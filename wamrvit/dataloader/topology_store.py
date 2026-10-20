"""An indexed, value-free topology store shared by all temporal windows.

SQLite supplies atomic publication, concurrent readers, and single-writer locking
without one filesystem object per frame. NPZ blobs contain numeric TreeDiff
columns only; reading never enables pickle. Legacy window caches are read only
by the explicit migration command, never implicitly redirected or overwritten.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from wamrvit.dataloader.selection.sources import canonical_source

TOPOLOGY_COLUMNS = (
    "topo_ids",
    "topo_tuples",
    "topo_ops",
    "topo_xmin",
    "topo_xmax",
    "topo_ymin",
    "topo_ymax",
    "topo_nx_tiles",
    "topo_ny_tiles",
    "topo_max_level_idx",
    "topo_channels",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def signature_key(signature):
    """Return the stable reference-store namespace for a topology build signature.

    Shared by pipeline assembly and legacy migration. Canonical JSON encoding is
    part of the persisted cache identity and must not change with orchestration.
    """
    return hashlib.sha256(_json(signature).encode()).hexdigest()


def validate_topology_store_options(options):
    """Reject unsupported keys at temporal validation, reference build and migration."""
    unsupported = set(options) - {"source_root", "dataset_id"}
    if unsupported:
        raise ValueError(
            f"Unsupported topology_store options {sorted(unsupported)}; only source_root and "
            "dataset_id are supported. Source fingerprints always use file size/mtime."
        )


def source_key(path, *, source_root=None, dataset_id=None):
    """Identify a reference by dataset/path and automatic source metadata.

    Used by: prepare_reference_store and migrate_legacy_cache.
    Relative identities are portable when copies preserve file sizes and mtimes.
    Source files must exist; there is no manual revision override of fingerprints.
    """
    source = canonical_source(path)
    physical, *suffix = source.split(":::")
    identity = source
    if source_root is not None:
        if not dataset_id:
            raise ValueError("topology_store.source_root requires an explicit dataset_id.")
        root = os.path.realpath(source_root)
        if os.path.commonpath([root, physical]) != root:
            raise ValueError(f"Source {physical} is outside topology_store.source_root={root}.")
        identity = ":::".join([os.path.relpath(physical, root), *suffix])
    source_path = Path(physical)
    if source_path.is_dir():
        files = [source_path / "Header", *sorted(source_path.glob("Level_*/Cell*"))]
    else:
        files = [source_path]
    # Protect against ordinary source replacement without rereading field values.
    # Relocation reuses entries only when the relevant file metadata are preserved.
    fingerprint = [
        (
            str(file.relative_to(source_path)) if source_path.is_dir() else "file",
            file.stat().st_size,
            file.stat().st_mtime_ns,
        )
        for file in files
        if file.is_file()
    ]
    if not fingerprint:
        raise FileNotFoundError(physical)
    # Keep the on-disk descriptor field name so existing metadata-based keys remain
    # unchanged. This is an automatic fingerprint, not a configurable revision.
    descriptor = {"dataset": dataset_id, "source": identity, "revision": fingerprint}
    return hashlib.sha256(_json(descriptor).encode()).hexdigest(), descriptor


def encode_topology(columns):
    """Encode numeric columns plus a conflict digest; used by TopologyStore.put."""
    arrays = {name: np.asarray(columns[name]) for name in TOPOLOGY_COLUMNS}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TypeError("Topology storage accepts numeric arrays only.")
    digest = hashlib.sha256()
    for name, array in arrays.items():
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(_json(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue(), digest.hexdigest()


def decode_topology(payload):
    """Decode trusted numeric structure without pickle; used by TopologyStore.get."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(TOPOLOGY_COLUMNS):
            raise ValueError("Invalid topology-store column schema.")
        return {name: archive[name] for name in TOPOLOGY_COLUMNS}


class TopologyStore:
    """Atomic numeric topology persistence, with one connection per owner.

    Used by: SampleMapper for reads; prepare_reference_store and explicit migration
    for writes. Stores geometry only, never temporal windows or field payloads.
    """

    def __init__(self, path, *, signature=None, writable=False):
        self.path = str(Path(path).absolute())
        self.writable = writable
        if writable and signature is None:
            raise ValueError("Creating a topology store requires a build signature.")
        if writable:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=120)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS topologies ("
                "frame_key TEXT PRIMARY KEY, source TEXT NOT NULL, "
                "digest TEXT NOT NULL, payload BLOB NOT NULL)"
            )
            with self.connection:
                self.connection.execute(
                    "INSERT OR IGNORE INTO metadata VALUES ('signature', ?)", (_json(signature),)
                )
        else:
            uri = Path(self.path).as_uri() + "?mode=ro"
            # A mapper owns each reader. Worker-stream shutdown joins its
            # executor before the consumer thread closes these connections.
            # Permit that quiescent ownership transfer; writers stay thread-bound.
            self.connection = sqlite3.connect(uri, uri=True, timeout=120, check_same_thread=False)
        stored = self.connection.execute(
            "SELECT value FROM metadata WHERE key='signature'"
        ).fetchone()
        if stored is None or (signature is not None and stored[0] != _json(signature)):
            self.close()
            raise ValueError("Topology-store build signature mismatch.")

    def missing(self, keys):
        """Return absent unique references for prepare_reference_store's bounded build."""
        keys = list(dict.fromkeys(keys))
        found = set()
        for offset in range(0, len(keys), 500):
            chunk = keys[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            found.update(
                row[0]
                for row in self.connection.execute(
                    f"SELECT frame_key FROM topologies WHERE frame_key IN ({placeholders})", chunk
                )
            )
        return [key for key in keys if key not in found]

    def put(self, key, source, columns):
        """Publish one topology atomically for builders/migration; reject conflicting data."""
        if not self.writable:
            raise RuntimeError("Cannot write through a read-only topology store.")
        payload, digest = encode_topology(columns)
        # A transaction makes conflict validation atomic even with two builders.
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO topologies VALUES (?, ?, ?, ?)",
                (key, _json(source), digest, payload),
            )
            existing = self.connection.execute(
                "SELECT digest FROM topologies WHERE frame_key=?", (key,)
            ).fetchone()[0]
            if existing != digest:
                raise ValueError(f"Conflicting topologies for reference frame {key}.")

    def get(self, key):
        """Fetch numeric columns for SampleMapper; missing references fail explicitly."""
        row = self.connection.execute(
            "SELECT payload FROM topologies WHERE frame_key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Reference topology {key} is missing from {self.path}.")
        return decode_topology(row[0])

    def __len__(self):
        return self.connection.execute("SELECT COUNT(*) FROM topologies").fetchone()[0]

    def close(self):
        """Release a connection after mapper shutdown or store context exit."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def iter_arrow_topology_rows(table):
    """Decode both fixed-shape list and variable-shape struct Ray columns.

    Used by: migrate_legacy_cache.
    The TreeDiff contract supplies the vector shapes (tuples are N×5), so
    numeric decoding does not need to unpickle historical extension metadata.
    Array slices keep their Arrow buffers alive for the duration of iteration.
    """
    import pyarrow as pa

    vectors = {"topo_ids", "topo_tuples", "topo_ops"}
    columns = {}
    table = table.combine_chunks()
    for name in TOPOLOGY_COLUMNS:
        column = table[name].chunk(0)
        if isinstance(column.type, pa.ExtensionType):
            column = column.storage
        if column.null_count:
            raise ValueError(f"Null topology entries in {name}.")
        if name in vectors:
            if pa.types.is_struct(column.type):
                column = column.field("data")
            if pa.types.is_fixed_size_list(column.type):
                offsets = (np.arange(len(column) + 1) + column.offset) * column.type.list_size
            elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
                offsets = column.offsets.to_numpy()
            else:
                raise TypeError(f"Unsupported numeric topology storage for {name}: {column.type}")
            if column.null_count or column.values.null_count:
                raise ValueError(f"Null topology vector values in {name}.")
            columns[name] = (offsets, column.values.to_numpy())
        else:
            columns[name] = column.to_numpy()
    for index in range(table.num_rows):
        result = {}
        for name, column in columns.items():
            if name in vectors:
                offsets, values = column
                value = values[offsets[index] : offsets[index + 1]]
                result[name] = value.reshape(-1, 5) if name == "topo_tuples" else value
            else:
                result[name] = np.asarray(column[index], dtype=column.dtype)
        if not (
            result["topo_ids"].size == result["topo_tuples"].shape[0] == result["topo_ops"].size
        ):
            raise ValueError("Topology ids, tuples, and ops have different lengths.")
        yield result


def migrate_legacy_cache(
    cache_dirs, destination, *, signature, key_options=None, trust_legacy_pickle=False
):
    """Explicit, non-destructive import; callers supply verified build provenance.

    Used by: this module's explicit migration CLI, never the training startup.
    Old Ray caches may pickle path arrays. Those are refused unless the operator
    explicitly trusts the source. Numeric topology columns never need pickle.
    """
    import pickle

    import pyarrow as pa
    import pyarrow.parquet as pq

    key_options = key_options or {}
    validate_topology_store_options(key_options)
    imported = 0
    with TopologyStore(destination, signature=signature, writable=True) as store:
        for directory in cache_dirs:
            for path in sorted(Path(directory).glob("*.parquet")):
                table = pq.read_table(path).combine_chunks()
                for index, columns in enumerate(iter_arrow_topology_rows(table)):
                    path_column = table["input_paths"].chunk(0)
                    if isinstance(path_column.type, pa.ExtensionType):
                        path_column = path_column.storage
                    paths = path_column[index].as_py()
                    if isinstance(paths, bytes):
                        if not trust_legacy_pickle:
                            raise ValueError(
                                "Legacy pickled paths require explicit --trust-legacy-pickle."
                            )
                        paths = pickle.loads(paths).tolist()
                    key, source = source_key(paths[-1], **key_options)
                    store.put(key, source, columns)
                    imported += 1
    return imported


def main():
    """CLI: python -m wamrvit.dataloader.topology_store; never alters legacy caches."""
    import argparse
    import ast

    from wamrvit.dataloader.payload.loaders import TopologyBuilder
    from wamrvit.dataloader.pipeline import builder_signature
    from wamrvit.utils import filter_valid_kwargs, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", action="append", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--trust-legacy-pickle", action="store_true")
    parser.add_argument("--accept-build-provenance", action="store_true")
    args, overrides = parser.parse_known_args()
    if not args.accept_build_provenance:
        parser.error(
            "Migration cannot infer implementation equivalence from legacy metadata. "
            "After verifying provenance, explicitly pass --accept-build-provenance."
        )
    config = load_config(args, overrides)
    options = config.get("topology_store", {})
    validate_topology_store_options(options)
    params = filter_valid_kwargs(TopologyBuilder, config["file_loader"].get("params", {}))
    signature = builder_signature(params)
    for directory in args.cache_dir:
        meta_path = Path(directory) / "meta.json"
        meta = json.loads(meta_path.read_text())
        if not list(Path(directory).glob("*.parquet")):
            raise ValueError(f"No Parquet files in {directory}.")
        old_params = {}
        for name, value in meta["topo_builder_params"].items():
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            old_params[name] = value
        old_signature = builder_signature(old_params)
        if _json(old_signature["params"]) != _json(signature["params"]):
            raise ValueError(f"Legacy topology parameters differ: {directory}.")
    build_key = signature_key(signature)
    destination = Path(args.destination_root) / "references" / f"{build_key}.sqlite"
    imported = migrate_legacy_cache(
        args.cache_dir,
        destination,
        signature=signature,
        key_options=options,
        trust_legacy_pickle=args.trust_legacy_pickle,
    )
    with TopologyStore(destination) as store:
        print(
            f"Imported {imported} rows into {len(store)} unique reference topologies: {destination}"
        )


if __name__ == "__main__":
    main()
