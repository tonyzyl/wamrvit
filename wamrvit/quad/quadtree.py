from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from wamrvit.quad.quadtree_kernels import (
    downsample_2x_block_mean_jit,
    morton2D_jit,
    normalize_centers_jit,
    resize_patch_bilinear_jit,
    resize_patch_nearest_jit,
)


class Quad(Enum):
    SW = 0  # (x-, y-)
    SE = 1  # (x+, y-)
    NW = 2  # (x-, y+)
    NE = 3  # (x+, y+)


class Dir(Enum):
    W = auto()
    E = auto()
    S = auto()
    N = auto()


class DiagDir(Enum):
    SW = auto()
    SE = auto()
    NW = auto()
    NE = auto()


# Child index helpers
Q = Quad
WEST_COL = {Q.SW.value, Q.NW.value}
EAST_COL = {Q.SE.value, Q.NE.value}
SOUTH_ROW = {Q.SW.value, Q.SE.value}
NORTH_ROW = {Q.NW.value, Q.NE.value}

# For descending along the side that abuts the caller
SIDE_CHILDREN = {
    Dir.W: (Q.SW.value, Q.NW.value),
    Dir.E: (Q.SE.value, Q.NE.value),
    Dir.S: (Q.SW.value, Q.SE.value),
    Dir.N: (Q.NW.value, Q.NE.value),
}


@dataclass
class QuadCell:
    """
    A single cell in the quadtree.

    Geometry: centroid (cx, cy) and half-extents (hx, hy). Children are in Morton
    order — SW(0), SE(1), NW(2), NE(3) — accessed via the `children` list.
    `is_leaf()` is True iff `children[0] is None`.

    The `value` patch has shape (C, H, W) where H, W depend on the parent
    Quadtree's storage mode. Uniform storage: H == patch_height, W == patch_width
    on every cell. Native storage: H, W scale with (max_level_idx - level_idx) so
    each cell resolves its physical area at the same per-cell density.

    `is_boundary` is set by `Quadtree.mark_boundary_cells()` — those calls are
    disabled in the active training/inference pipeline since the flag has no
    consumers there. Re-enable explicitly if you need it for visualization
    (`plot_quadtree_boundary`).
    """

    cx: float
    cy: float
    hx: float  # half-width
    hy: float  # half-height
    level_idx: int
    parent: QuadCell | None = None
    which: int | None = None
    value: np.ndarray | None = None
    children: list[QuadCell | None] = field(default_factory=lambda: [None, None, None, None])
    tile_ix: int = 0
    tile_iy: int = 0
    is_boundary: bool = False

    def is_leaf(self) -> bool:
        return self.children[0] is None

    def refine(
        self,
        init_child: Callable[[QuadCell, QuadCell], None] | None = None,
        interpolation: str = "bilinear",
    ) -> None:
        """
        Refines a leaf cell into 4 children by slicing its value patch into quadrants.

        Modes:
            'bilinear' / 'nearest': uniform-storage — slice + upsample back to parent (H, W).
                                    Conservation Property: Mean(Children) == Parent.
            'exact':                native-storage   — slice only; each child stores (C, H/2, W/2).
                                    Requires H and W to be even (enforced).

        Args:
            init_child: Optional callback(parent, child) after each child is created.
            interpolation: 'bilinear' | 'nearest' | 'exact'.
        """
        if not self.is_leaf():
            return
        hhx = self.hx * 0.5
        hhy = self.hy * 0.5

        val_splits = [None] * 4
        if self.value is not None:
            v = self.value
            _, H, W = v.shape

            # Determine split points based on quadrant sizes relative to parent patch
            h_mid = H // 2
            w_mid = W // 2

            # 1. Slice Quadrants from Parent
            v_sw_raw = v[:, 0:h_mid, 0:w_mid]
            v_se_raw = v[:, 0:h_mid, w_mid:W]
            v_nw_raw = v[:, h_mid:H, 0:w_mid]
            v_ne_raw = v[:, h_mid:H, w_mid:W]

            if interpolation == "exact":
                # Native mode: child stores half-size quadrant directly (no upsample).
                if H % 2 != 0 or W % 2 != 0:
                    raise ValueError(
                        f"refine(interpolation='exact') requires even (H, W); got ({H}, {W})."
                    )
                val_splits = [
                    np.ascontiguousarray(v_sw_raw),
                    np.ascontiguousarray(v_se_raw),
                    np.ascontiguousarray(v_nw_raw),
                    np.ascontiguousarray(v_ne_raw),
                ]
            else:
                # Uniform mode: upsample each quadrant back to (H, W)
                _resize = (
                    resize_patch_bilinear_jit
                    if interpolation == "bilinear"
                    else resize_patch_nearest_jit
                )
                val_splits = [
                    _resize(v_sw_raw, H, W),
                    _resize(v_se_raw, H, W),
                    _resize(v_nw_raw, H, W),
                    _resize(v_ne_raw, H, W),
                ]

        centers = {
            Q.SW.value: (self.cx - hhx, self.cy - hhy),
            Q.SE.value: (self.cx + hhx, self.cy - hhy),
            Q.NW.value: (self.cx - hhx, self.cy + hhy),
            Q.NE.value: (self.cx + hhx, self.cy + hhy),
        }

        for i in range(4):
            new_val = val_splits[i] if val_splits[i] is not None else None
            c = QuadCell(
                cx=centers[i][0],
                cy=centers[i][1],
                hx=hhx,
                hy=hhy,
                level_idx=self.level_idx + 1,
                parent=self,
                which=i,
                value=new_val,
                tile_ix=self.tile_ix,
                tile_iy=self.tile_iy,
            )
            if init_child:
                init_child(self, c)
            self.children[i] = c

    def coarsen(
        self,
        reduce_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None,
        stitch_only: bool = False,
    ) -> None:
        """
        Coarsens children into a single parent value.

        Modes:
            uniform (default):     stitch 4 children (C, H, W) -> (C, 2H, 2W), then 2x2 block-mean
                                   back to (C, H, W). Conservation: Parent = Average(Children).
            stitch_only=True:      native-storage — stitch 4 children (C, H, W) -> (C, 2H, 2W)
                                   and store directly (no block-mean). Lossless: exact inverse
                                   of refine(interpolation='exact').
            reduce_fn provided:    custom reducer over the child values (e.g., max/min).

        Args:
            reduce_fn: Optional custom reducer over the list of valid child values.
            stitch_only: If True, skip block-mean and store the stitched super-patch directly.
        """
        if self.is_leaf():
            return

        # Collect values, assume order SW, SE, NW, NE
        vals = [c.value if (c is not None and c.value is not None) else None for c in self.children]

        if all(v is None for v in vals):
            self.children = [None] * 4
            return

        if reduce_fn is not None:
            # Fallback for custom reducers (e.g., max, min)
            valid = [v for v in vals if v is not None]
            if valid:
                self.value = reduce_fn(valid)
        else:
            # 1. Get Geometry
            valid = next(v for v in vals if v is not None)
            C, H, W = valid.shape

            # Fill missing children with zeros (assuming strict layout)
            vals_safe = [
                v if v is not None else np.zeros((C, H, W), dtype=valid.dtype) for v in vals
            ]
            v_sw, v_se, v_nw, v_ne = vals_safe

            # 2. Stitch children into a 2x super-patch
            # Layout:
            #  [NW, NE]
            #  [SW, SE]
            # (Note: Standard grid layout has y-min at bottom (row 0), so SW is row 0-mid)
            bot = np.concatenate([v_sw, v_se], axis=2)  # Axis 2 is Width
            top = np.concatenate([v_nw, v_ne], axis=2)
            stitched = np.concatenate([bot, top], axis=1)  # Axis 1 is Height

            # stitched shape is now (C, 2H, 2W)

            if stitch_only:
                # Native mode: parent stores the full stitched super-patch.
                self.value = np.ascontiguousarray(stitched)
            else:
                # Uniform mode: 2x2 block-mean back to (C, H, W).
                self.value = downsample_2x_block_mean_jit(stitched)

        self.children = [None, None, None, None]

    def iter_leaves(self) -> Iterable[QuadCell]:
        if self.is_leaf():
            yield self
        else:
            for c in self.children:
                if c is not None:
                    yield from c.iter_leaves()

    def contains(self, x: float, y: float) -> bool:
        return (self.cx - self.hx) <= x <= (self.cx + self.hx) and (self.cy - self.hy) <= y <= (
            self.cy + self.hy
        )

    def locate_leaf(self, x: float, y: float) -> QuadCell | None:
        if not self.contains(x, y):
            return None
        if self.is_leaf():
            return self
        idx = quad_index(self, x, y)
        child = self.children[idx]
        if child is None:
            return None
        return child.locate_leaf(x, y)

    def face_neighbors(self, direction: Dir) -> list[QuadCell]:
        sib = self._sibling_same_level(direction)
        if sib is not None:
            if sib.is_leaf():
                return [sib]
            return descend_to_abutting_leaves(sib, direction.opposite())

        cur = self
        while cur.parent is not None:
            sib = cur._sibling_same_level(direction)
            if sib is not None:
                return descend_to_abutting_leaves(sib, direction.opposite())
            cur = cur.parent
        return []

    def _sibling_same_level(self, direction: Dir) -> QuadCell | None:
        if self.parent is None or self.which is None:
            return None
        i = self.which
        if direction == Dir.E and i in WEST_COL:
            return self.parent.children[i + 1]
        if direction == Dir.W and i in EAST_COL:
            return self.parent.children[i - 1]
        if direction == Dir.N and i in SOUTH_ROW:
            return self.parent.children[i + 2]
        if direction == Dir.S and i in NORTH_ROW:
            return self.parent.children[i - 2]
        return None


def quad_index(cell: QuadCell, x: float, y: float) -> int:
    east = x >= cell.cx
    north = y >= cell.cy
    if not east and not north:
        return Q.SW.value
    if east and not north:
        return Q.SE.value
    if not east and north:
        return Q.NW.value
    return Q.NE.value


def descend_to_abutting_leaves(cell: QuadCell, side_from_neighbor: Dir) -> list[QuadCell]:
    leaves: list[QuadCell] = []
    stack: list[QuadCell] = [cell]
    side_idxs = SIDE_CHILDREN[side_from_neighbor]
    while stack:
        cur = stack.pop()
        if cur.is_leaf():
            leaves.append(cur)
        else:
            for idx in side_idxs:
                ch = cur.children[idx]
                if ch is not None:
                    stack.append(ch)
    return leaves


def _dir_opposite(d: Dir) -> Dir:
    return {Dir.N: Dir.S, Dir.S: Dir.N, Dir.E: Dir.W, Dir.W: Dir.E}[d]


def _dir_opposite_from_child(d: Dir, child_idx: int | None) -> Dir:
    return _dir_opposite(d)


def _dir_opposite_patch():
    def opposite(self: Dir) -> Dir:
        return _dir_opposite(self)

    def opposite_from_child(self: Dir, child_idx: int | None) -> Dir:
        return _dir_opposite_from_child(self, child_idx)

    setattr(Dir, "opposite", opposite)
    setattr(Dir, "opposite_from_child", opposite_from_child)


_dir_opposite_patch()


def _ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def _child_bits(which: int) -> tuple[int, int]:
    east = 1 if which in EAST_COL else 0
    north = 1 if which in NORTH_ROW else 0
    return east, north


class Quadtree:
    """
    Rectangular domain [xmin, xmax] x [ymin, ymax] tiled by rectangular tiles, each
    a quadtree root. Leaves carry per-cell value patches; refinement subdivides into
    Morton-ordered children.

    Patch dimensions are derived from the tile dimensions and max_level_idx:
        patch_width  = tile_width  / 2^max_level_idx
        patch_height = tile_height / 2^max_level_idx
    If these values are not integers, a ValueError is raised at construction.

    Storage modes (see ``value_storage``):
        ``"uniform"`` (default): every leaf stores a (C, patch_h, patch_w) patch at
            the finest patch size. Coarse leaves carry an averaged version at the
            same shape — a single token represents one cell regardless of level.
        ``"native"``: each leaf stores its own native-resolution patch. Level-l
            leaves store (C, patch_h * 2**(max_level_idx - l), patch_w * 2**(...)),
            so all levels resolve the same physical area at the same per-cell density.
            Used by the multi-scale model path (`MultiScalePatchEmbed`).

    Conventions:
        ``level_idx`` is 0-based; ``max_level_idx == num_levels - 1``. Children are
        in Morton order: SW(0), SE(1), NW(2), NE(3). Mutating methods (`refine_leaf`,
        `coarsen_node`, `refine_where`, `coarsen_where`, `ensure_2to1_balance`)
        operate in place; copy/deepcopy yields an independent tree.

    Args:
        channels (int): Number of channels of the Quadtree.
        max_level_idx (int): Maximum refinement level_idx. Level 0 is the root tile,
            level_idx 1 is the first refinement, etc.
        tile_width (float, optional): Width of each root tile. If None, defaults to
            making square tiles.
        tile_height (float, optional): Height of each root tile. If None, defaults to
            making square tiles.
        value_storage (str): "uniform" or "native"; see Storage modes above.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        max_level_idx: int,
        channels: int = 1,
        tile_width: float = None,
        tile_height: float = None,
        periodic_right: bool = False,
        periodic_top: bool = False,
        value_storage: str = "uniform",
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.max_level_idx = max_level_idx
        self.tile_width = tile_width
        self.tile_height = tile_height

        assert xmax > xmin and ymax > ymin
        Lx = xmax - xmin
        Ly = ymax - ymin

        tw = (
            tile_width
            if tile_width is not None
            else (tile_height if tile_height is not None else min(Lx, Ly))
        )
        th = (
            tile_height
            if tile_height is not None
            else (tile_width if tile_width is not None else min(Lx, Ly))
        )

        nx_float = Lx / tw
        ny_float = Ly / th
        nx = int(round(nx_float))
        ny = int(round(ny_float))

        assert abs(nx_float - nx) < 1e-12 and abs(ny_float - ny) < 1e-12, (
            f"Domain ({Lx}x{Ly}) must be an integer multiple of the tile size ({tw}x{th})."
        )

        # Calculate minimal cell dimensions (Patch Size)
        factor = 2**max_level_idx
        min_w = tw / factor
        min_h = th / factor

        # Strict check for rounding issues
        if not (
            np.isclose(min_w, round(min_w), atol=1e-6)
            and np.isclose(min_h, round(min_h), atol=1e-6)
        ):
            raise ValueError(
                f"Tile dimensions ({tw}, {th}) at max_level_idx {max_level_idx} result in "
                f"non-integer minimal cell sizes ({min_w}, {min_h}). "
                "The grid must align perfectly with pixels."
            )

        self.patch_width = int(round(min_w))
        self.patch_height = int(round(min_h))

        self.min_grid_w = min_w
        self.min_grid_h = min_h
        self.max_grid_w = tw
        self.max_grid_h = th

        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.tile_width = tw
        self.tile_height = th
        self.nx_tiles = nx
        self.ny_tiles = ny
        self.max_level_idx = max_level_idx
        self.channels = channels

        if value_storage not in ("uniform", "native"):
            raise ValueError(f"value_storage must be 'uniform' or 'native'; got {value_storage!r}.")
        self.value_storage = value_storage

        # Root tile stores the native-resolution patch in 'native' mode (largest level).
        # In 'uniform' mode every leaf stores (C, patch_height, patch_width).
        root_factor = (1 << max_level_idx) if value_storage == "native" else 1
        root_ph = self.patch_height * root_factor
        root_pw = self.patch_width * root_factor

        self.roots: list[list[QuadCell]] = []
        for jy in range(ny):
            row: list[QuadCell] = []
            cy = ymin + (jy + 0.5) * th
            for ix in range(nx):
                cx = xmin + (ix + 0.5) * tw
                # Initialize with zeros of the correct patch shape
                init_val = np.zeros((channels, root_ph, root_pw), dtype=float)
                root = QuadCell(
                    cx=cx,
                    cy=cy,
                    hx=0.5 * tw,
                    hy=0.5 * th,
                    level_idx=0,
                    value=init_val,
                    tile_ix=ix,
                    tile_iy=jy,
                )
                row.append(root)
            self.roots.append(row)

        self.root: QuadCell = self.roots[0][0]
        self.periodic_right = periodic_right
        self.periodic_top = periodic_top

        # self.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline

    def mark_boundary_cells(self, eps: float = 1e-12) -> None:
        """Mark every leaf whose extent touches a domain edge."""
        for leaf in self._iter_all_leaves():
            leaf.is_boundary = (
                (leaf.cx - leaf.hx) <= self.xmin + eps
                or (leaf.cx + leaf.hx) >= self.xmax - eps
                or (leaf.cy - leaf.hy) <= self.ymin + eps
                or (leaf.cy + leaf.hy) >= self.ymax - eps
            )

    def get_domain_meta(self) -> dict[str, float]:
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "max_level_idx": self.max_level_idx,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
        }

    def _iter_all_leaves(self) -> Iterable[QuadCell]:
        for row in self.roots:
            for r in row:
                yield from r.iter_leaves()

    def refine_leaf(
        self,
        leaf: QuadCell,
        init_child: Callable[[QuadCell, QuadCell], None] | None = None,
        interpolation: str | None = None,
    ) -> None:
        """Refine a leaf using the quadtree's value_storage mode.

        In 'native' mode forces interpolation='exact' (lossless slice).
        In 'uniform' mode uses the caller-provided interpolation (default 'bilinear').
        """
        if self.value_storage == "native":
            leaf.refine(init_child=init_child, interpolation="exact")
        else:
            leaf.refine(init_child=init_child, interpolation=interpolation or "bilinear")

    def coarsen_node(
        self,
        node: QuadCell,
        reduce_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None,
    ) -> None:
        """Coarsen a node using the quadtree's value_storage mode.

        In 'native' mode uses stitch_only (lossless stitch, inverse of exact refine).
        In 'uniform' mode uses the default block-mean coarsen.
        """
        if self.value_storage == "native":
            node.coarsen(reduce_fn=reduce_fn, stitch_only=True)
        else:
            node.coarsen(reduce_fn=reduce_fn)

    def refine_where(self, predicate: Callable[[QuadCell], bool]) -> int:
        count = 0
        for leaf in list(self._iter_all_leaves()):
            if leaf.level_idx < self.max_level_idx and predicate(leaf):
                self.refine_leaf(leaf)
                count += 1
        # if count > 0:
        #     self.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline
        return count

    def coarsen_where(self, predicate: Callable[[QuadCell], bool]) -> int:
        count = 0

        def visit(node: QuadCell):
            nonlocal count
            if node.is_leaf():
                return
            for c in node.children:
                if c is not None:
                    visit(c)
            if all(c is not None and c.is_leaf() for c in node.children) and predicate(node):
                self.coarsen_node(node)
                count += 1

        for row in self.roots:
            for r in row:
                visit(r)
        # if count > 0:
        #     self.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline
        return count

    def face_neighbors(self, cell: QuadCell, direction: Dir) -> list[QuadCell]:
        n_intra = cell.face_neighbors(direction)
        if n_intra:
            return n_intra

        dix = 1 if direction == Dir.E else (-1 if direction == Dir.W else 0)
        diy = 1 if direction == Dir.N else (-1 if direction == Dir.S else 0)
        nix = cell.tile_ix + dix
        niy = cell.tile_iy + diy

        # Periodic wrapping
        if self.periodic_right and direction in (Dir.E, Dir.W):
            nix = nix % self.nx_tiles
        if self.periodic_top and direction in (Dir.N, Dir.S):
            niy = niy % self.ny_tiles

        if 0 <= nix < self.nx_tiles and 0 <= niy < self.ny_tiles:
            neighbor_root = self.roots[niy][nix]
            return descend_to_abutting_leaves(neighbor_root, direction.opposite())
        return []

    def corner_neighbors(self, cell: QuadCell, diag_dir: DiagDir) -> list[QuadCell]:
        """Return leaf cells diagonally adjacent to `cell` at the given corner.

        Uses locate_leaf at the corner vertex offset by a small epsilon into the
        diagonal neighbor's interior. Handles cross-tile and periodic boundaries.
        """
        eps = min(cell.hx, cell.hy) * 1e-6

        # Corner vertex of the cell
        if diag_dir == DiagDir.SW:
            vx, vy = cell.cx - cell.hx, cell.cy - cell.hy
            px, py = vx - eps, vy - eps
        elif diag_dir == DiagDir.SE:
            vx, vy = cell.cx + cell.hx, cell.cy - cell.hy
            px, py = vx + eps, vy - eps
        elif diag_dir == DiagDir.NW:
            vx, vy = cell.cx - cell.hx, cell.cy + cell.hy
            px, py = vx - eps, vy + eps
        elif diag_dir == DiagDir.NE:
            vx, vy = cell.cx + cell.hx, cell.cy + cell.hy
            px, py = vx + eps, vy + eps
        else:
            return []

        # Handle periodic wrapping
        if self.periodic_right:
            Lx = self.xmax - self.xmin
            if px < self.xmin:
                px += Lx
            elif px >= self.xmax:
                px -= Lx
        if self.periodic_top:
            Ly = self.ymax - self.ymin
            if py < self.ymin:
                py += Ly
            elif py >= self.ymax:
                py -= Ly

        leaf = self.locate_leaf(px, py)
        if leaf is None or leaf is cell:
            return []
        return [leaf]

    def ensure_2to1_balance(self) -> int:
        changed = 0
        pending: list[QuadCell] = []
        for leaf in self._iter_all_leaves():
            for d in (Dir.W, Dir.E, Dir.S, Dir.N):
                neighbors = self.face_neighbors(leaf, d)
                for n in neighbors:
                    if n.level_idx - leaf.level_idx > 1 and leaf.level_idx < self.max_level_idx:
                        pending.append(leaf)
                        break
                else:
                    continue
                break
        for leaf in pending:
            if leaf.is_leaf() and leaf.level_idx < self.max_level_idx:
                self.refine_leaf(leaf)
                changed += 1
        return changed

    def locate_leaf(self, x: float, y: float) -> QuadCell | None:
        if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
            return None
        # Clamp index to handle numerical edge cases at max boundary
        tx = min(self.nx_tiles - 1, max(0, int((x - self.xmin) / self.tile_width)))
        ty = min(self.ny_tiles - 1, max(0, int((y - self.ymin) / self.tile_height)))
        return self.roots[ty][tx].locate_leaf(x, y)

    def sample_leaves(self) -> list[tuple[tuple[float, float, float, float], int, np.ndarray]]:
        """
        Returns a list of (bbox, level_idx, value) where bbox=(cx, cy, hx, hy).
        Value is (C, patch_h, patch_w).
        """
        out = []
        for leaf in self._iter_all_leaves():
            out.append(
                (
                    (leaf.cx, leaf.cy, leaf.hx, leaf.hy),
                    leaf.level_idx,
                    leaf.value if leaf.value is not None else None,
                )
            )
        return out

    def get_num_leaves(self) -> int:
        return sum(1 for _ in self._iter_all_leaves())

    def assign_from_array(
        self,
        arr: np.ndarray,
        extent: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Assigns values to leaves by downsampling chunks of 'arr' to the leaf's fixed patch size.
        """
        assert arr.ndim == 3, "Array must be 3D (C,H,W)"
        C, H, W = arr.shape

        if C != self.channels:
            warnings.warn(f"Array has {C} channels, but quadtree has {self.channels}.")

        xmin = self.xmin if extent is None else extent[0]
        xmax = self.xmax if extent is None else extent[1]
        ymin = self.ymin if extent is None else extent[2]
        ymax = self.ymax if extent is None else extent[3]

        dx = (xmax - xmin) / max(W, 1)
        dy = (ymax - ymin) / max(H, 1)

        Ph = self.patch_height
        Pw = self.patch_width

        def clamp_int(v: int, lo: int, hi: int) -> int:
            return lo if v < lo else (hi if v > hi else v)

        for leaf in self._iter_all_leaves():
            x0 = max(xmin, leaf.cx - leaf.hx)
            x1 = min(xmax, leaf.cx + leaf.hx)
            y0 = max(ymin, leaf.cy - leaf.hy)
            y1 = min(ymax, leaf.cy + leaf.hy)

            j_start = int(np.floor((x0 - xmin) / dx))
            j_end = int(np.ceil((x1 - xmin) / dx))
            i_start = int(np.floor((y0 - ymin) / dy))
            i_end = int(np.ceil((y1 - ymin) / dy))

            j_start = clamp_int(j_start, 0, W)
            i_start = clamp_int(i_start, 0, H)
            j_end = clamp_int(j_end, 0, W)
            i_end = clamp_int(i_end, 0, H)

            # Expected Region Size in Pixels for this leaf
            # The leaf covers physical area (2*hx, 2*hy).
            # The root tile covers 2*root_hx, 2*root_hy.
            # The leaf scale is 2**(max_level_idx - leaf.level_idx) relative to the minimum cell.
            scale_factor = 1 << (self.max_level_idx - leaf.level_idx)
            expected_h = Ph * scale_factor
            expected_w = Pw * scale_factor

            # Extract
            if j_end <= j_start or i_end <= i_start:
                # Fallback: Leaf is smaller than a pixel or off-grid
                # Just sample the center
                jc = int(np.floor((leaf.cx - xmin) / dx))
                ic = int(np.floor((leaf.cy - ymin) / dy))
                jc = clamp_int(jc, 0, W - 1)
                ic = clamp_int(ic, 0, H - 1)
                pix = arr[:, ic, jc]  # (C,)
                # Broadcast to (C, Ph, Pw)
                val = np.tile(pix[:, None, None], (1, Ph, Pw))
            else:
                region = arr[:, i_start:i_end, j_start:j_end]

                # Downsample (Average Pooling) to (C, Ph, Pw)
                # We assume integer multiples enforced by init checks.
                # However, edge clipping might break exact multiples, so we use robust
                # reshaping if possible, or strict reshaping if dimensions match expectations.

                h_r, w_r = region.shape[1:]
                if h_r == expected_h and w_r == expected_w:
                    # Input at the quadtree's native (index-unit) resolution —
                    # original behavior preserved byte-for-byte.
                    if self.value_storage == "native":
                        # Native mode: store the raw region at its native (expected_h, expected_w).
                        val = region.astype(float)
                    elif scale_factor == 1:
                        # Uniform mode, finest level: raw region already matches (Ph, Pw).
                        val = region.astype(float)
                    else:
                        # Uniform mode, coarser level: block-average to (Ph, Pw).
                        reshaped = region.reshape(C, Ph, scale_factor, Pw, scale_factor)
                        val = reshaped.mean(axis=(2, 4))
                elif h_r == Ph and w_r == Pw:
                    # Downsampled input where extracted region already matches
                    # leaf storage — e.g. AMReX regular-model output at base-grid
                    # resolution projected onto an adaptive quadtree built at a
                    # finer index-unit resolution. No resize needed.
                    warnings.warn(
                        f"assign_from_array: region shape ({h_r}, {w_r}) matches leaf "
                        f"storage (Ph={Ph}, Pw={Pw}) but differs from native index "
                        f"resolution ({expected_h}, {expected_w}) at level_idx "
                        f"{leaf.level_idx}. Input appears downsampled; storing region "
                        f"as-is.",
                        stacklevel=2,
                    )
                    val = region.astype(float)
                elif h_r >= Ph and w_r >= Pw and h_r % Ph == 0 and w_r % Pw == 0:
                    # Input at an intermediate resolution (> leaf storage but < native,
                    # cleanly divisible). Block-average to (Ph, Pw).
                    sh, sw = h_r // Ph, w_r // Pw
                    warnings.warn(
                        f"assign_from_array: region shape ({h_r}, {w_r}) differs from "
                        f"native index resolution ({expected_h}, {expected_w}) at "
                        f"level_idx {leaf.level_idx}; block-averaging by ({sh}, {sw}) "
                        f"to ({Ph}, {Pw}).",
                        stacklevel=2,
                    )
                    reshaped = region.reshape(C, Ph, sh, Pw, sw)
                    val = reshaped.mean(axis=(2, 4))
                elif h_r <= Ph and w_r <= Pw and Ph % h_r == 0 and Pw % w_r == 0:
                    # Input coarser than leaf storage (sub-base-pixel leaves — common
                    # at fine quadtree levels when arr is at a downsampled resolution
                    # relative to the quadtree's native index space). Upsample via
                    # nearest-neighbor repeat.
                    rh, rw = Ph // h_r, Pw // w_r
                    warnings.warn(
                        f"assign_from_array: region shape ({h_r}, {w_r}) coarser than "
                        f"leaf storage ({Ph}, {Pw}) at level_idx {leaf.level_idx}; "
                        f"upsampling via nearest-neighbor repeat by ({rh}, {rw}). "
                        f"Expected native resolution ({expected_h}, {expected_w}).",
                        stacklevel=2,
                    )
                    val = np.repeat(np.repeat(region, rh, axis=1), rw, axis=2).astype(float)
                else:
                    raise ValueError(
                        f"Region shape {region.shape[1:]} cannot be aligned to "
                        f"({Ph}, {Pw}) at level_idx {leaf.level_idx}: expected "
                        f"({expected_h}, {expected_w}) at native index resolution, "
                        f"or a clean integer multiple/divisor of ({Ph}, {Pw})."
                    )

            leaf.value = val.astype(float)

    def cell_xy_index(self, cell: QuadCell) -> tuple[int, int, int]:
        x_idx = 0
        y_idx = 0
        level_idx = 0
        cur = cell
        while cur.parent is not None:
            assert cur.which is not None
            east, north = _child_bits(cur.which)
            x_idx |= (east & 1) << level_idx
            y_idx |= (north & 1) << level_idx
            level_idx += 1
            cur = cur.parent
        return level_idx, x_idx, y_idx

    def cell_uid_64bit(self, cell: QuadCell) -> np.uint64:
        level_idx, x_idx, y_idx = self.cell_xy_index(cell)
        XW = self.max_level_idx
        YW = self.max_level_idx
        LW = 6
        TXW = _ceil_log2(self.nx_tiles)
        TYW = _ceil_log2(self.ny_tiles)

        x_masked = x_idx & ((1 << XW) - 1) if XW > 0 else 0
        y_masked = y_idx & ((1 << YW) - 1) if YW > 0 else 0

        uid = 0
        shift = 0
        uid |= int(x_masked) << shift
        shift += XW
        uid |= int(y_masked) << shift
        shift += YW
        uid |= int(level_idx & ((1 << LW) - 1)) << shift
        shift += LW
        if TXW:
            uid |= int(cell.tile_ix & ((1 << TXW) - 1)) << shift
            shift += TXW
        if TYW:
            uid |= int(cell.tile_iy & ((1 << TYW) - 1)) << shift
            shift += TYW
        return np.uint64(uid)

    def decode_uid(self, uid: int) -> tuple[int, int, int, int, int]:
        XW = self.max_level_idx
        YW = self.max_level_idx
        LW = 6
        TXW = _ceil_log2(self.nx_tiles)
        TYW = _ceil_log2(self.ny_tiles)

        u = int(uid)
        shift = 0
        x_idx = (u >> shift) & ((1 << XW) - 1) if XW > 0 else 0
        shift += XW
        y_idx = (u >> shift) & ((1 << YW) - 1) if YW > 0 else 0
        shift += YW
        level_idx = (u >> shift) & ((1 << LW) - 1)
        shift += LW
        tile_ix = (u >> shift) & ((1 << TXW) - 1) if TXW else 0
        shift += TXW
        tile_iy = (u >> shift) & ((1 << TYW) - 1) if TYW else 0
        return int(tile_ix), int(tile_iy), int(level_idx), int(x_idx), int(y_idx)

    def _leaf_fine_coords(self, cell: QuadCell) -> tuple[int, int]:
        level_idx, x_idx, y_idx = self.cell_xy_index(cell)
        scale = 0 if self.max_level_idx - level_idx <= 0 else (self.max_level_idx - level_idx)
        x0 = (cell.tile_ix << self.max_level_idx) | (x_idx << scale)
        y0 = (cell.tile_iy << self.max_level_idx) | (y_idx << scale)
        return x0, y0

    def ordered_leaves(self, order: str = "morton") -> list[QuadCell]:
        """Return leaves sorted by the requested canonical order (morton/uid/raster)."""
        leaves = list(self._iter_all_leaves())
        if order == "morton":
            keys = [
                (
                    morton2D_jit(
                        *self._leaf_fine_coords(c),
                        bits=self.max_level_idx + _ceil_log2(max(self.nx_tiles, self.ny_tiles)),
                    ),
                    i,
                )
                for i, c in enumerate(leaves)
            ]
            order_idx = [i for _, i in sorted(keys, key=lambda t: t[0])]
        elif order == "uid":
            keys = [(int(self.cell_uid_64bit(c)), i) for i, c in enumerate(leaves)]
            order_idx = [i for _, i in sorted(keys, key=lambda t: t[0])]
        else:

            def key(i_c):
                i, c = i_c
                level_idx, x, y = self.cell_xy_index(c)
                return (c.tile_iy, c.tile_ix, y, x, level_idx)

            order_idx = [i for i, _ in sorted(enumerate(leaves), key=key)]
        return [leaves[i] for i in order_idx]

    def export_leaves(
        self, order: str = "morton", cell_scale_mode: str | None = None, eps: float = 1e-12
    ):
        """
        Export leaves.
        Values are [N, C, patch_h, patch_w] in uniform mode; skipped in native mode
        (variable per-leaf size — use quadtree_to_tensor_native instead).
        If cell_scale_mode is provided, 'centers' will be [N, 3] containing normalized
        (cx_norm, cy_norm, s_norm). Otherwise, it defaults to [N, 4] (cx, cy, hx, hy).
        """
        ordered = self.ordered_leaves(order)
        N = len(ordered)
        C = self.channels
        Ph, Pw = self.patch_height, self.patch_width

        ids = np.empty(N, dtype=np.uint64)

        # Allocate centers based on scale mode
        if cell_scale_mode is None:
            centers = np.empty((N, 4), dtype=np.float32)
        else:
            centers = np.empty((N, 3), dtype=np.float32)

        levels = np.empty(N, dtype=np.int16)
        tiles = np.empty((N, 2), dtype=np.int16)
        xy_idx = np.empty((N, 3), dtype=np.int32)
        # Skip flat values array in native mode — leaves have variable shape.
        if C == 0 or self.value_storage == "native":
            values = None
        else:
            values = np.empty((N, C, Ph, Pw), dtype=np.float32)

        for k, c in enumerate(ordered):
            ids[k] = self.cell_uid_64bit(c)

            # --- Center & Geometry Population ---
            if cell_scale_mode is None:
                centers[k, 0] = c.cx
                centers[k, 1] = c.cy
                centers[k, 2] = c.hx
                centers[k, 3] = c.hy
            else:
                grid_w = 2.0 * c.hx
                grid_h = 2.0 * c.hy

                if cell_scale_mode == "level_idx":
                    s_val = c.level_idx
                elif cell_scale_mode == "x":
                    s_val = grid_w
                elif cell_scale_mode == "y":
                    s_val = grid_h
                elif cell_scale_mode == "area":
                    s_val = grid_w * grid_h
                elif cell_scale_mode == "sqrt_area":
                    s_val = np.sqrt(grid_w * grid_h)
                elif cell_scale_mode == "log_area":
                    s_val = np.log(grid_w * grid_h + eps)
                elif cell_scale_mode == "diag":
                    s_val = np.sqrt(grid_w**2 + grid_h**2)
                else:
                    raise ValueError(f"Invalid cell_scale_mode '{cell_scale_mode}'")

                centers[k, 0] = c.cx
                centers[k, 1] = c.cy
                centers[k, 2] = s_val

            levels[k] = c.level_idx
            tiles[k, 0] = c.tile_ix
            tiles[k, 1] = c.tile_iy
            level_idx, xi, yi = self.cell_xy_index(c)
            xy_idx[k, 0] = level_idx
            xy_idx[k, 1] = xi
            xy_idx[k, 2] = yi

            if values is not None:
                if c.value is None:
                    values[k, ...] = 0.0
                else:
                    values[k, ...] = c.value

        # --- Post-Processing: Normalize Centers ---
        if cell_scale_mode is not None and N > 0:
            Lx = self.xmax - self.xmin
            Ly = self.ymax - self.ymin

            if cell_scale_mode == "level_idx":
                cell_min = 0.0
                Lcell = 1.0
            elif cell_scale_mode == "x":
                cell_min = self.min_grid_w
                Lcell = self.max_grid_w - self.min_grid_w
            elif cell_scale_mode == "y":
                cell_min = self.min_grid_h
                Lcell = self.max_grid_h - self.min_grid_h
            elif cell_scale_mode == "area":
                cell_min = self.min_grid_w * self.min_grid_h
                Lcell = (self.max_grid_w * self.max_grid_h) - cell_min
            elif cell_scale_mode == "sqrt_area":
                cell_min = np.sqrt(self.min_grid_w * self.min_grid_h)
                Lcell = np.sqrt(self.max_grid_w * self.max_grid_h) - cell_min
            elif cell_scale_mode == "log_area":
                cell_min = np.log(self.min_grid_w * self.min_grid_h + eps)
                Lcell = np.log(self.max_grid_w * self.max_grid_h + eps) - cell_min
            elif cell_scale_mode == "diag":
                cell_min = np.sqrt(self.min_grid_w**2 + self.min_grid_h**2)
                Lcell = np.sqrt(self.max_grid_w**2 + self.max_grid_h**2) - cell_min
            else:
                raise ValueError(f"Invalid cell_scale_mode '{cell_scale_mode}'")

            # Apply in-place Numba normalization (centers-min) / L
            normalize_centers_jit(centers, self.xmin, self.ymin, cell_min, Lx, Ly, Lcell)

        return {
            "ids": ids,
            "centers": centers,
            "levels": levels,
            "tiles": tiles,
            "xy_idx": xy_idx,
            "values": values,
        }

    def project_values_from(
        self,
        src: Quadtree,
        dst_offset: int = 0,
        num_channels: int | None = None,
        fill_value: float = 0.0,
    ) -> None:
        assert (
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.max_grid_h,
            self.max_grid_w,
            self.min_grid_h,
            self.min_grid_w,
        ) == (
            src.xmin,
            src.xmax,
            src.ymin,
            src.ymax,
            src.max_grid_h,
            src.max_grid_w,
            src.min_grid_h,
            src.min_grid_w,
        ), "Domain mismatch"
        # Since structure matches, we assume patch sizes are compatible or we project naively

        if self.channels == 0 or src.channels == 0:
            return
        ncopy = min(
            self.channels - dst_offset, src.channels if num_channels is None else num_channels
        )
        if ncopy <= 0:
            return

        for leaf in self._iter_all_leaves():
            if leaf.value is None:
                # Re-init if missing
                leaf.value = np.zeros(
                    (self.channels, self.patch_height, self.patch_width), dtype=float
                )

            vdst = leaf.value  # (C, Ph, Pw) reference

            sleaf = src.locate_leaf(leaf.cx, leaf.cy)
            if sleaf is None or sleaf.value is None:
                vdst[dst_offset : dst_offset + ncopy, ...] = fill_value
            else:
                # For now assume same patch size if tiling matches.
                vsrc = sleaf.value
                vdst[dst_offset : dst_offset + ncopy, ...] = vsrc[:ncopy, ...]

    def get_info(self) -> dict[str, Any]:
        """
        Returns a comprehensive dictionary of the quadtree's current state,
        including structural metadata and leaf statistics.
        """
        leaves = list(self._iter_all_leaves())
        num_leaves = len(leaves)

        # Calculate the actual max level_idx currently reached in the tree
        current_max_level = (
            max((leaf.level_idx for leaf in leaves), default=0) + 1
        )  # +1 to convert from 0-based index to count

        return {
            "num_leaves": num_leaves,
            "channels": self.channels,
            "current_max_level": current_max_level,
            "max_level_cap": self.max_level_idx + 1,
            "patch_size": (self.patch_height, self.patch_width),
            "num_root_tiles": (self.ny_tiles, self.nx_tiles),
            "domain": {"x": (self.xmin, self.xmax), "y": (self.ymin, self.ymax)},
        }

    def __repr__(self):
        """
        Returns a developer-friendly string representation.
        """
        info = self.get_info()
        return (
            f"<Quadtree: leaves={info['num_leaves']}, "
            f"channels={info['channels']}, "
            f"level_range=[0-{info['current_max_level'] - 1}] (cap {self.max_level_idx}), "
            f"patch={self.patch_height}x{self.patch_width}>"
        )

    def summary(self):
        """
        Prints a formatted summary of the Quadtree.
        """
        info = self.get_info()
        print("--- Quadtree Summary ---")
        print(
            f"Structure:       {info['num_leaves']} leaves across "
            f"{info['num_root_tiles'][0]}x{info['num_root_tiles'][1]} root tiles"
        )
        print(
            f"Refinement:      Current Max Level: {info['current_max_level']} "
            f"(Cap: {info['max_level_cap']})"
        )
        print(
            f"Data:            {info['channels']} channels, "
            f"Patch Size: {info['patch_size'][0]}x{info['patch_size'][1]}"
        )
        print(f"Domain Bounds:   X: {info['domain']['x']}, Y: {info['domain']['y']}")
        print("------------------------")


def merge_quadtrees(qt_a: Quadtree, qt_b: Quadtree, mode: str = "stack") -> Quadtree:
    assert (qt_a.xmin, qt_a.xmax, qt_a.ymin, qt_a.ymax) == (
        qt_b.xmin,
        qt_b.xmax,
        qt_b.ymin,
        qt_b.ymax,
    ), "Quadtrees must have identical domains"

    # Check tile compatibility
    assert (
        qt_a.nx_tiles == qt_b.nx_tiles
        and qt_a.ny_tiles == qt_b.ny_tiles
        and abs(qt_a.tile_width - qt_b.tile_width) < 1e-12
        and abs(qt_a.tile_height - qt_b.tile_height) < 1e-12
    ), "Quadtrees must have identical tilings"

    max_level_idx = max(qt_a.max_level_idx, qt_b.max_level_idx)

    if mode == "stack":
        out_channels = (qt_a.channels or 0) + (qt_b.channels or 0)
    elif mode == "a":
        out_channels = qt_a.channels
    elif mode == "b":
        out_channels = qt_b.channels
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = Quadtree(
        qt_a.xmin,
        qt_a.xmax,
        qt_a.ymin,
        qt_a.ymax,
        max_level_idx=max_level_idx,
        channels=out_channels,
        tile_width=qt_a.tile_width,
        tile_height=qt_a.tile_height,
    )

    diff_a = compute_tree_diff(out, qt_a, include_values=False)
    apply_tree_diff(
        out,
        diff_a,
        update_values=False,
        allow_refine=True,
        allow_coarsen=False,
        maintain_balance=True,
    )
    diff_b = compute_tree_diff(out, qt_b, include_values=False)
    apply_tree_diff(
        out,
        diff_b,
        update_values=False,
        allow_refine=True,
        allow_coarsen=False,
        maintain_balance=True,
    )
    while out.ensure_2to1_balance():
        pass

    if out_channels > 0:
        if mode == "a":
            out.project_values_from(qt_a, dst_offset=0)
        elif mode == "b":
            out.project_values_from(qt_b, dst_offset=0)
        else:
            if qt_a.channels > 0:
                out.project_values_from(qt_a, dst_offset=0)
            if qt_b.channels > 0:
                out.project_values_from(qt_b, dst_offset=qt_a.channels)

    return out


def merge_many_quadtrees(trees: list[Quadtree], mode: str = "stack") -> Quadtree:
    assert len(trees) >= 1
    merged = trees[0]
    for t in trees[1:]:
        merged = merge_quadtrees(merged, t, mode=mode)
    return merged


def _merge_with(self: Quadtree, other: Quadtree, mode: str = "stack") -> Quadtree:
    return merge_quadtrees(self, other, mode=mode)


setattr(Quadtree, "merge_with", _merge_with)


@dataclass
class TreeDiff:
    # Metadata
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    nx_tiles: int
    ny_tiles: int
    max_level_idx: int
    channels: int

    ids: np.ndarray
    tuples: np.ndarray
    # Values are 4D: (N, C, Ph, Pw)
    values: np.ndarray | None
    ops: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.ids.shape[0])


def _decode_uids_with_params(
    ids: np.ndarray, max_level_idx: int, nx_tiles: int, ny_tiles: int
) -> np.ndarray:
    ids_i = ids.astype(np.uint64)
    XW = max_level_idx
    YW = max_level_idx
    LW = 6
    TXW = _ceil_log2(nx_tiles)
    TYW = _ceil_log2(ny_tiles)

    u = ids_i.view(np.uint64)
    shift = 0
    x_mask = (np.uint64(1) << XW) - 1 if XW > 0 else np.uint64(0)
    y_mask = (np.uint64(1) << YW) - 1 if YW > 0 else np.uint64(0)
    l_mask = (np.uint64(1) << LW) - 1
    tx_mask = (np.uint64(1) << TXW) - 1 if TXW > 0 else np.uint64(0)
    ty_mask = (np.uint64(1) << TYW) - 1 if TYW > 0 else np.uint64(0)

    x_idx = ((u >> shift) & x_mask) if XW > 0 else np.zeros_like(u)
    shift += XW
    y_idx = ((u >> shift) & y_mask) if YW > 0 else np.zeros_like(u)
    shift += YW
    level_idx = (u >> shift) & l_mask
    shift += LW
    tile_ix = ((u >> shift) & tx_mask) if TXW > 0 else np.zeros_like(u)
    shift += TXW
    tile_iy = ((u >> shift) & ty_mask) if TYW > 0 else np.zeros_like(u)

    out = np.stack(
        [
            tile_ix.astype(np.int64),
            tile_iy.astype(np.int64),
            level_idx.astype(np.int64),
            x_idx.astype(np.int64),
            y_idx.astype(np.int64),
        ],
        axis=1,
    ).astype(np.int32)
    return out


def compute_tree_diff(tree_a: Quadtree, tree_b: Quadtree, include_values: bool = True) -> TreeDiff:
    assert (tree_a.xmin, tree_a.xmax, tree_a.ymin, tree_a.ymax) == (
        tree_b.xmin,
        tree_b.xmax,
        tree_b.ymin,
        tree_b.ymax,
    ), "Quadtrees must have identical domains"
    assert (
        tree_a.nx_tiles == tree_b.nx_tiles
        and tree_a.ny_tiles == tree_b.ny_tiles
        and abs(tree_a.tile_width - tree_b.tile_width) < 1e-12
        and abs(tree_a.tile_height - tree_b.tile_height) < 1e-12
    ), "Quadtrees must have identical tilings"
    assert tree_a.max_level_idx >= tree_b.max_level_idx, (
        "tree_a.max_level_idx must be >= tree_b.max_level_idx"
    )

    a_export = tree_a.export_leaves(order="uid")
    b_export = tree_b.export_leaves(order="uid")

    def build_exact_sets(exp):
        exact: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        tiles = exp["tiles"]
        xy = exp["xy_idx"]
        N = tiles.shape[0]
        for i in range(N):
            key = (int(tiles[i, 0]), int(tiles[i, 1]))
            if key not in exact:
                exact[key] = set()
            level_idx = int(xy[i, 0])
            xi = int(xy[i, 1])
            yi = int(xy[i, 2])
            exact[key].add((level_idx, xi, yi))
        return exact

    exact_a = build_exact_sets(a_export)
    b_tiles = b_export["tiles"]
    b_xy = b_export["xy_idx"]
    b_ids = b_export["ids"].astype(np.uint64)
    b_vals = b_export["values"]

    changed_rows: list[int] = []
    ops: list[int] = []

    for i in range(b_tiles.shape[0]):
        tile_key = (int(b_tiles[i, 0]), int(b_tiles[i, 1]))
        lvl_b = int(b_xy[i, 0])
        xb = int(b_xy[i, 1])
        yb = int(b_xy[i, 2])
        t_b = (lvl_b, xb, yb)
        if tile_key in exact_a and t_b in exact_a[tile_key]:
            # Exact match in position.
            # If values need to be checked, we could, but here we assume topology diff.
            # If standard use is diffing topology, we skip.
            continue
        is_refine = False
        if tile_key in exact_a:
            for la in range(lvl_b - 1, -1, -1):
                xa = xb >> (lvl_b - la)
                ya = yb >> (lvl_b - la)
                if (la, xa, ya) in exact_a[tile_key]:
                    is_refine = True
                    break
        changed_rows.append(i)
        ops.append(1 if is_refine else 2)

    if not changed_rows:
        empty_ids = np.zeros((0,), dtype=np.uint64)
        empty_tuples = np.zeros((0, 5), dtype=np.int32)
        # Empty values: 4D (0, C, Ph, Pw)
        empty_values = (
            None
            if not include_values
            else (
                np.zeros(
                    (0, tree_b.channels, tree_b.patch_height, tree_b.patch_width), dtype=np.float32
                )
            )
        )
        return TreeDiff(
            xmin=tree_b.xmin,
            xmax=tree_b.xmax,
            ymin=tree_b.ymin,
            ymax=tree_b.ymax,
            nx_tiles=tree_b.nx_tiles,
            ny_tiles=tree_b.ny_tiles,
            max_level_idx=tree_b.max_level_idx,
            channels=tree_b.channels,
            ids=empty_ids,
            tuples=empty_tuples,
            values=empty_values,
            ops=np.zeros((0,), dtype=np.int8),
        )

    changed_rows_arr = np.array(changed_rows, dtype=np.int64)
    ids = b_ids[changed_rows_arr]
    tuples = np.empty((changed_rows_arr.shape[0], 5), dtype=np.int32)
    tuples[:, 0] = b_tiles[changed_rows_arr, 0]
    tuples[:, 1] = b_tiles[changed_rows_arr, 1]
    tuples[:, 2] = b_xy[changed_rows_arr, 0]
    tuples[:, 3] = b_xy[changed_rows_arr, 1]
    tuples[:, 4] = b_xy[changed_rows_arr, 2]

    values = None
    if include_values and b_vals is not None:
        values = b_vals[changed_rows_arr].astype(np.float32)

    return TreeDiff(
        xmin=tree_b.xmin,
        xmax=tree_b.xmax,
        ymin=tree_b.ymin,
        ymax=tree_b.ymax,
        nx_tiles=tree_b.nx_tiles,
        ny_tiles=tree_b.ny_tiles,
        max_level_idx=tree_b.max_level_idx,
        channels=tree_b.channels,
        ids=ids,
        tuples=tuples,
        values=values,
        ops=np.asarray(ops, dtype=np.int8),
    )


def _build_target_index(
    diff: TreeDiff,
) -> tuple[
    dict[tuple[int, int], list[set[tuple[int, int]]]],
    dict[tuple[int, int], set[tuple[int, int, int]]],
]:
    occ: dict[tuple[int, int], list[set[tuple[int, int]]]] = {}
    exact: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
    maxL = diff.max_level_idx

    for row in diff.tuples:
        tile_ix, tile_iy, lvl, xb, yb = map(int, row)
        key = (tile_ix, tile_iy)
        if key not in occ:
            occ[key] = [set() for _ in range(maxL + 1)]
            exact[key] = set()
        exact[key].add((lvl, xb, yb))
        for L_iter in range(0, lvl + 1):
            shift = lvl - L_iter
            xp = xb >> shift if shift > 0 else xb
            yp = yb >> shift if shift > 0 else yb
            occ[key][L_iter].add((xp, yp))
    return occ, exact


def apply_tree_diff(
    tree_a: Quadtree,
    diff: TreeDiff,
    update_values: bool = True,
    max_iters: int = 64,
    maintain_balance: bool = True,
    allow_refine: bool = True,
    allow_coarsen: bool = True,
) -> None:
    assert (tree_a.xmin, tree_a.xmax, tree_a.ymin, tree_a.ymax) == (
        diff.xmin,
        diff.xmax,
        diff.ymin,
        diff.ymax,
    ), "Domain mismatch"
    assert tree_a.nx_tiles == diff.nx_tiles and tree_a.ny_tiles == diff.ny_tiles, "Tiling mismatch"

    assert tree_a.max_level_idx >= diff.max_level_idx, "tree_a.max_level_idx too small for diff"

    is_delta = getattr(diff, "ops", None) is not None and diff.tuples.shape[0] == diff.ids.shape[0]

    def build_refine_index(diff_: TreeDiff):
        occ_ref: dict[tuple[int, int], list[set[tuple[int, int]]]] = {}
        exact_ref: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        maxL = diff_.max_level_idx
        for row, op in zip(diff_.tuples, diff_.ops):
            if int(op) != 1:
                continue
            tile_ix, tile_iy, lvl, xb, yb = map(int, row)
            key = (tile_ix, tile_iy)
            if key not in occ_ref:
                occ_ref[key] = [set() for _ in range(maxL + 1)]
                exact_ref[key] = set()
            exact_ref[key].add((lvl, xb, yb))
            for L_iter in range(0, lvl + 1):
                shift = lvl - L_iter
                xp = xb >> shift if shift > 0 else xb
                yp = yb >> shift if shift > 0 else yb
                occ_ref[key][L_iter].add((xp, yp))
        return occ_ref, exact_ref

    def build_coarsen_targets(diff_: TreeDiff):
        targets: dict[tuple[int, int], dict[int, set[tuple[int, int]]]] = {}
        for row, op in zip(diff_.tuples, diff_.ops):
            if int(op) != 2:
                continue
            tile_ix, tile_iy, lvl, xb, yb = map(int, row)
            key = (tile_ix, tile_iy)
            if key not in targets:
                targets[key] = {}
            targets[key].setdefault(lvl, set()).add((xb, yb))
        return targets

    if not is_delta:
        occ, exact = _build_target_index(diff)

    def refine_pred(cell: QuadCell) -> bool:
        if not allow_refine:
            return False
        if cell.level_idx >= min(tree_a.max_level_idx, diff.max_level_idx):
            return False
        level_idx, xi, yi = tree_a.cell_xy_index(cell)
        key = (cell.tile_ix, cell.tile_iy)
        if is_delta:
            occ_ref, exact_ref = refine_index
            if key not in occ_ref:
                return False
            if (level_idx, xi, yi) in exact_ref[key]:
                return False
            return (xi, yi) in occ_ref[key][level_idx]
        else:
            if key not in occ_full:
                return False
            if (level_idx, xi, yi) in exact_full[key]:
                return False
            return (xi, yi) in occ_full[key][level_idx]

    def coarsen_pred(node: QuadCell) -> bool:
        if not allow_coarsen:
            return False
        key = (node.tile_ix, node.tile_iy)
        level_idx, xi, yi = tree_a.cell_xy_index(node)
        if level_idx > diff.max_level_idx:
            return False
        if is_delta:
            tmap = coarsen_targets.get(key, None)
            if not tmap:
                return False
            for L0, xyset in tmap.items():
                if level_idx < L0:
                    continue
                shift = level_idx - L0
                xp = xi >> shift if shift > 0 else xi
                yp = yi >> shift if shift > 0 else yi
                if (xp, yp) in xyset:
                    return True
            return False
        else:
            return key in exact_full and (level_idx, xi, yi) in exact_full[key]

    if is_delta:
        refine_index = build_refine_index(diff)
        coarsen_targets = build_coarsen_targets(diff)
        occ_full = exact_full = None
    else:
        occ_full, exact_full = _build_target_index(diff)
        refine_index = coarsen_targets = None

    maintain_bal = False if is_delta else maintain_balance
    for _ in range(max_iters):
        changed = 0
        if allow_refine:
            changed += tree_a.refine_where(refine_pred)
            if maintain_bal:
                changed += tree_a.ensure_2to1_balance()
        if allow_coarsen:
            changed += tree_a.coarsen_where(coarsen_pred)
            if maintain_bal:
                changed += tree_a.ensure_2to1_balance()
        if changed == 0:
            break

    if update_values and diff.values is not None:
        id2row: dict[int, int] = {int(u): i for i, u in enumerate(diff.ids.tolist())}
        # Cdst = tree_a.channels
        for leaf in tree_a._iter_all_leaves():
            uid = int(tree_a.cell_uid_64bit(leaf))
            i = id2row.get(uid, None)
            if i is None:
                continue
            vsrc = diff.values[i]  # (C, Ph, Pw)
            if vsrc is None:
                continue
            leaf.value = vsrc.copy()


def __sub__(self: Quadtree, other: Quadtree) -> TreeDiff:
    return compute_tree_diff(other, self, include_values=False)


setattr(Quadtree, "__sub__", __sub__)


def __add__(self: Quadtree, other: TreeDiff) -> Quadtree:
    return apply_tree_diff(self, other)


setattr(Quadtree, "__add__", __add__)


def __eq__(self: Quadtree, other: object) -> bool:
    if not isinstance(other, Quadtree):
        return NotImplemented

    if (
        self.nx_tiles != other.nx_tiles
        or self.ny_tiles != other.ny_tiles
        or self.max_level_idx != other.max_level_idx
        or self.channels != other.channels
        or not np.isclose(self.tile_width, other.tile_width)
        or not np.isclose(self.tile_height, other.tile_height)
        or not np.isclose(self.xmin, other.xmin)
        or not np.isclose(self.xmax, other.xmax)
        or not np.isclose(self.ymin, other.ymin)
        or not np.isclose(self.ymax, other.ymax)
    ):
        return False

    a_leaves = self.export_leaves(order="uid")
    b_leaves = other.export_leaves(order="uid")

    if not np.array_equal(a_leaves["ids"], b_leaves["ids"]):
        return False

    a_values = a_leaves["values"]
    b_values = b_leaves["values"]

    if a_values is None and b_values is None:
        return True
    if a_values is None or b_values is None:
        return False

    return np.allclose(a_values, b_values)


setattr(Quadtree, "__eq__", __eq__)
