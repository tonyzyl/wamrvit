"""Public entrypoints for array-backed adaptive regridding.

Implementation modules depend on their owning modules, never this public surface.
"""

from wamrvit.quad.array_regrid.core import configure_array_regrid_num_threads
from wamrvit.quad.array_regrid.native import (
    array_regrid_native_from_sequence,
    warm_array_regrid_native_kernels,
)
from wamrvit.quad.array_regrid.uniform import (
    array_regrid_from_sequence,
    array_regrid_from_tensor,
    array_regrid_topology_from_detectors,
    object_regrid_from_tensor,
    warm_array_regrid_kernels,
)

__all__ = [
    "array_regrid_from_sequence",
    "array_regrid_from_tensor",
    "array_regrid_topology_from_detectors",
    "array_regrid_native_from_sequence",
    "configure_array_regrid_num_threads",
    "object_regrid_from_tensor",
    "warm_array_regrid_kernels",
    "warm_array_regrid_native_kernels",
]
