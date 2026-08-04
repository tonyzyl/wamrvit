"""Public compatibility surface for array-backed adaptive regridding."""

from wamrvit.quad.array_regrid_native import (
    array_regrid_native_from_sequence,
    warm_array_regrid_native_kernels,
)
from wamrvit.quad.array_regrid_runtime import configure_array_regrid_num_threads
from wamrvit.quad.array_regrid_uniform import (
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
