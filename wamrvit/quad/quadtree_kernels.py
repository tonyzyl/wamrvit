import numpy as np
from numba import njit


# 1. Faster Nearest Neighbor Resizing (Replaces _resize_patch_nearest)
@njit(boundscheck=False, fastmath=True, cache=True)
def resize_patch_nearest_jit(patch: np.ndarray, target_h: int, target_w: int):
    C, H, W = patch.shape
    out = np.empty((C, target_h, target_w), dtype=patch.dtype)
    for c in range(C):
        for i in range(target_h):
            # fastmath allows faster float truncation
            r = min(int(i * (H / target_h)), H - 1)
            for j in range(target_w):
                c_idx = min(int(j * (W / target_w)), W - 1)
                out[c, i, j] = patch[c, r, c_idx]
    return out


# 1b. Bilinear Resizing (align_corners=True) — smooth upsampling that preserves
#     boundary values exactly and near-perfectly conserves the mean.
#     Output pixel i maps to input pixel i * (H-1) / (target_h-1), so corner
#     pixels of child and parent coincide — critical for continuity across
#     adjacent quadtree cells at all refinement levels.
@njit(boundscheck=False, fastmath=True, cache=True)
def resize_patch_bilinear_jit(patch: np.ndarray, target_h: int, target_w: int):
    C, H, W = patch.shape
    out = np.empty((C, target_h, target_w), dtype=np.float64)
    h_scale = (H - 1.0) / (target_h - 1.0) if target_h > 1 else 0.0
    w_scale = (W - 1.0) / (target_w - 1.0) if target_w > 1 else 0.0
    for c in range(C):
        for i in range(target_h):
            src_y = i * h_scale
            y0 = int(src_y)
            if y0 >= H - 1:
                y0 = H - 2 if H > 1 else 0
            y1 = y0 + 1
            dy = src_y - y0
            for j in range(target_w):
                src_x = j * w_scale
                x0 = int(src_x)
                if x0 >= W - 1:
                    x0 = W - 2 if W > 1 else 0
                x1 = x0 + 1
                dx = src_x - x0
                val = (
                    patch[c, y0, x0] * (1.0 - dy) * (1.0 - dx)
                    + patch[c, y1, x0] * dy * (1.0 - dx)
                    + patch[c, y0, x1] * (1.0 - dy) * dx
                    + patch[c, y1, x1] * dy * dx
                )
                out[c, i, j] = val
    return out


# 2. Faster 2x2 Block Averaging (Replaces the stitched.reshape().mean() in coarsen)
@njit(boundscheck=False, fastmath=True, cache=True)
def downsample_2x_block_mean_jit(stitched: np.ndarray):
    C, H2, W2 = stitched.shape
    H, W = H2 // 2, W2 // 2
    out = np.empty((C, H, W), dtype=np.float64)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                # Manual unrolling is significantly faster than numpy reshapes in Numba
                val = (
                    stitched[c, i * 2, j * 2]
                    + stitched[c, i * 2 + 1, j * 2]
                    + stitched[c, i * 2, j * 2 + 1]
                    + stitched[c, i * 2 + 1, j * 2 + 1]
                )
                out[c, i, j] = val * 0.25
    return out


# 3. Faster Morton Coding (Replaces _morton2D)
@njit(boundscheck=False, fastmath=True, cache=True)
def morton2D_jit(x: int, y: int, bits: int = 32):
    z = 0
    for i in range(bits):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z


@njit(boundscheck=False, fastmath=True, cache=True)
def normalize_centers_jit(
    centers: np.ndarray,
    xmin: float,
    ymin: float,
    cell_min: float,
    Lx: float,
    Ly: float,
    Lcell: float,
):
    N = centers.shape[0]

    for i in range(N):
        centers[i, 0] = (centers[i, 0] - xmin) / Lx
        centers[i, 1] = (centers[i, 1] - ymin) / Ly
        centers[i, 2] = (centers[i, 2] - cell_min) / Lcell
