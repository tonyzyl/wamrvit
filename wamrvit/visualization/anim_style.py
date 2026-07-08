"""Shared figure-layout engine for the animation scripts.

A single source of truth so `animate_regular.py` (imshow) and
`animate_adaptive.py` (plot_quadtree) place their data axes and colorbar at
the same figure-relative rectangles. Alignment is a consequence of
`compute_layout` being DETERMINISTIC: both paths feed the same data aspect
(regular: array H/W; adaptive: quadtree-bounds H/W, both crop-adjusted) and
therefore get an identical canvas. The data box is pre-sized to the data
aspect, so `set_aspect("equal")` fills it without recentering.

Pixel alignment depends on these fixed rects, NOT on `tight_layout()` /
`bbox_inches="tight"`, both of which reflow per figure and must be avoided in
the touched render paths.
"""
import matplotlib.pyplot as plt

TITLE_FONTSIZE = 14
TITLE_PAD = 10
AXIS_LABEL_FS = 12
TICK_FS = 10

# Layout budget in INCHES (dpi-independent; pixels = inches * dpi at
# plt.figure). Margins are constant so the canvas hugs the data box for any
# aspect while staying deterministic. Title space is ALWAYS reserved so a
# titled panel and an untitled one share a canvas.
DATA_LONG_IN = 5.0       # longest side of the data box
TITLE_IN = 0.45          # top margin (title may be present or absent)
LABEL_LEFT_IN = 0.70     # y-label + tick labels, when show_axes
LABEL_BOTTOM_IN = 0.55   # x-label + tick labels, when show_axes
NOAXES_PAD_IN = 0.10     # margin on a side with no labels (axes off)
CBAR_GAP_IN = 0.12       # gap between data box and colorbar
CBAR_W_IN = 0.18         # colorbar bar width
CBAR_LABEL_IN = 0.55     # colorbar tick labels
PAD_IN = 0.10            # right pad


def compute_layout(aspect, *, show_axes):
    """Return ((fig_w_in, fig_h_in), ax_rect, cax_rect) for a data box of the
    given aspect (= data_height / data_width).

    Deterministic: identical (aspect, show_axes) -> identical output, which is
    what aligns every panel of a dataset. Rects are matplotlib
    [left, bottom, width, height] in figure fractions.
    """
    aspect = float(aspect)
    if aspect >= 1.0:                      # portrait: height is the long side
        data_h = DATA_LONG_IN
        data_w = DATA_LONG_IN / aspect
    else:                                  # landscape: width is the long side
        data_w = DATA_LONG_IN
        data_h = DATA_LONG_IN * aspect

    left = LABEL_LEFT_IN if show_axes else NOAXES_PAD_IN
    bottom = LABEL_BOTTOM_IN if show_axes else NOAXES_PAD_IN
    top = TITLE_IN

    fig_w = left + data_w + CBAR_GAP_IN + CBAR_W_IN + CBAR_LABEL_IN + PAD_IN
    fig_h = top + data_h + bottom

    ax_rect = [left / fig_w, bottom / fig_h, data_w / fig_w, data_h / fig_h]
    cax_left = left + data_w + CBAR_GAP_IN
    cax_rect = [cax_left / fig_w, bottom / fig_h, CBAR_W_IN / fig_w, data_h / fig_h]
    return (fig_w, fig_h), ax_rect, cax_rect


def resolve_aspect(domain_h, domain_w, *, x_frac=None, y_frac=None, override=None):
    """Display aspect (H/W) after an optional fractional crop.

    `override` (from `animation.data_aspect`) wins verbatim when not None — the
    robustness escape hatch for when the regular and adaptive aspect sources
    ever diverge. `x_frac`/`y_frac` are (lo, hi) fractions over W and H.
    """
    if override is not None:
        return float(override)
    fh = 1.0 if y_frac is None else (float(y_frac[1]) - float(y_frac[0]))
    fw = 1.0 if x_frac is None else (float(x_frac[1]) - float(x_frac[0]))
    return (float(domain_h) * fh) / (float(domain_w) * fw)


def make_aligned_axes(fig, ax_rect, cax_rect):
    """Add a data axes and a colorbar axes at the given fixed rectangles.

    Used directly by the per-frame GIF `update` closures (which `fig.clf()`
    and re-add axes each frame) so every frame reuses one precomputed layout.
    """
    ax = fig.add_axes(ax_rect)
    cax = fig.add_axes(cax_rect)
    return ax, cax


def make_aligned_figure(aspect, dpi, *, show_axes):
    """Single-shot convenience: compute the layout, create the figure, add the
    two axes. Returns (fig, ax, cax). For PNG snapshots and the regular path.
    """
    figsize, ax_rect, cax_rect = compute_layout(aspect, show_axes=show_axes)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax, cax = make_aligned_axes(fig, ax_rect, cax_rect)
    return fig, ax, cax


def finalize_axes(ax, *, title, show_axes):
    """Apply the shared title/axis treatment. Call AFTER rendering."""
    if show_axes:
        ax.set_xlabel("x", fontsize=AXIS_LABEL_FS)
        ax.set_ylabel("y", fontsize=AXIS_LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
    else:
        ax.axis("off")
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)


def crop_to_frac(arr, *, x_frac, y_frac):
    """Crop a (..., H, W) array to fractional sub-ranges along H and W.

    `x_frac`/`y_frac` are (lo, hi) in [0, 1] over W and H respectively, or
    None for no crop on that axis. Mirrors the spatial crop that
    `plot_quadtree(x_frac=, y_frac=)` applies on the adaptive path, so a
    regular render of the same sub-region has the same extent.
    """
    if x_frac is None and y_frac is None:
        return arr
    h, w = arr.shape[-2], arr.shape[-1]
    y0, y1 = (0.0, 1.0) if y_frac is None else (float(y_frac[0]), float(y_frac[1]))
    x0, x1 = (0.0, 1.0) if x_frac is None else (float(x_frac[0]), float(x_frac[1]))
    ry0, ry1 = int(y0 * h), max(int(y0 * h) + 1, int(y1 * h))
    rx0, rx1 = int(x0 * w), max(int(x0 * w) + 1, int(x1 * w))
    return arr[..., ry0:ry1, rx0:rx1]
