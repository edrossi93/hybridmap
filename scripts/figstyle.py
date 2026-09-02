"""Shared plotting style for the manuscript figures.

Deliberately plain matplotlib so that anyone with numpy, pandas, scipy and
matplotlib can regenerate every figure from the deposited CSVs without
installing anything else. The look is the one used throughout the paper:
heavy fully boxed axes, large upright type, ticks pointing in on all four
sides.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Column widths used by Composites Part A, in inches.
SINGLE, ONE_HALF, DOUBLE = 3.50, 5.35, 7.20

# The palette of the submitted figures, kept so that the regenerated figures sit
# beside the untouched ones without a visible change of language.
PAPER = {
    "flax":         "#E8B33C",   # gold
    "matrix":       "#E0701B",   # orange
    "mixed-volume": "#7FC1E4",   # light blue
    "basalt":       "#2E7EB8",   # dark blue
    "qc":           "#B8B8B8",   # grey
    "voigt":        "#2E7EB8",
    "hill":         "#7FC1E4",
    "reuss":        "#C9E2F2",
    "corrected":    "#D9531E",
    "observed":     "#000000",
}

# Okabe-Ito, kept for anything the paper's own palette does not cover.
WONG = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#999999",
}

# One colour per constituent, used identically in every figure so that a
# reader can carry the mapping from one panel to the next.
CONSTITUENT = {k: PAPER[k] for k in
               ("flax", "matrix", "mixed-volume", "basalt", "qc")}

CONFIG = {
    "basalt_face": PAPER["basalt"],
    "flax_face":   PAPER["flax"],
}

# Canonical axis labels. The nomenclature is the one set by the untouched
# Figures 3 and 4 of the submission, "Quantity, symbol (unit)", and every
# regenerated figure draws from here so the wording cannot drift apart again.
# The measurement depth is deliberately absent: it belongs in the caption, not
# repeated on four axes.
AXIS = {
    "E":     "Elastic modulus, E (GPa)",
    "H":     "Hardness, H (GPa)",
    "PA":    "Phase angle, \u03c6 (\u00b0)",
    "depth": "Depth (nm)",
    "load":  "Load (mN)",
}

CONFIG_LABEL = {
    "basalt_face": "basalt-longitudinal",
    "flax_face":   "flax-longitudinal",
}


def apply():
    """Install the house style. Call once before building a figure."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        # fully boxed, heavy frame
        "axes.linewidth": 1.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        # keep text as text in the PDF so the publisher can reflow it
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, letter, dx=-0.175, dy=1.075):
    """Lowercase bold panel letter, placed outside the axes box.

    Offsets are in axes fractions, so the letter moves with the panel size.
    Prefer panel_letters() below, which keeps the letters aligned across
    panels of different sizes.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top", ha="left")


def panel_letters(fig, axes, letters=None, pad_in=0.02, fontsize=15,
                  where="left"):
    """Panel letters placed from the axes boxes, so they always line up.

    where="left" sets each letter back into the margin beside the panel, clear
    of the tick labels and the axis label. The set-back is measured, not
    guessed, and is shared within a column so a stack of panels puts its letters
    on one vertical line. Across a row each panel keeps its own set-back: a
    neighbour with long category labels would otherwise push its partner's
    letter off the canvas.

    where="above" puts the letter just above the panel's top-left corner
    instead. That sits much closer to the panel, because it does not have to
    clear the rotated axis label, and letters in a row still align through the
    shared top edge. It needs headroom, so the caller must leave a top margin;
    the figure is checked afterwards and a warning is printed if a letter would
    fall off the canvas.
    """
    if letters is None:
        letters = [chr(ord("a") + i) for i in range(len(axes))]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    w_in, h_in = fig.get_size_inches()

    if where == "above":
        for ax, letter in zip(axes, letters):
            box = ax.get_position()
            y = box.y1 * h_in + pad_in
            if y + fontsize / 72.0 <= h_in:
                fig.text(box.x0, y / h_in, letter, fontsize=fontsize,
                         fontweight="bold", va="bottom", ha="left")
            else:
                # No headroom, which happens under constrained layout: it fills
                # the canvas and will not reserve a top margin. The letter goes
                # just inside the same corner instead, on a light backing so it
                # reads over a micrograph. The corner is the same either way, so
                # the convention holds across the figure set.
                ax.text(0.012, 0.988, letter, transform=ax.transAxes,
                        fontsize=fontsize, fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="square,pad=0.12", fc="w", ec="none",
                                  alpha=0.75))
        return

    def own(ax):
        box = ax.get_window_extent(renderer)
        return (box.x0 - min(box.x0, ax.get_tightbbox(renderer).x0)) / dpi

    groups = {}
    for ax in axes:
        groups.setdefault(round(ax.get_position().x0, 3), []).append(ax)
    setback = {}
    for members in groups.values():
        common = max(own(ax) for ax in members) + pad_in + fontsize / 72.0 * 0.62
        for ax in members:
            setback[ax] = common

    for ax, letter in zip(axes, letters):
        width_in = ax.get_position().width * w_in
        ax.text(-setback[ax] / width_in, 1.0, letter, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", ha="left",
                clip_on=False)


# Point sizes and line weights of the house style the paper figures were drawn
# in. Sizes are multiplied by the per-figure font_scale; weights are not, so a
# dense multi-panel figure gets smaller type without thinner axes.
PRESETS = {
    "bold": dict(axes_label=14.5, tick_label=12.0, panel_label=17.0,
                 cbar_label=14.5, cbar_tick=11.5, legend=11.5, annot=11.5,
                 box_lw=1.9, tick_lw=1.5, line_lw=2.4, tick_len=5.5),
    "print": dict(axes_label=7.0, tick_label=6.0, panel_label=8.0,
                  cbar_label=7.0, cbar_tick=6.0, legend=6.0, annot=6.0,
                  box_lw=0.8, tick_lw=0.8, line_lw=1.2, tick_len=3.0),
}

# Real Helvetica or Arial first, then the metric-compatible clones, so the
# figures keep their proportions on a machine that has neither.
_SANS = ["Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans",
         "TeX Gyre Heros", "Liberation Sans", "FreeSans", "DejaVu Sans"]

# Wong's colour-blind-safe order, black last so the first categories are the
# colourful ones.
_CYCLE = [WONG["blue"], WONG["orange"], WONG["green"], WONG["red"],
          WONG["sky"], WONG["purple"], WONG["yellow"], WONG["grey"]]


def set_style(preset="bold", font_scale=1.0):
    """Apply the figure house style to matplotlib's rcParams.

    Returns the resolved size/weight dict. Every figure script calls this once
    before building, with the font_scale that suits its panel count.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    p = dict(PRESETS[preset])
    for k in ("axes_label", "tick_label", "panel_label", "cbar_label",
              "cbar_tick", "legend", "annot"):
        p[k] *= font_scale

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _SANS,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
        "font.size": p["tick_label"],
        "axes.labelsize": p["axes_label"],
        "axes.titlesize": p["axes_label"],
        "xtick.labelsize": p["tick_label"],
        "ytick.labelsize": p["tick_label"],
        "legend.fontsize": p["legend"],
        "figure.titlesize": p["axes_label"],

        # Fully boxed, thick black spines.
        "axes.linewidth": p["box_lw"],
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.facecolor": "white",
        "axes.labelcolor": "black",
        "axes.axisbelow": True,
        "axes.labelpad": 3.0,
        "axes.titlepad": 4.0,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": p["tick_lw"],
        "ytick.major.width": p["tick_lw"],
        "xtick.minor.width": p["tick_lw"] * 0.8,
        "ytick.minor.width": p["tick_lw"] * 0.8,
        "xtick.major.size": p["tick_len"],
        "ytick.major.size": p["tick_len"],
        "xtick.minor.size": p["tick_len"] * 0.55,
        "ytick.minor.size": p["tick_len"] * 0.55,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,

        "lines.linewidth": p["line_lw"],
        "lines.markersize": 6.0,
        "lines.markeredgewidth": p["tick_lw"],
        "patch.linewidth": p["tick_lw"],
        "errorbar.capsize": 3.0,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.4,
        "legend.handletextpad": 0.5,

        "axes.prop_cycle": mpl.cycler(color=_CYCLE),

        # Vector text stays text: journals need it editable, not outlined.
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 6,

        "image.interpolation": "nearest",
        "image.resample": False,

        # One layout engine throughout, so panels never need manual nudging.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": max(0.03, 0.22 * p["axes_label"] / 72.0),
        "figure.constrained_layout.h_pad": max(0.03, 0.22 * p["axes_label"] / 72.0),
        "figure.constrained_layout.wspace": max(0.025, 0.18 * p["axes_label"] / 72.0),
        "figure.constrained_layout.hspace": max(0.025, 0.18 * p["axes_label"] / 72.0),
    })
    return p


def _overlap_area(a, b):
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _live_texts(fig, axes):
    """Visible, non-empty text artists, minus tick labels for ticks that fall
    outside the current limits. Matplotlib keeps those artists around, and
    counting them produces collision reports for labels nobody can see."""
    dead = set()
    for ax in axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(map(float, lim))
            eps = max(hi - lo, 1.0) * 1e-9
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                try:
                    loc = float(tick.get_loc())
                except Exception:
                    continue
                if loc < lo - eps or loc > hi + eps:
                    dead.add(tick.label1)

    renderer = fig.canvas.get_renderer()
    out = []
    for art in fig.findobj(mpl.text.Text):
        if art in dead or not art.get_visible() or not art.get_text().strip():
            continue
        try:
            bb = art.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            out.append((art, bb))
    return out


def layout_report(fig, tolerance_px=1.5, min_overlap_frac=0.12):
    """Measure a drawn figure for clipping and collisions.

    Diagnostic only: it reads the rendered geometry and changes nothing. The
    figure scripts print whatever it returns, so a non-empty report is a
    prompt to adjust the layout by hand rather than an automatic repair.

    Returns a dict of finding-name to list of hits; every list empty means the
    layout is clean. Overlaps smaller than `min_overlap_frac` of the smaller
    artist are ignored, because antialiased glyph extents touch harmlessly.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figbox = fig.bbox
    axes = [a for a in fig.axes if a.get_visible()]

    def outside(bb):
        return (bb.x0 < figbox.x0 - tolerance_px or bb.y0 < figbox.y0 - tolerance_px
                or bb.x1 > figbox.x1 + tolerance_px or bb.y1 > figbox.y1 + tolerance_px)

    clipped_axes = []
    for i, ax in enumerate(axes):
        bb = ax.get_tightbbox(renderer)
        if bb is not None and outside(bb):
            clipped_axes.append(i)

    overlapping_axes = []
    for i, ai in enumerate(axes):
        for j, aj in enumerate(axes[i + 1:], start=i + 1):
            area = _overlap_area(ai.bbox, aj.bbox)
            if area > tolerance_px ** 2:
                overlapping_axes.append((i, j, round(area, 1)))

    texts = _live_texts(fig, axes)

    clipped_text = [(t.get_text(), tuple(round(v, 1) for v in bb.bounds))
                    for t, bb in texts if outside(bb)]

    overlapping_text = []
    for idx, (ta, ba) in enumerate(texts):
        for tb, bb in texts[idx + 1:]:
            area = _overlap_area(ba, bb)
            if area <= tolerance_px ** 2:
                continue
            smaller = max(min(ba.width * ba.height, bb.width * bb.height), 1e-9)
            if area / smaller >= min_overlap_frac:
                overlapping_text.append((ta.get_text(), tb.get_text(), round(area, 1)))

    text_in_wrong_panel = []
    for text, bb in texts:
        for i, ax in enumerate(axes):
            if ax is text.axes:
                continue
            area = _overlap_area(bb, ax.bbox)
            if area > tolerance_px ** 2 and area / max(bb.width * bb.height, 1e-9) > min_overlap_frac:
                text_in_wrong_panel.append((text.get_text(), i, round(area, 1)))

    return {"clipped axes": clipped_axes,
            "overlapping axes": overlapping_axes,
            "clipped text": clipped_text,
            "overlapping text": overlapping_text,
            "text over another panel": text_in_wrong_panel}


def save(fig, stem, outdir="."):
    """Write a figure as PDF and PNG under one stem.

    The paper figures each call savefig twice with the same 400 dpi and the
    same pair of extensions. The reply figures called a helper for it that was
    never actually written, so they raised AttributeError on their last line.
    This is that helper, doing exactly what the paper scripts do inline.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=400)
    return out / f"{stem}.pdf"


def assert_inside(ax, artists, name="", tol_px=1.0):
    """Check that text really sits inside its axes, and say so out loud.

    Eyeballing a rendered figure is how a label ends up straddling a spine:
    a string placed in data coordinates has a width in points, so whether it
    fits depends on the panel width, the font size and the data limits at once.
    This measures the rendered extent against the axes box and reports the
    overflow in pixels on each side.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = ax.get_window_extent(r)
    ok = True
    for a in artists if isinstance(artists, (list, tuple)) else [artists]:
        bb = a.get_window_extent(r)
        over = {"left": box.x0 - bb.x0, "right": bb.x1 - box.x1,
                "bottom": box.y0 - bb.y0, "top": bb.y1 - box.y1}
        bad = {k: round(v, 1) for k, v in over.items() if v > tol_px}
        label = repr(a.get_text())[:28]
        if bad:
            ok = False
            print(f"    OVERFLOW {name} {label}: {bad} px")
        else:
            print(f"    inside   {name} {label}")
    return ok
