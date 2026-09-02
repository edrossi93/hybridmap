"""
The three diagnostics a user needs to justify their own choices.

A histogram with the cuts drawn on it, because a cut is defended by the shape
of the distribution it sits in and by nothing else. A sweep, because the honest
report of a chosen cut is how much the answer moves when it is chosen
differently. A prediction against the measurement, because that is the question.

These are working figures. The paper's figures are built by the fig*.py scripts
and are not made here.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

# Written to file more often than shown, and a user on a machine with no
# display should not have to know about backends to get a PNG.
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

# figstyle lives beside the package rather than inside it: it carries the
# paper's figure conventions, which are not this tool's to impose. Used when
# present, ignored when the package is lifted out on its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import figstyle as _figstyle
except ImportError:
    _figstyle = None


def _style():
    if _figstyle is not None:
        _figstyle.apply()


def _bands(cfg):
    """
    Modulus intervals in the config, each with the phases that claim it.

    Phases that share an interval and differ only by zone, which is how a
    compliant band is usually split, are drawn once rather than shaded twice.
    """
    groups = {}
    for p in cfg.phases:
        lo = -np.inf if p.lower is None else cfg.cuts[p.lower]
        hi = np.inf if p.upper is None else cfg.cuts[p.upper]
        groups.setdefault((lo, hi), []).append(p)
    return sorted(groups.items())


def modulus_histogram(values, cfg, unit="GPa", bins=120, log=False, ax=None):
    """
    The distribution the cuts are drawn from, with the cuts on it and named.

    This is the figure that justifies a cut, or fails to. A cut is defended by
    the shape of the distribution it sits in, so the phases it produces are
    shaded in their own colours: a band with no visible population on either
    side of its edge is a cut that has not been earned.
    """
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    ax.hist(v, bins=bins, color="#9A9A9A", edgecolor="none")

    lo_lim, hi_lim = ax.get_xlim()
    colours = cfg.colours
    for i, ((lo, hi), phases) in enumerate(_bands(cfg)):
        colour = colours.get(phases[0].name) or f"C{i}"
        ax.axvspan(max(lo, lo_lim), min(hi, hi_lim), color=colour, alpha=0.18,
                   lw=0)
        centre = (max(lo, lo_lim) + min(hi, hi_lim)) / 2
        ax.annotate(" / ".join(p.name for p in phases), xy=(centre, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -8), textcoords="offset points",
                    ha="center", va="top", fontsize=8)

    for name, value in sorted(cfg.cuts.items(), key=lambda kv: kv[1]):
        ax.axvline(value, color="black", lw=1.4)
        ax.annotate(f"{name} = {value:g}", xy=(value, 0.55),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=8, rotation=90, va="center")

    ax.set_xlim(lo_lim, hi_lim)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(f"modulus ({unit})")
    ax.set_ylabel("indents")
    ax.set_title(f"{len(v)} indents", fontsize=10)
    return ax.get_figure()


def sweep_plot(table, cut_names, reference=None, ax=None):
    """
    Every bound in the swept grid, against the first cut.

    One line per combination of the remaining cuts, so a two-cut sweep reads as
    a small family of curves rather than as a table of numbers nobody checks.
    """
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")

    first, *rest = [f"{n}_GPa" for n in cut_names]
    if rest:
        for i, (key, group) in enumerate(table.groupby(rest)):
            # Labelled with the cut's own name, not the column it is stored
            # under: the reader wrote "upper" in the config and has no reason
            # to recognise "upper_GPa".
            label = ", ".join(f"{n} {v:g}" for n, v in
                              zip(cut_names[1:],
                                  key if isinstance(key, tuple) else (key,)))
            ax.plot(group[first], group.Voigt_GPa, "o-", ms=3, color=f"C{i}",
                    lw=1.2, label=label)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    else:
        ax.plot(table[first], table.Voigt_GPa, "o-", ms=3, lw=1.2)

    if reference is not None:
        ax.axhline(reference, color="k", lw=1.0, ls="--")
        ax.annotate("as configured", xy=(0.01, reference), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel(f"{cut_names[0]} cut (GPa)")
    ax.set_ylabel("Voigt bound (GPa)")
    return ax.get_figure()


def prediction_vs_measurement(bounds, measured, names=("Voigt", "Hill", "Reuss"),
                              unit="GPa", ax=None):
    """The bounds as vertical lines, each measurement as a point with its spread."""
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 2.2 + 0.4 * len(measured)),
                             layout="constrained")

    for i, (name, value) in enumerate(zip(names, bounds)):
        ax.axvline(value, color=f"C{i}", lw=1.4)
        ax.annotate(f"{name} {value:.2f}", xy=(value, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -10), textcoords="offset points",
                    fontsize=8, color=f"C{i}", rotation=90, va="top")

    y = np.arange(len(measured))
    ax.errorbar([m.value for m in measured], y,
                xerr=[m.sd if m.sd else 0.0 for m in measured],
                fmt="o", color="k", capsize=3, lw=1.2, ms=4)
    ax.set_yticks(y)
    ax.set_yticklabels([m.label for m in measured], fontsize=8)
    ax.set_ylim(-0.6, len(measured) - 0.4)
    ax.set_xlabel(f"modulus ({unit})")
    return ax.get_figure()


def save(fig, stem, outdir="."):
    """Write a figure, through figstyle's writer when it is available."""
    if _figstyle is not None:
        return _figstyle.save(fig, stem, outdir)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=400)
    return out / f"{stem}.pdf"
