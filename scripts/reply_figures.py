"""Diagnostic figures for the response to reviewers.

These support the reply rather than the paper. They answer questions about how
a number was obtained, which is exactly the material a reviewer wants to see
once and which would clutter the manuscript.

  R2  the CLT-normalised comparison against the prior laminates (R3-10)
  R3  the ASTM D790 Annex A1.1 toe-compensation construction (R3-17)

The reply's R1 panel, the void-threshold sensitivity, is not here. It is built
from the backscattered-electron pixels rather than from any deposited table, so
it cannot run on this data and is kept with the micrographs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# R2 inputs. Fibre volume fractions and measured flexural moduli of the prior
# laminates are taken from Sergi et al.; the moduli were measured independently
# of the volume fractions, which is what makes the comparison meaningful.
# ---------------------------------------------------------------------------
PRIOR = [
    ("PA11 flax, [0]",            0.377,  5.13, "flax"),
    ("PA11 hybrid [0/0], basalt", 0.424,  9.82, "hybrid"),
    ("PA11 hybrid [0/0], flax",   0.424,  8.34, "hybrid"),
    ("PA11 hybrid [0/90], basalt", 0.479, 9.40, "hybrid"),
    ("PA11 hybrid [0/90], flax",   0.479, 9.79, "hybrid"),
]
RHO_BASALT, RHO_FLAX = 2.67, 1.45      # g/cm3, for the 50/50 mass split


def fibre_modulus(kind, E_basalt, E_flax):
    """Rule-of-mixtures fibre modulus for the two fabric types."""
    if kind == "flax":
        return E_flax
    vb = (1 / RHO_BASALT) / (1 / RHO_BASALT + 1 / RHO_FLAX)
    return vb * E_basalt + (1 - vb) * E_flax


def clt(vf, Ef, Em, xi=2.0):
    """Balanced 0/90 crossply average of the axial and Halpin-Tsai transverse ply."""
    e1 = Ef * vf + Em * (1 - vf)
    eta = (Ef / Em - 1) / (Ef / Em + xi)
    e2 = Em * (1 + xi * eta * vf) / (1 - eta * vf)
    return 0.5 * (e1 + e2)


def r2_clt(outdir):
    moduli = (pd.read_csv(DATA / "flexural_moduli.csv")
              .groupby("configuration")["E_D790_toe_compensated_GPa"].mean())
    ours = [("this work, basalt-long", 0.649, moduli["basalt_face"], "hybrid"),
            ("this work, flax-long", 0.649, moduli["flax_face"], "hybrid")]

    Eb, Ef_, Em = 80.0, 60.0, 1.5
    rows = []
    for name, vf, meas, kind in PRIOR + ours:
        pred = clt(vf, fibre_modulus(kind, Eb, Ef_), Em)
        rows.append((name, vf, pred, meas, meas / pred, name.startswith("this work")))

    # the conclusion must not depend on the constituent moduli chosen, so the
    # ratio is recomputed over a grid of plausible inputs
    ratios = []
    for b in (70, 75, 80, 85, 90, 95):
        for f_ in (40, 50, 60, 70, 80):
            for m in (1.0, 1.5, 2.0):
                p = [clt(v, fibre_modulus(k, b, f_), m) for _, v, _, k in PRIOR]
                o = [clt(v, fibre_modulus(k, b, f_), m) for _, v, _, k in ours]
                rp = np.mean([mm / pp for (_, _, mm, _), pp in zip(PRIOR, p)])
                ro = np.mean([mm / pp for (_, _, mm, _), pp in zip(ours, o)])
                ratios.append(ro / rp)
    ratios = np.array(ratios)

    fs.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.42 * fs.DOUBLE),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]},
                                   layout="constrained")
    y = np.arange(len(rows))[::-1]
    cols = [fs.WONG["blue"] if r[5] else fs.WONG["grey"] for r in rows]
    ax1.barh(y, [r[4] for r in rows], 0.62, color=cols, edgecolor="k", linewidth=0.9)
    for yi, r in zip(y, rows):
        ax1.text(r[4] + 0.012, yi, f"{r[4]:.2f}", va="center", fontsize=7.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r[0]}  ($V_f$ = {r[1]:.3f})" for r in rows], fontsize=7.4)
    ax1.set_xlabel("Measured modulus / CLT prediction at its own $V_f$")
    ax1.set_xlim(0, 0.88)
    ax1.minorticks_on()
    ax1.tick_params(axis="y", which="minor", left=False, right=False)

    ax2.hist(ratios, bins=18, color=fs.WONG["green"], edgecolor="k", linewidth=0.8)
    ax2.axvline(1.0, color="k", lw=1.0, ls=(0, (4, 3)))
    ax2.set_xlabel("Ratio, this work / prior")
    ax2.set_ylabel("Constituent combinations")
    ax2.annotate(f"{ratios.mean():.2f} ± {ratios.std():.2f}\n"
                 f"{100 * (ratios > 1).mean():.0f} % above unity",
                 xy=(0.30, 0.94), xycoords="axes fraction", va="top", fontsize=7.6)
    ax2.minorticks_on()

    for ax, l in ((ax1, "a"), (ax2, "b")):
        fs.panel_label(ax, l, dx=-0.10 if l == "a" else -0.24)
    fs.save(fig, "Reply_R2_clt_normalised_comparison", outdir)
    for r in rows:
        print(f"  R2: {r[0]:28s} Vf {r[1]:.3f}  CLT {r[2]:5.2f}  measured {r[3]:5.2f}  "
              f"ratio {r[4]:.3f}")
    print(f"  R2: ours/prior = {ratios.mean():.2f} ± {ratios.std():.2f} over "
          f"{len(ratios)} constituent combinations, {100 * (ratios > 1).mean():.0f} % above 1")


def r3_toe(outdir):
    curves = pd.read_csv(DATA / "flexural_curves.csv")
    moduli = pd.read_csv(DATA / "flexural_moduli.csv").set_index("specimen_id")
    spec = "16Hybr_Bas_Long_1"
    g = curves[curves.specimen_id == spec].sort_values("strain_pct")
    e, s = g.strain_pct.to_numpy(), g.stress_MPa.to_numpy()
    toe = moduli.loc[spec, "toe_intercept_pct"]
    E = moduli.loc[spec, "E_D790_toe_compensated_GPa"]

    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.ONE_HALF, 0.72 * fs.ONE_HALF), layout="constrained")
    ax.plot(e, s, color=fs.WONG["grey"], lw=1.6, label="as recorded")
    ax.plot(e - toe, s, color=fs.WONG["blue"], lw=1.6, label="toe-compensated")
    xs = np.array([0.0, 0.42])
    ax.plot(xs + toe, xs * E * 10, color=fs.WONG["red"], lw=1.2, ls=(0, (5, 2)),
            label="steepest 0.15 % chord, extrapolated")
    ax.axvspan(0.05, 0.25, color=fs.WONG["grey"], alpha=0.18, lw=0)
    ax.annotate(f"toe intercept\n{toe:.4f} % strain", xy=(toe, 0),
                xytext=(0.16, 12), fontsize=7.6,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Flexural strain (%)")
    ax.set_ylabel("Flexural stress (MPa)")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=7.4, handlelength=1.8)
    ax.minorticks_on()
    fs.save(fig, "Reply_R3_toe_compensation", outdir)
    print(f"  R3: {spec} toe intercept {toe:.4f} %, chord modulus {E:.2f} GPa")


if __name__ == "__main__":
    import sys
    # Same convention as the paper figure scripts: an optional output directory,
    # defaulting to the current one. Previously both arguments were required and
    # the script raised IndexError when run the way the README describes.
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    r2_clt(out)
    r3_toe(out)
