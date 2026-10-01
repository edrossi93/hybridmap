"""The planned flexural comparisons, their effect sizes, and three corrections.

Supplementary Table S7 reports four comparisons under one Holm correction:
chord modulus, peak tangent modulus, flexural strength and softening. Two
alternative families are computed beside it, so that the dependence of the
conclusion on the choice of family is shown by computing rather than by
argument.

Stiffness as one comparison: the chord modulus and the peak tangent modulus are
two measures of the same property on the same specimens, so putting both in one
family is conservative, and because Holm enforces monotonicity the corrected
value for the chord follows from the tangent rather than from its own evidence.
This family counts the two stiffness measures as one comparison.

Strain at break in place of softening: chord modulus, flexural strength and
strain at break. This family supplies the corrected p for strain at break.

Under either alternative family the stiffness difference corrects to
p = 0.0357 rather than 0.0675, on the nine specimens deposited. The paper rests
nothing on that test: it reports that a difference of about 5 % cannot be
excluded and that it lies below what the construction itself can resolve.

Every quantity is defined here rather than inherited, so that each one has a
script behind it:

  chord modulus    ASTM D790 chord over 0.05 to 0.25 % strain with the A1.1
                   toe compensation, as tabulated in flexural_moduli.csv
  peak tangent     the largest slope of a straight line fitted to the curve
                   over a sliding 0.10 % strain window, assigned to the mean
                   strain of that window
  softening        the fall in that tangent modulus from its peak to 0.55 %
                   strain, as a percentage of the peak
  strength         the maximum stress of the record
  strain at break  the last strain of the record, descriptive only

Run: python flexural_comparisons.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

WINDOW_PCT = 0.10          # sliding window for the tangent modulus
SOFTEN_AT_PCT = 0.55       # where the softening is read off


def tangent_curve(g, window=WINDOW_PCT):
    """Tangent modulus against strain, in GPa, by local straight-line fit."""
    s = g.sort_values("strain_pct")
    e, y = s.strain_pct.to_numpy(), s.stress_MPa.to_numpy()
    strain, modulus = [], []
    for i in range(len(e)):
        m = (e >= e[i]) & (e <= e[i] + window)
        if m.sum() >= 5:
            strain.append(e[m].mean())
            # MPa per percent strain is ten times GPa per unit strain
            modulus.append(np.polyfit(e[m], y[m], 1)[0] / 10.0)
    return np.asarray(strain), np.asarray(modulus)


def per_specimen():
    fx = pd.read_csv(DATA / "flexural_curves.csv")
    fm = pd.read_csv(DATA / "flexural_moduli.csv")
    rows = []
    for (cfg, sid), g in fx.groupby(["configuration", "specimen_id"]):
        strain, modulus = tangent_curve(g)
        peak_tangent = float(modulus.max())
        at = float(np.interp(SOFTEN_AT_PCT, strain, modulus))
        rows.append({
            "configuration": cfg, "specimen_id": sid,
            "peak_tangent_GPa": peak_tangent,
            "softening_pct": 100.0 * (peak_tangent - at) / peak_tangent,
            "strength_MPa": float(g.stress_MPa.max()),
            "strain_at_break_pct": float(g.strain_pct.max())})
    d = pd.DataFrame(rows).merge(
        fm[["specimen_id", "E_D790_toe_compensated_GPa"]], on="specimen_id")
    return d.rename(columns={"E_D790_toe_compensated_GPa": "chord_GPa"})


def compare(a, b):
    """Welch's t, Hedges g and a 95 % interval on the difference of means."""
    n1, n2 = len(a), len(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    g = (1 - 3 / (4 * (n1 + n2) - 9)) * (a.mean() - b.mean()) / pooled
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    df = se ** 4 / (s1 ** 4 / (n1 ** 2 * (n1 - 1))
                    + s2 ** 4 / (n2 ** 2 * (n2 - 1)))
    half = stats.t.ppf(0.975, df) * se
    diff = a.mean() - b.mean()
    return {"p": float(p), "t": float(t), "hedges_g": float(g),
            "difference": float(diff),
            "ci95": [float(diff - half), float(diff + half)]}


def holm(pvalues):
    """Holm-Bonferroni, with the monotonicity the procedure requires."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def main():
    d = per_specimen()
    a = d[d.configuration == "basalt_face"]
    f = d[d.configuration == "flax_face"]

    quantities = {
        "chord modulus (GPa)": "chord_GPa",
        "peak tangent modulus (GPa)": "peak_tangent_GPa",
        "flexural strength (MPa)": "strength_MPa",
        "softening to 0.55 % strain (%)": "softening_pct",
        "strain at break (%)": "strain_at_break_pct",
    }
    stats_by = {name: compare(a[col], f[col])
                for name, col in quantities.items()}

    for name, col in quantities.items():
        stats_by[name]["basalt_mean"] = float(a[col].mean())
        stats_by[name]["basalt_sd"] = float(a[col].std(ddof=1))
        stats_by[name]["flax_mean"] = float(f[col].mean())
        stats_by[name]["flax_sd"] = float(f[col].std(ddof=1))

    families = {
        "four comparisons, Table S7":
            ["chord modulus (GPa)", "peak tangent modulus (GPa)",
             "flexural strength (MPa)", "softening to 0.55 % strain (%)"],
        "stiffness as one comparison: three":
            ["chord modulus (GPa)", "flexural strength (MPa)",
             "softening to 0.55 % strain (%)"],
        "strain at break in place of softening: three":
            ["chord modulus (GPa)", "flexural strength (MPa)",
             "strain at break (%)"],
    }

    out = {"per_specimen": d.to_dict(orient="records"),
           "comparisons": stats_by, "families": {}}

    print(f"  {'quantity':<32}{'basalt':>16}{'flax':>16}{'raw p':>10}")
    for name in quantities:
        s = stats_by[name]
        print(f"  {name:<32}{s['basalt_mean']:8.2f} ± {s['basalt_sd']:<5.2f}"
              f"{s['flax_mean']:8.2f} ± {s['flax_sd']:<5.2f}{s['p']:10.4f}")

    for label, members in families.items():
        corrected = holm(np.array([stats_by[n]["p"] for n in members]))
        out["families"][label] = {n: float(c)
                                  for n, c in zip(members, corrected)}
        print(f"\n  {label}")
        for n, c in zip(members, corrected):
            mark = "significant" if c < 0.05 else "not significant"
            print(f"    {n:<34} Holm p {c:.4f}   {mark}")

    s = stats_by["chord modulus (GPa)"]
    print(f"\n  the stiffness difference is {-s['difference']:.2f} GPa, "
          f"{-100 * s['difference'] / f.chord_GPa.mean():.1f} % of the "
          f"flax-longitudinal mean,")
    print(f"  with a 95 % interval of {-s['ci95'][1]:.2f} to "
          f"{-s['ci95'][0]:.2f} GPa on the difference")

    path = DATA / "flexural_comparisons.json"
    path.write_text(json.dumps(out, indent=2) + "\n", newline="\n")
    print(f"\n  written: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
