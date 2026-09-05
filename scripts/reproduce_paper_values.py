"""
Recompute every headline number in the paper from the deposited CSVs and print
it next to the published value.

Run this first if you want to know whether the deposit is intact. It takes a
few seconds and touches nothing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from classify import classify_shallow                      # noqa: E402
from rule_of_mixtures import class_table, bounds, substitute, threshold_sweep  # noqa: E402
from clustering_check import run as cluster_run            # noqa: E402

checks = []


def report(label, got, published, tol):
    ok = abs(got - published) <= tol
    checks.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.3f}   paper {published}")


print("classification")
raw = pd.read_csv(DATA / "shallow_map.csv")
lab = classify_shallow(raw)
report("shallow indents acquired", len(raw), 8553, 0)
report("shallow indents after resin cut", len(lab), 5553, 0)
for cls, n in [("flax", 2300), ("matrix", 894), ("mixed-volume", 1977), ("basalt", 380)]:
    report(f"n({cls})", (lab.constituent == cls).sum(), n, 0)

print("\nper-class moduli, shallow map")
sh = pd.read_csv(DATA / "classified_shallow.csv")
tab = class_table(sh)
for cls, E in [("flax", 4.70), ("matrix", 4.66), ("mixed-volume", 35.79), ("basalt", 85.99)]:
    report(f"median E({cls})", float(tab.loc[tab["class"] == cls, "E_median_GPa"].iloc[0]), E, 0.02)

print("\nrule of mixtures")
v, h, r = bounds(tab.fraction, tab.E_median_GPa)
report("Voigt bound", v, 21.33, 0.02)
report("Hill bound", h, 14.41, 0.02)
report("Reuss bound", r, 7.50, 0.02)
report("Voigt with depth-resolved basalt input", substitute(tab, "basalt", 49.47)[0], 18.83, 0.02)
report("Voigt with bulk PA-11 matrix (1.5 GPa)", substitute(tab, "matrix", 1.5)[0], 20.82, 0.02)

print("\nthreshold sensitivity (Supplementary Table S2)")
sw = threshold_sweep(raw)
ref = float(sw[(sw.lower_GPa == 15) & (sw.upper_GPa == 70)].Voigt_GPa.iloc[0])
report("reference construction", ref, 21.33, 0.02)
report("largest deviation across the grid", float(np.abs(sw.Voigt_GPa - ref).max()), 0.50, 0.02)

print("\nliterature-input comparison")
E_lit = np.array([60.0, 1.5, 80.0])
f_lit = np.array([0.25, 0.50, 0.25])
report("Voigt, textbook moduli and fabric fractions", float((f_lit * E_lit).sum()), 35.75, 0.02)
report("overshoot vs 20.1 GPa (%)", 100 * ((f_lit * E_lit).sum() - 20.1) / 20.1, 77.9, 0.3)

print("\nunsupervised cross-check")
dr = pd.read_csv(DATA / "classified_depth_resolved.csv")
c = cluster_run(dr)
report("ARI, rule vs k-means (k = 4)", c["ARI"], 0.398, 0.005)
report("ARI*, compliant clusters merged", c["ARI_star"], 0.678, 0.005)

print("\nthree-point bending")
# Read the replicates rather than restating summary statistics. An earlier
# version of this block hard-coded 20.1 +/- 0.8 and 18.5 +/- 0.7, the
# single-specimen values from the first submission, and then compared the
# resulting t, p and g against those same values. That check could not fail.
fm = pd.read_csv(DATA / "flexural_moduli.csv")
col = "E_D790_toe_compensated_GPa"
a = fm.loc[fm.configuration == "basalt_face", col]
f = fm.loc[fm.configuration == "flax_face", col]
n1 = n2 = len(a)

report("basalt-longitudinal modulus, mean", a.mean(), 18.85, 0.01)
report("basalt-longitudinal modulus, SD", a.std(ddof=1), 0.54, 0.01)
report("flax-longitudinal modulus, mean", f.mean(), 19.66, 0.01)
report("flax-longitudinal modulus, SD", f.std(ddof=1), 0.48, 0.01)

t, p = stats.ttest_ind(a, f, equal_var=False)
report("Welch t", t, -2.487, 0.01)
report("Welch p, uncorrected", p, 0.038, 0.001)

s1, s2 = a.std(ddof=1), f.std(ddof=1)
sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
g = (1 - 3 / (4 * (n1 + n2) - 9)) * (a.mean() - f.mean()) / sp
report("Hedges g", g, -1.423, 0.01)

# Strength and strain at break come from the curves, not from the modulus
# table. The configuration labels here are the ones the CSV actually uses.
fx = pd.read_csv(DATA / "flexural_curves.csv")
peak = fx.groupby(["configuration", "specimen_id"]).stress_MPa.max()
for cfg, published in (("basalt_face", 181.7), ("flax_face", 117.5)):
    report(f"flexural strength, {cfg}", peak.loc[cfg].mean(), published, 0.1)

ends = fx.groupby(["configuration", "specimen_id"]).strain_pct.max()
for cfg, published in (("basalt_face", 1.295), ("flax_face", 0.716)):
    report(f"strain at break, {cfg}", ends.loc[cfg].mean(), published, 0.01)

# The window above the tangent-modulus crossover reverses the ranking. This is
# the reason the orientation effect in the first submission did not survive
# re-analysis, so it is checked here rather than left as a remark in the text.
aw = fm.loc[fm.configuration == "basalt_face", "E_window_0p28_0p43_GPa"]
fw = fm.loc[fm.configuration == "flax_face", "E_window_0p28_0p43_GPa"]
report("modulus above the crossover, basalt-longitudinal", aw.mean(), 19.91, 0.01)
report("modulus above the crossover, flax-longitudinal", fw.mean(), 18.48, 0.01)

print("\n" + "-" * 72)
print(f"{sum(checks)} of {len(checks)} values reproduce")
if not all(checks):
    print("Values marked DIFF need looking at before the deposit is released.")
