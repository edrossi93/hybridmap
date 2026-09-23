"""
Recompute every number the paper prints, from the deposited tables, and put each
recomputed value beside the published one.

Run this first if you want to know whether the deposit is intact. It takes a few
seconds, needs nothing but the files in `data/`, and writes nothing.

What it answers. Whether each printed value follows from the deposited
measurements under the construction the paper describes: the two-level
construction of Eq. (1), with the class fractions and class medians measured
inside each architectural zone and the share of the two zones measured on the
imaged tiles. Every value is computed here or read from a deposited product whose
hash `provenance_check.py` holds to its own script. Nothing is a literal copied
from the manuscript except the published value each line is compared against.

What it cannot answer. It does not test whether the construction is the right
one; the sensitivity analyses do that, and they are checked here as published
numbers rather than as arguments. It does not check the acquisition either: the
per-indent tables are where this deposit starts. And one printed value has no
check below, which is said here so that the absence is visible rather than
silent: the Halpin-Tsai baseline of Figure 8, 27.1 GPa, is a literal in
`fig08_rom.py` with nothing behind it. The derivation added to that script for
the inputs §3.4 states gives 27.23 GPa, so the two disagree by 0.13 GPa and the
figure keeps the published literal until the authors settle which moves.

One more disagreement is left standing rather than resolved here. Supplementary
Table S8 gives the elementary-fibre axial modulus as 45.30 GPa and
`rule_of_mixtures.FLAX_INPUTS` gives 52.47 GPa for the same row, from the same
source read two ways. The estimates checked below are computed from the inputs
the table itself prints.

An earlier version of this script reported sixty-seven values reproducing while
checking eight the paper no longer prints, among them the whole-map Voigt sum of
21.33 GPa read as the headline estimate, the Hill and Reuss values that go with
it, and an overshoot against a 20.1 GPa target from the first submission. Those
checks are gone. A check against a value the paper does not carry is worse than
no check, because it reports success.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from classify import classify_shallow                      # noqa: E402
from rule_of_mixtures import class_table                   # noqa: E402
from clustering_check import run as cluster_run            # noqa: E402
from table2_intervals import intervals as _intervals       # noqa: E402
import zone_share as zs                                    # noqa: E402
import phase_angle_sensitivity as _pas                     # noqa: E402
import border_rejection as _border                         # noqa: E402
import flexural_comparisons as _fx                         # noqa: E402
import input_decomposition as _dec                         # noqa: E402

checks = []

# One unit in the last digit the paper prints. Three cells are rounded the other
# way from the recomputation and are the only places this is needed: the low end
# of the tabulated range in 3.4, which is exactly 28.25 and is printed 28.3; the
# modulus term of the same paragraph, which is 10.746 and is printed 10.8; and
# the last gap cell of Supplementary Table S8, which is 50.65 and is printed
# 50.6. Each is a rounding in the document and not a different quantity.
LAST_DIGIT = 0.06

# Half a unit in a figure the paper prints to one decimal, and a hair more. An
# exact decimal such as 21.65 is stored just below itself, so a half-unit
# tolerance written as 0.05 rejects the value the table rounds it to.
ROUNDED = 0.051


def report(label, got, published, tol):
    ok = abs(got - published) <= tol
    checks.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.3f}   paper {published}")


def report_same(label, got, expected):
    """A deposited product against the live computation of the same quantity.

    Not a published value. These lines exist so that a stale product cannot pass
    the checks that read it.
    """
    ok = abs(got - expected) <= 1e-9
    checks.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.3f}   "
          f"recomputed here")


def report_below(label, got, limit):
    """For a cell the paper prints as an inequality rather than as a value."""
    checks.append(got < limit)
    print(f"  [{'ok ' if got < limit else 'DIFF'}] {label:<52} {got:>9.2e}   "
          f"paper < {limit}")


print("classification, load-controlled map")
raw = pd.read_csv(DATA / "shallow_map.csv")
lab = classify_shallow(raw)
report("indents acquired", len(raw), 8553, 0)
report("indents after the resin cut", len(lab), 5553, 0)
counts = lab.constituent.value_counts()
for cls, n in [("flax", 2300), ("matrix", 894), ("mixed-volume", 1977),
               ("basalt", 380)]:
    report(f"n({cls})", int(counts.get(cls, 0)), n, 0)
classified = int(sum(counts.get(c, 0) for c in zs.CLASSES))
report("indents classified", classified, 5551, 0)
report("indents neither classified nor on resin", len(lab) - classified, 2, 0)
# The per-zone counts of 3.3. They are printed as a chain, and the chain is only
# worth printing if it closes, so the totals are checked with the parts.
for zone, n in (("basalt", 2760), ("flax", 2793)):
    report(f"n(indents in the {zone}-rich zone)", int((lab.zone == zone).sum()),
           n, 0)
cross = pd.crosstab(lab.zone, lab.constituent)
for zone, cls, n in [("basalt", "matrix", 894), ("basalt", "mixed-volume", 1525),
                     ("basalt", "basalt", 340), ("flax", "flax", 2300),
                     ("flax", "mixed-volume", 452), ("flax", "basalt", 40)]:
    report(f"n({cls}) in the {zone}-rich zone", int(cross.loc[zone, cls]), n, 0)
for cls, pct in [("flax", 41.4), ("matrix", 16.1), ("mixed-volume", 35.6),
                 ("basalt", 6.8)]:
    report(f"{cls}, per cent of the classified indents",
           100 * counts.get(cls, 0) / classified, pct, 0.05)

print("\nper-class moduli and intervals, load-controlled map (Table 3)")
sh = pd.read_csv(DATA / "classified_shallow.csv")
tab = class_table(sh)
_sh_ci = _intervals(sh, "E_GPa").set_index("class")
for cls, E, lo, hi in [("flax", 4.70, 4.56, 4.84), ("matrix", 4.66, 4.31, 4.99),
                       ("mixed-volume", 35.79, 34.4, 37.0),
                       ("basalt", 85.99, 83.7, 87.8)]:
    report(f"median E({cls})",
           float(tab.loc[tab["class"] == cls, "E_median_GPa"].iloc[0]), E, 0.02)
    report(f"CI low  E({cls})", float(_sh_ci.ci_low_GPa[cls]), lo, 0.05)
    report(f"CI high E({cls})", float(_sh_ci.ci_high_GPa[cls]), hi, 0.05)

# The depth-resolved column of Table 3, and the quality mask it depends on.
#
# This block was missing, and its absence is why the manuscript carried the old
# mask's counts through two rounds while the supplement carried the new ones:
# nothing compared the printed column against the data. Every cell of that
# column is checked here, the interval included, because an interval is as much
# a published number as a median.
print("\nper-class values, depth-resolved map at z = 150 nm (Table 3)")
_dr_all = pd.read_csv(DATA / "classified_depth_resolved.csv")
report("depth-resolved indents acquired", len(_dr_all), 2000, 0)
report("rejected by the phase-angle criterion",
       int(_dr_all.qc_flag_strict.sum()), 304, 0)
report("rejected, per cent", 100 * _dr_all.qc_flag_strict.mean(), 15.2, 0.05)
# The interval comes from the script that owns the caption's recipe, not from a
# second bootstrap written here. Two implementations of the same recipe drift,
# and then the table and the check disagree with each other rather than with the
# data.
_dr_ci = _intervals(_dr_all, "E_150").set_index("class")
for cls, n, E, lo, hi, H in [
        ("flax", 1120, 3.79, 3.67, 3.94, 0.203),
        ("matrix", 124, 5.32, 4.65, 6.39, 0.228),
        ("mixed-volume", 241, 29.76, 27.44, 31.17, 1.868),
        ("basalt", 211, 53.01, 50.51, 54.53, 4.192)]:
    g = _dr_all[_dr_all.constituent == cls]
    report(f"n({cls}), depth-resolved", len(g), n, 0)
    report(f"median E({cls}) @ 150 nm", float(g.E_150.median()), E, 0.02)
    report(f"median H({cls}) @ 150 nm", float(g.H_150.median()), H, 0.005)
    report(f"CI low  E({cls}) @ 150 nm", float(_dr_ci.ci_low_GPa[cls]), lo, 0.02)
    report(f"CI high E({cls}) @ 150 nm", float(_dr_ci.ci_high_GPa[cls]), hi, 0.02)

# Supplementary Table S1, which had no script behind it until the census of
# every printed number found 89.05 GPa with no route to it.
# Where the rejections fall, which 2.2 reports in two sentences. The border
# definition is imported rather than repeated: `border_rejection.py` owns it, and
# `border_effect.py` uses the same one on the other map.
print("\nwhere the rejected indents fall (2.2)")
_edge = _border.rejection_rates(_dr_all, "x_um")
report("rejected on the two outer rows of the short axis",
       _edge["border"]["pct"], 15.5, ROUNDED)
report("rejected inside them", _edge["interior"]["pct"], 15.2, ROUNDED)
for zone, published in (("basalt-centre", 11.0), ("flax-bottom", 20.2)):
    g = _dr_all[_dr_all.zone == zone]
    report(f"rejected in the {zone} band", 100 * g.qc_flag_strict.mean(),
           published, ROUNDED)

print("\nPoisson-ratio sensitivity (Supplementary Table S1)")
import poisson_sensitivity as _poi                          # noqa: E402
_s1 = _poi.table().set_index("class")
for cls, uniform, phase in (("flax", 4.70, 4.70), ("matrix", 4.66, 4.22),
                            ("mixed-volume", 35.79, 34.78),
                            ("basalt", 85.99, 89.05)):
    report(f"median E({cls}), uniform nu",
           float(_s1.E_uniform_nu_GPa[cls]), uniform, 0.005)
    report(f"median E({cls}), phase-specific nu",
           float(_s1.E_phase_specific_GPa[cls]), phase, 0.005)
_est = _poi.estimates(_poi.table())
report("estimate at phase-specific Poisson ratios", _est["phase_specific"],
       18.30, 0.005)

# Eq. (1) as the paper now builds it. The fractions are mixed inside each zone
# and the two zones are weighted by the share they occupy on the imaged section,
# not by how many indents landed in each. Weighting by indent counts is the
# construction of the previous submission; it is kept below because Table 2, 3.4
# and the Figure 8 caption all print what it gives, as the comparison that
# motivates the change.
print("\nEq. (1): the two-level construction (Table 2, Table 3, 2.4, 3.4)")
per_zone, medians, w_indents = zs.zone_fractions(sh)
w_tiles = zs.tile_share()
report("basalt-rich share of the section, imaged tiles", 100 * w_tiles, 36.2, 0.05)
report("flax-rich share of the section, imaged tiles",
       100 * (1 - w_tiles), 63.8, 0.05)
report("basalt-rich share implied by the indent counts",
       100 * w_indents, 49.7, 0.05)
fractions, (voigt, hill, reuss) = zs.at_share(per_zone, medians, w_tiles)
for cls, f in zip(zs.CLASSES, fractions):
    report(f"Eq. (1) fraction, {cls}", float(f),
           {"flax": 0.526, "matrix": 0.117, "mixed-volume": 0.303,
            "basalt": 0.054}[cls], 0.0005)
report("Eq. (1) fractions, sum", float(fractions.sum()), 1.000, 1e-9)
report("laminate estimate", voigt, 18.50, 0.005)
report("Hill average", hill, 12.7, 0.05)
report("Reuss value", reuss, 6.8, 0.05)
report("estimate with the zones weighted by indent counts",
       zs.at_share(per_zone, medians, w_indents)[1][0], 21.33, 0.005)

# Where the sum comes from, class by class. 4.3.4 and 4.3.6 quote these as the
# reason the flax input matters less than it looks.
E_class = np.array([medians[c] for c in zs.CLASSES])
terms = dict(zip(zs.CLASSES, fractions * E_class))
compliant = terms["flax"] + terms["matrix"]
report("compliant classes, contribution to the estimate", compliant, 3.02, 0.005)
report("compliant classes, per cent of the estimate",
       100 * compliant / voigt, 16, 0.5)
report("mixed-volume, contribution to the estimate", terms["mixed-volume"],
       10.9, 0.05)
report("mixed-volume, per cent of the estimate",
       100 * terms["mixed-volume"] / voigt, 59, 0.5)
report("pure basalt, per cent of the estimate",
       100 * terms["basalt"] / voigt, 25, 0.5)
report("the two stiff classes, contribution to the estimate",
       terms["mixed-volume"] + terms["basalt"], 15.48, 0.005)
report("the two stiff classes, per cent of the estimate",
       100 * (terms["mixed-volume"] + terms["basalt"]) / voigt, 84, 0.5)
# Interchanging the two compliant labels, which is the residual ambiguity of
# 4.3.5. Only the two medians swap; the fractions are unchanged.
swapped = dict(medians)
swapped["flax"], swapped["matrix"] = medians["matrix"], medians["flax"]
compliant_swapped = float(fractions[0] * swapped["flax"]
                          + fractions[1] * swapped["matrix"])
report("compliant contribution, labels interchanged", compliant_swapped,
       3.001, 0.005)
report("what interchanging the two labels moves",
       compliant - compliant_swapped, 0.018, 0.001)

print("\nuncertainty budget (Supplementary Table S5, 2.4)")
# Read from the deposited product rather than recomputed here: the budget is
# tens of thousands of resamplings and `provenance_check.py` holds the file to
# `uncertainty_budget.py`. The point estimate and the zone share are checked
# against the live computation above, so a stale budget cannot pass quietly.
budget = json.loads((DATA / "uncertainty_budget.json").read_text())
report_same("budget, point estimate", budget["voigt_point_estimate"], voigt)
report_same("budget, basalt-rich share used", budget["w_basalt_mosaic"], w_tiles)
report("u, class fractions within each zone",
       budget["type_a_sd_GPa"]["fractions"], 1.080, 0.001)
report("u, class medians", budget["type_a_sd_GPa"]["medians"], 0.345, 0.001)
report("u, share of the two zones", budget["type_a_sd_GPa"]["zone_share"],
       0.611, 0.001)
report("u, classification thresholds",
       budget["type_b_sd_GPa"]["classification_thresholds"], 0.243, 0.001)
report("u, Poisson ratio", budget["type_b_sd_GPa"]["poisson_ratio"], 0.056,
       0.001)
report("the three sampling terms together",
       budget["type_a_combined_sd_GPa"], 1.42, 0.005)
report("combined standard uncertainty", budget["combined_sd_GPa"], 1.441, 0.001)
report("combined uncertainty, per cent of the estimate",
       100 * budget["combined_sd_GPa"] / voigt, 7.8, 0.05)
report("95 % interval from the draws, low",
       budget["percentile_95_interval_GPa"][0], 15.78, 0.005)
report("95 % interval from the draws, high",
       budget["percentile_95_interval_GPa"][1], 21.34, 0.005)
report("u at a 10 um resampling block",
       budget["block_size_sensitivity_sd_GPa"]["2"], 1.21, 0.005)
report("u at a 41 um resampling block",
       budget["block_size_sensitivity_sd_GPa"]["8"], 1.68, 0.005)

# Where the map could have sat, which is the other half of the zone-share term:
# a window of the load-controlled map's own footprint slid over the mosaic.
# Written by zone_share.py, read here.
window = json.loads((DATA / "zone_share_sensitivity.json").read_text())
placements = window["windows"]["load-controlled map"]
report("placements of the load-controlled footprint on the mosaic",
       placements["n_placements"], 67025, 0)
report("basalt-rich share over those placements, mean",
       placements["share_mean"], 0.373, 0.0005)
report("basalt-rich share over those placements, SD",
       placements["share_sd"], 0.342, 0.0005)
report("estimate over those placements, mean", placements["voigt_mean"], 18.72,
       0.005)
report("estimate over those placements, SD", placements["voigt_sd"], 7.18, 0.005)

print("\nthreshold sensitivity (Supplementary Table S2)")
grid = zs.threshold_grid(raw, w_tiles)
published_S2 = {(10, 60): 18.84, (10, 65): 18.79, (10, 70): 18.74,
                (10, 75): 18.72, (10, 80): 18.59, (15, 60): 18.53,
                (15, 65): 18.42, (15, 70): 18.50, (15, 75): 18.48,
                (15, 80): 18.41, (20, 60): 18.24, (20, 65): 18.24,
                (20, 70): 18.21, (20, 75): 18.12, (20, 80): 18.00}
for (lower, upper), published in published_S2.items():
    row = grid[(grid.lower_GPa == lower) & (grid.upper_GPa == upper)]
    report(f"estimate at lo = {lower}, hi = {upper}",
           float(row.estimate_GPa.iloc[0]), published, 0.005)
reference = float(grid[(grid.lower_GPa == 15)
                       & (grid.upper_GPa == 70)].estimate_GPa.iloc[0])
report("largest deviation across the grid",
       float((grid.estimate_GPa - reference).abs().max()), 0.50, 0.005)
# The budget converts the extremes of this grid into its Type B term. Two
# implementations of one sweep are two chances to be wrong, so they are held
# against each other here.
report_same("grid low, against the budget's recorded range",
            float(grid.estimate_GPa.min()),
            budget["type_b_ranges"]["threshold_sweep_GPa"][0])
report_same("grid high, against the budget's recorded range",
            float(grid.estimate_GPa.max()),
            budget["type_b_ranges"]["threshold_sweep_GPa"][1])

print("\nphase-angle quality sensitivity (Supplementary Table S3)")
# Every cell of the table, including the first row, whose rejected count and
# basalt median are recorded values of the superseded mask and whose estimate is
# computed from that median like every other row.
rows_S3 = _pas.sweep()
published_S3 = [(433, 21.7, 49.47, 16.53), (473, 23.6, 53.14, 16.73),
                (304, 15.2, 53.01, 16.73), (217, 10.8, 52.63, 16.70)]
for row, (n, pct, basalt, estimate) in zip(rows_S3, published_S3):
    name = str(row["phase_angle_limit_deg"]).split(" ")[0]
    report(f"rejected at {name} deg", row["rejected"], n, 0)
    report(f"rejected at {name} deg, per cent", row["rejected_pct"], pct, ROUNDED)
    report(f"basalt input at {name} deg",
           row["alternative_basalt_input_GPa"], basalt, 0.005)
    report(f"estimate on that input at {name} deg",
           row["Voigt_with_that_input_GPa"], estimate, 0.005)
swept = [r["Voigt_with_that_input_GPa"] for r in rows_S3[1:]]
report("what the 10 to 20 deg sweep moves", max(swept) - min(swept), 0.03, 0.005)
report("what replacing the superseded mask moved",
       rows_S3[2]["Voigt_with_that_input_GPa"]
       - rows_S3[0]["Voigt_with_that_input_GPa"], 0.19, 0.015)

print("\ninput substitutions (Supplementary Tables S4 and S8, 3.4)")
flex = pd.read_csv(DATA / "flexural_moduli.csv")
measured = {c: float(g.E_D790_toe_compensated_GPa.mean())
            for c, g in flex.groupby("configuration")}


def with_input(cls, value):
    """The estimate with one class modulus replaced, everything else held."""
    return float(zs.at_share(per_zone, {**medians, cls: value}, w_tiles)[1][0])


def gap(estimate, configuration):
    """The relative gap the paper reports: (estimate - measured) / estimate."""
    return 100 * (estimate - measured[configuration]) / estimate


report("relative gap, basalt-longitudinal", gap(voigt, "basalt_face"), -1.9, 0.05)
report("relative gap, flax-longitudinal", gap(voigt, "flax_face"), -6.3, 0.05)
basalt_depth = float(_dr_all.loc[(_dr_all.qc_flag_strict == 0)
                                 & (_dr_all.constituent == "basalt"),
                                 "E_150"].median())
report("depth-resolved basalt input", basalt_depth, 53.01, 0.02)
estimate_depth = with_input("basalt", basalt_depth)
report("estimate with the depth-resolved basalt input", estimate_depth, 16.73,
       0.005)
report("relative gap on that input, basalt-longitudinal",
       gap(estimate_depth, "basalt_face"), -12.7, 0.05)
report("relative gap on that input, flax-longitudinal",
       gap(estimate_depth, "flax_face"), -17.5, 0.05)
report("estimate with the bulk PA-11 matrix, 1.5 GPa", with_input("matrix", 1.5),
       18.13, 0.005)
report("estimate with the matrix at 1.0 GPa", with_input("matrix", 1.0), 18.07,
       0.005)
# Supplementary Table S8. The three inputs are the ones printed in the table's
# own first column; they are three different physical quantities and none of
# them is what this work measures.
for value, estimate, g_basalt, g_flax in [(14.00, 23.38, 19.4, 15.9),
                                          (18.00, 25.49, 26.0, 22.9),
                                          (45.30, 39.84, 52.7, 50.6)]:
    got = with_input("flax", value)
    report(f"estimate with the flax input at {value:.2f} GPa", got, estimate,
           0.005)
    report("relative gap at that input, basalt-longitudinal",
           gap(got, "basalt_face"), g_basalt, LAST_DIGIT)
    report("relative gap at that input, flax-longitudinal",
           gap(got, "flax_face"), g_flax, LAST_DIGIT)

print("\ntabulated-input comparison (3.4)")
E_tab, f_nominal = _dec.TABULATED_E, _dec.NOMINAL_F
E_measured, f_measured = _dec.measured_moduli(), _dec.MEASURED_F
tabulated = _dec.voigt(E_tab, f_nominal)
report("Voigt sum, tabulated moduli and nominal fractions", tabulated, 35.8, 0.05)
for cfg, published in (("basalt_face", 47.3), ("flax_face", 45.0)):
    report(f"that sum overshoots {cfg} by", gap(tabulated, cfg), published, 0.05)
# The same comparison with the measurement as the denominator. Both sentences of
# 3.4 use one denominator each, so both are checked.
for cfg, published in (("basalt_face", 89.7), ("flax_face", 81.9)):
    report(f"overshoot on the measured denominator, {cfg}",
           100 * (tabulated - measured[cfg]) / measured[cfg], published, 0.05)
# What the tabulated route leaves open: the same fractions with the constituent
# moduli anywhere inside the ranges the Introduction cites.
for label, basalt, flax, published in (("low", 70.0, 40.0, 28.3),
                                       ("high", 95.0, 80.0, 44.5)):
    report(f"tabulated sum at the {label} end of the cited ranges",
           _dec.voigt({"basalt": basalt, "flax": flax, "matrix": 1.5},
                      f_nominal), published, LAST_DIGIT)
moduli_only = _dec.voigt(E_measured, f_nominal)
fractions_only = _dec.voigt(E_tab, f_measured)
both = _dec.voigt(E_measured, f_measured)
report("measured moduli at the nominal fractions", moduli_only, 25.0, 0.05)
report("tabulated moduli at the measured fractions", fractions_only, 43.8, 0.05)
report("measured moduli at the measured fractions", both, 24.5, 0.05)
report("what the moduli alone move", moduli_only - tabulated, -10.8, LAST_DIGIT)
report("what the fractions alone move", fractions_only - tabulated, 8.0, 0.05)
for cfg, published in (("basalt_face", 23.1), ("flax_face", 19.8)):
    report(f"the measured-fraction construction overshoots {cfg} by",
           gap(both, cfg), published, 0.05)

print("\nsegmentation of the mosaic (Figure 2, 3.4, Supplementary Table S6)")
# Both files here are primary measurements: the tile statistics and the
# leave-one-field-out validation need the tile images, which are not deposited.
# The lines below are therefore a check on the transcription into the paper, which
# is the failure this block exists to catch, and not on the segmentation itself.
seg = json.loads((DATA / "bse_segmentation_summary.json").read_text())
report("mosaic area, mm2", seg["area_mm2"], 6.65, 0.005)
report("imaged tiles", seg["n_tiles"], 91, 0)
report("unmarked interface band, per cent",
       100 * seg["zone_area_from_mosaic_mask"]["unmarked"], 4.0, 0.05)
for cls, published, ci in (("basalt", 24.4, 1.1), ("flax", 39.5, 3.2),
                           ("matrix", 36.1, None)):
    report(f"volume fraction, {cls}", 100 * seg["volume_fraction"][cls],
           published, 0.05)
    if ci is not None:
        report(f"its 95 % interval on the mean, {cls}",
               seg["ci_95_on_the_mean_pct"][cls], ci, 0.05)
report("fibre volume fraction, measured",
       seg["volume_fraction"]["basalt"] + seg["volume_fraction"]["flax"],
       0.639, 0.0005)
report("basalt packing inside its own zone",
       seg["in_zone_fibre_fraction_pooled_over_tiles"]["basalt_fib"], 0.665,
       0.0005)

val = json.loads((DATA / "bse_validation_metrics.json").read_text())
for zone, phases_only, all_three in (("basalt zone", 96.0, 82.8),
                                     ("flax zone", 97.9, 88.1)):
    report(f"{zone}, accuracy over the two solid phases",
           val[zone]["accuracy_pct_phases_only"], phases_only, 0.05)
    report(f"{zone}, accuracy with void as a third class",
           val[zone]["accuracy_pct"], all_three, 0.05)
for zone, cls, precision, recall, pixels in (
        ("basalt zone", "basalt", 92.5, 90.6, 33834),
        ("basalt zone", "matrix", 77.6, 74.4, 18465),
        ("basalt zone", "void", 72.5, 77.7, 20636),
        ("flax zone", "flax", 94.7, 94.2, 44000),
        ("flax zone", "matrix", 79.3, 78.1, 32493),
        ("flax zone", "void", 87.9, 88.9, 67127)):
    entry = val[zone]["classes"][cls]
    report(f"precision, {cls} in the {zone}", entry["precision_pct"], precision,
           0.05)
    report(f"recall, {cls} in the {zone}", entry["recall_pct"], recall, 0.05)
    report(f"labelled pixels, {cls} in the {zone}", entry["labelled_pixels"],
           pixels, 0)
# The three confusion figures 3.4 quotes, which are the argument that the pair
# backscattered contrast separates least is not the pair the classifier confuses.
flax_zone = val["flax zone"]["confusion_true_by_predicted"]
report("flax pixels called matrix", flax_zone["flax"]["matrix"], 635, 0)
report("flax pixels called matrix, per cent",
       100 * flax_zone["flax"]["matrix"] / val["flax zone"]["classes"]["flax"]["labelled_pixels"],
       1.4, 0.05)
report("matrix pixels called flax", flax_zone["matrix"]["flax"], 825, 0)
report("matrix pixels called flax, per cent",
       100 * flax_zone["matrix"]["flax"] / val["flax zone"]["classes"]["matrix"]["labelled_pixels"],
       2.5, 0.05)
report("matrix pixels called void, per cent",
       100 * flax_zone["matrix"]["void"] / val["flax zone"]["classes"]["matrix"]["labelled_pixels"],
       19.3, 0.05)

print("\nmixed-volume deconvolution (4.1, 4.3.6)")
# Read from the deposited product: 4000 bootstrap resamplings per acquisition,
# written by mixed_class_deconvolution.py and hash-recorded.
dec = json.loads((DATA / "mixed_class_deconvolution.json").read_text())
acq = dec["acquisitions"]
report("basalt fraction on the tie line, load-controlled",
       acq["shallow"]["phi_compliance"], 0.69, 0.005)
report("its interval, low", acq["shallow"]["phi_compliance_ci95"][0], 0.68, 0.005)
report("its interval, high", acq["shallow"]["phi_compliance_ci95"][1], 0.71, 0.005)
report("basalt fraction on the tie line, depth-resolved",
       acq["depth"]["phi_compliance"], 0.78, 0.005)
report("its interval, low", acq["depth"]["phi_compliance_ci95"][0], 0.75, 0.005)
report("its interval, high", acq["depth"]["phi_compliance_ci95"][1], 0.80, 0.005)
report("the same under linear mixing, load-controlled",
       acq["shallow"]["phi_linear"], 0.36, 0.005)
report("the same under linear mixing, depth-resolved",
       acq["depth"]["phi_linear"], 0.55, 0.005)
report("basalt packing in the zone, from the imaging",
       dec["independent_comparison"]["in_zone_basalt_packing_from_imaging"],
       0.665, 0.001)
report("two-class sum, load-controlled",
       dec["two_class_sum_GPa"]["compliance"]["shallow"], 61.0, 0.05)
report("two-class sum, depth-resolved",
       dec["two_class_sum_GPa"]["compliance"]["depth"], 67.7, 0.05)
report("the two sums, per cent apart",
       dec["two_class_sum_GPa"]["compliance"]["spread_pct"], 10.3, 0.05)
bulk = [e for e in acq["shallow"]["endpoint_sensitivity"]
        if e["e_matrix"] == 1.0 and e["e_basalt"] == 85.99][0]
report("load-controlled fraction at a bulk matrix end member",
       bulk["phi_compliance"], 0.91, 0.005)
bulk_depth = [e for e in acq["depth"]["endpoint_sensitivity"]
              if e["e_matrix"] == 1.0 and e["e_basalt"] == 53.14][0]
report("depth-resolved fraction at a bulk matrix end member",
       bulk_depth["phi_compliance"], 0.94, 0.005)
report("gap between the acquisitions across that sweep, low",
       dec["cross_acquisition_agreement"]["gap_span_pct"][0], 3.2, 0.05)
report("gap between the acquisitions across that sweep, high",
       dec["cross_acquisition_agreement"]["gap_span_pct"][1], 9.1, 0.05)
report("mixed-volume share of the basalt zone, load-controlled",
       100 * acq["shallow"]["mixed_class_fraction"], 55, 0.5)
report("mixed-volume share of the basalt zone, depth-resolved",
       100 * acq["depth"]["mixed_class_fraction"], 41, 0.5)

print("\nunsupervised cross-check")
dr = pd.read_csv(DATA / "classified_depth_resolved.csv")
c = cluster_run(dr)
report("ARI, rule vs k-means (k = 4)", c["ARI"], 0.408, 0.005)
report("ARI*, compliant clusters merged", c["ARI_star"], 0.652, 0.005)

print("\nthree-point bending (3.4, Supplementary Table S7)")
# Read the replicates rather than restating summary statistics. An earlier
# version of this block hard-coded 20.1 +/- 0.8 and 18.5 +/- 0.7, the
# single-specimen values from the first submission, and then compared the
# resulting t, p and g against those same values. That check could not fail.
#
# The five quantities come from flexural_comparisons.py, which defines each one
# and is the only place any of them is defined. The two moduli of Table S7 and
# the strength are also recomputed from the curves there, so this block reduces
# ten specimens rather than restating a summary.
spec = _fx.per_specimen()
a_all = spec[spec.configuration == "basalt_face"]
f_all = spec[spec.configuration == "flax_face"]
report("specimens per configuration", len(a_all), 5, 0)

for label, column, ba, bs, fa, fsd, g_pub in [
        ("chord modulus", "chord_GPa", 18.85, 0.54, 19.66, 0.49, -1.42),
        ("peak tangent modulus", "peak_tangent_GPa", 20.01, 0.55, 20.75, 0.24,
         -1.57),
        ("flexural strength", "strength_MPa", 181.68, 3.22, 117.55, 4.26, 15.34),
        ("softening to 0.55 % strain", "softening_pct", 8.40, 0.29, 34.55, 2.38,
         -13.92),
        ("strain at break", "strain_at_break_pct", 1.29, 0.08, 0.72, 0.03,
         8.79)]:
    x, y = a_all[column], f_all[column]
    report(f"{label}, basalt-longitudinal mean", x.mean(), ba, 0.01)
    report(f"{label}, basalt-longitudinal SD", x.std(ddof=1), bs, 0.005)
    report(f"{label}, flax-longitudinal mean", y.mean(), fa, 0.01)
    report(f"{label}, flax-longitudinal SD", y.std(ddof=1), fsd, 0.005)
    report(f"{label}, Hedges g", _fx.compare(x, y)["hedges_g"], g_pub, 0.005)

chord = _fx.compare(a_all.chord_GPa, f_all.chord_GPa)
report("chord modulus, Welch t", chord["t"], -2.487, 0.001)
report("chord modulus, Welch p uncorrected", chord["p"], 0.0380, 0.0001)
report("the stiffness difference", -chord["difference"], 0.81, 0.005)
report("that difference, per cent of the flax-longitudinal mean",
       -100 * chord["difference"] / f_all.chord_GPa.mean(), 4.1, 0.05)
report("its 95 % interval, low", -chord["ci95"][1], 0.06, 0.005)
report("its 95 % interval, high", -chord["ci95"][0], 1.56, 0.005)
report("peak tangent modulus, Welch p uncorrected",
       _fx.compare(a_all.peak_tangent_GPa, f_all.peak_tangent_GPa)["p"],
       0.0369, 0.0001)

# The Holm correction under all three families. The stiffness result depends on
# which family it is counted in, and the paper says so, so all three are checked
# rather than the one that was pre-specified.
print("\nHolm correction, three families (Supplementary Table S7, 3.4)")
quantities = {"chord modulus (GPa)": "chord_GPa",
              "peak tangent modulus (GPa)": "peak_tangent_GPa",
              "flexural strength (MPa)": "strength_MPa",
              "softening to 0.55 % strain (%)": "softening_pct",
              "strain at break (%)": "strain_at_break_pct"}
raw_p = {name: _fx.compare(a_all[col], f_all[col])["p"]
         for name, col in quantities.items()}
families = {
    "as reported: four comparisons":
        ["chord modulus (GPa)", "peak tangent modulus (GPa)",
         "flexural strength (MPa)", "softening to 0.55 % strain (%)"],
    "stiffness as one comparison: three":
        ["chord modulus (GPa)", "flexural strength (MPa)",
         "softening to 0.55 % strain (%)"],
    "the original family: three":
        ["chord modulus (GPa)", "flexural strength (MPa)",
         "strain at break (%)"]}
published_holm = {"as reported: four comparisons": 0.0738,
                  "stiffness as one comparison: three": 0.0380,
                  "the original family: three": 0.0380}
for label, members in families.items():
    corrected = dict(zip(members,
                         _fx.holm(np.array([raw_p[n] for n in members]))))
    report(f"Holm p on the chord modulus, {label}",
           corrected["chord modulus (GPa)"], published_holm[label], 0.0001)
    if "peak tangent modulus (GPa)" in corrected:
        report("Holm p on the peak tangent modulus, same family",
               corrected["peak tangent modulus (GPa)"], 0.0738, 0.0001)
    report_below(f"Holm p on the strength, {label}",
                 corrected["flexural strength (MPa)"], 0.0001)
    if "softening to 0.55 % strain (%)" in corrected:
        report_below("Holm p on the softening, same family",
                     corrected["softening to 0.55 % strain (%)"], 0.0001)

# The window above the tangent-modulus crossover reverses the ranking. This is
# the reason the orientation effect in the first submission did not survive
# re-analysis, so it is checked here rather than left as a remark in the text.
aw = flex.loc[flex.configuration == "basalt_face", "E_window_0p28_0p43_GPa"]
fw = flex.loc[flex.configuration == "flax_face", "E_window_0p28_0p43_GPa"]
report("modulus above the crossover, basalt-longitudinal", aw.mean(), 19.91, 0.01)
report("modulus above the crossover, flax-longitudinal", fw.mean(), 18.48, 0.01)

print("\n" + "-" * 72)
print(f"{sum(checks)} of {len(checks)} values reproduce")
if not all(checks):
    print("Values marked DIFF need looking at before the deposit is released.")
    sys.exit(1)
