"""What the stiff contacts of the basalt-centre band do with depth: the evidence
of 4.4, computed once and written down.

The paper chooses the load-controlled basalt median, 86 GPa, over the
depth-resolved one, 55 GPa, on the grounds that the two acquisitions agree on
the same contacts at 80 nm and that the deeper reading of those contacts then
falls with depth. Supplementary Figure S7 draws that fall; this script is the
arithmetic behind every figure 4.4 quotes about it, and behind a few more that
bear on what the fall is not, so that every one of them can be reproduced from
the deposit.

The selection is the figure's: every indent of the basalt-centre band whose
trace reads above the load-controlled map's own 70 GPa cut at 80 nm. It is made
at 80 nm and the behaviour below is read afterwards, so nothing about the trace
beyond 80 nm enters the choice of which traces to follow. Every indent that
meets it is classified basalt at 150 nm.

What is computed, in the order 4.4 reads it:

  * the median modulus of the selected contacts at each depth from 50 to 300 nm, and the
    ratio E(300)/E(80) indent by indent: its median, how many are at or below a
    half, its range, and whether any indent fails to soften;
  * the same ratio over every classified contact of the band, in four bands of
    the modulus read at 80 nm, so that the selection can be seen as the top of a
    gradient rather than as a population picked for a high reading;
  * the indents the depth-resolved classification itself calls basalt, a
    population selected at 150 nm and not at 80, on E(300)/E(80) and on
    E(300)/E(150);
  * whether the softening follows the distance to the nearest compliant
    contact, below 15 GPa at 80 nm, drawn from every indent in the band as the
    neighbourhoods of 4.1 are: a Spearman rank correlation, and the split
    between the indents more than two grid pitches from any such contact and
    the rest. The (2, 0) shell sits at exactly two pitches and is counted as
    near; the (2, 1) shell, at root five pitches, is the first shell beyond it;
  * the same question read through the eight-neighbour basalt fraction of 4.1,
    splitting the selection at six of eight neighbours called basalt;
  * the hardness of the same contacts between 80 and 300 nm;
  * two quantities that do not pass through the area function: S^2/P, which is
    independent of it, from 50 to 300 nm, and the contact stiffness S itself
    between 250 and 300 nm, where the load is still rising. The file carries
    two stiffness records, both exported channels: the stiffness channel, and
    the S^2/P channel, which agrees with the exported E and H to four decimals
    through (4/pi) Er^2/H and so carries the stiffness the instrument used
    for the modulus. The two do not agree with each other (the dictionary
    gives the ratio), and the step between them is not in the export. The
    paper reads both quantities off the S^2/P channel, S being taken as the
    square root of S^2/P times the load; the stiffness channel's own figures
    are written beside them and are not printed;
  * and two shares that involve no classification cut at 150 nm: the share of
    the band's classified contacts above 70 GPa at 80 nm, against the share of
    the load-controlled basalt-rich zone the 70 GPa cut calls basalt.

The reading-depth sensitivity of the laminate estimate, the basalt input read
from the selected traces at 80 and at 50 nm and substituted into the two-level
construction, is written by `basalt_input_sweep.py` beside the cut sweep it
belongs with; the medians it substitutes are computed here and imported there.

Run: python depth_softening.py
Writes: data/depth_softening.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

# The selection is the figure's, imported so that the two cannot drift apart:
# 80 nm, and the load-controlled map's own upper threshold of 70 GPa.
from figS7_basalt_depth import SELECT_AT_NM, SELECT_ABOVE_GPA      # noqa: E402
from figs_supplementary import neighbour_fraction, KNN            # noqa: E402

ZONE = "basalt-centre"
DEPTHS_REPORTED = (50.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0, 300.0)
DEEP_NM = 300.0
MID_NM = 150.0
# The compliant/stiff split of the classification, 15 GPa, read at the
# selection depth: a contact below it at 80 nm is a compliant neighbour.
COMPLIANT_BELOW_GPA = 15.0
# The four bands of the modulus at 80 nm over which the ratio is reported. The
# cuts are the classification's own, 15 and 70 GPa, plus the depth-resolved
# map's 40 GPa; each band is (lower, upper], open above on the last.
BANDS = ((70.0, np.inf), (40.0, 70.0), (15.0, 40.0), (-np.inf, 15.0))
# "More than two pitches": strictly beyond the second-neighbour shell along an
# axis. The tolerance is a thousandth of a pitch, enough to hold a shell that
# sits at exactly two pitches on the near side whatever the last digit of the
# stored coordinates does.
FAR_PITCHES = 2.0
PITCH_TOL = 1e-3
# The neighbourhood split: at least this many of the eight neighbours called
# basalt.
NEIGHBOURS_BASALT = 6


def traces(zone_ids, column, curves):
    """One row per indent of the band, one column per depth on the 5 nm grid."""
    return (curves[curves.indent_id.isin(zone_ids)]
            .pivot(index="indent_id", columns="depth_nm", values=column))


def grid_pitch(frame):
    """The spacing of the acquisition grid, read from the coordinates."""
    xs = np.sort(frame.x_um.unique())
    return float(np.min(np.diff(xs)))


def selected(E):
    """The indents read above the cut at the selection depth."""
    return E.index[E[SELECT_AT_NM] > SELECT_ABOVE_GPA]


def ratio(frame, top, bottom, ids):
    """Per-indent ratio of a trace at two depths, blanks dropped."""
    return (frame.loc[ids, top] / frame.loc[ids, bottom]).dropna()


def band_gradient(E, classified_ids):
    """E(300)/E(80) over the classified contacts, by the modulus at 80 nm."""
    e80 = E.loc[classified_ids, SELECT_AT_NM]
    r = E.loc[classified_ids, DEEP_NM] / e80
    rows = []
    for lower, upper in BANDS:
        inside = (e80 > lower) & ((e80 <= upper) if np.isfinite(upper) else True)
        rr = r[inside].dropna()
        rows.append({"lower_GPa": None if not np.isfinite(lower) else lower,
                     "upper_GPa": None if not np.isfinite(upper) else upper,
                     "n": int(inside.sum()),
                     "median_ratio": float(rr.median()),
                     "softening_pct": float(100 * (rr < 1).mean())})
    return rows


def distance_test(E, zone, ids, r):
    """Does the ratio follow the distance to the nearest compliant contact?"""
    pitch = grid_pitch(zone)
    xy = zone.set_index("indent_id")[["x_um", "y_um"]]
    compliant = E.index[E[SELECT_AT_NM] < COMPLIANT_BELOW_GPA]
    cxy = xy.loc[compliant].to_numpy()
    d = pd.Series([float(np.linalg.norm(cxy - xy.loc[i].to_numpy(),
                                        axis=1).min()) for i in ids],
                  index=ids)
    rho, p = stats.spearmanr(d.to_numpy(), r.loc[ids].to_numpy())
    far = (d / pitch) > FAR_PITCHES + PITCH_TOL
    return {"compliant_below_GPa": COMPLIANT_BELOW_GPA,
            "read_at_nm": SELECT_AT_NM,
            "pool": "every indent in the band, including quality-rejected ones",
            "n_compliant_contacts": int(len(compliant)),
            "pitch_um": round(pitch, 5),
            "distance_range_um": [round(float(d.min()), 3),
                                  round(float(d.max()), 3)],
            "spearman_rho": float(rho), "p": float(p),
            "far": {"criterion": f"more than {FAR_PITCHES:g} pitches from any "
                                 f"compliant contact",
                    "n": int(far.sum()),
                    "median_ratio": float(r.loc[ids][far.to_numpy()].median())},
            "near": {"n": int((~far).sum()),
                     "median_ratio":
                         float(r.loc[ids][(~far).to_numpy()].median())}}


def neighbourhood_test(zone, ids, r):
    """The same question through the eight-neighbour basalt fraction of 4.1."""
    sub = zone[zone.indent_id.isin(ids)].set_index("indent_id").loc[ids]
    f8 = neighbour_fraction(sub.reset_index(), zone, KNN)
    high = f8 >= NEIGHBOURS_BASALT / KNN
    rr = r.loc[ids].to_numpy()
    p = stats.mannwhitneyu(rr[high], rr[~high], alternative="two-sided").pvalue
    return {"neighbours": KNN,
            "criterion": f"at least {NEIGHBOURS_BASALT} of {KNN} neighbours "
                         f"called basalt",
            "n_at_or_above": int(high.sum()),
            "median_ratio_at_or_above": float(np.median(rr[high])),
            "n_rest": int((~high).sum()),
            "median_ratio_rest": float(np.median(rr[~high])),
            "mann_whitney_two_sided_p": float(p)}


def basalt_readings(depths=(50.0, SELECT_AT_NM)):
    """The median modulus of the selected contacts at the given depths.

    What `basalt_input_sweep.py` substitutes for the basalt input to read the
    laminate estimate at a shallower contact than the load-controlled map's
    own: the same selected traces, read at 80 nm and at 50 nm, the shallowest point of the deposited grid.
    """
    classified_depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    curves = pd.read_csv(DATA / "depth_resolved_curves.csv")
    zone = classified_depth[classified_depth.zone == ZONE]
    E = traces(zone.indent_id, "E_GPa", curves)
    ids = selected(E)
    return {"n": int(len(ids)),
            "median_GPa": {float(z): float(E.loc[ids, float(z)].median())
                           for z in depths}}


def compute():
    classified_depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    curves = pd.read_csv(DATA / "depth_resolved_curves.csv")
    shallow = pd.read_csv(DATA / "classified_shallow.csv")

    zone = classified_depth[classified_depth.zone == ZONE]
    labels = zone.set_index("indent_id").constituent
    classified_ids = labels.index[labels != "QC/invalid"]
    called = labels.index[labels == "basalt"]
    matrix = labels.index[labels == "matrix"]

    E = traces(zone.indent_id, "E_GPa", curves)
    H = traces(zone.indent_id, "H_GPa", curves)
    S = traces(zone.indent_id, "stiffness_N_per_m", curves)
    P = traces(zone.indent_id, "load_mN", curves)
    S2P = traces(zone.indent_id, "S2_over_P_GPa", curves)

    ids = selected(E)
    r = ratio(E, DEEP_NM, SELECT_AT_NM, ids)
    r_mid = ratio(E, DEEP_NM, MID_NM, ids)
    r_called = ratio(E, DEEP_NM, SELECT_AT_NM, called)
    r_called_mid = ratio(E, DEEP_NM, MID_NM, called)
    h = ratio(H, DEEP_NM, SELECT_AT_NM, ids)
    s2p = ratio(S2P, DEEP_NM, 50.0, ids)
    s_fall = (S.loc[ids, DEEP_NM] - S.loc[ids, 250.0]).dropna()
    s_ratio = ratio(S, DEEP_NM, SELECT_AT_NM, ids)
    s_ratio_matrix = ratio(S, DEEP_NM, SELECT_AT_NM, matrix)
    # The stiffness the S^2/P channel carries, in N/m: S = sqrt(S^2/P * P),
    # with S^2/P in GPa = 1e9 N/m^2 and P in mN = 1e-3 N.
    Sd = np.sqrt(S2P * 1e9 * P * 1e-3)
    sd_fall = (Sd.loc[ids, DEEP_NM] - Sd.loc[ids, 250.0]).dropna()
    sd_change = (Sd.loc[ids, DEEP_NM] / Sd.loc[ids, 250.0]).dropna()
    p_change = (P.loc[ids, DEEP_NM] / P.loc[ids, 250.0]).dropna()

    medians = {f"{z:g}": float(E.loc[ids, z].median()) for z in DEPTHS_REPORTED}
    shallow_zone = shallow[shallow.zone == "basalt"]
    shallow_classified = shallow_zone[shallow_zone.constituent != "QC/invalid"]

    return {
        "selection": {
            "zone": ZONE,
            "read_at_nm": SELECT_AT_NM,
            "above_GPa": SELECT_ABOVE_GPA,
            "n": int(len(ids)),
            "indents_in_zone": int(len(zone)),
            "classified_in_zone": int(len(classified_ids)),
            "selected_and_classified": int(labels.loc[ids].ne("QC/invalid").sum()),
            "class_at_150_nm": labels.loc[ids].value_counts().to_dict(),
            "indent_ids": [int(i) for i in ids],
        },
        "E_median_GPa_by_depth": medians,
        "E300_over_E80": {
            "n": int(len(r)),
            "median": float(r.median()),
            "at_or_below_half": int((r <= 0.5).sum()),
            "min": float(r.min()), "max": float(r.max()),
            "softening_pct": float(100 * (r < 1).mean()),
            "median_E_fall_pct": float(100 * (1 - medians["300"] / medians["80"])),
        },
        "E300_over_E150": {"n": int(len(r_mid)), "median": float(r_mid.median()),
                           "softening_pct": float(100 * (r_mid < 1).mean())},
        "bands_by_E80_over_classified": band_gradient(E, classified_ids),
        "called_basalt_at_150_nm": {
            "n": int(len(called)),
            "E300_over_E80": {"median": float(r_called.median()),
                              "softening_pct":
                                  float(100 * (r_called < 1).mean())},
            "E300_over_E150": {"median": float(r_called_mid.median()),
                               "softening_pct":
                                   float(100 * (r_called_mid < 1).mean())},
        },
        "distance_to_nearest_compliant_contact": distance_test(E, zone, ids, r),
        "eight_neighbour_basalt_fraction": neighbourhood_test(zone, ids, r),
        "H300_over_H80": {"n": int(len(h)), "median": float(h.median()),
                          "lower_at_300": int((h < 1).sum())},
        "S2_over_P": {
            "note": "S^2/P does not pass through the area function",
            "median_GPa_at_50_nm": float(S2P.loc[ids, 50.0].median()),
            "median_GPa_at_300_nm": float(S2P.loc[ids, DEEP_NM].median()),
            "median_ratio_300_over_50": float(s2p.median()),
            "ratio_of_medians": float(S2P.loc[ids, DEEP_NM].median()
                                      / S2P.loc[ids, 50.0].median()),
            "n_with_both": int(len(s2p)),
            "falling": int((s2p < 1).sum()),
        },
        "contact_stiffness_S": {
            "note": ("the paper reads S off the S^2/P channel, as "
                     "sqrt(S^2/P * P); the figures under 'stiffness_channel' "
                     "are the exported stiffness channel's own and are not "
                     "printed"),
            "median_kN_per_m_at_250_nm": float(Sd.loc[ids, 250.0].median() / 1e3),
            "median_kN_per_m_at_300_nm": float(Sd.loc[ids, DEEP_NM].median() / 1e3),
            "change_of_medians_250_to_300_nm_pct": float(
                100 * (Sd.loc[ids, DEEP_NM].median()
                       / Sd.loc[ids, 250.0].median() - 1)),
            "median_indent_change_250_to_300_nm_pct": float(
                100 * (sd_change.median() - 1)),
            "falling_250_to_300_nm": int((sd_fall < 0).sum()),
            "n": int(len(sd_fall)),
            "median_load_mN_at_250_nm": float(P.loc[ids, 250.0].median()),
            "median_load_mN_at_300_nm": float(P.loc[ids, DEEP_NM].median()),
            "load_change_of_medians_250_to_300_nm_pct": float(
                100 * (P.loc[ids, DEEP_NM].median()
                       / P.loc[ids, 250.0].median() - 1)),
            "S300_over_S80_median_selected": float(
                (Sd.loc[ids, DEEP_NM] / Sd.loc[ids, SELECT_AT_NM]).median()),
            "S300_over_S80_median_matrix": float(
                (Sd.loc[matrix, DEEP_NM] / Sd.loc[matrix, SELECT_AT_NM])
                .dropna().median()),
            "n_matrix": int((Sd.loc[matrix, DEEP_NM]
                             / Sd.loc[matrix, SELECT_AT_NM]).dropna().shape[0]),
        },
        "stiffness_channel": {
            "note": "the exported stiffness channel, for the record; not printed",
            "median_kN_per_m_at_250_nm": float(S.loc[ids, 250.0].median() / 1e3),
            "median_kN_per_m_at_300_nm": float(S.loc[ids, DEEP_NM].median() / 1e3),
            "falling_250_to_300_nm": int((s_fall < 0).sum()),
            "n": int(len(s_fall)),
            "median_load_mN_at_250_nm": float(P.loc[ids, 250.0].median()),
            "median_load_mN_at_300_nm": float(P.loc[ids, DEEP_NM].median()),
            "S300_over_S80_median_selected": float(s_ratio.median()),
            "S300_over_S80_median_matrix": float(s_ratio_matrix.median()),
            "n_matrix": int(len(s_ratio_matrix)),
        },
        "cut_free_shares": {
            "note": ("neither share uses the classification cut at 150 nm: the "
                     "first is the selection above, the second the "
                     "load-controlled map's own 70 GPa cut"),
            "depth_resolved_above_70_GPa_at_80_nm": {
                "n": int(len(ids)), "of": int(len(classified_ids)),
                "pct": float(100 * len(ids) / len(classified_ids))},
            "load_controlled_basalt_class": {
                "n": int((shallow_classified.constituent == "basalt").sum()),
                "of": int(len(shallow_classified)),
                "pct": float(100 * (shallow_classified.constituent
                                    == "basalt").mean())},
        },
    }


def main():
    out = compute()
    sel, rr = out["selection"], out["E300_over_E80"]
    print(f"  {sel['n']} indents of the {ZONE} band read above "
          f"{SELECT_ABOVE_GPA:g} GPa at {SELECT_AT_NM:g} nm, "
          f"{sel['selected_and_classified']} of them classified, all called "
          f"{', '.join(sel['class_at_150_nm'])}")
    print("  median E (GPa) at " + ", ".join(
        f"{z} nm {v:.1f}" for z, v in out["E_median_GPa_by_depth"].items()))
    print(f"  E(300)/E(80): median {rr['median']:.3f}, {rr['at_or_below_half']} "
          f"at or below 0.5, range {rr['min']:.3f} to {rr['max']:.3f}, "
          f"{rr['softening_pct']:.0f} % soften; the median falls "
          f"{rr['median_E_fall_pct']:.0f} %")
    print("  by the modulus at 80 nm, over the classified contacts of the band:")
    for b in out["bands_by_E80_over_classified"]:
        lo = "" if b["lower_GPa"] is None else f"above {b['lower_GPa']:g}"
        hi = "" if b["upper_GPa"] is None else f"to {b['upper_GPa']:g}"
        print(f"    {lo:>9} {hi:<6} n = {b['n']:>3}  median ratio "
              f"{b['median_ratio']:.3f}  softening {b['softening_pct']:.0f} %")
    c = out["called_basalt_at_150_nm"]
    print(f"  the {c['n']} called basalt at 150 nm: E(300)/E(80) median "
          f"{c['E300_over_E80']['median']:.2f}, "
          f"{c['E300_over_E80']['softening_pct']:.0f} % soften; E(300)/E(150) "
          f"median {c['E300_over_E150']['median']:.2f}, "
          f"{c['E300_over_E150']['softening_pct']:.0f} %")
    print(f"  the selected {sel['n']} between 150 and 300 nm: median ratio "
          f"{out['E300_over_E150']['median']:.2f}")
    d = out["distance_to_nearest_compliant_contact"]
    print(f"  distance to the nearest contact below {COMPLIANT_BELOW_GPA:g} GPa "
          f"at {SELECT_AT_NM:g} nm: {d['distance_range_um'][0]:.0f} to "
          f"{d['distance_range_um'][1]:.0f} um on a {d['pitch_um']:.0f} um "
          f"grid, Spearman rho {d['spearman_rho']:.2f}, p {d['p']:.2f}; "
          f"{d['far']['n']} far at {d['far']['median_ratio']:.2f} against "
          f"{d['near']['n']} near at {d['near']['median_ratio']:.2f}")
    n8 = out["eight_neighbour_basalt_fraction"]
    print(f"  {n8['criterion']}: {n8['n_at_or_above']} at "
          f"{n8['median_ratio_at_or_above']:.2f} against {n8['n_rest']} at "
          f"{n8['median_ratio_rest']:.2f}, p = {n8['mann_whitney_two_sided_p']:.2f}")
    h = out["H300_over_H80"]
    print(f"  H(300)/H(80): median {h['median']:.3f}, {h['lower_at_300']} of "
          f"{h['n']} lower")
    s2 = out["S2_over_P"]
    print(f"  S^2/P: median {s2['median_GPa_at_50_nm']:.0f} GPa at 50 nm to "
          f"{s2['median_GPa_at_300_nm']:.0f} at 300, median ratio "
          f"{s2['median_ratio_300_over_50']:.2f}, falling on "
          f"{s2['falling']} of {s2['n_with_both']}")
    s = out["contact_stiffness_S"]
    print(f"  S from the S^2/P channel: {s['median_kN_per_m_at_250_nm']:.1f} to "
          f"{s['median_kN_per_m_at_300_nm']:.1f} kN/m from 250 to 300 nm "
          f"({s['change_of_medians_250_to_300_nm_pct']:+.1f} % on the medians, "
          f"falling on {s['falling_250_to_300_nm']} of {s['n']}), while the "
          f"load rises {s['median_load_mN_at_250_nm']:.1f} to "
          f"{s['median_load_mN_at_300_nm']:.1f} mN "
          f"({s['load_change_of_medians_250_to_300_nm_pct']:+.0f} %); "
          f"S(300)/S(80) {s['S300_over_S80_median_selected']:.2f} against "
          f"{s['S300_over_S80_median_matrix']:.2f} for the {s['n_matrix']} "
          f"matrix contacts")
    sc = out["stiffness_channel"]
    print(f"  the stiffness channel itself: {sc['median_kN_per_m_at_250_nm']:.1f} "
          f"to {sc['median_kN_per_m_at_300_nm']:.1f} kN/m, falling on "
          f"{sc['falling_250_to_300_nm']} of {sc['n']} (not printed)")
    cf = out["cut_free_shares"]
    a, b = (cf["depth_resolved_above_70_GPa_at_80_nm"],
            cf["load_controlled_basalt_class"])
    print(f"  above 70 GPa with no 150 nm cut: {a['n']} of {a['of']} "
          f"({a['pct']:.1f} %) here, {b['n']} of {b['of']} ({b['pct']:.1f} %) "
          f"in the load-controlled basalt-rich zone")

    path = DATA / "depth_softening.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf8",
                    newline="\n")
    print(f"\n  written: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
