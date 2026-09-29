"""Sensitivity of the depth-resolved analysis to the phase-angle quality limit.

This is Supplementary Table S3. For each limit it answers three questions: how
many indents the criterion rejects, what the depth-resolved pure-axial basalt
median becomes once they are gone, and what the laminate estimate would be if
that median were used as the basalt input in place of the load-controlled one.

The criterion is the one stated in the Methods, and nothing else: an indent is
rejected when the magnitude of its corrected phase angle exceeds the limit at
any of the three analysis depths, or when one of those depths has no valid
window. At 15 degrees that rule is what `qc_flag_strict` carries in
`classified_depth_resolved.csv`. The sweep checks that the two agree indent by
indent rather than assuming it, because the whole point of the table is that the
rule is stated and applied in one place.

What the limit cannot move, and why the table says so. The reference estimate is
built on the load-controlled acquisition, which has no phase-angle channel at
all, so no choice of limit reaches it. What moves is the depth-resolved basalt
median, and with it the alternative input of Supplementary Table S4. The
estimate is recomputed on the two-level construction: class fractions
and class medians from the indents inside each zone, the share of the two zones
from the imaged tiles, with only the basalt median replaced.

The first row applies the instrument's own quality flag, which was computed on
the full continuous-stiffness trace and is deposited as `qc_flag_instrument` in
`depth_resolved_indents.csv`. It rejects 433 indents. No rule applied to the
deposited phase-angle values reproduces that set, so the flag is carried as a
recorded column; the class rule and the basalt median under it are computed
here on the deposited slice values, like every other row.

Run: python phase_angle_sensitivity.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from classify import classify, E_UPPER_DEPTH, PA_QC                # noqa: E402
import zone_share as zs                                            # noqa: E402

THRESHOLDS = (10.0, 15.0, 20.0)
CLASSES = zs.CLASSES

INSTRUMENT_LABEL = "15 (instrument flag, full CSM trace)"
N_DEPTH_RESOLVED = 2000


def fails(pa, limit):
    """The quality decision, one boolean per indent. True means rejected.

    Magnitude rather than signed value: a large negative phase angle is as
    unphysical as a large positive one. A missing slice is a rejection and not a
    pass by default, because an indent whose trace never covered one of the
    three depths has nothing to test there.
    """
    return (pa.abs() > limit).any(axis=1) | pa.isna().any(axis=1)


def reference_construction():
    """The two-level construction the estimate is built on.

    Returns the within-zone class fractions, the class medians and the
    basalt-zone share of the section. All three come from the load-controlled
    acquisition and the imaged tiles, so none of them depends on the
    phase-angle limit.
    """
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    per_zone, medians, _ = zs.zone_fractions(shallow)
    return per_zone, medians, zs.tile_share()


def estimate_with_basalt(per_zone, medians, w_basalt, basalt_GPa):
    """Eq. (1) with the basalt input swapped and everything else held fixed."""
    swapped = dict(medians)
    swapped["basalt"] = float(basalt_GPa)
    return float(zs.at_share(per_zone, swapped, w_basalt)[1][0])


def instrument_flag_basalt(depth=None):
    """Basalt median at 150 nm and its count under the instrument's own flag."""
    if depth is None:
        depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    flag = pd.read_csv(DATA / "depth_resolved_indents.csv")[
        ["indent_id", "qc_flag_instrument"]]
    d = depth.merge(flag, on="indent_id", validate="1:1")
    zone = ["basalt" if "basalt" in str(b) else "flax" for b in d.zone]
    label = classify(d.E_150.values, zone, e_upper=E_UPPER_DEPTH).values
    basalt = (label == "basalt") & (d.qc_flag_instrument.to_numpy() == 0)
    return (float(d.E_150[basalt].median()), int(basalt.sum()),
            int(d.qc_flag_instrument.sum()))


def instrument_flag_move(basalt_at_limit=None):
    """How far the estimate on the depth-resolved basalt input moves between the
    instrument's flag and the stated rule at 15 degrees.

    Unrounded: the difference of the two estimates themselves, not of the two
    cells the table prints.
    """
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    if basalt_at_limit is None:
        kept = depth[depth.qc_flag_strict == 0]
        basalt_at_limit = float(kept.loc[kept.constituent == "basalt",
                                         "E_150"].median())
    flag_basalt, _, _ = instrument_flag_basalt(depth)
    per_zone, medians, w = reference_construction()
    return (estimate_with_basalt(per_zone, medians, w, basalt_at_limit)
            - estimate_with_basalt(per_zone, medians, w, flag_basalt))


def sweep():
    """One row per limit, in the order Supplementary Table S3 prints them."""
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    pa = depth[["PA_100", "PA_150", "PA_200"]]

    # The class rule is applied once, on the modulus at 150 nm, and only the
    # quality decision moves with the limit. Reclassifying inside the loop would
    # let the two effects mix, and the table is about the quality decision.
    zone = ["basalt" if "basalt" in str(b) else "flax" for b in depth.zone]
    label = classify(depth.E_150.values, zone, e_upper=E_UPPER_DEPTH).values

    # The deposited flag is the same rule at 15 degrees. If this ever stops
    # holding, the table and the classification have come apart and the run
    # should stop rather than publish two answers.
    same = fails(pa, PA_QC).to_numpy() == depth.qc_flag_strict.astype(bool).to_numpy()
    if not same.all():
        raise SystemExit(f"the criterion at {PA_QC:g} degrees disagrees with "
                         f"qc_flag_strict on {(~same).sum()} indents")

    per_zone, medians, w = reference_construction()

    flag_basalt, flag_n, flag_rejected = instrument_flag_basalt(depth)
    rows = [{
        "phase_angle_limit_deg": INSTRUMENT_LABEL,
        "rejected": flag_rejected,
        "rejected_pct": 100.0 * flag_rejected / N_DEPTH_RESOLVED,
        "n_basalt": flag_n,
        "alternative_basalt_input_GPa": round(flag_basalt, 2),
        "Voigt_with_that_input_GPa": round(
            estimate_with_basalt(per_zone, medians, w, flag_basalt), 2),
        "source": ("the instrument's flag as deposited; class rule and "
                   "median recomputed from the deposited slice values"),
    }]

    for limit in THRESHOLDS:
        rejected = fails(pa, limit)
        kept = pd.Series(label).where(~rejected.to_numpy(), "QC/invalid")
        med = {c: float(depth.E_150[(kept == c).to_numpy()].median())
               for c in CLASSES}
        rows.append({
            "phase_angle_limit_deg": limit,
            "rejected": int(rejected.sum()),
            "rejected_pct": 100.0 * float(rejected.mean()),
            "n_basalt": int((kept == "basalt").sum()),
            **{f"E_{c}_GPa": round(med[c], 2) for c in CLASSES},
            "alternative_basalt_input_GPa": round(med["basalt"], 2),
            "Voigt_with_that_input_GPa": round(
                estimate_with_basalt(per_zone, medians, w, med["basalt"]), 2),
            "source": "recomputed from the deposited slice values",
        })
    return rows


def main():
    per_zone, medians, w = reference_construction()
    reference = float(zs.at_share(per_zone, medians, w)[1][0])
    rows = sweep()

    print(f"reference estimate, load-controlled inputs and the imaged zone "
          f"share: {reference:.2f} GPa")
    print("  no phase-angle limit can move it: that acquisition has no "
          "phase-angle channel\n")

    table = pd.DataFrame(rows).drop(columns="source")
    print(table.to_string(index=False))

    swept = rows[1:]
    est = [r["Voigt_with_that_input_GPa"] for r in swept]
    pct = [r["rejected_pct"] for r in swept]
    print(f"\nacross the sweep the rejected fraction moves "
          f"{min(pct):.0f} % to {max(pct):.0f} % and the estimate built on "
          f"that input moves {min(est):.2f} to {max(est):.2f} GPa, "
          f"a range of {max(est) - min(est):.2f} GPa")

    # The two figures Table 2 quotes on the phase-angle row: what the sweep
    # moves, and how far the instrument's flag sits from the stated rule.
    at15 = next(r for r in swept if r["phase_angle_limit_deg"] == PA_QC)
    print(f"the stated rule at {PA_QC:g} degrees rejects {at15['rejected']} "
          f"indents and gives {at15['Voigt_with_that_input_GPa']:.2f} GPa, "
          f"against {rows[0]['rejected']} and "
          f"{rows[0]['Voigt_with_that_input_GPa']:.2f} GPa under the "
          f"instrument's flag, a move of {instrument_flag_move():.2f} GPa")
    print("  that move is taken before the two estimates are rounded into the "
          "table")

    path = DATA / "phase_angle_sensitivity.json"
    path.write_text(json.dumps({
        "reference_voigt_GPa": reference,
        "note": ("The limit applies only to the depth-resolved acquisition, "
                 "which is the one carrying a continuous-stiffness trace. The "
                 "reference estimate is built on the load-controlled map and "
                 "the imaged zone share, and the load-controlled map has no "
                 "phase-angle channel, so no choice of limit can move it. What "
                 "moves is the depth-resolved basalt median and the "
                 "alternative input of Supplementary Table S4."),
        "criterion": ("reject when the magnitude of the corrected phase angle "
                      "exceeds the limit at any of the three analysis depths, "
                      "or when one of those depths has no valid window"),
        "rows": rows}, indent=2) + "\n", encoding="utf8", newline="\n")
    print(f"\nwritten to {path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
