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
rule is now stated and applied in one place.

What the limit cannot move, and why the table says so. The reference estimate is
built on the load-controlled acquisition, which has no phase-angle channel at
all, so no choice of limit reaches it. What moves is the depth-resolved basalt
median, and with it the alternative input of Supplementary Table S4. The
estimate is recomputed on the round-3 two-level construction: class fractions
and class medians from the indents inside each zone, the share of the two zones
from the imaged tiles, with only the basalt median replaced.

The first row is the quality mask of the earlier submission, kept because the
manuscript reports the move from that mask to this one. It rejected 433 indents
and left a basalt median of 49.47 GPa. No rule applied to the deposited
phase-angle values reproduces that set, so those two figures are carried here as
recorded values and are labelled as not reproducible; the estimate they imply is
computed like every other one. Nothing downstream may treat them as measurements
of this deposit.

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

# The superseded mask, as recorded in the previous submission. Not derivable
# from anything in the deposit: it was computed on the full continuous-stiffness
# trace, which is not deposited at the resolution it used, and no rule applied
# to the deposited phase-angle values reproduces the same set of 433 indents.
SUPERSEDED_LABEL = "15 (superseded flag, not reproducible)"
SUPERSEDED_REJECTED = 433
SUPERSEDED_BASALT_GPA = 49.47
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

    rows = [{
        "phase_angle_limit_deg": SUPERSEDED_LABEL,
        "rejected": SUPERSEDED_REJECTED,
        "rejected_pct": 100.0 * SUPERSEDED_REJECTED / N_DEPTH_RESOLVED,
        "alternative_basalt_input_GPa": SUPERSEDED_BASALT_GPA,
        "Voigt_with_that_input_GPa": round(
            estimate_with_basalt(per_zone, medians, w, SUPERSEDED_BASALT_GPA), 2),
        "source": ("recorded from the previous submission; the count and the "
                   "median are not reproducible from this deposit, the "
                   "estimate is computed from the median"),
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

    swept = [r for r in rows if r["source"].startswith("recomputed")]
    est = [r["Voigt_with_that_input_GPa"] for r in swept]
    pct = [r["rejected_pct"] for r in swept]
    print(f"\nacross the sweep the rejected fraction moves "
          f"{min(pct):.0f} % to {max(pct):.0f} % and the estimate built on "
          f"that input moves {min(est):.2f} to {max(est):.2f} GPa, "
          f"a range of {max(est) - min(est):.2f} GPa")

    # The two figures Table 2 quotes on the phase-angle row: what the sweep
    # moves, and what replacing the superseded mask with the stated rule moved.
    at15 = next(r for r in swept if r["phase_angle_limit_deg"] == PA_QC)
    print(f"the stated rule at {PA_QC:g} degrees rejects {at15['rejected']} "
          f"indents and gives {at15['Voigt_with_that_input_GPa']:.2f} GPa, "
          f"against {rows[0]['rejected']} and "
          f"{rows[0]['Voigt_with_that_input_GPa']:.2f} GPa under the "
          f"superseded mask, a move of "
          f"{at15['Voigt_with_that_input_GPa'] - rows[0]['Voigt_with_that_input_GPa']:.2f} GPa")

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
