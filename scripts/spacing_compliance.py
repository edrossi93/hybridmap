"""
Indent spacing against the Phani-Oliver criterion, per indent.

Reviewer 3 asked for the maximum contact depth per phase, so that compliance
with the spacing guideline could be demonstrated rather than asserted. The paper
reported medians, which understates the shortfall, and the reply told him it
reported maxima, which it did not.

A maximum is the wrong statistic here, for a reason worth stating. The largest
flax contact depth in the load-controlled map is 31 µm, which is a failed contact
rather than a deep one, so a raw maximum needs a rejection rule and the rule
then becomes another parameter to argue about. What the reviewer actually asked
for is a demonstration of compliance, and that can be given directly: count how
many indents in each class individually satisfy pitch >= 10 h_c.

The answer is unflattering and is the point of reporting it. Four indents in
five in the two compliant classes violate the criterion on the short axis of the
load-controlled map; the two stiff classes essentially never do. So the spacing
question is confined to flax and matrix, which is also where the border test
finds no evidence of an interaction.

Contact depth is reconstructed from the deposited peak load and hardness through
the Berkovich area function, A = P/H and A = 24.5 h_c^2, rather than stored: the
export carries load, stiffness, modulus and hardness per indent, and h_c follows
from two of them.

The reconstruction is an approximation, since it assumes an ideal tip rather
than a calibrated area function. It can be checked against the instrument's own
values, which Table 1 of the paper reports as class medians: 41, 78, 308 and
325 nm for basalt, mixed-volume, flax and matrix. This returns 54, 104, 327 and
343 nm, high by 5 to 7 % on the two classes that matter here. The compliance
fractions are therefore slightly conservative, which is the direction to err in
when the question is whether a criterion is met.

One indent of the 5551 has no reconstructed contact depth, and 2.2 says the depth
is reconstructed for every indent, so the shortfall is recorded here rather than
left to be found. Indent 8553, the last of the acquisition, is classified flax and
carries a hardness of exactly 0.000 GPa, so A = P/H is infinite and the square
root of it is not a number. It is not a borderline case: its stored contact depth
is 21.17 um, sixty-five times the flax class median, and its modulus reads
0.03 GPa. Like the 59 contacts below 1 GPa that the Supplementary Figure S3
caption reports, it is a runaway contact rather than a soft one, and it is the
extreme of that population. The load-controlled map carries no quality filter, so
it stays in the class and in every count the paper reports; what it cannot enter
is a spacing test, because there is no depth to test. The count of indents the
reconstruction covers, and this indent by name, are written into the JSON below so
that the difference between 5551 and 5550 is answered in the file rather than
noticed in it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Short axis of the load-controlled grid, in nm. The long axis is 5.096 µm and
# satisfies the criterion for every class, so the short axis is the binding one.
PITCH_SHORT_NM = 2477.0
BERKOVICH = 24.5          # A = 24.5 h_c^2 for an ideal Berkovich tip
CRITERION = 10            # Phani and Oliver: pitch at least ten times h_c
CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]


def contact_depth_nm(df):
    area = (df.peak_load_mN * 1e-3) / (df.H_GPa * 1e9)      # m^2
    return np.sqrt(area / BERKOVICH) * 1e9


def excluded(frame):
    """The classified indents the reconstruction cannot reach, one row each.

    Kept as a record rather than a count. A count invites the reader to assume
    which indents are missing; the identifier and the stored depth let them look.
    """
    return [{"indent_id": int(r.indent_id),
             "class": str(r.constituent),
             "zone": str(r.zone),
             "H_GPa": float(r.H_GPa),
             "stored_contact_depth_nm": (None if pd.isna(r.contact_depth_nm)
                                         else float(r.contact_depth_nm)),
             "reason": ("hardness of exactly 0.000 GPa, so A = P/H is infinite "
                        "and no contact depth follows from it; a runaway contact, "
                        "left in the class because the map carries no quality "
                        "filter")}
            for r in frame.itertuples()]


def reconstruct(frame=None):
    """The classified indents, split by whether a contact depth follows from them.

    Returns the ones it does, carrying an `hc_nm` column, and the ones it does
    not. Split out of main() so that anything checking these figures against the
    paper reads them from here instead of repeating the reconstruction.
    """
    if frame is None:
        frame = pd.read_csv(DATA / "classified_shallow.csv")
    classified = frame[frame.constituent.isin(CLASSES)]
    d = classified.assign(hc_nm=contact_depth_nm(classified))
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.dropna(subset=["hc_nm"]), d[d.hc_nm.isna()]


def violating_pct(kept, cls):
    """Share of a class breaching the criterion on the short axis, unrounded."""
    s = kept.loc[kept.constituent == cls, "hc_nm"]
    return 100.0 * float((s > PITCH_SHORT_NM / CRITERION).mean())


def main():
    kept, absent = reconstruct()
    missing = excluded(absent)
    d, classified = kept, len(kept) + len(absent)

    limit = PITCH_SHORT_NM / CRITERION
    rows = []
    for c in CLASSES:
        s = d.loc[d.constituent == c, "hc_nm"]
        rows.append({
            "class": c,
            "n": int(len(s)),
            "median_hc_nm": round(float(s.median()), 1),
            "violating_pct": round(violating_pct(d, c), 1),
            "required_pitch_at_median_um":
                round(CRITERION * float(s.median()) / 1000.0, 2),
        })

    out = pd.DataFrame(rows)
    print(f"load-controlled map, short-axis pitch {PITCH_SHORT_NM / 1000:.2f} µm, "
          f"so an indent complies when h_c < {limit:.0f} nm\n")
    print(out.to_string(index=False))
    compliant = [r for r in rows if r["class"] in ("flax", "matrix")]
    stiff = [r for r in rows if r["class"] in ("mixed-volume", "basalt")]
    print(f"\ncontact depth reconstructed for {len(d)} of {classified} "
          f"classified indents")
    for row in missing:
        print(f"  indent {row['indent_id']} ({row['class']}) has none: "
              f"{row['reason'].split(';')[0]}")
    print(f"\ncompliant classes violate on "
          f"{min(r['violating_pct'] for r in compliant):.0f} to "
          f"{max(r['violating_pct'] for r in compliant):.0f} % of indents; "
          f"stiff classes on "
          f"{min(r['violating_pct'] for r in stiff):.1f} to "
          f"{max(r['violating_pct'] for r in stiff):.1f} %")

    path = DATA / "spacing_compliance.json"
    path.write_text(json.dumps({
        "pitch_short_axis_nm": PITCH_SHORT_NM,
        "criterion": f"pitch >= {CRITERION} x contact depth",
        "method": ("contact depth from the deposited peak load and hardness, "
                   "A = P/H with A = 24.5 h_c^2"),
        "classified_indents": int(classified),
        "indents_with_a_reconstructed_contact_depth": int(len(d)),
        "indents_without_one": missing,
        "rows": rows}, indent=2) + "\n", encoding="utf8", newline="\n")
    print(f"\nwritten to {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
