"""
Indent spacing against the Phani-Oliver criterion, per indent.

Reviewer 3 asked for the maximum contact depth per phase, so that compliance
with the spacing guideline could be demonstrated rather than asserted. The paper
reported medians, which understates the shortfall, and the reply told him it
reported maxima, which it did not.

A maximum is the wrong statistic here, for a reason worth stating. The largest
flax contact depth in the fixed-depth map is 31 µm, which is a failed contact
rather than a deep one, so a raw maximum needs a rejection rule and the rule
then becomes another parameter to argue about. What the reviewer actually asked
for is a demonstration of compliance, and that can be given directly: count how
many indents in each class individually satisfy pitch >= 10 h_c.

The answer is unflattering and is the point of reporting it. Four indents in
five in the two compliant classes violate the criterion on the short axis of the
fixed-depth map; the two stiff classes essentially never do. So the spacing
question is confined to flax and matrix, which is also where the border test
finds no evidence of an interaction.

Contact depth is reconstructed from the deposited peak load and hardness through
the Berkovich area function, A = P/H and A = 24.5 h_c^2, rather than stored: the
export carries load, stiffness, modulus and hardness per indent, and h_c follows
from two of them.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Short axis of the fixed-depth grid, in nm. The long axis is 5.096 µm and
# satisfies the criterion for every class, so the short axis is the binding one.
PITCH_SHORT_NM = 2477.0
BERKOVICH = 24.5          # A = 24.5 h_c^2 for an ideal Berkovich tip
CRITERION = 10            # Phani and Oliver: pitch at least ten times h_c
CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]


def contact_depth_nm(df):
    area = (df.peak_load_mN * 1e-3) / (df.H_GPa * 1e9)      # m^2
    return np.sqrt(area / BERKOVICH) * 1e9


def main():
    d = pd.read_csv(DATA / "classified_shallow.csv")
    d = d.assign(hc_nm=contact_depth_nm(d))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["hc_nm"])

    limit = PITCH_SHORT_NM / CRITERION
    rows = []
    for c in CLASSES:
        s = d.loc[d.constituent == c, "hc_nm"]
        rows.append({
            "class": c,
            "n": int(len(s)),
            "median_hc_nm": round(float(s.median()), 1),
            "violating_pct": round(100.0 * float((s > limit).mean()), 1),
            "required_pitch_at_median_um":
                round(CRITERION * float(s.median()) / 1000.0, 2),
        })

    out = pd.DataFrame(rows)
    print(f"fixed-depth map, short-axis pitch {PITCH_SHORT_NM / 1000:.2f} µm, "
          f"so an indent complies when h_c < {limit:.0f} nm\n")
    print(out.to_string(index=False))
    compliant = [r for r in rows if r["class"] in ("flax", "matrix")]
    stiff = [r for r in rows if r["class"] in ("mixed-volume", "basalt")]
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
        "rows": rows}, indent=2) + "\n", encoding="utf8", newline="\n")
    print(f"\nwritten to {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
