"""
Rule-based assignment of indents to the four classes.

Three of them are stiffness levels, compliant, mixed-volume and pure fibre. The
fourth exists because the compliant level holds two constituents that stiffness
cannot tell apart, so flax and PA-11 are separated by where they sit instead.
The depth-resolved acquisition corroborates that split from the other side: the
two compliant medians are 4.10 and 5.50 GPa at 150 nm, p = 1.8e-5 (two-sided
Mann-Whitney).

This is the paper's configuration of the general rule in `hybridmap.classify`:
the same zone-and-threshold machinery, with our thresholds, our zone boundaries
and our four class names filled in. The values live here rather than in the
package because they are ours and would be wrong on anyone else's section. The
same values are written out as `examples/paper_shallow.yaml`, and
`tests/test_tool.py` checks that the two agree.

The rule has two ingredients: where the indent sits in the laminate (the
architectural zone) and how stiff it is. Neither on its own is enough. Modulus
alone cannot separate flax from PA-11, because their indentation moduli overlap
almost completely. Position alone cannot separate a fibre from the matrix
around it.

Zones are set by the two spatial cuts:

    x < 380 um            embedding resin, discarded
    380 <= x < 730 um     basalt-rich half
    x >= 730 um           flax-rich half

Inside the basalt-rich half a compliant indent is PA-11, because that is the
only compliant phase present there. Inside the flax-rich half a compliant
indent is flax, for the same reason. This is the step that makes the fractions
indent-class fractions rather than constituent volume fractions: matrix sitting
in the flax-rich half gets counted as flax. It does not move the Voigt sum
because the two compliant medians are within 1 % of each other, but it does
mean the numbers should not be read as constituent volume fractions.

Thresholds (GPa):

    lower  15    compliant / stiff split, from the shallow dip in the
                 basalt-zone modulus histogram
    upper  70    shallow map, pure-axial basalt above this
           40    depth-resolved map, lower because the stiff contacts read
                 lower with depth there: the contacts above 70 GPa at 80 nm
                 have a median of 84.7 GPa there and 46.3 GPa at 300 nm
                 (Supplementary Figure S7), for a reason the data do not
                 settle
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybridmap import classify as _hm                          # noqa: E402
from hybridmap.config import Band, Phase, QC, Zones            # noqa: E402

# Where the embedding resin ends, in the long-axis coordinate of the shallow
# map. Not assumed: the epoxy gives a uniform indentation modulus near
# 5.5 GPa out to x = 300 um, and the median rises steeply between 300 and
# 330 um on entering the laminate. The cut is placed 50 to 80 um beyond that
# transition so that indents disturbed by the resin/laminate edge are
# discarded along with the resin itself.
X_RESIN = 380.0

# Boundary between the basalt-rich and flax-rich halves, taken as the midpoint
# between the two plateaus of a 30-indent moving median of E along the long
# axis.
X_ZONE_SPLIT = 730.0

# Compliant/stiff split. Sits in the bimodal gap of the basalt-zone modulus
# histogram (Supplementary Figure S3a), so it separates polymer-dominated
# contacts from fibre-bearing ones without cutting through a population.
E_LOWER = 15.0

# Above this, a contact is taken as pure axial basalt. The two maps need
# different values because the stiff contacts read differently at the two
# depths: the contacts that read above 70 GPa at 80 nm in the depth-resolved
# map, where the two acquisitions agree, have fallen to a median of 44 GPa by
# 300 nm (Supplementary Figure S7), for a reason the data do not settle, so a
# 70 GPa cut at 150 nm would reject almost every genuine basalt contact.
E_UPPER_SHALLOW = 70.0
E_UPPER_DEPTH = 40.0

# Phase angle above this means the contact has not settled: the response is
# still dominated by the near-surface establishment transient rather than by
# the material. Consistent with the conservative criterion recommended for
# rejecting near-surface CSM measurements.
PA_QC = 15.0

# The zone bands and the phase rule in the form the package takes them. Written
# out here so the paper's rule is legible in one place; the thresholds are
# substituted at call time, which is what lets the sensitivity sweep move them.
ZONES = Zones(axis="position",
              bands=(Band("resin", X_RESIN, discard=True),
                     Band("basalt", X_ZONE_SPLIT),
                     Band("flax")))

PHASES = (Phase("matrix", upper="lower", zone="basalt"),
          Phase("flax", upper="lower", zone="flax"),
          Phase("mixed-volume", lower="lower", upper="upper"),
          Phase("basalt", lower="upper"))


def assign_zone(x_um):
    frame = pd.DataFrame({"x_um": np.asarray(x_um, dtype=float)})
    zones = _hm.assign_zones(frame, ZONES, {"position": "x_um"})
    return pd.Series(zones.values)


def classify(E_GPa, zone, e_lower=E_LOWER, e_upper=E_UPPER_SHALLOW):
    """Four-way label from modulus plus zone. Returns a pandas Series."""
    return _hm.classify(E_GPa, zone, PHASES,
                        {"lower": e_lower, "upper": e_upper})


def phase_angle_qc(pa_100, pa_150, pa_200, limit=PA_QC, near_surface_limit=20.0):
    """
    The reusable tool's phase-angle rule, which is not the paper's. True means
    keep.

    What this implements. An indent is rejected when the largest of the three
    slice values exceeds `limit`, or when the shallowest of them exceeds
    `near_surface_limit`. The comparison is on the signed value, and a slice that
    is missing is skipped rather than counted against the indent. This is the
    form `hybridmap.classify.qc_keep` takes from a configuration file, where a
    user's phase-angle channel may be signed either way and a gap in a trace may
    mean something other than a bad contact, so the package cannot decide either
    for them.

    What the paper implements. The Methods reject an indent when the phase angle
    exceeds 15 degrees *in magnitude* at any of the three analysis depths, or
    when one of those depths has no valid window. That rule is written twice, in
    `make_classified_depth.py`, which stores it as `qc_flag_strict`, and in
    `phase_angle_sensitivity.fails()`, which sweeps the limit; the sweep stops
    the run if the two ever disagree on a single indent.

    The two differ on 16 of the 2000 depth-resolved indents: fifteen with a
    missing slice, which the paper rejects and this passes, and one whose phase
    angle is large and negative, which the paper rejects on magnitude and the
    signed comparison here does not. So this rejects 288 where the paper's 304
    comes from, and every count, median and figure in the paper is the paper's
    rule, read from the stored flag. On the deposited data `near_surface_limit`
    at 20 degrees rejects nothing that the 15 degree limit has not already
    rejected; it bites only when it is set below `limit`.
    """
    frame = pd.DataFrame({"PA_100": np.asarray(pa_100, float),
                          "PA_150": np.asarray(pa_150, float),
                          "PA_200": np.asarray(pa_200, float)})
    qc = QC(columns=("PA_100", "PA_150", "PA_200"), limit=limit,
            first_limit=near_surface_limit)
    return _hm.qc_keep(frame, qc).to_numpy()


def classify_shallow(df):
    """df needs columns x_um and E_GPa. Resin indents are dropped."""
    d = df[df.x_um >= X_RESIN].copy()
    d["zone"] = assign_zone(d.x_um.values).values
    d["constituent"] = classify(d.E_GPa.values, d.zone.values,
                                e_upper=E_UPPER_SHALLOW).values
    return d


def classify_depth_resolved(df, zone_col="zone"):
    """
    df needs E_150, the three phase-angle columns and a zone column.

    The depth-resolved map covers a single strip, so its zones are three bands
    along y rather than the two halves along x used for the shallow map. The
    band names carry which constituent dominates -- "basalt-centre",
    "flax-bottom", "flax-top" -- and are reduced here to the two-value
    vocabulary classify() expects. Without that reduction every compliant
    indent falls through unlabelled, which is what happened while this function
    was passing the band names straight through.
    """
    d = df.copy()
    band = d[zone_col].astype(str)
    zone = np.where(band.str.contains("basalt"), "basalt", "flax")

    d["constituent"] = classify(d.E_150.values, zone,
                                e_upper=E_UPPER_DEPTH).values

    # The paper rejects 304 of the 2000 indents on the phase-angle criterion.
    # make_classified_depth.py is where that decision is taken and stored, so
    # use the stored flag whenever it is there. The fallback below is for a frame
    # that arrives without it and is not a second opinion about it:
    # phase_angle_qc() is the tool's rule rather than the paper's and rejects 288
    # of the same 2000. Its docstring says how the two differ.
    if "qc_flag_strict" in d:
        rejected = d.qc_flag_strict.astype(bool)
    else:
        rejected = ~phase_angle_qc(d.PA_100, d.PA_150, d.PA_200)
    d.loc[rejected, "constituent"] = "QC/invalid"
    return d


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "data"

    raw = pd.read_csv(data / "shallow_map.csv")
    got = classify_shallow(raw)
    print(f"shallow map: {len(raw)} acquired, {len(got)} after the resin cut")
    counts = got.constituent.value_counts()
    print(counts.to_string())

    # Write it. `classified_shallow.csv` is a product of this script, not a
    # record: any existing copy is compared against the new labels and then
    # overwritten, so a disagreement cannot persist in the deposit.
    out = data / "classified_shallow.csv"
    if out.exists():
        ref = pd.read_csv(out)
        merged = got.merge(ref[["indent_id", "constituent"]], on="indent_id",
                           suffixes=("_here", "_before"))
        agree = (merged.constituent_here == merged.constituent_before).mean()
        print(f"\nagreement with the previous file: {100 * agree:.2f} %")
    got.to_csv(out, index=False, lineterminator="\n")
    print(f"written: {out.name}")
