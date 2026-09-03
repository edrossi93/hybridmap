"""
Rule-based assignment of indents to the four classes.

Three of them are stiffness levels, compliant, mixed-volume and pure fibre. The
fourth exists because the compliant level holds two constituents that stiffness
cannot tell apart, so flax and PA-11 are separated by where they sit instead.
The depth-resolved acquisition corroborates that split from the other side: the
two compliant medians are 3.86 and 5.37 GPa at 150 nm, p = 8.2e-8.

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

    lower  15    compliant / stiff split, from the gap in the basalt-zone
                 modulus histogram
    upper  70    shallow map, pure-axial basalt above this
           40    depth-resolved map, lower because substrate compliance
                 depresses the fibre modulus at a 300 nm target depth
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
# histogram (Supplementary Figure S2a), so it separates polymer-dominated
# contacts from fibre-bearing ones without cutting through a population.
E_LOWER = 15.0

# Above this, a contact is taken as pure axial basalt. The two maps need
# different values because they sample mixed-volume contacts differently: the
# shallow map's small contact volume reaches nearly bare fibre, while at a
# 300 nm target depth the surrounding compliant material is inside the probed
# volume and depresses the apparent fibre modulus, so a 70 GPa cut there would
# reject almost every genuine basalt contact.
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
    True means keep. An indent is rejected if the phase angle is high anywhere
    in the analysed depth window, or very high right at the surface, which is
    the signature of a contact that has not settled.
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

    # The paper rejects 433 of the 2000 indents on the strict near-surface
    # phase-angle criterion. That flag is stored per indent, so use it when it
    # is present and recompute it only when it is not: phase_angle_qc() applies
    # the same thresholds but to the three slice values alone, and on a handful
    # of indents the full trace disagrees with the slices.
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

    # Cross-check against the labels used in the paper
    ref = pd.read_csv(data / "classified_shallow.csv")
    merged = got.merge(ref[["indent_id", "constituent"]], on="indent_id",
                       suffixes=("_here", "_paper"))
    agree = (merged.constituent_here == merged.constituent_paper).mean()
    print(f"\nagreement with the deposited labels: {100 * agree:.2f} %")
