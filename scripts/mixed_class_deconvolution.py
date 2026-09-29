"""
Inverting the mixed-volume class back onto the basalt/matrix tie line.

The mixed-volume fraction depends on the contact footprint, so it is a property
of the measurement and not of the laminate. The test worth running is therefore
to deconvolve the mixed class into its two end members and ask whether what
comes out is the same in both acquisitions, whose footprints differ by roughly a
factor of four in contact depth.

The inversion is applied to every indent in the basalt-rich zone, not only to
those the rule called mixed-volume. That is deliberate. A class fraction is
built out of the thresholds, so any quantity derived from the class counts
inherits the very parameter under test. Inverting each indent from its own
modulus removes the thresholds from the calculation entirely, which makes the
result a fair test rather than a restatement.

Two inversions are reported because the contact does not tell us which one it
obeys, and choosing the one that agrees best would be circular reasoning.

    linear      E = f E_b + (1 - f) E_m
    compliance  1/E = f/E_b + (1 - f)/E_m

The linear form treats the two phases beneath the tip as loaded in parallel; the
compliance-weighted form treats them as loaded in series. Neither is exact for a
stiff fibre embedded in a compliant medium, but they are not equally plausible.
The load path under a Berkovich tip runs down through whatever lies beneath it,
so a compliant layer under or beside a stiff one governs the displacement out of
proportion to its volume, which is why the modulus of a stiff film on a
compliant substrate falls away with depth far faster than any linear mixing
predicts. The compliance-weighted inversion is the closer first-order picture on
that argument alone, before any of its numbers are looked at.

End members are each acquisition's own basalt and matrix class medians, since
the point of the test is whether the two acquisitions agree once each is read on
its own scale. The two-class sum is then rebuilt on one common pair of end
members, the shallow one, because that is the constituent input the paper uses:
the two acquisitions agree on the same contacts at 80 nm, and the deeper reading
of those contacts then falls with depth for a reason the data do not settle
(4.4), so the depth-resolved basalt median is not taken as a second estimate of
the fibre modulus.

The independent number to compare against is the packing of basalt inside the
basalt-rich zone measured on the same polished section by backscattered-electron
imaging, which is 0.665 and shares no input with the indentation.

One more inversion lives here and is not part of the product this script
writes. 4.1 says the mixed-volume median "matches a Voigt-Reuss-Hill mixture of
basalt and PA-11" at a basalt fraction of 0.55 in the depth-resolved contacts and
0.67 in the shallower ones. Those two figures come from `hill_fraction`: the
class median is placed on the Hill curve, the arithmetic mean of the Voigt and
Reuss curves, between the two shallow end members, 86.0 and 4.66 GPa, and the
curve is solved for the basalt fraction. Both medians are inverted on the
shallow end members, the depth-resolved one included, because the question 4.1
asks is how much fibre a contact of that modulus holds, and the fibre modulus
the paper uses is the shallow one. On the linear and compliance forms above the
same medians give 0.31 and 0.38, and 0.89 and 0.92, so the printed pair holds
under the Hill convention only, and 4.1 has to say which convention it is.
`reproduce_paper_values.py` checks both figures through this function.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

SEED = 42
N_BOOT = 4000

# Where basalt, matrix and mixed contacts coexist, so that a basalt/matrix tie
# line is the right one. The flax-rich zone has no third end member to invert
# onto and is excluded.
ZONE = {"shallow": "basalt", "depth": "basalt-centre"}
COLUMN = {"shallow": "E_GPa", "depth": "E_150"}


def zone_moduli(kind):
    """Valid moduli in the basalt-rich zone, with the two end members."""
    name = "classified_shallow.csv" if kind == "shallow" \
        else "classified_depth_resolved.csv"
    df = pd.read_csv(DATA / name)
    col = COLUMN[kind]
    z = df[(df.zone == ZONE[kind]) & (df.constituent != "QC/invalid")]
    e_b = float(z.loc[z.constituent == "basalt", col].median())
    e_m = float(z.loc[z.constituent == "matrix", col].median())
    return z[col].to_numpy(dtype=float), e_b, e_m


def invert(e, e_b, e_m):
    """Basalt fraction per indent under each mixing assumption."""
    lin = (e - e_m) / (e_b - e_m)
    ser = (1.0 / e_m - 1.0 / e) / (1.0 / e_m - 1.0 / e_b)
    # Values outside the tie line are contacts stiffer than the basalt median or
    # more compliant than the matrix median, i.e. the scatter of the end members
    # themselves. They are pinned rather than dropped, so that the mean stays a
    # mean over every indent in the zone.
    return np.clip(lin, 0.0, 1.0), np.clip(ser, 0.0, 1.0)


def hill_fraction(e, e_b, e_m):
    """The basalt fraction at which the Hill average of the two end members is `e`.

    The Hill average is half the Voigt sum plus half the Reuss value. It rises
    monotonically from e_m at zero basalt to e_b at all basalt, so the fraction
    is found by bisection on that interval; a modulus outside it has no
    fraction and raises.
    """
    if not e_m < e < e_b:
        raise ValueError(f"{e} GPa is not between the end members {e_m} and "
                         f"{e_b} GPa")

    def hill(f):
        return 0.5 * (f * e_b + (1.0 - f) * e_m) \
            + 0.5 / (f / e_b + (1.0 - f) / e_m)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hill(mid) < e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def boot(f, rng):
    """95 % interval on the mean, resampling indents."""
    n = len(f)
    draws = [f[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def endpoint_sensitivity(e, e_b, e_m):
    """How far the answer moves if the end members are chosen differently.

    The basalt end member is swept over the interval the paper already carries
    on it, and the matrix end member from the bulk tensile value of PA-11 up to
    the measured one, which is the full span of plausible matrix stiffness.
    """
    out = []
    for b in (e_b - 2.5, e_b, e_b + 2.5):
        for m in (1.0, 2.5, e_m):
            _, ser = invert(e, b, m)
            out.append({"e_basalt": round(b, 2), "e_matrix": round(m, 2),
                        "phi_compliance": round(float(ser.mean()), 4)})
    return out


def main():
    rng = np.random.default_rng(SEED)
    result = {"seed": SEED, "n_bootstrap": N_BOOT, "acquisitions": {}}

    for kind in ("shallow", "depth"):
        e, e_b, e_m = zone_moduli(kind)
        lin, ser = invert(e, e_b, e_m)
        result["acquisitions"][kind] = {
            "zone": ZONE[kind],
            "n_indents": int(len(e)),
            "e_basalt_end_member_GPa": round(e_b, 3),
            "e_matrix_end_member_GPa": round(e_m, 3),
            "phi_linear": round(float(lin.mean()), 4),
            "phi_linear_ci95": [round(v, 4) for v in boot(lin, rng)],
            "phi_compliance": round(float(ser.mean()), 4),
            "phi_compliance_ci95": [round(v, 4) for v in boot(ser, rng)],
            "endpoint_sensitivity": endpoint_sensitivity(e, e_b, e_m),
        }

    # The two-class sum the comment asks about, rebuilt on one common pair of
    # end members so that what is being compared is the fraction and not the
    # scale each acquisition reads the fibre on.
    sh = result["acquisitions"]["shallow"]
    e_b = sh["e_basalt_end_member_GPa"]
    e_m = sh["e_matrix_end_member_GPa"]
    two = {}
    for kind, r in result["acquisitions"].items():
        for model in ("linear", "compliance"):
            phi = r[f"phi_{model}"]
            two.setdefault(model, {})[kind] = round(phi * e_b + (1 - phi) * e_m, 2)
    for model, v in two.items():
        lo, hi = sorted(v.values())
        v["spread_pct"] = round(100.0 * (hi - lo) / ((hi + lo) / 2), 1)
    result["two_class_sum_GPa"] = {
        "end_members_from": "shallow",
        "e_basalt_GPa": e_b, "e_matrix_GPa": e_m, **two}

    # Class fractions, for the contrast the answer turns on.
    for kind in ("shallow", "depth"):
        name = "classified_shallow.csv" if kind == "shallow" \
            else "classified_depth_resolved.csv"
        df = pd.read_csv(DATA / name)
        z = df[(df.zone == ZONE[kind]) & (df.constituent != "QC/invalid")]
        result["acquisitions"][kind]["mixed_class_fraction"] = round(
            float((z.constituent == "mixed-volume").mean()), 4)

    # Whether the two acquisitions agree is a different question from where the
    # answer sits, and the sweep separates them: matched end members move the
    # absolute fraction a great deal and the gap between the acquisitions
    # hardly at all.
    def sweep(kind):
        r = result["acquisitions"][kind]
        base = r["e_basalt_end_member_GPa"]
        return {(x["e_matrix"], round(x["e_basalt"] - base, 1)):
                x["phi_compliance"] for x in r["endpoint_sensitivity"]}

    a_s, a_d = sweep("shallow"), sweep("depth")
    matched = []
    for key in sorted(set(a_s) & set(a_d)):
        s_v, d_v = a_s[key], a_d[key]
        matched.append({"e_matrix_GPa": key[0], "e_basalt_offset_GPa": key[1],
                        "phi_shallow": s_v, "phi_depth": d_v,
                        "gap_pct": round(200.0 * abs(d_v - s_v) / (d_v + s_v), 1)})
    result["cross_acquisition_agreement"] = {
        "matched_end_members": matched,
        "phi_span_over_sweep": [round(min(list(a_s.values()) + list(a_d.values())), 3),
                                round(max(list(a_s.values()) + list(a_d.values())), 3)],
        "gap_span_pct": [min(m["gap_pct"] for m in matched),
                         max(m["gap_pct"] for m in matched)],
    }

    imaged = json.loads((DATA / "bse_segmentation_summary.json").read_text())
    result["independent_comparison"] = {
        "in_zone_basalt_packing_from_imaging": round(
            imaged["in_zone_fibre_fraction_pooled_over_tiles"]["basalt_fib"], 4),
        "note": "Backscattered-electron segmentation of the same polished "
                "section, pooled over 91 fields. Shares no input with the "
                "indentation.",
    }

    out = DATA / "mixed_class_deconvolution.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    a, b = result["acquisitions"]["shallow"], result["acquisitions"]["depth"]
    print("mixed-volume class fraction   %.3f   %.3f   (factor %.1f apart)"
          % (a["mixed_class_fraction"], b["mixed_class_fraction"],
             a["mixed_class_fraction"] / b["mixed_class_fraction"]))
    print("deconvolved, linear           %.3f   %.3f" %
          (a["phi_linear"], b["phi_linear"]))
    print("deconvolved, compliance       %.3f   %.3f" %
          (a["phi_compliance"], b["phi_compliance"]))
    print("imaged in-zone basalt packing %.3f"
          % result["independent_comparison"]
                  ["in_zone_basalt_packing_from_imaging"])
    print("two-class sum, compliance     %.2f   %.2f GPa  (%.1f %% apart)"
          % (result["two_class_sum_GPa"]["compliance"]["shallow"],
             result["two_class_sum_GPa"]["compliance"]["depth"],
             result["two_class_sum_GPa"]["compliance"]["spread_pct"]))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
