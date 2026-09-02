"""
Voigt, Reuss and Hill bounds from the per-class medians and indent-class
fractions, plus the sensitivity checks reported in the Supplementary
Information.

Two things worth stating plainly, because they are easy to misread:

  * The fractions are indent-class fractions. They are the number of indents
    carrying each label divided by the total number of classified indents. They
    are not measured constituent volume fractions. See classify.py for why.

  * Weighting the within-zone class frequencies by the zone indent counts and
    summing over zones gives exactly the global class counts over the global
    total. The two descriptions in the paper are the same arithmetic; the
    function below checks that rather than asserting it.
"""

import numpy as np
import pandas as pd

CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]


def bounds(fractions, moduli):
    """Voigt, Reuss and Hill for a set of parallel phases."""
    f = np.asarray(fractions, float)
    E = np.asarray(moduli, float)
    # Renormalise rather than fail. The caller usually passes fractions over
    # the four labelled classes, which already sum to one; but a threshold
    # sweep can drop a class entirely, and a silent 0.98 there would bias the
    # bound low without anything saying so.
    if not np.isclose(f.sum(), 1.0, atol=1e-6):
        f = f / f.sum()
    voigt = float((f * E).sum())
    reuss = float(1.0 / (f / E).sum())
    return voigt, 0.5 * (voigt + reuss), reuss


def class_table(df, modulus_col="E_GPa", classes=CLASSES):
    """Counts, fractions and median modulus per class, ignoring QC failures."""
    # QC/invalid and unclassified are excluded here rather than earlier, so the
    # fractions are over classified indents only. That is why they are
    # indent-class fractions and not constituent volume fractions: an indent
    # that failed the phase-angle check simply does not vote.
    d = df[df.constituent.isin(classes)]
    n = np.array([(d.constituent == c).sum() for c in classes], float)
    # Median, not mean. Each class has a long tail from contacts that partly
    # sample a neighbouring phase, and a mean would let those tails set the
    # input to the rule of mixtures.
    E = np.array([d.loc[d.constituent == c, modulus_col].median() for c in classes])
    return pd.DataFrame({"class": classes, "n": n.astype(int),
                         "fraction": n / n.sum(), "E_median_GPa": E})


def check_zone_weighting(df, classes=CLASSES):
    """
    Confirm that zone weighting and global counting agree. Returns the two
    fraction vectors so the caller can print them side by side.
    """
    d = df[df.constituent.isin(classes)]
    glob = np.array([(d.constituent == c).mean() for c in classes])
    zoned = np.zeros(len(classes))
    for z, sub in d.groupby("zone"):
        w = len(sub) / len(d)
        zoned += w * np.array([(sub.constituent == c).mean() for c in classes])
    return glob, zoned


def threshold_sweep(raw_shallow, lowers=(10, 15, 20), uppers=(60, 65, 70, 75, 80)):
    """
    Redo the classification and the Voigt bound for a grid of thresholds. This
    is Supplementary Table S1. Note that the table in the first submission was
    generated from an earlier classification pass and did not match the main
    text; this function regenerates it from the deposited data.
    """
    from classify import classify_shallow, classify

    # The resin cut and the zone boundaries are held fixed while the modulus
    # thresholds move. They are set by position, not by stiffness, so sweeping
    # them would be a different sensitivity test and would confound this one.
    d = raw_shallow[raw_shallow.x_um >= 380.0].copy()
    from classify import assign_zone
    d["zone"] = assign_zone(d.x_um.values).values

    out = []
    for lo in lowers:
        for hi in uppers:
            d["constituent"] = classify(d.E_GPa.values, d.zone.values,
                                        e_lower=lo, e_upper=hi).values
            t = class_table(d)
            v, h, r = bounds(t.fraction, t.E_median_GPa)
            out.append({"lower_GPa": lo, "upper_GPa": hi, "Voigt_GPa": v,
                        "Hill_GPa": h, "Reuss_GPa": r})
    return pd.DataFrame(out)


def substitute(table, cls, new_modulus):
    """Swap one class modulus and return the new bounds. Used for the basalt
    input sensitivity and for the bulk PA-11 comparison."""
    t = table.copy()
    t.loc[t["class"] == cls, "E_median_GPa"] = new_modulus
    return bounds(t.fraction, t.E_median_GPa)


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    data = Path(__file__).resolve().parents[1] / "data"

    sh = pd.read_csv(data / "classified_shallow.csv")
    tab = class_table(sh)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    v, h, r = bounds(tab.fraction, tab.E_median_GPa)
    print(f"\nVoigt {v:.2f}   Hill {h:.2f}   Reuss {r:.2f}  GPa")

    g, z = check_zone_weighting(sh)
    print(f"\nglobal fractions      {np.round(g, 5)}")
    print(f"zone-weighted          {np.round(z, 5)}")
    print(f"identical              {np.allclose(g, z)}")

    print(f"\nbasalt input swapped to the depth-resolved value (49.47 GPa): "
          f"Voigt {substitute(tab, 'basalt', 49.47)[0]:.2f} GPa")
    for Em in (1.0, 1.2, 1.5):
        print(f"matrix set to bulk PA-11 {Em} GPa: "
              f"Voigt {substitute(tab, 'matrix', Em)[0]:.2f} GPa")

    raw = pd.read_csv(data / "shallow_map.csv")
    sweep = threshold_sweep(raw)
    ref = sweep[(sweep.lower_GPa == 15) & (sweep.upper_GPa == 70)].Voigt_GPa.iloc[0]
    print(f"\nthreshold sweep: reference {ref:.2f} GPa, "
          f"range {sweep.Voigt_GPa.min():.2f} to {sweep.Voigt_GPa.max():.2f}, "
          f"max deviation {np.abs(sweep.Voigt_GPa - ref).max():.2f} GPa")
