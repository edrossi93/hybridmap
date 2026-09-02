"""
Reduce labelled indents to one row per phase: how many, what share, how stiff.

Two decisions are carried here and both are worth stating where the numbers are
made rather than in a paper somewhere.

The fractions are indent-class fractions: indents carrying a label over indents
carrying any label. They are not measured volume fractions. Anything excluded,
a quality failure or an indent no phase claimed, simply does not vote.

The central value is the median. Every phase has a tail from contacts that
partly sampled a neighbour, and a mean would let those tails set the input to
the rule of mixtures.
"""

import numpy as np
import pandas as pd


def class_table(df, classes, modulus_col, label_col="constituent",
                median_col="median"):
    """Counts, fractions and median modulus per phase, in the config's order."""
    d = df[df[label_col].isin(classes)]
    n = np.array([(d[label_col] == c).sum() for c in classes], float)
    if n.sum() == 0:
        raise ValueError(
            f"no indent carries any of the phase labels {list(classes)}. "
            f"Labels present: {sorted(set(map(str, df[label_col])))}")
    E = np.array([d.loc[d[label_col] == c, modulus_col].median()
                  for c in classes])
    return pd.DataFrame({"class": list(classes), "n": n.astype(int),
                         "fraction": n / n.sum(), median_col: E})


def check_zone_weighting(df, classes, label_col="constituent", zone_col="zone"):
    """
    Weighting the within-zone frequencies by zone size and summing over zones
    is the same arithmetic as counting globally. Papers assert this; returning
    both vectors lets the reader check it instead.
    """
    d = df[df[label_col].isin(classes)]
    glob = np.array([(d[label_col] == c).mean() for c in classes])
    zoned = np.zeros(len(classes))
    for _, sub in d.groupby(zone_col):
        w = len(sub) / len(d)
        zoned += w * np.array([(sub[label_col] == c).mean() for c in classes])
    return glob, zoned
