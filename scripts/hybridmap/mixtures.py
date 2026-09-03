"""
Voigt, Reuss and Hill from measured inputs, and the sensitivity of the answer
to the cuts that produced them.

The sweep is not an optional extra. A cut chosen from a histogram gap is a
judgement, and a tool that lets a user set one without showing how far the
answer moves when it is set differently is hiding the weakest step in the
chain. `sweep` re-runs the whole classification for each combination of cut
values, so what moves is the actual answer and not a linearisation of it.
"""

import itertools

import numpy as np
import pandas as pd

from .aggregate import class_table
from .classify import classify


def bounds(fractions, moduli):
    """Voigt, Hill and Reuss for a set of parallel phases."""
    f = np.asarray(fractions, float)
    E = np.asarray(moduli, float)
    # Renormalise rather than fail. A sweep can drop a phase entirely, and a
    # silent 0.98 would bias the bound low without anything saying so.
    if not np.isclose(f.sum(), 1.0, atol=1e-6):
        f = f / f.sum()
    voigt = float((f * E).sum())
    reuss = float(1.0 / (f / E).sum())
    return voigt, 0.5 * (voigt + reuss), reuss


def substitute(table, cls, new_modulus, median_col="median"):
    """Swap one phase's modulus and rebound. For asking what one input is worth."""
    t = table.copy()
    t.loc[t["class"] == cls, median_col] = new_modulus
    return bounds(t.fraction, t[median_col])


def sweep(frame, phases, cuts, grids, modulus_col, classes=None,
          label_col="constituent", zone_col="zone", median_col="median"):
    """
    Redo the classification and the bounds across a grid of cut values.

    `frame` must already carry its zones and have any discarded band dropped.
    The spatial cuts are deliberately held still while the modulus cuts move:
    they are set by position rather than by stiffness, so sweeping them would
    be a different sensitivity test and would confound this one.
    """
    if not grids:
        raise ValueError("nothing to sweep: add a sweep section naming cuts "
                         "and the values to try for each")

    names = list(grids)
    # Phase order fixes the column order of every table downstream, so it comes
    # from the config, not from whatever order the labels happen to appear in.
    classes = list(classes) if classes else [p.name for p in phases]
    E = frame[modulus_col].to_numpy()
    zone = frame[zone_col].to_numpy() if zone_col in frame else None

    rows = []
    for combo in itertools.product(*(grids[n] for n in names)):
        moved = dict(cuts)
        moved.update(zip(names, combo))
        labels = classify(E, zone, phases, moved).values
        # "allow" here: a sweep exists to try cuts the user has not committed
        # to, and one that empties a phase is a result to show as nan.
        t = class_table(frame.assign(**{label_col: labels}), classes,
                        modulus_col, label_col, median_col, on_empty="allow")
        v, h, r = bounds(t.fraction, t[median_col])
        row = {f"{n}_GPa": value for n, value in zip(names, combo)}
        row.update({"Voigt_GPa": v, "Hill_GPa": h, "Reuss_GPa": r})
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_summary(table, cuts):
    """
    The reference construction and how far the grid moves it.

    The caller decides how to say it. The reference row is found here because getting it wrong, comparing the grid
    against its own mean instead of against the configured cuts, would flatter
    the answer.
    """
    mask = pd.Series(True, index=table.index)
    for name, value in cuts.items():
        col = f"{name}_GPa"
        if col in table:
            mask &= np.isclose(table[col], value)
    if not mask.any():
        return None
    ref = float(table.loc[mask, "Voigt_GPa"].iloc[0])
    return {"reference_GPa": ref,
            "min_GPa": float(table.Voigt_GPa.min()),
            "max_GPa": float(table.Voigt_GPa.max()),
            "max_deviation_GPa": float(np.abs(table.Voigt_GPa - ref).max())}
