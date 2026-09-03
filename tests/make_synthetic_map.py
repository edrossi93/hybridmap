"""
SYNTHETIC DATA. Nothing produced by this file is a measurement.

It writes a made-up indentation map so that the tool can be tested on something
that is not the paper's laminate: different column names, three phases instead
of four, one zone instead of two. If the tool only ever ran on the data it was
written for, "reusable" would be an assertion rather than a demonstration.

The map is three normal populations in modulus, well separated, scattered over
a rectangle. Because the populations are planted, the true phase of every point
is known and is written out alongside, which lets tests/test_tool.py ask the
only question worth asking of a classifier on synthetic data: does it recover
what was put in.

    python tests/make_synthetic_map.py [--out tests/_generated]

The output directory is ignored by git. The generator ships; the file it writes
does not, so that nothing here can be mistaken for data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed, so a failing test means the code changed.
SEED = 20240902

# Three populations: a compliant matrix, a stiff reinforcement, and contacts
# that caught both. Separated by several standard deviations, because the
# fixture is there to test the plumbing, not to be a hard classification
# problem; a test that fails because two populations genuinely overlap would
# tell us nothing about the tool.
POPULATIONS = [
    ("polymer",  600, 3.0, 0.5),
    ("boundary", 250, 20.0, 4.0),
    ("ceramic",  150, 60.0, 8.0),
]

# A handful of points with no usable modulus, so the quality path is exercised
# rather than assumed. Real maps always have some.
N_INVALID = 20


def build(seed=SEED):
    rng = np.random.default_rng(seed)

    modulus, truth = [], []
    for name, count, mean, sd in POPULATIONS:
        modulus.append(rng.normal(mean, sd, count))
        truth += [name] * count
    modulus = np.concatenate(modulus)

    modulus = np.append(modulus, np.full(N_INVALID, np.nan))
    truth += ["invalid"] * N_INVALID

    n = len(modulus)
    order = rng.permutation(n)
    frame = pd.DataFrame({
        # Column names chosen to share nothing with the paper's. If the tool
        # has a name baked in anywhere, this is what finds it.
        "point_id": np.arange(1, n + 1),
        "x_position_mm": rng.uniform(0.0, 4.0, n).round(4),
        "y_position_mm": rng.uniform(0.0, 1.5, n).round(4),
        "reduced_modulus_GPa": modulus[order].round(4),
        "true_phase": np.array(truth)[order],
    })
    return frame


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=here / "_generated")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "synthetic_map.csv"
    build().to_csv(target, index=False, lineterminator="\n")

    print(f"SYNTHETIC, not a measurement: {target}")
    print(f"{sum(c for _, c, _, _ in POPULATIONS) + N_INVALID} points, "
          f"{len(POPULATIONS)} planted populations, seed {SEED}")
