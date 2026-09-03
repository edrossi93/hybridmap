"""
How well the bound is known, by resampling rather than by propagation.

Two things vary and both are resampled from what was actually measured.

The per-phase medians are bootstrapped: resample the indents within a phase
with replacement, take the median of each resample. The bootstrap distribution
is used as it stands rather than summarised into an interval first, because
summarising it and then drawing from the summary throws away the shape and
misrepresents the tails.

The fractions are drawn from a Dirichlet with concentration equal to the
observed counts, which is the posterior for multinomial proportions under a
uniform prior.

Each resample of a phase has the size of that phase, which is what makes the
spread reflect how well the phase is actually sampled: a class carrying a few
hundred indents is less certain than one carrying a few thousand, and the
budget should say so instead of treating them alike.

What is deliberately outside this budget is anything that is a choice rather
than a random quantity. Where the cuts go is a choice, reported by
`mixtures.sweep`. Which value to use for a phase is a choice, reported by
`mixtures.substitute`. Folding a choice into a random budget would present it
as noise and let the interval absorb a disagreement the reader should see.
"""

import numpy as np

from .mixtures import bounds


def bootstrap_medians(df, classes, modulus_col, n_draws=10_000, seed=0,
                      label_col="constituent", seed_per_phase=False):
    """One column of median replicates per phase, in the order given.

    `seed_per_phase` picks between two readings of a stated seed, which give
    the same interval to the first decimal and can differ in the second:

      False   one generator, phases drawn in the order `classes` lists them,
              so a phase's replicates depend on how many were drawn before it.
              This is what the budget uses, and changing it would move the
              published spread on the bound.
      True    each phase gets its own generator from the same seed, so a phase
              is independent of its position in the list and of which other
              phases were requested.

    The second is the better default for a per-phase table, because a reader
    recomputing one row should not have to know what was computed before it.
    The first is kept because the budget was published under it.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in classes:
        if seed_per_phase:
            rng = np.random.default_rng(seed)
        values = df.loc[df[label_col] == name, modulus_col].to_numpy(float)
        # Non-finite values are dropped rather than counted as zero. They are
        # failed fits, and a zero would pull the median down.
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"phase '{name}' has no usable values to resample")
        index = rng.integers(0, len(values), size=(n_draws, len(values)))
        out[name] = np.median(values[index], axis=1)
    return out


def intervals(replicates, level=95.0):
    """Percentile interval per phase, from its bootstrap replicates."""
    tail = (100.0 - level) / 2.0
    return {name: (float(np.percentile(r, tail)),
                   float(np.percentile(r, 100.0 - tail)))
            for name, r in replicates.items()}


def budget(table, replicates, classes, seed=0, n_draws=10_000,
           median_col="median", level=95.0):
    """
    The spread on the Voigt bound, and where it comes from.

    Three sums from the same replicates: everything varying, then each source
    varying alone with the other held at its point value. That is how the
    components are attributed, and why they are reported as separate standard
    deviations and not as one number.
    """
    E = np.column_stack([replicates[c] for c in classes])

    # A separate generator, seeded the same way. The two draws are independent
    # by construction: the medians resample indents within a phase, the
    # fractions resample how many indents each phase holds. Treating them as
    # uncorrelated is what lets the two component standard deviations below add
    # in quadrature to the combined one.
    rng = np.random.default_rng(seed)
    f = rng.dirichlet(table.n.to_numpy(float), size=n_draws)

    fractions = table.fraction.to_numpy()
    medians = table[median_col].to_numpy()
    everything = (f * E).sum(axis=1)
    medians_only = (fractions * E).sum(axis=1)
    fractions_only = (f * medians).sum(axis=1)

    tail = (100.0 - level) / 2.0
    return {
        "point": float(bounds(fractions, medians)[0]),
        "n_draws": n_draws,
        "seed": seed,
        "components_sd": {
            "class_medians": float(medians_only.std(ddof=1)),
            "class_fractions": float(fractions_only.std(ddof=1)),
        },
        "combined_sd": float(everything.std(ddof=1)),
        "interval": [float(np.percentile(everything, tail)),
                     float(np.percentile(everything, 100.0 - tail))],
        "level": level,
        "draws": everything,
    }
