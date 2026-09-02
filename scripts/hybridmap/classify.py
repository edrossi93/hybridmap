"""
Assign each indent to a zone and then to a phase.

The rule has two ingredients and needs both. Modulus alone cannot separate two
compliant constituents whose indentation moduli overlap. Position alone cannot
separate a fibre from the matrix around it. So a phase is a modulus interval,
optionally qualified by the zone it has to sit in, and the zone carries whatever
architectural knowledge the user has about their own section.

Phases are matched in the order the config lists them and the first match wins.
An indent matching no phase is left unclassified rather than forced into the
nearest one: it does not then count towards any fraction, which is the honest
outcome and is why `config.coverage_warnings` says out loud when an interval is
unclaimed.
"""

import numpy as np
import pandas as pd

from . import io
from .config import ConfigError

ZONE_COL = "zone"
LABEL_COL = "constituent"

# An indent whose modulus is not a finite number never reached a usable
# contact, so it is separated from the merely unclassified: one is a failed
# measurement, the other is a gap in the rule, and they should not be counted
# together.
INVALID = "QC/invalid"
UNMATCHED = "unclassified"


def assign_zones(df, zones, cols):
    """Zone name per row, as a Series on the frame's own index."""
    if zones is None:
        # One zone, so that everything downstream can group by zone without a
        # special case for the modulus-only configuration.
        return pd.Series(["all"] * len(df), index=df.index)

    if zones.column:
        if zones.column not in df.columns:
            raise ConfigError(
                f"zones.from_column.column is '{zones.column}', which is not "
                f"in the map file. Available: {', '.join(map(str, df.columns))}")
        raw = df[zones.column].astype(str)
        if zones.mapping:
            # Values with no entry keep their own name, so a column that
            # already holds the zone names needs no map at all.
            return raw.map(lambda v: zones.mapping.get(v, v))
        return raw

    axis = cols.get(zones.axis)
    if axis is None:
        raise ConfigError(f"zones.axis is '{zones.axis}', which this map does "
                          f"not declare in map.columns")
    x = df[axis].to_numpy(dtype=float)
    conditions = [x < b.below for b in zones.bands if b.below is not None]
    names = [b.name for b in zones.bands if b.below is not None]
    if not conditions:
        # A single open band: everything is in it. Worth handling rather than
        # forbidding, because one zone is the honest description of a specimen
        # with no architecture to exploit, and np.select cannot be asked for a
        # default with nothing to select on.
        return pd.Series(zones.bands[-1].name, index=df.index)
    return pd.Series(np.select(conditions, names, default=zones.bands[-1].name),
                     index=df.index)


def discarded(zone_series, zones):
    """True where the row falls in a band the config marks for discard."""
    if zones is None or zones.column:
        return pd.Series(False, index=zone_series.index)
    drop = {b.name for b in zones.bands if b.discard}
    return zone_series.isin(drop)


def classify(modulus, zone, phases, cuts,
             unmatched=UNMATCHED, invalid=INVALID):
    """Phase label per indent, from modulus and zone. Returns a Series."""
    E = np.asarray(modulus, dtype=float)
    zone = np.asarray(zone) if zone is not None else None
    out = np.full(E.shape, unmatched, dtype=object)

    # Reverse order with unconditional assignment gives first-match-wins
    # without a per-row loop: a later phase writes first and an earlier one
    # overwrites it.
    for phase in reversed(phases):
        lo = -np.inf if phase.lower is None else cuts[phase.lower]
        hi = np.inf if phase.upper is None else cuts[phase.upper]
        hit = (E >= lo) & (E < hi)
        if phase.zone is not None:
            if zone is None:
                raise ConfigError(f"phase '{phase.name}' names a zone but no "
                                  f"zones were assigned")
            hit &= (zone == phase.zone)
        out[hit] = phase.name

    out[~np.isfinite(E)] = invalid
    return pd.Series(out)


def qc_keep(df, qc):
    """
    True means keep the indent.

    A stored flag wins when the file has one. Whoever wrote it saw the whole
    trace; the threshold form below sees only the slices it is given, and on a
    handful of indents the two disagree.
    """
    if qc is None:
        return pd.Series(True, index=df.index)

    if qc.flag_column and qc.flag_column in df.columns:
        return ~df[qc.flag_column].astype(bool)

    if not qc.columns or qc.limit is None:
        return pd.Series(True, index=df.index)

    io.require(df, qc.columns, "the phase-angle quality check")
    values = np.vstack([df[c].to_numpy(float) for c in qc.columns])
    bad = np.nanmax(values, axis=0) > qc.limit
    if qc.first_limit is not None:
        # The first column is the shallowest. A contact can settle by the
        # deeper slices and still have been dominated by the establishment
        # transient where it was measured first.
        bad |= df[qc.columns[0]].to_numpy(float) > qc.first_limit
    return pd.Series(~bad, index=df.index)


def apply(df, cfg, cols, zone_col=ZONE_COL, label_col=LABEL_COL):
    """
    The whole labelling step: zones, the discard, the phase rule, then QC.

    Returns a new frame carrying the zone and the label. Discarded rows are
    dropped here rather than flagged, because a band marked for discard is
    material the user has said is not part of the specimen.
    """
    zones = assign_zones(df, cfg.zones, cols)
    kept = df.loc[~discarded(zones, cfg.zones)].copy()
    kept[zone_col] = zones.loc[kept.index]

    kept[label_col] = classify(kept[cols["modulus"]].to_numpy(),
                               kept[zone_col].to_numpy(),
                               cfg.phases, cfg.cuts).values

    rejected = ~qc_keep(kept, cfg.qc)
    kept.loc[rejected.values, label_col] = INVALID
    return kept
