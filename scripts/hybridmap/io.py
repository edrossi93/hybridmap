"""
Read a map table and work out which column plays which role.

Nothing here renames a column. It would be easy to load the file, rename the
user's modulus column to "modulus" and let the rest of the package assume that
name, and it would break the moment a config asked for clustering features or a
zone column that happened to be the one just renamed away. So the frame keeps
the user's own names and every function downstream is told which column to use.

Nothing here converts units either. The deposit's convention is that the unit
lives in the column name, and a tool that silently rescaled someone's data
would be doing the one thing this analysis cannot afford.
"""

from pathlib import Path

import pandas as pd

from .config import ConfigError

# Column roles this package understands. Anything else in map.columns is the
# user's business and is passed through untouched.
ROLES = ("id", "modulus", "position")


def read_map(spec):
    """Load the map table named by a validated MapSpec."""
    path = Path(spec.path)
    if not path.exists():
        raise ConfigError(f"map.path points at {path}, which does not exist. "
                          f"Paths in a config are read relative to the config "
                          f"file, not to where you are standing.")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ConfigError(f"could not read {path.name}: {exc}")


def resolve(df, spec):
    """
    Map role to actual column name, checking each one is really there.

    Returns a dict with the roles the config declared. Missing 'id' is allowed:
    plenty of exported tables have no identifier column, and nothing in the
    chain needs one except a join the user asks for themselves.
    """
    cols = {}
    for role in ROLES:
        name = spec.columns.get(role)
        if name is None:
            continue
        if name not in df.columns:
            raise ConfigError(
                f"map.columns.{role} is '{name}', which is not in "
                f"{Path(spec.path).name}. That file has: "
                f"{', '.join(map(str, df.columns))}")
        cols[role] = name
    if "modulus" not in cols:
        raise ConfigError("map.columns.modulus is required")
    return cols


def require(df, names, why):
    """Check a set of columns exists, saying what wanted them."""
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ConfigError(f"{why} needs columns that are not in the file: "
                          f"{', '.join(missing)}. Available: "
                          f"{', '.join(map(str, df.columns))}")


def depth_columns(spec, prop="modulus"):
    """
    The per-depth columns for one property, in ascending depth order.

    Empty for a single-depth map, which is the case the caller has to handle
    rather than paper over: a map with one modulus per indent has no depth
    axis, and inventing one would misrepresent it.
    """
    group = spec.by_depth.get(prop) or {}
    return [group[k] for k in sorted(group, key=float)]
