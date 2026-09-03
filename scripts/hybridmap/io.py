"""
Read a table, work out which column plays which role, and refuse clearly when
the file is not what the config says it is.

The contract is deliberately small. One row per indent for a map, one row per
recorded point for a curve, and the columns named in the config have to be
numbers. Everything else in the file is carried through untouched.

Nothing here renames a column. It would be easy to load the file, rename the
user's modulus column to "modulus" and let the rest of the package assume that
name, and it would break the moment a config asked for clustering features or a
zone column that happened to be the one just renamed away. So the frame keeps
the user's own names and every function downstream is told which column to use.

Nothing here converts units either. The deposit's convention is that the unit
lives in the column name, and a tool that rescaled someone's data behind
their back would be doing the one thing this analysis cannot afford.
"""

from pathlib import Path

import pandas as pd

from .config import ConfigError

# Column roles this package understands. Anything else in map.columns is the
# user's business and is passed through untouched.
ROLES = ("id", "modulus", "position")


def read_table(path, options=None, what="table"):
    """Read a CSV, with the reader options the config declared."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"the {what} path points at {path}, which does not "
                          f"exist. Paths in a config are read relative to the "
                          f"config file, not to where you are standing.")
    try:
        frame = pd.read_csv(path, **(options or {}))
    except Exception as exc:
        raise ConfigError(f"could not read {path.name}: {exc}")

    if frame.empty:
        raise ConfigError(f"{path.name} has no rows")
    if len(frame.columns) == 1:
        # One column almost always means the separator is wrong. The symptom
        # otherwise surfaces much later, as a missing column.
        raise ConfigError(
            f"{path.name} read as a single column, {frame.columns[0]!r}. If it "
            f"is not comma separated, say so with a read block, for example "
            f"map.read.sep: ';'")
    return frame


def read_map(spec):
    """Load the map table named by a validated MapSpec."""
    return read_table(spec.path, spec.read, "map")


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


def numeric(df, column, why="this column"):
    """
    A column as floats, or a refusal that says what is wrong with it.

    Empty cells are allowed and become NaN: a failed fit is a real thing for a
    map to contain, and the classification has a place for it. Text where a
    number belongs is not allowed: the alternative is dropping every row the
    reader could not parse and reporting a median over what is left.
    """
    values = df[column]
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)

    coerced = pd.to_numeric(values, errors="coerce")
    unreadable = values[coerced.isna() & values.notna()]
    if unreadable.empty:
        return coerced.astype(float)

    examples = list(dict.fromkeys(str(v) for v in unreadable))[:3]
    hint = ""
    if any("," in v for v in examples):
        # By far the most common cause, and the one whose failure is otherwise
        # a numpy conversion error six frames down.
        hint = (" These look like decimal commas. Add a read block: "
                "map.read.decimal: ',' and, if the file is semicolon "
                "separated, map.read.sep: ';'")
    raise ConfigError(
        f"column '{column}', which {why}, is not numeric: {len(unreadable)} of "
        f"{len(values)} values cannot be read as numbers, for example "
        f"{', '.join(repr(e) for e in examples)}.{hint}")


def describe_map(df, cols, cfg):
    """
    What the tool sees in the file, as plain numbers.

    Returned rather than printed so the caller decides how to say it. This is
    the report that answers the two questions a user actually has when a run
    looks wrong: did it read my file the way I meant, and are my cuts anywhere
    near my data.
    """
    modulus = numeric(df, cols["modulus"], "the phases are cut on")
    finite = modulus[modulus.notna()]

    report = {
        "rows": len(df),
        "modulus_column": cols["modulus"],
        "usable": int(finite.size),
        "missing": int(modulus.isna().sum()),
        "min": float(finite.min()) if finite.size else None,
        "median": float(finite.median()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "warnings": [],
    }

    if "id" in cols:
        repeated = int(df[cols["id"]].duplicated().sum())
        if repeated:
            report["warnings"].append(
                f"{repeated} rows repeat an identifier in "
                f"'{cols['id']}'; a map should have one row per indent")

    if "position" in cols:
        position = numeric(df, cols["position"], "the zones are cut on")
        report["position"] = (float(position.min()), float(position.max()))
        if position.isna().any():
            report["warnings"].append(
                f"{int(position.isna().sum())} rows have no position, so they "
                f"fall in the last zone band by default")

    # A cut outside the data is not an error: a user may be reusing a config
    # across specimens. It does mean one phase will be empty, and that
    # should be said before the bound comes out of it.
    if finite.size:
        for name, value in cfg.cuts.items():
            if value < finite.min() or value > finite.max():
                report["warnings"].append(
                    f"cut '{name}' at {value:g} is outside the range of "
                    f"{cols['modulus']}, {finite.min():.3g} to "
                    f"{finite.max():.3g}, so a phase will have no indents")
    return report


def depth_columns(spec, prop="modulus"):
    """
    The per-depth columns for one property, in ascending depth order.

    Empty for a single-depth map, which is the case the caller has to handle
    rather than paper over: a map with one modulus per indent has no depth
    axis, and inventing one would misrepresent it.
    """
    group = spec.by_depth.get(prop) or {}
    return [group[k] for k in sorted(group, key=float)]
