"""
Load a configuration, and refuse it clearly when it cannot work.

The whole point of this file is the second half of that sentence. A tool that
accepts a config with a phase referencing a zone that does not exist, and then
silently labels nothing, is worse than one that will not start.

Two conventions are worth reading before writing a config.

Cuts are named once and referenced by name. A phase says it sits below the cut
called "lower", not below 15.0. If each phase carried its own number, the same
threshold would appear twice, the two copies would eventually disagree, and the
sensitivity sweep would have nothing well defined to move. Naming them is what
makes `hybridmap sweep` possible at all.

Phases are matched in the order they are written and the first match wins. That
is a rule the reader can follow by eye, which matters more here than
flexibility: the classification is the step a sceptical reader will want to
reconstruct by hand.
"""

# Annotations are not evaluated at import with this in place, which keeps the
# `float | None` spellings below from raising on Python 3.8 and 3.9. Without it
# the whole deposit, reproduction path included, silently needs 3.10.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """Raised with a message written for whoever has to fix the file."""


@dataclass(frozen=True)
class Band:
    """One zone band along a spatial axis, open at the top if `below` is None."""
    name: str
    below: float | None = None
    discard: bool = False


@dataclass(frozen=True)
class Zones:
    """
    Either bands cut on a coordinate, or zone names read from a column.

    The second form exists because a zone is sometimes already decided before
    the analysis starts, by hand on a micrograph or by an earlier pass, and
    re-deriving it from a coordinate would be inventing agreement.
    """
    axis: str | None = None
    bands: tuple[Band, ...] = ()
    column: str | None = None
    mapping: dict = field(default_factory=dict)

    @property
    def names(self):
        """Zone names a phase may legitimately name, discarded bands excluded."""
        if self.column:
            return sorted(set(self.mapping.values())) if self.mapping else []
        return [b.name for b in self.bands if not b.discard]


@dataclass(frozen=True)
class Phase:
    """
    One mechanical population: a modulus interval, optionally inside one zone.

    `lower` and `upper` hold cut *names*, not values, so a sweep can move the
    cut without rewriting the phases. Either may be None, meaning unbounded on
    that side.
    """
    name: str
    lower: str | None = None
    upper: str | None = None
    zone: str | None = None
    colour: str | None = None


@dataclass(frozen=True)
class MapSpec:
    path: Path
    kind: str
    columns: dict
    by_depth: dict = field(default_factory=dict)
    read: dict = field(default_factory=dict)


# What a `read:` block may set, passed straight to pandas.read_csv. Deliberately
# short. These four cover the exports people actually have to deal with, a
# semicolon-separated European file with decimal commas and a placeholder for a
# failed fit, and each one is a statement about the file rather than about the
# analysis. Anything longer would be a second configuration language.
READ_OPTIONS = {"sep", "decimal", "encoding", "na_values"}


@dataclass(frozen=True)
class QC:
    """
    Contact-quality rejection. A stored per-indent flag is trusted when the
    file carries one, because whoever produced it saw the full trace; the
    threshold form is the fallback for a file that does not.
    """
    flag_column: str | None = None
    columns: tuple[str, ...] = ()
    limit: float | None = None
    first_limit: float | None = None


@dataclass(frozen=True)
class Measured:
    """A macroscopic measurement to compare the prediction against."""
    label: str
    value: float
    sd: float | None = None
    n: int | None = None


@dataclass(frozen=True)
class Curves:
    """
    The bending-curve route to a measured modulus.

    Two input shapes are accepted. Stress against strain, which is what a curve
    already reduced by the testing software looks like, and force against
    displacement, which is what the machine actually records. The second needs
    the specimen geometry, and is converted with the same standard the chord
    and the toe correction come from.
    """
    path: Path
    columns: dict
    chord: tuple[float, float]
    toe_window: float | None = None
    extra_window: tuple[float, float] | None = None
    geometry: dict = field(default_factory=dict)
    modulus_scale: float = 0.1
    read: dict = field(default_factory=dict)

    @property
    def is_raw(self):
        """True when the file holds force and displacement rather than stress."""
        return "force" in self.columns and "displacement" in self.columns


# What the geometry block has to give, in millimetres, to turn a force and a
# displacement into a stress and a strain.
GEOMETRY = ("span", "width", "thickness")

# The chord is a slope in stress-unit per per-cent-strain, so with stress in
# MPa it is MPa per per cent, and a tenth of that is GPa. This default is right
# for the two shapes documented above, N and mm giving MPa, and wrong for
# anyone feeding in stress already in GPa, who should set it to 1.
DEFAULT_MODULUS_SCALE = 0.1


@dataclass(frozen=True)
class Macro:
    measured: tuple[Measured, ...] = ()
    curves: Curves | None = None


@dataclass(frozen=True)
class Clustering:
    features: tuple[str, ...] = ()
    k: int = 2
    seed: int = 0
    n_init: int = 10
    merge_lowest: int = 0


@dataclass(frozen=True)
class Config:
    name: str
    map: MapSpec
    cuts: dict
    phases: tuple[Phase, ...]
    zones: Zones | None = None
    sweep: dict = field(default_factory=dict)
    macro: Macro | None = None
    clustering: Clustering | None = None
    qc: QC | None = None
    source: Path | None = None

    @property
    def phase_names(self):
        return [p.name for p in self.phases]

    @property
    def colours(self):
        return {p.name: p.colour for p in self.phases if p.colour}


# ----------------------------------------------------------------------------
# reading

def _read_document(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no config file at {path}")
    text = path.read_text(encoding="utf8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise ConfigError(
                "reading a YAML config needs PyYAML, which is in "
                "requirements.txt: pip install -r requirements.txt. A JSON "
                "config with the same keys works without it.")
        doc = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        doc = json.loads(text)
    else:
        raise ConfigError(f"{path.name}: expected a .yaml, .yml or .json file")
    if not isinstance(doc, dict):
        raise ConfigError(f"{path.name}: the top level must be a mapping of "
                          f"sections, not {type(doc).__name__}")
    return doc


def apply_overrides(doc, assignments):
    """
    Apply `--set cuts.lower=12` style overrides to the raw document.

    They are applied before validation, not after, so an override is checked
    exactly as if it had been written into the file. An override that creates a
    key nobody reads is a typo, so a path whose parent does not exist is an
    error rather than a new branch of the document.
    """
    for item in assignments or ():
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = doc
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise ConfigError(f"--set {dotted}: there is no section "
                                  f"{'.'.join(keys[:-1])} in this config")
            node = node[k]
        if not isinstance(node, dict):
            raise ConfigError(f"--set {dotted}: {'.'.join(keys[:-1])} is not a "
                              f"section")
        node[keys[-1]] = _scalar(raw)
    return doc


def _scalar(raw):
    """Read an override value as the type it looks like, string as a fallback."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


# ----------------------------------------------------------------------------
# validation

def _need(doc, key, where):
    if key not in doc:
        raise ConfigError(f"{where}: missing '{key}'")
    return doc[key]


def _float(value, where):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: {value!r} is not a number")


def _zones(doc, columns):
    if doc is None:
        return None
    if ("bands" in doc) == ("from_column" in doc):
        raise ConfigError("zones: give either 'bands', to cut the zones on a "
                          "coordinate, or 'from_column', to read them from a "
                          "column that already holds them, but not both")

    if "from_column" in doc:
        spec = doc["from_column"]
        column = _need(spec, "column", "zones.from_column")
        mapping = spec.get("map") or {}
        if not isinstance(mapping, dict):
            raise ConfigError("zones.from_column.map: expected a mapping of "
                              "the values in the column to zone names")
        return Zones(column=column, mapping={str(k): str(v)
                                             for k, v in mapping.items()})

    axis = doc.get("axis", "position")
    if axis not in columns:
        raise ConfigError(f"zones.axis is '{axis}', which is not one of the "
                          f"columns you declared: {', '.join(sorted(columns))}")

    raw = _need(doc, "bands", "zones")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("zones.bands: expected a list of bands, lowest first")

    bands, last = [], None
    for i, b in enumerate(raw):
        name = _need(b, "name", f"zones.bands[{i}]")
        has_edge = "below" in b
        if i < len(raw) - 1 and not has_edge:
            raise ConfigError(f"zones.bands[{i}] '{name}': every band except "
                              f"the last needs 'below', the coordinate where "
                              f"it ends")
        if i == len(raw) - 1 and has_edge:
            raise ConfigError(f"zones.bands[{i}] '{name}': the last band is "
                              f"open at the top, so it must not give 'below'")
        edge = _float(b["below"], f"zones.bands[{i}].below") if has_edge else None
        if edge is not None and last is not None and edge <= last:
            raise ConfigError(f"zones.bands[{i}] '{name}': below={edge} is not "
                              f"above the previous band's {last}; bands are "
                              f"listed lowest first")
        last = edge if edge is not None else last
        bands.append(Band(name, edge, bool(b.get("discard", False))))

    if all(b.discard for b in bands):
        raise ConfigError("zones: every band is discarded, so nothing would be "
                          "classified")
    return Zones(axis=axis, bands=tuple(bands))


def _phases(raw, cuts, zones):
    if not isinstance(raw, list) or not raw:
        raise ConfigError("phases: expected a list of at least one phase")

    known = zones.names if zones else []
    out, seen = [], set()
    for i, p in enumerate(raw):
        name = _need(p, "name", f"phases[{i}]")
        if name in seen:
            raise ConfigError(f"phases: '{name}' is declared twice")
        seen.add(name)

        lower = upper = None
        if "between" in p:
            pair = p["between"]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ConfigError(f"phases[{i}] '{name}': 'between' takes two "
                                  f"cut names, low first")
            lower, upper = pair
        if "below" in p:
            upper = p["below"]
        if "above" in p:
            lower = p["above"]
        if lower is None and upper is None:
            raise ConfigError(f"phases[{i}] '{name}': give 'below', 'above' or "
                              f"'between' so the phase has a modulus range")

        for ref in (lower, upper):
            if ref is not None and ref not in cuts:
                raise ConfigError(
                    f"phases[{i}] '{name}' refers to the cut '{ref}', which is "
                    f"not in the cuts section. Declared cuts: "
                    f"{', '.join(cuts) or 'none'}")
        if lower is not None and upper is not None and cuts[lower] >= cuts[upper]:
            raise ConfigError(f"phases[{i}] '{name}': cut '{lower}' "
                              f"({cuts[lower]}) is not below cut '{upper}' "
                              f"({cuts[upper]}), so the phase is empty")

        zone = p.get("zone")
        if zone is not None:
            if zones is None:
                raise ConfigError(
                    f"phases[{i}] '{name}' names the zone '{zone}', but this "
                    f"config has no zones section. Either add one or drop the "
                    f"zone from the phase; a modulus-only classification is a "
                    f"legitimate thing to want.")
            if known and zone not in known:
                raise ConfigError(
                    f"phases[{i}] '{name}' names the zone '{zone}', which is "
                    f"not one of yours: {', '.join(known)}")
        out.append(Phase(name, lower, upper, zone, p.get("colour")))
    return tuple(out)


def coverage_warnings(cfg):
    """
    Report modulus intervals no phase claims, and phases no indent can reach.

    Not an error. Leaving an interval unclaimed is sometimes deliberate, and
    the tool should say what it will do rather than refuse. But an unclaimed
    interval is also how indents quietly end up unlabelled and drop out of the
    fractions, so it is said out loud.
    """
    warnings = []
    edges = sorted({v for v in cfg.cuts.values()})
    zones = cfg.zones.names if cfg.zones else [None]

    for zone in zones:
        applicable = [p for p in cfg.phases
                      if p.zone is None or p.zone == zone]
        intervals = [(-float("inf"), edges[0] if edges else float("inf"))]
        for i, e in enumerate(edges):
            hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
            intervals.append((e, hi))

        for lo, hi in intervals:
            covered = any(
                (p.lower is None or cfg.cuts[p.lower] <= lo) and
                (p.upper is None or cfg.cuts[p.upper] >= hi)
                for p in applicable)
            if not covered:
                where = f" in zone '{zone}'" if zone else ""
                warnings.append(
                    f"no phase covers modulus {_interval(lo, hi)}{where}; "
                    f"indents there will be left unclassified and will not "
                    f"count towards the fractions")

    for p in cfg.phases:
        shadowed = False
        for q in cfg.phases:
            if q is p:
                break
            same_zone = q.zone is None or q.zone == p.zone
            q_lo = -float("inf") if q.lower is None else cfg.cuts[q.lower]
            q_hi = float("inf") if q.upper is None else cfg.cuts[q.upper]
            p_lo = -float("inf") if p.lower is None else cfg.cuts[p.lower]
            p_hi = float("inf") if p.upper is None else cfg.cuts[p.upper]
            if same_zone and q_lo <= p_lo and q_hi >= p_hi:
                shadowed = True
        if shadowed:
            warnings.append(
                f"phase '{p.name}' is listed after a phase that already claims "
                f"its whole range, so it can never match; phases are matched "
                f"in order and the first match wins")
    return warnings


def _interval(lo, hi):
    left = "anything" if lo == -float("inf") else f"{lo:g}"
    right = "up" if hi == float("inf") else f"{hi:g}"
    return f"{left} to {right}"


def _macro(doc, base):
    if doc is None:
        return None
    if "measured" not in doc and "from_curves" not in doc:
        raise ConfigError("macro: give 'measured', a value with its "
                          "uncertainty, or 'from_curves' to compute one from "
                          "load-displacement data")

    measured = []
    for i, m in enumerate(doc.get("measured") or ()):
        measured.append(Measured(
            label=str(m.get("label", f"measurement {i + 1}")),
            value=_float(_need(m, "value", f"macro.measured[{i}]"),
                         f"macro.measured[{i}].value"),
            sd=_float(m["sd"], f"macro.measured[{i}].sd") if "sd" in m else None,
            n=int(m["n"]) if "n" in m else None))

    curves = None
    if "from_curves" in doc:
        c = doc["from_curves"]
        chord = _need(c, "chord", "macro.from_curves")
        if not isinstance(chord, (list, tuple)) or len(chord) != 2:
            raise ConfigError("macro.from_curves.chord: expected [low, high], "
                              "the strain interval the chord is measured over")
        columns = dict(_need(c, "columns", "macro.from_curves"))
        if "specimen" not in columns:
            raise ConfigError("macro.from_curves.columns.specimen is required; "
                              "the curves are grouped by it")

        geometry = dict(c.get("geometry") or {})
        reduced = {"strain", "stress"} <= set(columns)
        raw = {"displacement", "force"} <= set(columns)
        if reduced == raw:
            raise ConfigError(
                "macro.from_curves.columns: name either 'strain' and 'stress', "
                "for curves the testing software has already reduced, or "
                "'displacement' and 'force', for what the machine recorded, "
                "but not both and not neither")
        if raw:
            missing = [k for k in GEOMETRY if k not in geometry]
            if missing:
                raise ConfigError(
                    f"macro.from_curves.geometry needs {', '.join(missing)} to "
                    f"turn force and displacement into stress and strain. All "
                    f"three of {', '.join(GEOMETRY)} in millimetres.")
            geometry = {k: _float(geometry[k], f"macro.from_curves.geometry.{k}")
                        for k in GEOMETRY}
            if any(v <= 0 for v in geometry.values()):
                raise ConfigError("macro.from_curves.geometry: span, width and "
                                  "thickness must all be positive")

        read = dict(c.get("read") or {})
        unknown = sorted(set(read) - READ_OPTIONS)
        if unknown:
            raise ConfigError(f"macro.from_curves.read: {', '.join(unknown)} is "
                              f"not something this reads. Allowed: "
                              f"{', '.join(sorted(READ_OPTIONS))}")

        extra = c.get("extra_window")
        curves = Curves(
            path=(base / str(_need(c, "path", "macro.from_curves"))).resolve(),
            columns=columns,
            chord=(float(chord[0]), float(chord[1])),
            toe_window=_float(c["toe_window"], "macro.from_curves.toe_window")
            if "toe_window" in c else None,
            extra_window=(float(extra[0]), float(extra[1])) if extra else None,
            geometry=geometry,
            modulus_scale=_float(c["modulus_scale"],
                                 "macro.from_curves.modulus_scale")
            if "modulus_scale" in c else DEFAULT_MODULUS_SCALE,
            read=read)
    return Macro(tuple(measured), curves)


def _qc(doc):
    if doc is None:
        return None
    pa = doc.get("phase_angle") or {}
    return QC(flag_column=doc.get("flag_column"),
              columns=tuple(pa.get("columns", ())),
              limit=_float(pa["max"], "qc.phase_angle.max") if "max" in pa else None,
              first_limit=_float(pa["first_max"], "qc.phase_angle.first_max")
              if "first_max" in pa else None)


def _clustering(doc):
    if doc is None:
        return None
    features = _need(doc, "features", "clustering")
    if not isinstance(features, list) or len(features) < 2:
        raise ConfigError("clustering.features: expected a list of at least "
                          "two column names")
    return Clustering(features=tuple(str(f) for f in features),
                      k=int(doc.get("k", 2)),
                      seed=int(doc.get("seed", 0)),
                      n_init=int(doc.get("n_init", 10)),
                      merge_lowest=int(doc.get("merge_lowest", 0)))


def from_document(doc, base, source=None):
    """Validate a raw document. `base` is what relative paths are read against."""
    base = Path(base)

    m = _need(doc, "map", "config")
    columns = _need(m, "columns", "map")
    if "modulus" not in columns:
        raise ConfigError("map.columns: 'modulus' is required; it is the "
                          "property the phases are cut on")
    kind = m.get("kind", "single_depth")
    if kind not in {"single_depth", "depth_resolved"}:
        raise ConfigError(f"map.kind is '{kind}'; expected 'single_depth', one "
                          f"row per indent, or 'depth_resolved', properties at "
                          f"several depths")

    read = dict(m.get("read") or {})
    unknown = sorted(set(read) - READ_OPTIONS)
    if unknown:
        raise ConfigError(f"map.read: {', '.join(unknown)} is not something "
                          f"this reads. Allowed: {', '.join(sorted(READ_OPTIONS))}")

    spec = MapSpec(path=(base / str(_need(m, "path", "map"))).resolve(),
                   kind=kind, columns=dict(columns),
                   by_depth=dict(columns.get("by_depth") or {}),
                   read=read)
    if spec.by_depth and kind == "single_depth":
        raise ConfigError("map.columns.by_depth is set but map.kind is "
                          "'single_depth'. A single-depth map has no depth "
                          "axis; either drop by_depth or say "
                          "kind: depth_resolved")

    cuts_raw = _need(doc, "cuts", "config")
    if not isinstance(cuts_raw, dict) or not cuts_raw:
        raise ConfigError("cuts: expected at least one named modulus cut, for "
                          "example  cuts: {split: 15.0}")
    cuts = {str(k): _float(v, f"cuts.{k}") for k, v in cuts_raw.items()}

    # Zones are cut on one of the declared roles, so by_depth is not a
    # candidate axis even though it sits in the same section.
    zones = _zones(doc.get("zones"),
                   {k: v for k, v in spec.columns.items() if k != "by_depth"})
    phases = _phases(_need(doc, "phases", "config"), cuts, zones)

    sweep = {}
    for name, values in (doc.get("sweep") or {}).items():
        if name not in cuts:
            raise ConfigError(f"sweep: '{name}' is not a declared cut. "
                              f"Declared cuts: {', '.join(cuts)}")
        if not isinstance(values, list) or not values:
            raise ConfigError(f"sweep.{name}: expected a list of values to try")
        sweep[str(name)] = tuple(_float(v, f"sweep.{name}") for v in values)

    return Config(name=str(doc.get("name", source.stem if source else "map")),
                  map=spec, cuts=cuts, phases=phases, zones=zones, sweep=sweep,
                  macro=_macro(doc.get("macro"), base),
                  clustering=_clustering(doc.get("clustering")),
                  qc=_qc(doc.get("qc")), source=source)


def load(path, overrides=None):
    """Read, override, and validate. Paths inside are read relative to the config."""
    path = Path(path).resolve()
    doc = apply_overrides(_read_document(path), overrides)
    return from_document(doc, path.parent, source=path)
