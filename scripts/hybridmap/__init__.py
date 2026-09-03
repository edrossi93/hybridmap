"""
hybridmap: measured-input rule of mixtures from an indentation map.

The chain is classify, aggregate, bound, compare against a macroscopic
measurement. Every choice that makes it one laboratory's analysis instead of a
method, where the specimen ends, where the phases divide, what the phases are
called, which column holds what, is read from a config file.

The reproduction boundary is the exported per-indent table. This package reads
a CSV and never an instrument file.

    from hybridmap import config, io, classify, aggregate, mixtures
    cfg = config.load("examples/paper_shallow.yaml")
    frame = io.read_map(cfg.map)
    cols = io.resolve(frame, cfg.map)
    labelled = classify.apply(frame, cfg, cols)
    table = aggregate.class_table(labelled, cfg.phase_names, cols["modulus"])
    voigt, hill, reuss = mixtures.bounds(table.fraction, table["median"])

`plots` is deliberately not imported here: it selects a matplotlib backend, and
a module that does that as a side effect of an unrelated import is a nuisance.
"""

# plots is left out of this list deliberately: importing it selects a
# matplotlib backend, and doing that as a side effect of an unrelated import
# is a nuisance to anyone who wanted only the arithmetic.
from . import aggregate, classify, cluster, config, io, macro, mixtures
from .config import Config, ConfigError, load

__all__ = ["aggregate", "classify", "cluster", "config", "io", "macro",
           "mixtures", "Config", "ConfigError", "load"]
