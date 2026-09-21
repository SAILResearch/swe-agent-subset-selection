"""Loading of the selection result files (``<family>_w<W>.json``, ``determinism_w<W>.{json,npz}``).

Methods are named ``<family>:<method>``. Families are read in alphabetical order, which fixes the order of the
methods wherever ties are broken by position.
"""
import json
import re
from pathlib import Path

import numpy as np

from selection.families import FAMILIES

FAMILY_PREFIXES = tuple(f"{family}:" for family in FAMILIES)
STORED_FIELDS = ("dist_means", "dist_maxerr", "dist_maxerr_mean", "dist_p95")


def family_of(name):
    """Family of a method name, "unknown" without a known prefix."""
    return name.split(":", 1)[0] if name.startswith(FAMILY_PREFIXES) else "unknown"


def local_name(name):
    """Method name without its family prefix."""
    return name.split(":", 1)[1] if name.startswith(FAMILY_PREFIXES) else name


def single_file(folder, stem, extension):
    """The file ``<stem>[_d<first>-<last>]<extension>`` of a folder, or None."""
    pattern = re.compile(re.escape(stem) + r"(_d\d+-\d+)?" + re.escape(extension) + "$")
    matches = sorted(p for p in Path(folder).iterdir() if pattern.match(p.name))
    if len(matches) > 1:
        raise SystemExit(f"More than one file for {stem}{extension} in {folder}: {[m.name for m in matches]}")
    return matches[0] if matches else None


def available_windows(folder):
    """Window sizes with at least one family result file."""
    found = set()
    for path in Path(folder).iterdir():
        match = re.match(r"(.+)_w(\d+)(_d\d+-\d+)?\.json$", path.name)
        if match and match.group(1) in FAMILIES:
            found.add(int(match.group(2)))
    return sorted(found)


def load_window(folder, window, with_splits=True):
    """All family result files of one window as one table.

    Returns {"window", "methods", "n_distributions", "positions", "levels", "sizes", "splits", "run_names", "families",
    "results": {subset key: {"<family>:<method>": {dist_means, dist_maxerr, dist_maxerr_mean, dist_p95}}}}; with
    ``with_splits`` every method also holds ``split_errors``.
    """
    table = None
    for family in sorted(FAMILIES):
        path = single_file(folder, f"{family}_w{window}", ".json")
        if path is None:
            continue
        stored = json.loads(path.read_text())
        if table is None:
            table = {"window": window, "methods": [], "n_distributions": len(stored["distribution_positions"]),
                     "positions": stored["distribution_positions"], "levels": stored["distribution_levels"],
                     "sizes": stored["distribution_sizes"], "splits": stored["splits"], "run_names": stored["run_names"],
                     "families": {}, "results": {key: {} for key in stored["results"]}}
        elif stored["distribution_positions"] != table["positions"] or stored["splits"] != table["splits"]:
            raise SystemExit(f"{path.name} covers other distributions or splits than the other families of window {window}")
        table["families"][family] = list(stored["methods"])
        for method in stored["methods"]:
            table["methods"].append(f"{family}:{method}")
            for key in table["results"]:
                entry = stored["results"][key][method]
                kept = {field: np.asarray(entry[field], dtype=float) for field in STORED_FIELDS}
                if with_splits:
                    kept["split_errors"] = entry["split_errors"]
                table["results"][key][f"{family}:{method}"] = kept
    if table is None:
        raise SystemExit(f"No family result files of window {window} in {folder}")
    return table


def load_draws(folder, window):
    """Per-seed baseline draws of a window: {baseline: {subset key: RMSE[distributions, seeds],
    "<subset key>__maxerr": MaxErr[distributions, seeds]}} (float32, as stored), or None."""
    path = single_file(folder, f"determinism_w{window}", ".npz")
    if path is None:
        return None
    draws = {}
    with np.load(path) as stored:
        for name in stored.files:
            parts = name.replace("__maxerr", "").split("__")
            if len(parts) == 2:
                draws.setdefault(parts[1], {})[parts[0] + ("__maxerr" if name.endswith("__maxerr") else "")] = stored[name]
    return draws


def load_comparisons(folder, window):
    """Comparisons of the determinism stage for one window, or None."""
    path = single_file(folder, f"determinism_w{window}", ".json")
    return json.loads(path.read_text()) if path else None
