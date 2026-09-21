"""Trajectory parsers: one module per raw trajectory format.

Every parser module exposes ``parse_run(run_dir, output_dir)``, which converts the raw trajectories of
one run into unified-schema JSON files (``unified_<instance>.json``) and returns a list of log lines.
"""
import importlib


def load(parser_name):
    """Import and return the parser module of a trajectory format."""
    return importlib.import_module(f"parsers.{parser_name}")
