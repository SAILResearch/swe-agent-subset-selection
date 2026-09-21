"""Per-dataset sanitization rules: the term lists and patterns fixed after the one-time manual review
of the Phase 1 (corpus discovery) and Phase 2 (outcome leakage) reports of each dataset."""
import importlib


def load(rules_name):
    """Import and return the rules module of a dataset."""
    return importlib.import_module(f"sanitization_rules.{rules_name}")
