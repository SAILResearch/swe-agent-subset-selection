"""Evaluation protocols: how populations, temporal splits and seeds are laid out for a dataset.

``synthetic_distributions``  multi_model and multi_agent: 1,000 synthetic distributions of one instance pool
``run_groups``               single_setup: six run groups, each with its own instance pool and runs
"""
import importlib


def load(protocol_name):
    """Import and return a protocol module."""
    return importlib.import_module(f"protocols.{protocol_name}")
