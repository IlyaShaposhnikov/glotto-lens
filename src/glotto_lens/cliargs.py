"""Argument types shared by the command lines."""

import argparse


def non_negative(text: str) -> int:
    """An integer >= 0. SQLite reads LIMIT -1 as "no limit", and a negative slice drops the
    last items, so a negative count is never what the user meant."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value
