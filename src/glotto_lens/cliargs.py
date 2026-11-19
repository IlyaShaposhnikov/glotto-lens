"""Argument types shared by the command lines."""

import argparse


def non_negative(text: str) -> int:
    """An integer >= 0. SQLite reads LIMIT -1 as "no limit", and a negative slice drops the
    last items, so a negative count is never what the user meant."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a non-negative integer")
    return value
