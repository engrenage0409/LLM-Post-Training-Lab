"""Parameter validation helpers for DPO experiments."""

import math
from argparse import ArgumentTypeError


def positive_beta(value):
    """argparse type: DPO beta must be a finite positive float."""
    try:
        beta = float(value)
    except (TypeError, ValueError) as exc:
        raise ArgumentTypeError(f"Invalid DPO beta: {value!r}") from exc
    if not math.isfinite(beta) or beta <= 0:
        raise ArgumentTypeError(f"DPO beta must be finite and > 0, got {value!r}")
    return beta
