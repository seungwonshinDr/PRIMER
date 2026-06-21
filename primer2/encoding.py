"""Sequence encoding helpers.

Bases are encoded as integers ``A=1, U=2, G=3, C=4`` to stay compatible with the
original DIPS code. Sequences are handled as RNA (``T`` is treated as ``U``).
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

BASES = "AUGC"
CODE = {b: i + 1 for i, b in enumerate(BASES)}
DECODE = {i + 1: b for i, b in enumerate(BASES)}
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def codes_to_seq(codes: Iterable[int]) -> str:
    return "".join(DECODE[int(c)] for c in codes)


def seq_to_codes(seq: str) -> List[int]:
    seq = seq.upper().replace("T", "U")
    return [CODE[b] for b in seq]


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def random_population(rng: np.random.Generator, pop_size: int, length: int) -> np.ndarray:
    """Random population of probe sequences as integer codes (1..4)."""
    return rng.integers(1, 5, size=(pop_size, length))
