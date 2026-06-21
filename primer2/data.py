"""Input loading and synthetic-data generation.

Real inputs come from small-RNA-seq quantification (e.g. miRBase-annotated
isomiRs). Two CSV files are expected:

* sequences CSV with columns ``id,sequence`` (one row per target; sequences may
  have different lengths to represent isomiRs).
* concentrations CSV with a leading ``id`` column followed by one column per
  sample; values are molar concentrations.

:func:`make_demo_dataset` synthesizes a small isomiR-style dataset so the
pipeline can be exercised without private data or NUPACK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .encoding import codes_to_seq


@dataclass
class Dataset:
    ids: List[str]
    sequences: List[str]
    concentrations: np.ndarray  # (n_targets, n_samples)
    samples: List[str]


def load_dataset(sequences_csv: str, concentrations_csv: str) -> Dataset:
    seq_df = pd.read_csv(sequences_csv)
    conc_df = pd.read_csv(concentrations_csv)

    seq_df = seq_df.set_index("id")
    conc_df = conc_df.set_index("id")
    conc_df = conc_df.reindex(seq_df.index)

    ids = list(seq_df.index.astype(str))
    sequences = [str(s).upper().replace("T", "U") for s in seq_df["sequence"]]
    samples = list(conc_df.columns.astype(str))
    concentrations = conc_df.to_numpy(dtype=float)
    return Dataset(ids=ids, sequences=sequences,
                   concentrations=concentrations, samples=samples)


def make_demo_dataset(n_base: int = 3, n_samples: int = 3, base_len: int = 18,
                      seed: int = 0) -> Dataset:
    """Synthetic isomiR-style dataset.

    Each base miRNA gets 5'/3' isomiR variants of differing lengths, and each
    sample over-expresses one base miRNA family to create a discriminable
    signature.
    """
    rng = np.random.default_rng(seed)
    ids: List[str] = []
    sequences: List[str] = []
    family: List[int] = []

    for b in range(n_base):
        core = codes_to_seq(rng.integers(1, 5, size=base_len))
        variants = {
            "canonical": core,
            "iso5p_trim": core[1:],          # 5' shortened isomiR
            "iso3p_add": core + codes_to_seq(rng.integers(1, 5, size=2)),  # 3' extended
        }
        for tag, seq in variants.items():
            ids.append("miR-%d.%s" % (b, tag))
            sequences.append(seq)
            family.append(b)

    family_arr = np.array(family)
    concentrations = np.zeros((len(sequences), n_samples))
    for s in range(n_samples):
        enriched = s % n_base
        base_level = rng.uniform(0.5, 1.5, size=len(sequences)) * 1e-6
        base_level[family_arr == enriched] *= 6.0  # family-specific enrichment
        concentrations[:, s] = base_level

    samples = ["sample_%d" % s for s in range(n_samples)]
    return Dataset(ids=ids, sequences=sequences,
                   concentrations=concentrations, samples=samples)
