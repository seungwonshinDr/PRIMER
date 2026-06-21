"""High-level probe and panel design.

* :func:`design_probe` evolves a probe that is *selective* for one target
  sample/class: it maximizes the probe captured in that sample minus the best
  capture across the other samples (one-vs-rest selectivity).
* :func:`design_panel` designs one probe per class and reports the panel's
  cross-hybridization (orthogonality) matrix, addressing the move from single
  markers to multi-probe panels.

Targets are arbitrary-length RNA sequences (canonical miRNAs and/or isomiRs);
``target_conc`` is a ``(n_targets, n_samples)`` expression matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ga
from .encoding import codes_to_seq, random_population
from .thermo import ThermoBackend


@dataclass
class ProbeResult:
    target_index: int
    sequence: str
    fitness: float
    history: List[float] = field(default_factory=list)


@dataclass
class PanelResult:
    probes: Dict[int, ProbeResult]
    orthogonality: np.ndarray  # pairwise probe-probe duplex energy (kcal/mol)

    @property
    def worst_cross_hybridization(self) -> float:
        n = self.orthogonality.shape[0]
        if n < 2:
            return 0.0
        off_diag = self.orthogonality[~np.eye(n, dtype=bool)]
        return float(off_diag.min())  # most negative = strongest cross-hyb


def _signal_profile(backend: ThermoBackend, probe: str, targets: Sequence[str],
                    target_conc: np.ndarray, probe_conc: float,
                    temp_k: float) -> np.ndarray:
    n_samples = target_conc.shape[1]
    return np.array([
        backend.bound_probe_conc(probe, targets, target_conc[:, s], probe_conc, temp_k)
        for s in range(n_samples)
    ])


def design_probe(backend: ThermoBackend, targets: Sequence[str],
                 target_conc: np.ndarray, target_index: int, probe_len: int, *,
                 pop_size: int = 20, n_elite: int = 2, n_gen: int = 50,
                 point_prob: float = 0.1, shift_prob: float = 0.7,
                 temp_k: float = 310.0, probe_conc: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None,
                 log: Optional[callable] = None) -> ProbeResult:
    target_conc = np.asarray(target_conc, dtype=float)
    rng = rng or np.random.default_rng()
    if probe_conc is None:
        probe_conc = float(target_conc.sum() / target_conc.shape[1])

    population = random_population(rng, pop_size, probe_len)
    best_seq: Optional[str] = None
    best_fit = -np.inf
    history: List[float] = []

    for gen in range(n_gen):
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            seq = codes_to_seq(population[i])
            signal = _signal_profile(backend, seq, targets, target_conc,
                                     probe_conc, temp_k)
            rest = np.delete(signal, target_index)
            fitness[i] = signal[target_index] - (rest.max() if rest.size else 0.0)

        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fit:
            best_fit = float(fitness[gen_best_idx])
            best_seq = codes_to_seq(population[gen_best_idx])
        history.append(best_fit)
        if log is not None:
            log(target_index, gen, best_fit, probe_conc)

        population = ga.next_generation(population, fitness, n_elite,
                                        point_prob, shift_prob, rng)

    return ProbeResult(target_index=target_index, sequence=best_seq,
                       fitness=best_fit, history=history)


def design_panel(backend: ThermoBackend, targets: Sequence[str],
                 target_conc: np.ndarray, probe_len: int, *,
                 classes: Optional[Sequence[int]] = None,
                 **kwargs) -> PanelResult:
    target_conc = np.asarray(target_conc, dtype=float)
    if classes is None:
        classes = list(range(target_conc.shape[1]))

    probes: Dict[int, ProbeResult] = {}
    for class_index in classes:
        probes[class_index] = design_probe(backend, targets, target_conc,
                                           class_index, probe_len, **kwargs)

    seqs = [probes[c].sequence for c in classes]
    n = len(seqs)
    orthogonality = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            orthogonality[i, j] = backend.duplex_energy(seqs[i], seqs[j])

    return PanelResult(probes=probes, orthogonality=orthogonality)
