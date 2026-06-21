"""Thermodynamics backends.

The genetic algorithm only needs two things from thermodynamics:

* ``bound_probe_conc`` - equilibrium concentration of probe captured into
  probe-target duplexes within one sample (the assay signal).
* ``duplex_energy`` - free energy of a probe-probe duplex, used to score panel
  orthogonality (cross-hybridization).

:class:`NupackBackend` implements both with the NUPACK 4 Python API (test-tube
ensemble analysis). :class:`HeuristicBackend` provides NUPACK-free approximations
so the engine and its tests run anywhere; its energies are a simple
complementarity heuristic, NOT calibrated thermodynamics, and must not be used
for production probe selection.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

from .encoding import COMPLEMENT
from .equilibrium import bound_probe_conc as _solve_bound_probe

R_KCAL = 0.001987  # gas constant, kcal / (mol * K)
WATER_MOLAR = 55.14  # standard-state conversion used by the original DIPS code


class ThermoBackend(ABC):
    def __init__(self, material: str = "rna", celsius: float = 37.0,
                 sodium: float = 1.0, magnesium: float = 0.0):
        self.material = material
        self.celsius = celsius
        self.sodium = sodium
        self.magnesium = magnesium

    def keq(self, gibbs: float, temp_k: float) -> float:
        return math.exp(-gibbs / (R_KCAL * temp_k)) / WATER_MOLAR

    @abstractmethod
    def duplex_energy(self, seq_a: str, seq_b: str) -> float:
        ...

    @abstractmethod
    def self_energy(self, seq: str) -> float:
        ...

    @abstractmethod
    def bound_probe_conc(self, probe: str, targets: Sequence[str],
                         target_conc: Sequence[float], probe_conc: float,
                         temp_k: float) -> float:
        ...


def _antiparallel_pairs(a: str, b: str) -> int:
    """Max complementary base pairs over all ungapped antiparallel registers."""
    br = b[::-1]
    la, lb = len(a), len(br)
    best = 0
    for offset in range(-(lb - 1), la):
        matches = 0
        for i in range(la):
            j = i - offset
            if 0 <= j < lb and COMPLEMENT.get(a[i]) == br[j]:
                matches += 1
        if matches > best:
            best = matches
    return best


class HeuristicBackend(ThermoBackend):
    """NUPACK-free approximation for testing / CI (not real thermodynamics)."""

    _DUPLEX_PER_PAIR = -2.0
    _SELF_PER_PAIR = -0.6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._duplex_cache: Dict[Tuple[str, str], float] = {}
        self._self_cache: Dict[str, float] = {}

    def duplex_energy(self, seq_a: str, seq_b: str) -> float:
        key = (seq_a, seq_b)
        cached = self._duplex_cache.get(key)
        if cached is None:
            cached = self._DUPLEX_PER_PAIR * _antiparallel_pairs(seq_a, seq_b)
            self._duplex_cache[key] = cached
        return cached

    def self_energy(self, seq: str) -> float:
        cached = self._self_cache.get(seq)
        if cached is None:
            # Intramolecular hairpin proxy: half the antiparallel self-overlap,
            # leaving a minimal loop, weighted lower than an intermolecular duplex.
            stem = _antiparallel_pairs(seq, seq) // 2
            cached = self._SELF_PER_PAIR * stem
            self._self_cache[seq] = cached
        return cached

    def bound_probe_conc(self, probe, targets, target_conc, probe_conc, temp_k):
        K = [self.keq(self.duplex_energy(probe, t), temp_k) for t in targets]
        khp_probe = self.keq(self.self_energy(probe), temp_k)
        khp_target = [self.keq(self.self_energy(t), temp_k) for t in targets]
        return _solve_bound_probe(K, khp_probe, khp_target, target_conc, probe_conc)


class NupackBackend(ThermoBackend):
    """NUPACK 4 backend: real thermodynamics via test-tube ensemble analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import nupack  # licensed package, see https://docs.nupack.org

        self._nupack = nupack
        self._model = nupack.Model(
            material=self.material,
            celsius=self.celsius,
            sodium=self.sodium,
            magnesium=self.magnesium,
        )

    def duplex_energy(self, seq_a: str, seq_b: str) -> float:
        nupack = self._nupack
        strands = [nupack.Strand(seq_a, name="a"), nupack.Strand(seq_b, name="b")]
        result = nupack.mfe(strands=strands, model=self._model)
        return float(result[0].energy)

    def self_energy(self, seq: str) -> float:
        nupack = self._nupack
        result = nupack.mfe(strands=[nupack.Strand(seq, name="a")], model=self._model)
        return float(result[0].energy)

    def bound_probe_conc(self, probe, targets, target_conc, probe_conc, temp_k):
        nupack = self._nupack
        probe_strand = nupack.Strand(probe, name="P")
        strands = {probe_strand: max(float(probe_conc), 0.0)}
        for i, (seq, conc) in enumerate(zip(targets, target_conc)):
            strands[nupack.Strand(seq, name="T%d" % i)] = max(float(conc), 0.0)

        tube = nupack.Tube(strands=strands,
                           complexes=nupack.SetSpec(max_size=2), name="assay")
        result = nupack.tube_analysis(tubes=[tube], model=self._model)
        concentrations = result.tubes[tube].complex_concentrations

        bound = 0.0
        for complex_, conc in concentrations.items():
            members = list(complex_.strands)
            if probe_strand in members and len(members) >= 2:
                bound += float(conc)
        return bound


def get_backend(name: str = "auto", **kwargs) -> ThermoBackend:
    """Return a thermodynamics backend.

    ``name='auto'`` uses NUPACK when importable, otherwise the heuristic backend.
    """
    if name in ("nupack", "auto"):
        try:
            return NupackBackend(**kwargs)
        except Exception as exc:
            if name == "nupack":
                raise RuntimeError(
                    "NUPACK 4 backend requested but unavailable (%s). Install the "
                    "licensed package from https://docs.nupack.org/start/ or use "
                    "--backend heuristic." % exc
                ) from exc
    return HeuristicBackend(**kwargs)
