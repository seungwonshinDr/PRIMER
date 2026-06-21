"""PRIMER2: modernized probe-design pipeline.

A refresh of the original DIPS genetic-algorithm pipeline that:

1. Targets the NUPACK 4 Python API (via a pluggable thermodynamics backend).
2. Computes hybridization yields from a multi-strand test-tube equilibrium
   instead of a hand-rolled, unstable Newton solver.
3. Designs multi-probe panels and supports variable-length isomiR targets.

The thermodynamics are isolated behind :class:`primer2.thermo.ThermoBackend`.
When the licensed ``nupack`` package is installed, :class:`NupackBackend` runs
real NUPACK 4 test-tube analysis. Otherwise a dependency-free
:class:`HeuristicBackend` provides approximate energies so the engine (and its
tests) can run without NUPACK.
"""
from .thermo import ThermoBackend, HeuristicBackend, NupackBackend, get_backend
from .design import design_probe, design_panel, PanelResult, ProbeResult

__all__ = [
    "ThermoBackend",
    "HeuristicBackend",
    "NupackBackend",
    "get_backend",
    "design_probe",
    "design_panel",
    "PanelResult",
    "ProbeResult",
]
