"""Software & reproducibility critic."""
from __future__ import annotations

import inspect
from typing import List

from ...thermo import HeuristicBackend, NupackBackend
from ..base import CriticAgent, CritiqueContext, Finding, Severity


class SoftwareCritic(CriticAgent):
    name = "Software critic"
    aspect = "Software & reproducibility"

    def review(self, context: CritiqueContext) -> List[Finding]:
        findings: List[Finding] = []
        findings.append(self._unused_temp_k())

        findings.append(self._finding(
            Severity.HIGH,
            "Heuristic backend results are easily mistaken for real thermodynamics",
            "get_backend('auto') silently falls back to the uncalibrated "
            "HeuristicBackend when NUPACK is absent, so a run can produce "
            "scientifically meaningless probes with no loud warning in the output.",
            "primer2/thermo.py: get_backend; run_primer2.py",
            recommendation="Emit a prominent warning / tag outputs when the "
                           "heuristic backend is used, and forbid it for production "
                           "design via an explicit opt-in flag.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "No input validation or QC on sequences and concentrations",
            "load_dataset casts values directly with no checks for alphabet, "
            "negative/zero concentrations, missing values, or below-detection "
            "targets, so malformed inputs propagate silently into the model.",
            "primer2/data.py: load_dataset",
            recommendation="Validate the sequence alphabet, reject/flag negative or "
                           "missing concentrations, and document a zeros/LOD policy.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "Legacy reproducibility is broken; seeding is inconsistent",
            "Run_DIPS seeds with np.random.seed(None) and legacy operators use "
            "global numpy RNG plus multiprocessing, so legacy runs are not "
            "reproducible, while primer2 uses an injectable Generator only.",
            "Run_DIPS.py:6; theprobes.py; primer2 (Generator)",
            recommendation="Thread a single np.random.Generator through all code "
                           "paths and seed worker processes explicitly.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "No provenance recorded with designed probes",
            "Outputs do not record the backend, NUPACK model/version, salt, "
            "temperature, or RNG seed, so a designed probe cannot be reproduced or "
            "audited later.",
            "primer2/design.py: ProbeResult / PanelResult; run_primer2.py",
            recommendation="Attach a provenance record (backend, model params, "
                           "seed, dataset hash) to results.",
        ))

        return findings

    def _unused_temp_k(self) -> Finding:
        sig = inspect.signature(NupackBackend.bound_probe_conc)
        has_temp = "temp_k" in sig.parameters
        source = inspect.getsource(NupackBackend.bound_probe_conc)
        # temp_k appears only in the signature line, never used in the body.
        used_in_body = source.count("temp_k") > 1
        evidence = ("NupackBackend.bound_probe_conc accepts temp_k but never uses "
                    "it (temperature is fixed at Model construction)."
                    if has_temp and not used_in_body else "signature changed")
        return self._finding(
            Severity.LOW,
            "Per-call temperature is silently ignored by the NUPACK backend",
            "bound_probe_conc takes a temp_k argument suggesting per-call "
            "temperature sweeps, but the NUPACK Model fixes temperature at "
            "construction, so the parameter is a silent no-op.",
            "primer2/thermo.py: NupackBackend.bound_probe_conc",
            evidence=evidence,
            recommendation="Rebuild the Model when temp_k changes, or drop the "
                           "parameter to avoid a misleading API.",
        )
