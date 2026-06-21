"""Thermodynamics critic: the energy model and equilibrium chemistry."""
from __future__ import annotations

from typing import List

from ..base import CriticAgent, CritiqueContext, Finding, Severity


class ThermodynamicsCritic(CriticAgent):
    name = "Thermodynamics critic"
    aspect = "Thermodynamics & energy model"

    def review(self, context: CritiqueContext) -> List[Finding]:
        findings: List[Finding] = []
        backend = context.backend

        # Live evidence: heuristic energies are GC/AU-blind.
        au = backend.duplex_energy("AAAAAA", "UUUUUU")
        gc = backend.duplex_energy("GGGGGG", "CCCCCC")
        findings.append(self._finding(
            Severity.CRITICAL,
            "Affinity model is not nearest-neighbor thermodynamics",
            "Duplex stability is scored as a flat energy per matched pair with no "
            "GC-vs-AU distinction, no stacking context, and no loop penalties. "
            "Rankings produced this way are scientifically meaningless for design.",
            "primer2/thermo.py: HeuristicBackend.duplex_energy / _antiparallel_pairs",
            recommendation="Use Turner/SantaLucia RNA nearest-neighbor parameters "
                           "(ViennaRNA or NUPACK), or restrict the heuristic "
                           "backend to smoke tests only.",
            evidence=("6-bp all-AU vs all-GC duplex score identically: "
                      "AU=%.1f kcal/mol == GC=%.1f kcal/mol (real RNA NN params "
                      "differ by ~1 kcal/mol per step)." % (au, gc)),
        ))

        findings.append(self._finding(
            Severity.CRITICAL,
            "MFE free energy used as a two-state binding constant",
            "Both legacy Gibbs_*_NUPACK and NupackBackend convert a single MFE "
            "structure energy into Keq. MFE is the minimum-energy structure of the "
            "ensemble, not the association free energy; folding-dominated species "
            "are mis-read as binders.",
            "Gibbs_multi_NUPACK.py:16; primer2/thermo.py: NupackBackend.duplex_energy",
            recommendation="Use partition-function / complex free energies "
                           "(ΔG_bind) from NUPACK complex_analysis, not single MFE.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "NupackBackend counts probe-probe homodimers as assay signal",
            "bound_probe_conc sums every complex containing the probe with "
            "size >= 2, which includes probe-probe dimers. The GA can be rewarded "
            "for self-dimerizing probes that produce no target signal.",
            "primer2/thermo.py: NupackBackend.bound_probe_conc",
            recommendation="Sum only complexes containing the probe AND >=1 target "
                           "strand; exclude probe-probe complexes from the signal.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "Reduced 1:1 equilibrium omits major competing reactions",
            "The fallback equilibrium models only 1:1 probe-target duplexes plus a "
            "single hairpin per species: no probe-probe, target-target, or higher "
            "order complexes that materially deplete free probe in multiplex tubes.",
            "primer2/equilibrium.py; cpt_NUPACK.py:16-22",
            recommendation="Use full tube_analysis with a complex set that includes "
                           "dimers; deprecate the hand-rolled reduced model.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "Two-state hairpin from a single MFE overestimates folded fraction",
            "Self-structure Keq is derived from one MFE structure, ignoring the "
            "conformational ensemble of short RNAs and binding-site accessibility, "
            "which is central for structured miRNA precursors / isomiRs.",
            "primer2/thermo.py: self_energy; theprobes.py:25-34",
            recommendation="Use partition-function folding and weight binding by "
                           "unpaired probability at the intended register.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "Ambiguous /55.14 standard-state conversion",
            "Keq divides exp(-G/RT) by water molarity (55.14) with no documented "
            "standard state. If inconsistent with the backend that computes yields, "
            "a ~55x systematic error propagates into every hybridization yield.",
            "Keq.py:13; primer2/thermo.py: ThermoBackend.keq",
            recommendation="Derive K from the same backend used for yields, or "
                           "document/verify the standard state and unit-test vs "
                           "published oligo Tm/Keq.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "Non-physiological, non-configurable salt; temperature decoupled",
            "Salt is fixed at 1.0 M Na+ / 0 Mg2+ and legacy NUPACK hardcodes T=37 "
            "while Keq uses Temp=310 K, so changing temperature rescales Keq but "
            "not the underlying ΔG.",
            "Gibbs_*_NUPACK.py:13,16; primer2/thermo.py defaults",
            recommendation="Thread one temperature and assay-realistic salt through "
                           "all energy and equilibrium calls; expose on the CLI.",
        ))

        return findings
