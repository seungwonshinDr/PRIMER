"""Biology critic: probe manufacturability and miRNA-specific constraints."""
from __future__ import annotations

import re
from typing import List

import numpy as np

from ...design import design_probe
from ..base import CriticAgent, CritiqueContext, Finding, Severity


def _gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def _longest_homopolymer(seq: str) -> int:
    return max((len(m.group(0)) for m in re.finditer(r"(.)\1*", seq)), default=0)


class BiologyCritic(CriticAgent):
    name = "Biology critic"
    aspect = "RNA biology & manufacturability"

    def review(self, context: CritiqueContext) -> List[Finding]:
        findings: List[Finding] = []
        findings.append(self._unconstrained_probe(context))

        findings.append(self._finding(
            Severity.CRITICAL,
            "No seed-region or binding-register constraint for isomiRs",
            "isomiRs are encoded only as length variants; the seed (~nt 2-8) that "
            "drives miRNA family specificity is not annotated or weighted, and the "
            "free-register heuristic lets a probe bind anywhere on the target "
            "without biologically meaningful alignment.",
            "primer2/data.py: make_demo_dataset; primer2/thermo.py: _antiparallel_pairs",
            recommendation="Annotate seed/non-seed regions, constrain the binding "
                           "register, and penalize seed matches to off-target "
                           "miRNA families.",
        ))

        findings.append(self._finding(
            Severity.CRITICAL,
            "Off-targets limited to the input list (no miRNome-wide scan)",
            "Selectivity is scored only against the loaded targets. A probe "
            "selective against ~20 miRNAs can still cross-hybridize thousands of "
            "seed-matched transcripts in a real sample.",
            "primer2/design.py: _signal_profile; primer2/data.py: load_dataset",
            recommendation="Add an optional transcriptome / miRBase seed-family "
                           "off-target scan and penalize strong off-target binding.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "RNA:RNA only despite a 'DNA/RNA probes' claim",
            "Every energy/equilibrium call uses material='rna'. Most practical "
            "miRNA detection probes are DNA or LNA-modified DNA against RNA, which "
            "use different nearest-neighbor parameters and Tm, so RNA:RNA energies "
            "mis-rank real probe candidates.",
            "AGENTS.md; primer2/thermo.py defaults; Gibbs_*_NUPACK.py",
            recommendation="Support material='dna' / chimeric models and select "
                           "chemistry per assay format.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "Probe concentration scales with total target load",
            "probe_conc is set to the mean total target concentration, but real "
            "assays use a fixed probe concentration independent of sample RNA load, "
            "which changes selective pressure and reported yields.",
            "primer2/design.py: design_probe; Run_DIPS.py:18",
            recommendation="Use a fixed, user-specified probe concentration and "
                           "report bound fraction at that concentration.",
        ))

        return findings

    def _unconstrained_probe(self, context: CritiqueContext) -> Finding:
        result = design_probe(context.backend, context.dataset.sequences,
                              context.dataset.concentrations, target_index=0,
                              probe_len=18, pop_size=20, n_gen=40,
                              rng=np.random.default_rng(context.seed))
        seq = result.sequence
        gc = _gc_fraction(seq)
        run = _longest_homopolymer(seq)
        bad = gc < 0.3 or gc > 0.7 or run >= 4
        sev = Severity.CRITICAL if bad else Severity.HIGH
        return self._finding(
            sev,
            "Designed probes have no GC%, Tm, or homopolymer constraints",
            "The GA can converge on biologically implausible probes (AU-only, "
            "poly-G, extreme Tm, long homopolymer runs) that are hard to "
            "synthesize, prone to secondary structure, and nonspecific.",
            "primer2/encoding.py: random_population; primer2/ga.py mutation operators",
            evidence="a designed 18-mer was %r with GC=%.0f%% and a %d-nt "
                     "homopolymer run." % (seq, gc * 100, run),
            recommendation="Add hard filters / penalties: GC 30-70%%, Tm within a "
                           "window of the assay temperature, and a max homopolymer "
                           "run; reject violating sequences before scoring.",
        )
