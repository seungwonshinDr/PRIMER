"""Statistics / experimental-design critic: the objective and data handling."""
from __future__ import annotations

from typing import List

import numpy as np

from ...design import _signal_profile
from ...encoding import codes_to_seq
from ..base import CriticAgent, CritiqueContext, Finding, Severity


class StatisticsCritic(CriticAgent):
    name = "Statistics critic"
    aspect = "Statistics & experimental design"

    def review(self, context: CritiqueContext) -> List[Finding]:
        findings: List[Finding] = []
        findings.append(self._winner_flip(context))

        findings.append(self._finding(
            Severity.CRITICAL,
            "Objective is in-sample with no validation / cross-validation",
            "Fitness is evaluated on the same concentration matrix used for "
            "selection across pop_size x n_gen (x100 restarts in legacy). This is "
            "in-sample optimization: the winner fits this matrix, not new "
            "patients/batches. The 100 restarts inflate the winner's curse.",
            "primer2/design.py: design_probe; Run_DIPS.py:23-37",
            recommendation="Leave-one-sample-out / k-fold CV over sample columns; "
                           "select the final probe by held-out selectivity.",
        ))

        findings.append(self._finding(
            Severity.CRITICAL,
            "Massive implicit multiple testing with no null calibration",
            "Thousands of candidate sequences are screened and the best fitness is "
            "reported as an extreme order statistic with no empirical null, no "
            "permutation control, and no multiplicity correction, so the reported "
            "'% selectivity' carries no significance.",
            "primer2/design.py; theprobes.py:146; Run_DIPS.py",
            recommendation="Build a null from random/label-shuffled sequences and "
                           "report the GA result against its 95th/99th percentile.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "No expression-noise model, replicates, or uncertainty propagation",
            "Concentrations are treated as exact point values, but sRNA-seq data "
            "are noisy, compositional, and often below detection. Probe ranks can "
            "change under plausible 2x error, giving false precision.",
            "primer2/data.py: load_dataset / make_demo_dataset",
            recommendation="Model concentrations as distributions and optimize "
                           "E[margin] or P(margin > tau) via Monte Carlo.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "Panel optimizes probes independently; crosstalk only checked post hoc",
            "design_panel runs an independent one-vs-rest GA per class and computes "
            "orthogonality afterwards. Real multiplex readout is a joint inverse "
            "problem; probe-probe binding and shared-target competition are invisible "
            "during optimization.",
            "primer2/design.py: design_panel",
            recommendation="Optimize a joint panel objective in a multi-probe tube "
                           "(e.g. maximize the minimum diagonal margin of the "
                           "probe x sample response, penalize off-diagonal leakage).",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "Demo data are separable by construction",
            "make_demo_dataset enriches one family 6x per sample, so any selective "
            "objective trivially succeeds; passing tests on it is weak evidence of "
            "method quality.",
            "primer2/data.py: make_demo_dataset",
            recommendation="Add benchmark suites with controlled overlap, "
                           "near-duplicate isomiRs, and known non-separable cases.",
        ))

        return findings

    def _winner_flip(self, context: CritiqueContext) -> Finding:
        backend = context.backend
        ds = context.dataset
        rng = np.random.default_rng(context.seed)
        conc = ds.concentrations
        probe_len = 16
        target_index = 0

        candidates = [codes_to_seq(rng.integers(1, 5, size=probe_len))
                      for _ in range(24)]

        def margin_for(matrix):
            best_i, best_m = -1, -np.inf
            for i, seq in enumerate(candidates):
                signal = _signal_profile(backend, seq, ds.sequences, matrix,
                                         float(matrix.sum() / matrix.shape[1]), 310.0)
                rest = np.delete(signal, target_index)
                m = signal[target_index] - rest.max()
                if m > best_m:
                    best_i, best_m = i, m
            return best_i

        base_winner = margin_for(conc)
        trials = 16
        flips = sum(margin_for(conc * rng.lognormal(0.0, 0.35, size=conc.shape))
                    != base_winner for _ in range(trials))

        sev = Severity.CRITICAL if flips >= trials // 2 else (
            Severity.HIGH if flips else Severity.MEDIUM)
        return self._finding(
            sev,
            "max-of-rest fitness only beats the single worst competitor",
            "fitness = signal[target] - max(signal[rest]) is a hard max over "
            "nuisance samples: it is blind to the rest of the off-target "
            "distribution (second-worst, mean, tail mass) and its non-smooth "
            "argmax can flip the selected probe under concentration noise.",
            "primer2/design.py: design_probe; theprobes.py:103-106",
            evidence="under +/-35%% log-normal concentration noise the best of 24 "
                     "candidate probes changed in %d of %d trials." % (flips, trials),
            recommendation="Use a distribution-aware margin (log-sum-exp or a high "
                           "quantile of the off-target signal) and report stability.",
        )
