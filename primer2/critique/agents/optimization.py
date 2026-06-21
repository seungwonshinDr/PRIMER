"""Optimization critic: the genetic algorithm itself."""
from __future__ import annotations

from typing import List

import numpy as np

from ...design import design_probe
from ..base import CriticAgent, CritiqueContext, Finding, Severity


class OptimizationCritic(CriticAgent):
    name = "Optimization critic"
    aspect = "Optimization & genetic algorithm"

    def review(self, context: CritiqueContext) -> List[Finding]:
        findings: List[Finding] = []
        findings.append(self._search_dynamics(context))
        findings.append(self._roulette_flat_fitness())
        findings.append(self._point_mutation_waste(context))

        findings.append(self._finding(
            Severity.CRITICAL,
            "Elites are double-counted in the mating pool",
            "next_generation copies the top n_elite individuals AND draws the "
            "roulette mating pool from the full population (including those "
            "elites). The legacy roulette even takes an `elites` arg it never uses. "
            "Effective population size collapses and super-fit clones dominate.",
            "primer2/ga.py: next_generation; roulette.py: roulette (elites unused)",
            recommendation="Exclude elite indices from the roulette pool, or use a "
                           "(mu+lambda) scheme with duplicate suppression.",
        ))

        findings.append(self._finding(
            Severity.CRITICAL,
            "No real benchmark; the GA is never shown to beat random",
            "The only optimization test asserts best-so-far history is "
            "non-decreasing, which is true by construction (elitism). There is no "
            "comparison against random sequences, reverse-complement, or greedy "
            "search, so search-quality regressions are invisible.",
            "tests/test_design.py: test_selectivity_optimization_improves",
            recommendation="Add fixed-seed benchmarks vs random / best-of-N "
                           "baselines and report effect size on held-out samples.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "Tiny population + strong elitism + min-subtraction roulette",
            "pop_size=20 against a ~4^L (L=18 -> ~10^10) space, 10% elitism, and "
            "weights = fitness - min (which zeroes the worst individual) drive "
            "premature convergence; legacy Run_DIPS hides this behind 100 blind "
            "restarts instead of diversity maintenance.",
            "primer2/design.py defaults; primer2/ga.py: roulette",
            recommendation="Larger population, rank/tournament selection, fitness "
                           "sharing / crowding, and a per-base mutation rate ~1/L.",
        ))

        findings.append(self._finding(
            Severity.HIGH,
            "Destructive shift mutation at 70% with single-point crossover",
            "shift_prob=0.7 applies a scrambling structural move to most "
            "individuals each generation while recombination is single-point only "
            "(and the odd parent is never crossed), giving poor mixing of "
            "epistatic sequence motifs.",
            "primer2/ga.py: crossover, shift_mutation; design.py defaults",
            recommendation="Lower structural-move rate (~0.01-0.1), add "
                           "two-point/uniform crossover, and tune mutation per locus.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "No termination or convergence criterion",
            "Search runs a fixed number of generations with no stagnation "
            "detection, diversity floor, or evaluation budget, so it wastes compute "
            "after convergence or stops mid-climb.",
            "primer2/design.py: design_probe loop",
            recommendation="Stop on fitness stagnation / diversity threshold and "
                           "expose an evaluation budget.",
        ))

        findings.append(self._finding(
            Severity.MEDIUM,
            "Fixed probe length; no length / register search",
            "Probe length is a hard hyperparameter and selectivity is strongly "
            "length- and register-dependent, so the GA cannot move in the "
            "(length, sequence) space that isomiR discrimination needs.",
            "primer2/design.py: design_probe(probe_len)",
            recommendation="Add insert/delete operators or an outer length sweep "
                           "and an explicit binding-register gene.",
        ))

        return findings

    def _search_dynamics(self, context: CritiqueContext) -> Finding:
        pop_size = 20
        unique_counts: List[int] = []

        def hook(gen, population, fitness):
            unique_counts.append(len({tuple(row) for row in population}))

        design_probe(context.backend, context.dataset.sequences,
                     context.dataset.concentrations, target_index=0,
                     probe_len=16, pop_size=pop_size, n_gen=25,
                     rng=np.random.default_rng(context.seed), on_generation=hook)

        first = unique_counts[0] if unique_counts else 0
        last = unique_counts[-1] if unique_counts else 0
        if last <= max(2, first // 2):
            return self._finding(
                Severity.HIGH,
                "Population diversity collapses (premature convergence)",
                "Unique genotypes drop sharply, so the search converges on a few "
                "lucky genotypes instead of exploring the sequence space.",
                "primer2/ga.py: next_generation; primer2/design.py defaults",
                evidence="unique genotypes went %d -> %d over 25 generations "
                         "(pop_size=%d)." % (first, last, pop_size),
                recommendation="Add diversity maintenance and weaker selection "
                               "pressure.",
            )
        return self._finding(
            Severity.HIGH,
            "Operators over-disrupt: the search never consolidates gains",
            "With shift_prob=0.7 the population stays almost fully diverse for the "
            "whole run, so the GA churns like repeated random sampling rather than "
            "exploiting good genotypes -- the opposite failure mode of refinement.",
            "primer2/ga.py: shift_mutation; primer2/design.py defaults",
            evidence="unique genotypes stayed %d/%d after 25 generations (near-"
                     "maximal churn, little exploitation)." % (last, pop_size),
            recommendation="Lower the structural-move rate, add elitist refinement, "
                           "and balance exploration vs exploitation.",
        )

    def _roulette_flat_fitness(self) -> Finding:
        import warnings

        from roulette import roulette as legacy_roulette  # legacy module

        population = np.ones((4, 5))
        fitness = np.ones((4, 1))  # flat fitness -> min-subtracted weights sum to 0
        evidence = "no defect observed"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                legacy_roulette(np.zeros((1, 5)), population, fitness)
            except Exception as exc:  # ZeroDivisionError / IndexError etc.
                evidence = "legacy roulette raised %s on flat fitness." % type(exc).__name__
            else:
                divide = any("invalid value" in str(w.message)
                             or "divide" in str(w.message) for w in caught)
                if divide:
                    evidence = ("legacy roulette triggers a divide-by-zero "
                                "RuntimeWarning (NaN selection probabilities) on "
                                "flat fitness.")
        return self._finding(
            Severity.HIGH,
            "Legacy roulette is undefined when fitness is flat",
            "With equal fitness the min-subtracted weights sum to zero, producing "
            "NaN selection probabilities. The primer2 roulette guards this; the "
            "legacy path does not.",
            "roulette.py: roulette",
            evidence=evidence,
            recommendation="Fall back to uniform selection when total weight <= 0 "
                           "(as primer2/ga.py does).",
        )

    def _point_mutation_waste(self, context: CritiqueContext) -> Finding:
        from primer2 import ga

        rng = np.random.default_rng(context.seed)
        pop = np.full((200, 30), 1)  # all base 'A' (code 1)
        mutated = ga.point_mutation(pop, prob=1.0, rng=rng)
        still_same = float(np.mean(mutated == pop))
        return self._finding(
            Severity.MEDIUM,
            "Point mutation can resample the same base",
            "Mutation draws uniformly from all four bases including the current "
            "one, so ~1/4 of mutation events are no-ops, wasting expensive fitness "
            "evaluations.",
            "primer2/ga.py: point_mutation; pointmut.py",
            evidence="with prob=1.0 every base 'mutated' yet %.0f%% stayed "
                     "identical." % (still_same * 100),
            recommendation="Sample uniformly from the other three bases.",
        )
