"""Adversarial multi-agent critique of the PRIMER/DIPS probe-design algorithm.

A panel of specialized critic agents reviews the algorithm from independent
angles (thermodynamics & biology, optimization / genetic algorithm, statistics
& experimental design, software & reproducibility). Each agent emits structured
:class:`Finding` objects; wherever feasible a finding is backed by a *live*
empirical probe of the real code (not just an assertion), so the criticism is
evidence-based. :class:`CritiquePanel` aggregates findings and renders a report.
"""
from .base import CriticAgent, CritiqueContext, Finding, Severity
from .panel import CritiquePanel, CritiqueReport, default_panel

__all__ = [
    "CriticAgent",
    "CritiqueContext",
    "Finding",
    "Severity",
    "CritiquePanel",
    "CritiqueReport",
    "default_panel",
]
