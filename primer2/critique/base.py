"""Core types for the critique framework."""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..data import Dataset, make_demo_dataset
from ..thermo import ThermoBackend, HeuristicBackend


class Severity(enum.IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def label(self) -> str:
        return self.name


@dataclass
class Finding:
    """A single criticism from one critic agent."""

    aspect: str
    severity: Severity
    title: str
    problem: str
    recommendation: str
    location: str = ""
    evidence: Optional[str] = None

    def to_markdown(self) -> str:
        lines = [f"- **[{self.severity.label()}] {self.title}**"]
        if self.location:
            lines.append(f"  - *Where:* `{self.location}`")
        lines.append(f"  - *Problem:* {self.problem}")
        if self.evidence:
            lines.append(f"  - *Evidence:* {self.evidence}")
        lines.append(f"  - *Fix:* {self.recommendation}")
        return "\n".join(lines)


@dataclass
class CritiqueContext:
    """Shared resources critic agents may use to substantiate findings.

    ``dataset`` and ``backend`` let agents run live empirical probes against the
    real algorithm. Agents must treat the context as read-only.
    """

    backend: ThermoBackend
    dataset: Dataset
    seed: int = 0

    @classmethod
    def default(cls, seed: int = 0) -> "CritiqueContext":
        return cls(backend=HeuristicBackend(),
                   dataset=make_demo_dataset(seed=seed), seed=seed)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


class CriticAgent(ABC):
    """Base class for an adversarial critic agent."""

    #: Human-readable name of the agent.
    name: str = "critic"
    #: The aspect/lens this agent reviews from.
    aspect: str = "general"

    @abstractmethod
    def review(self, context: CritiqueContext) -> List[Finding]:
        ...

    def _finding(self, severity: Severity, title: str, problem: str,
                 location: str = "", recommendation: str = "",
                 evidence: Optional[str] = None) -> Finding:
        return Finding(aspect=self.aspect, severity=severity, title=title,
                       problem=problem, recommendation=recommendation,
                       location=location, evidence=evidence)
