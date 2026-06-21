"""Concrete critic agents."""
from .thermodynamics import ThermodynamicsCritic
from .optimization import OptimizationCritic
from .statistics import StatisticsCritic
from .biology import BiologyCritic
from .software import SoftwareCritic

ALL_AGENTS = [
    ThermodynamicsCritic,
    OptimizationCritic,
    StatisticsCritic,
    BiologyCritic,
    SoftwareCritic,
]

__all__ = [c.__name__ for c in ALL_AGENTS] + ["ALL_AGENTS"]
