"""Genetic-algorithm operators for fixed-length probe populations.

Probes in a population share a fixed design length, so the operators work on an
integer matrix of shape ``(pop_size, probe_len)`` with base codes 1..4. Targets
may have arbitrary (isomiR) lengths; that variability is handled entirely by the
thermodynamics backend, not here.

This is a cleaned-up, NumPy-Generator-based reimplementation of the original
``elite`` / ``roulette`` / ``crossover`` / ``pointmut`` / ``shiftmut`` modules.
"""
from __future__ import annotations

import numpy as np


def elitism(population: np.ndarray, fitness: np.ndarray, n_elite: int) -> np.ndarray:
    order = np.argsort(fitness)[::-1]
    return population[order[:n_elite]].copy()


def roulette(population: np.ndarray, fitness: np.ndarray, n_select: int,
             rng: np.random.Generator) -> np.ndarray:
    weights = fitness - fitness.min()
    total = weights.sum()
    if total <= 0:
        idx = rng.integers(0, population.shape[0], size=n_select)
        return population[idx].copy()
    probabilities = weights / total
    idx = rng.choice(population.shape[0], size=n_select, p=probabilities)
    return population[idx].copy()


def crossover(parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, length = parents.shape
    children = parents.copy()
    for i in range(0, n - 1, 2):
        point = int(rng.integers(0, length))
        children[i, point:], children[i + 1, point:] = (
            parents[i + 1, point:], parents[i, point:])
    return children


def point_mutation(population: np.ndarray, prob: float,
                   rng: np.random.Generator) -> np.ndarray:
    pop = population.copy()
    mask = rng.random(pop.shape) < prob
    pop[mask] = rng.integers(1, 5, size=int(mask.sum()))
    return pop


def shift_mutation(population: np.ndarray, prob: float,
                   rng: np.random.Generator) -> np.ndarray:
    pop = population.copy()
    n, length = pop.shape
    for i in range(n):
        if rng.random() > prob:
            continue
        point = int(rng.integers(0, length))
        if rng.integers(0, 2) == 0:
            pop[i, :point] = pop[i, 1:point + 1] if point > 0 else pop[i, :point]
            pop[i, point] = rng.integers(1, 5)
        else:
            if point < length - 1:
                pop[i, point + 1:] = pop[i, point:-1]
            pop[i, point] = rng.integers(1, 5)
    return pop


def next_generation(population: np.ndarray, fitness: np.ndarray, n_elite: int,
                    point_prob: float, shift_prob: float,
                    rng: np.random.Generator) -> np.ndarray:
    pop_size = population.shape[0]
    elites = elitism(population, fitness, n_elite)
    parents = roulette(population, fitness, pop_size - n_elite, rng)
    children = crossover(parents, rng)
    children = point_mutation(children, point_prob, rng)
    children = shift_mutation(children, shift_prob, rng)
    return np.concatenate([elites, children], axis=0)
