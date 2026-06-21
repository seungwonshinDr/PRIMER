"""Hybridization equilibrium.

Replaces the original ``cpt_NUPACK.Conc`` Newton solver (whose own comment noted
instability and initial-value sensitivity) with a robustly bracketed root find.

Model: a dilute test tube containing one probe species and ``N`` target species.
Each can fold into an inactive hairpin, and probe+target form a 1:1 duplex.

  total probe   P0   = p (1 + sum_j K_j t_j + Khp_p)
  total target  T0_j = t_j (1 + K_j p + Khp_t_j)

Substituting ``t_j = T0_j / (1 + K_j p + Khp_t_j)`` reduces the system to a
single monotonic equation in the free-probe concentration ``p`` on ``[0, P0]``,
which we solve with :func:`scipy.optimize.brentq` (guaranteed to converge given
the sign change at the bracket ends).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def solve_free_probe(K, Khp_probe, Khp_target, target_conc, probe_conc) -> float:
    K = np.asarray(K, dtype=float)
    Khp_target = np.asarray(Khp_target, dtype=float)
    T0 = np.asarray(target_conc, dtype=float)
    P0 = float(probe_conc)

    if P0 <= 0.0:
        return 0.0

    def residual(p):
        t = T0 / (1.0 + K * p + Khp_target)
        return p * (1.0 + np.sum(K * t) + float(Khp_probe)) - P0

    f_hi = residual(P0)
    if f_hi <= 0.0:
        # Only possible when all binding/hairpin constants are ~0; root is at P0.
        return P0

    return brentq(residual, 0.0, P0, xtol=1e-30, rtol=8.9e-16, maxiter=200)


def bound_probe_conc(K, Khp_probe, Khp_target, target_conc, probe_conc) -> float:
    """Concentration of probe captured in probe-target duplexes (the signal)."""
    p = solve_free_probe(K, Khp_probe, Khp_target, target_conc, probe_conc)
    p_hairpin = p * float(Khp_probe)
    return max(float(probe_conc) - p - p_hairpin, 0.0)
