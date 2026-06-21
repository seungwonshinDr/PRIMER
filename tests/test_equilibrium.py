import numpy as np

from primer2.equilibrium import solve_free_probe, bound_probe_conc


def _mass_balance_residuals(K, khp_probe, khp_target, T0, P0):
    K = np.asarray(K, float)
    khp_target = np.asarray(khp_target, float)
    T0 = np.asarray(T0, float)
    p = solve_free_probe(K, khp_probe, khp_target, T0, P0)
    t = T0 / (1.0 + K * p + khp_target)
    probe_total = p * (1.0 + np.sum(K * t) + khp_probe)
    target_total = t * (1.0 + K * p + khp_target)
    return probe_total - P0, target_total - T0


def test_mass_balance_is_conserved():
    K = [5e5, 2e6, 8e4]
    khp_target = [0.3, 1.2, 0.05]
    T0 = [2e-6, 3e-6, 1e-6]
    P0 = 2e-6
    dprobe, dtargets = _mass_balance_residuals(K, 0.4, khp_target, T0, P0)
    assert abs(dprobe) < 1e-18
    assert np.all(np.abs(dtargets) < 1e-18)


def test_free_probe_within_bounds():
    p = solve_free_probe([1e6, 1e6], 0.5, [0.2, 0.2], [1e-6, 1e-6], 2e-6)
    assert 0.0 <= p <= 2e-6


def test_no_binding_returns_full_probe():
    # All constants zero -> nothing binds, free probe == total probe.
    assert solve_free_probe([0.0, 0.0], 0.0, [0.0, 0.0], [1e-6, 1e-6], 2e-6) == 2e-6
    assert bound_probe_conc([0.0], 0.0, [0.0], [1e-6], 2e-6) == 0.0


def test_stronger_binding_captures_more_probe():
    weak = bound_probe_conc([1e4], 0.0, [0.0], [5e-6], 2e-6)
    strong = bound_probe_conc([1e8], 0.0, [0.0], [5e-6], 2e-6)
    assert strong > weak
