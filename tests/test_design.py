import numpy as np

from primer2 import HeuristicBackend, design_panel, design_probe
from primer2.data import make_demo_dataset
from primer2.encoding import codes_to_seq, reverse_complement, seq_to_codes


def test_encoding_roundtrip():
    codes = [1, 2, 3, 4, 1, 1, 2]
    assert seq_to_codes(codes_to_seq(codes)) == codes
    assert reverse_complement("AUGC") == "GCAU"


def test_variable_length_isomir_targets_supported():
    backend = HeuristicBackend()
    targets = ["AUGCAUGCAU", "UGCAUGCA", "AUGCAUGCAUGG"]  # different lengths
    conc = np.array([[2e-6, 1e-6], [1e-6, 2e-6], [1e-6, 1e-6]])
    result = design_probe(backend, targets, conc, target_index=0, probe_len=10,
                          pop_size=8, n_gen=5, rng=np.random.default_rng(0))
    assert len(result.sequence) == 10
    assert np.isfinite(result.fitness)


def test_panel_design_returns_probe_per_class():
    backend = HeuristicBackend()
    dataset = make_demo_dataset(seed=1)
    panel = design_panel(backend, dataset.sequences, dataset.concentrations,
                         probe_len=14, pop_size=8, n_gen=6,
                         rng=np.random.default_rng(1))
    assert set(panel.probes) == set(range(dataset.concentrations.shape[1]))
    n = dataset.concentrations.shape[1]
    assert panel.orthogonality.shape == (n, n)


def test_selectivity_optimization_improves():
    backend = HeuristicBackend()
    dataset = make_demo_dataset(seed=2)
    result = design_probe(backend, dataset.sequences, dataset.concentrations,
                          target_index=0, probe_len=16, pop_size=16, n_gen=25,
                          rng=np.random.default_rng(2))
    # Best-so-far fitness is monotonic non-decreasing across generations.
    assert result.history[-1] >= result.history[0]
