#!/usr/bin/env python3
"""PRIMER2 command-line runner (the modernized replacement for Run_DIPS.py).

Examples
--------
Run a self-contained demo (synthetic isomiR data, no NUPACK needed):

    python run_primer2.py --demo

Design a panel from real data with NUPACK (when installed):

    python run_primer2.py --sequences seqs.csv --concentrations conc.csv \
        --backend nupack --probe-len 18
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from primer2 import design_panel, get_backend
from primer2.data import load_dataset, make_demo_dataset


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PRIMER2 selective probe/panel design")
    p.add_argument("--demo", action="store_true",
                   help="run on a synthetic isomiR dataset")
    p.add_argument("--sequences", help="targets CSV (id,sequence)")
    p.add_argument("--concentrations", help="concentrations CSV (id + samples)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "nupack", "heuristic"])
    p.add_argument("--probe-len", type=int, default=18)
    p.add_argument("--pop-size", type=int, default=20)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--n-elite", type=int, default=2)
    p.add_argument("--temp", type=float, default=310.0, help="temperature (K)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.demo:
        dataset = make_demo_dataset(seed=args.seed)
    elif args.sequences and args.concentrations:
        dataset = load_dataset(args.sequences, args.concentrations)
    else:
        print("error: provide --demo or both --sequences and --concentrations",
              file=sys.stderr)
        return 2

    backend = get_backend(args.backend, celsius=args.temp - 273.15)
    print("PRIMER2  backend=%s  targets=%d  samples=%d  probe_len=%d" % (
        type(backend).__name__, len(dataset.sequences),
        len(dataset.samples), args.probe_len))
    print("-" * 64)

    def log(target_index, gen, best_fit, probe_conc):
        if gen % 10 == 0 or gen == args.generations - 1:
            pct = best_fit / probe_conc * 100 if probe_conc else 0.0
            print("  [%s] gen %3d  best-selectivity = %7.3f %%" % (
                dataset.samples[target_index], gen, pct))

    rng = np.random.default_rng(args.seed)
    panel = design_panel(
        backend, dataset.sequences, dataset.concentrations, args.probe_len,
        pop_size=args.pop_size, n_gen=args.generations, n_elite=args.n_elite,
        temp_k=args.temp, rng=rng, log=log)

    print("-" * 64)
    print("Designed panel:")
    for class_index, probe in panel.probes.items():
        print("  %-12s -> %s  (fitness=%.3e)" % (
            dataset.samples[class_index], probe.sequence, probe.fitness))

    print("-" * 64)
    print("Panel orthogonality (probe-probe duplex energy, kcal/mol;"
          " less negative = more orthogonal):")
    np.set_printoptions(precision=2, suppress=True)
    print(panel.orthogonality)
    print("worst cross-hybridization energy: %.2f kcal/mol"
          % panel.worst_cross_hybridization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
