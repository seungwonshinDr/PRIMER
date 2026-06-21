# PRIMER / DIPS

A research codebase that designs DNA/RNA probes for selective miRNA detection
using a genetic algorithm. Probes are scored by how selectively they hybridize
to a target cell's miRNA profile versus other cells, using thermodynamic
equilibrium (Gibbs free energy -> equilibrium constants -> hybridization
concentrations).

## Layout

- `Run_DIPS.py` - top-level driver (loads data, runs repeated GA searches).
- `theprobes.py` - the genetic-algorithm loop (`probes(...)`).
- `Gibbs_single_NUPACK.py` / `Gibbs_multi_NUPACK.py` - free-energy calculations
  via the proprietary `nupack` package (probe self-structure / probe-target).
- `Keq.py` - Gibbs free energy -> equilibrium constant.
- `cpt_NUPACK.py` - solves hybridization concentrations (scipy Newton solver).
- `crossover.py`, `pointmut.py`, `shiftmut.py`, `roulette.py` - GA operators.

## Cursor Cloud specific instructions

Python dependencies are installed into a virtualenv at `~/.venvs/primer`
(created/refreshed by the startup update script). Run things with
`~/.venvs/primer/bin/python`. The repo has no package config, so add the repo
root to the path when importing its modules from elsewhere, e.g.
`PYTHONPATH=/workspace ~/.venvs/primer/bin/python your_script.py`.

All `.py` modules are now Python-3 syntax clean (`python -m py_compile *.py`
passes for every file). There is no lint config; `py_compile` is the basic
syntax check. The installable core (GA operators + thermodynamics: `Keq`,
`cpt_NUPACK`, `crossover`, `pointmut`, `shiftmut`, `roulette`, `elite`) imports
and runs under Python 3 with only `numpy`/`scipy`.

Already fixed in this repo:

- Python 3 port: `Run_DIPS.py` and `theprobes.py` use `print()` calls.
- `elite.py` provides `elitism(nElite, population, fitness)` (top-nElite probes),
  which `theprobes.py` imports.
- `Run_DIPS.py` imports `theprobes` (previously the non-existent
  `theprobes_NUPACK`).

Remaining external blockers (NOT code issues - need licensed software / private
data) before `Run_DIPS.py` runs end-to-end against real inputs:

- Proprietary `nupack`. `Gibbs_single_NUPACK.py` / `Gibbs_multi_NUPACK.py` (and
  the drivers) `import nupack`, which is not on PyPI and needs a licensed
  download from https://nupack.org. The repo uses an old NUPACK 3.x-style
  `nupack.mfe(...)` API.
- Missing data files. `Run_DIPS.py` loads `miRsequences` and `miRconc_reduced`,
  which are not in the repo.

The GA engine itself has been verified end-to-end by running the real
`theprobes.probes()` loop with a mocked `nupack` and synthetic data; the
selectivity fitness improves across generations as expected.
