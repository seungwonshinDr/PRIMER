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

The installable core (GA operators + thermodynamics: `Keq`, `cpt_NUPACK`,
`crossover`, `pointmut`, `shiftmut`, `roulette`) imports and runs under Python
3 with only `numpy`/`scipy`. There is no lint config; `python -m py_compile *.py`
is the basic syntax check.

Known blockers (require source/data changes or proprietary software, NOT just
environment setup) that stop `Run_DIPS.py` / `theprobes.py` from running as-is:

- Python 2 syntax. `Run_DIPS.py` and `theprobes.py` use `print` statements, so
  they do not parse under Python 3 (the only interpreter available). The other
  8 modules are Python-3 syntax clean.
- Proprietary `nupack`. `Gibbs_single_NUPACK.py` / `Gibbs_multi_NUPACK.py`
  (and the drivers) `import nupack`, which is not on PyPI and needs a licensed
  download from https://nupack.org. The repo also uses an old NUPACK 3.x-style
  `nupack.mfe(...)` API.
- Missing module `elite`. `theprobes.py` does `import elite` and calls
  `elite.elitism(...)`, but there is no `elite.py` in the repo or git history.
- Module-name mismatch. `Run_DIPS.py` does `import theprobes_NUPACK`, but the
  file is named `theprobes.py`.
- Missing data files. `Run_DIPS.py` loads `miRsequences` and `miRconc_reduced`,
  which are not in the repo.

Because of the above, the full `Run_DIPS.py` pipeline cannot be executed in this
environment without modifying source code and supplying proprietary `nupack`
plus the private miRNA data files.
