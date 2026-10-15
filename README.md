# QCGD

Code and source data for the QCGD manuscript.

## Repository contents

- `algorithm_data.ipynb` runs the simulations and saves results as pickle files.
- `data_figures.ipynb` loads those pickle files and recreates the figures and
  tables.
- `qcgd.py`, `qite.py`, `run_qcgd.py`, and `run_qite.py` contain the algorithm
  and execution routines used by the notebooks.
- `saveload.py` contains the `save_data` and `load_data` pickle helpers.
- `unitary_descent.py` and `plot_comparison_energies.py` support the comparison
  calculations and plots.
- `source_data/` contains the CSV values underlying the manuscript figures and
  tables.
- `DBIvsGCI/` contains the small saved comparison data used for Figure 4.

## Installation

The code was checked with Python 3.11. Create an isolated environment and
install the recorded dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the calculations

Open `algorithm_data.ipynb` from this directory and run the cells in order. 
The manuscript calculation uses:

- 2 Trotter steps;
- 3 QCGD/DB-QITE iterations;
- 20 equally spaced parameter values on the interval `[0, 0.5]`.

The notebook saves results below `alpha1beta1/` and saves the spectrum of th Heisenberg Hamiltonian 
as `spectra.pickle`. These files can be large, so they are not committed. Keeping
the notebook and repository root together is important because `saveload.py`
uses paths relative to its own location.

After generating the pickle files, open `data_figures.ipynb` and run its cells
in order. It writes plots to `plots/`. LaTeX is used for plot text when a local
LaTeX executable is available and Matplotlib's built-in text renderer is used
otherwise.

The extended `[0, 1.5]` scans retained in the plotting notebook are only used
for the extended-landscape comparison in Figure 5. The principal convergence
runs use `[0, 0.5]`.

## Source data

The shareable numerical values are provided as CSV files in `source_data/`.
They were exported from the existing pickle result files; see
`source_data/README.md` and `source_data/manifest.json`. The original pickle
workflow is unchanged for new calculations.

## Tests

From the repository root, run:

```bash
python -m unittest discover -v
```

The tests cover the unitary-descent implementation and the DB-QITE/unitary
gradient-descent comparison.

## Licence

This code is distributed under the terms in `LICENSE` and `NOTICE`:
EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0.

