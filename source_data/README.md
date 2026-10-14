# Source data

These CSV files contain the numerical values underlying the manuscript figures
and tables. They are compact exports from the authors' existing local pickle
files; no values were recalculated during the cleanup.

| File | Contents |
| --- | --- |
| `figure4_gd_cgd.csv` | GD and CGD energy trajectories in Figure 4 |
| `figure5_landscapes.csv` | Complete parameter grids in Figure 5 |
| `figure5_slices.csv` | One-dimensional slices in Figure 5 |
| `figures6_8_results.csv` | Energies, fidelities, and circuit resources in Figures 6–8 |
| `selected_parameters.csv` | Selected DB-QITE and QCGD parameters |
| `table1_excess_energy_advantage.csv` | Unrounded values for Table I |
| `table2_final_fidelities.csv` | Unrounded values for Table II |
| `table3_matched_fidelity_resources.csv` | Unrounded values for Table III |

`manifest.json` records the canonical configuration and `SHA256SUMS` records
file checksums. The calculation notebook continues to write pickle files, as in
the original workflow; these CSV files are the portable source-data release.

