"""Three-step comparison of DB-QITE, QCGD, geodesic GD, and geodesic CGD.

Run directly with the Qrisp environment:

    python test_dbqite_vs_unitary_gd.py

The initial point is labelled ``k=0`` and the three updates are ``k=1,2,3``.
Every method performs its own grid search. The selected trajectories,
parameters, and complete energy scans are saved to one pickle dictionary.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import sympy as sp

from qrisp import QuantumVariable
from qrisp.algorithms.vqe.problems.heisenberg import create_heisenberg_init_function
from qrisp.operators import X, Y, Z

from qcgd import QCGD
from qite import QITE
from saveload import load_data, save_data
from unitary_descent import unitary_cgd, unitary_gd


N_QUBITS = 12
N_STEPS = 3
MU_VALUES = np.linspace(0.0, 0.5, 5)
GAMMA_VALUES = np.linspace(0.0, 0.5, 5)
DEFAULT_FILENAME = f"DBIvsGCI_L={N_QUBITS}"
DEFAULT_FOLDER = ["DBIvsGCI"]
METHODS = ("dbqite", "qcgd", "gd", "cgd")


def create_heisenberg_hamiltonian(n_qubits: int):
    """Return the open-boundary XXX Hamiltonian used by the project."""
    return sum(
        X(index) * X(index + 1)
        + Y(index) * Y(index + 1)
        + Z(index) * Z(index + 1)
        for index in range(n_qubits - 1)
    )


def create_singlet_initialisation(n_qubits: int):
    if n_qubits % 2:
        raise ValueError(
            "The tensor-product singlet initialisation needs an even number of qubits."
        )
    matching = [(index, index + 1) for index in range(0, n_qubits, 2)]
    return create_heisenberg_init_function(matching)


def get_system_statevector(qc, n_qubits: int, substitutions=None):
    """Extract and normalize the system register, following the project helpers."""
    bound_qc = qc.bind_parameters(substitutions) if substitutions else qc.copy()
    for _ in range(bound_qc.num_qubits() - n_qubits):
        bound_qc.qubits.insert(0, bound_qc.qubits.pop(-1))
    state = np.asarray(bound_qc.statevector_array()[: 2**n_qubits], dtype=complex)
    return state / np.linalg.norm(state)


def energy(state, hamiltonian_matrix) -> float:
    return float(np.real(np.vdot(state, hamiltonian_matrix @ state)))


def _scan_mu(qc, n_qubits, hamiltonian_matrix, mu, mu_values):
    return np.asarray(
        [
            energy(
                get_system_statevector(qc, n_qubits, {mu: float(mu_value)}),
                hamiltonian_matrix,
            )
            for mu_value in mu_values
        ]
    )


def _scan_gamma_mu(
    qc,
    n_qubits,
    hamiltonian_matrix,
    mu,
    gamma,
    mu_values,
    gamma_values,
):
    """Return a complete energy grid with axes ``(gamma, mu)``."""
    return np.asarray(
        [
            [
                energy(
                    get_system_statevector(
                        qc,
                        n_qubits,
                        {mu: float(mu_value), gamma: float(gamma_value)},
                    ),
                    hamiltonian_matrix,
                )
                for mu_value in mu_values
            ]
            for gamma_value in gamma_values
        ]
    )


def _select_gradient_step(
    method,
    n_qubits,
    hamiltonian,
    hamiltonian_matrix,
    initialisation,
    exp_H,
    previous_mus,
    step,
    mu_values,
):
    mu = sp.Symbol(f"{method}_mu_{step}")
    parameters = previous_mus + [mu]
    qv = QuantumVariable(n_qubits)

    if method == "dbqite":
        QITE(qv, initialisation, exp_H, parameters, step, method="GC")
    elif method == "gd":
        unitary_gd(qv, initialisation, hamiltonian, parameters, step)
    else:
        raise ValueError(f"Unknown gradient method: {method}")

    energy_scan = _scan_mu(
        qv.qs.compile(), n_qubits, hamiltonian_matrix, mu, mu_values
    )
    mu_index = int(np.argmin(energy_scan))
    return float(mu_values[mu_index]), float(energy_scan[mu_index]), energy_scan


def _select_conjugate_step(
    method,
    n_qubits,
    hamiltonian,
    hamiltonian_matrix,
    initialisation,
    exp_H,
    previous_mus,
    previous_gammas,
    step,
    mu_values,
    gamma_values,
):
    mu = sp.Symbol(f"{method}_mu_{step}")
    parameters = previous_mus + [mu]
    qv = QuantumVariable(n_qubits)

    # The first conjugate-gradient update is the ordinary gradient update and
    # has no gamma. Its full energy scan is therefore one-dimensional.
    if step == 1:
        if method == "qcgd":
            QCGD(qv, initialisation, exp_H, parameters, [], step, method="GC")
        elif method == "cgd":
            unitary_cgd(qv, initialisation, hamiltonian, parameters, [], step)
        else:
            raise ValueError(f"Unknown conjugate-gradient method: {method}")

        energy_scan = _scan_mu(
            qv.qs.compile(), n_qubits, hamiltonian_matrix, mu, mu_values
        )
        mu_index = int(np.argmin(energy_scan))
        return (
            float(mu_values[mu_index]),
            None,
            float(energy_scan[mu_index]),
            energy_scan,
            energy_scan,
        )

    gamma = sp.Symbol(f"{method}_gamma_{step}")
    gammas = previous_gammas + [gamma]
    if method == "qcgd":
        QCGD(qv, initialisation, exp_H, parameters, gammas, step, method="GC")
    elif method == "cgd":
        unitary_cgd(qv, initialisation, hamiltonian, parameters, gammas, step)
    else:
        raise ValueError(f"Unknown conjugate-gradient method: {method}")

    energy_grid = _scan_gamma_mu(
        qv.qs.compile(),
        n_qubits,
        hamiltonian_matrix,
        mu,
        gamma,
        mu_values,
        gamma_values,
    )
    gamma_index, mu_index = np.unravel_index(
        np.argmin(energy_grid), energy_grid.shape
    )
    energy_vs_mu = energy_grid[gamma_index].copy()
    return (
        float(mu_values[mu_index]),
        float(gamma_values[gamma_index]),
        float(energy_grid[gamma_index, mu_index]),
        energy_vs_mu,
        energy_grid,
    )


def run_three_step_comparison(
    *,
    n_qubits=N_QUBITS,
    mu_values=MU_VALUES,
    gamma_values=GAMMA_VALUES,
    filename=DEFAULT_FILENAME,
    folder=DEFAULT_FOLDER,
):
    """Run all four methods, save one results dictionary, and return it."""
    mu_values = np.asarray(mu_values, dtype=float)
    gamma_values = np.asarray(gamma_values, dtype=float)
    if mu_values.ndim != 1 or len(mu_values) == 0:
        raise ValueError("mu_values must be a non-empty one-dimensional sequence.")
    if gamma_values.ndim != 1 or len(gamma_values) == 0:
        raise ValueError("gamma_values must be a non-empty one-dimensional sequence.")

    hamiltonian = create_heisenberg_hamiltonian(n_qubits)
    hamiltonian_matrix = hamiltonian.to_sparse_matrix()
    initialisation = create_singlet_initialisation(n_qubits)

    def exp_H(qv, time):
        hamiltonian.trotterization(method="commuting")(qv, time, 5)

    initial_qv = QuantumVariable(n_qubits)
    initialisation(initial_qv)
    initial_state = get_system_statevector(initial_qv.qs.compile(), n_qubits)
    initial_energy = energy(initial_state, hamiltonian_matrix)

    energies = {method: [initial_energy] for method in METHODS}
    selected_mus = {method: [] for method in METHODS}
    selected_gammas = {"qcgd": [], "cgd": []}
    energy_vs_mu = {method: [] for method in METHODS}
    energy_grids = {"qcgd": [], "cgd": []}

    for step in range(1, N_STEPS + 1):
        for method in ("dbqite", "gd"):
            selected_mu, selected_energy, scan = _select_gradient_step(
                method,
                n_qubits,
                hamiltonian,
                hamiltonian_matrix,
                initialisation,
                exp_H,
                selected_mus[method],
                step,
                mu_values,
            )
            selected_mus[method].append(selected_mu)
            energies[method].append(selected_energy)
            energy_vs_mu[method].append(scan)

        for method in ("qcgd", "cgd"):
            print("STARTING", method, "step", step)
            selected_mu, selected_gamma, selected_energy, scan, grid = (
                _select_conjugate_step(
                    method,
                    n_qubits,
                    hamiltonian,
                    hamiltonian_matrix,
                    initialisation,
                    exp_H,
                    selected_mus[method],
                    selected_gammas[method],
                    step,
                    mu_values,
                    gamma_values,
                )
            )
            selected_mus[method].append(selected_mu)
            if selected_gamma is not None:
                selected_gammas[method].append(selected_gamma)
            energies[method].append(selected_energy)
            energy_vs_mu[method].append(scan)
            energy_grids[method].append(grid)

    results = {
        "n_qubits": n_qubits,
        "steps": np.arange(N_STEPS + 1),
        "mu_values": mu_values,
        "gamma_values": gamma_values,
        "energies": {
            method: np.asarray(values) for method, values in energies.items()
        },
        "selected_mus": {
            method: np.asarray(values) for method, values in selected_mus.items()
        },
        # Entry zero corresponds to update 2 because update 1 has no gamma.
        "selected_gammas": {
            method: np.asarray(values)
            for method, values in selected_gammas.items()
        },
        # Each entry is directly plottable against mu_values. For CGD/QCGD,
        # updates 2 and 3 use the slice at that update's selected gamma.
        "energy_vs_mu": energy_vs_mu,
        # Complete CGD/QCGD searches. Update 1 is 1D; updates 2 and 3 have
        # shape (len(gamma_values), len(mu_values)).
        "energy_grids": energy_grids,
    }
    save_data(results, filename, folder=list(folder))
    return results


class TestFourMethodComparison(unittest.TestCase):
    def test_three_steps_save_all_energies_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            results = run_three_step_comparison(
                n_qubits=2,
                mu_values=[0.0, 0.1],
                gamma_values=[0.0, 0.1],
                filename="three_step_comparison_test",
                folder=[temporary_folder],
            )
            saved = load_data(
                "three_step_comparison_test", folder=[temporary_folder]
            )

        self.assertEqual(len(results["steps"]), N_STEPS + 1)
        self.assertEqual(set(saved["energies"]), set(METHODS))
        for method in METHODS:
            self.assertEqual(len(saved["energies"][method]), N_STEPS + 1)
            self.assertEqual(len(saved["selected_mus"][method]), N_STEPS)
            self.assertEqual(len(saved["energy_vs_mu"][method]), N_STEPS)
            self.assertTrue(np.all(np.isfinite(saved["energies"][method])))
            self.assertTrue(np.all(np.diff(saved["energies"][method]) <= 1e-8))
            for scan in saved["energy_vs_mu"][method]:
                self.assertEqual(scan.shape, (2,))

        for method in ("qcgd", "cgd"):
            self.assertEqual(len(saved["selected_gammas"][method]), N_STEPS - 1)
            self.assertEqual(saved["energy_grids"][method][0].shape, (2,))
            self.assertEqual(saved["energy_grids"][method][1].shape, (2, 2))
            self.assertEqual(saved["energy_grids"][method][2].shape, (2, 2))


if __name__ == "__main__":
    comparison = run_three_step_comparison()
    print("Energies:", comparison["energies"])
    print("Selected mus:", comparison["selected_mus"])
    print("Selected gammas:", comparison["selected_gammas"])
    print(f"Saved to {DEFAULT_FOLDER[0]}/{DEFAULT_FILENAME}.pickle")
