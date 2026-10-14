"""Exact geodesic GD and CGD updates for small-system Qrisp simulations.

The routines in this module intentionally complement, rather than replace,
``QITE`` and ``QCGD``.  They construct the *actual* Riemannian geodesic

    U_{j+1} = exp(-mu_j W_j) U_j,

where ``R_j = [H, |psi_j><psi_j|]`` and either ``W_j = R_j`` (GD) or
``W_j = R_j + gamma_j W_{j-1}`` (CGD).  The state-dependent generators are
dense, so this implementation is an exact classical reference for small
systems.  It is not an efficient gate decomposition for quantum hardware.

The final step may contain SymPy parameters.  Qrisp keeps them as abstract
parameters in a custom operation; bind them before simulation, just as for
the symbolic circuits produced by ``QITE`` and ``QCGD``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from scipy.linalg import expm

from qrisp import Operation, QuantumArray


DEFAULT_MAX_QUBITS = 12


def _is_numeric(value) -> bool:
    if isinstance(value, Number):
        return True
    return isinstance(value, sp.Expr) and not value.free_symbols


def _as_real(value, name: str) -> float:
    if not _is_numeric(value):
        raise ValueError(f"{name} must be numeric at this recursion level; got {value!r}.")
    result = complex(value)
    if abs(result.imag) > 1e-12:
        raise ValueError(f"{name} must be real; got {value!r}.")
    return float(result.real)


def _qubits(qarg):
    if isinstance(qarg, QuantumArray):
        return sum((qv.reg for qv in qarg.flatten()), [])
    return list(qarg.reg)


def _hamiltonian_matrix(H, dimension: int) -> np.ndarray:
    if hasattr(H, "to_sparse_matrix"):
        matrix = H.to_sparse_matrix()
    elif hasattr(H, "to_array"):
        matrix = H.to_array()
    else:
        matrix = H

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"H has shape {matrix.shape}, but the quantum argument requires "
            f"a {(dimension, dimension)} matrix."
        )
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
        raise ValueError("H must be Hermitian.")
    return matrix


def _initial_state(qarg, U_0: Callable) -> np.ndarray:
    """Evaluate U_0|0> in an isolated session matching ``qarg``."""
    probe = qarg.duplicate()
    U_0(probe)
    qc = probe.qs.compile()
    state = np.asarray(qc.statevector_array(), dtype=complex).reshape(-1)
    state /= np.linalg.norm(state)
    return state


def _projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def _gradient(H: np.ndarray, state: np.ndarray) -> np.ndarray:
    P = _projector(state)
    R = H @ P - P @ H
    # Remove round-off in the anti-Hermitian structure.
    return 0.5 * (R - R.conj().T)


class GeodesicEvolution(Operation):
    """Qrisp operation representing ``exp(-mu * (R + gamma W))``.

    The matrix is evaluated only after all parameters have been bound.  This
    avoids constructing enormous symbolic matrix exponentials while retaining
    Qrisp's normal ``abstract_params``/``bind_parameters`` workflow.
    """

    def __init__(
        self,
        gradient: np.ndarray,
        mu,
        previous_direction: np.ndarray | None = None,
        gamma=None,
        name: str = "geodesic",
    ):
        gradient = np.asarray(gradient, dtype=complex)
        dimension = gradient.shape[0]
        n_qubits = int(np.log2(dimension))
        if gradient.shape != (dimension, dimension) or 2**n_qubits != dimension:
            raise ValueError("The geodesic generator must be a square 2**n matrix.")

        if previous_direction is None:
            params = [mu]
        else:
            previous_direction = np.asarray(previous_direction, dtype=complex)
            if previous_direction.shape != gradient.shape:
                raise ValueError("The current and previous directions must have equal shape.")
            params = [mu, gamma]

        super().__init__(name=name, num_qubits=n_qubits, params=params)
        self.gradient = gradient
        self.previous_direction = previous_direction
        self.is_qfree = False

    @property
    def mu(self):
        return self.params[0]

    @property
    def gamma(self):
        return None if self.previous_direction is None else self.params[1]

    def direction(self, substitutions: dict | None = None) -> np.ndarray:
        gamma = self.gamma
        if substitutions and isinstance(gamma, sp.Expr):
            gamma = gamma.subs(substitutions)
        if self.previous_direction is None:
            W = self.gradient
        else:
            W = self.gradient + _as_real(gamma, "gamma") * self.previous_direction
        return 0.5 * (W - W.conj().T)

    def matrix(self, substitutions: dict | None = None) -> np.ndarray:
        mu = self.mu
        if substitutions and isinstance(mu, sp.Expr):
            mu = mu.subs(substitutions)
        return expm(-_as_real(mu, "mu") * self.direction(substitutions))

    def get_unitary(self, decimals=-1):
        # Qrisp's statevector simulator uses complex64 internally. Returning
        # SciPy's default complex128 result works on its dense path, but fails
        # in the sparse contraction path used by larger systems because Numba
        # requires both dot-product operands to have the same dtype.
        result = np.asarray(self.matrix(), dtype=np.complex64)
        return result if decimals == -1 else np.round(result, decimals)

    def bind_parameters(self, substitutions):
        params = []
        for value in self.params:
            if isinstance(value, sp.Expr):
                value = value.subs(substitutions)
            params.append(value)

        result = GeodesicEvolution(
            self.gradient,
            params[0],
            previous_direction=self.previous_direction,
            gamma=None if self.previous_direction is None else params[1],
            name=self.name,
        )
        return result

    def inverse(self):
        return GeodesicEvolution(
            self.gradient,
            -self.mu,
            previous_direction=self.previous_direction,
            gamma=self.gamma,
            name=self.name + "_dg" if not self.name.endswith("_dg") else self.name[:-3],
        )


@dataclass(frozen=True)
class GeodesicData:
    """Classical data used to construct a geodesic circuit."""

    states: tuple[np.ndarray, ...]
    gradients: tuple[np.ndarray, ...]
    directions: tuple[np.ndarray, ...]


def _validate_inputs(qarg, M: Sequence, Gamma: Sequence | None, k: int, max_qubits: int):
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")
    n_qubits = len(_qubits(qarg))
    if n_qubits > max_qubits:
        raise ValueError(
            f"Exact geodesic construction is dense and was limited to {max_qubits} "
            f"qubits, but qarg contains {n_qubits}. Use QITE/QCGD for scalable "
            "circuit approximations, or raise max_qubits explicitly for a small "
            "classical benchmark."
        )
    if len(M) < k:
        raise ValueError(f"Step {k} needs {k} mu values; received {len(M)}.")
    if Gamma is not None and len(Gamma) < max(0, k - 1):
        raise ValueError(f"CGD step {k} needs {k - 1} gamma values; received {len(Gamma)}.")
    return n_qubits


def _build_data(qarg, U_0, H, M, Gamma, k: int, conjugate_gradient: bool) -> GeodesicData:
    dimension = 2 ** len(_qubits(qarg))
    H_matrix = _hamiltonian_matrix(H, dimension)
    state = _initial_state(qarg, U_0)
    states = [state]
    gradients = []
    directions = []

    for step in range(k):
        R = _gradient(H_matrix, state)
        if conjugate_gradient and step:
            gamma = Gamma[step - 1]
            if _is_numeric(gamma):
                W = R + _as_real(gamma, f"Gamma[{step - 1}]") * directions[-1]
                W = 0.5 * (W - W.conj().T)
            elif step == k - 1:
                # Keep the final search direction queryable as a symbolic
                # object array.  It is not exponentiated until gamma is bound.
                W = R.astype(object) + gamma * directions[-1].astype(object)
            else:
                raise ValueError(
                    f"Gamma[{step - 1}] must be numeric at this recursion level; "
                    f"got {gamma!r}."
                )
        else:
            W = R
        gradients.append(R)
        directions.append(W)

        # Only completed steps are needed to construct the next projector.  The
        # last mu may therefore remain symbolic for parameter scanning.
        if step < k - 1:
            mu = _as_real(M[step], f"M[{step}]")
            state = expm(-mu * W) @ state
            state /= np.linalg.norm(state)
            states.append(state)

    return GeodesicData(tuple(states), tuple(gradients), tuple(directions))


def _append_steps(qarg, U_0, M, Gamma, data: GeodesicData, k: int, cgd: bool):
    if k == 0:
        U_0(qarg)
        return

    _append_steps(qarg, U_0, M, Gamma, data, k - 1, cgd)
    step = k - 1
    if cgd and step:
        op = GeodesicEvolution(
            data.gradients[step],
            M[step],
            previous_direction=data.directions[step - 1],
            gamma=Gamma[step - 1],
            name=f"cgd_geodesic_{step}",
        )
    else:
        op = GeodesicEvolution(
            data.gradients[step], M[step], name=f"gd_geodesic_{step}"
        )
    qarg.qs.append(op, _qubits(qarg))


def unitary_gd(qarg, U_0, H, M, k, *, max_qubits=DEFAULT_MAX_QUBITS):
    """Apply ``k`` exact geodesic gradient-descent updates to ``qarg``.

    Previous entries of ``M`` must be numeric; ``M[k-1]`` may be a SymPy
    symbol and can subsequently be bound on the compiled Qrisp circuit.
    The function returns :class:`GeodesicData` for inspecting the recursively
    constructed states, gradients, and search directions.
    """
    _validate_inputs(qarg, M, None, k, max_qubits)
    data = _build_data(qarg, U_0, H, M, (), k, conjugate_gradient=False)
    _append_steps(qarg, U_0, M, (), data, k, cgd=False)
    return data


def unitary_cgd(qarg, U_0, H, M, Gamma, k, *, max_qubits=DEFAULT_MAX_QUBITS):
    """Apply ``k`` exact geodesic conjugate-gradient updates to ``qarg``.

    For the current step, both ``M[k-1]`` and ``Gamma[k-2]`` may be symbolic.
    Parameters belonging to earlier steps must be numeric because they define
    the current-state projector.  The first CGD step is the GD step.
    """
    _validate_inputs(qarg, M, Gamma, k, max_qubits)
    data = _build_data(qarg, U_0, H, M, Gamma, k, conjugate_gradient=True)
    _append_steps(qarg, U_0, M, Gamma, data, k, cgd=True)
    return data


__all__ = [
    "DEFAULT_MAX_QUBITS",
    "GeodesicData",
    "GeodesicEvolution",
    "unitary_gd",
    "unitary_cgd",
]
