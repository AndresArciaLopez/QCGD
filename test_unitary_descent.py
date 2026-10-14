import numpy as np
import sympy as sp
import unittest
from scipy.linalg import expm

from qrisp import QuantumVariable, x
from qrisp.operators import X, Z

from unitary_descent import unitary_cgd, unitary_gd


def _statevector(qv, substitutions=None):
    qc = qv.qs.compile()
    if substitutions:
        qc = qc.bind_parameters(substitutions)
    return np.asarray(qc.statevector_array(), dtype=complex)


class TestUnitaryDescent(unittest.TestCase):
    def test_gd_symbolic_step_matches_dense_definition(self):
        mu = sp.Symbol("mu")
        qv = QuantumVariable(1)
        data = unitary_gd(qv, lambda qarg: None, X(0), [mu], 1)

        self.assertIn(mu, qv.qs.abstract_params)
        actual = _statevector(qv, {mu: 0.2})
        expected = expm(-0.2 * data.directions[0]) @ np.array([1.0, 0.0])
        self.assertTrue(np.allclose(actual, expected))

    def test_gd_recursion_uses_updated_projector(self):
        qv = QuantumVariable(1)
        data = unitary_gd(qv, lambda qarg: None, X(0), [0.2, 0.3], 2)

        actual = _statevector(qv)
        expected = expm(-0.3 * data.directions[1]) @ data.states[1]
        self.assertTrue(np.allclose(actual, expected))
        self.assertFalse(np.allclose(data.gradients[0], data.gradients[1]))

    def test_cgd_supports_symbolic_current_mu_and_gamma(self):
        mu = sp.Symbol("mu")
        gamma = sp.Symbol("gamma")
        qv = QuantumVariable(1)
        data = unitary_cgd(
            qv, lambda qarg: None, X(0) + 0.2 * Z(0), [0.2, mu], [gamma], 2
        )

        self.assertTrue({mu, gamma}.issubset(qv.qs.abstract_params))
        self.assertTrue(
            any(gamma in getattr(entry, "free_symbols", set()) for entry in data.directions[-1].flat)
        )
        substitutions = {mu: 0.3, gamma: 0.4}
        actual = _statevector(qv, substitutions)
        W = data.gradients[1] + substitutions[gamma] * data.directions[0]
        expected = expm(-substitutions[mu] * W) @ data.states[1]
        self.assertTrue(np.allclose(actual, expected))

    def test_non_numeric_previous_parameter_is_rejected(self):
        qv = QuantumVariable(1)
        with self.assertRaisesRegex(ValueError, r"M\[0\] must be numeric"):
            unitary_gd(qv, lambda qarg: None, X(0), [sp.Symbol("old"), 0.2], 2)

    def test_dense_size_guard_is_explicit(self):
        qv = QuantumVariable(2)
        with self.assertRaisesRegex(ValueError, "dense"):
            unitary_gd(qv, lambda qarg: x(qarg[0]), X(0), [0.1], 1, max_qubits=1)


if __name__ == "__main__":
    unittest.main()
