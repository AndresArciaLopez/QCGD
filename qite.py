from qrisp import QuantumArray, mcp, conjugate, invert
import sympy as sp


def QITE(qarg, U_0, exp_H, s, k, method="GC", alpha=1, beta=1):
    if k == 0:
        U_0(qarg)
    else:
        s_ = s[k - 1]

        def conjugator(qarg):
            with invert():
                QITE(qarg, U_0, exp_H, s, k - 1, method=method, alpha=alpha, beta=beta)

        def reflection(qarg, t_):
            with conjugate(conjugator)(qarg):
                if isinstance(qarg, QuantumArray):
                    qubits = sum([qv.reg for qv in qarg.flatten()], [])
                    mcp(alpha/beta * t_, qubits, ctrl_state=0, method="khattar")
                else:
                    mcp(alpha/beta * t_, qarg, ctrl_state=0, method="khattar")

        if method == "GC":

            QITE(qarg, U_0, exp_H, s, k - 1, method=method, alpha=alpha, beta=beta)

            with conjugate(exp_H)(qarg, s_):
                reflection(qarg, s_)

        if method == "HOPF":

            phi = (sp.sqrt(5) - 1) / 2

            QITE(qarg, U_0, exp_H, s, k - 1, method=method, alpha=alpha, beta=beta)

            # exp_H performs forward evolution $e^{-itH}
            exp_H(qarg, -(1 - phi) * s_)
            reflection(qarg, -(1 + phi) * s_)
            exp_H(qarg, s_)
            reflection(qarg, phi * s_)
            exp_H(qarg, -phi * s_)