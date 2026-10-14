from qrisp import QuantumArray, mcp, conjugate, invert
from qite import QITE
import sympy as sp
from math import prod

def QCGD(qarg1, U_0, exp_H, M, Gamma, k, method="GC", alpha=1, beta=1):

    if k == 1:
        
        M_ = M[0]
        
        QITE(qarg1, U_0, exp_H, [M_], 1, method=method, alpha=alpha, beta=beta)
    
    else:
        M_ = M[k-1]

        conjugators = []
        
        def U_0_conjugator(qarg1):
            with invert():
                U_0(qarg1)
    
        conjugators.append(U_0_conjugator)        
        
        for i in range(1, k):
            def conjugator(qarg1, i=i):
                with invert():
                    QCGD(qarg1, U_0, exp_H, M, Gamma, i, method=method, alpha=alpha, beta=beta)
            conjugators.append(conjugator)
        
        reflections = []
    
        for i in range(k):
            def reflection(qarg1, t_, i=i): # exp{it qarg_r} qarg_t
                with conjugate(conjugators[i])(qarg1):
                    if isinstance(qarg1, QuantumArray):
                        qubits = sum([qv.reg for qv in qarg1.flatten()], [])
                        mcp(t_, qubits, ctrl_state=0, method="khattar")
                    else:
                        mcp(t_, qarg1, ctrl_state=0, method="khattar")
    
            reflections.append(reflection)

        if method == "GC":
            
            QCGD(qarg1, U_0, exp_H, M, Gamma, k-1, method=method, alpha=alpha, beta=beta)

            # Only gamma values up to the current recursion level belong to
            # this update. Recursive calls receive the full parameter lists,
            # so using Gamma[i:] would incorrectly include future gammas.
            coefficients = [prod(Gamma[i:k - 1]) for i in range(k - 1)]

            # The recursive call has already prepared U_(k-1). When mu=0,
            # the negative and positive historical-reflection sequences are
            # exact inverses, while the current-state reflection has angle
            # zero. The complete kth update is therefore the identity for
            # every value of the current gamma.
            if M_ == 0:
                return
            
            for i in range(k-2,-1,-1):
                c = coefficients[i]
                if c != 0:
                    reflections[i](qarg1, -c)

            with conjugate(exp_H)(qarg1, M_):

                for i in range(0, k - 1):
                    c = coefficients[i]
                    if c != 0:
                        reflections[i](qarg1, c)

                if M_ != 0:
                    reflections[k - 1](qarg1, M_)
        
        if method == "HOPF":
            # UPDATE HOPF METHOD HERE
            phi = (sp.sqrt(5) - 1) / 2
            
            QCGD(qarg1, U_0, exp_H, M, Gamma, k-1, method=method, alpha=alpha, beta=beta)
            
            exp_H(qarg1, -(1 - phi) * M_)

            reflections[k-1](qarg1, -(1 + phi) * M_)

            for i in range(k-2,-1,-1):
                c = prod(Gamma[i:k - 1])
                reflections[i](qarg1, -(1 + phi) * c)

            exp_H(qarg1, M_)

            reflections[k-1](qarg1, phi * M_)

            for i in range(k-2,-1,-1):
                c = prod(Gamma[i:k - 1])
                reflections[i](qarg1, phi * c)
            
            exp_H(qarg1, -phi * M_)
