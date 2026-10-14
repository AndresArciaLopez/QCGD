from qrisp import QuantumVariable
from qite import QITE
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from saveload import save_data

def get_statevector(qc, n, subs_dic=None):
    if subs_dic is not None:
        bqc = qc.bind_parameters(subs_dic) 
    else:
        bqc = qc.copy()
    
    for i in range(bqc.num_qubits() - n):
        bqc.qubits.insert(0, bqc.qubits.pop(-1))

    statevector = bqc.statevector_array()[:2**n]
    statevector = statevector/np.linalg.norm(statevector)

    return statevector

def compute_moments(psi, H):
    E = (psi.conj().T @ H.dot(psi)).real
    S = (psi.conj().T @ (H @ H).dot(psi)).real
    return E, S, S - E**2

def run_qite(N, steps, H, exp_H, U_0, formula="GC", use_statevectors=False, title="", mu_max=0.5, alpha=1,beta=1):
    # System Hamiltonian provided by H and Hamiltonian evolution represented by exp_H. The initial state is prepared by U_0. The number of steps is specified by steps. The formula for QITE can be specified by formula, which can be either "GC" or "LS".
    # Number of QITE steps given by steps.
    # Size of system is N and initialisation is U_0
    H_matrix = H.to_sparse_matrix()
    statevectors = []
    circuits = []
    
    def initial_step():
        qv = QuantumVariable(N)
        U_0(qv)
        return qv
    
    qv = initial_step()
    qc = qv.qs.compile()
    circuits.append(qc)

    if use_statevectors:
        E_0, _, _ = compute_moments(get_statevector(qc, N), H_matrix)
    else:
        E_0 = H.expectation_value(initial_step, precision=0.01, precompiled_qc=qc, diagonalisation_method='commuting')()

    s_values = np.linspace(0, mu_max, 20)

    theta = sp.Symbol('theta')
    optimal_s = [theta]
    optimal_energies = [E_0]

    for k in range(1,steps+1):

        # Perform k steps of QITE
        def state_prep():
            qv = QuantumVariable(N)
            QITE(qv, U_0, exp_H, optimal_s, k, method=formula, alpha=alpha, beta=beta)
            return qv

        qv = state_prep()
        qc = qv.qs.compile()
        circuits.append(qc)

        # Find optimal evolution time 
        if use_statevectors:
            energies = [compute_moments(get_statevector(qc, N, subs_dic={theta:s_}), H_matrix)[0] for s_ in s_values]
        else:
            energies = [H.expectation_value(state_prep, precision=0.01, diagonalisation_method='commuting', subs_dic={theta:s_}, precompiled_qc=qc)() for s_ in s_values]

        index = np.argmin(energies)
        s_min = s_values[index]
        min_energy = energies[index]
        print("[QITE] THIS IS MIN ENERGY AFTER QITE step", k, ": ", min_energy)
        
        save_data({"L": N, "k": k, "s_values": s_values, "energies": energies, "s_min": s_min, 
                   "min_energy": min_energy}, 
                   f"EnergyL={N}step{k}{title}", 
                   [f"alpha{alpha}beta{beta}", f"resultsL={N}", f"energies{N}mu{max(s_values):.1f}"])
        
        optimal_s.insert(-1,s_min)
        optimal_energies.append(min_energy)

    circuit_ops = {}
    circuit_qubits = {}
    circuit_depth = {}

    # Collect data for circuits and statevectors
    for k, qc in enumerate(circuits):
        # We need to bind the symbolic parameter theta to transpile to specific basis gates
        if k>0:
            qc = qc.bind_parameters(subs_dic={theta:optimal_s[k-1]})   
 
        tqc = qc.transpile(basis_gates=["cz","u"])
        circuit_ops[k] = tqc.count_ops()
        circuit_qubits[k] = tqc.num_qubits()
        circuit_depth[k] = tqc.depth()

        statevectors.append(get_statevector(qc, N))

    circuit_data = [circuit_ops, circuit_qubits, circuit_depth]


    print("[QITE] THESE ARE THE OPTIMAL ENERGIES:", optimal_energies)
    print("[QITE] THESE ARE THE OPTIMAL S:", optimal_s)
    
    return {"energies": optimal_energies, "circuit_data": circuit_data, "statevectors": statevectors}
