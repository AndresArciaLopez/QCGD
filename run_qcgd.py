from qrisp import QuantumVariable 
from qcgd import QCGD
import numpy as np
import sympy as sp
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

def build_numeric_qcgd_circuit(N,U_0,exp_H,
    M_values,G_values,k,formula="GC",
    alpha=1,beta=1,):
    """
    Construct and compile a QCGD circuit using numerical parameters only.
    """

    if len(M_values) < k:
        raise ValueError(
            f"Step {k} requires {k} M values, "
            f"but only {len(M_values)} were supplied."
        )

    if k > 1 and len(G_values) < k - 1:
        raise ValueError(
            f"Step {k} requires {k - 1} Gamma values, "
            f"but only {len(G_values)} were supplied."
        )

    qv = QuantumVariable(N)

    QCGD(qv,U_0,exp_H,
        list(M_values[:k]),list(G_values[:k - 1]),k,
        method=formula,alpha=alpha,beta=beta,)

    qc = qv.qs.compile()

    if qc.abstract_params:
        raise RuntimeError(
            f"The rebuilt circuit still contains symbolic parameters: "
            f"{qc.abstract_params}"
        )

    return qv, qc

def compute_moments(psi, H):
    E = (psi.conj().T @ H.dot(psi)).real
    S = (psi.conj().T @ (H @ H).dot(psi)).real
    return E, S, S - E**2

def run_qcgd(N, steps, H, exp_H, U_0, formula="GC", use_statevectors=False, title="", param_max=0.5, alpha=1, beta=1):
#N = Number of Qubits, steps = number of iteration steps,
# H = Hamiltonian, exp_H = Hamiltonian evolution, U_0 = starting unitary
    H_matrix = H.to_sparse_matrix()    
    statevectors = []
    circuits = []
    
    qv = QuantumVariable(N)
    U_0(qv)
    qc = qv.qs.compile()
    circuits.append(qc)

    if use_statevectors:
        E_0, _, _ = compute_moments(get_statevector(qc, N), H_matrix)
    else:
        E_0 = H.expectation_value(qv, precision=0.01, precompiled_qc=qc, diagonalisation_method='commuting')()
        

    M_values = np.linspace(0, param_max, 20)
    G_values = np.linspace(0, param_max, 20)

    mu = sp.Symbol('mu')
    gamma = sp.Symbol('gamma')

    optimal_M= [mu]
    optimal_G = [gamma]
    optimal_energies = [E_0]

    # take QITE step
    def qite_step():
        qv = QuantumVariable(N)
        QCGD(qv, U_0, exp_H, optimal_M, optimal_G, 1, method=formula, alpha=alpha, beta=beta)
        return qv

    qv = qite_step()
    qc = qv.qs.compile()
    # circuits.append(qc)
    
    # Find optimal evolution time 
    if use_statevectors:
        energies = [compute_moments(get_statevector(qc, N, subs_dic={mu:m_}), H_matrix)[0] for m_ in M_values]
    else:
        energies = [H.expectation_value(qite_step, precision=0.01, diagonalisation_method='commuting', subs_dic={mu:m_}, precompiled_qc=qc)() for m_ in M_values]

    print("[QCGD] THIS IS MIN ENERGY AFTER QITE step:", np.min(energies))
    index = np.argmin(energies)
    M_min = M_values[index]

    optimal_M.insert(-1,M_min)
    optimal_energies.append(energies[index])

    numeric_M = optimal_M[:-1]
    numeric_G = optimal_G[:-1]

    qv_numeric, qc_numeric = build_numeric_qcgd_circuit(
        N,
        U_0,
        exp_H,
        numeric_M,
        numeric_G,
        1,
        formula=formula,
        alpha=alpha,
        beta=beta,
    )

    circuits.append(qc_numeric)

    for k in range(2,steps+1):

        # Perform k steps of qcgd
        def state_prep():
            qv = QuantumVariable(N)
            QCGD(qv, U_0, exp_H, optimal_M, optimal_G, k, method=formula, alpha=alpha, beta=beta)
            return qv
                
        qv = state_prep()
        qc = qv.qs.compile()
        
        if use_statevectors:
            energies = np.array([[compute_moments(get_statevector(qc, N, subs_dic={mu:m_, gamma:g_}), H_matrix)[0] for m_ in M_values] for g_ in G_values])
        else:
            energies = np.array([[H.expectation_value(state_prep, precision=0.01, diagonalisation_method='commuting', subs_dic={mu:m_, gamma:g_}, precompiled_qc=qc)() for m_ in M_values] for g_ in G_values])

        print("[QCGD] THIS IS ENERGY at mu=0,gamma=0 AFTER QCGD step", k, ": ", energies[0, 0])
        print("[QCGD] THIS IS ENERGY at mu=0,gamma=random AFTER QCGD step", k, ": ", energies[11,0])
        print("[QCGD] THIS IS MIN ENERGY AFTER QCGD step", k, ": ", np.min(energies))

        index = np.unravel_index(np.argmin(energies), energies.shape)
        gamma_zero_index = 0 
        gamma_index = index[0]
        mu_index = index[1]
        M_min = M_values[mu_index]
        G_min = G_values[gamma_index]
        energy_min = energies[index]

        save_data({"L": N, "k": k, "M_values": M_values, "G_values": G_values, 
                   "energies": energies, "M_min": M_min, "G_min": G_min, "energy_min": energy_min}, 
                   f"EnergyL={N}step{k}{title}", 
                   [f"alpha{alpha}beta{beta}",f"resultsL={N}", f"energies{N}range{max(M_values):.1f}x{max(G_values):.1f}"])

        # Find the sampled gamma value closest to zero
        gamma_zero = G_values[gamma_zero_index]

        energies_gamma_zero = energies[gamma_zero_index, :]
        energies_gamma_optimal = energies[gamma_index, :]

        save_data({"L": N, "k": k, "M_values": M_values, "energies_gamma_zero": energies_gamma_zero, 
                   "energies_gamma_optimal": energies_gamma_optimal, "M_min": M_min, "G_min": G_min, 
                   "energy_min": energy_min}, 
                   f"EnergyL={N}step{k}{title}", 
                   [f"alpha{alpha}beta{beta}", f"resultsL={N}", f"energies{N}mu{max(M_values):.1f}compare"])

        optimal_M.insert(-1,M_min)
        optimal_G.insert(-1,G_min)
        optimal_energies.append(energies[index])

        numeric_M = optimal_M[:-1]
        numeric_G = optimal_G[:-1]

        qv_numeric, qc_numeric = build_numeric_qcgd_circuit(N,U_0,exp_H,numeric_M,numeric_G,
                                                            k,formula=formula,alpha=alpha,beta=beta,)
        circuits.append(qc_numeric)

    circuit_ops = {}
    circuit_qubits = {}
    circuit_depth = {}

    # Collect data for circuits and statevectors
    for k, qc in enumerate(circuits):

        tqc = qc.transpile(basis_gates=["cz", "u"])
        circuit_ops[k] = tqc.count_ops()
        circuit_qubits[k] = tqc.num_qubits()
        circuit_depth[k] = tqc.depth()
        statevectors.append(get_statevector(qc, N))

    circuit_data = [circuit_ops, circuit_qubits, circuit_depth]

    print("[QCGD] THESE ARE THE OPTIMAL ENERGIES:", optimal_energies)
    print("[QCGD] THESE ARE THE OPTIMAL M:", optimal_M)
    print("[QCGD] THESE ARE THE OPTIMAL G:", optimal_G)

    return {"energies": optimal_energies, "circuit_data": circuit_data, 'statevectors': statevectors}
