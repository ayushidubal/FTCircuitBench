import math

import numpy as np
from numpy import linalg as LA

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.circuit.library import QFT, HamiltonianGate, Initialize, RYGate


# Create the HHL circuit
def hhl_circuit(A, b, t0=2 * np.pi):  # get hhl circuit
    """
    A: The matrix representing the linear system.
    b: The vector representing the right-hand side of the linear system.
    t: A time parameter used in the controlled-Hamiltonian operations.
    r: A parameter used to determine the rotation angles for the ancilla qubit.
    shots: The number of shots (repetitions) for the quantum circuit execution.

    Returns:
    QuantumCircuit: The quantum circuit for solving the linear system using HHL.
    """
    # ==========================================================================
    # Preprocessing
    # ==========================================================================

    # Check the hermitian matrix
    def check_hermitian(mat):
        mat = np.asarray(mat)
        assert np.allclose(
            mat, mat.T.conjugate(), rtol=1e-05, atol=1e-08
        ), "Sorry! The input matrix should be Hermitian."

    check_hermitian(A)

    # Normalize A and b
    norm_b = LA.norm(b)
    A = A / norm_b
    b = b / norm_b

    # Calculate condition number and eigenvalues of A
    eigs = LA.eigvals(A)
    # ==========================================================================
    # Quantum Circuit
    # ==========================================================================

    # Qubit registers
    ancilla_qbit = QuantumRegister(1, name="anc")
    q_reg = QuantumRegister(math.log2(len(b)), name="q")
    b_reg = QuantumRegister(math.log2(len(b)), name="b")
    ancilla_result = ClassicalRegister(1, name="anc_result")
    b_vec = ClassicalRegister(math.log2(len(b)), name="b_vec")

    # Define empty circuit
    circ = QuantumCircuit(
        ancilla_qbit,
        q_reg,
        b_reg,
        ancilla_result,
        b_vec,
        name=f"HHL {len(b)} by {len(b)}",
    )

    # Encode vector b
    init = Initialize(list(b))
    circ.append(init, b_reg)

    # ===========================================================================
    circ.barrier()
    # Hadamard
    circ.h(q_reg)

    # ================================================================================================
    circ.barrier()
    # Apply controlled-Hamiltonian operators on register b
    for i in range(len(q_reg)):
        time = t0 / (2 ** (len(q_reg) - i))
        U = HamiltonianGate(A, time)
        G = U.control(1)
        qubit = [i + 1] + [len(q_reg) + j + 1 for j in range(int(math.log2(len(b))))]
        circ.append(G, qubit)
        # circ.G(c_reg[i], b_reg[0]) #might need to use .append instead of circ.G to this doesn't work
        # Ramin's code: qubit = [i+1]+[c_num+j+1 for j in range(b_num)]

    # ================================================================================================
    circ.barrier()
    # IQFT
    iqft = QFT(
        len(q_reg), approximation_degree=0, do_swaps=True, inverse=True, name="IQFT"
    )
    circ.append(iqft, q_reg)

    # ================================================================================================
    circ.barrier()

    for i in range(len(q_reg) + 1):
        if i != 0:
            U = RYGate((2 * np.pi) / eigs[i - 1]).control(
                1
            )  # or 2**(len(q_reg)+1-i) factor?
            circ.append(U, [i, 0])

    # ================================================================================================
    # Uncompute
    # ================================================================================================
    circ.barrier()

    # Measure ancilla qubit
    circ.measure(ancilla_qbit, ancilla_result)

    # ================================================================================================
    circ.barrier()

    # QFT
    qft = QFT(
        len(q_reg), approximation_degree=0, do_swaps=True, inverse=False, name="QFT"
    )
    circ.append(qft, q_reg)

    # ================================================================================================
    circ.barrier()
    # Inverse controlled-Hamiltonian operators on register b
    for i in range(len(q_reg))[::-1]:
        time = t0 / (2 ** (len(q_reg) - i))
        U = HamiltonianGate(A, -time)
        G = U.control(1)
        qubit = [i + 1] + [len(q_reg) + j + 1 for j in range(int(math.log2(len(b))))]
        circ.append(G, qubit)

    # ================================================================================================
    circ.barrier()
    # Hadamards
    circ.h(q_reg)

    # ================================================================================================
    circ.barrier()
    # Measure b
    circ.measure(b_reg, b_vec)

    return circ
