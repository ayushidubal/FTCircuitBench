import random

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli

from ftcircuitbench.k_pbc.segmented_litinski import compile_clifford_t_to_kpbc

_KPBC_OUTPUT_OPS = {"x", "y", "z", "h", "s", "sdg", "cx", "t_pauli"}
_RANDOM_1Q_GATES = ("id", "x", "y", "z", "h", "s", "sdg", "t", "tdg")


def _random_clifford_t_circuit(rng, num_qubits, depth):
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        if rng.random() < 0.3:
            control, target = rng.sample(range(num_qubits), 2)
            qc.cx(control, target)
        else:
            getattr(qc, rng.choice(_RANDOM_1Q_GATES))(rng.randrange(num_qubits))
    return qc


def _kpbc_unitary(ops, num_qubits):
    # A t_pauli with signed Pauli P is exp(-i pi/8 P), which equals T for P = +Z up to phase.
    dim = 2**num_qubits
    unitary = np.eye(dim, dtype=complex)
    for op in ops:
        if op.op == "t_pauli":
            labels = ["I"] * num_qubits
            for qubit, pauli in op.term.pairs:
                labels[int(qubit[1:])] = pauli
            pauli_matrix = Pauli("".join(reversed(labels))).to_matrix()
            gate = np.cos(np.pi / 8) * np.eye(dim) - 1j * op.term.sign * np.sin(
                np.pi / 8
            ) * pauli_matrix
        else:
            qc = QuantumCircuit(num_qubits)
            getattr(qc, op.op)(*(int(qubit[1:]) for qubit in op.qubits))
            gate = Operator(qc).data
        unitary = gate @ unitary
    return unitary


def _equal_up_to_global_phase(left, right):
    index = np.unravel_index(np.argmax(np.abs(right)), right.shape)
    phase = left[index] / right[index]
    return np.isclose(abs(phase), 1.0) and np.allclose(left, phase * right, atol=1e-9)


def test_segmented_litinski_keeps_support_at_k():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(0)

    header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert header.k == 1
    assert all(op.term is None or op.term.weight <= 1 for op in ops)


def test_segmented_litinski_k_equals_n_allows_full_support_case():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.t(1)

    header, ops = compile_clifford_t_to_kpbc(qc, k=2)

    assert header.k == 2
    assert any(op.op == "t_pauli" and op.term.weight == 2 for op in ops)


def test_segmented_litinski_emits_boundary_clifford_when_growth_would_exceed_k():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.t(1)

    header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert header.k == 1
    assert any(op.op == "cx" for op in ops)
    assert all(op.term is None or op.term.weight <= 1 for op in ops)


def test_segmented_litinski_moves_t_before_x_and_keeps_x():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.t(0)

    header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert header.data_qubits == 1
    assert [op.op for op in ops] == ["t_pauli", "x"]
    assert ops[0].term.pairs == (("q0", "Z"),)
    assert ops[0].term.sign == -1


def test_segmented_litinski_keeps_absorbed_clifford_after_rotation():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)

    _header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert [op.op for op in ops] == ["t_pauli", "h"]
    assert ops[0].term.pairs == (("q0", "X"),)
    assert ops[0].term.sign == 1


@pytest.mark.parametrize(("phase_gate", "expected_sign"), [("s", -1), ("sdg", 1)])
def test_segmented_litinski_conjugates_t_backward_through_phase_gate(
    phase_gate, expected_sign
):
    # T moved before (phase_gate; h) becomes the rotation phase_gate^dag H Z H phase_gate.
    qc = QuantumCircuit(1)
    getattr(qc, phase_gate)(0)
    qc.h(0)
    qc.t(0)

    _header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert [op.op for op in ops] == ["t_pauli", phase_gate, "h"]
    assert ops[0].term.pairs == (("q0", "Y"),)
    assert ops[0].term.sign == expected_sign


@pytest.mark.parametrize("k", [1, 2, 3])
def test_segmented_litinski_preserves_unitary_on_random_circuits(k):
    rng = random.Random(1000 + k)
    for _trial in range(150):
        qc = _random_clifford_t_circuit(rng, num_qubits=3, depth=rng.randint(1, 16))

        _header, ops = compile_clifford_t_to_kpbc(qc, k=k)

        assert {op.op for op in ops} <= _KPBC_OUTPUT_OPS
        assert all(op.term is None or op.term.weight <= k for op in ops)
        assert _equal_up_to_global_phase(
            Operator(qc).data, _kpbc_unitary(ops, qc.num_qubits)
        ), qc.draw(output="text")


def test_segmented_litinski_emits_native_xyz_when_not_absorbed():
    qc = QuantumCircuit(3)
    qc.t(0)
    qc.x(0)
    qc.y(1)
    qc.z(2)

    _header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert [op.op for op in ops] == ["t_pauli", "x", "y", "z"]


def test_segmented_litinski_drops_identity_gate():
    qc = QuantumCircuit(1)
    qc.id(0)
    qc.t(0)
    qc.id(0)

    _header, ops = compile_clifford_t_to_kpbc(qc, k=1)

    assert [op.op for op in ops] == ["t_pauli"]


def test_segmented_litinski_rejects_unsupported_gate():
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)

    with pytest.raises(ValueError, match="unsupported"):
        compile_clifford_t_to_kpbc(qc, k=1)
