import pytest
from qiskit import QuantumCircuit

from ftcircuitbench.k_pbc.segmented_litinski import compile_clifford_t_to_kpbc


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


def test_segmented_litinski_rejects_unsupported_gate():
    qc = QuantumCircuit(1)
    qc.x(0)

    with pytest.raises(ValueError, match="unsupported"):
        compile_clifford_t_to_kpbc(qc, k=1)
