from __future__ import annotations

import numpy as np
import pytest

from ftcircuitbench.semi_pbc.lowering import (
    lower_pauli_measurement,
    lower_pauli_rotation,
)
from ftcircuitbench.semi_pbc.pauli import PauliTerm

_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_PAULI_MATRICES = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
_CLIFFORD_MATRICES = {"h": _H, "s": _S, "sdg": _SDG}


def test_rotation_lowering_matches_original_unitary_for_xyz():
    term = PauliTerm.from_pairs([("q0", "X"), ("q1", "Y"), ("q2", "Z")])
    lowered = lower_pauli_rotation(start_id=0, term=term, k=1, source_id="line1")
    original = pauli_rotation_matrix(term, data_qubits=3)
    compiled = semi_pbc_unitary(lowered, data_qubits=3)
    assert_allclose_up_to_global_phase(compiled, original)


def test_negative_measurement_lowering_matches_source_projectors():
    term = PauliTerm.from_pairs([("q0", "X"), ("q1", "Z")], sign=-1)
    lowered = lower_pauli_measurement(
        start_id=0,
        term=term,
        k=1,
        result="src0",
        source_id="line1",
        next_ancilla=0,
        next_classical=0,
    )
    expected_zero, expected_one = signed_pauli_projectors(term, data_qubits=2)
    actual_zero, actual_one = induced_source_projectors(
        lowered.ops,
        source="src0",
        data_qubits=2,
    )
    assert np.allclose(actual_zero, expected_zero)
    assert np.allclose(actual_one, expected_one)


def test_rotation_lowering_uses_basis_changes_and_max_k_rotation():
    term = PauliTerm.from_pairs(
        [("q0", "X"), ("q1", "Y"), ("q2", "Z"), ("q3", "X")]
    )
    ops = lower_pauli_rotation(start_id=0, term=term, k=2, source_id="line4")
    assert [op.op for op in ops] == [
        "h",
        "sdg",
        "h",
        "h",
        "cx",
        "cx",
        "t_pauli",
        "cx",
        "cx",
        "h",
        "h",
        "s",
        "h",
    ]
    assert [op.qubits for op in ops if op.op == "cx"] == [
        ("q2", "q0"),
        ("q3", "q0"),
        ("q3", "q0"),
        ("q2", "q0"),
    ]
    t_ops = [op for op in ops if op.op == "t_pauli"]
    assert len(t_ops) == 1
    assert t_ops[0].term.pairs == (("q0", "Z"), ("q1", "Z"))
    assert t_ops[0].source_id == "line4"
    assert (
        max(
            (op.term.weight for op in ops if op.op in {"t_pauli", "m_pauli"}),
            default=0,
        )
        == 2
    )


def pauli_rotation_matrix(term: PauliTerm, data_qubits: int) -> np.ndarray:
    qubit_order = [f"q{i}" for i in range(data_qubits)]
    pauli = signed_pauli_matrix(term, qubit_order)
    ident = np.eye(pauli.shape[0], dtype=complex)
    theta = np.pi / 8
    return np.cos(theta) * ident - 1j * np.sin(theta) * pauli


def semi_pbc_unitary(ops, data_qubits: int) -> np.ndarray:
    qubit_order = _qubit_order(ops, data_qubits)
    dim = 2 ** len(qubit_order)
    unitary = np.eye(dim, dtype=complex)
    for op in ops:
        if op.op in _CLIFFORD_MATRICES:
            gate = _single_qubit_gate(
                _CLIFFORD_MATRICES[op.op],
                qubit_order.index(op.qubits[0]),
                len(qubit_order),
            )
        elif op.op == "cx":
            gate = _cx_gate(
                qubit_order.index(op.qubits[0]),
                qubit_order.index(op.qubits[1]),
                len(qubit_order),
            )
        elif op.op == "t_pauli":
            gate = _pauli_rotation_matrix_for_order(op.term, qubit_order)
        else:
            raise ValueError(f"unsupported unitary op {op.op!r}")
        unitary = gate @ unitary
    return unitary


def assert_allclose_up_to_global_phase(
    actual: np.ndarray, expected: np.ndarray
) -> None:
    overlap = np.vdot(expected.reshape(-1), actual.reshape(-1))
    assert abs(overlap) > 1e-12
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(actual, phase * expected, atol=1e-12)


def signed_pauli_projectors(
    term: PauliTerm,
    data_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    qubit_order = [f"q{i}" for i in range(data_qubits)]
    pauli = signed_pauli_matrix(term, qubit_order)
    ident = np.eye(pauli.shape[0], dtype=complex)
    return (ident + pauli) / 2, (ident - pauli) / 2


def induced_source_projectors(
    ops,
    source: str,
    data_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    qubit_order = _qubit_order(ops, data_qubits)
    total_qubits = len(qubit_order)
    full_dim = 2**total_qubits
    prefix = np.eye(full_dim, dtype=complex)
    measurement = None
    for op in ops:
        if op.op == "alloc":
            continue
        if op.op == "m_pauli":
            measurement = op
            break
        if op.op in _CLIFFORD_MATRICES:
            gate = _single_qubit_gate(
                _CLIFFORD_MATRICES[op.op],
                qubit_order.index(op.qubits[0]),
                total_qubits,
            )
        elif op.op == "cx":
            gate = _cx_gate(
                qubit_order.index(op.qubits[0]),
                qubit_order.index(op.qubits[1]),
                total_qubits,
            )
        elif op.op == "t_pauli":
            gate = _pauli_rotation_matrix_for_order(op.term, qubit_order)
        else:
            raise ValueError(f"unsupported prefix op {op.op!r}")
        prefix = gate @ prefix

    if measurement is None:
        raise ValueError("expected physical measurement")

    xor = next(op for op in ops if op.op == "xor" and op.target == source)
    if xor.terms != (measurement.result,):
        raise ValueError("test helper expects xor of the physical measurement result")

    physical_pauli = signed_pauli_matrix(measurement.term, qubit_order)
    full_ident = np.eye(full_dim, dtype=complex)
    physical_projectors = (
        (full_ident + physical_pauli) / 2,
        (full_ident - physical_pauli) / 2,
    )
    embed = _data_to_full_zero_ancilla_isometry(data_qubits, total_qubits)
    source_projectors = [
        np.zeros((2**data_qubits, 2**data_qubits), dtype=complex),
        np.zeros((2**data_qubits, 2**data_qubits), dtype=complex),
    ]
    isometry = prefix @ embed
    for raw, physical_projector in enumerate(physical_projectors):
        source_bit = raw ^ xor.const
        source_projectors[source_bit] += (
            isometry.conj().T @ physical_projector @ isometry
        )
    return source_projectors[0], source_projectors[1]


def signed_pauli_matrix(term: PauliTerm, qubit_order: list[str]) -> np.ndarray:
    by_qubit = dict(term.pairs)
    matrix = np.array([[1]], dtype=complex)
    for qubit in qubit_order:
        matrix = np.kron(matrix, _PAULI_MATRICES[by_qubit.get(qubit, "I")])
    return term.sign * matrix


def _pauli_rotation_matrix_for_order(
    term: PauliTerm,
    qubit_order: list[str],
) -> np.ndarray:
    pauli = signed_pauli_matrix(term, qubit_order)
    ident = np.eye(pauli.shape[0], dtype=complex)
    theta = np.pi / 8
    return np.cos(theta) * ident - 1j * np.sin(theta) * pauli


def _single_qubit_gate(gate: np.ndarray, qubit_index: int, total_qubits: int):
    matrix = np.array([[1]], dtype=complex)
    for index in range(total_qubits):
        matrix = np.kron(matrix, gate if index == qubit_index else _I)
    return matrix


def _cx_gate(control: int, target: int, total_qubits: int) -> np.ndarray:
    dim = 2**total_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        bits = _index_to_bits(column, total_qubits)
        if bits[control]:
            bits[target] ^= 1
        row = _bits_to_index(bits)
        matrix[row, column] = 1
    return matrix


def _data_to_full_zero_ancilla_isometry(
    data_qubits: int,
    total_qubits: int,
) -> np.ndarray:
    data_dim = 2**data_qubits
    full_dim = 2**total_qubits
    isometry = np.zeros((full_dim, data_dim), dtype=complex)
    for column in range(data_dim):
        bits = _index_to_bits(column, data_qubits) + [0] * (total_qubits - data_qubits)
        isometry[_bits_to_index(bits), column] = 1
    return isometry


def _qubit_order(ops, data_qubits: int) -> list[str]:
    ancillas = {
        op.qubit for op in ops if op.op in {"alloc", "release"} and op.qubit is not None
    }
    return [f"q{i}" for i in range(data_qubits)] + sorted(ancillas, key=_qubit_key)


def _qubit_key(qubit: str) -> tuple[int, int]:
    return (0 if qubit[0] == "q" else 1, int(qubit[1:]))


def _index_to_bits(index: int, width: int) -> list[int]:
    return [(index >> (width - offset - 1)) & 1 for offset in range(width)]


def _bits_to_index(bits: list[int]) -> int:
    index = 0
    for bit in bits:
        index = (index << 1) | bit
    return index


def test_rotation_passes_through_when_weight_within_k():
    term = PauliTerm.from_pairs([("q0", "Z"), ("q2", "Z")], sign=-1)
    ops = lower_pauli_rotation(start_id=10, term=term, k=2, source_id="line8")
    assert len(ops) == 1
    assert ops[0].op == "t_pauli"
    assert ops[0].id == 10
    assert ops[0].term == term


def test_rotation_lowering_rejects_identity_pauli_term():
    term = PauliTerm.from_pairs([])
    with pytest.raises(ValueError, match="identity|weight"):
        lower_pauli_rotation(start_id=0, term=term, k=1)


def test_lowering_rejects_invalid_k():
    term = PauliTerm.from_pairs([("q0", "Z")])
    with pytest.raises(ValueError, match="k"):
        lower_pauli_rotation(start_id=0, term=term, k=0)


def test_rotation_lowering_rejects_negative_start_id():
    term = PauliTerm.from_pairs([("q0", "Z")])
    with pytest.raises(ValueError, match="start_id"):
        lower_pauli_rotation(start_id=-1, term=term, k=1)


def test_measurement_lowering_uses_max_k_data_measurement_and_xor():
    term = PauliTerm.from_pairs(
        [("q0", "X"), ("q1", "Z"), ("q2", "Z"), ("q3", "Z")],
        sign=-1,
    )
    lowered = lower_pauli_measurement(
        start_id=0,
        term=term,
        k=2,
        result="src4",
        source_id="line4",
        next_ancilla=0,
        next_classical=0,
    )
    ops = lowered.ops
    assert lowered.next_ancilla == 0
    assert lowered.next_classical == 1
    assert [op.op for op in ops] == [
        "h",
        "cx",
        "cx",
        "m_pauli",
        "cx",
        "cx",
        "h",
        "xor",
    ]
    assert [op.qubits for op in ops if op.op == "cx"] == [
        ("q2", "q0"),
        ("q3", "q0"),
        ("q3", "q0"),
        ("q2", "q0"),
    ]
    measure = next(op for op in ops if op.op == "m_pauli")
    assert measure.result == "c0"
    assert measure.source_id == "line4"
    assert measure.term.sign == 1
    assert measure.term.pairs == (("q0", "Z"), ("q1", "Z"))
    xor = ops[-1]
    assert xor.target == "src4"
    assert xor.terms == ("c0",)
    assert xor.const == 1

    expected_zero, expected_one = signed_pauli_projectors(term, data_qubits=4)
    actual_zero, actual_one = induced_source_projectors(
        ops,
        source="src4",
        data_qubits=4,
    )
    assert np.allclose(actual_zero, expected_zero)
    assert np.allclose(actual_one, expected_one)


def test_measurement_passes_through_when_weight_within_k():
    term = PauliTerm.from_pairs([("q1", "Z")])
    lowered = lower_pauli_measurement(
        start_id=3,
        term=term,
        k=1,
        result="src2",
        source_id="line2",
        next_ancilla=0,
        next_classical=0,
    )
    assert [op.op for op in lowered.ops] == ["m_pauli", "xor"]
    assert lowered.ops[0].result == "c0"
    assert lowered.ops[1].target == "src2"
    assert lowered.ops[1].terms == ("c0",)
    assert lowered.ops[1].const == 0


def test_negative_low_weight_measurement_normalizes_physical_sign():
    term = PauliTerm.from_pairs([("q1", "Z")], sign=-1)
    lowered = lower_pauli_measurement(
        start_id=0,
        term=term,
        k=1,
        result="src9",
        source_id="line9",
        next_ancilla=0,
        next_classical=0,
    )
    assert [op.op for op in lowered.ops] == ["m_pauli", "xor"]
    measure = lowered.ops[0]
    assert measure.term.sign == 1
    assert measure.term.pairs == (("q1", "Z"),)
    assert lowered.ops[1].terms == ("c0",)
    assert lowered.ops[1].const == 1


def test_measurement_lowering_rejects_identity_pauli_term():
    term = PauliTerm.from_pairs([])
    with pytest.raises(ValueError, match="identity|weight"):
        lower_pauli_measurement(
            start_id=0,
            term=term,
            k=1,
            result="src0",
            source_id="line1",
            next_ancilla=0,
            next_classical=0,
        )


def test_measurement_lowering_rejects_invalid_k():
    term = PauliTerm.from_pairs([("q0", "Z")])
    with pytest.raises(ValueError, match="k"):
        lower_pauli_measurement(
            start_id=0,
            term=term,
            k=0,
            result="src0",
            source_id="line1",
            next_ancilla=0,
            next_classical=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_id": -1}, "start_id"),
        ({"result": "c0"}, "source.*result|result"),
        ({"next_ancilla": -1}, "next_ancilla"),
        ({"next_classical": -1}, "next_classical"),
    ],
)
def test_measurement_lowering_rejects_invalid_identifiers(kwargs, message):
    term = PauliTerm.from_pairs([("q0", "Z")])
    args = {
        "start_id": 0,
        "term": term,
        "k": 1,
        "result": "src0",
        "source_id": "line1",
        "next_ancilla": 0,
        "next_classical": 0,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        lower_pauli_measurement(**args)
