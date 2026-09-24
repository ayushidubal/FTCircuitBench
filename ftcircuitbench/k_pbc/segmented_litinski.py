from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from ftcircuitbench.k_pbc.ir import KPBCHeader, KPBCOp
from ftcircuitbench.pbc_converter.tab_gate import TableauPauliBasis
from ftcircuitbench.semi_pbc.pauli import PauliTerm

_CLIFFORD_GATES = {"h", "s", "sdg", "cx"}
_SKIPPED_GATES = {"barrier", "measure"}
_T_GATES = {"t", "tdg"}
_CLIFFORD_DECOMPOSITIONS = {
    "x": (("h", 0), ("s", 0), ("s", 0), ("h", 0)),
    "z": (("s", 0), ("s", 0)),
    "y": (("h", 0), ("s", 0), ("s", 0), ("h", 0), ("s", 0), ("s", 0)),
}


def compile_clifford_t_to_kpbc(
    qc: QuantumCircuit, *, k: int
) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    """Compile a Clifford+T circuit into capped segmented k-PBC candidates."""
    header = KPBCHeader(k=k, data_qubits=qc.num_qubits)
    pending_rows: list[np.ndarray] = []
    reverse_items: list[tuple[str, Any, Any]] = []

    for gate_name, qubits in reversed(_expanded_instruction_names_and_qubits(qc)):
        if gate_name in _SKIPPED_GATES:
            continue
        if gate_name in _T_GATES:
            pending_rows.append(
                _z_rotation_row(qc.num_qubits, qubits[0], tdg=gate_name == "tdg")
            )
            continue
        if gate_name in _CLIFFORD_GATES:
            prospective = _prospective_rows_after_gate(
                pending_rows, gate_name, qubits, qc.num_qubits
            )
            if all(_row_weight(row, qc.num_qubits) <= k for row in prospective):
                pending_rows = prospective
                continue
            _append_pending(reverse_items, pending_rows)
            reverse_items.append(("clifford", gate_name, tuple(qubits)))
            pending_rows = []
            continue
        raise ValueError(f"unsupported gate {gate_name!r}")

    _append_pending(reverse_items, pending_rows)
    ops = _items_to_ops(reversed(reverse_items), qc.num_qubits)
    header.validate_ops(ops)
    return header, ops


def _z_rotation_row(num_qubits: int, qubit: int, tdg: bool) -> np.ndarray:
    row = np.zeros(2 * num_qubits + 1, dtype=bool)
    row[num_qubits + qubit] = True
    row[-1] = tdg
    return row


def _row_weight(row: np.ndarray, num_qubits: int) -> int:
    return int(np.count_nonzero(row[:num_qubits] | row[num_qubits : 2 * num_qubits]))


def _row_to_pauli_term(row: np.ndarray, num_qubits: int) -> PauliTerm:
    pairs: list[tuple[str, str]] = []
    for qubit in range(num_qubits):
        x_bit = bool(row[qubit])
        z_bit = bool(row[num_qubits + qubit])
        if x_bit and z_bit:
            pairs.append((f"q{qubit}", "Y"))
        elif x_bit:
            pairs.append((f"q{qubit}", "X"))
        elif z_bit:
            pairs.append((f"q{qubit}", "Z"))
    sign = -1 if bool(row[-1]) else 1
    return PauliTerm.from_pairs(pairs, sign=sign)


def _prospective_rows_after_gate(
    rows: list[np.ndarray], gate_name: str, qubits: list[int], num_qubits: int
) -> list[np.ndarray]:
    if not rows:
        return []
    _validate_clifford_shape(gate_name, qubits)
    tableau = TableauPauliBasis(np.array(rows, dtype=bool, copy=True))
    if tableau.qubits != num_qubits:
        raise ValueError("pending rows do not match circuit width")
    if gate_name == "sdg":
        for _ in range(3):
            tableau.apply_gate("s", qubits)
    else:
        tableau.apply_gate(gate_name, qubits)
    return [row.copy() for row in tableau.tableau]


def _append_pending(
    reverse_items: list[tuple[str, Any, Any]], pending_rows: list[np.ndarray]
) -> None:
    for row in pending_rows:
        reverse_items.append(("t_pauli", row.copy(), None))


def _items_to_ops(items, num_qubits: int) -> tuple[KPBCOp, ...]:
    ops: list[KPBCOp] = []
    for op_id, item in enumerate(items):
        kind, payload, extra = item
        if kind == "t_pauli":
            ops.append(KPBCOp.t_pauli(op_id, _row_to_pauli_term(payload, num_qubits)))
        elif kind == "clifford":
            ops.append(
                KPBCOp.clifford(
                    op_id,
                    payload,
                    tuple(f"q{qubit}" for qubit in extra),
                )
            )
        else:
            raise ValueError(f"unsupported segmented item {kind!r}")
    return tuple(ops)


def _instruction_name_and_qubits(
    qc: QuantumCircuit, instruction
) -> tuple[str, list[int]]:
    operation = getattr(instruction, "operation", None)
    qargs = getattr(instruction, "qubits", None)
    if operation is None or qargs is None:
        operation, qargs, _clbits = instruction
    return operation.name, [qc.find_bit(qubit).index for qubit in qargs]


def _expanded_instruction_names_and_qubits(qc: QuantumCircuit) -> list[tuple[str, list[int]]]:
    expanded: list[tuple[str, list[int]]] = []
    for instruction in qc.data:
        gate_name, qubits = _instruction_name_and_qubits(qc, instruction)
        decomposition = _CLIFFORD_DECOMPOSITIONS.get(gate_name)
        if decomposition is None:
            expanded.append((gate_name, qubits))
            continue
        if len(qubits) != 1:
            raise ValueError(f"{gate_name} requires exactly one qubit")
        for decomposed_gate, qubit_index in decomposition:
            expanded.append((decomposed_gate, [qubits[qubit_index]]))
    return expanded


def _validate_clifford_shape(gate_name: str, qubits: list[int]) -> None:
    if gate_name in {"h", "s", "sdg"}:
        if len(qubits) != 1:
            raise ValueError(f"{gate_name} requires exactly one qubit")
        return
    if gate_name == "cx":
        if len(qubits) != 2 or qubits[0] == qubits[1]:
            raise ValueError("cx requires two distinct qubits")
        return
    raise ValueError(f"unsupported gate {gate_name!r}")
