from __future__ import annotations

from dataclasses import dataclass

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.pauli import PauliTerm


@dataclass(frozen=True)
class LoweringResult:
    ops: list[SemiPBCOp]
    next_ancilla: int
    next_classical: int


def _basis_change_ops(term: PauliTerm) -> list[tuple[str, str]]:
    ops: list[tuple[str, str]] = []
    for qubit, pauli in term.pairs:
        if pauli == "X":
            ops.append(("h", qubit))
        elif pauli == "Y":
            ops.append(("sdg", qubit))
            ops.append(("h", qubit))
    return ops


def _inverse_basis_change_ops(term: PauliTerm) -> list[tuple[str, str]]:
    ops: list[tuple[str, str]] = []
    for qubit, pauli in reversed(term.pairs):
        if pauli == "X":
            ops.append(("h", qubit))
        elif pauli == "Y":
            ops.append(("h", qubit))
            ops.append(("s", qubit))
    return ops


def lower_pauli_measurement(
    start_id: int,
    term: PauliTerm,
    k: int,
    result: str,
    source_id: str | None,
    next_ancilla: int,
    next_classical: int,
) -> LoweringResult:
    _validate_k(k)
    _validate_non_identity_term(term)
    c_raw = f"c{next_classical}"
    xor_const = 1 if term.sign == -1 else 0
    if term.weight <= k:
        positive_term = PauliTerm.from_pairs(term.pairs, sign=1)
        return LoweringResult(
            ops=[
                SemiPBCOp.measurement(
                    start_id,
                    positive_term,
                    result=c_raw,
                    source_id=source_id,
                ),
                SemiPBCOp.xor(
                    start_id + 1,
                    target=result,
                    terms=(c_raw,),
                    const=xor_const,
                    source_id=source_id,
                ),
            ],
            next_ancilla=next_ancilla,
            next_classical=next_classical + 1,
        )

    ops: list[SemiPBCOp] = []
    next_id = start_id
    ancilla = f"a{next_ancilla}"
    active_qubits = [qubit for qubit, _pauli in term.pairs]

    ops.append(SemiPBCOp.alloc(next_id, ancilla, source_id=source_id))
    next_id += 1

    for op, qubit in _basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
        next_id += 1

    for qubit in active_qubits:
        ops.append(
            SemiPBCOp.clifford(next_id, "cx", (qubit, ancilla), source_id=source_id)
        )
        next_id += 1

    z_term = PauliTerm.from_pairs([(ancilla, "Z")], sign=1)
    ops.append(
        SemiPBCOp.measurement(
            next_id,
            z_term,
            result=c_raw,
            source_id=source_id,
        )
    )
    next_id += 1

    for op, qubit in _inverse_basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
        next_id += 1

    ops.append(SemiPBCOp.release(next_id, ancilla, source_id=source_id))
    next_id += 1

    ops.append(
        SemiPBCOp.xor(
            next_id,
            target=result,
            terms=(c_raw,),
            const=xor_const,
            source_id=source_id,
        )
    )
    return LoweringResult(
        ops=ops,
        next_ancilla=next_ancilla + 1,
        next_classical=next_classical + 1,
    )


def lower_pauli_rotation(
    start_id: int,
    term: PauliTerm,
    k: int,
    source_id: str | None = None,
) -> list[SemiPBCOp]:
    _validate_k(k)
    _validate_non_identity_term(term)
    if term.weight <= k:
        return [SemiPBCOp.pauli_rotation(start_id, term, source_id=source_id)]

    ops: list[SemiPBCOp] = []
    next_id = start_id
    active_qubits = [qubit for qubit, _pauli in term.pairs]
    target = active_qubits[0]

    for op, qubit in _basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
        next_id += 1

    for qubit in active_qubits[1:]:
        ops.append(
            SemiPBCOp.clifford(next_id, "cx", (qubit, target), source_id=source_id)
        )
        next_id += 1

    z_term = PauliTerm.from_pairs([(target, "Z")], sign=term.sign)
    ops.append(SemiPBCOp.pauli_rotation(next_id, z_term, source_id=source_id))
    next_id += 1

    for qubit in reversed(active_qubits[1:]):
        ops.append(
            SemiPBCOp.clifford(next_id, "cx", (qubit, target), source_id=source_id)
        )
        next_id += 1

    for op, qubit in _inverse_basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
        next_id += 1

    return ops


def _validate_k(k: int) -> None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be an integer >= 1")


def _validate_non_identity_term(term: PauliTerm) -> None:
    if term.weight < 1:
        raise ValueError("cannot lower identity Pauli term with weight 0")
