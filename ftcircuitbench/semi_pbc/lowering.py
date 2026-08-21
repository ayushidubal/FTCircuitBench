from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.pauli import PauliTerm

_SOURCE_CLASSICAL_RE = re.compile(r"src(?:0|[1-9][0-9]*)\Z")


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
    retained_qubits: Sequence[str] | None = None,
) -> LoweringResult:
    _validate_non_negative_int(start_id, "start_id")
    _validate_k(k)
    _validate_non_identity_term(term)
    _validate_source_result(result)
    _validate_non_negative_int(next_ancilla, "next_ancilla")
    _validate_non_negative_int(next_classical, "next_classical")
    c_raw = f"c{next_classical}"
    xor_const = 1 if term.sign == -1 else 0
    if term.weight <= k:
        if retained_qubits is not None:
            _kept_and_extra_qubits(term, k, retained_qubits)
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
    kept_qubits, extra_qubits = _kept_and_extra_qubits(term, k, retained_qubits)
    target = kept_qubits[0]

    next_id = _append_basis_change_ops(ops, next_id, term, source_id)
    next_id = _append_compression_cx_ops(
        ops,
        next_id,
        extra_qubits=extra_qubits,
        target=target,
        source_id=source_id,
    )

    z_term = PauliTerm.from_pairs(((qubit, "Z") for qubit in kept_qubits), sign=1)
    ops.append(
        SemiPBCOp.measurement(
            next_id,
            z_term,
            result=c_raw,
            source_id=source_id,
        )
    )
    next_id += 1

    next_id = _append_compression_cx_ops(
        ops,
        next_id,
        extra_qubits=reversed(extra_qubits),
        target=target,
        source_id=source_id,
    )
    for op, qubit in _inverse_basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
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
        next_ancilla=next_ancilla,
        next_classical=next_classical + 1,
    )


def lower_pauli_rotation(
    start_id: int,
    term: PauliTerm,
    k: int,
    source_id: str | None = None,
    retained_qubits: Sequence[str] | None = None,
) -> list[SemiPBCOp]:
    _validate_non_negative_int(start_id, "start_id")
    _validate_k(k)
    _validate_non_identity_term(term)
    if term.weight <= k:
        if retained_qubits is not None:
            _kept_and_extra_qubits(term, k, retained_qubits)
        return [SemiPBCOp.pauli_rotation(start_id, term, source_id=source_id)]

    ops: list[SemiPBCOp] = []
    next_id = start_id
    kept_qubits, extra_qubits = _kept_and_extra_qubits(term, k, retained_qubits)
    target = kept_qubits[0]

    next_id = _append_basis_change_ops(ops, next_id, term, source_id)
    next_id = _append_compression_cx_ops(
        ops,
        next_id,
        extra_qubits=extra_qubits,
        target=target,
        source_id=source_id,
    )

    z_term = PauliTerm.from_pairs(
        ((qubit, "Z") for qubit in kept_qubits),
        sign=term.sign,
    )
    ops.append(SemiPBCOp.pauli_rotation(next_id, z_term, source_id=source_id))
    next_id += 1

    next_id = _append_compression_cx_ops(
        ops,
        next_id,
        extra_qubits=reversed(extra_qubits),
        target=target,
        source_id=source_id,
    )
    for op, qubit in _inverse_basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
        next_id += 1

    return ops


def _kept_and_extra_qubits(
    term: PauliTerm,
    k: int,
    retained_qubits: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    active_qubits = [qubit for qubit, _pauli in term.pairs]
    if retained_qubits is None:
        return active_qubits[:k], active_qubits[k:]

    if isinstance(retained_qubits, (str, bytes)):
        raise TypeError("retained_qubits must be a sequence of qubit ids")
    kept_qubits = list(retained_qubits)
    required_count = min(k, len(active_qubits))
    if len(kept_qubits) != required_count:
        raise ValueError(
            "retained_qubits must contain exactly "
            f"{required_count} active qubits"
        )

    active_set = set(active_qubits)
    seen: set[str] = set()
    for qubit in kept_qubits:
        if qubit in seen:
            raise ValueError(f"duplicate retained qubit {qubit!r}")
        if qubit not in active_set:
            raise ValueError(f"retained qubit {qubit!r} is not active in the term")
        seen.add(qubit)
    return kept_qubits, [qubit for qubit in active_qubits if qubit not in seen]


def _append_basis_change_ops(
    ops: list[SemiPBCOp],
    next_id: int,
    term: PauliTerm,
    source_id: str | None,
) -> int:
    for op, qubit in _basis_change_ops(term):
        ops.append(SemiPBCOp.clifford(next_id, op, (qubit,), source_id=source_id))
        next_id += 1
    return next_id


def _append_compression_cx_ops(
    ops: list[SemiPBCOp],
    next_id: int,
    *,
    extra_qubits: Iterable[str],
    target: str,
    source_id: str | None,
) -> int:
    for qubit in extra_qubits:
        ops.append(
            SemiPBCOp.clifford(next_id, "cx", (qubit, target), source_id=source_id)
        )
        next_id += 1
    return next_id


def _validate_k(k: int) -> None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be an integer >= 1")


def _validate_non_negative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_source_result(result: str) -> None:
    if not isinstance(result, str) or _SOURCE_CLASSICAL_RE.fullmatch(result) is None:
        raise ValueError("result must be a source classical id src<N>")


def _validate_non_identity_term(term: PauliTerm) -> None:
    if not isinstance(term, PauliTerm):
        raise TypeError("term must be a PauliTerm")
    if term.weight < 1:
        raise ValueError("cannot lower identity Pauli term with weight 0")
