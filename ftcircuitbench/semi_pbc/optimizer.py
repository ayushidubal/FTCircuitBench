from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.pauli import PauliTerm


@dataclass(frozen=True)
class RetainedBlock:
    retained_qubits: tuple[str, ...]
    extra_qubits: tuple[str, ...]
    target: str


def choose_retained_block(
    term: PauliTerm,
    *,
    k: int,
    neighbor_terms: Iterable[PauliTerm] = (),
    enumeration_limit: int = 2000,
) -> RetainedBlock:
    _validate_term(term)
    _validate_positive_int(k, "k")
    _validate_positive_int(enumeration_limit, "enumeration_limit")

    active_qubits = tuple(qubit for qubit, _pauli in term.pairs)
    retained_count = min(k, len(active_qubits))
    neighbor_counts = _neighbor_overlap_counts(active_qubits, neighbor_terms)
    retained = _choose_retained_qubits(
        active_qubits,
        retained_count=retained_count,
        neighbor_counts=neighbor_counts,
        enumeration_limit=enumeration_limit,
    )
    target = _choose_target(retained, active_qubits, neighbor_counts)
    retained_order = (target,) + tuple(qubit for qubit in retained if qubit != target)
    extra_qubits = tuple(qubit for qubit in active_qubits if qubit not in retained)
    return RetainedBlock(retained_order, extra_qubits, target)


def cancel_adjacent_inverse_cliffords(
    ops: Iterable[SemiPBCOp],
) -> list[SemiPBCOp]:
    optimized: list[SemiPBCOp] = []
    for op in ops:
        if optimized and _are_inverse_cliffords(optimized[-1], op):
            optimized.pop()
        else:
            optimized.append(op)
    return optimized


def _choose_retained_qubits(
    active_qubits: tuple[str, ...],
    *,
    retained_count: int,
    neighbor_counts: Counter[str],
    enumeration_limit: int,
) -> tuple[str, ...]:
    if math.comb(len(active_qubits), retained_count) <= enumeration_limit:
        return min(
            combinations(active_qubits, retained_count),
            key=lambda candidate: _candidate_score(
                candidate,
                active_qubits,
                neighbor_counts,
            ),
        )

    active_order = {qubit: index for index, qubit in enumerate(active_qubits)}
    greedy = sorted(
        active_qubits,
        key=lambda qubit: (-neighbor_counts[qubit], active_order[qubit]),
    )[:retained_count]
    return tuple(sorted(greedy, key=active_order.__getitem__))


def _candidate_score(
    candidate: tuple[str, ...],
    active_qubits: tuple[str, ...],
    neighbor_counts: Counter[str],
) -> tuple[int, tuple[int, ...]]:
    active_order = {qubit: index for index, qubit in enumerate(active_qubits)}
    return (
        -sum(neighbor_counts[qubit] for qubit in candidate),
        tuple(active_order[qubit] for qubit in candidate),
    )


def _choose_target(
    retained_qubits: tuple[str, ...],
    active_qubits: tuple[str, ...],
    neighbor_counts: Counter[str],
) -> str:
    active_order = {qubit: index for index, qubit in enumerate(active_qubits)}
    return min(
        retained_qubits,
        key=lambda qubit: (-neighbor_counts[qubit], active_order[qubit]),
    )


def _neighbor_overlap_counts(
    active_qubits: tuple[str, ...],
    neighbor_terms: Iterable[PauliTerm],
) -> Counter[str]:
    active = set(active_qubits)
    counts: Counter[str] = Counter()
    for neighbor in neighbor_terms:
        _validate_term(neighbor)
        for qubit, _pauli in neighbor.pairs:
            if qubit in active:
                counts[qubit] += 1
    return counts


def _are_inverse_cliffords(left: SemiPBCOp, right: SemiPBCOp) -> bool:
    if left.op in {"h", "cx"}:
        return left.op == right.op and left.qubits == right.qubits
    if left.op == "s":
        return right.op == "sdg" and left.qubits == right.qubits
    if left.op == "sdg":
        return right.op == "s" and left.qubits == right.qubits
    return False


def _validate_term(term: PauliTerm) -> None:
    if not isinstance(term, PauliTerm):
        raise TypeError("term must be a PauliTerm")
    if term.weight < 1:
        raise ValueError("cannot optimize identity Pauli term with weight 0")


def _validate_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
