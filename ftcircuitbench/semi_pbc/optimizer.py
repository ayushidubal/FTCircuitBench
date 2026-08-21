from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import combinations

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.lowering import lower_pauli_rotation
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.reducer import ReducedSourceOp

_DEFAULT_ENUMERATION_LIMIT = 128


@dataclass(frozen=True)
class RetainedBlock:
    retained_qubits: tuple[str, ...]
    extra_qubits: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class RotationRunOptimization:
    ops: list[SemiPBCOp]
    source_ops_by_output_id: dict[int, ReducedSourceOp]


@dataclass(frozen=True)
class _RotationFrameCandidate:
    block: RetainedBlock
    prefix: tuple[SemiPBCOp, ...]
    rotation: SemiPBCOp
    suffix: tuple[SemiPBCOp, ...]


def choose_retained_block(
    term: PauliTerm,
    *,
    k: int,
    neighbor_terms: Iterable[PauliTerm] = (),
    enumeration_limit: int = _DEFAULT_ENUMERATION_LIMIT,
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


def candidate_retained_blocks(
    term: PauliTerm,
    *,
    k: int,
    neighbor_terms: Iterable[PauliTerm] = (),
    enumeration_limit: int = _DEFAULT_ENUMERATION_LIMIT,
) -> tuple[RetainedBlock, ...]:
    _validate_term(term)
    _validate_positive_int(k, "k")
    _validate_positive_int(enumeration_limit, "enumeration_limit")

    neighbor_terms = tuple(neighbor_terms)
    active_qubits = tuple(qubit for qubit, _pauli in term.pairs)
    retained_count = min(k, len(active_qubits))
    if retained_count == len(active_qubits):
        return (
            _block_from_subset(active_qubits, active_qubits, active_qubits[0]),
        )

    active_order = {qubit: index for index, qubit in enumerate(active_qubits)}
    candidate_limit = math.comb(len(active_qubits), retained_count) * retained_count
    if candidate_limit <= enumeration_limit:
        blocks = [
            _block_from_subset(active_qubits, subset, target)
            for subset in combinations(active_qubits, retained_count)
            for target in subset
        ]
    else:
        canonical_subset = active_qubits[:retained_count]
        blocks = [
            _block_from_subset(active_qubits, canonical_subset, canonical_subset[0]),
            choose_retained_block(
                term,
                k=k,
                neighbor_terms=neighbor_terms,
                enumeration_limit=enumeration_limit,
            ),
        ]

    preferred = choose_retained_block(
        term,
        k=k,
        neighbor_terms=neighbor_terms,
        enumeration_limit=enumeration_limit,
    )
    return tuple(
        sorted(
            _unique_blocks((preferred, *blocks)),
            key=lambda block: (
                0 if block == preferred else 1,
                tuple(active_order[qubit] for qubit in block.retained_qubits),
            ),
        )
    )


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


def optimize_rotation_run(
    source_ops: Iterable[ReducedSourceOp],
    *,
    start_id: int,
    k: int,
    enumeration_limit: int = _DEFAULT_ENUMERATION_LIMIT,
) -> RotationRunOptimization:
    _validate_non_negative_int(start_id, "start_id")
    _validate_positive_int(k, "k")
    source_ops = tuple(source_ops)
    if not source_ops:
        return RotationRunOptimization([], {})
    for source_op in source_ops:
        _validate_rotation_source_op(source_op, k)

    candidate_sets = [
        tuple(
            _rotation_frame_candidate(source_op, block, k)
            for block in candidate_retained_blocks(
                source_op.term,
                k=k,
                neighbor_terms=_rotation_run_neighbor_terms(source_ops, index),
                enumeration_limit=enumeration_limit,
            )
        )
        for index, source_op in enumerate(source_ops)
    ]
    selected = _select_rotation_frames(candidate_sets)
    tagged = _emit_selected_rotation_frames(source_ops, selected)
    return _renumber_tagged_ops(tagged, start_id)


def _block_from_subset(
    active_qubits: tuple[str, ...],
    subset: tuple[str, ...],
    target: str,
) -> RetainedBlock:
    retained = (target,) + tuple(qubit for qubit in subset if qubit != target)
    subset_set = set(subset)
    extra = tuple(qubit for qubit in active_qubits if qubit not in subset_set)
    return RetainedBlock(retained, extra, target)


def _unique_blocks(blocks: Iterable[RetainedBlock]) -> list[RetainedBlock]:
    seen: set[tuple[str, ...]] = set()
    unique = []
    for block in blocks:
        if block.retained_qubits in seen:
            continue
        seen.add(block.retained_qubits)
        unique.append(block)
    return unique


def _rotation_run_neighbor_terms(
    source_ops: tuple[ReducedSourceOp, ...],
    index: int,
) -> tuple[PauliTerm, ...]:
    terms = []
    if index > 0:
        terms.append(source_ops[index - 1].term)
    if index + 1 < len(source_ops):
        terms.append(source_ops[index + 1].term)
    return tuple(terms)


def _rotation_frame_candidate(
    source_op: ReducedSourceOp,
    block: RetainedBlock,
    k: int,
) -> _RotationFrameCandidate:
    lowered = lower_pauli_rotation(
        0,
        source_op.term,
        k,
        source_id=source_op.source_id,
        retained_qubits=block.retained_qubits,
    )
    rotation_index = next(
        index for index, op in enumerate(lowered) if op.op == "t_pauli"
    )
    return _RotationFrameCandidate(
        block=block,
        prefix=tuple(lowered[:rotation_index]),
        rotation=lowered[rotation_index],
        suffix=tuple(lowered[rotation_index + 1 :]),
    )


def _select_rotation_frames(
    candidate_sets: list[tuple[_RotationFrameCandidate, ...]],
) -> list[_RotationFrameCandidate]:
    scores = [len(candidate.prefix) for candidate in candidate_sets[0]]
    paths = [[index] for index, _candidate in enumerate(candidate_sets[0])]

    for run_index, candidates in enumerate(candidate_sets[1:], start=1):
        next_scores: list[int] = []
        next_paths: list[list[int]] = []
        for candidate_index, candidate in enumerate(candidates):
            options = [
                (
                    scores[prior_index]
                    + _transition_cost(prior_candidate, candidate),
                    [*paths[prior_index], candidate_index],
                )
                for prior_index, prior_candidate in enumerate(
                    candidate_sets[run_index - 1]
                )
            ]
            score, path = min(options, key=lambda option: (option[0], option[1]))
            next_scores.append(score)
            next_paths.append(path)
        scores = next_scores
        paths = next_paths

    best_index = min(
        range(len(scores)),
        key=lambda index: (
            scores[index] + len(candidate_sets[-1][index].suffix),
            paths[index],
        ),
    )
    return [
        candidate_sets[run_index][candidate_index]
        for run_index, candidate_index in enumerate(paths[best_index])
    ]


def _transition_cost(
    left: _RotationFrameCandidate,
    right: _RotationFrameCandidate,
) -> int:
    return len(cancel_adjacent_inverse_cliffords((*left.suffix, *right.prefix)))


def _emit_selected_rotation_frames(
    source_ops: tuple[ReducedSourceOp, ...],
    selected: list[_RotationFrameCandidate],
) -> list[tuple[SemiPBCOp, ReducedSourceOp]]:
    tagged: list[tuple[SemiPBCOp, ReducedSourceOp]] = []
    tagged.extend((op, source_ops[0]) for op in selected[0].prefix)
    tagged.append((selected[0].rotation, source_ops[0]))
    for index in range(1, len(selected)):
        transition = _cancel_tagged_adjacent_inverse_cliffords(
            (
                *((op, source_ops[index - 1]) for op in selected[index - 1].suffix),
                *((op, source_ops[index]) for op in selected[index].prefix),
            )
        )
        tagged.extend(transition)
        tagged.append((selected[index].rotation, source_ops[index]))
    tagged.extend((op, source_ops[-1]) for op in selected[-1].suffix)
    return tagged


def _cancel_tagged_adjacent_inverse_cliffords(
    tagged_ops: Iterable[tuple[SemiPBCOp, ReducedSourceOp]],
) -> list[tuple[SemiPBCOp, ReducedSourceOp]]:
    optimized: list[tuple[SemiPBCOp, ReducedSourceOp]] = []
    for op, source_op in tagged_ops:
        if optimized and _are_inverse_cliffords(optimized[-1][0], op):
            optimized.pop()
        else:
            optimized.append((op, source_op))
    return optimized


def _renumber_tagged_ops(
    tagged_ops: Iterable[tuple[SemiPBCOp, ReducedSourceOp]],
    start_id: int,
) -> RotationRunOptimization:
    ops = []
    source_ops_by_output_id = {}
    next_id = start_id
    for op, source_op in tagged_ops:
        renumbered = replace(op, id=next_id)
        ops.append(renumbered)
        source_ops_by_output_id[next_id] = source_op
        next_id += 1
    return RotationRunOptimization(ops, source_ops_by_output_id)


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


def _validate_non_negative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_rotation_source_op(source_op: ReducedSourceOp, k: int) -> None:
    if not isinstance(source_op, ReducedSourceOp):
        raise TypeError("source_ops must contain ReducedSourceOp instances")
    if source_op.op != "t_pauli":
        raise ValueError("rotation-DP runs may contain only t_pauli source ops")
    if source_op.term.weight <= k:
        raise ValueError("rotation-DP runs may contain only high-weight rotations")
