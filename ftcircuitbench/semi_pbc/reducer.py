from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pbc_input import SourcePBCOp


@dataclass(frozen=True)
class ReducedSourceOp:
    id: int
    op: str
    term: PauliTerm
    source_id: str
    result_terms: tuple[str, ...] = ()
    result_const: int = 0
    used_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PriorMeasurement:
    index: int
    op: ReducedSourceOp
    source_term: PauliTerm


@dataclass(frozen=True)
class _Candidate:
    term: PauliTerm
    result_terms: tuple[str, ...]
    result_const: int
    used_source_ids: tuple[str, ...]


def reduce_measurements(
    source_ops: Sequence[SourcePBCOp],
    greedy_order: int = 1,
) -> list[ReducedSourceOp]:
    if greedy_order not in {0, 1, 2}:
        raise ValueError("greedy_order must be 0, 1, or 2")

    reduced: list[ReducedSourceOp] = []
    prior_measurements: list[_PriorMeasurement] = []

    for index, source_op in enumerate(source_ops):
        reduced_op = _wrap_source_op(source_op)
        if source_op.op == "m_pauli":
            eligible = [
                prior
                for prior in prior_measurements
                if _is_eligible(prior, source_ops, current_index=index)
            ]
            candidate = _select_candidate(source_op.term, eligible, greedy_order)
            if candidate is not None:
                reduced_op = ReducedSourceOp(
                    id=source_op.id,
                    op=source_op.op,
                    term=candidate.term,
                    source_id=source_op.source_id,
                    result_terms=candidate.result_terms,
                    result_const=candidate.result_const,
                    used_source_ids=candidate.used_source_ids,
                )
            prior_measurements.append(
                _PriorMeasurement(
                    index=index,
                    op=reduced_op,
                    source_term=source_op.term,
                )
            )
        reduced.append(reduced_op)

    return reduced


def _wrap_source_op(source_op: SourcePBCOp) -> ReducedSourceOp:
    return ReducedSourceOp(
        id=source_op.id,
        op=source_op.op,
        term=source_op.term,
        source_id=source_op.source_id,
    )


def _is_eligible(
    prior: _PriorMeasurement,
    source_ops: Sequence[SourcePBCOp],
    *,
    current_index: int,
) -> bool:
    return all(
        op.term.commutes_with(prior.op.term)
        for op in source_ops[prior.index + 1 : current_index]
        if op.op == "t_pauli"
    )


def _select_candidate(
    current_term: PauliTerm,
    eligible: Sequence[_PriorMeasurement],
    greedy_order: int,
) -> _Candidate | None:
    candidates = [
        candidate
        for priors in _candidate_prior_groups(eligible, greedy_order)
        if (candidate := _build_candidate(current_term, priors)) is not None
        and candidate.term.weight < current_term.weight
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_candidate_sort_key)[0]


def _candidate_prior_groups(
    eligible: Sequence[_PriorMeasurement],
    greedy_order: int,
) -> list[tuple[_PriorMeasurement, ...]]:
    if greedy_order == 0:
        return [(eligible[-1],)] if eligible else []

    groups = [(prior,) for prior in eligible]
    if greedy_order == 2:
        groups.extend(combinations(eligible, 2))
    return groups


def _build_candidate(
    current_term: PauliTerm,
    priors: tuple[_PriorMeasurement, ...],
) -> _Candidate | None:
    product = current_term
    for prior in priors:
        try:
            product = product.multiply_real(prior.source_term)
        except ValueError:
            return None

    result_const = 1 if product.sign == -1 else 0
    positive_term = PauliTerm.from_pairs(product.pairs, sign=1)
    return _Candidate(
        term=positive_term,
        result_terms=tuple(f"src{prior.op.id}" for prior in priors),
        result_const=result_const,
        used_source_ids=tuple(prior.op.source_id for prior in priors),
    )


def _candidate_sort_key(
    candidate: _Candidate,
) -> tuple[int, int, tuple[str, ...], tuple[tuple[str, str], ...]]:
    return (
        candidate.term.weight,
        len(candidate.used_source_ids),
        candidate.used_source_ids,
        candidate.term.pairs,
    )
