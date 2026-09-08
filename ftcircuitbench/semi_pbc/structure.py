from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pbc_input import SourcePBCOp, parse_pbc_text

_SUPPORTED_AI_WINDOW_WIDTHS = {4, 5, 6}


def summarize_pbc_file(path: str | Path, *, k: int) -> dict[str, Any]:
    return summarize_pbc_text(Path(path).read_text(encoding="utf-8"), k=k)


def summarize_pbc_text(text: str, *, k: int) -> dict[str, Any]:
    if type(k) is not int or k < 1:
        raise ValueError("k must be an integer >= 1")
    program = parse_pbc_text(text)
    ops = tuple(program.ops)
    return {
        "format": "semi-pbc-structure-summary",
        "version": 1,
        "k": k,
        "data_qubits": program.data_qubits,
        "source_op_count": len(ops),
        "op_counts": _op_counts(ops),
        "max_weight": max((op.term.weight for op in ops), default=0),
        "weight_histogram": _weight_histogram(ops),
        "high_weight_counts": _high_weight_counts(ops, k),
        "rotation_runs": _high_weight_rotation_runs(ops, k),
        "rotation_windows": _rotation_windows(ops, k),
        "adjacent_support_overlap": _adjacent_support_overlap(ops),
        "adjacent_commutation": _adjacent_commutation(ops),
        "repeated_supports": _repeated_supports(ops),
        "repeated_terms": _repeated_terms(ops, program.data_qubits),
    }


def compare_pbc_files(
    left_path: str | Path,
    right_path: str | Path,
    *,
    k: int,
    left_label: str | None = None,
    right_label: str | None = None,
) -> dict[str, Any]:
    left_text = Path(left_path).read_text(encoding="utf-8")
    right_text = Path(right_path).read_text(encoding="utf-8")
    left_program = parse_pbc_text(left_text)
    right_program = parse_pbc_text(right_text)
    left = summarize_pbc_text(left_text, k=k)
    right = summarize_pbc_text(right_text, k=k)
    left["label"] = left_label or Path(left_path).stem
    right["label"] = right_label or Path(right_path).stem
    return _comparison(left, right, left_program.data_qubits, right_program.data_qubits)


def compare_pbc_texts(
    left_text: str,
    right_text: str,
    *,
    k: int,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, Any]:
    left_program = parse_pbc_text(left_text)
    right_program = parse_pbc_text(right_text)
    left = summarize_pbc_text(left_text, k=k)
    right = summarize_pbc_text(right_text, k=k)
    left["label"] = left_label
    right["label"] = right_label
    return _comparison(left, right, left_program.data_qubits, right_program.data_qubits)


def _comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    left_data_qubits: int,
    right_data_qubits: int,
) -> dict[str, Any]:
    return {
        "format": "semi-pbc-structure-comparison",
        "version": 1,
        "k": left["k"],
        "left": left,
        "right": right,
        "deltas": {
            "data_qubits": right_data_qubits - left_data_qubits,
            "source_op_count": right["source_op_count"] - left["source_op_count"],
            "high_weight_t_pauli_count": (
                right["high_weight_counts"]["t_pauli"]
                - left["high_weight_counts"]["t_pauli"]
            ),
            "high_weight_m_pauli_count": (
                right["high_weight_counts"]["m_pauli"]
                - left["high_weight_counts"]["m_pauli"]
            ),
            "high_weight_rotation_run_count": (
                right["rotation_runs"]["count"] - left["rotation_runs"]["count"]
            ),
            "max_high_weight_rotation_run": (
                right["rotation_runs"]["max_length"]
                - left["rotation_runs"]["max_length"]
            ),
            "ai_candidate_window_count": (
                right["rotation_windows"]["candidate_count"]
                - left["rotation_windows"]["candidate_count"]
            ),
            "ai_eligible_window_count": (
                right["rotation_windows"]["eligible_count"]
                - left["rotation_windows"]["eligible_count"]
            ),
            "adjacent_support_total_overlap": (
                right["adjacent_support_overlap"]["total_overlap"]
                - left["adjacent_support_overlap"]["total_overlap"]
            ),
            "adjacent_noncommuting_pairs": (
                right["adjacent_commutation"]["noncommuting_pairs"]
                - left["adjacent_commutation"]["noncommuting_pairs"]
            ),
        },
    }


def _op_counts(ops: Sequence[SourcePBCOp]) -> dict[str, int]:
    counts = Counter(op.op for op in ops)
    return {op: counts[op] for op in sorted(counts)}


def _weight_histogram(ops: Sequence[SourcePBCOp]) -> dict[str, dict[str, int]]:
    histograms = {"all": Counter(), "m_pauli": Counter(), "t_pauli": Counter()}
    for op in ops:
        histograms["all"][str(op.term.weight)] += 1
        histograms[op.op][str(op.term.weight)] += 1
    return {
        op: {weight: histogram[weight] for weight in sorted(histogram, key=int)}
        for op, histogram in histograms.items()
    }


def _high_weight_counts(ops: Sequence[SourcePBCOp], k: int) -> dict[str, int]:
    counts = Counter(op.op for op in ops if op.term.weight > k)
    return {"m_pauli": counts["m_pauli"], "t_pauli": counts["t_pauli"]}


def _high_weight_rotation_runs(ops: Sequence[SourcePBCOp], k: int) -> dict[str, Any]:
    lengths: list[int] = []
    current = 0
    for op in ops:
        if op.op == "t_pauli" and op.term.weight > k:
            current += 1
            continue
        if current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    histogram = Counter(str(length) for length in lengths)
    return {
        "count": len(lengths),
        "max_length": max(lengths, default=0),
        "ops_in_runs": sum(lengths),
        "length_histogram": {
            length: histogram[length] for length in sorted(histogram, key=int)
        },
    }


def _rotation_windows(
    ops: Sequence[SourcePBCOp],
    k: int,
    *,
    window_terms: int = 4,
) -> dict[str, Any]:
    union_widths: list[int] = []
    eligible_widths: list[int] = []
    high_weight_terms = 0
    for index in range(max(0, len(ops) - window_terms + 1)):
        chunk = ops[index : index + window_terms]
        if any(op.op != "t_pauli" for op in chunk):
            continue
        if not any(op.term.weight > k for op in chunk):
            continue
        support = _support_union(chunk)
        union_width = len(support)
        union_widths.append(union_width)
        high_weight_terms += sum(1 for op in chunk if op.term.weight > k)
        if union_width in _SUPPORTED_AI_WINDOW_WIDTHS:
            eligible_widths.append(union_width)
    return {
        "window_terms": window_terms,
        "supported_union_widths": sorted(_SUPPORTED_AI_WINDOW_WIDTHS),
        "candidate_count": len(union_widths),
        "eligible_count": len(eligible_widths),
        "eligible_fraction": _ratio(len(eligible_widths), len(union_widths)),
        "high_weight_terms_in_candidates": high_weight_terms,
        "min_union_width": min(union_widths, default=0),
        "max_union_width": max(union_widths, default=0),
        "mean_union_width": _mean(union_widths),
        "union_width_histogram": _counter_histogram(union_widths),
        "eligible_union_width_histogram": _counter_histogram(eligible_widths),
    }


def _adjacent_support_overlap(ops: Sequence[SourcePBCOp]) -> dict[str, Any]:
    supports = [_support(op.term) for op in ops if op.term.weight > 0]
    overlaps = [
        len(supports[index] & supports[index + 1])
        for index in range(len(supports) - 1)
    ]
    return {
        "pairs": len(overlaps),
        "nonzero_pairs": sum(1 for overlap in overlaps if overlap),
        "total_overlap": sum(overlaps),
        "max_overlap": max(overlaps, default=0),
        "mean_overlap": _mean(overlaps),
    }


def _adjacent_commutation(ops: Sequence[SourcePBCOp]) -> dict[str, Any]:
    pairs = [
        (ops[index].term, ops[index + 1].term) for index in range(len(ops) - 1)
    ]
    noncommuting = sum(1 for left, right in pairs if not left.commutes_with(right))
    return {
        "pairs": len(pairs),
        "commuting_pairs": len(pairs) - noncommuting,
        "noncommuting_pairs": noncommuting,
        "commuting_fraction": _ratio(len(pairs) - noncommuting, len(pairs)),
    }


def _repeated_supports(ops: Sequence[SourcePBCOp]) -> dict[str, Any]:
    counts = Counter(tuple(sorted(_support(op.term))) for op in ops)
    repeated = [count for count in counts.values() if count > 1]
    return {
        "unique": len(counts),
        "count": len(repeated),
        "max_multiplicity": max(repeated, default=0),
    }


def _repeated_terms(ops: Sequence[SourcePBCOp], data_qubits: int) -> dict[str, Any]:
    counts = Counter(op.term.to_full_width(data_qubits) for op in ops)
    repeated = [count for count in counts.values() if count > 1]
    return {
        "unique": len(counts),
        "count": len(repeated),
        "max_multiplicity": max(repeated, default=0),
    }


def _support_union(ops: Sequence[SourcePBCOp]) -> set[str]:
    return {qubit for op in ops for qubit, _pauli in op.term.pairs}


def _support(term: PauliTerm) -> set[str]:
    return {qubit for qubit, _pauli in term.pairs}


def _counter_histogram(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {value: counts[value] for value in sorted(counts, key=int)}


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
