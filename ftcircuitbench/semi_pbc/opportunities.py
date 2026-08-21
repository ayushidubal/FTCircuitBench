from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ftcircuitbench.semi_pbc.pbc_input import SourcePBCOp, parse_pbc_text
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_text
from ftcircuitbench.semi_pbc.reducer import ReducedSourceOp, reduce_measurements

_OPTIMIZATION_MODES = ("none", "local-window", "rotation-dp")


def analyze_pbc_file_opportunities(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return analyze_pbc_text_opportunities(Path(path).read_text(encoding="utf-8"), **kwargs)


def analyze_pbc_text_opportunities(
    text: str,
    *,
    k: int,
    measurement_reducer: str = "peres-galvao-greedy",
    greedy_order: int = 1,
) -> dict[str, Any]:
    if type(k) is not int or k < 1:
        raise ValueError("k must be an integer >= 1")
    program = parse_pbc_text(text)
    source_ops = _reduce_source_ops(
        program.ops,
        measurement_reducer=measurement_reducer,
        greedy_order=greedy_order,
    )
    return {
        "format": "semi-pbc-opportunity-report",
        "version": 1,
        "k": k,
        "measurement_reducer": measurement_reducer,
        "greedy_order": greedy_order,
        "data_qubits": program.data_qubits,
        "input_op_count": len(program.ops),
        "reduced_op_count": len(source_ops),
        "op_counts": _op_counts(source_ops),
        "max_input_weight": max((op.term.weight for op in program.ops), default=0),
        "max_reduced_weight": max((op.term.weight for op in source_ops), default=0),
        "weight_histogram": _weight_histogram(source_ops),
        "high_weight_counts": _high_weight_counts(source_ops, k),
        "rotation_runs": _rotation_run_summary(source_ops, k),
        "adjacent_support_overlap": _adjacent_support_overlap(source_ops),
        "repeated_pauli_supports": _repeated_pauli_supports(source_ops),
        "compile_comparisons": _compile_comparisons(
            text,
            k=k,
            measurement_reducer=measurement_reducer,
            greedy_order=greedy_order,
        ),
    }


def _reduce_source_ops(
    source_ops: Iterable[SourcePBCOp],
    *,
    measurement_reducer: str,
    greedy_order: int,
) -> list[ReducedSourceOp]:
    source_ops = tuple(source_ops)
    if measurement_reducer == "none":
        return [
            ReducedSourceOp(
                id=op.id,
                op=op.op,
                term=op.term,
                source_id=op.source_id,
            )
            for op in source_ops
        ]
    if measurement_reducer != "peres-galvao-greedy":
        raise ValueError("measurement_reducer must be 'none' or 'peres-galvao-greedy'")
    return reduce_measurements(source_ops, greedy_order=greedy_order)


def _op_counts(source_ops: Iterable[ReducedSourceOp]) -> dict[str, int]:
    counts = Counter(op.op for op in source_ops)
    return {op: counts[op] for op in sorted(counts)}


def _weight_histogram(source_ops: Iterable[ReducedSourceOp]) -> dict[str, dict[str, int]]:
    histograms = {
        "all": Counter(),
        "m_pauli": Counter(),
        "t_pauli": Counter(),
    }
    for op in source_ops:
        histograms["all"][str(op.term.weight)] += 1
        histograms[op.op][str(op.term.weight)] += 1
    return {
        op: {weight: histogram[weight] for weight in sorted(histogram, key=int)}
        for op, histogram in histograms.items()
    }


def _high_weight_counts(source_ops: Iterable[ReducedSourceOp], k: int) -> dict[str, int]:
    counts = Counter(
        op.op for op in source_ops if op.op in {"m_pauli", "t_pauli"} and op.term.weight > k
    )
    return {"m_pauli": counts["m_pauli"], "t_pauli": counts["t_pauli"]}


def _rotation_run_summary(
    source_ops: Iterable[ReducedSourceOp],
    k: int,
) -> dict[str, Any]:
    lengths = []
    current = 0
    for op in source_ops:
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


def _adjacent_support_overlap(source_ops: Iterable[ReducedSourceOp]) -> dict[str, Any]:
    supports = [
        {qubit for qubit, _pauli in op.term.pairs}
        for op in source_ops
        if op.term.weight > 0
    ]
    overlaps = [
        len(supports[index] & supports[index + 1])
        for index in range(len(supports) - 1)
    ]
    return {
        "pairs": len(overlaps),
        "nonzero_pairs": sum(1 for overlap in overlaps if overlap),
        "total_overlap": sum(overlaps),
        "max_overlap": max(overlaps, default=0),
    }


def _repeated_pauli_supports(source_ops: Iterable[ReducedSourceOp]) -> dict[str, int]:
    counts = Counter(tuple(qubit for qubit, _pauli in op.term.pairs) for op in source_ops)
    repeated = [count for count in counts.values() if count > 1]
    return {
        "count": len(repeated),
        "max_multiplicity": max(repeated, default=0),
    }


def _compile_comparisons(
    text: str,
    *,
    k: int,
    measurement_reducer: str,
    greedy_order: int,
) -> dict[str, dict[str, Any]]:
    comparisons = {}
    for optimization in _OPTIMIZATION_MODES:
        result = compile_pbc_text(
            text,
            k=k,
            measurement_reducer=measurement_reducer,
            greedy_order=greedy_order,
            optimization=optimization,
            emit_sidecar=False,
        )
        comparisons[optimization] = {
            "output_op_count": result.summary["output_op_count"],
            "latency_weighted_depth": result.summary["latency_weighted_depth"],
            "max_output_weight": result.summary["max_output_weight"],
            "ancilla_count": result.summary["ancilla_count"],
            "max_live_ancillas": result.summary["max_live_ancillas"],
        }
    return comparisons
