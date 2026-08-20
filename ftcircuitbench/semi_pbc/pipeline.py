from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp, validate_program
from ftcircuitbench.semi_pbc.lowering import (
    lower_pauli_measurement,
    lower_pauli_rotation,
)
from ftcircuitbench.semi_pbc.pbc_input import SourcePBCOp, parse_pbc_text
from ftcircuitbench.semi_pbc.reducer import ReducedSourceOp, reduce_measurements
from ftcircuitbench.semi_pbc.schedule import compute_summary


@dataclass(frozen=True)
class CompileResult:
    header: SemiPBCHeader
    ops: tuple[SemiPBCOp, ...]
    summary: dict[str, Any]
    sidecar: dict[str, Any] | None = None

    def jsonl_records(self) -> list[dict[str, Any]]:
        return [self.header.to_record(), *(op.to_record() for op in self.ops)]


def compile_pbc_text(
    text: str,
    *,
    k: int,
    objective: str = "latency-depth",
    measurement_reducer: str = "peres-galvao-greedy",
    greedy_order: int = 1,
    rotation_lowering: str = "parity-network",
    measurement_lowering: str = "coherent-parity",
    ancilla_budget: int | None = None,
    emit_sidecar: bool = True,
) -> CompileResult:
    _validate_options(
        k=k,
        objective=objective,
        measurement_reducer=measurement_reducer,
        greedy_order=greedy_order,
        rotation_lowering=rotation_lowering,
        measurement_lowering=measurement_lowering,
        ancilla_budget=ancilla_budget,
    )
    program = parse_pbc_text(text)
    header = SemiPBCHeader(k=k, data_qubits=program.data_qubits)
    source_ops = _reduce_source_ops(
        program.ops,
        measurement_reducer=measurement_reducer,
        greedy_order=greedy_order,
    )
    ops, provenance = _lower_source_ops(
        source_ops,
        k=k,
        ancilla_budget=ancilla_budget,
    )
    validate_program(header, ops)
    summary = compute_summary(
        header,
        ops,
        input_op_count=len(program.ops),
        max_input_weight=max((op.term.weight for op in program.ops), default=0),
    )
    _enforce_ancilla_budget(summary, ancilla_budget)
    return CompileResult(
        header=header,
        ops=tuple(ops),
        summary=summary,
        sidecar=_build_sidecar(k, provenance) if emit_sidecar else None,
    )


def compile_pbc_file(path: str | Path, **kwargs: Any) -> CompileResult:
    return compile_pbc_text(Path(path).read_text(encoding="utf-8"), **kwargs)


def _validate_options(
    *,
    k: int,
    objective: str,
    measurement_reducer: str,
    greedy_order: int,
    rotation_lowering: str,
    measurement_lowering: str,
    ancilla_budget: int | None,
) -> None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be an integer >= 1")
    if objective != "latency-depth":
        raise ValueError("objective must be 'latency-depth'")
    if measurement_reducer not in {"none", "peres-galvao-greedy"}:
        raise ValueError("measurement_reducer must be 'none' or 'peres-galvao-greedy'")
    if greedy_order not in {0, 1, 2}:
        raise ValueError("greedy_order must be 0, 1, or 2")
    if rotation_lowering != "parity-network":
        raise ValueError("rotation_lowering must be 'parity-network'")
    if measurement_lowering != "coherent-parity":
        raise ValueError("measurement_lowering must be 'coherent-parity'")
    if ancilla_budget is not None and (
        type(ancilla_budget) is not int or ancilla_budget < 0
    ):
        raise ValueError("ancilla_budget must be None or a non-negative integer")


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
    return reduce_measurements(source_ops, greedy_order=greedy_order)


def _lower_source_ops(
    source_ops: Iterable[ReducedSourceOp],
    *,
    k: int,
    ancilla_budget: int | None,
) -> tuple[list[SemiPBCOp], list[dict[str, Any]]]:
    ops: list[SemiPBCOp] = []
    provenance: list[dict[str, Any]] = []
    next_id = 0
    next_ancilla = 0
    next_classical = 0

    for source_op in source_ops:
        source_result = f"src{source_op.id}"
        if source_op.op == "t_pauli":
            lowered = lower_pauli_rotation(
                next_id,
                source_op.term,
                k,
                source_id=source_op.source_id,
            )
        elif source_op.op == "m_pauli":
            if source_op.term.weight == 0:
                lowered = [
                    SemiPBCOp.xor(
                        next_id,
                        target=source_result,
                        terms=source_op.result_terms,
                        const=source_op.result_const,
                        source_id=source_op.source_id,
                    )
                ]
                ops.extend(lowered)
                provenance.extend(_provenance_records(source_op, lowered))
                next_id = lowered[-1].id + 1
                continue
            measurement = lower_pauli_measurement(
                start_id=next_id,
                term=source_op.term,
                k=k,
                result=source_result,
                source_id=source_op.source_id,
                next_ancilla=next_ancilla,
                next_classical=next_classical,
            )
            lowered = _apply_reducer_result_mapping(
                measurement.ops,
                result_terms=source_op.result_terms,
                result_const=source_op.result_const,
            )
            next_ancilla = measurement.next_ancilla
            next_classical = measurement.next_classical
        else:
            raise ValueError(f"unsupported PBC operation {source_op.op!r}")

        ops.extend(lowered)
        provenance.extend(_provenance_records(source_op, lowered))
        next_id = lowered[-1].id + 1

    return ops, provenance


def _enforce_ancilla_budget(
    summary: dict[str, Any], ancilla_budget: int | None
) -> None:
    if ancilla_budget is None:
        return
    max_live = summary["max_live_ancillas"]
    if max_live > ancilla_budget:
        raise ValueError(
            "ancilla budget exceeded: "
            f"peak live ancillas {max_live} > budget {ancilla_budget}"
        )


def _apply_reducer_result_mapping(
    ops: list[SemiPBCOp],
    *,
    result_terms: tuple[str, ...],
    result_const: int,
) -> list[SemiPBCOp]:
    if not result_terms and result_const == 0:
        return ops
    final = ops[-1]
    if final.op != "xor":
        raise ValueError("measurement lowering did not end with source-result xor")
    combined_terms = _xor_terms(final.terms, result_terms)
    ops[-1] = SemiPBCOp.xor(
        final.id,
        target=final.target,
        terms=combined_terms,
        const=final.const ^ result_const,
        source_id=final.source_id,
    )
    return ops


def _xor_terms(*groups: tuple[str, ...]) -> tuple[str, ...]:
    active: set[str] = set()
    for group in groups:
        for term in group:
            if term in active:
                active.remove(term)
            else:
                active.add(term)
    return tuple(sorted(active, key=_classical_sort_key))


def _classical_sort_key(term: str) -> tuple[int, int, str]:
    if term.startswith("c") and term[1:].isdigit():
        return (0, int(term[1:]), "")
    if term.startswith("src") and term[3:].isdigit():
        return (1, int(term[3:]), "")
    return (2, 0, term)


def _provenance_records(
    source_op: ReducedSourceOp, lowered: list[SemiPBCOp]
) -> list[dict[str, Any]]:
    gadget_id = f"g{source_op.id}"
    records = []
    for op in lowered:
        record: dict[str, Any] = {
            "id": op.id,
            "op": op.op,
            "source_id": source_op.source_id,
        }
        if source_op.used_source_ids:
            record["used_source_ids"] = list(source_op.used_source_ids)
        if (
            source_op.used_source_ids
            or source_op.result_terms
            or source_op.result_const != 0
        ):
            record["result_terms"] = list(source_op.result_terms)
            record["result_const"] = source_op.result_const
        if len(lowered) > 1:
            record["gadget_id"] = gadget_id
        records.append(record)
    return records


def _build_sidecar(k: int, provenance: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "semi-pbc-sidecar",
        "version": 1,
        "k": k,
        "provenance": provenance,
    }
