from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ftcircuitbench.semi_pbc.ai_pauli_network import (
    PauliNetworkWindow,
    analyze_ai_pauli_window_trajectory,
    build_ai_trajectory_prefix_ops,
)
from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp, validate_program
from ftcircuitbench.semi_pbc.lowering import (
    lower_pauli_measurement,
    lower_pauli_rotation,
)
from ftcircuitbench.semi_pbc.optimizer import (
    cancel_adjacent_inverse_cliffords,
    choose_retained_block,
    optimize_rotation_run,
)
from ftcircuitbench.semi_pbc.pauli import PauliTerm
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
    optimization: str = "none",
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
        optimization=optimization,
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
        data_qubits=program.data_qubits,
        k=k,
        ancilla_budget=ancilla_budget,
        optimization=optimization,
    )
    validate_program(header, ops)
    summary = compute_summary(
        header,
        ops,
        input_op_count=len(program.ops),
        max_input_weight=max((op.term.weight for op in program.ops), default=0),
    )
    summary["optimization"] = optimization
    _enforce_ancilla_budget(summary, ancilla_budget)
    return CompileResult(
        header=header,
        ops=tuple(ops),
        summary=summary,
        sidecar=_build_sidecar(k, optimization, provenance) if emit_sidecar else None,
    )


def compile_pbc_file(path: str | Path, **kwargs: Any) -> CompileResult:
    return compile_pbc_text(Path(path).read_text(encoding="utf-8"), **kwargs)


def _validate_options(
    *,
    k: int,
    objective: str,
    measurement_reducer: str,
    greedy_order: int,
    optimization: str,
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
    if optimization not in {
        "none",
        "local-window",
        "rotation-dp",
        "ai-trajectory-prefix",
    }:
        raise ValueError(
            "optimization must be 'none', 'local-window', 'rotation-dp', "
            "or 'ai-trajectory-prefix'"
        )
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
    data_qubits: int,
    k: int,
    ancilla_budget: int | None,
    optimization: str,
) -> tuple[list[SemiPBCOp], list[dict[str, Any]]]:
    source_ops = tuple(source_ops)
    ops: list[SemiPBCOp] = []
    provenance: list[dict[str, Any]] = []
    next_id = 0
    next_ancilla = 0
    next_classical = 0

    index = 0
    while index < len(source_ops):
        ai_prefix = _try_ai_trajectory_prefix_window(
            source_ops,
            index=index,
            data_qubits=data_qubits,
            start_id=next_id,
            k=k,
            optimization=optimization,
        )
        if ai_prefix is not None:
            optimized_ops, source_ops_by_output_id, run_end = ai_prefix
            ops.extend(optimized_ops)
            provenance.extend(
                _provenance_records_from_mapping(
                    optimized_ops,
                    source_ops_by_output_id,
                )
            )
            next_id = optimized_ops[-1].id + 1
            index = run_end
            continue

        run_end = _rotation_dp_run_end(source_ops, index, k, optimization)
        if run_end is not None:
            run = source_ops[index:run_end]
            optimized = optimize_rotation_run(run, start_id=next_id, k=k)
            ops.extend(optimized.ops)
            provenance.extend(
                _provenance_records_from_mapping(
                    optimized.ops,
                    optimized.source_ops_by_output_id,
                )
            )
            next_id = optimized.ops[-1].id + 1
            index = run_end
            continue

        source_op = source_ops[index]
        source_result = f"src{source_op.id}"
        retained_qubits = _retained_qubits_for_source_op(
            source_ops,
            index=index,
            k=k,
            optimization=optimization,
        )
        if source_op.op == "t_pauli":
            lowered = lower_pauli_rotation(
                next_id,
                source_op.term,
                k,
                source_id=source_op.source_id,
                retained_qubits=retained_qubits,
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
                index += 1
                continue
            measurement = lower_pauli_measurement(
                start_id=next_id,
                term=source_op.term,
                k=k,
                result=source_result,
                source_id=source_op.source_id,
                next_ancilla=next_ancilla,
                next_classical=next_classical,
                retained_qubits=retained_qubits,
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
        index += 1

    if optimization in {"local-window", "rotation-dp", "ai-trajectory-prefix"}:
        ops = cancel_adjacent_inverse_cliffords(ops)
        provenance = _filter_provenance(provenance, ops)
    if optimization == "rotation-dp":
        fallback_ops, fallback_provenance = _lower_source_ops(
            source_ops,
            data_qubits=data_qubits,
            k=k,
            ancilla_budget=ancilla_budget,
            optimization="local-window",
        )
        if len(fallback_ops) < len(ops):
            return fallback_ops, fallback_provenance

    return ops, provenance


def _rotation_dp_run_end(
    source_ops: tuple[ReducedSourceOp, ...],
    index: int,
    k: int,
    optimization: str,
) -> int | None:
    if optimization != "rotation-dp":
        return None
    if not _is_high_weight_rotation(source_ops[index], k):
        return None
    run_end = index + 1
    while run_end < len(source_ops) and _is_high_weight_rotation(
        source_ops[run_end], k
    ):
        run_end += 1
    return run_end if run_end - index > 1 else None


def _try_ai_trajectory_prefix_window(
    source_ops: tuple[ReducedSourceOp, ...],
    *,
    index: int,
    data_qubits: int,
    start_id: int,
    k: int,
    optimization: str,
) -> tuple[list[SemiPBCOp], dict[int, ReducedSourceOp], int] | None:
    if optimization != "ai-trajectory-prefix":
        return None
    window_terms = 4
    run_end = index + window_terms
    chunk = source_ops[index:run_end]
    if len(chunk) != window_terms:
        return None
    if any(source_op.op != "t_pauli" for source_op in chunk):
        return None
    if not any(source_op.term.weight > k for source_op in chunk):
        return None

    window = _ai_window_from_source_ops(
        chunk,
        data_qubits=data_qubits,
        index=index,
    )
    if window is None:
        return None
    try:
        analysis = analyze_ai_pauli_window_trajectory(
            window,
            k=k,
            deterministic=True,
        )
        if not _is_usable_ai_prefix_analysis(analysis):
            return None
        prefix_ops = build_ai_trajectory_prefix_ops(
            start_id=start_id,
            trajectory=analysis,
            k=k,
            original_qubits=window.original_qubits,
            source_ids=window.source_ids,
        )
    except Exception:  # noqa: BLE001 - optional AI path falls back on any failure.
        return None

    source_ops_by_output_id = {
        output_id: chunk[source_index]
        for output_id, source_index in prefix_ops.source_indices_by_output_id.items()
    }
    return prefix_ops.ops, source_ops_by_output_id, run_end


def _ai_window_from_source_ops(
    source_ops: tuple[ReducedSourceOp, ...],
    *,
    data_qubits: int,
    index: int,
) -> PauliNetworkWindow | None:
    original_qubits = tuple(
        sorted(
            {
                int(qubit[1:])
                for op in source_ops
                for qubit, _pauli in op.term.pairs
                if qubit.startswith("q")
            }
        )
    )
    coupling_map = _line_coupling_map_for_ai_window(len(original_qubits))
    if coupling_map is None:
        return None
    reindex = {qubit: local for local, qubit in enumerate(original_qubits)}
    return PauliNetworkWindow(
        source_path="<compile_pbc_text>",
        start_op_id=source_ops[0].id,
        stop_op_id=source_ops[-1].id,
        source_ids=tuple(op.source_id for op in source_ops),
        original_qubits=original_qubits,
        num_qubits=len(original_qubits),
        signed_paulis=tuple(
            _compress_source_pauli(op.term.to_full_width(data_qubits), reindex)
            for op in source_ops
        ),
        total_pauli_weight=sum(op.term.weight for op in source_ops),
        multi_qubit_terms=sum(op.term.weight > 1 for op in source_ops),
        topology="line",
        coupling_map=tuple(coupling_map),
    )


def _line_coupling_map_for_ai_window(
    num_qubits: int,
) -> tuple[tuple[int, int], ...] | None:
    if num_qubits not in {4, 5, 6}:
        return None
    return tuple((qubit, qubit + 1) for qubit in range(num_qubits - 1))


def _compress_source_pauli(signed_pauli: str, reindex: dict[int, int]) -> str:
    compressed = ["I"] * len(reindex)
    for original_index, compressed_index in reindex.items():
        compressed[compressed_index] = signed_pauli[original_index + 1]
    return signed_pauli[0] + "".join(compressed)


def _is_usable_ai_prefix_analysis(result: dict[str, Any]) -> bool:
    if result.get("status") != "ok":
        return False
    if "nan" in str(result.get("solver_output_excerpt", "")).lower():
        return False
    return all(
        key in result
        for key in (
            "decoded_solution",
            "gateset",
            "replay_terms",
            "rotation_angle_signs",
            "k_terminal_prefix_length",
        )
    )


def _is_high_weight_rotation(source_op: ReducedSourceOp, k: int) -> bool:
    return source_op.op == "t_pauli" and source_op.term.weight > k


def _retained_qubits_for_source_op(
    source_ops: tuple[ReducedSourceOp, ...],
    *,
    index: int,
    k: int,
    optimization: str,
) -> tuple[str, ...] | None:
    source_op = source_ops[index]
    if (
        optimization not in {"local-window", "rotation-dp", "ai-trajectory-prefix"}
        or source_op.term.weight <= k
    ):
        return None
    block = choose_retained_block(
        source_op.term,
        k=k,
        neighbor_terms=_neighbor_terms(source_ops, index),
    )
    return block.retained_qubits


def _neighbor_terms(
    source_ops: tuple[ReducedSourceOp, ...],
    index: int,
) -> tuple[PauliTerm, ...]:
    terms = []
    for prior in range(index - 1, -1, -1):
        if source_ops[prior].term.weight > 0:
            terms.append(source_ops[prior].term)
            break
    for following in range(index + 1, len(source_ops)):
        if source_ops[following].term.weight > 0:
            terms.append(source_ops[following].term)
            break
    return tuple(terms)


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
    include_gadget = len(lowered) > 1
    return [
        _provenance_record(source_op, op, include_gadget=include_gadget)
        for op in lowered
    ]


def _provenance_records_from_mapping(
    ops: list[SemiPBCOp],
    source_ops_by_output_id: dict[int, ReducedSourceOp],
) -> list[dict[str, Any]]:
    source_output_counts: dict[int, int] = {}
    for source_op in source_ops_by_output_id.values():
        source_output_counts[source_op.id] = (
            source_output_counts.get(source_op.id, 0) + 1
        )
    return [
        _provenance_record(
            source_ops_by_output_id[op.id],
            op,
            include_gadget=source_output_counts[source_ops_by_output_id[op.id].id] > 1,
        )
        for op in ops
    ]


def _provenance_record(
    source_op: ReducedSourceOp,
    op: SemiPBCOp,
    *,
    include_gadget: bool,
) -> dict[str, Any]:
    gadget_id = f"g{source_op.id}"
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
    if include_gadget:
        record["gadget_id"] = gadget_id
    return record


def _filter_provenance(
    provenance: list[dict[str, Any]],
    ops: list[SemiPBCOp],
) -> list[dict[str, Any]]:
    retained_ids = {op.id for op in ops}
    return [record for record in provenance if record["id"] in retained_ids]


def _build_sidecar(
    k: int,
    optimization: str,
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "semi-pbc-sidecar",
        "version": 1,
        "k": k,
        "optimization": optimization,
        "provenance": provenance,
    }
