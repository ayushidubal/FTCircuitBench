from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp

_PHASE1_LATENCIES = {
    "h": 1,
    "s": 1,
    "sdg": 1,
    "cx": 1,
    "t_pauli": 1,
    "m_pauli": 1,
    "alloc": 0,
    "release": 0,
    "xor": 0,
}


def compute_summary(
    header: SemiPBCHeader,
    ops: Iterable[SemiPBCOp],
    *,
    input_op_count: int,
    max_input_weight: int,
) -> dict[str, Any]:
    op_list = list(ops)
    max_output_weight = max(
        (op.term.weight for op in op_list if op.op in {"t_pauli", "m_pauli"}),
        default=0,
    )
    return {
        "record_type": "summary",
        "k": header.k,
        "data_qubits": header.data_qubits,
        "input_op_count": input_op_count,
        "output_op_count": len(op_list),
        "max_input_weight": max_input_weight,
        "max_output_weight": max_output_weight,
        "ancilla_count": len({op.qubit for op in op_list if op.op == "alloc"}),
        "max_live_ancillas": _max_live_ancillas(op_list),
        "latency_weighted_depth": sum(_op_latency(op) for op in op_list),
        "latency_model": "phase1_default",
    }


def _op_latency(op: SemiPBCOp) -> int:
    try:
        return _PHASE1_LATENCIES[op.op]
    except KeyError as exc:
        raise ValueError(f"unsupported semi-PBC operation {op.op!r}") from exc


def _max_live_ancillas(ops: Iterable[SemiPBCOp]) -> int:
    active: set[str] = set()
    peak = 0
    for op in ops:
        if op.op == "alloc":
            active.add(op.qubit)
            peak = max(peak, len(active))
        elif op.op == "release":
            active.discard(op.qubit)
    return peak
