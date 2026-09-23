from __future__ import annotations

from ftcircuitbench.k_pbc.ir import KPBCHeader, KPBCOp
from ftcircuitbench.semi_pbc.pipeline import CompileResult


def semi_pbc_result_to_kpbc(
    result: CompileResult,
) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    header = KPBCHeader(k=result.header.k, data_qubits=result.header.data_qubits)
    ops: list[KPBCOp] = []
    for op in result.ops:
        if op.op in {"alloc", "release"}:
            continue
        if op.op in {"h", "s", "sdg", "cx"}:
            ops.append(KPBCOp.clifford(op.id, op.op, op.qubits, source_id=op.source_id))
        elif op.op == "t_pauli":
            ops.append(KPBCOp.t_pauli(op.id, op.term, source_id=op.source_id))
        elif op.op == "m_pauli":
            ops.append(KPBCOp.m_pauli(op.id, op.term, op.result, source_id=op.source_id))
        elif op.op == "xor":
            ops.append(
                KPBCOp.xor(
                    op.id,
                    target=op.target,
                    terms=op.terms,
                    const=op.const,
                    source_id=op.source_id,
                )
            )
        else:
            raise ValueError(f"unsupported semi-PBC op {op.op!r}")
    header.validate_ops(ops)
    return header, tuple(ops)
