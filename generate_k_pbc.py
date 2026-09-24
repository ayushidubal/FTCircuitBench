from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from qiskit import QuantumCircuit

from ftcircuitbench.k_pbc.export import semi_pbc_result_to_kpbc
from ftcircuitbench.k_pbc.ir import KPBCHeader, KPBCOp, write_kpbc_jsonl
from ftcircuitbench.k_pbc.segmented_litinski import compile_clifford_t_to_kpbc
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate k-PBC candidate JSONL from PBC text or Clifford+T QASM."
    )
    parser.add_argument("--qasm", type=Path)
    parser.add_argument("--pbc", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", required=True, type=_positive_int)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("pbc-naive-ladder", "ct-segmented-litinski"),
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    try:
        header, ops = _compile_candidate(args)
        write_kpbc_jsonl(args.out, header, ops)
        summary = _summary(args.mode, header, ops)
        if args.summary is not None:
            _write_json(args.summary, summary)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.out} ops={summary['op_count']} "
        f"max_pauli_weight={summary['max_pauli_weight']} mode={args.mode}"
    )
    return 0


def _compile_candidate(args: argparse.Namespace) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    if args.mode == "ct-segmented-litinski":
        if args.qasm is None or args.pbc is not None:
            raise ValueError("ct-segmented-litinski requires --qasm and no --pbc")
        return _compile_segmented_litinski(args.qasm, k=args.k)
    if args.mode == "pbc-naive-ladder":
        if args.pbc is None or args.qasm is not None:
            raise ValueError("pbc-naive-ladder requires --pbc and no --qasm")
        return _compile_pbc_naive_ladder(args.pbc, k=args.k)
    raise ValueError(f"unsupported mode {args.mode!r}")


def _compile_segmented_litinski(
    qasm_path: Path, *, k: int
) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    return compile_clifford_t_to_kpbc(circuit, k=k)


def _compile_pbc_naive_ladder(
    pbc_path: Path, *, k: int
) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    result = compile_pbc_file(
        pbc_path,
        k=k,
        measurement_reducer="none",
        optimization="none",
        emit_sidecar=False,
    )
    return semi_pbc_result_to_kpbc(result)


def _summary(mode: str, header: KPBCHeader, ops: tuple[KPBCOp, ...]) -> dict[str, object]:
    op_counts = Counter(op.op for op in ops)
    return {
        "mode": mode,
        "k": header.k,
        "data_qubits": header.data_qubits,
        "op_count": len(ops),
        "op_counts": dict(sorted(op_counts.items())),
        "max_pauli_weight": max(
            (op.term.weight for op in ops if op.term is not None),
            default=0,
        ),
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--k must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--k must be an integer >= 1")
    return parsed


def _write_json(path: Path, data: object) -> None:
    output_path = Path(path)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as output:
            temp_path = Path(output.name)
            output.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        temp_path.replace(output_path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
