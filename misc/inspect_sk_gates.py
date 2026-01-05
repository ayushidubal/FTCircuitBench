#!/usr/bin/env python3
"""
inspect_sk_gates.py

Run the Solovay–Kitaev (SK) transpiler on one or more QASM files and print:
- the unique gate names present in the resulting circuit
- per-gate counts
- basic validation vs the expected SK basis

Usage examples:
  python inspect_sk_gates.py \
      --degree 1 \
      qasm/qft/qft_18q.qasm qasm/adder/adder_10q.qasm

  python inspect_sk_gates.py --degree 2 --canonicalize \
      qasm/hhl/hhl_7q.qasm

If no QASM paths are provided, a small default set is used.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _ensure_repo_on_sys_path() -> None:
    """Ensure the repository root (containing ftcircuitbench/) is importable."""
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_sys_path()

from qiskit import QuantumCircuit, transpile  # noqa: E402

from ftcircuitbench import load_qasm_circuit  # noqa: E402
from ftcircuitbench.transpilers import transpile_to_solovay_kitaev_clifford_t  # noqa: E402

# This is the declared target basis for SK in our codebase
SK_BASIS = ["cx", "h", "s", "t", "tdg", "x", "z", "sdg"]


def _format_counts(counter: collections.Counter) -> str:
    items = sorted(counter.items(), key=lambda kv: (kv[0] != "cx", kv[0]))
    return ", ".join(f"{name}={count}" for name, count in items)


def analyze_gate_set(circuit: QuantumCircuit) -> Tuple[set[str], collections.Counter]:
    gate_names: set[str] = set()
    counts: collections.Counter = collections.Counter()
    for instruction in circuit.data:
        name = instruction.operation.name
        gate_names.add(name)
        counts[name] += 1
    return gate_names, counts


def run_sk_once(
    qasm_path: Path,
    recursion_degree: int,
    canonicalize: bool,
) -> Tuple[QuantumCircuit, set[str], collections.Counter]:
    # Load
    qc = load_qasm_circuit(str(qasm_path), is_file=True)

    # Transpile via SK
    sk_qc = transpile_to_solovay_kitaev_clifford_t(qc, recursion_degree=recursion_degree)

    # Optionally canonicalize to the declared SK basis to see if composition changes
    if canonicalize:
        sk_qc = transpile(sk_qc, basis_gates=SK_BASIS, optimization_level=0)

    names, counts = analyze_gate_set(sk_qc)
    return sk_qc, names, counts


def default_qasm_paths(repo_root: Path) -> List[Path]:
    candidates = [
        repo_root / "qasm/qft/qft_18q.qasm",
        repo_root / "qasm/adder/adder_10q.qasm",
        repo_root / "qasm/hhl/hhl_7q.qasm",
        repo_root / "qasm/qpe/qpe_H2_1_0_12q.qasm",
    ]
    return [p for p in candidates if p.exists()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SK transpiler on QASM files and print resulting gate sets",
    )
    parser.add_argument(
        "qasm_paths",
        nargs="*",
        help="Paths to QASM files to analyze. If omitted, a small default set is used.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        dest="degree",
        help="Solovay–Kitaev recursion degree (default: 1)",
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help=(
            "After SK, run a final transpile to the declared SK basis "
            "to see if gate composition changes."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent

    qasm_files: List[Path]
    if args.qasm_paths:
        qasm_files = [Path(p).resolve() for p in args.qasm_paths]
    else:
        qasm_files = default_qasm_paths(repo_root)
        if not qasm_files:
            print("No default QASM files found. Please pass paths explicitly.")
            return 1

    results = []
    for qasm_path in qasm_files:
        try:
            sk_qc, names, counts = run_sk_once(
                qasm_path=qasm_path,
                recursion_degree=args.degree,
                canonicalize=bool(args.canonicalize),
            )
            extra = {}
            unexpected = sorted(set(names) - set(SK_BASIS))
            if unexpected:
                extra["unexpected_gates"] = unexpected
            results.append(
                {
                    "file": str(qasm_path),
                    "num_qubits": sk_qc.num_qubits,
                    "depth": sk_qc.depth(),
                    "unique_gates": sorted(names),
                    "counts": dict(sorted(counts.items())),
                    "t_family": {k: counts.get(k, 0) for k in ("t", "tdg")},
                    "clifford_subset": {
                        k: counts.get(k, 0) for k in ("h", "s", "sdg", "x", "z", "cx")
                    },
                    **extra,
                }
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                {
                    "file": str(qasm_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if args.json_out:
        print(json.dumps({"degree": args.degree, "results": results}, indent=2))
    else:
        print(
            f"SK results (degree={args.degree}, canonicalize={bool(args.canonicalize)}):"
        )
        for r in results:
            print("\n==>", r.get("file"))
            if "error" in r:
                print("  ERROR:", r["error"])
                continue
            print("  qubits:", r["num_qubits"], " depth:", r["depth"])
            print("  unique gates:", ", ".join(r["unique_gates"]))
            print("  counts:", _format_counts(collections.Counter(r["counts"])))
            tf = r.get("t_family", {})
            print("  T family:", ", ".join(f"{k}={v}" for k, v in tf.items()))
            cl_sub = r.get("clifford_subset", {})
            print(
                "  Clifford subset:",
                ", ".join(f"{k}={v}" for k, v in cl_sub.items() if v),
            )
            if r.get("unexpected_gates"):
                print("  unexpected gates:", ", ".join(r["unexpected_gates"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
