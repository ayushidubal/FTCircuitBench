from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftcircuitbench.semi_pbc.ai_pauli_network import (
    run_ai_pauli_network_synthesis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Try Qiskit's AI Pauli-network synthesis pass on a toy circuit."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backend-name")
    parser.add_argument("--max-threads", type=int)
    parser.add_argument(
        "--pauli",
        action="append",
        dest="signed_paulis",
        default=[],
        help="Signed full-width Pauli string, for example +ZZI. May be repeated.",
    )
    parser.add_argument("--num-qubits", type=int)
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="Replace even if the AI synthesized circuit is not better.",
    )
    args = parser.parse_args()

    signed_paulis = args.signed_paulis or ["+ZZII", "+IZZI", "-IIZZ", "+ZIIZ"]
    num_qubits = args.num_qubits or (len(signed_paulis[0]) - 1)
    coupling_map = [(qubit, qubit + 1) for qubit in range(num_qubits - 1)]
    result = run_ai_pauli_network_synthesis(
        num_qubits=num_qubits,
        signed_paulis=signed_paulis,
        backend_name=args.backend_name,
        coupling_map=coupling_map if args.backend_name is None else None,
        replace_only_if_better=not args.force_replace,
        max_threads=args.max_threads,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"{result['status']} wrote {args.out} "
        f"before={result['before']} after={result['after']} "
        f"error={result['error']!r}"
    )
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
