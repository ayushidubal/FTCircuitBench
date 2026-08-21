from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ftcircuitbench.semi_pbc.ir import write_jsonl
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile nwqec PBC text to semi-PBC JSONL."
    )
    parser.add_argument("--pbc", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--emit-sidecar", dest="emit_sidecar", action="store_true")
    parser.set_defaults(emit_sidecar=True)
    parser.add_argument("--k", required=True, type=_positive_int)
    parser.add_argument("--objective", default="latency-depth", choices=("latency-depth",))
    parser.add_argument(
        "--measurement-reducer",
        default="peres-galvao-greedy",
        choices=("none", "peres-galvao-greedy"),
    )
    parser.add_argument("--greedy-order", default=1, type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--optimization",
        default="none",
        choices=("none", "local-window", "rotation-dp"),
    )
    parser.add_argument(
        "--rotation-lowering",
        default="parity-network",
        choices=("parity-network",),
    )
    parser.add_argument(
        "--measurement-lowering",
        default="coherent-parity",
        choices=("coherent-parity",),
    )
    parser.add_argument(
        "--ancilla-budget",
        default=None,
        type=_parse_ancilla_budget,
        metavar="unlimited|N",
    )
    args = parser.parse_args()

    emit_sidecar = args.emit_sidecar or args.sidecar is not None
    try:
        result = compile_pbc_file(
            args.pbc,
            k=args.k,
            objective=args.objective,
            measurement_reducer=args.measurement_reducer,
            greedy_order=args.greedy_order,
            optimization=args.optimization,
            rotation_lowering=args.rotation_lowering,
            measurement_lowering=args.measurement_lowering,
            ancilla_budget=args.ancilla_budget,
            emit_sidecar=emit_sidecar,
        )

        write_jsonl(args.out, result.header, result.ops)
        if args.summary is not None:
            _write_json(args.summary, result.summary)
        if emit_sidecar:
            sidecar_path = args.sidecar or args.out.with_suffix(
                f"{args.out.suffix}.sidecar.json"
            )
            _write_json(sidecar_path, result.sidecar)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.out} ops={len(result.ops)} "
        f"max_output_weight={result.summary['max_output_weight']} "
        f"latency_weighted_depth={result.summary['latency_weighted_depth']}"
    )
    return 0


def _parse_ancilla_budget(value: str) -> int | None:
    if value == "unlimited":
        return None
    try:
        budget = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ancilla-budget must be 'unlimited' or a non-negative integer"
        ) from exc
    if budget < 0:
        raise argparse.ArgumentTypeError(
            "--ancilla-budget must be 'unlimited' or a non-negative integer"
        )
    return budget


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
