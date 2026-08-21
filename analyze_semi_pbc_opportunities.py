from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ftcircuitbench.semi_pbc.opportunities import analyze_pbc_file_opportunities


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze capped semi-PBC optimization opportunities in PBC text."
    )
    parser.add_argument("--pbc", required=True, type=Path)
    parser.add_argument("--k", required=True, type=_positive_int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--measurement-reducer",
        default="peres-galvao-greedy",
        choices=("none", "peres-galvao-greedy"),
    )
    parser.add_argument("--greedy-order", default=1, type=int, choices=(0, 1, 2))
    args = parser.parse_args()

    try:
        report = analyze_pbc_file_opportunities(
            args.pbc,
            k=args.k,
            measurement_reducer=args.measurement_reducer,
            greedy_order=args.greedy_order,
        )
        _write_json(args.out, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.out} "
        f"high_weight_rotations={report['high_weight_counts']['t_pauli']} "
        f"rotation_runs={report['rotation_runs']['count']}"
    )
    return 0


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
