from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ftcircuitbench.semi_pbc.structure import compare_pbc_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare structural features of two nwqec PBC files."
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    parser.add_argument("--k", required=True, type=_positive_int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        report = compare_pbc_files(
            args.left,
            args.right,
            k=args.k,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        _write_json(args.out, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.out} "
        f"left_ai_windows={report['left']['rotation_windows']['eligible_count']} "
        f"right_ai_windows={report['right']['rotation_windows']['eligible_count']}"
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
