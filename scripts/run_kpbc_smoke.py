from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_k_pbc import main as generate_main


DEFAULT_DECODER_MODEL = {
    "name": "unit-line",
    "t0": 0.0,
    "alpha": 1.0,
    "beta_l": 1.0,
    "beta_d": 1.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local k-PBC smoke workflow.")
    parser.add_argument("--qasm", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--k", required=True, type=_positive_int)
    parser.add_argument(
        "--compiler-mode", required=True, choices=("segmented-litinski", "naive")
    )
    parser.add_argument("--router-bin", required=True, type=Path)
    parser.add_argument("--router-arg", action="append", default=[])
    parser.add_argument("--tracegen-python", type=Path)
    parser.add_argument("--decoder-model", type=Path)
    parser.add_argument("--d", required=True, type=_positive_int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run.log"
    candidate_path = args.out_dir / "candidate.kpbc.jsonl"
    routed_path = args.out_dir / "routed.json"
    trace_path = args.out_dir / "trace.jsonl"
    summary_path = args.out_dir / "summary.json"

    log_lines: list[str] = []
    try:
        _generate_candidate(args, candidate_path, log_lines)
        _run_router(args, candidate_path, routed_path, log_lines)
        routed_ops = json.loads(routed_path.read_text(encoding="utf-8"))
        events, summary = _time_routed_ops(
            routed_ops,
            decoder_config=_load_decoder_config(args.decoder_model),
            d=args.d,
        )
        _write_trace_jsonl(trace_path, events, summary)
        _write_json(summary_path, summary)
        log_lines.append(f"wrote {trace_path}")
        log_lines.append(f"wrote {summary_path}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        log_lines.append(f"error: {exc}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        parser.error(str(exc))

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir}")
    return 0


def _generate_candidate(args: argparse.Namespace, out_path: Path, log_lines: list[str]) -> None:
    argv = [
        "generate_k_pbc.py",
        "--qasm",
        str(args.qasm),
        "--out",
        str(out_path),
        "--k",
        str(args.k),
        "--mode",
        args.compiler_mode,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        generate_main()
    finally:
        sys.argv = old_argv
    log_lines.append(f"wrote {out_path}")


def _run_router(
    args: argparse.Namespace, candidate_path: Path, routed_path: Path, log_lines: list[str]
) -> None:
    command = [
        str(args.router_bin),
        *args.router_arg,
        "--input",
        str(candidate_path),
        "--output",
        str(routed_path),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    log_lines.append(" ".join(command))
    log_lines.append(f"wrote {routed_path}")


def _time_routed_ops(
    routed_ops: list[dict[str, Any]], *, decoder_config: dict[str, Any], d: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        tracegen_root = Path("/Users/ayushidubal/qmem/.worktrees/tracegen")
        if tracegen_root.exists():
            sys.path.insert(0, str(tracegen_root))
        from qmem_utils.decoder_model import PowerLawDecoderModel
        from qmem_utils.kpbc_tracegen import time_routed_kpbc_ops

        model = PowerLawDecoderModel(**decoder_config)
        return time_routed_kpbc_ops(routed_ops, decoder_model=model, d=d)
    except ImportError:
        return _fallback_time_routed_ops(routed_ops, decoder_config=decoder_config, d=d)


def _fallback_time_routed_ops(
    routed_ops: list[dict[str, Any]], *, decoder_config: dict[str, Any], d: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = []
    t = 0
    for op in routed_ops:
        duration = int(
            round(
                float(decoder_config["t0"])
                + float(decoder_config["alpha"])
                * (float(op["routed_l"]) ** float(decoder_config["beta_l"]))
                * (float(d) ** float(decoder_config["beta_d"]))
            )
        )
        events.append(
            {
                "t": t,
                "end": t + duration,
                "op_time": duration,
                "qubits": list(op.get("logical_qubits", [])),
                "op_name": op["op"],
                "op_type": op["op"],
                "routed_l": op.get("routed_l"),
                "routed_area": op.get("routed_area"),
            }
        )
        t += duration
    return events, {
        "decoder_model": decoder_config["name"],
        "makespan": t,
        "model_extrapolation_count": 0,
    }


def _load_decoder_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_DECODER_MODEL)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_trace_jsonl(
    path: Path, events: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event, separators=(",", ":")) + "\n")
        output.write(
            json.dumps({"record_type": "summary", **summary}, separators=(",", ":"))
            + "\n"
        )


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected integer >= 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
