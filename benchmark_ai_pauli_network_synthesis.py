from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qiskit import qasm2

from ftcircuitbench.semi_pbc.ai_pauli_network import (
    build_pauli_network_circuit,
    build_rotation_region_circuit,
    circuits_equivalent,
    extract_rotation_regions,
    extract_supported_rotation_windows,
    run_ai_pauli_network_synthesis,
    run_ai_pauli_network_synthesis_on_circuit,
)
from ftcircuitbench.semi_pbc.pbc_input import parse_pbc_file


def benchmark_pbc_files(
    *,
    paths: list[Path],
    out_dir: Path,
    max_windows_per_file: int,
    max_files: int | None,
    window_terms: int,
    topology: str = "line",
    max_threads: int | None = None,
    include_qasm: bool = False,
    selection: str = "ranked-window",
    max_regions_per_file: int = 4,
    min_region_terms: int = 4,
    max_region_terms: int | None = None,
    capture_pass_output: bool = True,
    emit_candidates: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    candidate_path = out_dir / "candidate_replacements.jsonl"

    summary: dict[str, Any] = {
        "files_seen": 0,
        "files_with_windows": 0,
        "files_with_regions": 0,
        "windows_run": 0,
        "regions_run": 0,
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "changed": 0,
        "metric_improved": 0,
        "metric_unchanged": 0,
        "metric_worse": 0,
        "delta_ops": 0,
        "delta_depth": 0,
        "delta_two_qubit_ops": 0,
        "candidates_emitted": 0,
        "candidate_equivalent": 0,
        "candidate_not_equivalent": 0,
        "seconds": 0.0,
    }

    selected_paths = paths[:max_files] if max_files is not None else paths
    with result_path.open("w", encoding="utf-8") as result_file:
        candidate_file = (
            candidate_path.open("w", encoding="utf-8") if emit_candidates else None
        )
        for path in selected_paths:
            try:
                summary["files_seen"] += 1
                program = parse_pbc_file(path)
                if selection == "ranked-window":
                    windows = extract_supported_rotation_windows(
                        program,
                        source_path=path,
                        max_windows=max_windows_per_file,
                        window_terms=window_terms,
                        topology=topology,
                    )
                    if windows:
                        summary["files_with_windows"] += 1
                    for window in windows:
                        result = run_ai_pauli_network_synthesis(
                            num_qubits=window.num_qubits,
                            signed_paulis=window.signed_paulis,
                            coupling_map=list(window.coupling_map),
                            max_threads=max_threads,
                            capture_pass_output=capture_pass_output,
                        )
                        window_dict = asdict(window)
                        _write_result(
                            result_file,
                            selection=selection,
                            result=result,
                            include_qasm=include_qasm,
                            window=window_dict,
                        )
                        _update_summary(summary, result, kind="window")
                        if candidate_file is not None:
                            _maybe_write_candidate(
                                candidate_file,
                                summary,
                                selection=selection,
                                result=result,
                                window=window_dict,
                            )
                elif selection == "collect-pauli-networks":
                    regions = extract_rotation_regions(
                        program,
                        source_path=path,
                        max_regions=max_regions_per_file,
                        min_terms=min_region_terms,
                        max_terms=max_region_terms,
                        topology=topology,
                    )
                    if regions:
                        summary["files_with_regions"] += 1
                    for region in regions:
                        result = run_ai_pauli_network_synthesis_on_circuit(
                            circuit=build_rotation_region_circuit(region),
                            coupling_map=list(region.coupling_map),
                            max_threads=max_threads,
                            capture_pass_output=capture_pass_output,
                        )
                        _write_result(
                            result_file,
                            selection=selection,
                            result=result,
                            include_qasm=include_qasm,
                            region=asdict(region),
                        )
                        _update_summary(summary, result, kind="region")
                else:
                    raise ValueError(f"unsupported selection {selection!r}")
            finally:
                if candidate_file is not None:
                    candidate_file.flush()
        if candidate_file is not None:
            candidate_file.close()

    summary["seconds"] = round(float(summary["seconds"]), 6)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def find_pbc_files(inputs: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob(pattern) if item.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(files))


def _write_result(
    result_file: Any,
    *,
    selection: str,
    result: dict[str, Any],
    include_qasm: bool,
    window: dict[str, Any] | None = None,
    region: dict[str, Any] | None = None,
) -> None:
    if not include_qasm:
        result = {key: value for key, value in result.items()}
        result.pop("optimized_qasm", None)
        result.pop("pass_output", None)
    record: dict[str, Any] = {"selection": selection, "result": result}
    if window is not None:
        record["window"] = window
    if region is not None:
        record["region"] = region
    result_file.write(json.dumps(record, sort_keys=True) + "\n")


def _maybe_write_candidate(
    candidate_file: Any,
    summary: dict[str, Any],
    *,
    selection: str,
    result: dict[str, Any],
    window: dict[str, Any],
) -> None:
    before = result.get("before")
    after = result.get("after")
    optimized_qasm = result.get("optimized_qasm")
    if result.get("status") != "ok" or before is None or after is None:
        return
    delta = _metric_delta(before, after)
    if (delta["two_qubit_ops"], delta["depth"], delta["ops"]) >= (0, 0, 0):
        return
    if not isinstance(optimized_qasm, str) or not optimized_qasm:
        return

    original = build_pauli_network_circuit(
        num_qubits=window["num_qubits"],
        signed_paulis=window["signed_paulis"],
    )
    candidate = qasm2.loads(optimized_qasm)
    equivalent = circuits_equivalent(original, candidate)
    summary["candidates_emitted"] += 1
    if equivalent:
        summary["candidate_equivalent"] += 1
    else:
        summary["candidate_not_equivalent"] += 1
    candidate_file.write(
        json.dumps(
            {
                "selection": selection,
                "window": window,
                "delta": delta,
                "equivalent": equivalent,
                "optimized_qasm": optimized_qasm,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _update_summary(
    summary: dict[str, Any],
    result: dict[str, Any],
    *,
    kind: str,
) -> None:
    status = result["status"]
    if kind == "window":
        summary["windows_run"] += 1
    elif kind == "region":
        summary["regions_run"] += 1
    else:
        raise ValueError(f"unsupported summary kind {kind!r}")
    summary[status] += 1
    summary["seconds"] += float(result.get("seconds", 0.0))
    if result.get("changed"):
        summary["changed"] += 1
    before = result.get("before")
    after = result.get("after")
    if before is None or after is None:
        return
    delta = _metric_delta(before, after)
    summary["delta_ops"] += delta["ops"]
    summary["delta_depth"] += delta["depth"]
    summary["delta_two_qubit_ops"] += delta["two_qubit_ops"]
    objective_delta = (delta["two_qubit_ops"], delta["depth"], delta["ops"])
    if objective_delta < (0, 0, 0):
        summary["metric_improved"] += 1
    elif objective_delta > (0, 0, 0):
        summary["metric_worse"] += 1
    else:
        summary["metric_unchanged"] += 1


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        "ops": int(after["ops"]) - int(before["ops"]),
        "depth": int(after["depth"]) - int(before["depth"]),
        "two_qubit_ops": int(after["two_qubit_ops"]) - int(before["two_qubit_ops"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Qiskit's AI Pauli-network synthesis on PBC windows."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_pbc_post_opt.txt")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-windows-per-file", type=int, default=4)
    parser.add_argument("--window-terms", type=int, default=4)
    parser.add_argument(
        "--selection",
        default="ranked-window",
        choices=["ranked-window", "collect-pauli-networks"],
    )
    parser.add_argument("--max-regions-per-file", type=int, default=4)
    parser.add_argument("--min-region-terms", type=int, default=4)
    parser.add_argument("--max-region-terms", type=int)
    parser.add_argument("--topology", default="line", choices=["line", "t", "y"])
    parser.add_argument("--max-threads", type=int)
    parser.add_argument("--include-qasm", action="store_true")
    parser.add_argument(
        "--emit-candidates",
        action="store_true",
        help="Write improved ranked-window replacements with equivalence checks.",
    )
    parser.add_argument(
        "--show-pass-output",
        action="store_true",
        help="Let IBM synthesis diagnostics print to the terminal.",
    )
    args = parser.parse_args()

    paths = find_pbc_files(args.inputs, args.pattern)
    summary = benchmark_pbc_files(
        paths=paths,
        out_dir=args.out_dir,
        max_windows_per_file=args.max_windows_per_file,
        max_files=args.max_files,
        window_terms=args.window_terms,
        topology=args.topology,
        max_threads=args.max_threads,
        include_qasm=args.include_qasm,
        selection=args.selection,
        max_regions_per_file=args.max_regions_per_file,
        min_region_terms=args.min_region_terms,
        max_region_terms=args.max_region_terms,
        capture_pass_output=not args.show_pass_output,
        emit_candidates=args.emit_candidates,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
