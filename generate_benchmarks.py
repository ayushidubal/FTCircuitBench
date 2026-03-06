# generate_benchmarks.py (Corrected)
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from qiskit import QuantumCircuit

# Add cabal bin to PATH for gridsynth
os.environ["PATH"] = (
    os.path.expanduser("~/.cabal/bin") + ":" + os.environ.get("PATH", "")
)

from ftcircuitbench.benchmark_utils import (
    combine_pbc_files_same_dir,
    find_all_qasm_files,
    get_clifford_t_qasm_path,
    get_output_param_str,
    get_pbc_prefix,
    get_stats_json_path,
    save_json,
)
from ftcircuitbench import load_qasm_circuit
from ftcircuitbench.api import PipelineConfig, run_pipeline
from ftcircuitbench.reports.summary_markdown import generate_summary_markdown

# Default parameter sets
DEFAULT_SK_DEGREES = [1, 2]
DEFAULT_GS_PRECISIONS = [3, 10]
BASE_OUTPUT_DIR = "circuit_benchmarks"


def _flatten_benchmark_instances(
    benchmarks_config: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str, str]]:
    """Build a deterministic list of (category, instance_name, qasm_path)."""
    instances: List[Tuple[str, str, str]] = []
    for category in sorted(benchmarks_config.keys()):
        for instance_name in sorted(benchmarks_config[category].keys()):
            instances.append(
                (category, instance_name, benchmarks_config[category][instance_name])
            )
    return instances


def quick_qubit_count_check(qasm_file_path: str) -> int:
    """
    Quickly check the number of qubits in a QASM file by parsing only the quantum register declarations.
    This is much faster than loading the full circuit for large files.

    This optimization is critical for the generate_benchmarks.py script when using --max-qubits,
    as it prevents the script from spending significant time loading large circuits that will
    be skipped anyway.

    Supports both OpenQASM 2.0 (qreg name[size];) and OpenQASM 3.0 (qubit[size] name;) syntax.

    Args:
        qasm_file_path (str): Path to the QASM file

    Returns:
        int: Total number of qubits declared in the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be parsed or has invalid quantum register declarations

    Example:
        >>> quick_qubit_count_check('qasm/hamiltonians/fermi_hubbard_2d_128q.qasm')
        128
    """
    if not os.path.exists(qasm_file_path):
        raise FileNotFoundError(f"QASM file not found: {qasm_file_path}")

    total_qubits = 0

    try:
        with open(qasm_file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("//"):
                    continue

                # Look for OpenQASM 2.0 qreg declarations: qreg name[size];
                qreg_match = re.match(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\];", line)
                if qreg_match:
                    register_size = int(qreg_match.group(2))
                    total_qubits += register_size

                # Look for OpenQASM 3.0 qubit declarations: qubit[size] name;
                qubit_match = re.match(r"qubit\s*\[\s*(\d+)\s*\]\s+(\w+);", line)
                if qubit_match:
                    register_size = int(qubit_match.group(1))
                    total_qubits += register_size

                # Stop parsing if we hit the first non-declaration line (gates, etc.)
                # But only if we've already found at least one quantum register declaration
                # This handles cases where gate definitions come before register declarations
                if (
                    total_qubits > 0
                    and line
                    and not line.startswith("qreg")
                    and not line.startswith("qubit")
                    and not line.startswith("creg")
                    and not line.startswith("bit")
                    and not line.startswith("OPENQASM")
                    and not line.startswith("include")
                    and not line.startswith("gate")
                    and not line.startswith("}")
                ):
                    # We've hit the first gate or operation after finding register declarations, so we can stop
                    break

    except Exception as e:
        raise ValueError(f"Error parsing QASM file {qasm_file_path}: {e}")

    if total_qubits == 0:
        raise ValueError(f"No quantum registers found in {qasm_file_path}")

    return total_qubits


def _get_param_str(pipeline_type: str, parameter_value: int) -> str:
    param_type = "precision_level" if pipeline_type == "gs" else "recursion_degree"
    return get_output_param_str(pipeline_type, parameter_value, param_type)


def _get_param_dir(
    output_dir: str,
    category: str,
    instance_name: str,
    pipeline_type: str,
    parameter_value: int,
) -> str:
    if pipeline_type == "gs":
        return os.path.join(
            output_dir,
            category,
            instance_name,
            "GS",
            f"precision_level_{parameter_value}",
        )
    return os.path.join(
        output_dir,
        category,
        instance_name,
        "SK",
        f"recursion_degree_{parameter_value}",
    )


def _variation_outputs_exist(
    pipeline_type: str,
    parameter_value: int,
    param_specific_dir: str,
    instance_name: str,
) -> bool:
    param_str = _get_param_str(pipeline_type, parameter_value)
    clifford_t_qasm_path = get_clifford_t_qasm_path(
        param_specific_dir, instance_name, param_str
    )
    stats_json_path = get_stats_json_path(param_specific_dir, instance_name, param_str)
    return os.path.isfile(clifford_t_qasm_path) and os.path.isfile(stats_json_path)


def _load_existing_stats_if_present(
    pipeline_type: str,
    parameter_value: int,
    param_specific_dir: str,
    instance_name: str,
) -> Optional[Dict[str, Any]]:
    param_str = _get_param_str(pipeline_type, parameter_value)
    stats_json_path = get_stats_json_path(param_specific_dir, instance_name, param_str)
    if not os.path.isfile(stats_json_path):
        return None
    try:
        with open(stats_json_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"    ⚠️ Could not read existing stats at {stats_json_path}: {exc}")
        return None


def _all_requested_outputs_exist(
    args: argparse.Namespace,
    category: str,
    instance_name: str,
    sk_params: List[int],
    gs_params: List[int],
) -> bool:
    checks: List[bool] = []
    if not args.skip_sk:
        for degree in sk_params:
            param_dir = _get_param_dir(
                args.output_dir, category, instance_name, "sk", degree
            )
            checks.append(
                _variation_outputs_exist("sk", degree, param_dir, instance_name)
            )
    if not args.skip_gs:
        for precision in gs_params:
            param_dir = _get_param_dir(
                args.output_dir, category, instance_name, "gs", precision
            )
            checks.append(
                _variation_outputs_exist("gs", precision, param_dir, instance_name)
            )
    return bool(checks) and all(checks)


def _load_instance_filter(instances_file: str) -> Set[str]:
    """Load normalized `category/instance` keys from a plain-text file."""
    selected: Set[str] = set()
    with open(instances_file, "r") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            key = raw
            if key.startswith("qasm/"):
                key = key[len("qasm/") :]
            if key.endswith(".qasm"):
                key = key[: -len(".qasm")]
            parts = key.split("/")
            if len(parts) < 2:
                print(f"  ⚠️ Ignoring malformed instance key: {raw}")
                continue
            selected.add(f"{parts[0]}/{parts[1]}")
    return selected


def process_pipeline_variation(
    pipeline_type: str,
    parameter_value: int,
    original_qc: QuantumCircuit,
    param_specific_dir: str,
    instance_name: str,
    args: argparse.Namespace,
    layering_method_override: Optional[str] = None,
    optimize_t_maxiter_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    A unified function to run either the GS or SK pipeline for one parameter variation.
    """
    param_type = "precision_level" if pipeline_type == "gs" else "recursion_degree"
    param_str = get_output_param_str(pipeline_type, parameter_value, param_type)
    circuit_name = instance_name
    clifford_t_qasm_path = get_clifford_t_qasm_path(
        param_specific_dir, circuit_name, param_str
    )
    full_pbc_output_prefix = (
        get_pbc_prefix(param_specific_dir, circuit_name, param_str)
        if not args.skip_pbc
        else None
    )

    if args.skip_pbc:
        print("    Skipping PBC conversion...")
    else:
        print("    Converting to PBC...")
    sys.stdout.flush()

    effective_layering_method = (
        layering_method_override if layering_method_override else args.layering_method
    )
    effective_optimize_t_maxiter = (
        optimize_t_maxiter_override
        if optimize_t_maxiter_override is not None
        else args.optimize_t_maxiter
    )

    # Prefer C++ Gridsynth by default; the library will safely fall back to Python
    # if nwqec is not importable/available.
    prefer_cpp = not args.force_python

    config = PipelineConfig(
        pipeline=pipeline_type,
        gridsynth_precision=parameter_value if pipeline_type == "gs" else None,
        sk_recursion=parameter_value if pipeline_type == "sk" else None,
        layering_method=effective_layering_method,
        layering_max_checks=args.layering_max_checks,
        optimize_pbc=args.optimize_pbc,
        optimize_t_maxiter=effective_optimize_t_maxiter,
        prefer_cpp=prefer_cpp,
        calculate_fidelity=not args.skip_fidelity,
        return_intermediate=True,
        clifford_output_path=clifford_t_qasm_path,
        pbc_output_prefix=full_pbc_output_prefix,
        run_pbc=not args.skip_pbc,
    )

    result = run_pipeline(original_qc, config)
    stats: Dict[str, Any] = {}
    stats.update(result.clifford_stats)
    stats.update(result.pbc_stats)

    if result.fidelity:
        stats.update(result.fidelity)
        stats["fidelity_method"] = result.fidelity.get("method", "N/A")
    else:
        stats.update({"fidelity": None, "fidelity_method": "skipped"})

    stats["transpilation_clifford_t_time"] = result.timings.get(
        "transpilation_clifford_t_time"
    )
    stats["pbc_conversion_time"] = result.timings.get("pbc_conversion_time")
    numeric_timings = [
        t for t in result.timings.values() if isinstance(t, (int, float))
    ]
    if numeric_timings:
        stats["total_time"] = sum(numeric_timings)

    if full_pbc_output_prefix:
        combine_pbc_files_same_dir(
            os.path.dirname(full_pbc_output_prefix),
            os.path.basename(full_pbc_output_prefix),
            "pre_opt",
        )
        combine_pbc_files_same_dir(
            os.path.dirname(full_pbc_output_prefix),
            os.path.basename(full_pbc_output_prefix),
            "post_opt",
        )

    stats.update(
        {
            "pipeline": pipeline_type.upper(),
            "parameter_type": (
                "precision_level" if pipeline_type == "gs" else "recursion_degree"
            ),
            "parameter_value": parameter_value,
            "pbc_layering_method": result.parameters.get("layering_method"),
            "pbc_layering_max_checks": result.parameters.get("layering_max_checks"),
            "pbc_optimize_t_maxiter": result.parameters.get("optimize_t_maxiter"),
        }
    )
    # --- Save all stats to JSON file in the parameter-specific directory ---
    stats_json_path = get_stats_json_path(param_specific_dir, circuit_name, param_str)
    save_json(stats, stats_json_path)

    return stats


def main():
    """Main function to generate benchmarks for all algorithms and instances."""
    parser = argparse.ArgumentParser(description="Generate FTCircuitBench benchmarks")
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=1000,
        help="Maximum number of qubits for circuits to process (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=BASE_OUTPUT_DIR,
        help="Output directory for benchmarks",
    )
    parser.add_argument(
        "--skip-fidelity",
        action="store_true",
        help="Skip expensive fidelity calculations",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use only recursion degree 1 for SK and precision 3 for GS",
    )
    parser.add_argument(
        "--skip-sk", action="store_true", help="Skip Solovay-Kitaev pipeline"
    )
    parser.add_argument(
        "--skip-gs", action="store_true", help="Skip Gridsynth pipeline"
    )
    parser.add_argument(
        "--skip-pbc",
        action="store_true",
        help="Skip PBC conversion and only benchmark Clifford+T synthesis",
    )
    parser.add_argument(
        "--optimize-t-maxiter",
        type=int,
        default=3,
        help="Maximum PBC optimization iterations. Set to 0 to disable. (default: 3)",
    )
    parser.add_argument(
        "--layering-method",
        choices=["bare", "v2", "v3", "singleton"],
        default="v2",
        help="PBC layering method to use (default: v2)",
    )
    parser.add_argument(
        "--layering-max-checks",
        type=int,
        default=None,
        help="If set, bound layering by checking only the last K layers",
    )
    parser.add_argument(
        "--optimize-pbc",
        action="store_true",
        help="Enable PBC Tfuse/T-merging optimization",
    )
    parser.add_argument(
        "--prefer-cpp",
        action="store_true",
        help="(Deprecated) Prefer nwqec C++ backend (now default unless --force-python)",
    )
    parser.add_argument(
        "--force-python",
        action="store_true",
        help="Force Python backend even if nwqec is available",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of disjoint shards (default: 1, i.e. no sharding)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to execute (default: 0)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip work when expected output artifacts already exist",
    )
    parser.add_argument(
        "--instances-file",
        type=str,
        default=None,
        help="Optional file with one `category/instance` per line to restrict work",
    )

    group_detailed = parser.add_mutually_exclusive_group()
    group_detailed.add_argument(
        "--detailed",
        dest="detailed",
        action="store_true",
        help="Show detailed PBC conversion progress",
    )
    group_detailed.add_argument(
        "--no-detailed",
        dest="detailed",
        action="store_false",
        help="Hide detailed PBC progress",
    )
    parser.set_defaults(detailed=False)  # Default to false for cleaner benchmark logs

    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    # Benchmark each circuit across the default parameter grids
    sk_params = [DEFAULT_SK_DEGREES[0]] if args.quick else DEFAULT_SK_DEGREES
    gs_params = [DEFAULT_GS_PRECISIONS[0]] if args.quick else DEFAULT_GS_PRECISIONS

    print(f"Processing circuits with <= {args.max_qubits} qubits.")
    print(f"Output directory: {args.output_dir}")
    if args.skip_pbc:
        print("PBC conversion is DISABLED.")
    else:
        print("Using parallel PBC conversion (Python fallback only).")
    if not args.skip_pbc and args.optimize_t_maxiter == 0:
        print("PBC T-gate optimization is DISABLED.")

    benchmarks_config = find_all_qasm_files("qasm")
    os.makedirs(args.output_dir, exist_ok=True)

    all_instances = _flatten_benchmark_instances(benchmarks_config)
    total_discovered_instances = len(all_instances)

    if args.instances_file:
        requested_instance_keys = _load_instance_filter(args.instances_file)
        discovered_map: Dict[str, Tuple[str, str, str]] = {
            f"{category}/{instance_name}": (category, instance_name, qasm_file_path)
            for category, instance_name, qasm_file_path in all_instances
        }
        missing_keys = sorted(
            key for key in requested_instance_keys if key not in discovered_map
        )
        if missing_keys:
            print(
                f"⚠️ {len(missing_keys)} entries from {args.instances_file} "
                "were not found under qasm/"
            )
            for key in missing_keys[:10]:
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more")

        all_instances = [
            item
            for item in all_instances
            if f"{item[0]}/{item[1]}" in requested_instance_keys
        ]
        print(
            f"Instance filter active: {len(all_instances)}/{total_discovered_instances} "
            "circuits selected"
        )

    total_instances = len(all_instances)
    sharding_pool = all_instances
    if args.skip_existing:
        sharding_pool = [
            (category, instance_name, qasm_file_path)
            for category, instance_name, qasm_file_path in all_instances
            if not _all_requested_outputs_exist(
                args, category, instance_name, sk_params, gs_params
            )
        ]
        print(
            f"Skip-existing prefilter: {len(sharding_pool)}/{total_instances} "
            "circuits still need requested outputs"
        )

    selected_instances = [
        item
        for idx, item in enumerate(sharding_pool)
        if idx % args.num_shards == args.shard_index
    ]
    shard_total = len(selected_instances)

    print(
        f"Shard configuration: index {args.shard_index}/{args.num_shards - 1} "
        f"({shard_total}/{len(sharding_pool)} circuits assigned)"
    )

    for processed_instances, (category, instance_name, qasm_file_path) in enumerate(
        selected_instances, start=1
    ):
        print(
            f"\n[{processed_instances}/{shard_total}] 📄 Processing: {category}/{instance_name}"
        )

        try:
            # Use quick_qubit_count_check to check if the circuit exceeds max_qubits
            if args.skip_existing and _all_requested_outputs_exist(
                args, category, instance_name, sk_params, gs_params
            ):
                print("  ⏭️ Skipping: all requested outputs already exist")
                continue

            qubit_count = quick_qubit_count_check(qasm_file_path)
            if qubit_count > args.max_qubits:
                print(f"  Skipping: {qubit_count} qubits > {args.max_qubits}")
                continue

            original_qc = load_qasm_circuit(qasm_file_path, is_file=True)
            original_qc_for_fidelity = original_qc.copy()
            original_qc_for_fidelity.remove_final_measurements(inplace=True)

            # Decide if singleton layering should be forced for this instance
            # ================================================================
            force_singleton = False
            # Category-based rules
            if category.startswith("hamiltonians"):
                force_singleton = True
            if category.startswith("qpe"):
                force_singleton = True
            if category.startswith("qsvt"):
                force_singleton = True
            # Instance-based size rules
            # hhl greater than 7 qubits
            if category.startswith("hhl") and qubit_count > 7:
                force_singleton = True
            # qft greater than 18 qubits
            if category.startswith("qft") and qubit_count > 18:
                force_singleton = True

            layering_override = "singleton" if force_singleton else None
            # When forcing singleton, skip optimization explicitly for speed
            optimize_override = 0 if force_singleton else None
            # ================================================================

            # Process SK pipeline
            if not args.skip_sk:
                all_sk_stats = []
                for degree in sk_params:
                    param_dir = _get_param_dir(
                        args.output_dir, category, instance_name, "sk", degree
                    )
                    if args.skip_existing and _variation_outputs_exist(
                        "sk", degree, param_dir, instance_name
                    ):
                        print(f"  ⏭️ SK (degree {degree}) already exists, skipping")
                        existing_stats = _load_existing_stats_if_present(
                            "sk", degree, param_dir, instance_name
                        )
                        if existing_stats:
                            all_sk_stats.append(existing_stats)
                        continue
                    try:
                        run_stats = process_pipeline_variation(
                            "sk",
                            degree,
                            original_qc_for_fidelity,
                            param_dir,
                            instance_name,
                            args,
                            layering_method_override=layering_override,
                            optimize_t_maxiter_override=optimize_override,
                        )
                        all_sk_stats.append(run_stats)
                        print(f"  ✅ SK (degree {degree})")
                    except Exception as e:
                        print(f"  ❌ SK (degree {degree}) FAILED: {e}")

                if all_sk_stats:
                    summary_path = os.path.join(
                        args.output_dir,
                        category,
                        instance_name,
                        "SK",
                        "comparison_summary.md",
                    )
                    with open(summary_path, "w") as f:
                        for s in all_sk_stats:
                            f.write(
                                generate_summary_markdown(
                                    s, instance_name, "SK", s["parameter_value"]
                                )
                            )
                            f.write("\n\n---\n\n")

            # Process GS pipeline
            if not args.skip_gs:
                all_gs_stats = []
                for precision in gs_params:
                    param_dir = _get_param_dir(
                        args.output_dir, category, instance_name, "gs", precision
                    )
                    if args.skip_existing and _variation_outputs_exist(
                        "gs", precision, param_dir, instance_name
                    ):
                        print(
                            f"  ⏭️ GS (precision {precision}) already exists, skipping"
                        )
                        existing_stats = _load_existing_stats_if_present(
                            "gs", precision, param_dir, instance_name
                        )
                        if existing_stats:
                            all_gs_stats.append(existing_stats)
                        continue
                    try:
                        run_stats = process_pipeline_variation(
                            "gs",
                            precision,
                            original_qc_for_fidelity,
                            param_dir,
                            instance_name,
                            args,
                            layering_method_override=layering_override,
                            optimize_t_maxiter_override=optimize_override,
                        )
                        all_gs_stats.append(run_stats)
                        print(f"  ✅ GS (precision {precision})")
                    except Exception as e:
                        print(f"  ❌ GS (precision {precision}) FAILED: {e}")

                if all_gs_stats:
                    summary_path = os.path.join(
                        args.output_dir,
                        category,
                        instance_name,
                        "GS",
                        "comparison_summary.md",
                    )
                    with open(summary_path, "w") as f:
                        for s in all_gs_stats:
                            f.write(
                                generate_summary_markdown(
                                    s, instance_name, "GS", s["parameter_value"]
                                )
                            )
                            f.write("\n\n---\n\n")

        except Exception as e:
            print(f"  ❌ FATAL ERROR processing instance {instance_name}: {e}")

    print("\n" + "=" * 60)
    print("✅ BENCHMARK GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
