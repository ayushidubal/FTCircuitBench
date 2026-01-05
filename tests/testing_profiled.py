# examples/run_all_pipelines.py (Modified)
import argparse

# Profiling imports
import cProfile
import os
import sys
import time

from qiskit.quantum_info import Operator

from ftcircuitbench import (
    MAX_QUBITS_FOR_FIDELITY,
    calculate_circuit_fidelity,
    convert_to_pbc_circuit,
    load_qasm_circuit,
    transpile_to_gridsynth_clifford_t,
    transpile_to_solovay_kitaev_clifford_t,
)

# Import the analyzer functions
from ftcircuitbench.analyzer import analyze_clifford_t_circuit, analyze_pbc_circuit


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 0.01 and seconds != 0:
        return f"{seconds*1000:.2f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def print_circuit_stats(title: str, stats: dict, show_detailed: bool = False):
    """Print circuit statistics in a clean, formatted way."""
    print(f"\n=== {title} ===")

    # Basic circuit info
    print("Circuit Size:")
    print(f"  Qubits: {stats.get('num_qubits', 'N/A')}")
    print(f"  Total Gates: {stats.get('total_gate_count', 'N/A')}")
    print(f"  Depth: {stats.get('depth', 'N/A')}")

    # T-gate statistics
    if "t_count" in stats or "tdg_count" in stats or "total_t_family_count" in stats:
        print("\nT-gate Analysis:")
        print(f"  T-count: {stats.get('t_count', 0)}")
        print(f"  Tdg-count: {stats.get('tdg_count', 0)}")
        print(f"  Total T-family: {stats.get('total_t_family_count', 0)}")
        if "t_gates_per_qubit" in stats and stats["t_gates_per_qubit"]:
            max_qubit_item = max(
                stats["t_gates_per_qubit"].items(),
                key=lambda x: x[1],
                default=(None, 0),
            )
            if max_qubit_item[0] is not None:
                print(
                    f"  Max T-gates on qubit {max_qubit_item[0]}: {max_qubit_item[1]}"
                )

    # Clifford gate statistics
    if "clifford_gate_count" in stats:
        print("\nClifford Gates:")
        print(f"  Total: {stats['clifford_gate_count']}")
        if "detailed_clifford_counts" in stats and show_detailed:
            for gate, count in stats["detailed_clifford_counts"].items():
                print(f"    {gate}: {count}")

    # Two-qubit gate statistics
    if "total_two_qubit_gates" in stats:
        print("\nTwo-qubit Gates:")
        print(f"  Total: {stats['total_two_qubit_gates']}")
        if "avg_qubit_interaction_degree" in stats:
            print("  Modularity Statistics:")
            print(
                f"    Average interaction degree: {stats['avg_qubit_interaction_degree']:.2f}"
            )
            print(
                f"    Std dev interaction degree: {stats['std_qubit_interaction_degree']:.2f}"
            )
            print(
                f"    Interaction graph density: {stats['interaction_graph_density']:.4f}"
            )
        elif "two_qubit_gate_interaction_pairs" in stats and show_detailed:
            print("  Interaction pairs:")
            for (q1, q2), count in sorted(
                stats["two_qubit_gate_interaction_pairs"].items()
            ):
                print(f"    ({q1}, {q2}): {count}")

    # PBC specific statistics
    if "pbc_t_operators" in stats:
        print("\nPBC Circuit Metrics:")
        print(f"  Rotation operators: {stats.get('pbc_t_operators', 'N/A')}")
        print(
            f"  Measurement operators: {stats.get('pbc_measurement_operators', 'N/A')}"
        )
        if "pbc_num_utility_operators" in stats:
            print(
                f"  Utility operators (barriers): {stats.get('pbc_num_utility_operators',0)}"
            )

        # PBC Pauli weight statistics
        if "pbc_avg_pauli_weight" in stats:
            print("  Rotation Pauli Weight Statistics:")
            print(f"    Average weight: {stats['pbc_avg_pauli_weight']:.2f}")
            print(f"    Std dev weight: {stats['pbc_std_pauli_weight']:.2f}")
            print(f"    Min weight: {stats['pbc_min_pauli_weight']}")
            print(f"    Max weight: {stats['pbc_max_pauli_weight']}")

        if "pbc_avg_measurement_pauli_weight" in stats:
            print("  Measurement Pauli Weight Statistics:")
            print(
                f"    Average weight: {stats['pbc_avg_measurement_pauli_weight']:.2f}"
            )
            print(
                f"    Std dev weight: {stats['pbc_std_measurement_pauli_weight']:.2f}"
            )
            print(f"    Min weight: {stats['pbc_min_measurement_pauli_weight']}")
            print(f"    Max weight: {stats['pbc_max_measurement_pauli_weight']}")

        # PBC Modularity statistics
        if "pbc_avg_qubit_interaction_degree" in stats:
            pbc_degrees = list(stats["pbc_avg_qubit_interaction_degree"].values())
            if pbc_degrees:
                pbc_avg_degree = sum(pbc_degrees) / len(pbc_degrees)
                pbc_std_degree = (
                    sum((x - pbc_avg_degree) ** 2 for x in pbc_degrees)
                    / len(pbc_degrees)
                ) ** 0.5
                print("  PBC Modularity Statistics:")
                print(f"    Average interaction degree: {pbc_avg_degree:.2f}")
                print(f"    Std dev interaction degree: {pbc_std_degree:.2f}")

    if "optimized_rpc_t_gates" in stats:  # This indicates PBC stats are present
        print("\nPBC Optimization Result:")
        print(
            f"  Initial C+T T-gates: {stats.get('initial_clifford_t_t_gates_for_pbc', stats.get('total_t_family_count','N/A'))}"
        )
        print(f"  Optimized PBC T-gate equivalents: {stats['optimized_rpc_t_gates']}")
        initial_t = stats.get(
            "initial_clifford_t_t_gates_for_pbc", stats.get("total_t_family_count", 0)
        )
        if isinstance(initial_t, (int, float)) and initial_t > 0:
            reduction = (initial_t - stats["optimized_rpc_t_gates"]) / initial_t * 100
            print(f"  T-gate reduction by PBC: {reduction:.2f}%")
        else:
            print("  T-gate reduction by PBC: N/A")

    # Show detailed statistics only if requested and available
    if show_detailed:
        print("\nPipeline Timings:")
        if "transpilation_clifford_t_time" in stats:
            print(
                f"  Clifford+T Transpilation: {format_time(stats['transpilation_clifford_t_time'])}"
            )
        if "analysis_clifford_t_time" in stats:
            print(
                f"  Clifford+T Analysis: {format_time(stats['analysis_clifford_t_time'])}"
            )
        if "pbc_conversion_time" in stats:
            print(f"  PBC Conversion: {format_time(stats['pbc_conversion_time'])}")
        if "analysis_pbc_time" in stats:
            print(f"  PBC Analysis: {format_time(stats['analysis_pbc_time'])}")
        if "total_time" in stats:  # Overall pipeline time
            print(f"  Total Pipeline Time: {format_time(stats['total_time'])}")

        print("\nOther Detailed Statistics:")
        # List keys that are already handled or too verbose for default detailed view
        handled_keys = [
            "num_qubits",
            "total_gate_count",
            "depth",
            "t_count",
            "tdg_count",
            "total_t_family_count",
            "t_gates_per_qubit",
            "clifford_gate_count",
            "detailed_clifford_counts",
            "total_two_qubit_gates",
            "two_qubit_gate_interaction_pairs",
            "avg_qubit_interaction_degree",
            "std_qubit_interaction_degree",
            "interaction_graph_density",
            "pbc_t_operators",
            "pbc_measurement_operators",
            "pbc_num_utility_operators",
            "pbc_avg_pauli_weight",
            "pbc_std_pauli_weight",
            "pbc_min_pauli_weight",
            "pbc_max_pauli_weight",
            "pbc_avg_measurement_pauli_weight",
            "pbc_std_measurement_pauli_weight",
            "pbc_min_measurement_pauli_weight",
            "pbc_max_measurement_pauli_weight",
            "pbc_avg_qubit_interaction_degree",
            "pbc_std_qubit_interaction_degree",
            "optimized_rpc_t_gates",
            "initial_clifford_t_t_gates_for_pbc",
            "transpilation_clifford_t_time",
            "analysis_clifford_t_time",
            "pbc_conversion_time",
            "analysis_pbc_time",
            "total_time",
            "initial_sk_t_gates",  # from pbc_stats
            "compilation_precision_digits",
            "fidelity",
            "pbc_conversion_stats_from_converter",  # internal holder
        ]
        if "pbc_conversion_stats_from_converter" in stats and isinstance(
            stats["pbc_conversion_stats_from_converter"], dict
        ):
            for key, value in stats["pbc_conversion_stats_from_converter"].items():
                if key not in handled_keys:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    elif isinstance(value, dict) and key.endswith("distribution"):
                        print(f"  {key}: {value}")  # Print distributions as is
                    elif isinstance(value, list) and key.endswith("per_layer"):
                        print(f"  {key}: {value}")  # Print layer lists as is
                    else:
                        print(f"  {key}: {value}")

        for key, value in stats.items():
            if key not in handled_keys and key != "pbc_conversion_stats_from_converter":
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def print_pbc_optimization_summary(stats: dict, pipeline_name: str):
    """Print PBC optimization information for a single pipeline (used in comparison)."""
    if "optimized_rpc_t_gates" in stats:
        print(f"\nPBC Optimization ({pipeline_name}):")
        initial_t = stats.get(
            "initial_clifford_t_t_gates_for_pbc",
            stats.get("total_t_family_count", "N/A"),
        )
        print(f"  Initial C+T T-gates: {initial_t}")
        print(f"  Optimized PBC T-gate equivalents: {stats['optimized_rpc_t_gates']}")

        if isinstance(initial_t, (int, float)) and initial_t > 0:
            reduction = (initial_t - stats["optimized_rpc_t_gates"]) / initial_t * 100
            print(f"  T-gate reduction by PBC: {reduction:.2f}%")
        else:
            print("  T-gate reduction by PBC: N/A")


def print_pipeline_comparison(gs_results: dict, sk_results: dict):
    """Print a comparison of the two pipelines."""
    print("\n\n===========================")
    print("=== Pipeline Comparison ===")
    print("===========================")

    # Timing comparison
    print("\nOverall Pipeline Timing:")
    print(f"  Gridsynth Pipeline: {format_time(gs_results.get('total_time', 0))}")
    print(f"  Solovay-Kitaev Pipeline: {format_time(sk_results.get('total_time', 0))}")

    # Clifford+T Circuit size comparison
    print("\nClifford+T Circuit Size (before PBC):")
    gs_ct_stats = gs_results.get("clifford_t_stats", {})
    sk_ct_stats = sk_results.get("clifford_t_stats", {})
    print(
        f"  Gridsynth C+T: {gs_ct_stats.get('total_gate_count', 'N/A')} gates, depth {gs_ct_stats.get('depth', 'N/A')}"
    )
    print(
        f"  Solovay-Kitaev C+T: {sk_ct_stats.get('total_gate_count', 'N/A')} gates, depth {sk_ct_stats.get('depth', 'N/A')}"
    )

    # Clifford+T T-gate comparison
    print("\nClifford+T T-gate Count (before PBC):")
    print(f"  Gridsynth C+T: {gs_ct_stats.get('total_t_family_count', 0)} T-gates")
    print(f"  Solovay-Kitaev C+T: {sk_ct_stats.get('total_t_family_count', 0)} T-gates")

    # Clifford+T Fidelity comparison
    if "fidelity" in gs_ct_stats and "fidelity" in sk_ct_stats:
        print("\nClifford+T Circuit Fidelity (vs original):")
        print(f"  Gridsynth C+T: {gs_ct_stats['fidelity']:.7f}")
        print(f"  Solovay-Kitaev C+T: {sk_ct_stats['fidelity']:.7f}")

    # PBC optimization comparison
    gs_pbc_stats = gs_results.get("pbc_stats", {})
    sk_pbc_stats = sk_results.get("pbc_stats", {})
    if (
        "optimized_rpc_t_gates" in gs_pbc_stats
        and "optimized_rpc_t_gates" in sk_pbc_stats
    ):
        print("\nPBC Optimization (T-gate equivalents):")
        print(f"  Gridsynth PBC: {gs_pbc_stats['optimized_rpc_t_gates']}")
        print(f"  Solovay-Kitaev PBC: {sk_pbc_stats['optimized_rpc_t_gates']}")

        gs_initial_t_for_pbc = gs_pbc_stats.get(
            "initial_clifford_t_t_gates_for_pbc",
            gs_ct_stats.get("total_t_family_count", 0),
        )
        sk_initial_t_for_pbc = sk_pbc_stats.get(
            "initial_clifford_t_t_gates_for_pbc",
            sk_ct_stats.get("total_t_family_count", 0),
        )

        if gs_initial_t_for_pbc > 0:
            gs_reduction = (
                (gs_initial_t_for_pbc - gs_pbc_stats["optimized_rpc_t_gates"])
                / gs_initial_t_for_pbc
                * 100
            )
            print(
                f"  T-gate reduction (Gridsynth Path): {gs_reduction:.2f}% (from {gs_initial_t_for_pbc} C+T T-gates)"
            )
        if sk_initial_t_for_pbc > 0:
            sk_reduction = (
                (sk_initial_t_for_pbc - sk_pbc_stats["optimized_rpc_t_gates"])
                / sk_initial_t_for_pbc
                * 100
            )
            print(
                f"  T-gate reduction (Solovay-Kitaev Path): {sk_reduction:.2f}% (from {sk_initial_t_for_pbc} C+T T-gates)"
            )


def parse_arguments():
    parser = argparse.ArgumentParser(description="FTCircuitBench Analysis Tool")
    parser.add_argument("qasm_file", help="Path to the QASM file to analyze")
    parser.add_argument(
        "--gridsynth-precision",
        type=int,
        default=10,
        help="Precision for Gridsynth compilation (default: 10)",
    )
    parser.add_argument(
        "--sk-recursion",
        type=int,
        default=3,
        help="Recursion degree for Solovay-Kitaev compilation (default: 3)",
    )
    parser.add_argument(
        "--layering-method",
        choices=["bare", "v2"],
        default="bare",
        help="Method to use for PBC layering (default: bare)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["gs", "sk", "both"],
        default="gs",
        help="Pipeline to run: gs (Gridsynth), sk (Solovay-Kitaev), or both (default: gs)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    qasm_filename = args.qasm_file

    if not os.path.exists(qasm_filename):
        print(f"Error: QASM file '{qasm_filename}' not found.")
        sys.exit(1)

    print("=== FTCircuitBench Analysis ===")
    print(f"Input: {qasm_filename}")
    print(f"Using layering method: {args.layering_method}")
    print(f"Pipeline: {args.pipeline}")

    try:
        initial_qc_obj = load_qasm_circuit(qasm_filename, is_file=True)
    except Exception as e:
        print(f"Error loading QASM file: {e}")
        sys.exit(1)

    initial_qc_for_unitary = initial_qc_obj.copy()
    initial_qc_for_unitary.remove_final_measurements(inplace=True)

    if initial_qc_for_unitary.num_qubits <= MAX_QUBITS_FOR_FIDELITY:
        print("\nComputing unitary for input circuit...")
        unitary_start_time = time.time()
        try:
            Operator(initial_qc_for_unitary)
            print(
                f"Original circuit ({initial_qc_for_unitary.num_qubits} qubits) unitary computed in {format_time(time.time() - unitary_start_time)}."
            )
        except Exception as e:
            print(f"Warning: Could not compute original unitary: {e}")
    else:
        print(
            f"\nNote: Circuit has {initial_qc_for_unitary.num_qubits} qubits. Traditional unitary-based fidelity calculation is limited to circuits with <= {MAX_QUBITS_FOR_FIDELITY} qubits."
        )
        print(
            "Using RZ product fidelity method: fidelity proxy calculated as product of individual RZ gate decomposition fidelities."
        )

    gs_results = {}
    sk_results = {}

    if args.pipeline in ["gs", "both"]:
        print("\n\n=============================================")
        print("=== Pipeline: Gridsynth-based -> PBC ===")
        print("=============================================")
        pipeline_total_start_time = time.time()
        current_pipeline_stats = {}
        try:
            print("\nStep 1: Transpiling to Gridsynth Clifford+T...")
            ct_transpile_start_time = time.time()
            gs_clifford_t_circuit = transpile_to_gridsynth_clifford_t(
                initial_qc_obj.copy(),
                gridsynth_precision=args.gridsynth_precision,
                ensure_pbc_input_basis=True,
            )
            current_pipeline_stats["transpilation_clifford_t_time"] = (
                time.time() - ct_transpile_start_time
            )
            print(
                f"Clifford+T transpilation time: {format_time(current_pipeline_stats['transpilation_clifford_t_time'])}"
            )

            ct_analysis_start_time = time.time()
            clifford_t_stats = analyze_clifford_t_circuit(
                gs_clifford_t_circuit,
                gridsynth_precision_used=args.gridsynth_precision,
            )
            current_pipeline_stats["analysis_clifford_t_time"] = (
                time.time() - ct_analysis_start_time
            )
            print(
                f"Clifford+T analysis time: {format_time(current_pipeline_stats['analysis_clifford_t_time'])}"
            )
            current_pipeline_stats.update(clifford_t_stats)  # Add all C+T stats
            gs_results["clifford_t_stats"] = (
                clifford_t_stats.copy()
            )  # Store for comparison

            # Calculate fidelity for Clifford+T circuit using the new scalable method
            fidelity_start_time = time.time()
            fidelity_result = calculate_circuit_fidelity(
                initial_qc_for_unitary,
                gs_clifford_t_circuit,
                gridsynth_precision=args.gridsynth_precision,
            )

            if fidelity_result["fidelity"] is not None:
                current_pipeline_stats["fidelity"] = fidelity_result["fidelity"]
                current_pipeline_stats["fidelity_method"] = fidelity_result["method"]
                gs_results["clifford_t_stats"]["fidelity"] = fidelity_result["fidelity"]
                gs_results["clifford_t_stats"]["fidelity_method"] = fidelity_result[
                    "method"
                ]

                print(
                    f"Clifford+T circuit fidelity: {fidelity_result['fidelity']:.15e} (method: {fidelity_result['method']}, computed in {format_time(time.time()-fidelity_start_time)})"
                )

                # Add additional info for rz_product_fidelity
                if fidelity_result["method"] == "rz_product_fidelity":
                    print(
                        f"  RZ gates processed: {fidelity_result.get('rz_gate_count', 'N/A')}"
                    )
                    if "individual_fidelities" in fidelity_result:
                        ind_fids = fidelity_result["individual_fidelities"]
                        if ind_fids:
                            print(
                                f"  Individual fidelity range: [{min(ind_fids):.15f}, {max(ind_fids):.15f}]"
                            )
            else:
                print(
                    f"Warning: Could not compute Gridsynth Clifford+T circuit fidelity: {fidelity_result['status']}"
                )

            print("\nStep 2: Converting to PBC...")
            # Pass the T-count from the C+T circuit to PBC for accurate reduction stats
            initial_t_for_pbc = current_pipeline_stats.get("total_t_family_count", 0)
            current_pipeline_stats["initial_clifford_t_t_gates_for_pbc"] = (
                initial_t_for_pbc
            )

            pbc_qc_from_gs, pbc_conv_stats = convert_to_pbc_circuit(
                gs_clifford_t_circuit, layering_method=args.layering_method
            )
            current_pipeline_stats["pbc_conversion_time"] = pbc_conv_stats.get(
                "pbc_conversion_time", 0
            )
            print(
                f"PBC conversion time: {format_time(current_pipeline_stats['pbc_conversion_time'])}"
            )
            current_pipeline_stats["pbc_conversion_stats_from_converter"] = (
                pbc_conv_stats  # Store raw stats
            )

            pbc_analysis_start_time = time.time()
            pbc_analysis_results = analyze_pbc_circuit(
                pbc_qc_from_gs, pbc_conversion_stats=pbc_conv_stats
            )
            current_pipeline_stats.update(pbc_analysis_results)
            current_pipeline_stats["analysis_pbc_time"] = (
                time.time() - pbc_analysis_start_time
            )
            print(
                f"PBC analysis time: {format_time(current_pipeline_stats['analysis_pbc_time'])}"
            )
            gs_results["pbc_stats"] = (
                pbc_analysis_results.copy()
            )  # Store for comparison
            gs_results["pbc_stats"][
                "initial_clifford_t_t_gates_for_pbc"
            ] = initial_t_for_pbc

            current_pipeline_stats["total_time"] = (
                time.time() - pipeline_total_start_time
            )
            gs_results["total_time"] = current_pipeline_stats["total_time"]
            print_circuit_stats(
                "Gridsynth Pipeline Full Analysis",
                current_pipeline_stats,
                show_detailed=True,
            )

        except Exception as e:
            print(f"Error in Gridsynth pipeline: {e}")
            import traceback

            traceback.print_exc()

    if args.pipeline in ["sk", "both"]:
        print("\n\n=================================================")
        print("=== Pipeline: Solovay-Kitaev based -> PBC ===")
        print("=================================================")
        pipeline_total_start_time = time.time()
        current_pipeline_stats = {}
        try:
            print("\nStep 1: Transpiling to Solovay-Kitaev Clifford+T...")
            ct_transpile_start_time = time.time()
            sk_clifford_t_circuit = transpile_to_solovay_kitaev_clifford_t(
                initial_qc_obj.copy(), recursion_degree=args.sk_recursion
            )
            current_pipeline_stats["transpilation_clifford_t_time"] = (
                time.time() - ct_transpile_start_time
            )
            print(
                f"Clifford+T transpilation time: {format_time(current_pipeline_stats['transpilation_clifford_t_time'])}"
            )

            ct_analysis_start_time = time.time()
            clifford_t_stats = analyze_clifford_t_circuit(sk_clifford_t_circuit)
            current_pipeline_stats["analysis_clifford_t_time"] = (
                time.time() - ct_analysis_start_time
            )
            print(
                f"Clifford+T analysis time: {format_time(current_pipeline_stats['analysis_clifford_t_time'])}"
            )
            current_pipeline_stats.update(clifford_t_stats)
            sk_results["clifford_t_stats"] = clifford_t_stats.copy()

            # Calculate fidelity for Clifford+T circuit using the new scalable method
            fidelity_start_time = time.time()
            fidelity_result = calculate_circuit_fidelity(
                initial_qc_for_unitary,
                sk_clifford_t_circuit,
                gridsynth_precision=args.gridsynth_precision,
            )

            if fidelity_result["fidelity"] is not None:
                current_pipeline_stats["fidelity"] = fidelity_result["fidelity"]
                current_pipeline_stats["fidelity_method"] = fidelity_result["method"]
                sk_results["clifford_t_stats"]["fidelity"] = fidelity_result["fidelity"]
                sk_results["clifford_t_stats"]["fidelity_method"] = fidelity_result[
                    "method"
                ]

                print(
                    f"Clifford+T circuit fidelity: {fidelity_result['fidelity']:.15e} (method: {fidelity_result['method']}, computed in {format_time(time.time()-fidelity_start_time)})"
                )

                # Add additional info for rz_product_fidelity
                if fidelity_result["method"] == "rz_product_fidelity":
                    print(
                        f"  RZ gates processed: {fidelity_result.get('rz_gate_count', 'N/A')}"
                    )
                    if "individual_fidelities" in fidelity_result:
                        ind_fids = fidelity_result["individual_fidelities"]
                        if ind_fids:
                            print(
                                f"  Individual fidelity range: [{min(ind_fids):.15f}, {max(ind_fids):.15f}]"
                            )
            else:
                print(
                    f"Warning: Could not compute Solovay-Kitaev Clifford+T circuit fidelity: {fidelity_result['status']}"
                )

            print("\nStep 2: Converting to PBC...")
            initial_t_for_pbc = current_pipeline_stats.get("total_t_family_count", 0)
            current_pipeline_stats["initial_clifford_t_t_gates_for_pbc"] = (
                initial_t_for_pbc
            )

            pbc_qc_from_sk, pbc_conv_stats = convert_to_pbc_circuit(
                sk_clifford_t_circuit, layering_method=args.layering_method
            )
            current_pipeline_stats["pbc_conversion_time"] = pbc_conv_stats.get(
                "pbc_conversion_time", 0
            )
            print(
                f"PBC conversion time: {format_time(current_pipeline_stats['pbc_conversion_time'])}"
            )
            current_pipeline_stats["pbc_conversion_stats_from_converter"] = (
                pbc_conv_stats
            )

            pbc_analysis_start_time = time.time()
            pbc_analysis_results = analyze_pbc_circuit(
                pbc_qc_from_sk, pbc_conversion_stats=pbc_conv_stats
            )
            current_pipeline_stats.update(pbc_analysis_results)
            current_pipeline_stats["analysis_pbc_time"] = (
                time.time() - pbc_analysis_start_time
            )
            print(
                f"PBC analysis time: {format_time(current_pipeline_stats['analysis_pbc_time'])}"
            )
            sk_results["pbc_stats"] = pbc_analysis_results.copy()
            sk_results["pbc_stats"][
                "initial_clifford_t_t_gates_for_pbc"
            ] = initial_t_for_pbc

            current_pipeline_stats["total_time"] = (
                time.time() - pipeline_total_start_time
            )
            sk_results["total_time"] = current_pipeline_stats["total_time"]
            print_circuit_stats(
                "Solovay-Kitaev Pipeline Full Analysis",
                current_pipeline_stats,
                show_detailed=True,
            )

        except Exception as e:
            print(f"Error in Solovay-Kitaev pipeline: {e}")
            import traceback

            traceback.print_exc()

    if args.pipeline == "both" and gs_results and sk_results:
        print_pipeline_comparison(gs_results, sk_results)

    print("\n--- FTCircuitBench Analysis Complete ---")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()  # Call your main analysis function

    profiler.disable()
    stats_file = "testing_profile.prof"
    profiler.dump_stats(stats_file)
    print("\n--- Profiling Complete ---")
    print(f"Profiling data saved to: {stats_file}")

    # Attempt to automatically open snakeviz
    import shutil
    import subprocess

    snakeviz_cmd = shutil.which("snakeviz")
    if snakeviz_cmd:
        print(f"Attempting to launch snakeviz with '{stats_file}'...")
        try:
            # Use Popen to launch snakeviz as a separate process
            # and not wait for it to complete.
            subprocess.Popen([snakeviz_cmd, stats_file])
            print("Snakeviz should open in your default web browser shortly.")
        except Exception as e:
            print(f"Could not automatically launch snakeviz: {e}")
            print("Please open it manually.")
            print(f"  Command: snakeviz {stats_file}")
    else:
        print("\nSnakeviz command not found in PATH.")
        print("To view stats graphically, install snakeviz (`pip install snakeviz`)")
        print(f"and then run: snakeviz {stats_file}")

    print("\nAlternatively, to view stats in Python interactive mode:")
    print("  import pstats")
    print(f"  p = pstats.Stats('{stats_file}')")
    print("  p.sort_stats('cumulative').print_stats(30) # Show top 30 cumulative time")
    print("  p.sort_stats('tottime').print_stats(30)    # Show top 30 self time")
