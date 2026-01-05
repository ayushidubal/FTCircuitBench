# ftcircuitbench/reports/summary_markdown.py (FINAL CORRECTED VERSION)

import numbers
from typing import Any, Dict, Optional

from ftcircuitbench.benchmark_utils import (
    format_time,
    get_clifford_rows,
    get_general_info,
    get_interaction_graph_rows,
    percent_reduction,
)

# This file now only contains the markdown generation function.
# The generate_comparison_summary_string function is not used by generate_benchmarks.py,
# so it has been removed to simplify. If needed, it can be added back.


def generate_summary_markdown(
    stats: Dict[str, Any],
    circuit_name: str,
    pipeline_type: str,
    parameter_value: Optional[str] = None,
) -> str:
    def fmt(val, width=8):
        if isinstance(val, float):
            return f"{val:.2f}".rjust(width)
        if isinstance(val, int):
            return f"{val}".rjust(width)
        return str(val).rjust(width)

    general_info = get_general_info(stats)
    clifford_rows = get_clifford_rows(stats)

    # --- PBC Circuit Optimization Summary Table ---
    pbc_rows = [
        [
            "Rotation Operators",
            stats.get("pre_opt_rotation_operators", "N/A"),
            stats.get("pbc_rotation_operators", "N/A"),
            percent_reduction(
                stats.get("pre_opt_rotation_operators"),
                stats.get("pbc_rotation_operators"),
            ),
        ],
        [
            "Rotation Layers",
            stats.get("pre_opt_rotation_layers", "N/A"),
            stats.get("pbc_rotation_layers", "N/A"),
            percent_reduction(
                stats.get("pre_opt_rotation_layers"), stats.get("pbc_rotation_layers")
            ),
        ],
        [
            "Avg Rotations per Layer",
            stats.get("pre_opt_avg_rotations_per_layer", "N/A"),
            stats.get("pbc_avg_rotations_per_layer", "N/A"),
            percent_reduction(
                stats.get("pre_opt_avg_rotations_per_layer"),
                stats.get("pbc_avg_rotations_per_layer"),
            ),
        ],
        [
            "Measurement Operators",
            stats.get("pre_opt_measurement_operators", "N/A"),
            stats.get("pbc_measurement_operators", "N/A"),
            percent_reduction(
                stats.get("pre_opt_measurement_operators"),
                stats.get("pbc_measurement_operators"),
            ),
        ],
        ["", "", "", ""],
        [
            "Avg Operator Pauli Weight",
            stats.get("pre_opt_avg_operator_pauli_weight", "N/A"),
            stats.get("pbc_avg_operator_pauli_weight", "N/A"),
            percent_reduction(
                stats.get("pre_opt_avg_operator_pauli_weight"),
                stats.get("pbc_avg_operator_pauli_weight"),
            ),
        ],
        [
            "Std Operator Pauli Weight",
            stats.get("pre_opt_std_operator_pauli_weight", "N/A"),
            stats.get("pbc_std_operator_pauli_weight", "N/A"),
            percent_reduction(
                stats.get("pre_opt_std_operator_pauli_weight"),
                stats.get("pbc_std_operator_pauli_weight"),
            ),
        ],
        [
            "Max Operator Pauli Weight",
            stats.get("pre_opt_max_operator_pauli_weight", "N/A"),
            stats.get("pbc_max_operator_pauli_weight", "N/A"),
            percent_reduction(
                stats.get("pre_opt_max_operator_pauli_weight"),
                stats.get("pbc_max_operator_pauli_weight"),
            ),
        ],
        ["", "", "", ""],
        [
            "Avg Qubit Interaction Degree",
            stats.get("pre_opt_avg_qubit_interaction_degree", "N/A"),
            stats.get("pbc_avg_qubit_interaction_degree", "N/A"),
            percent_reduction(
                stats.get("pre_opt_avg_qubit_interaction_degree"),
                stats.get("pbc_avg_qubit_interaction_degree"),
            ),
        ],
        [
            "Std Qubit Interaction Degree",
            stats.get("pre_opt_std_qubit_interaction_degree", "N/A"),
            stats.get("pbc_std_qubit_interaction_degree", "N/A"),
            percent_reduction(
                stats.get("pre_opt_std_qubit_interaction_degree"),
                stats.get("pbc_std_qubit_interaction_degree"),
            ),
        ],
        [
            "Max Qubit Interaction Degree",
            stats.get("pre_opt_max_qubit_interaction_degree", "N/A"),
            stats.get("pbc_max_qubit_interaction_degree", "N/A"),
            percent_reduction(
                stats.get("pre_opt_max_qubit_interaction_degree"),
                stats.get("pbc_max_qubit_interaction_degree"),
            ),
        ],
    ]
    # Note: Graph density is not included for PBC circuits as it's not a relevant statistic
    # due to the potential for many interactions on the same edge

    # --- Timings Table ---
    timing_rows = [
        [
            "Clifford+T Transpilation",
            format_time(stats.get("transpilation_clifford_t_time", 0.0)),
        ],
        ["PBC Conversion (Total)", format_time(stats.get("pbc_conversion_time", 0.0))],
    ]

    # --- Markdown Output ---
    md = []
    md.append(f"# {circuit_name} - {pipeline_type} Pipeline Summary")
    if parameter_value:
        md.append(f"**Parameter:** {parameter_value}")
    md.append("")
    md.append("## General Circuit Info")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    for row in general_info:
        md.append(f"| {row[0]} | {row[1]} |")
    md.append("")

    if clifford_rows:
        md.append("## Clifford+T Circuit Summary")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|---|---|")
        for row in clifford_rows:
            md.append(f"| {row[0]} | {row[1]} |")
        md.append("")

    # --- Clifford+T Interaction Graph Statistics ---
    clifford_interaction_rows = get_interaction_graph_rows(stats, "")
    if clifford_interaction_rows:
        md.append("## Clifford+T Interaction Graph Statistics")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|---|---|")
        for row in clifford_interaction_rows:
            md.append(f"| {row[0]} | {row[1]} |")
        md.append("")

    # --- PBC Interaction Graph Statistics ---
    pbc_interaction_rows = get_interaction_graph_rows(stats, "pbc_")
    if pbc_interaction_rows:
        md.append("## PBC Interaction Graph Statistics")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|---|---|")
        for row in pbc_interaction_rows:
            md.append(f"| {row[0]} | {row[1]} |")
        md.append("")

    md.append("## PBC Circuit Optimization Summary")
    md.append("")
    md.append("| Metric | Value Pre Opt | Value Post Opt | % Reduction |")
    md.append("|---|---|---|---|")
    for row in pbc_rows:
        # Format numbers for better readability in the table
        pre_val_str = fmt(row[1], 8) if isinstance(row[1], numbers.Number) else row[1]
        post_val_str = fmt(row[2], 8) if isinstance(row[2], numbers.Number) else row[2]
        md.append(f"| {row[0]} | {pre_val_str} | {post_val_str} | {row[3]} |")
    md.append("")

    if timing_rows:
        md.append("## Pipeline Timings")
        md.append("")
        md.append("| Stage | Time |")
        md.append("|---|---|")
        for row in timing_rows:
            md.append(f"| {row[0]} | {row[1]} |")
        md.append("")

    return "\n".join(md)
