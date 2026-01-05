#!/usr/bin/env python3
"""
Script to show the actual rotation and measurement layers from a PBC circuit.
This displays the specific Pauli operators in each layer.
"""

import sys

# Import the correct parser from the analyzer module
from ftcircuitbench.analyzer.pbc_analyzer import parse_pbc_gate_name
from ftcircuitbench.parser import load_qasm_circuit

# Import the new PBC file reader functions
from ftcircuitbench.pbc_converter import (
    analyze_pbc_file_content,
    convert_to_pbc_circuit,
    parse_pauli_string,
    read_combined_pbc_file,
    validate_pbc_file,
)
from ftcircuitbench.transpilers.gs_transpiler import transpile_to_gridsynth_clifford_t


def show_pbc_layers_from_file(pbc_file_path: str):
    """
    Show PBC layers from a combined PBC file.

    Args:
        pbc_file_path (str): Path to the combined PBC file
    """
    print("=== PBC Layer Analysis from File ===")
    print(f"Input file: {pbc_file_path}")
    print()

    # Validate the file first
    is_valid, errors = validate_pbc_file(pbc_file_path)
    if not is_valid:
        print("✗ PBC file is invalid:")
        for error in errors:
            print(f"  - {error}")
        return

    print("✓ PBC file is valid")
    print()

    # Read and analyze the file
    pbc_data = read_combined_pbc_file(pbc_file_path)
    analysis = analyze_pbc_file_content(pbc_data)

    print("=" * 60)
    print("PBC FILE STRUCTURE")
    print("=" * 60)

    print(f"Sections found: {', '.join(pbc_data['sections_found'])}")
    print(f"Number of T-layers: {analysis['num_t_layers']}")
    print(f"Number of measurement operators: {analysis['num_measurement_operators']}")
    print(f"Total Pauli operators: {analysis['total_pauli_operators']}")
    print()

    # Display T-layers
    if pbc_data["t_layers"]:
        print("T-LAYERS:")
        print("-" * 30)

        for layer_idx, layer in enumerate(pbc_data["t_layers"]):
            print(f"\nLayer {layer_idx} ({len(layer)} operators):")

            if not layer:
                print("  (empty layer)")
                continue

            for op_idx, pauli_str in enumerate(layer):
                parsed = parse_pauli_string(pauli_str)
                print(
                    f"  {op_idx:3d}. {pauli_str:15s} (weight: {parsed['weight']}, qubits: {parsed['qubit_indices']})"
                )

    # Display measurement basis
    if pbc_data["measurement_basis"]:
        print("\nMEASUREMENT BASIS:")
        print("-" * 30)

        for op_idx, pauli_str in enumerate(pbc_data["measurement_basis"]):
            parsed = parse_pauli_string(pauli_str)
            print(
                f"  {op_idx:3d}. {pauli_str:15s} (weight: {parsed['weight']}, qubits: {parsed['qubit_indices']})"
            )

    # Display statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print("Pauli Weight Distribution:")
    for weight, count in sorted(analysis["pauli_weight_distribution"].items()):
        print(f"  Weight {weight}: {count} operators")

    print("\nQubit Interaction Degree:")
    for qubit_idx, count in sorted(analysis["qubit_interaction_degree"].items()):
        print(f"  Qubit {qubit_idx}: {count} interactions")

    if analysis["t_layer_sizes"]:
        print("\nT-Layer Sizes:")
        print(
            f"  Average: {sum(analysis['t_layer_sizes']) / len(analysis['t_layer_sizes']):.2f}"
        )
        print(f"  Min: {min(analysis['t_layer_sizes'])}")
        print(f"  Max: {max(analysis['t_layer_sizes'])}")
        print(f"  Distribution: {analysis['t_layer_sizes']}")


def show_pbc_layers_corrected(
    qasm_file_path, gridsynth_precision=10, layering_method="bare"
):
    """
    Load a QASM file, convert it to PBC, and show the actual rotation and measurement layers.
    Corrected version.
    Args:
        qasm_file_path (str): Path to the QASM file
        gridsynth_precision (int): Gridsynth precision for RZ decomposition
        layering_method (str): PBC layering method ('bare' or 'v2')
    """
    print("=== PBC Layer Analysis (Corrected) ===")
    print(f"Input file: {qasm_file_path}")
    print(f"Gridsynth precision: {gridsynth_precision}")
    print(f"Layering method: {layering_method}")
    print()

    # Step 1: Load and transpile to Clifford+T
    print("Step 1: Loading and transpiling to Clifford+T...")
    circuit = load_qasm_circuit(qasm_file_path, is_file=True)
    clifford_t_circuit = transpile_to_gridsynth_clifford_t(
        circuit, gridsynth_precision=gridsynth_precision
    )

    # Step 2: Convert to PBC
    print("Step 2: Converting to PBC...")
    pbc_circuit, pbc_conversion_stats = convert_to_pbc_circuit(
        clifford_t_circuit, layering_method=layering_method
    )

    # Analyze the PBC circuit to get overall counts (optional here, but good for verification)
    # pbc_analysis_stats = analyze_pbc_circuit(pbc_circuit, pbc_conversion_stats)
    # print(f"Total rotation ops from analyzer: {pbc_analysis_stats.get('pbc_t_operators', 0)}")
    # print(f"Total measurement ops from analyzer: {pbc_analysis_stats.get('pbc_measurement_operators', 0)}")

    print("\n" + "=" * 60)
    print("PBC CIRCUIT STRUCTURE (Grouped by Barriers)")
    print("=" * 60)

    # Extract and display layers based on barriers
    grouped_layers = []
    current_group = []

    for instruction in pbc_circuit.data:
        op = instruction.operation
        op_name = op.name

        if op_name == "barrier":
            if current_group:  # Save previous group if not empty
                grouped_layers.append(current_group)
            current_group = []  # Start a new group
            continue  # Don't add barrier itself to the group content

        qargs = instruction.qubits
        op_type, pauli_str_in_name, params = parse_pbc_gate_name(op_name)

        # The pauli_str_in_name from the correct parser is just the Pauli letters (e.g., "XYZ")
        # For weight calculation, count non-Identity characters if 'I' can be in pauli_str_in_name
        # Assuming parse_pbc_gate_name returns active Paulis (no 'I's), len is weight.
        # If 'I's can be present, weight should be sum(1 for c in pauli_str_in_name if c != 'I')
        # The provided parse_pbc_gate_name from analyzer is good.

        # Get qubit indices
        qubit_indices = [pbc_circuit.find_bit(q).index for q in qargs]

        # Store processed gate info
        current_group.append(
            {
                "type": op_type,
                "pauli": pauli_str_in_name,
                "params": params[0] if params else None,  # params is a list from parser
                "qubits": qubit_indices,
                "name": op_name,
            }
        )

    if current_group:  # Add the last group if any
        grouped_layers.append(current_group)

    rotation_operator_count = 0
    measurement_operator_count = 0

    for i, group in enumerate(grouped_layers):
        print(f"\nSegment {i+1} (contains {len(group)} operations):")
        segment_type = "Mixed"
        if group:
            # Check if all ops in group are same type (typically rotation or measurement)
            first_op_type = group[0]["type"]
            if all(op_info["type"] == first_op_type for op_info in group):
                segment_type = first_op_type.capitalize()

        print(f"  Type: {segment_type}")
        for op_info in group:
            if op_info["type"] == "rotation":
                rotation_operator_count += 1
                print(
                    f"    ROTATION: Pauli {op_info['pauli']} on qubits {op_info['qubits']} with angle {op_info['params']}"
                )
            elif op_info["type"] == "measurement":
                measurement_operator_count += 1
                print(
                    f"    MEASUREMENT: Pauli {op_info['pauli']} on qubits {op_info['qubits']}"
                )
            elif op_info["type"] == "utility":
                print(f"    UTILITY: {op_info['name']} on qubits {op_info['qubits']}")
            else:
                print(
                    f"    UNKNOWN: {op_info['name']} (Pauli: {op_info['pauli']}, Qubits: {op_info['qubits']})"
                )

    print("\n" + "=" * 60)
    print("LAYER STATISTICS (from show_pbc_layers.py)")
    print("=" * 60)
    print(f"Total segments (groups separated by barriers): {len(grouped_layers)}")
    print(f"Total rotation operators found: {rotation_operator_count}")
    print(f"Total measurement operators found: {measurement_operator_count}")

    # Pauli weight distribution
    rotation_weights = []
    measurement_weights = []
    for group in grouped_layers:
        for op_info in group:
            # Weight is the number of non-Identity Paulis.
            # The parse_pbc_gate_name in analyzer.py already gives a pauli_str_in_name
            # that consists of X, Y, Z for the active qubits.
            # If the pbm.py ensures no 'I's in the pauli string for the gate name,
            # then len(op_info['pauli']) is the weight.
            # Otherwise, count non-'I's if 'I's could be part of the string.
            # Let's assume op_info['pauli'] is the compact string like "XZ" not "IXIZ".
            weight = len(
                op_info["pauli"]
            )  # Or sum(1 for c in op_info['pauli'] if c != 'I')
            if op_info["type"] == "rotation":
                rotation_weights.append(weight)
            elif op_info["type"] == "measurement":
                measurement_weights.append(weight)

    if rotation_weights:
        print("\nRotation operator Pauli weights:")
        weight_counts = {}
        for weight in rotation_weights:
            weight_counts[weight] = weight_counts.get(weight, 0) + 1
        for weight, count in sorted(weight_counts.items()):
            print(f"  Weight {weight}: {count} operators")

    if measurement_weights:
        print("\nMeasurement operator Pauli weights:")
        weight_counts = {}
        for weight in measurement_weights:
            weight_counts[weight] = weight_counts.get(weight, 0) + 1
        for weight, count in sorted(weight_counts.items()):
            print(f"  Weight {weight}: {count} operators")

    # Qubit interaction patterns
    all_qubits_involved = set()
    qubit_interactions_map = {q_idx: 0 for q_idx in range(pbc_circuit.num_qubits)}

    for group in grouped_layers:
        for op_info in group:
            if op_info["type"] in ["rotation", "measurement"]:
                for q_idx in op_info["qubits"]:
                    all_qubits_involved.add(q_idx)
                    if q_idx in qubit_interactions_map:
                        qubit_interactions_map[q_idx] += 1

    if all_qubits_involved:
        print("\nQubit participation in PBC operators:")
        for qubit_idx in sorted(list(all_qubits_involved)):
            print(
                f"  Qubit {qubit_idx}: involved in {qubit_interactions_map[qubit_idx]} operators"
            )


# Keep the raw display function if needed
def show_raw_pbc_circuit(
    qasm_file_path, gridsynth_precision=10, layering_method="bare"
):
    # ... (content remains the same, ensure it uses the correct parse_pbc_gate_name)
    print("\n" + "=" * 60)
    print("RAW PBC CIRCUIT STRUCTURE")
    print("=" * 60)

    # Load and convert to PBC
    circuit = load_qasm_circuit(qasm_file_path, is_file=True)
    clifford_t_circuit = transpile_to_gridsynth_clifford_t(
        circuit, gridsynth_precision=gridsynth_precision
    )
    pbc_circuit, _ = convert_to_pbc_circuit(
        clifford_t_circuit, layering_method=layering_method
    )

    print(f"Total PBC operations: {len(pbc_circuit.data)}")
    print(f"Circuit depth: {pbc_circuit.depth()}")
    print(f"Number of qubits: {pbc_circuit.num_qubits}")

    print("\nOperation sequence:")
    for i, instruction in enumerate(pbc_circuit.data):
        op = instruction.operation
        op_name = op.name
        qargs = instruction.qubits
        qubit_indices = [pbc_circuit.find_bit(q).index for q in qargs]

        op_type, pauli_str_in_name, params = parse_pbc_gate_name(
            op_name
        )  # Uses imported correct parser

        param_str = params[0] if params else ""
        if param_str:
            print(
                f"  {i+1:3d}. {op_type.upper():12s}: Pauli {pauli_str_in_name:8s} on qubits {qubit_indices} (angle: {param_str})"
            )
        else:
            print(
                f"  {i+1:3d}. {op_type.upper():12s}: Pauli {pauli_str_in_name:8s} on qubits {qubit_indices}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python show_pbc_layers.py <qasm_file> [gridsynth_precision] [layering_method]"
        )
        print(
            "       python show_pbc_layers.py <qasm_file> --raw  # Show raw circuit structure"
        )
        print(
            "       python show_pbc_layers.py --file <pbc_file>  # Show PBC layers from file"
        )
        sys.exit(1)

    # Check if we're reading from a PBC file
    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Error: --file requires a PBC file path")
            sys.exit(1)
        pbc_file = sys.argv[2]
        show_pbc_layers_from_file(pbc_file)
    else:
        qasm_file = sys.argv[1]
        show_raw_flag = "--raw" in sys.argv

        if show_raw_flag:
            # Determine gridsynth_precision and layering_method, allowing them to be optional
            gs_prec_val = 10
            layer_method_val = "bare"

            non_flag_args = [arg for arg in sys.argv[2:] if arg != "--raw"]
            if len(non_flag_args) > 0:
                try:
                    gs_prec_val = int(non_flag_args[0])
                except ValueError:
                    print(
                        f"Warning: Could not parse '{non_flag_args[0]}' as Gridsynth precision. Using default {gs_prec_val}."
                    )
            if len(non_flag_args) > 1:
                if non_flag_args[1] in ["bare", "v2"]:
                    layer_method_val = non_flag_args[1]
                else:
                    print(
                        f"Warning: Invalid layering method '{non_flag_args[1]}'. Using default '{layer_method_val}'."
                    )

            show_raw_pbc_circuit(qasm_file, gs_prec_val, layer_method_val)
        else:
            gridsynth_precision = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            layering_method = sys.argv[3] if len(sys.argv) > 3 else "bare"
            show_pbc_layers_corrected(qasm_file, gridsynth_precision, layering_method)
