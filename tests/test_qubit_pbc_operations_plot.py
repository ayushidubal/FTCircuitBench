#!/usr/bin/env python3
"""
Test script for the new show_qubit_pbc_operations_plot function.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from ftcircuitbench.analyzer.visualization import show_qubit_pbc_operations_plot
from ftcircuitbench.pbc_converter.pbc_circuit_reader import read_combined_pbc_file


def create_test_pbc_circuit():
    """Create a simple test PBC circuit with proper PBC gate names."""
    qc = QuantumCircuit(6, 6)

    # Create PBC-style rotation gates
    def create_pbc_rotation(pauli_str, angle):
        """Create a PBC rotation gate with proper name."""
        gate = Gate(f"R{pauli_str}({angle})", len(pauli_str), [])
        return gate

    def create_pbc_measurement(pauli_str):
        """Create a PBC measurement gate with proper name."""
        gate = Gate(f"Meas{pauli_str}", len(pauli_str), [])
        return gate

    # Add single-qubit PBC rotations
    qc.append(create_pbc_rotation("X", "pi/8"), [0])
    qc.append(create_pbc_rotation("Y", "pi/6"), [1])
    qc.append(create_pbc_rotation("Z", "pi/4"), [2])

    # Add two-qubit PBC rotations
    qc.append(create_pbc_rotation("XX", "pi/8"), [0, 1])
    qc.append(create_pbc_rotation("YY", "pi/6"), [1, 2])
    qc.append(create_pbc_rotation("ZZ", "pi/4"), [2, 3])
    qc.append(create_pbc_rotation("XY", "pi/8"), [3, 4])
    qc.append(create_pbc_rotation("XZ", "pi/6"), [4, 5])

    # Add three-qubit PBC rotation
    qc.append(create_pbc_rotation("XYZ", "pi/8"), [0, 1, 2])

    # Add single-qubit measurements
    qc.append(create_pbc_measurement("X"), [0])
    qc.append(create_pbc_measurement("Y"), [1])
    qc.append(create_pbc_measurement("Z"), [2])

    # Add two-qubit measurements
    qc.append(create_pbc_measurement("XX"), [3, 4])
    qc.append(create_pbc_measurement("YY"), [4, 5])

    # Add three-qubit measurement
    qc.append(create_pbc_measurement("XYZ"), [0, 1, 2])

    return qc


def create_pbc_circuit_from_file(pbc_file_path):
    """Create a QuantumCircuit from PBC file data for testing."""
    try:
        # Read PBC file
        pbc_data = read_combined_pbc_file(pbc_file_path)

        # Count total qubits needed (estimate from Pauli strings)
        max_qubits = 0
        for layer in pbc_data.get("t_layers", []):
            for pauli_str in layer:
                max_qubits = max(max_qubits, len(pauli_str))

        for pauli_str in pbc_data.get("measurement_basis", []):
            max_qubits = max(max_qubits, len(pauli_str))

        if max_qubits == 0:
            max_qubits = 4  # Default fallback

        # Create circuit
        qc = QuantumCircuit(max_qubits, max_qubits)

        # Create PBC-style gates based on actual PBC data
        def create_pbc_rotation(pauli_str, angle="pi/8"):
            """Create a PBC rotation gate with proper name."""
            # Remove sign from Pauli string for gate name
            clean_pauli = pauli_str.lstrip("+-")
            gate = Gate(f"R{clean_pauli}({angle})", len(clean_pauli), [])
            return gate

        def create_pbc_measurement(pauli_str):
            """Create a PBC measurement gate with proper name."""
            # Remove sign from Pauli string for gate name
            clean_pauli = pauli_str.lstrip("+-")
            gate = Gate(f"Meas{clean_pauli}", len(clean_pauli), [])
            return gate

        # Add rotation operations from T-layers
        for layer_idx, layer in enumerate(pbc_data.get("t_layers", [])):
            for pauli_str in layer:
                if pauli_str and pauli_str != "(empty layer)":
                    # Create qubit list for this Pauli string
                    clean_pauli = pauli_str.lstrip("+-")
                    qubits = list(range(len(clean_pauli)))
                    qc.append(create_pbc_rotation(pauli_str), qubits)

        # Add measurement operations from measurement basis
        for pauli_str in pbc_data.get("measurement_basis", []):
            if pauli_str:
                clean_pauli = pauli_str.lstrip("+-")
                qubits = list(range(len(clean_pauli)))
                qc.append(create_pbc_measurement(pauli_str), qubits)

        return qc, pbc_data

    except Exception as e:
        print(f"Error reading PBC file: {e}")
        return None, None


def main():
    """Test the new visualization function."""
    print("=== Testing with synthetic PBC circuit ===")
    circuit = create_test_pbc_circuit()

    print(f"Circuit has {circuit.num_qubits} qubits")
    print(f"Circuit has {len(circuit.data)} operations")

    # Print some gate names to verify they're PBC-style
    print("Sample gate names:")
    for i, instruction in enumerate(circuit.data[:5]):
        print(f"  {i}: {instruction.operation.name}")

    print("\nTesting show_qubit_pbc_operations_plot...")
    show_qubit_pbc_operations_plot(
        circuit,
        title="Test PBC Operations per Qubit (Synthetic)",
        figsize=(10, 6),
        color="coral",
        show_stats=True,
        show_grid=True,
    )

    print("\n" + "=" * 50)
    print("=== Testing with actual PBC file ===")

    # Try to find and use an actual PBC file
    pbc_file_path = "circuit_benchmarks/adder/adder_4q/GS/precision_level_3/adder_4q_gs_prec3_pbc_post_opt.txt"

    if os.path.exists(pbc_file_path):
        print(f"Using PBC file: {pbc_file_path}")
        circuit, pbc_data = create_pbc_circuit_from_file(pbc_file_path)

        if circuit:
            print(f"Created circuit with {circuit.num_qubits} qubits")
            print(f"PBC data sections: {pbc_data.get('sections_found', [])}")

            # Print some gate names to verify they're PBC-style
            print("Sample gate names from PBC file:")
            for i, instruction in enumerate(circuit.data[:5]):
                print(f"  {i}: {instruction.operation.name}")

            show_qubit_pbc_operations_plot(
                circuit,
                title="PBC Operations per Qubit (Real Data)",
                figsize=(10, 6),
                color="steelblue",
                show_stats=True,
                show_grid=True,
            )
        else:
            print("Failed to create circuit from PBC file")
    else:
        print(f"PBC file not found: {pbc_file_path}")
        print("Skipping real data test")


if __name__ == "__main__":
    main()
