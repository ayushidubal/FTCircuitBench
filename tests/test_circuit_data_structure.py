#!/usr/bin/env python3
"""
Test script to demonstrate the structure of circuit.data for Clifford+T and PBC circuits.
"""

import sys

sys.path.append(".")

from ftcircuitbench import (
    analyze_clifford_t_circuit,
    analyze_pbc_circuit,
    convert_to_pbc_circuit,
    load_qasm_circuit,
    transpile_to_gridsynth_clifford_t,
)


def examine_circuit_data_structure(circuit, circuit_type):
    """
    Examine and print the structure of circuit.data for a given circuit.

    Args:
        circuit: QuantumCircuit object
        circuit_type: String describing the circuit type
    """
    print(f"\n{'='*60}")
    print(f"EXAMINING {circuit_type.upper()} CIRCUIT DATA STRUCTURE")
    print(f"{'='*60}")

    print("Circuit info:")
    print(f"  - Number of qubits: {circuit.num_qubits}")
    print(f"  - Total instructions: {len(circuit.data)}")
    print(f"  - Circuit depth: {circuit.depth()}")

    print("\nFirst 10 instructions in circuit.data:")
    print(f"{'Index':<6} {'Gate Name':<15} {'Qubits':<15} {'Type':<10}")
    print(f"{'-'*6} {'-'*15} {'-'*15} {'-'*10}")

    for i, instruction in enumerate(circuit.data[:10]):
        op_name = instruction.operation.name
        qargs = instruction.qubits
        qubit_indices = [circuit.find_bit(q).index for q in qargs]

        # Determine gate type
        gate_type = "Other"  # Default type

        if circuit_type == "Clifford+T":
            if op_name in ["t", "tdg"]:
                gate_type = "T-family"
            elif op_name in ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"]:
                gate_type = "Clifford"
            elif op_name in ["barrier", "id", "snapshot", "delay", "reset"]:
                gate_type = "Utility"
        elif circuit_type == "PBC":
            if op_name.startswith("R") and "(" in op_name:
                gate_type = "Rotation"
            elif op_name.startswith("Meas"):
                gate_type = "Measurement"
            elif op_name in ["barrier", "id"]:
                gate_type = "Utility"
        elif circuit_type == "Original":
            if op_name in ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"]:
                gate_type = "Clifford"
            elif op_name in ["rz", "ry", "rx"]:
                gate_type = "Rotation"
            elif op_name in ["barrier", "id", "snapshot", "delay", "reset"]:
                gate_type = "Utility"

        print(f"{i:<6} {op_name:<15} {str(qubit_indices):<15} {gate_type:<10}")

    if len(circuit.data) > 10:
        print(f"... and {len(circuit.data) - 10} more instructions")

    # Show some statistics
    print("\nGate distribution:")
    gate_counts = {}
    for instruction in circuit.data:
        op_name = instruction.operation.name
        gate_counts[op_name] = gate_counts.get(op_name, 0) + 1

    for gate_name, count in sorted(gate_counts.items()):
        print(f"  {gate_name}: {count}")

    # Show qubit interaction patterns
    print("\nQubit interaction patterns (first 5 two-qubit gates):")
    two_qubit_count = 0
    for instruction in circuit.data:
        if len(instruction.qubits) == 2 and two_qubit_count < 5:
            q1_idx = circuit.find_bit(instruction.qubits[0]).index
            q2_idx = circuit.find_bit(instruction.qubits[1]).index
            print(f"  {instruction.operation.name} on qubits ({q1_idx}, {q2_idx})")
            two_qubit_count += 1


def main():
    # Load a simple example circuit
    qasm_file = "qasm/example_circuit_for_comparison_2q.qasm"
    print(f"Loading circuit from: {qasm_file}")

    # Load original circuit
    original_circuit = load_qasm_circuit(qasm_file, is_file=True)
    examine_circuit_data_structure(original_circuit, "Original")

    # Transpile to Clifford+T
    print("\n🔄 Transpiling to Clifford+T basis...")
    intermediate_circuit, clifford_t_circuit = transpile_to_gridsynth_clifford_t(
        original_circuit.copy(),
        gridsynth_precision=3,
        return_intermediate=True,
    )
    examine_circuit_data_structure(clifford_t_circuit, "Clifford+T")

    # Convert to PBC
    print("\n🔄 Converting to PBC format...")
    pbc_circuit, pbc_stats = convert_to_pbc_circuit(
        clifford_t_circuit,
        optimize_t_maxiter=2,
        if_print_rpc=False,
        layering_method="v2",
        parallel=True,
    )
    examine_circuit_data_structure(pbc_circuit, "PBC")

    # Show analysis results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")

    clifford_t_analysis = analyze_clifford_t_circuit(clifford_t_circuit)
    print("\nClifford+T Analysis:")
    print(f"  - T gates: {clifford_t_analysis.get('t_count', 0)}")
    print(f"  - T† gates: {clifford_t_analysis.get('tdg_count', 0)}")
    print(f"  - Total T-family: {clifford_t_analysis.get('total_t_family_count', 0)}")
    print(f"  - Clifford gates: {clifford_t_analysis.get('clifford_gate_count', 0)}")
    print(f"  - Total gates: {clifford_t_analysis.get('total_gate_count', 0)}")

    pbc_analysis = analyze_pbc_circuit(pbc_circuit, pbc_stats)
    print("\nPBC Analysis:")
    print(f"  - T operators: {pbc_analysis.get('pbc_t_operators', 0)}")
    print(
        f"  - Measurement operators: {pbc_analysis.get('pbc_measurement_operators', 0)}"
    )
    print(
        f"  - Average Pauli weight: {pbc_analysis.get('pbc_avg_pauli_weight', 'N/A')}"
    )
    print(f"  - Total gates: {len(pbc_circuit.data)}")


if __name__ == "__main__":
    main()
