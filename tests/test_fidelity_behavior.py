#!/usr/bin/env python3
"""
Comprehensive test script to verify fidelity behavior.
Tests both unitary-based fidelity (small circuits) and RZ product fidelity (large circuits)
with different precision levels and circuit types.
"""

import sys

sys.path.append(".")

import numpy as np
from qiskit import QuantumCircuit

from ftcircuitbench import (
    calculate_circuit_fidelity,
    load_qasm_circuit,
    transpile_to_gridsynth_clifford_t,
)


def test_small_circuit_unitary_fidelity():
    """Test unitary-based fidelity on small circuits."""
    print("=" * 60)
    print("TESTING SMALL CIRCUIT (UNITARY-BASED FIDELITY)")
    print("=" * 60)

    # Create a small circuit with RZ gates
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rz(np.pi / 3, 0)
    qc.cx(0, 1)
    qc.rz(np.pi / 4, 1)

    print(f"Original circuit: {qc.count_ops()}")

    # Test with different precision levels
    for precision in [1, 3, 5, 10]:
        print(f"\n--- Testing precision {precision} ---")

        # Get both intermediate and final circuits
        intermediate_qc, final_qc = transpile_to_gridsynth_clifford_t(
            qc, gridsynth_precision=precision, return_intermediate=True
        )

        print(f"Intermediate circuit: {intermediate_qc.count_ops()}")
        print(f"Final circuit: {final_qc.count_ops()}")

        # Test old fidelity method (should work for small circuits)
        try:
            old_result = calculate_circuit_fidelity(
                qc, final_qc, gridsynth_precision=precision
            )
            if isinstance(old_result["fidelity"], (int, float)):
                print(
                    f"Old method fidelity: {old_result['fidelity']:.6f} ({old_result['method']})"
                )
            else:
                print(
                    f"Old method fidelity: {old_result['fidelity']} ({old_result['method']})"
                )
        except Exception as e:
            print(f"Old method failed: {e}")

        # Test new fidelity method with intermediate circuit
        try:
            new_result = calculate_circuit_fidelity(
                qc,
                final_qc,
                gridsynth_precision=precision,
                intermediate_qc=intermediate_qc,
            )
            if isinstance(new_result["fidelity"], (int, float)):
                print(
                    f"New method fidelity: {new_result['fidelity']:.6f} ({new_result['method']})"
                )
            else:
                print(
                    f"New method fidelity: {new_result['fidelity']} ({new_result['method']})"
                )
        except Exception as e:
            print(f"New method failed: {e}")


def test_large_circuit_rz_fidelity():
    """Test RZ product fidelity on large circuits with custom gates."""
    print("\n" + "=" * 60)
    print("TESTING LARGE CIRCUIT (RZ PRODUCT FIDELITY)")
    print("=" * 60)

    # Load the adder circuit (has custom gates)
    try:
        qc = load_qasm_circuit("qasm/adder/adder_10q.qasm", is_file=True)
        qc.remove_final_measurements(inplace=True)
        print(f"Original circuit: {qc.count_ops()}")
        print(f"Circuit qubits: {qc.num_qubits}")

        # Test with different precision levels
        for precision in [1, 3, 5, 10]:
            print(f"\n--- Testing precision {precision} ---")

            # Get both intermediate and final circuits
            intermediate_qc, final_qc = transpile_to_gridsynth_clifford_t(
                qc, gridsynth_precision=precision, return_intermediate=True
            )

            print(f"Intermediate circuit: {intermediate_qc.count_ops()}")
            print(f"Final circuit: {final_qc.count_ops()}")

            # Count RZ gates in intermediate circuit
            rz_count = sum(
                1
                for instruction in intermediate_qc.data
                if instruction.operation.name == "rz"
            )
            print(f"RZ gates in intermediate circuit: {rz_count}")

            # Test old fidelity method (should return N/A for large circuits without intermediate)
            try:
                old_result = calculate_circuit_fidelity(
                    qc, final_qc, gridsynth_precision=precision
                )
                if isinstance(old_result["fidelity"], (int, float)):
                    print(
                        f"Old method fidelity: {old_result['fidelity']:.6f} ({old_result['method']})"
                    )
                else:
                    print(
                        f"Old method fidelity: {old_result['fidelity']} ({old_result['method']})"
                    )
                print(f"Old method RZ count: {old_result.get('rz_gate_count', 'N/A')}")
            except Exception as e:
                print(f"Old method failed: {e}")

            # Test new fidelity method with intermediate circuit
            try:
                new_result = calculate_circuit_fidelity(
                    qc,
                    final_qc,
                    gridsynth_precision=precision,
                    intermediate_qc=intermediate_qc,
                )
                if isinstance(new_result["fidelity"], (int, float)):
                    print(
                        f"New method fidelity: {new_result['fidelity']:.6f} ({new_result['method']})"
                    )
                else:
                    print(
                        f"New method fidelity: {new_result['fidelity']} ({new_result['method']})"
                    )
                print(f"New method RZ count: {new_result.get('rz_gate_count', 'N/A')}")

                # Check if fidelity decreases with lower precision (only for numeric values)
                if isinstance(new_result["fidelity"], (int, float)):
                    if precision == 1:
                        low_precision_fidelity = new_result["fidelity"]
                    elif precision == 10:
                        high_precision_fidelity = new_result["fidelity"]
                        if low_precision_fidelity < high_precision_fidelity:
                            print(
                                "✅ Fidelity correctly decreases with lower precision!"
                            )
                        else:
                            print("❌ Fidelity should decrease with lower precision")

            except Exception as e:
                print(f"New method failed: {e}")

    except Exception as e:
        print(f"Error loading adder circuit: {e}")


def test_circuit_with_explicit_rz():
    """Test with a circuit that has explicit RZ gates."""
    print("\n" + "=" * 60)
    print("TESTING CIRCUIT WITH EXPLICIT RZ GATES")
    print("=" * 60)

    # Create a circuit with explicit RZ gates
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.rz(np.pi / 3, 0)
    qc.rz(np.pi / 4, 1)
    qc.cx(0, 1)
    qc.rz(np.pi / 6, 2)
    qc.cx(1, 2)

    print(f"Original circuit: {qc.count_ops()}")
    print(f"Circuit qubits: {qc.num_qubits}")

    # Test with different precision levels
    for precision in [1, 3, 5, 10]:
        print(f"\n--- Testing precision {precision} ---")

        # Get both intermediate and final circuits
        intermediate_qc, final_qc = transpile_to_gridsynth_clifford_t(
            qc, gridsynth_precision=precision, return_intermediate=True
        )

        print(f"Intermediate circuit: {intermediate_qc.count_ops()}")
        print(f"Final circuit: {final_qc.count_ops()}")

        # Count RZ gates in intermediate circuit
        rz_count = sum(
            1
            for instruction in intermediate_qc.data
            if instruction.operation.name == "rz"
        )
        print(f"RZ gates in intermediate circuit: {rz_count}")

        # Test both methods
        try:
            old_result = calculate_circuit_fidelity(
                qc, final_qc, gridsynth_precision=precision
            )
            if isinstance(old_result["fidelity"], (int, float)):
                print(
                    f"Old method fidelity: {old_result['fidelity']:.6f} ({old_result['method']})"
                )
            else:
                print(
                    f"Old method fidelity: {old_result['fidelity']} ({old_result['method']})"
                )
            print(f"Old method RZ count: {old_result.get('rz_gate_count', 'N/A')}")
        except Exception as e:
            print(f"Old method failed: {e}")

        try:
            new_result = calculate_circuit_fidelity(
                qc,
                final_qc,
                gridsynth_precision=precision,
                intermediate_qc=intermediate_qc,
            )
            if isinstance(new_result["fidelity"], (int, float)):
                print(
                    f"New method fidelity: {new_result['fidelity']:.6f} ({new_result['method']})"
                )
            else:
                print(
                    f"New method fidelity: {new_result['fidelity']} ({new_result['method']})"
                )
            print(f"New method RZ count: {new_result.get('rz_gate_count', 'N/A')}")
        except Exception as e:
            print(f"New method failed: {e}")


def test_edge_cases():
    """Test edge cases for fidelity calculation."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # Test empty circuit
    print("\n--- Empty Circuit ---")
    qc_empty = QuantumCircuit(2)
    intermediate_qc, final_qc = transpile_to_gridsynth_clifford_t(
        qc_empty, return_intermediate=True
    )

    try:
        result = calculate_circuit_fidelity(
            qc_empty, final_qc, intermediate_qc=intermediate_qc
        )
        print(f"Empty circuit fidelity: {result['fidelity']:.6f}")
    except Exception as e:
        print(f"Empty circuit test failed: {e}")

    # Test circuit with only Clifford gates
    print("\n--- Clifford-Only Circuit ---")
    qc_clifford = QuantumCircuit(2)
    qc_clifford.h(0)
    qc_clifford.cx(0, 1)
    qc_clifford.s(1)

    intermediate_qc, final_qc = transpile_to_gridsynth_clifford_t(
        qc_clifford, return_intermediate=True
    )

    try:
        result = calculate_circuit_fidelity(
            qc_clifford, final_qc, intermediate_qc=intermediate_qc
        )
        print(f"Clifford-only fidelity: {result['fidelity']:.6f}")
    except Exception as e:
        print(f"Clifford-only test failed: {e}")


def main():
    """Run all fidelity tests."""
    print("🚀 COMPREHENSIVE FIDELITY BEHAVIOR TEST")
    print(
        "Testing fidelity calculation with different circuit types and precision levels"
    )

    # Test small circuits (unitary-based fidelity)
    test_small_circuit_unitary_fidelity()

    # Test large circuits with custom gates (RZ product fidelity)
    test_large_circuit_rz_fidelity()

    # Test circuits with explicit RZ gates
    test_circuit_with_explicit_rz()

    # Test edge cases
    test_edge_cases()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
