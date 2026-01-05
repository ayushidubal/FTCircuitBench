# examples/validate_clifford_t_decomposition.py
import sys

import numpy as np
from qiskit import QuantumCircuit

# Add the project root to the path
sys.path.append("..")
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Operator, process_fidelity

from ftcircuitbench import (
    MAX_QUBITS_FOR_FIDELITY,
    rz_product_fidelity,
    transpile_to_gridsynth_clifford_t,
    transpile_to_solovay_kitaev_clifford_t,
)

# --- Configuration ---
FIDELITY_THRESHOLD = 0.989  # Minimum acceptable fidelity for a "pass"

GRIDSYNTH_PRECISION = 10
SK_RECURSION_DEGREE = 3  # Increased for better fidelity in validation


def generate_test_circuits(
    num_qubits_list: list[int],
) -> list[tuple[str, QuantumCircuit]]:
    """Generates a list of test circuits."""
    circuits = []

    for nq in num_qubits_list:
        if nq < 1:
            continue

        # Circuit 1: Simple Rz
        qc_rz = QuantumCircuit(nq, name=f"simple_rz_n{nq}")
        qc_rz.rz(np.pi / 5, 0)
        if nq > 1:
            qc_rz.cx(0, 1 % nq)
        circuits.append((f"simple_rz_n{nq}", qc_rz))

        # Circuit 2: General U gate
        qc_u = QuantumCircuit(nq, name=f"general_u_n{nq}")
        qc_u.append(UGate(np.pi / 3, np.pi / 4, np.pi / 5), [0])
        if nq > 1:
            qc_u.h(1 % nq)
            qc_u.cx(0, 1 % nq)
        circuits.append((f"general_u_n{nq}", qc_u))

        # Circuit 3: More complex sequence
        if nq >= 2:
            qc_complex = QuantumCircuit(nq, name=f"complex_n{nq}")
            qc_complex.h(0)
            qc_complex.s(1)
            qc_complex.append(UGate(0.1, 0.2, 0.3), [0])
            qc_complex.cx(0, 1)
            qc_complex.rz(-np.pi / 4, 0)  # Tdg equivalent
            qc_complex.t(1)
            if nq >= 3:
                qc_complex.ccx(0, 1, 2)
            circuits.append((f"complex_n{nq}", qc_complex))

        # Circuit 4: Multiple RZ gates (new for larger circuits)
        if nq >= 3:
            qc_multiple_rz = QuantumCircuit(nq, name=f"multiple_rz_n{nq}")
            qc_multiple_rz.rz(np.pi / 6, 0)
            qc_multiple_rz.rz(np.pi / 8, 1)
            qc_multiple_rz.cx(0, 1)
            qc_multiple_rz.rz(np.pi / 12, 2)
            if nq >= 4:
                qc_multiple_rz.rz(np.pi / 10, 3)
                qc_multiple_rz.cx(1, 2)
            if nq >= 5:
                qc_multiple_rz.rz(np.pi / 16, 4)
                qc_multiple_rz.cx(2, 3)
            if nq >= 6:
                qc_multiple_rz.rz(np.pi / 20, 5)
                qc_multiple_rz.cx(3, 4)
            if nq >= 7:
                qc_multiple_rz.rz(np.pi / 24, 6)
                qc_multiple_rz.cx(4, 5)
            circuits.append((f"multiple_rz_n{nq}", qc_multiple_rz))

        # Circuit 5: Quantum Fourier Transform-like structure (new for larger circuits)
        if nq >= 4:
            qc_qft_like = QuantumCircuit(nq, name=f"qft_like_n{nq}")
            # Apply H and RZ gates in QFT pattern
            for i in range(nq):
                qc_qft_like.h(i)
                if i < nq - 1:
                    qc_qft_like.rz(np.pi / (2 ** (i + 1)), i)
                    qc_qft_like.cx(i, i + 1)
            # Add some additional rotations
            for i in range(min(nq, 3)):
                qc_qft_like.rz(np.pi / (3 ** (i + 1)), i)
            circuits.append((f"qft_like_n{nq}", qc_qft_like))

        # Circuit 6: Random rotation pattern (new for larger circuits)
        if nq >= 5:
            qc_random_rot = QuantumCircuit(nq, name=f"random_rot_n{nq}")
            # Apply random rotations to each qubit
            angles = [
                np.pi / 7,
                np.pi / 11,
                np.pi / 13,
                np.pi / 17,
                np.pi / 19,
                np.pi / 23,
                np.pi / 29,
            ]
            for i in range(min(nq, len(angles))):
                qc_random_rot.rz(angles[i], i)
                if i > 0:
                    qc_random_rot.cx(i - 1, i)
            # Add some controlled operations
            if nq >= 3:
                qc_random_rot.ccx(0, 1, 2)
            if nq >= 4:
                qc_random_rot.rz(np.pi / 31, 3)
                qc_random_rot.cx(2, 3)
            circuits.append((f"random_rot_n{nq}", qc_random_rot))

    # Circuit 7: All T-gates (should remain largely unchanged if basis includes t, tdg)
    qc_all_t = QuantumCircuit(1, name="all_t_n1")
    qc_all_t.t(0)
    qc_all_t.t(0)
    qc_all_t.tdg(0)  # Qiskit tdg is TdgGate
    circuits.append(("all_t_n1", qc_all_t))

    return circuits


def validate_decomposition(
    original_qc: QuantumCircuit,
    decomposed_qc: QuantumCircuit,
    method_name: str,
    original_unitary: Operator = None,
) -> dict:
    """
    Validates a single decomposition by comparing unitaries.
    """
    print(f"\n--- Validating: {method_name} ({original_qc.name}) ---")
    validation_results = {"method": method_name, "circuit_name": original_qc.name}

    print(f"Original circuit ops: {original_qc.count_ops()}")
    print(f"Decomposed circuit ops: {decomposed_qc.count_ops()}")

    if original_qc.num_qubits > MAX_QUBITS_FOR_FIDELITY:
        print(
            f"Skipping unitary validation for {original_qc.num_qubits}-qubit circuit (limit: {MAX_QUBITS_FOR_FIDELITY})."
        )
        validation_results["status"] = "skipped_large"
        validation_results["fidelity"] = None
        return validation_results

    if original_unitary is None:
        try:
            original_unitary = Operator(original_qc)
        except Exception as e:
            print(f"Error calculating original unitary: {e}")
            validation_results["status"] = "error_original_unitary"
            validation_results["fidelity"] = None
            return validation_results

    decomposed_unitary = None
    try:
        decomposed_unitary = Operator(decomposed_qc)
    except Exception as e:
        print(f"Error calculating decomposed unitary: {e}")
        validation_results["status"] = "error_decomposed_unitary"
        validation_results["fidelity"] = None
        return validation_results

    try:
        fidelity = process_fidelity(decomposed_unitary, original_unitary)
        validation_results["fidelity"] = fidelity
        if fidelity >= FIDELITY_THRESHOLD:
            validation_results["status"] = "pass"
            print(f"Fidelity: {fidelity:.9f} -> PASS")
        else:
            validation_results["status"] = "fail"
            print(f"Fidelity: {fidelity:.9f} -> FAIL (Threshold: {FIDELITY_THRESHOLD})")
            # For debugging failed cases:
            # print("Original Unitary:\n", np.round(original_unitary.data, 3))
            # print("Decomposed Unitary:\n", np.round(decomposed_unitary.data, 3))

    except Exception as e:
        print(f"Error calculating fidelity: {e}")
        validation_results["status"] = "error_fidelity_calc"
        validation_results["fidelity"] = None

    return validation_results


def main():
    print("--- Clifford+T Decomposition Validation Test ---")

    # Define qubit numbers for generated test circuits
    # Include larger circuits up to 7 qubits
    test_qubit_counts = [1, 2, 3, 4, 5, 6, 7]

    test_circuits_data = generate_test_circuits(test_qubit_counts)

    all_results = []
    rz_fidelity_results = []  # Store results for new RZ fidelity method

    for name, original_qc_template in test_circuits_data:
        print("\n\n=================================================")
        print(
            f"Testing Original Circuit: {name} ({original_qc_template.num_qubits} qubits)"
        )
        print("=================================================")

        # Create a fresh copy for each pipeline and remove measurements for unitary comparison
        original_qc = original_qc_template.copy()
        original_qc.remove_final_measurements(inplace=True)

        # --- NEW: Test RZ Decomposition Fidelity Method ---
        print("\n--- Testing New RZ Decomposition Fidelity Method ---")
        try:
            rz_fid_result = rz_product_fidelity(
                original_qc, gridsynth_precision=GRIDSYNTH_PRECISION
            )
            rz_fid_result["circuit_name"] = name
            rz_fid_result["method"] = "RZ Decomposition Fidelity"
            rz_fidelity_results.append(rz_fid_result)
        except Exception as e:
            print(f"ERROR in RZ fidelity calculation for {name}: {e}")
            rz_fidelity_results.append(
                {
                    "circuit_name": name,
                    "method": "RZ Decomposition Fidelity",
                    "overall_fidelity": None,
                    "status": "error",
                    "rz_gate_count": 0,
                }
            )

        # Pre-calculate original unitary if possible
        current_original_unitary = None
        if original_qc.num_qubits <= MAX_QUBITS_FOR_FIDELITY:
            try:
                current_original_unitary = Operator(original_qc)
            except Exception:  # Will be handled in validate_decomposition
                pass

        # --- Test Gridsynth-based Clifford+T pipeline ---
        try:
            # ensure_pbc_input_basis=False: gives C+T from Gridsynth directly.
            # ensure_pbc_input_basis=True: also ensures output gates are t, tdg (suitable for PBC)
            # Let's test the direct Gridsynth C+T output first.
            gs_ct_qc = transpile_to_gridsynth_clifford_t(
                original_qc.copy(),  # Pass a copy
                gridsynth_precision=GRIDSYNTH_PRECISION,
            )
            results_gs = validate_decomposition(
                original_qc, gs_ct_qc, "Gridsynth C+T", current_original_unitary
            )
            all_results.append(results_gs)

            # Optionally, also test the PBC-compatible version from Gridsynth path
            gs_ct_pbc_basis_qc = transpile_to_gridsynth_clifford_t(
                original_qc.copy(),
                gridsynth_precision=GRIDSYNTH_PRECISION,
            )
            results_gs_pbc_basis = validate_decomposition(
                original_qc,
                gs_ct_pbc_basis_qc,
                "Gridsynth C+T (PBC Basis)",
                current_original_unitary,
            )
            all_results.append(results_gs_pbc_basis)

        except Exception as e:
            print(f"ERROR in Gridsynth pipeline for {name}: {e}")
            all_results.append(
                {
                    "method": "Gridsynth C+T",
                    "circuit_name": name,
                    "status": "pipeline_error",
                    "fidelity": None,
                }
            )
            # import traceback; traceback.print_exc()

        # --- Test Solovay-Kitaev-based Clifford+T pipeline ---
        try:
            sk_ct_qc = transpile_to_solovay_kitaev_clifford_t(
                original_qc.copy(), recursion_degree=SK_RECURSION_DEGREE  # Pass a copy
            )
            results_sk = validate_decomposition(
                original_qc, sk_ct_qc, "Solovay-Kitaev C+T", current_original_unitary
            )
            all_results.append(results_sk)

        except Exception as e:
            print(f"ERROR in Solovay-Kitaev pipeline for {name}: {e}")
            all_results.append(
                {
                    "method": "Solovay-Kitaev C+T",
                    "circuit_name": name,
                    "status": "pipeline_error",
                    "fidelity": None,
                }
            )
            # import traceback; traceback.print_exc()

    # --- Summary of Results ---
    print("\n\n=================================================")
    print("Validation Summary:")
    print("=================================================")
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    error_count = 0

    for res in all_results:
        status_str = f"{res['method']} - {res['circuit_name']}: Status={res['status']}"
        if res["fidelity"] is not None:
            status_str += f", Fidelity={res['fidelity']:.7f}"
        print(status_str)

        if res["status"] == "pass":
            passed_count += 1
        elif res["status"] == "fail":
            failed_count += 1
        elif "skipped" in res["status"]:
            skipped_count += 1
        else:  # errors
            error_count += 1

    print(f"\nTotal Tests: {len(all_results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (due to size/errors): {skipped_count + error_count}")

    if failed_count > 0 or error_count > 0:
        print("\nSOME VALIDATIONS FAILED OR ENCOUNTERED ERRORS.")
    elif passed_count > 0:
        print("\nALL APPLICABLE VALIDATIONS PASSED.")
    else:
        print("\nNO VALIDATIONS WERE APPLICABLE OR COMPLETED SUCCESSFULLY.")

    # --- NEW: RZ Fidelity Method Summary ---
    print("\n\n=================================================")
    print("RZ Decomposition Fidelity Method Summary:")
    print("=================================================")

    for rz_res in rz_fidelity_results:
        status_str = f"{rz_res['method']} - {rz_res['circuit_name']}: "
        if rz_res["overall_fidelity"] is not None:
            status_str += f"Overall Fidelity={rz_res['overall_fidelity']:.7f}, "
            status_str += f"RZ Gates={rz_res['rz_gate_count']}, "
            avg_fid = rz_res.get("avg_individual_fidelity", "N/A")
            if isinstance(avg_fid, (int, float)):
                status_str += f"Avg Individual={avg_fid:.7f}"
            else:
                status_str += f"Avg Individual={avg_fid}"
        else:
            status_str += f"Status={rz_res.get('status', 'unknown')}"
        print(status_str)

    # --- Comparison Analysis ---
    print("\n\n=================================================")
    print("Fidelity Method Comparison:")
    print("=================================================")

    # Create a mapping of circuit names to their results
    circuit_results = {}
    for res in all_results:
        if res["circuit_name"] not in circuit_results:
            circuit_results[res["circuit_name"]] = {}
        circuit_results[res["circuit_name"]][res["method"]] = res

    for rz_res in rz_fidelity_results:
        circuit_name = rz_res["circuit_name"]
        if circuit_name in circuit_results:
            print(f"\nCircuit: {circuit_name}")
            print(f"  RZ Decomposition Fidelity: {rz_res['overall_fidelity']:.9f}")

            # Compare with other methods
            for method, res in circuit_results[circuit_name].items():
                if res["fidelity"] is not None:
                    print(f"  {method}: {res['fidelity']:.9f}")
                    if rz_res["overall_fidelity"] is not None:
                        diff = abs(rz_res["overall_fidelity"] - res["fidelity"])
                        print(f"    Difference: {diff:.9f}")
                else:
                    print(f"  {method}: {res['status']}")
        else:
            print(
                f"\nCircuit: {circuit_name} - RZ method only: {rz_res['overall_fidelity']:.9f}"
            )


if __name__ == "__main__":
    main()
