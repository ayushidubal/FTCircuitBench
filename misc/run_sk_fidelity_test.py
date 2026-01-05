#!/usr/bin/env python3
import os
from typing import List

os.environ["PATH"] = (
    os.path.expanduser("~/.cabal/bin") + ":" + os.environ.get("PATH", "")
)

from ftcircuitbench import (
    MAX_QUBITS_FOR_FIDELITY,
    calculate_circuit_fidelity,
    load_qasm_circuit,
    transpile_to_solovay_kitaev_clifford_t,
)


def run_for_file(qasm_path: str, degrees: List[int]) -> None:
    print(f"Testing SK fidelity on: {qasm_path}")
    original = load_qasm_circuit(qasm_path, is_file=True)
    original_no_meas = original.copy()
    try:
        original_no_meas.remove_final_measurements(inplace=True)
    except Exception:
        pass

    for d in degrees:
        print(f"  - SK recursion degree {d}")
        # Get intermediate and final SK circuit
        intermediate_rz, sk_ct = transpile_to_solovay_kitaev_clifford_t(
            original_no_meas.copy(), recursion_degree=d, return_intermediate=True
        )

        # Small circuits: unitary-based; Large: SK RZ-product via intermediate
        if original_no_meas.num_qubits <= MAX_QUBITS_FOR_FIDELITY:
            res = calculate_circuit_fidelity(
                original_no_meas, sk_ct, gridsynth_precision=3
            )
        else:
            res = calculate_circuit_fidelity(
                original_no_meas,
                sk_ct,
                gridsynth_precision=3,
                sk_recursion_degree=d,
                intermediate_qc=intermediate_rz,
            )

        fid = res.get("fidelity", "N/A")
        method = res.get("method", "?")
        try:
            fid_str = f"{float(fid):.6f}"
        except Exception:
            fid_str = str(fid)
        print(f"    fidelity={fid_str}  method={method}")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    adder_10q = os.path.join(root, "qasm", "adder", "adder_10q.qasm")
    qft_18q = os.path.join(root, "qasm", "qft", "qft_18q.qasm")

    # Prefer small circuit to see unitary-based fidelity differences clearly
    target = adder_10q if os.path.exists(adder_10q) else qft_18q
    run_for_file(target, degrees=[1, 2])


if __name__ == "__main__":
    main()
