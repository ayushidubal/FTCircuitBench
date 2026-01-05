# ./ftcircuitbench/decomposer/decomposer.py
import re
import subprocess
from typing import Optional, Union

import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.circuit.library import RZGate

# For __main__ fidelity check:
from qiskit.quantum_info import Operator, process_fidelity

# --- Gridsynth CLI Based Decomposition ---


def _run_gridsynth_cli(angle: Union[float, str], precision: int = 10) -> str:
    """
    Decompose an angle into a series of S, H, T gates using the gridsynth CLI.

    Gridsynth must be installed and accessible in the system's PATH.
    It can parse numerical values and symbolic expressions like "pi/4".
    Scientific notation will be converted to decimal format.

    Note: This function is primarily used with string inputs in the codebase.
    Float inputs are supported mainly for testing purposes.

    :param angle: Angle in radians as a float or string (e.g., 0.785398, "pi/4").
                 String inputs are preferred, especially for symbolic expressions.
    :param precision: Number of digits of precision for gridsynth.
    :return: String of the gate sequence (e.g., "THTHTHS").
             The order of gates in the string is intended for direct application
             from left to right to construct the desired rotation.
    :raises RuntimeError: If gridsynth command fails, is not found, or returns an error.
    """
    try:
        # Convert float to string if needed (mainly for testing)
        if isinstance(angle, float):
            # Convert to decimal format with sufficient precision
            angle_str = f"{angle:.20f}"
        else:
            # For string inputs, only convert numeric strings to decimal format
            # Leave symbolic expressions (like "pi/4") as is
            if isinstance(angle, str) and "pi" not in angle.lower():
                try:
                    # Convert to float and then to decimal format
                    angle_str = f"{float(angle):.20f}"
                except ValueError:
                    # If conversion fails, use the string as is
                    angle_str = angle
            else:
                angle_str = angle

        # Construct Gridsynth command with parentheses around the angle
        # This prevents negative values from being interpreted as command line arguments
        cmd = f'gridsynth "({angle_str})" --digits={precision}'

        # Run Gridsynth
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"gridsynth command failed with error: {error_message}")
    except (ValueError, SyntaxError) as e:
        raise RuntimeError(f"Could not parse angle value '{angle}'. Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error running Gridsynth: {str(e)}")


def _run_gridsynth_cli_unpack(args):
    return _run_gridsynth_cli(*args)


def decompose_rz_gates_gridsynth(
    original_circuit: QuantumCircuit, precision: int = 10, progress_bar=None
) -> QuantumCircuit:
    """
    Decomposes all RZ gates in a quantum circuit into S, H, T (and possibly X) gates
    using the gridsynth CLI.

    Note:
    - Gridsynth must be installed and in the system PATH.
    - Circuit parameters in RZ gates must be bound to numerical values or
      expressions that gridsynth can parse (e.g., "pi/4"). Unbound Qiskit
      `ParameterExpression` objects (e.g., containing `Parameter('my_angle')`)
      will cause an error.
    - Identity operations (rotations of 0 degrees) are automatically removed.

    Args:
        original_circuit: The Qiskit QuantumCircuit to decompose.
        precision: Number of digits of precision for gridsynth.
        progress_bar: Optional tqdm progress bar to update during decomposition.

    Returns:
        A new QuantumCircuit with RZ gates replaced by their decompositions.

    Raises:
        RuntimeError: If gridsynth command fails or is not found.
        ValueError: If RZ gate parameters are unsuitable for gridsynth.
    """
    new_circuit = QuantumCircuit(
        *original_circuit.qregs,
        *original_circuit.cregs,
        name=(
            original_circuit.name + "_decomposed_rz"
            if original_circuit.name
            else "decomposed_rz"
        ),
    )

    ZERO_THRESHOLD = 10 ** (-precision)
    # Step 1: Collect all operations and RZ gate info
    ops_info = []  # (is_rz, op, qargs, cargs, rz_info)
    rz_jobs = []  # (index, qubit, theta_str)
    for idx, (op, qargs, cargs) in enumerate(original_circuit.data):
        if isinstance(op, RZGate):
            theta = op.params[0]
            qubit = qargs[0]
            # Convert angle to string
            if isinstance(theta, (int, float)):
                theta_str = f"{float(theta):.15g}"
            elif isinstance(theta, ParameterExpression):
                try:
                    theta_val = float(theta)
                    theta_str = f"{theta_val:.15g}"
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"RZ gate parameter {theta} is a ParameterExpression that "
                        f"could not be evaluated to a number. Gridsynth requires "
                        f"numerical values. Error: {e}"
                    )
            else:
                raise ValueError(
                    f"RZ gate parameter {theta} is of unsupported type {type(theta)}. "
                    f"Gridsynth requires numerical values."
                )
            # Skip identity
            if abs(float(theta_str)) < ZERO_THRESHOLD:
                if progress_bar:
                    progress_bar.update(1)
                ops_info.append((False, None, None, None, None))
                continue
            rz_jobs.append((idx, qubit, theta_str))
            ops_info.append((True, None, None, None, (qubit, theta_str)))
        else:
            ops_info.append((False, op, qargs, cargs, None))

    # Step 2: Process RZ gates
    rz_results = {}
    if rz_jobs:
        for idx, qubit, theta_str in rz_jobs:
            try:
                # Get gate sequence from gridsynth
                decomp_str = _run_gridsynth_cli(theta_str, precision)
                rz_results[idx] = (qubit, decomp_str)
                if progress_bar:
                    progress_bar.update(1)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decompose RZ gate at index {idx} with angle {theta_str}: {str(e)}"
                )

    # Step 3: Reconstruct circuit in original order
    for idx, (is_rz, op, qargs, cargs, rz_info) in enumerate(ops_info):
        if is_rz:
            qubit, theta_str = rz_info
            decomp_str = rz_results.get(idx, (None, ""))[1]
            # Skip if empty or only identity
            if not decomp_str or all(g in "IW" for g in decomp_str):
                if progress_bar:
                    progress_bar.update(1)
                continue
            for gate_char in decomp_str:
                if gate_char == "S":
                    new_circuit.s(qubit)
                elif gate_char == "T":
                    new_circuit.t(qubit)
                elif gate_char == "H":
                    new_circuit.h(qubit)
                elif gate_char == "X":
                    new_circuit.x(qubit)
                elif gate_char in "IW":
                    continue
                else:
                    print(
                        f"Warning: Encountered unexpected gate character '{gate_char}' "
                        f"in gridsynth output for Rz({theta_str}). This gate will be ignored."
                    )
            if progress_bar:
                progress_bar.update(1)
        elif op is not None:
            new_circuit.append(op, qargs, cargs)
    return new_circuit


def create_circuit_from_gate_string(gate_sequence: str) -> QuantumCircuit:
    """
    Generates a 1-qubit Qiskit circuit from a sequence of gate characters.
    Operators are applied from left to right as they appear in the string.
    e.g., "STH" will apply S, then T, then H. The resulting unitary is U_H U_T U_S.

    :param gate_sequence: String of gate characters (e.g., "SHTH").
                          Supported characters: S, T, H, X, Z, Y.
                          Case-insensitive for input characters.
    :return: A QuantumCircuit object with one qubit.
    :raises TypeError: if gate_sequence is not a string.
    :raises ValueError: if gate_sequence contains unsupported characters.
    """
    if not isinstance(gate_sequence, str):
        raise TypeError("gate_sequence must be a string.")

    qc = QuantumCircuit(1, name=f"seq_{gate_sequence[:10]}")

    gate_methods = {
        "S": qc.s,
        "T": qc.t,
        "H": qc.h,
        "X": qc.x,
        "Z": qc.z,
        "Y": qc.y,
        "W": qc.id,
    }

    for gate_char_orig in gate_sequence:
        gate_char = gate_char_orig.upper()
        method = gate_methods.get(gate_char)
        if method:
            method(0)
        else:
            raise ValueError(
                f"Unsupported gate character '{gate_char_orig}' in sequence. "
                f"Supported characters are: {', '.join(gate_methods.keys())}."
            )
    return qc


def parse_angle_from_gate_name(gate_name: str) -> Optional[float]:
    """
    Extracts and evaluates an angle from a gate name string like "rz(angle_expression)".
    Example: "rz(0.785)", "rz(Pi/4)", "rz(numpy.pi/2)".
    Uses `eval()` safely.
    """
    if not isinstance(gate_name, str):
        raise TypeError("gate_name must be a string")

    match = re.search(r"\(([^)]+)\)", gate_name)
    if match:
        angle_str = match.group(1)
        try:
            allowed_globals = {"__builtins__": {}}
            allowed_locals = {
                "pi": np.pi,
                "Pi": np.pi,
                "numpy": np,
                "np": np,
                "sqrt": np.sqrt,
                "cos": np.cos,
                "sin": np.sin,
                "exp": np.exp,
            }
            angle = eval(angle_str, allowed_globals, allowed_locals)
            return float(angle)
        except Exception:
            return None
    return None


# --- PyGridsynth Based Decomposition (Placeholders) ---


def _pygridsynth_decompose_angle(angle: float, precision_digits: int = 10) -> str:
    raise NotImplementedError(
        "PyGridsynth _pygridsynth_decompose_angle is not yet implemented."
    )


def decompose_rz_gates_pygridsynth(
    original_circuit: QuantumCircuit, precision_digits: int = 10
) -> QuantumCircuit:
    raise NotImplementedError(
        "PyGridsynth RZ gate decomposition (decompose_rz_gates_pygridsynth) "
        "is not yet implemented."
    )


if __name__ == "__main__":
    print("--- Running Decomposer Module Standalone Test ---")

    # --- Test decompose_rz_gates_gridsynth ---
    print("\nTesting RZ gate decomposition with gridsynth CLI...")
    qc_orig = QuantumCircuit(2, name="Original")
    qc_orig.h(0)
    qc_orig.rz(np.pi / 4, 0)
    qc_orig.cx(0, 1)
    qc_orig.rz(0.12345, 1)

    theta_param = Parameter("θ")
    qc_orig.rz(theta_param / 2, 0)
    qc_orig_bound = qc_orig.assign_parameters({theta_param: np.pi})

    print("\nOriginal Circuit (parameter bound):")
    print(qc_orig_bound.draw(output="text"))

    gridsynth_precision = 10  # Precision for gridsynth decomposition

    try:
        qc_decomposed = decompose_rz_gates_gridsynth(
            qc_orig_bound, precision=gridsynth_precision
        )
        print("\nDecomposed Circuit (using gridsynth):")
        print(qc_decomposed.draw(output="text"))
        print("\nGate counts in decomposed circuit:")
        print(qc_decomposed.count_ops())

        # --- Fidelity Validation Section ---
        print("\n\n--- Fidelity Validation for RZ Decomposition ---")
        test_angle = np.pi / 16  # Example: RZ(pi/8) is the T gate
        # test_angle = 0.123456789 # A more generic angle
        print(
            f"Target: RZ({test_angle:.4f}) with precision {gridsynth_precision} digits"
        )

        # 1. Ideal Rz unitary
        ideal_rz_qc = QuantumCircuit(1)
        ideal_rz_qc.rz(test_angle, 0)
        ideal_rz_unitary = Operator(ideal_rz_qc)
        # print("\nIdeal RZ Unitary:")
        # print(np.round(ideal_rz_unitary.data, 3))

        # 2. Decompose angle using _run_gridsynth_cli
        angle_str = str(
            test_angle
        )  # Or f"{test_angle/np.pi}*pi" if gridsynth prefers symbolic
        decomposed_sequence_str = _run_gridsynth_cli(
            angle_str, precision=gridsynth_precision
        )
        print(
            f"Gridsynth decomposed sequence for RZ({test_angle:.4f}): '{decomposed_sequence_str}'"
        )

        if not decomposed_sequence_str:
            print("Gridsynth returned an empty sequence. Cannot compute fidelity.")
        else:
            # 3. Circuit from decomposed sequence
            approx_qc = create_circuit_from_gate_string(decomposed_sequence_str)
            approx_unitary = Operator(approx_qc)
            print("\nApproximate Unitary from Decomposed Sequence:")
            print(np.round(approx_unitary.data, 3))

            # 4. Calculate Process Fidelity
            # For unitaries U, V, F_p = |Tr(U_ideal^dagger V_approx)|^2 / d^2
            # qiskit.quantum_info.process_fidelity handles this.
            # It computes state fidelity of Choi matrices: F_p(Λ, E) = F_s(J(Λ), J(E))
            # For unitary channels, this is equivalent to |Tr(U†V)|^2/d^2
            fid = process_fidelity(
                approx_unitary, ideal_rz_unitary, require_cp=False, require_tp=False
            )
            print(
                f"Process Fidelity between ideal RZ({test_angle:.4f}) and decomposition: {fid:.9f}"
            )

            # For reference, T gate is RZ(pi/4) up to global phase.
            # If test_angle = np.pi/4, sequence should be "T"
            if test_angle == np.pi / 4 and decomposed_sequence_str.upper() == "T":
                print(
                    "Matches T gate as expected (RZ(pi/4) -> T). Fidelity should be 1.0."
                )
            elif (
                test_angle == np.pi / 8 and decomposed_sequence_str.upper() == "T"
            ):  # Gridsynth T is Rz(pi/4)
                print(
                    "Note: Gridsynth 'T' is Rz(pi/4). For Rz(pi/8), sequence is different."
                )

    except RuntimeError as e:
        print(f"\nERROR during RZ decomposition or fidelity check: {e}")
        print(
            "Please ensure 'gridsynth' (the CLI tool) is installed and in your system PATH."
        )
    except ValueError as e:
        print(f"\nVALUE ERROR during RZ decomposition or fidelity check: {e}")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during RZ decomposition or fidelity check: {e}")

    # --- Test create_circuit_from_gate_string ---
    print("\n\nTesting create_circuit_from_gate_string...")
    gate_seq = "SHTHTS"
    try:
        seq_qc = create_circuit_from_gate_string(gate_seq)
        print(f"Circuit from sequence '{gate_seq}':")
        print(seq_qc.draw(output="text"))
    except Exception as e:
        print(f"ERROR creating circuit from string: {e}")

    gate_seq_invalid = "SHAZ"
    print(f"\nAttempting to create circuit from invalid sequence '{gate_seq_invalid}':")
    try:
        seq_qc_invalid = create_circuit_from_gate_string(gate_seq_invalid)
        print(seq_qc_invalid.draw(output="text"))
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")
    except Exception as e:
        print(f"Unexpected error for invalid sequence: {e}")

    # --- Test parse_angle_from_gate_name ---
    print("\n\nTesting parse_angle_from_gate_name...")
    gate_names_to_test = [
        "rz(pi/4)",
        "another_gate_type(0.123)",
        "rz( (numpy.pi * 2) / 8 )",
        "rz(sqrt(2)/2)",
        "rz(invalid_expression)",
        "not_an_rz_gate",
        "rz()",
    ]
    for name in gate_names_to_test:
        angle = parse_angle_from_gate_name(name)
        print(
            f"Parsed angle from '{name}': {angle if angle is not None else 'Could not parse'}"
        )

    print("\n--- Decomposer Module Standalone Test Complete ---")
