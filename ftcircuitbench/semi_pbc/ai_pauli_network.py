from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import math
import sys
import time
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.transpiler import PassManager

from ftcircuitbench.semi_pbc.pbc_input import PBCProgram

_TWO_QUBIT_OPS = {"cx", "cz", "ecr", "swap", "iswap", "rxx", "ryy", "rzz"}
_SUPPORTED_TOPOLOGIES = {
    ("line", 4): [(0, 1), (1, 2), (2, 3)],
    ("line", 5): [(0, 1), (1, 2), (2, 3), (3, 4)],
    ("line", 6): [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    ("t", 4): [(0, 1), (1, 2), (1, 3)],
    ("t", 5): [(0, 1), (1, 2), (2, 3), (1, 4)],
    ("t", 6): [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)],
    ("y", 6): [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)],
}


@dataclass(frozen=True)
class PauliNetworkWindow:
    source_path: str
    start_op_id: int
    stop_op_id: int
    source_ids: tuple[str, ...]
    original_qubits: tuple[int, ...]
    num_qubits: int
    signed_paulis: tuple[str, ...]
    total_pauli_weight: int
    multi_qubit_terms: int
    topology: str
    coupling_map: tuple[tuple[int, int], ...]


def ai_pauli_network_dependency_status() -> dict[str, Any]:
    qiskit_version = _package_version("qiskit")
    transpiler_version = _package_version("qiskit-ibm-transpiler")
    if importlib.util.find_spec("qiskit_ibm_transpiler") is None:
        return {
            "available": False,
            "qiskit_version": qiskit_version,
            "qiskit_ibm_transpiler_version": transpiler_version,
            "reason": "missing qiskit_ibm_transpiler",
        }
    _install_qiskit_compatibility_shims()
    try:
        __import__("qiskit_ibm_transpiler")
    except Exception as exc:  # noqa: BLE001 - optional dependency probes must not fail importers.
        return {
            "available": False,
            "qiskit_version": qiskit_version,
            "qiskit_ibm_transpiler_version": transpiler_version,
            "reason": f"qiskit_ibm_transpiler import failed: {exc}",
        }
    return {
        "available": True,
        "qiskit_version": qiskit_version,
        "qiskit_ibm_transpiler_version": transpiler_version,
        "reason": "",
    }


def build_pauli_network_circuit(
    *,
    num_qubits: int,
    signed_paulis: Sequence[str],
) -> QuantumCircuit:
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError("num_qubits must be an integer >= 1")

    circuit = QuantumCircuit(num_qubits)
    for signed_pauli in signed_paulis:
        sign, pauli = _parse_signed_pauli(signed_pauli, num_qubits)
        _append_pauli_rotation(circuit, pauli, sign)
    return circuit


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    counts = {name: int(count) for name, count in sorted(circuit.count_ops().items())}
    two_qubit_ops = sum(
        count for name, count in counts.items() if name in _TWO_QUBIT_OPS
    )
    return {
        "num_qubits": circuit.num_qubits,
        "ops": len(circuit.data),
        "depth": circuit.depth(),
        "two_qubit_ops": two_qubit_ops,
        "counts": counts,
    }


def run_ai_pauli_network_synthesis(
    *,
    num_qubits: int,
    signed_paulis: Sequence[str],
    backend_name: str | None = None,
    coupling_map: list[tuple[int, int]] | None = None,
    local_mode: bool = True,
    replace_only_if_better: bool = True,
    max_threads: int | None = None,
) -> dict[str, Any]:
    circuit = build_pauli_network_circuit(
        num_qubits=num_qubits,
        signed_paulis=signed_paulis,
    )
    before = circuit_metrics(circuit)
    status = ai_pauli_network_dependency_status()
    if not status["available"]:
        return {
            "status": "skipped",
            "dependency": status,
            "before": before,
            "after": None,
            "seconds": 0.0,
            "error": status["reason"],
        }

    try:
        _install_qiskit_compatibility_shims()
        from qiskit_ibm_transpiler.ai.collection import CollectPauliNetworks
        from qiskit_ibm_transpiler.ai.synthesis import AIPauliNetworkSynthesis

        if backend_name is None and coupling_map is None:
            coupling_map = _all_to_all_coupling_map(num_qubits)
        kwargs: dict[str, Any] = {
            "local_mode": local_mode,
            "replace_only_if_better": replace_only_if_better,
        }
        if backend_name is not None:
            kwargs["backend_name"] = backend_name
        if coupling_map is not None:
            kwargs["coupling_map"] = coupling_map
        if max_threads is not None:
            kwargs["max_threads"] = max_threads

        pass_manager = PassManager(
            [
                CollectPauliNetworks(),
                AIPauliNetworkSynthesis(**kwargs),
            ]
        )
        start = time.perf_counter()
        optimized = pass_manager.run(circuit)
        seconds = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - report pass failures as experiment data.
        return {
            "status": "failed",
            "dependency": status,
            "before": before,
            "after": None,
            "seconds": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    after = circuit_metrics(optimized)
    return {
        "status": "ok",
        "dependency": status,
        "before": before,
        "after": after,
        "seconds": seconds,
        "error": "",
        "changed": after != before,
        "optimized_qasm": qasm2.dumps(optimized),
    }


def extract_supported_rotation_windows(
    program: PBCProgram,
    *,
    source_path: str | Path,
    max_windows: int,
    window_terms: int = 4,
    topology: str = "line",
) -> list[PauliNetworkWindow]:
    if max_windows < 1:
        return []
    if window_terms < 1:
        raise ValueError("window_terms must be >= 1")
    if topology not in {key[0] for key in _SUPPORTED_TOPOLOGIES}:
        raise ValueError(f"unsupported topology {topology!r}")

    rotations = [op for op in program.ops if op.op == "t_pauli"]
    windows: list[PauliNetworkWindow] = []
    for start in range(len(rotations) - window_terms + 1):
        chunk = rotations[start : start + window_terms]
        original_qubits = tuple(
            sorted(
                {
                    int(qubit[1:])
                    for op in chunk
                    for qubit, _pauli in op.term.pairs
                    if qubit.startswith("q")
                }
            )
        )
        coupling_map = _SUPPORTED_TOPOLOGIES.get((topology, len(original_qubits)))
        if coupling_map is None:
            continue
        reindex = {qubit: index for index, qubit in enumerate(original_qubits)}
        signed_paulis = tuple(
            _compress_source_pauli(op.term.to_full_width(program.data_qubits), reindex)
            for op in chunk
        )
        windows.append(
            PauliNetworkWindow(
                source_path=str(source_path),
                start_op_id=chunk[0].id,
                stop_op_id=chunk[-1].id,
                source_ids=tuple(op.source_id for op in chunk),
                original_qubits=original_qubits,
                num_qubits=len(original_qubits),
                signed_paulis=signed_paulis,
                total_pauli_weight=sum(op.term.weight for op in chunk),
                multi_qubit_terms=sum(op.term.weight > 1 for op in chunk),
                topology=topology,
                coupling_map=tuple(coupling_map),
            )
        )
    return sorted(
        windows,
        key=lambda window: (
            -window.total_pauli_weight,
            -window.multi_qubit_terms,
            window.start_op_id,
        ),
    )[:max_windows]


def _append_pauli_rotation(circuit: QuantumCircuit, pauli: str, sign: int) -> None:
    active = [index for index, label in enumerate(pauli) if label != "I"]
    if not active:
        raise ValueError("identity Pauli rotations are not supported")

    for qubit in active:
        label = pauli[qubit]
        if label == "X":
            circuit.h(qubit)
        elif label == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)

    target = active[0]
    for qubit in active[1:]:
        circuit.cx(qubit, target)
    circuit.rz(sign * math.pi / 4, target)
    for qubit in reversed(active[1:]):
        circuit.cx(qubit, target)

    for qubit in reversed(active):
        label = pauli[qubit]
        if label == "X":
            circuit.h(qubit)
        elif label == "Y":
            circuit.h(qubit)
            circuit.s(qubit)


def _parse_signed_pauli(signed_pauli: str, num_qubits: int) -> tuple[int, str]:
    if not isinstance(signed_pauli, str) or len(signed_pauli) != num_qubits + 1:
        raise ValueError("signed Pauli must have one sign and num_qubits Pauli labels")
    sign_label = signed_pauli[0]
    if sign_label not in {"+", "-"}:
        raise ValueError("signed Pauli must start with '+' or '-'")
    pauli = signed_pauli[1:]
    invalid = sorted(set(pauli) - {"I", "X", "Y", "Z"})
    if invalid:
        raise ValueError(f"unsupported Pauli labels: {invalid}")
    return (1 if sign_label == "+" else -1), pauli


def _compress_source_pauli(signed_pauli: str, reindex: dict[int, int]) -> str:
    compressed = ["I"] * len(reindex)
    for original_index, compressed_index in reindex.items():
        compressed[compressed_index] = signed_pauli[original_index + 1]
    return signed_pauli[0] + "".join(compressed)


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _all_to_all_coupling_map(num_qubits: int) -> list[tuple[int, int]]:
    return [
        (control, target)
        for control in range(num_qubits)
        for target in range(num_qubits)
        if control != target
    ]


def _install_qiskit_compatibility_shims() -> None:
    module_name = "qiskit.synthesis.linear.linear_matrix_utils"
    if importlib.util.find_spec(module_name) is not None:
        return
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
    module.random_invertible_binary_matrix = _random_invertible_binary_matrix
    sys.modules[module_name] = module


def _random_invertible_binary_matrix(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        matrix = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        if _binary_rank(matrix) == n:
            return matrix


def _binary_rank(matrix: np.ndarray) -> int:
    echelon = matrix.copy()
    rows, columns = echelon.shape
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, rows) if echelon[row, column]),
            None,
        )
        if pivot is None:
            continue
        if pivot != rank:
            echelon[[rank, pivot]] = echelon[[pivot, rank]]
        for row in range(rows):
            if row != rank and echelon[row, column]:
                echelon[row] ^= echelon[rank]
        rank += 1
    return rank
