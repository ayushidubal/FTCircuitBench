from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import io
import math
import os
import sys
import time
import types
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Operator, Pauli
from qiskit.transpiler import PassManager

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pbc_input import PBCProgram

_TWO_QUBIT_OPS = {"cx", "cz", "ecr", "swap", "iswap", "rxx", "ryy", "rzz"}
_AI_ROTATION_MARKER = 0x80000000
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


@dataclass(frozen=True)
class PauliNetworkRegion:
    source_path: str
    start_op_id: int
    stop_op_id: int
    source_ids: tuple[str, ...]
    num_qubits: int
    signed_paulis: tuple[str, ...]
    total_pauli_weight: int
    multi_qubit_terms: int
    topology: str
    coupling_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PauliReplayEmission:
    prefix_length: int
    op: str
    qubit: int
    rotation_index: int
    phase_mult: int
    emitted_pauli: str


@dataclass(frozen=True)
class PauliReplaySnapshot:
    prefix_length: int
    pending_terms: tuple[str, ...]
    pending_weights: tuple[int, ...]
    pending_indices: tuple[int, ...]


@dataclass(frozen=True)
class PauliReplayResult:
    snapshots: tuple[PauliReplaySnapshot, ...]
    emissions: tuple[PauliReplayEmission, ...]


@dataclass(frozen=True)
class AITrajectoryPrefixOps:
    ops: list[SemiPBCOp]
    source_indices_by_output_id: dict[int, int]


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


def decode_ai_pauli_solution(
    encoded_solution: Sequence[int],
) -> list[tuple[str, int, int, int]]:
    decoded = []
    axis_names = ("rx", "ry", "rz")
    for value in encoded_solution:
        if value >= _AI_ROTATION_MARKER:
            axis_code = (value >> 21) & 0x3
            if axis_code >= len(axis_names):
                raise ValueError(f"invalid AI Pauli rotation axis code {axis_code}")
            qubit = (value >> 11) & 0x3FF
            rotation_index = (value >> 1) & 0x3FF
            phase_mult = 1 if value & 1 else -1
            decoded.append((axis_names[axis_code], qubit, rotation_index, phase_mult))
        else:
            decoded.append(("gate", int(value), 0, 0))
    return decoded


def replay_ai_pauli_solution(
    *,
    num_qubits: int,
    signed_paulis: Sequence[str],
    gateset: Sequence[tuple[str, Sequence[int]]],
    solution: Sequence[int] | Sequence[tuple[str, int, int, int]],
) -> PauliReplayResult:
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    decoded_solution = _normalise_ai_pauli_solution(solution)
    pending = [_canonical_signed_pauli(pauli, num_qubits) for pauli in signed_paulis]
    emitted: set[int] = set()
    emissions: list[PauliReplayEmission] = []
    snapshots = [_pauli_replay_snapshot(0, pending, emitted)]

    for prefix_length, (op, arg1, arg2, arg3) in enumerate(decoded_solution, start=1):
        if op == "gate":
            if arg1 < 0 or arg1 >= len(gateset):
                raise ValueError(f"gate action index {arg1} is outside the gateset")
            clifford = _clifford_for_ai_pauli_gate(num_qubits, gateset[arg1])
            for index, pauli in enumerate(pending):
                if index not in emitted:
                    pending[index] = _evolve_signed_pauli(pauli, clifford)
        elif op in {"rx", "ry", "rz"}:
            qubit, rotation_index, phase_mult = arg1, arg2, arg3
            if rotation_index < 0 or rotation_index >= len(pending):
                raise ValueError(f"rotation index {rotation_index} is out of range")
            if rotation_index in emitted:
                raise ValueError(f"rotation index {rotation_index} was already emitted")
            expected_op, expected_qubit, expected_phase = _single_rotation_for_pauli(
                pending[rotation_index]
            )
            if (op, qubit, phase_mult) != (
                expected_op,
                expected_qubit,
                expected_phase,
            ):
                raise ValueError(
                    "rotation emission does not match pending Pauli: "
                    f"got {(op, qubit, phase_mult)}, "
                    f"expected {(expected_op, expected_qubit, expected_phase)}"
                )
            emissions.append(
                PauliReplayEmission(
                    prefix_length=prefix_length,
                    op=op,
                    qubit=qubit,
                    rotation_index=rotation_index,
                    phase_mult=phase_mult,
                    emitted_pauli=pending[rotation_index],
                )
            )
            emitted.add(rotation_index)
        else:
            raise ValueError(f"unsupported AI Pauli solution op {op!r}")
        snapshots.append(_pauli_replay_snapshot(prefix_length, pending, emitted))

    return PauliReplayResult(
        snapshots=tuple(snapshots),
        emissions=tuple(emissions),
    )


def find_k_terminal_prefix(
    replay: PauliReplayResult,
    *,
    k: int,
) -> PauliReplaySnapshot:
    if k < 1:
        raise ValueError("k must be >= 1")
    for snapshot in replay.snapshots:
        if all(weight <= k for weight in snapshot.pending_weights):
            return snapshot
    raise ValueError(f"no replay prefix has all pending Pauli weights <= {k}")


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


def circuits_equivalent(
    original: QuantumCircuit,
    candidate: QuantumCircuit,
) -> bool:
    if original.num_qubits != candidate.num_qubits:
        return False
    return bool(Operator(original).equiv(Operator(candidate)))


def run_ai_pauli_network_synthesis(
    *,
    num_qubits: int,
    signed_paulis: Sequence[str],
    backend_name: str | None = None,
    coupling_map: list[tuple[int, int]] | None = None,
    local_mode: bool = True,
    replace_only_if_better: bool = True,
    max_threads: int | None = None,
    capture_pass_output: bool = False,
) -> dict[str, Any]:
    circuit = build_pauli_network_circuit(
        num_qubits=num_qubits,
        signed_paulis=signed_paulis,
    )
    return run_ai_pauli_network_synthesis_on_circuit(
        circuit=circuit,
        backend_name=backend_name,
        coupling_map=coupling_map,
        local_mode=local_mode,
        replace_only_if_better=replace_only_if_better,
        max_threads=max_threads,
        capture_pass_output=capture_pass_output,
    )


def run_ai_pauli_network_synthesis_on_circuit(
    *,
    circuit: QuantumCircuit,
    backend_name: str | None = None,
    coupling_map: list[tuple[int, int]] | None = None,
    local_mode: bool = True,
    replace_only_if_better: bool = True,
    max_threads: int | None = None,
    capture_pass_output: bool = False,
) -> dict[str, Any]:
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
            coupling_map = _all_to_all_coupling_map(circuit.num_qubits)
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
        if capture_pass_output:
            with _capture_process_output() as output:
                optimized = pass_manager.run(circuit)
        else:
            output = io.StringIO()
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
        "pass_output": output.getvalue() if capture_pass_output else "",
    }


def extract_ai_pauli_trajectory(
    *,
    circuit: QuantumCircuit,
    coupling_map: Sequence[tuple[int, int]],
    qargs: Sequence[int] | None = None,
    deterministic: bool = False,
    num_searches: int = 10,
    num_mcts_searches: int = 0,
    c: float = 2**0.5,
    max_expand_depth: int = 1,
) -> dict[str, Any]:
    before = circuit_metrics(circuit)
    status = ai_pauli_network_dependency_status()
    if not status["available"]:
        return {
            "status": "skipped",
            "dependency": status,
            "before": before,
            "raw_actions": None,
            "decoded_solution": None,
            "gateset": None,
            "num_qubits": circuit.num_qubits,
            "replay_terms": None,
            "rotation_angle_signs": None,
            "solver_output_lines": 0,
            "solver_output_excerpt": "",
            "error": status["reason"],
        }

    replay_terms = None
    rotation_angle_signs = None
    try:
        selected_qargs = (
            list(range(circuit.num_qubits)) if qargs is None else list(qargs)
        )
        model_repo = _load_ai_pauli_model_repository()
        record, subgraph_perm = _select_ai_pauli_model_record(
            model_repo,
            coupling_map,
            selected_qargs,
        )
        model = record.model
        model_n_qubits = int(model.env_config.get("num_qubits", len(selected_qargs)))
        prepared_input = _prepare_ai_pauli_input(
            circuit,
            subgraph_perm,
            model_n_qubits,
        )
        replay_terms, rotation_angle_signs = _parse_ai_pauli_circuit_rotations(
            prepared_input
        )
        state = model.env.get_state(prepared_input)
        with _capture_process_output() as solver_output:
            raw_actions = model.algorithm.solve(
                state,
                deterministic,
                num_searches,
                num_mcts_searches,
                c,
                max_expand_depth,
            )
        solver_output_fields = _solver_output_fields(solver_output.getvalue())
        if raw_actions is None:
            return {
                "status": "failed",
                "dependency": status,
                "before": before,
                "raw_actions": None,
                "decoded_solution": None,
                "gateset": _normalise_gateset(model.env_config.get("gateset", [])),
                "num_qubits": model_n_qubits,
                "replay_terms": list(replay_terms),
                "rotation_angle_signs": list(rotation_angle_signs),
                **solver_output_fields,
                "error": "AI Pauli solver returned no trajectory",
            }
        raw_action_list = [int(action) for action in raw_actions]
    except Exception as exc:  # noqa: BLE001 - experiments should report failures.
        return {
            "status": "failed",
            "dependency": status,
            "before": before,
            "raw_actions": None,
            "decoded_solution": None,
            "gateset": None,
            "num_qubits": circuit.num_qubits,
            "replay_terms": list(replay_terms) if replay_terms is not None else None,
            "rotation_angle_signs": (
                list(rotation_angle_signs) if rotation_angle_signs is not None else None
            ),
            "solver_output_lines": 0,
            "solver_output_excerpt": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "ok",
        "dependency": status,
        "before": before,
        "raw_actions": raw_action_list,
        "decoded_solution": decode_ai_pauli_solution(raw_action_list),
        "gateset": _normalise_gateset(model.env_config.get("gateset", [])),
        "num_qubits": model_n_qubits,
        "subgraph_perm": list(subgraph_perm),
        "replay_terms": list(replay_terms),
        "rotation_angle_signs": list(rotation_angle_signs),
        **solver_output_fields,
        "error": "",
    }


def analyze_ai_pauli_window_trajectory(
    window: PauliNetworkWindow | PauliNetworkRegion,
    *,
    k: int,
    deterministic: bool = False,
    num_searches: int = 10,
    num_mcts_searches: int = 0,
    c: float = 2**0.5,
    max_expand_depth: int = 1,
) -> dict[str, Any]:
    circuit = build_pauli_network_circuit(
        num_qubits=window.num_qubits,
        signed_paulis=window.signed_paulis,
    )
    trajectory = extract_ai_pauli_trajectory(
        circuit=circuit,
        coupling_map=window.coupling_map,
        deterministic=deterministic,
        num_searches=num_searches,
        num_mcts_searches=num_mcts_searches,
        c=c,
        max_expand_depth=max_expand_depth,
    )
    if trajectory["status"] != "ok":
        return trajectory

    replay_terms = trajectory.get("replay_terms")
    rotation_angle_signs = trajectory.get("rotation_angle_signs")
    if replay_terms is None or rotation_angle_signs is None:
        replay_terms, rotation_angle_signs = _parse_ai_pauli_circuit_rotations(circuit)

    try:
        replay = replay_ai_pauli_solution(
            num_qubits=int(trajectory["num_qubits"]),
            signed_paulis=replay_terms,
            gateset=trajectory["gateset"],
            solution=trajectory["decoded_solution"],
        )
        terminal = find_k_terminal_prefix(replay, k=k)
        emissions_before_prefix = sum(
            emission.prefix_length <= terminal.prefix_length
            for emission in replay.emissions
        )
    except Exception as exc:  # noqa: BLE001 - experiments should report failures.
        return {
            **trajectory,
            "status": "failed",
            "full_trajectory_length": len(trajectory.get("raw_actions") or []),
            "replay_terms": list(replay_terms),
            "rotation_angle_signs": list(rotation_angle_signs),
            "error": f"replay failed: {type(exc).__name__}: {exc}",
        }

    return {
        **trajectory,
        "full_trajectory_length": len(trajectory["raw_actions"]),
        "k_terminal_prefix_length": terminal.prefix_length,
        "pending_terms": list(terminal.pending_terms),
        "pending_weights": list(terminal.pending_weights),
        "pending_rotation_indices": list(terminal.pending_indices),
        "prefix_emissions": [
            {
                "prefix_length": emission.prefix_length,
                "op": emission.op,
                "qubit": emission.qubit,
                "rotation_index": emission.rotation_index,
                "phase_mult": emission.phase_mult,
                "emitted_pauli": emission.emitted_pauli,
            }
            for emission in replay.emissions
            if emission.prefix_length <= terminal.prefix_length
        ],
        "rotation_angle_signs": list(rotation_angle_signs),
        "emissions_before_prefix": emissions_before_prefix,
        "total_emissions": len(replay.emissions),
    }


def build_ai_trajectory_prefix_ops(
    *,
    start_id: int,
    trajectory: dict[str, Any],
    k: int,
    original_qubits: Sequence[int] | None = None,
    source_ids: Sequence[str] | None = None,
) -> AITrajectoryPrefixOps:
    if trajectory.get("status") != "ok":
        raise ValueError("AI trajectory result is not ok")
    if type(start_id) is not int or start_id < 0:
        raise ValueError("start_id must be a non-negative integer")
    if type(k) is not int or k < 1:
        raise ValueError("k must be an integer >= 1")

    num_qubits = int(trajectory["num_qubits"])
    original_qubits = (
        tuple(range(num_qubits))
        if original_qubits is None
        else tuple(int(qubit) for qubit in original_qubits)
    )
    if len(original_qubits) != num_qubits:
        raise ValueError("original_qubits must match the trajectory width")

    replay_terms = tuple(str(term) for term in trajectory["replay_terms"])
    rotation_angle_signs = tuple(
        int(sign) for sign in trajectory["rotation_angle_signs"]
    )
    source_ids = (
        tuple(f"ai_rotation_{index}" for index in range(len(replay_terms)))
        if source_ids is None
        else tuple(str(source_id) for source_id in source_ids)
    )
    if len(rotation_angle_signs) != len(replay_terms):
        raise ValueError("rotation_angle_signs must match replay_terms")
    if len(source_ids) != len(replay_terms):
        raise ValueError("source_ids must match replay_terms")

    replay = replay_ai_pauli_solution(
        num_qubits=num_qubits,
        signed_paulis=replay_terms,
        gateset=trajectory["gateset"],
        solution=trajectory["decoded_solution"],
    )
    prefix_length = int(trajectory["k_terminal_prefix_length"])
    terminal = next(
        (
            snapshot
            for snapshot in replay.snapshots
            if snapshot.prefix_length == prefix_length
        ),
        None,
    )
    if terminal is None:
        raise ValueError(f"trajectory has no prefix length {prefix_length}")
    if any(weight > k for weight in terminal.pending_weights):
        raise ValueError("trajectory prefix does not satisfy k cap")

    emission_by_prefix = {
        emission.prefix_length: emission
        for emission in replay.emissions
        if emission.prefix_length <= prefix_length
    }
    ops: list[SemiPBCOp] = []
    source_indices_by_output_id: dict[int, int] = {}
    prefix_cliffords: list[SemiPBCOp] = []
    next_id = start_id

    decoded_solution = _normalise_ai_pauli_solution(trajectory["decoded_solution"])
    gateset = _normalise_gateset(trajectory["gateset"])
    shared_source = "+".join(source_ids)
    for step_index, (op, arg1, _arg2, _arg3) in enumerate(
        decoded_solution[:prefix_length],
        start=1,
    ):
        if op == "gate":
            clifford = _semi_pbc_clifford_for_ai_gate(
                next_id,
                gateset[arg1],
                original_qubits=original_qubits,
                source_id=shared_source,
            )
            ops.append(clifford)
            prefix_cliffords.append(clifford)
            source_indices_by_output_id[next_id] = 0
            next_id += 1
            continue
        if op in {"rx", "ry", "rz"}:
            emission = emission_by_prefix[step_index]
            source_index = emission.rotation_index
            term = _semi_pbc_term_from_signed_pauli(
                emission.emitted_pauli,
                original_qubits=original_qubits,
                sign_multiplier=rotation_angle_signs[source_index],
                source_id=source_ids[source_index],
            )
            if term.weight > k:
                raise ValueError("emitted Pauli term exceeds k cap")
            ops.append(
                SemiPBCOp.pauli_rotation(
                    next_id,
                    term,
                    source_id=source_ids[source_index],
                )
            )
            source_indices_by_output_id[next_id] = source_index
            next_id += 1
            continue
        raise ValueError(f"unsupported trajectory op {op!r}")

    for source_index, pending_term in zip(
        terminal.pending_indices,
        terminal.pending_terms,
        strict=True,
    ):
        term = _semi_pbc_term_from_signed_pauli(
            pending_term,
            original_qubits=original_qubits,
            sign_multiplier=rotation_angle_signs[source_index],
            source_id=source_ids[source_index],
        )
        if term.weight > k:
            raise ValueError("pending Pauli term exceeds k cap")
        ops.append(
            SemiPBCOp.pauli_rotation(
                next_id,
                term,
                source_id=source_ids[source_index],
            )
        )
        source_indices_by_output_id[next_id] = source_index
        next_id += 1

    for clifford in reversed(prefix_cliffords):
        inverse = _inverse_semi_pbc_clifford(next_id, clifford)
        ops.append(inverse)
        source_indices_by_output_id[next_id] = source_indices_by_output_id[clifford.id]
        next_id += 1

    return AITrajectoryPrefixOps(ops, source_indices_by_output_id)


def extract_rotation_regions(
    program: PBCProgram,
    *,
    source_path: str | Path,
    max_regions: int,
    min_terms: int = 4,
    max_terms: int | None = None,
    topology: str = "line",
) -> list[PauliNetworkRegion]:
    if max_regions < 1:
        return []
    if min_terms < 1:
        raise ValueError("min_terms must be >= 1")
    if max_terms is not None and max_terms < min_terms:
        raise ValueError("max_terms must be >= min_terms")
    if topology not in {key[0] for key in _SUPPORTED_TOPOLOGIES}:
        raise ValueError(f"unsupported topology {topology!r}")

    regions: list[PauliNetworkRegion] = []
    current = []
    for op in program.ops:
        if op.op == "t_pauli":
            current.append(op)
            continue
        _append_rotation_region(
            regions,
            current,
            program=program,
            source_path=source_path,
            min_terms=min_terms,
            max_terms=max_terms,
            topology=topology,
        )
        current = []
    _append_rotation_region(
        regions,
        current,
        program=program,
        source_path=source_path,
        min_terms=min_terms,
        max_terms=max_terms,
        topology=topology,
    )
    return sorted(
        regions,
        key=lambda region: (
            -region.total_pauli_weight,
            -region.multi_qubit_terms,
            region.start_op_id,
        ),
    )[:max_regions]


def build_rotation_region_circuit(region: PauliNetworkRegion) -> QuantumCircuit:
    return build_pauli_network_circuit(
        num_qubits=region.num_qubits,
        signed_paulis=region.signed_paulis,
    )


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


def _normalise_ai_pauli_solution(
    solution: Sequence[int] | Sequence[tuple[str, int, int, int]],
) -> list[tuple[str, int, int, int]]:
    if not solution:
        return []
    first = solution[0]
    if isinstance(first, int):
        return decode_ai_pauli_solution(solution)  # type: ignore[arg-type]
    return [(op, int(arg1), int(arg2), int(arg3)) for op, arg1, arg2, arg3 in solution]  # type: ignore[misc]


def _canonical_signed_pauli(signed_pauli: str, num_qubits: int) -> str:
    sign, pauli = _parse_signed_pauli(signed_pauli, num_qubits)
    return ("+" if sign > 0 else "-") + pauli


def _clifford_for_ai_pauli_gate(
    num_qubits: int,
    gate: tuple[str, Sequence[int]],
) -> Clifford:
    gate_name, raw_args = gate
    args = tuple(int(arg) for arg in raw_args)
    if any(arg < 0 or arg >= num_qubits for arg in args):
        raise ValueError(f"gate {gate_name} uses qubits outside width {num_qubits}")
    circuit = QuantumCircuit(num_qubits)
    method_name = gate_name.lower()
    if method_name == "cx":
        args = args[::-1]
    try:
        getattr(circuit, method_name)(*args)
    except AttributeError as exc:
        raise ValueError(f"unsupported gate {gate_name!r}") from exc
    return Clifford(circuit)


def _evolve_signed_pauli(signed_pauli: str, clifford: Clifford) -> str:
    sign = signed_pauli[0]
    pauli = signed_pauli[1:]
    qiskit_label = pauli[::-1]
    if sign == "-":
        qiskit_label = "-" + qiskit_label
    evolved = Pauli(qiskit_label).evolve(clifford, frame="s")
    return _qiskit_pauli_to_signed_pauli(evolved)


def _qiskit_pauli_to_signed_pauli(pauli: Pauli) -> str:
    label = pauli.to_label()
    sign = "+"
    if label.startswith("-"):
        sign = "-"
        label = label[1:]
    elif label.startswith("+"):
        label = label[1:]
    if label.startswith(("i", "-i")):
        raise ValueError(f"non-real Pauli phase after Clifford evolution: {pauli}")
    return sign + label[::-1]


def _single_rotation_for_pauli(signed_pauli: str) -> tuple[str, int, int]:
    sign = signed_pauli[0]
    pauli = signed_pauli[1:]
    active = [(index, axis) for index, axis in enumerate(pauli) if axis != "I"]
    if len(active) != 1:
        raise ValueError(f"pending Pauli is not weight 1: {signed_pauli}")
    qubit, axis = active[0]
    return f"r{axis.lower()}", qubit, (1 if sign == "+" else -1)


def _pauli_replay_snapshot(
    prefix_length: int,
    pending: Sequence[str],
    emitted: set[int],
) -> PauliReplaySnapshot:
    pending_pairs = tuple(
        (index, pauli) for index, pauli in enumerate(pending) if index not in emitted
    )
    return PauliReplaySnapshot(
        prefix_length=prefix_length,
        pending_terms=tuple(pauli for _index, pauli in pending_pairs),
        pending_weights=tuple(_pauli_weight(pauli) for _index, pauli in pending_pairs),
        pending_indices=tuple(index for index, _pauli in pending_pairs),
    )


def _pauli_weight(signed_pauli: str) -> int:
    return sum(label != "I" for label in signed_pauli[1:])


def _compress_source_pauli(signed_pauli: str, reindex: dict[int, int]) -> str:
    compressed = ["I"] * len(reindex)
    for original_index, compressed_index in reindex.items():
        compressed[compressed_index] = signed_pauli[original_index + 1]
    return signed_pauli[0] + "".join(compressed)


def _append_rotation_region(
    regions: list[PauliNetworkRegion],
    ops: list[Any],
    *,
    program: PBCProgram,
    source_path: str | Path,
    min_terms: int,
    max_terms: int | None,
    topology: str,
) -> None:
    if len(ops) < min_terms:
        return
    selected_ops = ops[:max_terms] if max_terms is not None else ops
    coupling_map = _region_coupling_map(program.data_qubits, topology)
    if coupling_map is None:
        return
    regions.append(
        PauliNetworkRegion(
            source_path=str(source_path),
            start_op_id=selected_ops[0].id,
            stop_op_id=selected_ops[-1].id,
            source_ids=tuple(op.source_id for op in selected_ops),
            num_qubits=program.data_qubits,
            signed_paulis=tuple(
                op.term.to_full_width(program.data_qubits) for op in selected_ops
            ),
            total_pauli_weight=sum(op.term.weight for op in selected_ops),
            multi_qubit_terms=sum(op.term.weight > 1 for op in selected_ops),
            topology=topology,
            coupling_map=tuple(coupling_map),
        )
    )


def _region_coupling_map(
    num_qubits: int,
    topology: str,
) -> list[tuple[int, int]] | None:
    if topology == "line":
        return [(qubit, qubit + 1) for qubit in range(num_qubits - 1)]
    return _SUPPORTED_TOPOLOGIES.get((topology, num_qubits))


def _load_ai_pauli_model_repository():
    _install_qiskit_compatibility_shims()
    from qiskit_ibm_transpiler.model_bootstrap import ensure_models_loaded

    return ensure_models_loaded("pauli")


def _select_ai_pauli_model_record(
    model_repo,
    coupling_map: Sequence[tuple[int, int]],
    qargs: Sequence[int],
):
    from qiskit_ibm_transpiler.wrappers.ai_local_synthesis import (
        get_coupling_map_graph,
        get_formatted_coupling_map,
        get_mapping_perm,
    )

    formatted = get_formatted_coupling_map(list(coupling_map))
    graph = get_coupling_map_graph(coupling_map=formatted)
    subgraph_perm, cmap_hash = get_mapping_perm(graph, list(qargs), model_repo)
    return model_repo.get(cmap_hash), subgraph_perm


def _prepare_ai_pauli_input(
    circuit: QuantumCircuit,
    subgraph_perm: Sequence[int],
    target_qubits: int,
) -> QuantumCircuit:
    input_circuit = circuit.decompose(
        ["swap", "rxx", "ryy", "rzz", "rzx", "rzy", "ryx"]
    )
    input_circuit_perm = QuantumCircuit(input_circuit.num_qubits).compose(
        input_circuit,
        qubits=np.argsort(subgraph_perm),
    )
    if input_circuit_perm.num_qubits > target_qubits:
        raise ValueError(
            f"model expects {target_qubits} qubits but circuit uses "
            f"{input_circuit_perm.num_qubits}"
        )
    if input_circuit_perm.num_qubits == target_qubits:
        return input_circuit_perm

    embedded = QuantumCircuit(target_qubits)
    embedded.compose(
        input_circuit_perm,
        qubits=range(input_circuit_perm.num_qubits),
        inplace=True,
    )
    return embedded


def _parse_ai_pauli_circuit_rotations(
    circuit: QuantumCircuit,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    num_qubits = circuit.num_qubits
    clifford = Clifford(QuantumCircuit(num_qubits))
    rotations: list[str] = []
    angle_signs: list[int] = []

    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if gate_name in {"rx", "ry", "rz"}:
            pauli_chars = ["I"] * num_qubits
            pauli_chars[num_qubits - 1 - qubits[0]] = gate_name[1].upper()
            evolved = Pauli("".join(pauli_chars)).evolve(clifford)
            rotations.append(_qiskit_pauli_to_signed_pauli(evolved.adjoint()))
            angle_signs.append(_rotation_angle_sign(instruction.operation.params[0]))
            continue
        try:
            clifford = clifford.compose(instruction.operation, qubits)
        except QiskitError as exc:
            raise TypeError(
                f"Gate {gate_name} on qubits {qubits} not supported."
            ) from exc

    return tuple(rotations), tuple(angle_signs)


def _rotation_angle_sign(angle: Any) -> int:
    return 1 if float(angle) >= 0 else -1


def _solver_output_fields(output: str) -> dict[str, Any]:
    return {
        "solver_output_lines": len(output.splitlines()),
        "solver_output_excerpt": output[:2000],
    }


def _normalise_gateset(
    gateset: Sequence[Sequence[Any]],
) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (str(gate_name).lower(), tuple(int(qubit) for qubit in qubits))
        for gate_name, qubits in gateset
    ]


def _semi_pbc_clifford_for_ai_gate(
    id: int,
    gate: tuple[str, Sequence[int]],
    *,
    original_qubits: Sequence[int],
    source_id: str,
) -> SemiPBCOp:
    gate_name, raw_args = gate
    method_name = gate_name.lower()
    args = tuple(int(arg) for arg in raw_args)
    if method_name == "cx":
        args = args[::-1]
    if method_name not in {"h", "s", "sdg", "cx"}:
        raise ValueError(f"unsupported semi-PBC Clifford gate {gate_name!r}")
    try:
        qubits = tuple(f"q{original_qubits[arg]}" for arg in args)
    except IndexError as exc:
        raise ValueError(f"gate {gate_name} uses qubits outside local window") from exc
    return SemiPBCOp.clifford(id, method_name, qubits, source_id=source_id)


def _inverse_semi_pbc_clifford(id: int, op: SemiPBCOp) -> SemiPBCOp:
    inverse_op = {"h": "h", "s": "sdg", "sdg": "s", "cx": "cx"}[op.op]
    return SemiPBCOp.clifford(id, inverse_op, op.qubits, source_id=op.source_id)


def _semi_pbc_term_from_signed_pauli(
    signed_pauli: str,
    *,
    original_qubits: Sequence[int],
    sign_multiplier: int,
    source_id: str,
) -> PauliTerm:
    if sign_multiplier not in {-1, 1}:
        raise ValueError("sign_multiplier must be -1 or 1")
    sign, pauli = _parse_signed_pauli(signed_pauli, len(original_qubits))
    return PauliTerm.from_pairs(
        (
            (f"q{original_qubits[index]}", label)
            for index, label in enumerate(pauli)
            if label != "I"
        ),
        sign=sign * sign_multiplier,
        source_id=source_id,
    )


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


@contextmanager
def _capture_process_output():
    captured = io.StringIO()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with TemporaryFile(mode="w+b") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield captured
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            sink.seek(0)
            captured.write(sink.read().decode("utf-8", errors="replace"))


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
