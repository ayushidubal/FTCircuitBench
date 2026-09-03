from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

import pytest

from ftcircuitbench.semi_pbc.ai_pauli_network import (
    PauliNetworkWindow,
    _capture_process_output,
    ai_pauli_network_dependency_status,
    analyze_ai_pauli_window_trajectory,
    build_pauli_network_circuit,
    build_rotation_region_circuit,
    circuit_metrics,
    circuits_equivalent,
    decode_ai_pauli_solution,
    extract_ai_pauli_trajectory,
    extract_rotation_regions,
    extract_supported_rotation_windows,
    find_k_terminal_prefix,
    replay_ai_pauli_solution,
    run_ai_pauli_network_synthesis,
)
from ftcircuitbench.semi_pbc.pbc_input import parse_pbc_text


def test_dependency_status_has_stable_shape() -> None:
    status = ai_pauli_network_dependency_status()

    assert set(status) == {
        "available",
        "qiskit_version",
        "qiskit_ibm_transpiler_version",
        "reason",
    }
    assert isinstance(status["available"], bool)
    assert isinstance(status["qiskit_version"], str)


def test_dependency_status_reports_import_broken_package(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qiskit_ibm_transpiler":
            raise ModuleNotFoundError("broken dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network.importlib.util.find_spec",
        lambda name: object() if name == "qiskit_ibm_transpiler" else None,
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    status = ai_pauli_network_dependency_status()

    assert status["available"] is False
    assert "broken dependency" in status["reason"]


def test_build_pauli_network_circuit_uses_one_rz_per_term() -> None:
    circuit = build_pauli_network_circuit(
        num_qubits=3,
        signed_paulis=("+ZZI", "-IXY"),
    )

    counts = circuit.count_ops()
    assert counts["rz"] == 2
    assert counts["cx"] == 4
    assert circuit.num_qubits == 3


def test_circuit_metrics_are_json_serializable() -> None:
    circuit = build_pauli_network_circuit(
        num_qubits=2,
        signed_paulis=("+ZZ",),
    )

    metrics = circuit_metrics(circuit)

    assert metrics["num_qubits"] == 2
    assert metrics["ops"] == 3
    assert metrics["two_qubit_ops"] == 2
    json.dumps(metrics)


def test_runner_skips_cleanly_when_dependency_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network.ai_pauli_network_dependency_status",
        lambda: {
            "available": False,
            "qiskit_version": "2.0.2",
            "qiskit_ibm_transpiler_version": "",
            "reason": "missing qiskit_ibm_transpiler",
        },
    )

    result = run_ai_pauli_network_synthesis(
        num_qubits=2,
        signed_paulis=("+ZZ",),
    )

    assert result["status"] == "skipped"
    assert result["before"]["two_qubit_ops"] == 2
    assert result["after"] is None


def test_extract_supported_rotation_windows_reindexes_active_qubits() -> None:
    program = parse_pbc_text(
        """qreg q[8];
t_pauli +ZIZIZIII;
m_pauli +ZZIIIIII;
t_pauli -IIZIZZII;
t_pauli +IIZIZZII;
t_pauli +ZIIIIZII;"""
    )

    windows = extract_supported_rotation_windows(
        program,
        source_path=Path("toy_pbc.txt"),
        max_windows=1,
        window_terms=4,
    )

    assert len(windows) == 1
    assert windows[0].source_path == "toy_pbc.txt"
    assert windows[0].source_ids == ("line2", "line4", "line5", "line6")
    assert windows[0].num_qubits == 4
    assert windows[0].original_qubits == (0, 2, 4, 5)
    assert windows[0].signed_paulis == (
        "+ZZZI",
        "-IZZZ",
        "+IZZZ",
        "+ZIIZ",
    )


def test_benchmark_pbc_files_writes_jsonl_and_summary(tmp_path, monkeypatch) -> None:
    from benchmark_ai_pauli_network_synthesis import benchmark_pbc_files

    pbc = tmp_path / "toy_pbc_post_opt.txt"
    pbc.write_text(
        """qreg q[4];
t_pauli +ZZII;
t_pauli +IZZI;
t_pauli -IIZZ;
t_pauli +ZIIZ;""",
        encoding="utf-8",
    )

    def fake_run(**kwargs):
        assert kwargs["num_qubits"] == 4
        assert kwargs["coupling_map"] == [(0, 1), (1, 2), (2, 3)]
        assert kwargs["capture_pass_output"] is True
        return {
            "status": "ok",
            "dependency": {"available": True},
            "before": {"ops": 12, "depth": 12, "two_qubit_ops": 8},
            "after": {"ops": 11, "depth": 10, "two_qubit_ops": 7},
            "seconds": 0.5,
            "error": "",
            "changed": True,
            "optimized_qasm": "OPENQASM 2.0;",
        }

    monkeypatch.setattr(
        "benchmark_ai_pauli_network_synthesis.run_ai_pauli_network_synthesis",
        fake_run,
    )

    summary = benchmark_pbc_files(
        paths=[pbc],
        out_dir=tmp_path / "out",
        max_windows_per_file=1,
        max_files=None,
        window_terms=4,
        max_threads=1,
        include_qasm=False,
    )

    assert summary["windows_run"] == 1
    assert summary["changed"] == 1
    assert summary["metric_improved"] == 1
    assert summary["metric_worse"] == 0
    assert summary["delta_depth"] == -2
    result_lines = (tmp_path / "out" / "results.jsonl").read_text().splitlines()
    assert len(result_lines) == 1
    assert "optimized_qasm" not in json.loads(result_lines[0])["result"]
    assert (tmp_path / "out" / "summary.json").exists()


def test_extract_rotation_regions_splits_on_measurements() -> None:
    program = parse_pbc_text(
        """qreg q[4];
t_pauli +ZZII;
t_pauli +IZZI;
m_pauli +ZIII;
t_pauli -IIZZ;
t_pauli +ZIIZ;"""
    )

    regions = extract_rotation_regions(
        program,
        source_path=Path("toy_pbc.txt"),
        max_regions=10,
        min_terms=2,
    )

    assert len(regions) == 2
    assert regions[0].source_ids == ("line2", "line3")
    assert regions[1].source_ids == ("line5", "line6")
    assert regions[0].signed_paulis == ("+ZZII", "+IZZI")


def test_build_rotation_region_circuit_uses_full_program_width() -> None:
    program = parse_pbc_text(
        """qreg q[4];
t_pauli +ZZII;
t_pauli +IIIZ;"""
    )
    region = extract_rotation_regions(
        program,
        source_path=Path("toy_pbc.txt"),
        max_regions=1,
        min_terms=2,
    )[0]

    circuit = build_rotation_region_circuit(region)

    assert circuit.num_qubits == 4
    assert circuit.count_ops()["rz"] == 2


def test_benchmark_pbc_files_can_use_collect_pauli_networks_selection(
    tmp_path,
    monkeypatch,
) -> None:
    from benchmark_ai_pauli_network_synthesis import benchmark_pbc_files

    pbc = tmp_path / "toy_pbc_post_opt.txt"
    pbc.write_text(
        """qreg q[4];
t_pauli +ZZII;
t_pauli +IZZI;
t_pauli -IIZZ;
t_pauli +ZIIZ;""",
        encoding="utf-8",
    )

    def fake_run(**kwargs):
        assert kwargs["circuit"].num_qubits == 4
        assert kwargs["coupling_map"] == [(0, 1), (1, 2), (2, 3)]
        assert kwargs["capture_pass_output"] is True
        return {
            "status": "ok",
            "dependency": {"available": True},
            "before": {"ops": 12, "depth": 12, "two_qubit_ops": 8},
            "after": {"ops": 10, "depth": 9, "two_qubit_ops": 6},
            "seconds": 0.25,
            "error": "",
            "changed": True,
            "optimized_qasm": "OPENQASM 2.0;",
        }

    monkeypatch.setattr(
        "benchmark_ai_pauli_network_synthesis.run_ai_pauli_network_synthesis_on_circuit",
        fake_run,
    )

    summary = benchmark_pbc_files(
        paths=[pbc],
        out_dir=tmp_path / "out",
        max_windows_per_file=1,
        max_files=None,
        window_terms=4,
        max_threads=1,
        include_qasm=False,
        selection="collect-pauli-networks",
        max_regions_per_file=1,
        min_region_terms=4,
    )

    assert summary["regions_run"] == 1
    assert summary["windows_run"] == 0
    assert summary["metric_improved"] == 1
    record = json.loads((tmp_path / "out" / "results.jsonl").read_text())
    assert record["selection"] == "collect-pauli-networks"
    assert record["region"]["source_ids"] == ["line2", "line3", "line4", "line5"]


def test_capture_process_output_catches_fd_writes() -> None:
    with _capture_process_output() as captured:
        os.write(2, b"fd-level diagnostic\n")

    assert "fd-level diagnostic" in captured.getvalue()


def test_circuits_equivalent_accepts_same_unitary() -> None:
    original = build_pauli_network_circuit(
        num_qubits=2,
        signed_paulis=("+ZZ",),
    )
    same = build_pauli_network_circuit(
        num_qubits=2,
        signed_paulis=("+ZZ",),
    )
    different = build_pauli_network_circuit(
        num_qubits=2,
        signed_paulis=("-ZZ",),
    )

    assert circuits_equivalent(original, same)
    assert not circuits_equivalent(original, different)


def test_benchmark_emits_equivalent_improved_ranked_window_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    from qiskit import qasm2

    from benchmark_ai_pauli_network_synthesis import benchmark_pbc_files

    pbc = tmp_path / "toy_pbc_post_opt.txt"
    pbc.write_text(
        """qreg q[4];
t_pauli +ZZII;
t_pauli +IZZI;
t_pauli -IIZZ;
t_pauli +ZIIZ;""",
        encoding="utf-8",
    )

    def fake_run(**kwargs):
        original = build_pauli_network_circuit(
            num_qubits=kwargs["num_qubits"],
            signed_paulis=kwargs["signed_paulis"],
        )
        return {
            "status": "ok",
            "dependency": {"available": True},
            "before": {"ops": 12, "depth": 12, "two_qubit_ops": 8},
            "after": {"ops": 10, "depth": 9, "two_qubit_ops": 6},
            "seconds": 0.1,
            "error": "",
            "changed": True,
            "optimized_qasm": qasm2.dumps(original),
        }

    monkeypatch.setattr(
        "benchmark_ai_pauli_network_synthesis.run_ai_pauli_network_synthesis",
        fake_run,
    )

    summary = benchmark_pbc_files(
        paths=[pbc],
        out_dir=tmp_path / "out",
        max_windows_per_file=1,
        max_files=None,
        window_terms=4,
        max_threads=1,
        include_qasm=False,
        emit_candidates=True,
    )

    assert summary["candidates_emitted"] == 1
    assert summary["candidate_equivalent"] == 1
    candidate = json.loads((tmp_path / "out" / "candidate_replacements.jsonl").read_text())
    assert candidate["equivalent"] is True
    assert candidate["selection"] == "ranked-window"
    assert candidate["window"]["source_ids"] == ["line2", "line3", "line4", "line5"]
    assert candidate["delta"] == {
        "depth": -3,
        "ops": -2,
        "two_qubit_ops": -2,
    }


def test_benchmark_pbc_files_can_analyze_ranked_window_trajectories(
    tmp_path,
    monkeypatch,
) -> None:
    from benchmark_ai_pauli_network_synthesis import benchmark_pbc_files

    pbc = tmp_path / "toy_pbc_post_opt.txt"
    pbc.write_text(
        """qreg q[4];
t_pauli +ZZII;
t_pauli +IZZI;
t_pauli -IIZZ;
t_pauli +ZIIZ;""",
        encoding="utf-8",
    )

    def fake_analyze(window, **kwargs):
        assert window.num_qubits == 4
        assert kwargs["k"] == 2
        return {
            "status": "ok",
            "full_trajectory_length": 12,
            "k_terminal_prefix_length": 5,
            "pending_terms": ["+ZZII", "+IIZZ"],
            "pending_weights": [2, 2],
            "emissions_before_prefix": 1,
            "total_emissions": 4,
            "error": "",
        }

    monkeypatch.setattr(
        "benchmark_ai_pauli_network_synthesis.analyze_ai_pauli_window_trajectory",
        fake_analyze,
    )

    summary = benchmark_pbc_files(
        paths=[pbc],
        out_dir=tmp_path / "out",
        max_windows_per_file=1,
        max_files=None,
        window_terms=4,
        max_threads=1,
        include_qasm=False,
        trajectory_k="floor-half",
    )

    assert summary["windows_run"] == 1
    assert summary["ok"] == 1
    assert summary["trajectory_prefix_length_total"] == 5
    record = json.loads((tmp_path / "out" / "results.jsonl").read_text())
    assert record["result"]["k_terminal_prefix_length"] == 5
    assert record["window"]["source_ids"] == ["line2", "line3", "line4", "line5"]


def test_decode_ai_pauli_solution_decodes_gate_and_rotation_markers() -> None:
    rotation_marker = 0x80000000 | (2 << 21) | (3 << 11) | (5 << 1) | 1

    decoded = decode_ai_pauli_solution([7, rotation_marker])

    assert decoded == [("gate", 7, 0, 0), ("rz", 3, 5, 1)]


def test_decode_ai_pauli_solution_matches_qiskit_gym_decoder_when_available() -> None:
    qiskit_gym_synthesis = pytest.importorskip("qiskit_gym.envs.synthesis")
    encoded = [
        7,
        0x80000000 | (0 << 21) | (2 << 11) | (4 << 1),
        0x80000000 | (1 << 21) | (1 << 11) | (3 << 1) | 1,
        0x80000000 | (2 << 21) | (3 << 11) | (5 << 1) | 1,
    ]

    assert decode_ai_pauli_solution(encoded) == (
        qiskit_gym_synthesis.decode_pauli_solution(encoded)
    )


def test_replay_ai_pauli_solution_uses_reversed_cx_and_s_frame() -> None:
    cx_replay = replay_ai_pauli_solution(
        num_qubits=2,
        signed_paulis=("+ZZ",),
        gateset=(("cx", (0, 1)),),
        solution=[("gate", 0, 0, 0), ("rz", 0, 0, 1)],
    )

    assert cx_replay.snapshots[1].pending_terms == ("+ZI",)
    assert cx_replay.emissions[0].emitted_pauli == "+ZI"

    s_replay = replay_ai_pauli_solution(
        num_qubits=1,
        signed_paulis=("+X",),
        gateset=(("s", (0,)),),
        solution=[("gate", 0, 0, 0), ("ry", 0, 0, 1)],
    )

    assert s_replay.snapshots[1].pending_terms == ("+Y",)
    assert s_replay.emissions[0].phase_mult == 1


def test_find_k_terminal_prefix_returns_first_all_pending_weights_under_cap() -> None:
    replay = replay_ai_pauli_solution(
        num_qubits=2,
        signed_paulis=("+ZZ", "+XI"),
        gateset=(("cx", (0, 1)),),
        solution=[
            ("gate", 0, 0, 0),
            ("rz", 0, 0, 1),
            ("rx", 0, 1, 1),
        ],
    )

    terminal = find_k_terminal_prefix(replay, k=1)

    assert terminal.prefix_length == 1
    assert terminal.pending_terms == ("+ZI", "+XI")
    assert terminal.pending_weights == (1, 1)


def test_extract_ai_pauli_trajectory_skips_when_dependency_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network.ai_pauli_network_dependency_status",
        lambda: {
            "available": False,
            "qiskit_version": "2.0.2",
            "qiskit_ibm_transpiler_version": "",
            "reason": "missing qiskit_ibm_transpiler",
        },
    )

    result = extract_ai_pauli_trajectory(
        circuit=build_pauli_network_circuit(num_qubits=1, signed_paulis=("+Z",)),
        coupling_map=[],
    )

    assert result["status"] == "skipped"
    assert result["raw_actions"] is None
    assert result["error"] == "missing qiskit_ibm_transpiler"


def test_extract_ai_pauli_trajectory_uses_model_algorithm_actions(monkeypatch) -> None:
    calls = {}

    class FakeEnv:
        def get_state(self, circuit):
            calls["prepared_width"] = circuit.num_qubits
            return [9, 8, 7]

    class FakeAlgorithm:
        def solve(
            self,
            state,
            deterministic,
            num_searches,
            num_mcts_searches,
            c,
            max_expand_depth,
        ):
            os.write(2, b"solver diagnostic\n")
            calls["solve"] = (
                state,
                deterministic,
                num_searches,
                num_mcts_searches,
                c,
                max_expand_depth,
            )
            return [0, 0x80000000 | (2 << 21) | (0 << 11) | (0 << 1) | 1]

    class FakeModel:
        def __init__(self):
            self.env = FakeEnv()
            self.algorithm = FakeAlgorithm()
            self.env_config = {
                "num_qubits": 2,
                "gateset": [("cx", (0, 1))],
            }

    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network.ai_pauli_network_dependency_status",
        lambda: {
            "available": True,
            "qiskit_version": "2.5.2",
            "qiskit_ibm_transpiler_version": "0.18.0",
            "reason": "",
        },
    )
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network._load_ai_pauli_model_repository",
        lambda: object(),
    )
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network._select_ai_pauli_model_record",
        lambda repo, coupling_map, qargs: (
            type("Record", (), {"model": FakeModel(), "coupling_map": [(0, 1)]})(),
            [0, 1],
        ),
    )
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network._prepare_ai_pauli_input",
        lambda circuit, subgraph_perm, target_qubits: circuit,
    )

    result = extract_ai_pauli_trajectory(
        circuit=build_pauli_network_circuit(num_qubits=2, signed_paulis=("+ZZ",)),
        coupling_map=[(0, 1)],
        deterministic=True,
        num_searches=3,
        num_mcts_searches=2,
        c=1.25,
        max_expand_depth=4,
    )

    assert result["status"] == "ok"
    assert result["raw_actions"] == [
        0,
        0x80000000 | (2 << 21) | (0 << 11) | (0 << 1) | 1,
    ]
    assert result["decoded_solution"] == [("gate", 0, 0, 0), ("rz", 0, 0, 1)]
    assert result["gateset"] == [("cx", (0, 1))]
    assert result["solver_output_lines"] == 1
    assert "solver diagnostic" in result["solver_output_excerpt"]
    assert calls["solve"] == ([9, 8, 7], True, 3, 2, 1.25, 4)


def test_analyze_ai_pauli_window_trajectory_reports_k_terminal_metrics(
    monkeypatch,
) -> None:
    window = PauliNetworkWindow(
        source_path="toy_pbc.txt",
        start_op_id=0,
        stop_op_id=1,
        source_ids=("line2", "line3"),
        original_qubits=(0, 1),
        num_qubits=2,
        signed_paulis=("+ZZ", "+XI"),
        total_pauli_weight=3,
        multi_qubit_terms=1,
        topology="line",
        coupling_map=((0, 1),),
    )
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network.extract_ai_pauli_trajectory",
        lambda **kwargs: {
            "status": "ok",
            "raw_actions": [0, 0x80400001, 0x80000003],
            "decoded_solution": [
                ("gate", 0, 0, 0),
                ("rz", 0, 0, 1),
                ("rx", 0, 1, 1),
            ],
            "gateset": [("cx", (0, 1))],
            "num_qubits": 2,
            "error": "",
        },
    )

    result = analyze_ai_pauli_window_trajectory(window, k=1)

    assert result["status"] == "ok"
    assert result["full_trajectory_length"] == 3
    assert result["k_terminal_prefix_length"] == 1
    assert result["pending_terms"] == ["+ZI", "+XI"]
    assert result["pending_weights"] == [1, 1]
    assert result["emissions_before_prefix"] == 0


def test_analyze_ai_pauli_window_trajectory_uses_parsed_rotation_signs(
    monkeypatch,
) -> None:
    window = PauliNetworkWindow(
        source_path="toy_pbc.txt",
        start_op_id=0,
        stop_op_id=0,
        source_ids=("line2",),
        original_qubits=(0, 1),
        num_qubits=2,
        signed_paulis=("-ZZ",),
        total_pauli_weight=2,
        multi_qubit_terms=1,
        topology="line",
        coupling_map=((0, 1),),
    )
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.ai_pauli_network.extract_ai_pauli_trajectory",
        lambda **kwargs: {
            "status": "ok",
            "raw_actions": [0, 0x80400001],
            "decoded_solution": [("gate", 0, 0, 0), ("rz", 0, 0, 1)],
            "gateset": [("cx", (0, 1))],
            "num_qubits": 2,
            "error": "",
        },
    )

    result = analyze_ai_pauli_window_trajectory(window, k=1)

    assert result["status"] == "ok"
    assert result["pending_terms"] == ["+ZI"]
    assert result["rotation_angle_signs"] == [-1]
