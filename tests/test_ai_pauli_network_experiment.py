from __future__ import annotations

import builtins
import json
from pathlib import Path

from ftcircuitbench.semi_pbc.ai_pauli_network import (
    ai_pauli_network_dependency_status,
    build_pauli_network_circuit,
    circuit_metrics,
    extract_supported_rotation_windows,
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
