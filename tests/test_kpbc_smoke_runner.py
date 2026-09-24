import json
import subprocess
import sys


def test_kpbc_smoke_runner_creates_artifacts(tmp_path):
    qasm = tmp_path / "toy.qasm"
    qasm.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nt q[0];\n',
        encoding="utf-8",
    )
    router = tmp_path / "router.py"
    router.write_text(
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "json.dump([{'id': 0, 'op': 't_pauli', 'logical_qubits': [0], "
        "'routed_l': 1, 'routed_area': 1}], open(out, 'w'))\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_kpbc_smoke.py",
            "--qasm",
            str(qasm),
            "--out-dir",
            str(out_dir),
            "--k",
            "1",
            "--compiler-mode",
            "segmented-litinski",
            "--router-bin",
            sys.executable,
            "--router-arg",
            str(router),
            "--d",
            "5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (out_dir / "candidate.kpbc.jsonl").exists()
    assert (out_dir / "routed.json").exists()
    assert (out_dir / "trace.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "run.log").exists()
