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
            "ct-segmented-litinski",
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


def test_kpbc_smoke_runner_accepts_pbc_input_and_tracegen_repo(tmp_path):
    pbc = tmp_path / "toy_pbc_post_opt.txt"
    pbc.write_text("qreg q[2];\nt_pauli +ZZ;\n", encoding="utf-8")
    router = tmp_path / "router.py"
    router.write_text(
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "json.dump([{'id': 0, 'op': 't_pauli', 'logical_qubits': [0], "
        "'routed_l': 1, 'routed_area': 1}], open(out, 'w'))\n",
        encoding="utf-8",
    )
    tracegen_repo = tmp_path / "tracegen"
    tracegen_repo.mkdir()
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_kpbc_smoke.py",
            "--pbc",
            str(pbc),
            "--out-dir",
            str(out_dir),
            "--k",
            "1",
            "--compiler-mode",
            "pbc-naive-ladder",
            "--router-bin",
            sys.executable,
            "--router-arg",
            str(router),
            "--tracegen-repo",
            str(tracegen_repo),
            "--d",
            "5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (out_dir / "candidate.kpbc.jsonl").exists()
    assert f"tracegen_repo={tracegen_repo}" in (out_dir / "run.log").read_text()
