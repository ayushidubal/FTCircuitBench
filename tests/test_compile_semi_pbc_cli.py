import json
import subprocess
import sys


def test_compile_semi_pbc_cli_writes_jsonl_and_summary(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "toy.semi_pbc.jsonl"
    summary = tmp_path / "toy.summary.json"
    sidecar = tmp_path / "toy.sidecar.json"
    pbc.write_text("qreg q[2];\nt_pauli +ZZ;\nm_pauli -ZZ;\n")
    proc = subprocess.run(
        [
            sys.executable,
            "compile_semi_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--summary",
            str(summary),
            "--sidecar",
            str(sidecar),
            "--emit-sidecar",
            "--k",
            "1",
            "--measurement-reducer",
            "none",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["format"] == "semi-pbc"
    assert rows[0]["k"] == 1
    assert json.loads(summary.read_text())["max_output_weight"] == 1
    assert json.loads(sidecar.read_text())["format"] == "semi-pbc-sidecar"
    assert "max_output_weight=1" in proc.stdout


def test_compile_semi_pbc_cli_emits_default_sidecar(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "toy.semi_pbc.jsonl"
    pbc.write_text("qreg q[1];\nt_pauli +Z;\n")

    subprocess.run(
        [
            sys.executable,
            "compile_semi_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--k",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    sidecar = out.with_suffix(f"{out.suffix}.sidecar.json")
    assert json.loads(sidecar.read_text())["format"] == "semi-pbc-sidecar"


def test_compile_semi_pbc_cli_accepts_local_window_optimization(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "toy.semi_pbc.jsonl"
    summary = tmp_path / "toy.summary.json"
    pbc.write_text("qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n")

    subprocess.run(
        [
            sys.executable,
            "compile_semi_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--summary",
            str(summary),
            "--k",
            "2",
            "--measurement-reducer",
            "none",
            "--optimization",
            "local-window",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(summary.read_text())["optimization"] == "local-window"


def test_compile_semi_pbc_cli_accepts_rotation_dp_optimization(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "toy.semi_pbc.jsonl"
    summary = tmp_path / "toy.summary.json"
    pbc.write_text("qreg q[4];\nt_pauli +ZZZI;\nt_pauli +ZIZZ;\n")

    subprocess.run(
        [
            sys.executable,
            "compile_semi_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--summary",
            str(summary),
            "--k",
            "2",
            "--measurement-reducer",
            "none",
            "--optimization",
            "rotation-dp",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads(summary.read_text())
    assert data["optimization"] == "rotation-dp"
    assert data["output_op_count"] == 4


def test_compile_semi_pbc_cli_reports_invalid_inputs_without_traceback(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "toy.semi_pbc.jsonl"
    pbc.write_text("qreg q[1];\nt_pauli +Z;\n")

    proc = subprocess.run(
        [
            sys.executable,
            "compile_semi_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--k",
            "1",
            "--ancilla-budget",
            "-1",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr
    assert "ancilla-budget" in proc.stderr
    assert not out.exists()


def test_importing_semi_pbc_pipeline_does_not_import_matplotlib():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ftcircuitbench.semi_pbc.pipeline; "
                "print('matplotlib' in sys.modules)"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert proc.stdout.strip() == "False"
    assert proc.stderr == ""
