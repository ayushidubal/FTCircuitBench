import json
import subprocess
import sys


def test_prepare_kpbc_server_run_writes_manifest_without_datetime_names(tmp_path):
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    (circuits / "adder_4q.qasm").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nt q[0];\n',
        encoding="utf-8",
    )
    (circuits / "qft_2q.qasm").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nt q[1];\n',
        encoding="utf-8",
    )
    decoders = tmp_path / "decoders.json"
    decoders.write_text(
        json.dumps(
            [
                {
                    "name": "toy",
                    "t0": 0.0,
                    "alpha": 1.0,
                    "beta_l": 1.0,
                    "beta_d": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "kpbc_runs"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_kpbc_server_run.py",
            "--circuits-dir",
            str(circuits),
            "--out-dir",
            str(out_dir),
            "--compiler-modes",
            "segmented-litinski",
            "naive",
            "--k-values",
            "1,n,mid",
            "--decoders",
            str(decoders),
            "--d",
            "5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 12
    assert {(row["decoder"], row["compiler_mode"], row["k"]) for row in rows} >= {
        ("toy", "segmented-litinski", 1),
        ("toy", "segmented-litinski", 2),
        ("toy", "segmented-litinski", 4),
        ("toy", "naive", 1),
    }
    assert all("202" not in row["run_dir"] for row in rows)
    assert 'xargs -P "$JOBS"' in (out_dir / "run_server.sh").read_text()
    assert (out_dir / "README.md").exists()
