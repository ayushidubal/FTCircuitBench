import json
import subprocess
import sys


def test_prepare_kpbc_server_run_writes_smallest_per_family_manifest(tmp_path):
    pbc_dir = tmp_path / "pbc"
    (pbc_dir / "adder" / "adder_4q" / "GS" / "precision_level_10").mkdir(
        parents=True
    )
    (pbc_dir / "adder" / "adder_10q" / "GS" / "precision_level_10").mkdir(
        parents=True
    )
    (pbc_dir / "qft" / "qft_4q" / "GS" / "precision_level_10").mkdir(parents=True)
    (pbc_dir / "adder" / "adder_4q" / "GS" / "precision_level_10" / "adder_4q_gs_prec10_pbc_post_opt.txt").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nt_pauli +ZIII;\n',
        encoding="utf-8",
    )
    (pbc_dir / "adder" / "adder_10q" / "GS" / "precision_level_10" / "adder_10q_gs_prec10_pbc_post_opt.txt").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[10];\nt_pauli +ZIIIIIIIII;\n',
        encoding="utf-8",
    )
    (pbc_dir / "qft" / "qft_4q" / "GS" / "precision_level_10" / "qft_4q_gs_prec10_pbc_post_opt.txt").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nt_pauli +ZIII;\n',
        encoding="utf-8",
    )
    ct_dir = tmp_path / "ct"
    (ct_dir / "adder").mkdir(parents=True)
    (ct_dir / "qft").mkdir(parents=True)
    (ct_dir / "adder" / "adder_4q.qasm").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nt q[0];\n',
        encoding="utf-8",
    )
    (ct_dir / "qft" / "qft_4q.qasm").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nt q[0];\n',
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
            "--pbc-dir",
            str(pbc_dir),
            "--ct-dir",
            str(ct_dir),
            "--out-dir",
            str(out_dir),
            "--compiler-modes",
            "ct-segmented-litinski",
            "pbc-naive-ladder",
            "--k-values",
            "mid",
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
    assert len(rows) == 4
    assert {row["family"] for row in rows} == {"adder", "qft"}
    assert {row["circuit_name"] for row in rows} == {
        "adder_4q_gs_prec10_pbc_post_opt",
        "qft_4q_gs_prec10_pbc_post_opt",
    }
    assert {(row["decoder"], row["compiler_mode"], row["k"]) for row in rows} == {
        ("toy", "ct-segmented-litinski", 2),
        ("toy", "pbc-naive-ladder", 2),
    }
    assert all("adder_10q" not in row["input"] for row in rows)
    assert all("202" not in row["run_dir"] for row in rows)
    assert 'xargs -P "$JOBS"' in (out_dir / "run_server.sh").read_text()
    assert (out_dir / "README.md").exists()
