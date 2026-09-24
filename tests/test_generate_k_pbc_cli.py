import json
import subprocess
import sys


def test_generate_k_pbc_cli_writes_candidate(tmp_path):
    qasm = tmp_path / "toy.qasm"
    out = tmp_path / "toy.kpbc.jsonl"
    qasm.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nt q[0];\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "generate_k_pbc.py",
            "--qasm",
            str(qasm),
            "--out",
            str(out),
            "--k",
            "1",
            "--mode",
            "ct-segmented-litinski",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert out.exists()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[0] == {"format": "k-pbc", "version": 1, "k": 1, "data_qubits": 1}
    assert records[1]["op"] == "t_pauli"
    assert records[1]["angle_num"] == 1
    assert records[1]["angle_den"] == 8
    assert "wrote" in result.stdout


def test_generate_k_pbc_cli_writes_pbc_naive_ladder_candidate(tmp_path):
    pbc = tmp_path / "toy_pbc_post_opt.txt"
    out = tmp_path / "toy.kpbc.jsonl"
    pbc.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        "t_pauli +ZZ;\nm_pauli -ZI;\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "generate_k_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--k",
            "1",
            "--mode",
            "pbc-naive-ladder",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[0] == {"format": "k-pbc", "version": 1, "k": 1, "data_qubits": 2}
    assert [record["op"] for record in records[1:]] == [
        "cx",
        "t_pauli",
        "cx",
        "m_pauli",
        "xor",
    ]
    assert all(
        len(record.get("terms", ())) <= 1
        for record in records[1:]
        if record["op"] in {"t_pauli", "m_pauli"}
    )
