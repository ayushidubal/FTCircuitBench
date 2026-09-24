import json

import pytest

from ftcircuitbench.k_pbc.ir import (
    KPBCHeader,
    KPBCOp,
    read_kpbc_jsonl,
    write_kpbc_jsonl,
)
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_kpbc_round_trips_gate_set(tmp_path):
    path = tmp_path / "toy.kpbc.jsonl"
    header = KPBCHeader(k=2, data_qubits=3)
    ops = [
        KPBCOp.clifford(0, "i", ("q0",)),
        KPBCOp.clifford(1, "x", ("q0",)),
        KPBCOp.clifford(2, "y", ("q1",)),
        KPBCOp.clifford(3, "z", ("q2",)),
        KPBCOp.clifford(4, "h", ("q0",)),
        KPBCOp.clifford(5, "s", ("q1",)),
        KPBCOp.clifford(6, "sdg", ("q1",)),
        KPBCOp.clifford(7, "cx", ("q0", "q2")),
        KPBCOp.t_pauli(8, PauliTerm.from_pairs([("q0", "Z"), ("q2", "X")])),
        KPBCOp.m_pauli(9, PauliTerm.from_pairs([("q1", "Z")]), result="c0"),
    ]
    write_kpbc_jsonl(path, header, ops)
    loaded_header, loaded_ops = read_kpbc_jsonl(path)
    assert loaded_header == header
    assert loaded_ops == tuple(ops)


def test_kpbc_rejects_pauli_above_k():
    header = KPBCHeader(k=1, data_qubits=2)
    op = KPBCOp.t_pauli(0, PauliTerm.from_pairs([("q0", "Z"), ("q1", "Z")]))
    with pytest.raises(ValueError, match="exceeds k"):
        header.validate_ops([op])


def test_kpbc_t_pauli_emits_fixed_pi_over_eight_angle(tmp_path):
    path = tmp_path / "angle.kpbc.jsonl"
    header = KPBCHeader(k=1, data_qubits=1)
    op = KPBCOp.t_pauli(0, PauliTerm.from_pairs([("q0", "Z")]))

    write_kpbc_jsonl(path, header, [op])

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[1]["angle_num"] == 1
    assert records[1]["angle_den"] == 8


def test_kpbc_reads_fastkpbc_t_pauli_angle_fixture(tmp_path):
    path = tmp_path / "fastkpbc_fixture.kpbc.jsonl"
    path.write_text(
        '{"format":"k-pbc","version":1,"k":2,"data_qubits":2}\n'
        '{"id":2,"op":"t_pauli","terms":[["q0","Z"],["q1","Z"]],'
        '"sign":1,"angle_num":1,"angle_den":8}\n'
    )

    loaded_header, loaded_ops = read_kpbc_jsonl(path)

    assert loaded_header == KPBCHeader(k=2, data_qubits=2)
    assert loaded_ops == (
        KPBCOp.t_pauli(2, PauliTerm.from_pairs([("q0", "Z"), ("q1", "Z")])),
    )


def test_kpbc_rejects_non_t_angle(tmp_path):
    path = tmp_path / "bad_angle.kpbc.jsonl"
    path.write_text(
        '{"format":"k-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"t_pauli","terms":[["q0","Z"]],"sign":1,'
        '"angle_num":1,"angle_den":4}\n'
    )

    with pytest.raises(ValueError, match="angle_num=1 and angle_den=8"):
        read_kpbc_jsonl(path)
