from __future__ import annotations

import json

import pytest

from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp, read_jsonl, write_jsonl
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_jsonl_round_trip_canonical_terms(tmp_path):
    path = tmp_path / "toy.semi_pbc.jsonl"
    header = SemiPBCHeader(k=2, data_qubits=3)
    ops = [
        SemiPBCOp.pauli_rotation(
            0, PauliTerm.from_pairs([("q2", "Z"), ("q0", "X")]), source_id="line1"
        ),
        SemiPBCOp.measurement(
            1,
            PauliTerm.from_pairs([("q1", "Z")], sign=-1),
            result="c0",
            source_id="line2",
        ),
        SemiPBCOp.xor(2, target="src1", terms=["c0"], const=1),
    ]
    write_jsonl(path, header, ops)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {"format": "semi-pbc", "version": 1, "k": 2, "data_qubits": 3}
    assert rows[1]["terms"] == [["q0", "X"], ["q2", "Z"]]
    loaded_header, loaded_ops = read_jsonl(path)
    assert loaded_header == header
    assert loaded_ops == ops


def test_write_jsonl_rejects_non_monotonic_ids(tmp_path):
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.clifford(1, "h", ("q0",)),
        SemiPBCOp.clifford(1, "s", ("q0",)),
    ]
    with pytest.raises(ValueError, match="monotonic"):
        write_jsonl(tmp_path / "bad.jsonl", header, ops)


def test_read_jsonl_rejects_non_monotonic_ids(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":1,"op":"h","qubits":["q0"]}\n'
        '{"id":1,"op":"s","qubits":["q0"]}\n'
    )
    with pytest.raises(ValueError, match="monotonic"):
        read_jsonl(path)


def test_read_jsonl_rejects_malformed_operation_record(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"m_pauli","terms":[["q0","Z"]]}\n'
    )
    with pytest.raises(ValueError, match="result"):
        read_jsonl(path)


def test_header_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        SemiPBCHeader(k=0, data_qubits=3)


def test_pauli_op_rejects_weight_above_k_when_validated():
    header = SemiPBCHeader(k=1, data_qubits=2)
    op = SemiPBCOp.pauli_rotation(0, PauliTerm.from_pairs([("q0", "Z"), ("q1", "Z")]))
    with pytest.raises(ValueError, match="weight"):
        op.validate(header)


def test_pauli_op_rejects_data_qubit_outside_header_width():
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp.pauli_rotation(0, PauliTerm.from_pairs([("q1", "Z")]))
    with pytest.raises(ValueError, match="outside"):
        op.validate(header)
