from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftcircuitbench.semi_pbc.ir import (
    SemiPBCHeader,
    SemiPBCOp,
    read_jsonl,
    validate_program,
    write_jsonl,
)
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


def test_jsonl_round_trip_preserves_non_pauli_source_ids(tmp_path):
    path = tmp_path / "source_ids.semi_pbc.jsonl"
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp(0, "h", qubits=("q0",), source_id="src-h"),
        SemiPBCOp(1, "alloc", qubit="a0", basis="zero", source_id="src-alloc"),
        SemiPBCOp(2, "release", qubit="a0", source_id="src-release"),
        SemiPBCOp(3, "xor", target="src0", terms=(), const=1, source_id="src-xor"),
    ]
    write_jsonl(path, header, ops)
    assert read_jsonl(path) == (header, ops)


def test_direct_clifford_list_qubits_canonicalize_and_round_trip(tmp_path):
    path = tmp_path / "direct_clifford.semi_pbc.jsonl"
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp(0, "h", qubits=["q0"])

    assert op.qubits == ("q0",)

    write_jsonl(path, header, [op])
    assert read_jsonl(path) == (header, [op])


def test_direct_xor_list_terms_canonicalize_and_round_trip(tmp_path):
    path = tmp_path / "direct_xor.semi_pbc.jsonl"
    header = SemiPBCHeader(k=1, data_qubits=1)
    measurement = SemiPBCOp.measurement(
        0,
        PauliTerm.from_pairs([("q0", "Z")]),
        result="c0",
    )
    op = SemiPBCOp(1, "xor", target="src0", terms=["c0"])

    assert op.terms == ("c0",)

    write_jsonl(path, header, [measurement, op])
    assert read_jsonl(path) == (header, [measurement, op])


def test_write_jsonl_rejects_non_monotonic_ids(tmp_path):
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.clifford(1, "h", ("q0",)),
        SemiPBCOp.clifford(1, "s", ("q0",)),
    ]
    with pytest.raises(ValueError, match="monotonic"):
        write_jsonl(tmp_path / "bad.jsonl", header, ops)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SemiPBCOp.clifford(0, "h", None),
        lambda: SemiPBCOp.clifford(0, "h", (0,)),
        lambda: SemiPBCOp.xor(0, "src0", None),
        lambda: SemiPBCOp.xor(0, "src0", [0]),
    ],
)
def test_public_constructors_reject_missing_string_sequences(factory):
    with pytest.raises(ValueError, match="list or tuple"):
        factory()


def test_public_constructors_accept_list_and_tuple_string_sequences():
    clifford = SemiPBCOp.clifford(0, "h", ["q0"])
    xor = SemiPBCOp.xor(1, "src0", ("c0", "src1"))

    assert clifford.qubits == ("q0",)
    assert xor.terms == ("c0", "src1")


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


def test_read_jsonl_rejects_ancilla_use_before_alloc(tmp_path):
    path = tmp_path / "bad_ancilla_lifetime.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"h","qubits":["a0"]}\n'
    )
    with pytest.raises(ValueError, match="allocated|ancilla"):
        read_jsonl(path)


def test_read_jsonl_rejects_ancilla_use_after_release(tmp_path):
    path = tmp_path / "bad_ancilla_release.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"alloc","qubit":"a0","basis":"zero"}\n'
        '{"id":1,"op":"release","qubit":"a0"}\n'
        '{"id":2,"op":"h","qubits":["a0"]}\n'
    )
    with pytest.raises(ValueError, match="allocated|ancilla"):
        read_jsonl(path)


def test_read_jsonl_rejects_out_of_order_ancilla_allocation(tmp_path):
    path = tmp_path / "bad_ancilla_order.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"alloc","qubit":"a1","basis":"zero"}\n'
    )
    with pytest.raises(ValueError, match="allocation order|a0"):
        read_jsonl(path)


def test_validate_program_rejects_unreleased_ancilla_at_end():
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [SemiPBCOp.alloc(0, "a0")]

    with pytest.raises(ValueError, match="release|active|ancilla"):
        validate_program(header, ops)


def test_validate_program_allows_release_after_uncomputed_ancilla_use():
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.alloc(0, "a0"),
        SemiPBCOp.clifford(1, "h", ("a0",)),
        SemiPBCOp.clifford(2, "h", ("a0",)),
        SemiPBCOp.release(3, "a0"),
    ]

    validate_program(header, ops)


def test_read_jsonl_allows_release_after_uncomputed_ancilla_use(tmp_path):
    path = tmp_path / "uncomputed_ancilla_release.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"alloc","qubit":"a0","basis":"zero"}\n'
        '{"id":1,"op":"cx","qubits":["q0","a0"]}\n'
        '{"id":2,"op":"cx","qubits":["q0","a0"]}\n'
        '{"id":3,"op":"release","qubit":"a0"}\n'
    )

    header, ops = read_jsonl(path)
    assert header == SemiPBCHeader(k=1, data_qubits=1)
    assert [op.op for op in ops] == ["alloc", "cx", "cx", "release"]


def test_read_jsonl_rejects_undefined_classical_input(tmp_path):
    path = tmp_path / "bad_classical_dataflow.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"xor","target":"src0","terms":["c0"],"const":0}\n'
    )
    with pytest.raises(ValueError, match="defined|classical|c0"):
        read_jsonl(path)


def test_read_jsonl_rejects_duplicate_classical_definitions(tmp_path):
    path = tmp_path / "bad_classical_duplicate.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"m_pauli","terms":[["q0","Z"]],"sign":1,"result":"c0"}\n'
        '{"id":1,"op":"m_pauli","terms":[["q0","Z"]],"sign":1,"result":"c0"}\n'
    )
    with pytest.raises(ValueError, match="already defined|c0"):
        read_jsonl(path)


def test_read_jsonl_rejects_duplicate_xor_terms(tmp_path):
    path = tmp_path / "bad_duplicate_xor_terms.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"m_pauli","terms":[["q0","Z"]],"sign":1,"result":"c0"}\n'
        '{"id":1,"op":"xor","target":"src0","terms":["c0","c0"],"const":0}\n'
    )
    with pytest.raises(ValueError, match="duplicate|xor"):
        read_jsonl(path)


def test_read_jsonl_rejects_unsupported_operation_record(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":2}\n'
        '{"id":0,"op":"cz","qubits":["q0","q1"]}\n'
    )
    with pytest.raises(ValueError, match="unsupported"):
        read_jsonl(path)


def test_read_jsonl_rejects_leading_zero_qubit_with_line_context(tmp_path):
    path = tmp_path / "bad_leading_zero_qubit.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":2}\n'
        '{"id":0,"op":"t_pauli","terms":[["q01","Z"]],'
        '"sign":1,"angle_num":1,"angle_den":8}\n'
    )
    with pytest.raises(ValueError, match=r"line 2: .*leading zero|line 2: .*q01"):
        read_jsonl(path)


def test_read_jsonl_rejects_unknown_operation_field(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"h","qubits":["q0"],"extra":true}\n'
    )
    with pytest.raises(ValueError, match="unknown|unexpected"):
        read_jsonl(path)


def test_read_jsonl_rejects_unknown_header_field(tmp_path):
    path = tmp_path / "bad_header.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1,"extra":true}\n'
    )
    with pytest.raises(ValueError, match=r"line 1: .*unknown"):
        read_jsonl(path)


def test_read_jsonl_streams_without_path_read_text(tmp_path, monkeypatch):
    path = tmp_path / "streamed_read.semi_pbc.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"h","qubits":["q0"]}\n'
    )

    def fail_read_text(self, *args, **kwargs):
        raise AssertionError(f"{Path.read_text.__name__} should not be called")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    assert read_jsonl(path) == (
        SemiPBCHeader(k=1, data_qubits=1),
        [SemiPBCOp.clifford(0, "h", ("q0",))],
    )


def test_read_jsonl_rejects_blank_lines_with_line_context(tmp_path):
    path = tmp_path / "blank.jsonl"
    path.write_text('{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n' "\n")
    with pytest.raises(ValueError, match=r"line 2: .*blank"):
        read_jsonl(path)


def test_read_jsonl_wraps_operation_errors_with_line_context(tmp_path):
    path = tmp_path / "bad_op.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"h","qubits":["q0"]}\n'
        '{"id":1,"op":"m_pauli","terms":[["q0","Z"]]}\n'
    )
    with pytest.raises(ValueError, match=r"line 3: .*result"):
        read_jsonl(path)


def test_write_jsonl_does_not_commit_partial_output_on_iterator_failure(tmp_path):
    path = tmp_path / "streamed.semi_pbc.jsonl"
    header = SemiPBCHeader(k=1, data_qubits=1)

    def ops():
        yield SemiPBCOp.clifford(0, "h", ("q0",))
        raise RuntimeError("iterator stopped after first op")

    with pytest.raises(RuntimeError, match="iterator stopped"):
        write_jsonl(path, header, ops())

    assert not path.exists()


def test_write_jsonl_does_not_clobber_existing_output_on_iterator_failure(tmp_path):
    path = tmp_path / "existing.semi_pbc.jsonl"
    original = '{"format":"semi-pbc","version":1,"k":1,"data_qubits":0}\n'
    path.write_text(original)
    header = SemiPBCHeader(k=1, data_qubits=1)

    def ops():
        yield SemiPBCOp.clifford(0, "h", ("q0",))
        raise RuntimeError("iterator stopped after first op")

    with pytest.raises(RuntimeError, match="iterator stopped"):
        write_jsonl(path, header, ops())

    assert path.read_text() == original


def test_header_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        SemiPBCHeader(k=0, data_qubits=3)


def test_header_rejects_boolean_version():
    with pytest.raises(ValueError, match="version"):
        SemiPBCHeader(k=1, data_qubits=1, version=True)


def test_write_jsonl_rejects_non_string_source_id(tmp_path):
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp.clifford(0, "h", ("q0",), source_id=123)

    with pytest.raises(ValueError, match="source_id"):
        write_jsonl(tmp_path / "bad_source_id.jsonl", header, [op])


def test_pauli_term_invalid_label_is_rejected_before_jsonl_write():
    with pytest.raises(ValueError, match="Pauli"):
        PauliTerm((("q0", "A"),), sign=1)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PauliTerm((1,), sign=1), "pair"),
        (lambda: PauliTerm(None, sign=1), "terms"),
        (lambda: PauliTerm((("q0", "X"), ("q0", "Z")), sign=1), "duplicate"),
    ],
)
def test_pauli_term_constructor_rejects_malformed_pairs(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("qubit", ["q01", "a01"])
def test_validate_rejects_runtime_pauli_term_leading_zero_qubits(qubit):
    with pytest.raises(ValueError, match="leading zero|qubit"):
        PauliTerm(((qubit, "Z"),), sign=1)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SemiPBCOp(0, [], qubits=("q0",)), "op"),
        (SemiPBCOp(0, "h", qubits=("q0", 0)), "qubits"),
        (SemiPBCOp(0, "xor", target="src0", terms=None), "terms"),
        (SemiPBCOp(0, "xor", target="src0", terms=("c0", 0)), "terms"),
    ],
)
def test_validate_rejects_runtime_container_type_mismatches(op, message):
    header = SemiPBCHeader(k=1, data_qubits=1)
    with pytest.raises(ValueError, match=message):
        op.validate(header)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SemiPBCOp(0, "h", qubits=("q0",), target="src0"), "target.*h"),
        (SemiPBCOp(0, "release", qubit="a0", basis="zero"), "basis.*release"),
        (
            SemiPBCOp(0, "xor", target="src0", terms=("c0",), qubit="a0"),
            "qubit.*xor",
        ),
    ],
)
def test_validate_rejects_irrelevant_runtime_fields(op, message):
    header = SemiPBCHeader(k=1, data_qubits=1)
    with pytest.raises(ValueError, match=message):
        op.validate(header)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SemiPBCOp(0, "h", qubits=("q0",), target="src0"), "target.*h"),
        (SemiPBCOp(0, "release", qubit="a0", basis="zero"), "basis.*release"),
        (
            SemiPBCOp(0, "xor", target="src0", terms=("c0",), qubit="a0"),
            "qubit.*xor",
        ),
    ],
)
def test_to_record_rejects_irrelevant_runtime_fields(op, message):
    with pytest.raises(ValueError, match=message):
        op.to_record()


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SemiPBCOp(0, [], qubits=("q0",)), "op"),
        (SemiPBCOp(0, "h", qubits=("q0", 0)), "qubits"),
        (SemiPBCOp(0, "h", qubits=()), "one qubit"),
        (SemiPBCOp(0, "cx", qubits=("q0",)), "two qubits"),
        (SemiPBCOp(0, "alloc", basis="zero"), "ancilla"),
        (SemiPBCOp(0, "release"), "ancilla"),
        (
            SemiPBCOp(
                0,
                "t_pauli",
                term=PauliTerm.from_pairs([("q0", "Z")]),
                angle_num=True,
                angle_den=8,
            ),
            "integer",
        ),
        (
            SemiPBCOp(
                0,
                "t_pauli",
                term=PauliTerm.from_pairs([("q0", "Z")]),
                angle_num=1,
                angle_den=4,
            ),
            "angle_num=1",
        ),
        (
            SemiPBCOp(0, "m_pauli", term=PauliTerm.from_pairs([("q0", "Z")])),
            "result",
        ),
        (SemiPBCOp(0, "xor", target="src0", terms=("c0",), const=True), "integer"),
        (SemiPBCOp(0, "xor", target="src0", terms=None), "terms"),
        (SemiPBCOp(0, "h", qubits=("q0",), source_id=123), "source_id"),
    ],
)
def test_to_record_rejects_malformed_runtime_shape(op, message):
    with pytest.raises(ValueError, match=message):
        op.to_record()


def test_to_record_serializes_representative_valid_operations():
    term = PauliTerm.from_pairs([("q0", "Z")], sign=-1)

    assert SemiPBCOp.clifford(0, "h", ("q0",), source_id="src-h").to_record() == {
        "id": 0,
        "op": "h",
        "qubits": ["q0"],
        "source_id": "src-h",
    }
    assert SemiPBCOp.alloc(1, "a0").to_record() == {
        "id": 1,
        "op": "alloc",
        "qubit": "a0",
        "basis": "zero",
    }
    assert SemiPBCOp.release(2, "a0").to_record() == {
        "id": 2,
        "op": "release",
        "qubit": "a0",
    }
    assert SemiPBCOp.pauli_rotation(3, term).to_record() == {
        "id": 3,
        "op": "t_pauli",
        "terms": [["q0", "Z"]],
        "sign": -1,
        "angle_num": 1,
        "angle_den": 8,
    }
    assert SemiPBCOp.measurement(4, term, result="c0").to_record() == {
        "id": 4,
        "op": "m_pauli",
        "terms": [["q0", "Z"]],
        "sign": -1,
        "result": "c0",
    }
    assert SemiPBCOp.xor(5, target="src0", terms=("c0",), const=1).to_record() == {
        "id": 5,
        "op": "xor",
        "target": "src0",
        "terms": ["c0"],
        "const": 1,
    }


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


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SemiPBCOp(0, "h", qubits=(0,)), "qubit"),
        (SemiPBCOp.alloc(0, 0), "qubit"),
        (SemiPBCOp.release(0, 0), "qubit"),
        (
            SemiPBCOp.measurement(0, PauliTerm.from_pairs([("q0", "Z")]), result=0),
            "result",
        ),
        (SemiPBCOp.xor(0, target=0, terms=["c0"]), "target"),
        (SemiPBCOp(0, "xor", target="src0", terms=(0,)), "xor term"),
    ],
)
def test_validate_rejects_non_string_qubit_and_classical_fields(op, message):
    header = SemiPBCHeader(k=1, data_qubits=1)
    with pytest.raises(ValueError, match=message):
        op.validate(header)


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SemiPBCOp.clifford(-1, "h", ("q0",)), "non-negative"),
        (SemiPBCOp.clifford(0, "h", ()), "one qubit"),
        (SemiPBCOp.clifford(0, "s", ("q0", "q1")), "one qubit"),
        (SemiPBCOp.clifford(0, "cx", ("q0",)), "two qubits"),
        (SemiPBCOp.clifford(0, "cx", ("q0", "q0")), "distinct|duplicate"),
        (SemiPBCOp.alloc(0, "q0"), "ancilla"),
        (SemiPBCOp.alloc(0, "a0", basis="plus"), "basis"),
        (SemiPBCOp.release(0, "q0"), "ancilla"),
        (
            SemiPBCOp(
                0,
                "t_pauli",
                term=PauliTerm.from_pairs([("q0", "Z")]),
                angle_num=1,
                angle_den=4,
            ),
            "angle_num=1",
        ),
        (
            SemiPBCOp(
                0,
                "t_pauli",
                term=PauliTerm.from_pairs([("q0", "Z")]),
                angle_num=True,
                angle_den=8,
            ),
            "integer",
        ),
        (
            SemiPBCOp(0, "m_pauli", term=PauliTerm.from_pairs([("q0", "Z")])),
            "result",
        ),
        (SemiPBCOp.pauli_rotation(0, PauliTerm.from_pairs([])), "weight"),
        (
            SemiPBCOp.measurement(0, PauliTerm.from_pairs([("q0", "Z")]), result="m0"),
            "classical",
        ),
        (
            SemiPBCOp.measurement(0, PauliTerm.from_pairs([("q0", "Z")]), result="c01"),
            "leading zero|classical",
        ),
        (SemiPBCOp.xor(0, target="bit0", terms=["c0"]), "classical"),
        (SemiPBCOp.xor(0, target="src01", terms=["c0"]), "leading zero|classical"),
        (SemiPBCOp.xor(0, target="src0", terms=["bit0"]), "classical"),
        (SemiPBCOp.xor(0, target="src0", terms=["c01"]), "leading zero|classical"),
        (SemiPBCOp.xor(0, target="src0", terms=["c0"], const=2), "const"),
        (SemiPBCOp.xor(0, target="src0", terms=["c0"], const=True), "integer"),
        (SemiPBCOp.clifford(0, "cz", ("q0", "q1")), "unsupported"),
        (SemiPBCOp(0, "reset", qubit="q0"), "unsupported"),
    ],
)
def test_validate_rejects_malformed_operation_branches(op, message):
    header = SemiPBCHeader(k=1, data_qubits=2)
    with pytest.raises(ValueError, match=message):
        op.validate(header)


def test_measurement_result_requires_physical_classical_id():
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp.measurement(0, PauliTerm.from_pairs([("q0", "Z")]), result="src0")

    with pytest.raises(ValueError, match="physical|c<N>|result"):
        op.validate(header)


def test_xor_target_requires_source_classical_id():
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp.xor(0, target="c0", terms=("c0",))

    with pytest.raises(ValueError, match="source|src<N>|target"):
        op.validate(header)


def test_xor_allows_physical_and_source_inputs():
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp.xor(0, target="src0", terms=("c0", "src1"))

    op.validate(header)
