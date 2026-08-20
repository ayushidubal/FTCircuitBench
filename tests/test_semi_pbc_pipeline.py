import json

import pytest

from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp, write_jsonl
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_file, compile_pbc_text
from ftcircuitbench.semi_pbc.schedule import compute_summary


def test_compute_summary_uses_default_phase1_latencies():
    header = SemiPBCHeader(k=1, data_qubits=2)
    ops = [
        SemiPBCOp.clifford(0, "h", ("q0",)),
        SemiPBCOp.clifford(1, "cx", ("q0", "q1")),
        SemiPBCOp.pauli_rotation(2, PauliTerm.from_pairs([("q0", "Z")])),
        SemiPBCOp.xor(3, target="src0", terms=(), const=0),
    ]
    summary = compute_summary(header, ops, input_op_count=1, max_input_weight=2)
    assert summary["output_op_count"] == 4
    assert summary["max_output_weight"] == 1
    assert summary["latency_weighted_depth"] == 3


def test_compute_summary_reports_unique_allocated_ancillas():
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.alloc(0, "a0"),
        SemiPBCOp.release(1, "a0"),
        SemiPBCOp.alloc(2, "a1"),
        SemiPBCOp.release(3, "a1"),
    ]
    summary = compute_summary(header, ops, input_op_count=1, max_input_weight=2)
    assert summary["ancilla_count"] == 2
    assert summary["max_live_ancillas"] == 1


def test_compute_summary_reports_peak_live_ancillas():
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.alloc(0, "a0"),
        SemiPBCOp.alloc(1, "a1"),
        SemiPBCOp.release(2, "a1"),
        SemiPBCOp.release(3, "a0"),
    ]
    summary = compute_summary(header, ops, input_op_count=1, max_input_weight=2)
    assert summary["ancilla_count"] == 2
    assert summary["max_live_ancillas"] == 2


def test_compile_pbc_text_lowers_to_hard_cap_and_preserves_source_results():
    text = "qreg q[3];\nt_pauli +XYZ;\nm_pauli -ZZZ;\n"
    result = compile_pbc_text(text, k=1, measurement_reducer="none")
    assert result.summary["max_input_weight"] == 3
    assert result.summary["max_output_weight"] == 1
    assert result.summary["input_op_count"] == 2
    assert any(
        op.op == "xor" and op.target == "src1" and op.const == 1 for op in result.ops
    )
    for op in result.ops:
        op.validate(result.header)


def test_compile_pbc_text_uses_reducer_before_lowering():
    text = "qreg q[3];\nm_pauli +ZZI;\nm_pauli +ZZZ;\n"
    result = compile_pbc_text(
        text,
        k=1,
        measurement_reducer="peres-galvao-greedy",
        greedy_order=1,
    )
    physical_measurements = [op for op in result.ops if op.op == "m_pauli"]
    assert any(
        op.source_id == "line3" and op.term.pairs == (("q2", "Z"),)
        for op in physical_measurements
    )
    assert any(
        op.op == "xor"
        and op.target == "src1"
        and op.terms == ("c1", "src0")
        and op.const == 0
        for op in result.ops
    )


def test_compile_pbc_text_lowers_reduced_representative_to_max_k_block():
    text = "qreg q[4];\nm_pauli +ZIII;\nm_pauli +ZZZZ;\n"
    result = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="peres-galvao-greedy",
        greedy_order=1,
    )

    assert not any(op.op == "alloc" for op in result.ops)
    assert any(
        op.op == "m_pauli"
        and op.source_id == "line3"
        and op.term.pairs == (("q1", "Z"), ("q2", "Z"))
        for op in result.ops
    )
    assert any(
        op.op == "xor"
        and op.target == "src1"
        and op.terms == ("c1", "src0")
        and op.const == 0
        for op in result.ops
    )
    assert result.summary["max_output_weight"] == 2


def test_compile_pbc_text_combines_reducer_sign_adjustment_into_final_xor():
    text = "qreg q[2];\nm_pauli +ZI;\nm_pauli -ZZ;\n"
    result = compile_pbc_text(
        text,
        k=1,
        measurement_reducer="peres-galvao-greedy",
        greedy_order=1,
    )
    assert any(
        op.op == "m_pauli"
        and op.source_id == "line3"
        and op.term.pairs == (("q1", "Z"),)
        for op in result.ops
    )
    assert any(
        op.op == "xor"
        and op.target == "src1"
        and op.terms == ("c1", "src0")
        and op.const == 1
        for op in result.ops
    )


def test_compile_pbc_text_is_deterministic():
    text = "qreg q[3];\nm_pauli +ZZI;\nm_pauli +ZZZ;\n"
    first = compile_pbc_text(text, k=1).jsonl_records()
    second = compile_pbc_text(text, k=1).jsonl_records()
    assert first == second


def test_compile_pbc_file_reads_source_path(tmp_path):
    path = tmp_path / "toy.pbc"
    path.write_text("qreg q[1];\nt_pauli +Z;\n")
    result = compile_pbc_file(path, k=1)
    assert result.header == SemiPBCHeader(k=1, data_qubits=1)
    assert result.summary["input_op_count"] == 1


def test_compile_result_jsonl_records_can_be_written(tmp_path):
    result = compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1)
    path = tmp_path / "out.jsonl"
    write_jsonl(path, result.header, result.ops)
    assert result.jsonl_records() == [
        json.loads(line) for line in path.read_text().splitlines()
    ]


def test_compile_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=0)


def test_compile_validates_options_before_parsing_input():
    with pytest.raises(ValueError, match="k"):
        compile_pbc_text("not pbc", k=0)
    with pytest.raises(ValueError, match="greedy_order"):
        compile_pbc_text(
            "not pbc",
            k=1,
            measurement_reducer="peres-galvao-greedy",
            greedy_order=3,
        )


def test_compile_rejects_unsupported_reducer_option():
    with pytest.raises(ValueError, match="measurement_reducer"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, measurement_reducer="bad")


def test_compile_allows_zero_ancilla_budget_for_data_compression():
    result = compile_pbc_text("qreg q[2];\nm_pauli +ZZ;\n", k=1, ancilla_budget=0)

    assert result.summary["max_output_weight"] == 1
    assert result.summary["ancilla_count"] == 0
    assert result.summary["max_live_ancillas"] == 0
    assert not any(op.op == "alloc" for op in result.ops)


def test_compile_lowers_sequential_measurements_without_ancillas():
    result = compile_pbc_text(
        "qreg q[2];\nm_pauli +ZZ;\nm_pauli +ZZ;\n",
        k=1,
        measurement_reducer="none",
        ancilla_budget=0,
    )

    assert result.summary["max_output_weight"] == 1
    assert [op.op for op in result.ops].count("m_pauli") == 2
    assert result.summary["ancilla_count"] == 0
    assert result.summary["max_live_ancillas"] == 0


def test_compile_emits_only_xor_for_identity_reduced_measurement():
    result = compile_pbc_text(
        "qreg q[2];\nm_pauli +ZZ;\nm_pauli +ZZ;\n",
        k=1,
        measurement_reducer="peres-galvao-greedy",
        greedy_order=1,
    )

    assert not any(op.op == "m_pauli" and op.source_id == "line3" for op in result.ops)
    assert any(
        op.op == "xor"
        and op.source_id == "line3"
        and op.target == "src1"
        and op.terms == ("src0",)
        and op.const == 0
        for op in result.ops
    )


def test_compile_rejects_unsupported_strategy_options():
    with pytest.raises(ValueError, match="objective"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, objective="space")
    with pytest.raises(ValueError, match="rotation_lowering"):
        compile_pbc_text(
            "qreg q[1];\nt_pauli +Z;\n",
            k=1,
            rotation_lowering="moflic-paler-k2",
        )
    with pytest.raises(ValueError, match="measurement_lowering"):
        compile_pbc_text(
            "qreg q[1];\nm_pauli +Z;\n", k=1, measurement_lowering="chunks"
        )


def test_compile_result_sidecar_can_be_disabled():
    result = compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, emit_sidecar=False)
    assert result.sidecar is None


def test_compile_result_sidecar_records_output_provenance_when_enabled():
    result = compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, emit_sidecar=True)
    assert result.sidecar["format"] == "semi-pbc-sidecar"
    assert result.sidecar["k"] == 1
    assert result.sidecar["provenance"][0]["id"] == 0
    assert result.sidecar["provenance"][0]["source_id"] == "line2"


def test_compile_result_sidecar_records_reducer_dependencies():
    result = compile_pbc_text(
        "qreg q[3];\nm_pauli +ZZI;\nm_pauli +ZZZ;\n",
        k=1,
        emit_sidecar=True,
    )

    assert any(
        record["source_id"] == "line3"
        and record["used_source_ids"] == ["line2"]
        and record["result_terms"] == ["src0"]
        and record["result_const"] == 0
        for record in result.sidecar["provenance"]
    )
