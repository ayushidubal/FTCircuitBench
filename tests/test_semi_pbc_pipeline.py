import json

import numpy as np
import pytest

from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp, write_jsonl
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_file, compile_pbc_text
from ftcircuitbench.semi_pbc.schedule import compute_summary
from tests.test_semi_pbc_lowering import (
    assert_allclose_up_to_global_phase,
    induced_source_projectors,
    pauli_rotation_matrix,
    semi_pbc_unitary,
    signed_pauli_projectors,
)


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


def test_compile_local_window_reduces_adjacent_identical_rotations():
    text = "qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n"
    baseline = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="none",
    )
    optimized = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="local-window",
    )

    assert baseline.summary["output_op_count"] == 10
    assert optimized.summary["output_op_count"] == 6
    assert optimized.summary["optimization"] == "local-window"
    assert optimized.summary["max_output_weight"] == 2
    assert [op.op for op in optimized.ops] == [
        "cx",
        "cx",
        "t_pauli",
        "t_pauli",
        "cx",
        "cx",
    ]


def test_compile_local_window_rotation_unitary_matches_source_after_cancellation():
    term = PauliTerm.from_full_width("+ZZZZ")
    result = compile_pbc_text(
        "qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n",
        k=2,
        measurement_reducer="none",
        optimization="local-window",
    )

    single_rotation = pauli_rotation_matrix(term, data_qubits=4)
    original = single_rotation @ single_rotation
    compiled = semi_pbc_unitary(result.ops, data_qubits=4)
    assert_allclose_up_to_global_phase(compiled, original)


def test_compile_local_window_measurement_projectors_match_source():
    term = PauliTerm.from_full_width("-ZZZZ")
    result = compile_pbc_text(
        "qreg q[4];\nm_pauli -ZZZZ;\n",
        k=2,
        measurement_reducer="none",
        optimization="local-window",
    )

    expected_zero, expected_one = signed_pauli_projectors(term, data_qubits=4)
    actual_zero, actual_one = induced_source_projectors(
        result.ops,
        source="src0",
        data_qubits=4,
    )
    assert np.allclose(actual_zero, expected_zero)
    assert np.allclose(actual_one, expected_one)


def test_compile_local_window_filters_sidecar_after_cancellation():
    result = compile_pbc_text(
        "qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n",
        k=2,
        measurement_reducer="none",
        optimization="local-window",
        emit_sidecar=True,
    )
    op_ids = {op.id for op in result.ops}
    sidecar_ids = {record["id"] for record in result.sidecar["provenance"]}

    assert sidecar_ids == op_ids
    assert result.sidecar["optimization"] == "local-window"


def test_compile_rotation_dp_reduces_run_beyond_local_window():
    text = "qreg q[4];\nt_pauli +ZZZI;\nt_pauli +ZIZZ;\n"
    local = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="local-window",
    )
    dp = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="rotation-dp",
    )

    assert local.summary["output_op_count"] == 6
    assert dp.summary["output_op_count"] == 4
    assert dp.summary["optimization"] == "rotation-dp"
    assert [op.op for op in dp.ops] == ["cx", "t_pauli", "t_pauli", "cx"]
    assert dp.summary["max_output_weight"] == 2


def test_compile_rotation_dp_unitary_matches_source():
    first_term = PauliTerm.from_full_width("+ZZZI")
    second_term = PauliTerm.from_full_width("+ZIZZ")
    result = compile_pbc_text(
        "qreg q[4];\nt_pauli +ZZZI;\nt_pauli +ZIZZ;\n",
        k=2,
        measurement_reducer="none",
        optimization="rotation-dp",
    )

    original = pauli_rotation_matrix(
        second_term, data_qubits=4
    ) @ pauli_rotation_matrix(first_term, data_qubits=4)
    compiled = semi_pbc_unitary(result.ops, data_qubits=4)
    assert_allclose_up_to_global_phase(compiled, original)


def test_compile_rotation_dp_sidecar_matches_output_ops():
    result = compile_pbc_text(
        "qreg q[4];\nt_pauli +ZZZI;\nt_pauli +ZIZZ;\n",
        k=2,
        measurement_reducer="none",
        optimization="rotation-dp",
        emit_sidecar=True,
    )
    op_ids = {op.id for op in result.ops}
    sidecar_ids = {record["id"] for record in result.sidecar["provenance"]}

    assert sidecar_ids == op_ids
    assert result.sidecar["optimization"] == "rotation-dp"


def test_compile_ai_trajectory_prefix_emits_shared_frame(monkeypatch):
    text = (
        "qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n"
    )
    analyze_calls = []

    def fake_analyze(window, **kwargs):
        analyze_calls.append(kwargs)
        assert window.num_qubits == 4
        assert window.signed_paulis == ("+ZZZZ", "+ZZZZ", "+ZZZZ", "+ZZZZ")
        assert kwargs["k"] == 2
        return {
            "status": "ok",
            "raw_actions": [0, 1],
            "decoded_solution": [("gate", 0, 0, 0), ("gate", 1, 0, 0)],
            "gateset": [("cx", (0, 2)), ("cx", (0, 3))],
            "num_qubits": 4,
            "replay_terms": ["+ZZZZ", "+ZZZZ", "+ZZZZ", "+ZZZZ"],
            "rotation_angle_signs": [1, 1, 1, 1],
            "k_terminal_prefix_length": 2,
            "pending_terms": ["+ZZII", "+ZZII", "+ZZII", "+ZZII"],
            "pending_weights": [2, 2, 2, 2],
            "pending_rotation_indices": [0, 1, 2, 3],
            "prefix_emissions": [],
            "solver_output_excerpt": "",
            "error": "",
        }

    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.pipeline.analyze_ai_pauli_window_trajectory",
        fake_analyze,
    )

    result = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="ai-trajectory-prefix",
    )

    assert result.summary["optimization"] == "ai-trajectory-prefix"
    assert result.summary["max_output_weight"] == 2
    assert [op.op for op in result.ops] == [
        "cx",
        "cx",
        "t_pauli",
        "t_pauli",
        "t_pauli",
        "t_pauli",
        "cx",
        "cx",
    ]
    assert result.ops[0].qubits == ("q2", "q0")
    assert result.ops[1].qubits == ("q3", "q0")
    assert [op.term.to_full_width(4) for op in result.ops if op.op == "t_pauli"] == [
        "+ZZII",
        "+ZZII",
        "+ZZII",
        "+ZZII",
    ]
    assert analyze_calls[0]["deterministic"] is True


def test_compile_ai_trajectory_prefix_unitary_matches_source(monkeypatch):
    text = (
        "qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n"
    )

    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.pipeline.analyze_ai_pauli_window_trajectory",
        lambda window, **kwargs: {
            "status": "ok",
            "raw_actions": [0, 1],
            "decoded_solution": [("gate", 0, 0, 0), ("gate", 1, 0, 0)],
            "gateset": [("cx", (0, 2)), ("cx", (0, 3))],
            "num_qubits": 4,
            "replay_terms": ["+ZZZZ", "+ZZZZ", "+ZZZZ", "+ZZZZ"],
            "rotation_angle_signs": [1, 1, 1, 1],
            "k_terminal_prefix_length": 2,
            "pending_terms": ["+ZZII", "+ZZII", "+ZZII", "+ZZII"],
            "pending_weights": [2, 2, 2, 2],
            "pending_rotation_indices": [0, 1, 2, 3],
            "prefix_emissions": [],
            "solver_output_excerpt": "",
            "error": "",
        },
    )

    result = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="ai-trajectory-prefix",
    )

    term = PauliTerm.from_full_width("+ZZZZ")
    original = np.linalg.matrix_power(pauli_rotation_matrix(term, data_qubits=4), 4)
    compiled = semi_pbc_unitary(result.ops, data_qubits=4)
    assert_allclose_up_to_global_phase(compiled, original)


def test_compile_ai_trajectory_prefix_falls_back_when_solver_fails(monkeypatch):
    text = (
        "qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n"
    )
    monkeypatch.setattr(
        "ftcircuitbench.semi_pbc.pipeline.analyze_ai_pauli_window_trajectory",
        lambda window, **kwargs: {
            "status": "failed",
            "error": "AI Pauli solver returned no trajectory",
            "solver_output_excerpt": "NaN",
        },
    )

    fallback = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="local-window",
    )
    result = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="ai-trajectory-prefix",
    )

    assert [op.to_record() for op in result.ops] == [
        op.to_record() for op in fallback.ops
    ]
    assert result.summary["optimization"] == "ai-trajectory-prefix"


def test_compile_rotation_dp_does_not_regress_local_window_at_run_boundaries():
    text = (
        "qreg q[5];\n"
        "t_pauli +IIIIZ;\n"
        "t_pauli +IZZIZ;\n"
        "t_pauli +IZZZZ;\n"
        "m_pauli +IZZZZ;\n"
    )
    local = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="local-window",
    )
    dp = compile_pbc_text(
        text,
        k=2,
        measurement_reducer="none",
        optimization="rotation-dp",
    )

    assert local.summary["output_op_count"] == 11
    assert dp.summary["output_op_count"] <= local.summary["output_op_count"]


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
    with pytest.raises(ValueError, match="optimization"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, optimization="global")
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
