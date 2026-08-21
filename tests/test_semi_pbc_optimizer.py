from __future__ import annotations

import pytest

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.optimizer import (
    cancel_adjacent_inverse_cliffords,
    candidate_retained_blocks,
    choose_retained_block,
    optimize_rotation_run,
)
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.reducer import ReducedSourceOp


def test_choose_retained_block_prefers_neighbor_overlap():
    term = PauliTerm.from_full_width("+ZZZZ")
    neighbor = PauliTerm.from_full_width("+IIZZ")

    block = choose_retained_block(term, k=2, neighbor_terms=(neighbor,))

    assert block.retained_qubits == ("q2", "q3")
    assert block.extra_qubits == ("q0", "q1")
    assert block.target == "q2"


def test_choose_retained_block_uses_canonical_tie_break():
    term = PauliTerm.from_full_width("+ZZZZ")

    block = choose_retained_block(term, k=2, neighbor_terms=())

    assert block.retained_qubits == ("q0", "q1")
    assert block.extra_qubits == ("q2", "q3")
    assert block.target == "q0"


def test_choose_retained_block_rejects_invalid_k():
    term = PauliTerm.from_full_width("+Z")

    with pytest.raises(ValueError, match="k"):
        choose_retained_block(term, k=0)


def test_candidate_retained_blocks_include_target_choices():
    term = PauliTerm.from_full_width("+ZZZI")

    blocks = candidate_retained_blocks(term, k=2)

    assert any(block.retained_qubits == ("q0", "q1") for block in blocks)
    assert any(block.retained_qubits == ("q1", "q0") for block in blocks)
    assert any(block.retained_qubits == ("q0", "q2") for block in blocks)
    assert any(block.retained_qubits == ("q2", "q0") for block in blocks)


def test_candidate_retained_blocks_returns_single_low_weight_candidate():
    term = PauliTerm.from_full_width("+ZZ")

    blocks = candidate_retained_blocks(term, k=2)

    assert blocks == (
        type(blocks[0])(
            retained_qubits=("q0", "q1"),
            extra_qubits=(),
            target="q0",
        ),
    )


def test_candidate_retained_blocks_put_neighbor_choice_first():
    term = PauliTerm.from_full_width("+ZZZI")
    neighbor = PauliTerm.from_full_width("+ZIZZ")

    blocks = candidate_retained_blocks(term, k=2, neighbor_terms=(neighbor,))

    assert blocks[0].retained_qubits == ("q0", "q2")


def test_cancel_adjacent_inverse_cliffords_removes_only_exact_pairs():
    ops = [
        SemiPBCOp.clifford(0, "h", ("q0",)),
        SemiPBCOp.clifford(1, "h", ("q0",)),
        SemiPBCOp.clifford(2, "s", ("q1",)),
        SemiPBCOp.clifford(3, "sdg", ("q1",)),
        SemiPBCOp.clifford(4, "cx", ("q0", "q1")),
        SemiPBCOp.clifford(5, "cx", ("q0", "q1")),
        SemiPBCOp.clifford(6, "h", ("q2",)),
        SemiPBCOp.clifford(7, "h", ("q3",)),
    ]

    optimized = cancel_adjacent_inverse_cliffords(ops)

    assert [op.id for op in optimized] == [6, 7]


def test_optimize_rotation_run_reuses_nonlocal_frame_choice():
    source_ops = (
        ReducedSourceOp(
            id=0,
            op="t_pauli",
            term=PauliTerm.from_full_width("+ZZZI", source_id="line2"),
            source_id="line2",
        ),
        ReducedSourceOp(
            id=1,
            op="t_pauli",
            term=PauliTerm.from_full_width("+ZIZZ", source_id="line3"),
            source_id="line3",
        ),
    )

    result = optimize_rotation_run(source_ops, start_id=10, k=2)

    assert [op.id for op in result.ops] == [10, 11, 12, 13]
    assert [op.op for op in result.ops] == ["cx", "t_pauli", "t_pauli", "cx"]
    assert result.ops[0].qubits == ("q2", "q0")
    assert result.ops[-1].qubits == ("q2", "q0")
    assert result.ops[1].term.pairs == (("q0", "Z"), ("q1", "Z"))
    assert result.ops[2].term.pairs == (("q0", "Z"), ("q3", "Z"))
    assert result.source_ops_by_output_id[10].source_id == "line2"
    assert result.source_ops_by_output_id[12].source_id == "line3"
