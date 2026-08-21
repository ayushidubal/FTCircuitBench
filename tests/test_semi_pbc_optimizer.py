from __future__ import annotations

import pytest

from ftcircuitbench.semi_pbc.ir import SemiPBCOp
from ftcircuitbench.semi_pbc.optimizer import (
    cancel_adjacent_inverse_cliffords,
    choose_retained_block,
)
from ftcircuitbench.semi_pbc.pauli import PauliTerm


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
