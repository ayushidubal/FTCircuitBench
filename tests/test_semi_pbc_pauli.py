from __future__ import annotations

import pytest

from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_pauli_term_canonicalizes_data_then_ancilla():
    term = PauliTerm.from_pairs(
        [("a2", "X"), ("q10", "Z"), ("q2", "Y"), ("a0", "Z")]
    )
    assert term.pairs == (("q2", "Y"), ("q10", "Z"), ("a0", "Z"), ("a2", "X"))
    assert term.weight == 4


def test_pauli_term_rejects_duplicate_non_identity_qubit():
    with pytest.raises(ValueError, match="duplicate"):
        PauliTerm.from_pairs([("q0", "X"), ("q0", "Z")])


def test_full_width_conversion_omits_identity_in_sparse_form():
    term = PauliTerm.from_full_width("+IXYZ", source_id="line7")
    assert term.sign == 1
    assert term.pairs == (("q1", "X"), ("q2", "Y"), ("q3", "Z"))
    assert term.to_full_width(4) == "+IXYZ"


def test_pauli_multiplication_tracks_real_signed_result():
    left = PauliTerm.from_pairs([("q0", "X"), ("q1", "Z")])
    right = PauliTerm.from_pairs([("q0", "X"), ("q2", "Y")], sign=-1)
    product = left.multiply_real(right)
    assert product.sign == -1
    assert product.pairs == (("q1", "Z"), ("q2", "Y"))


def test_pauli_multiplication_rejects_imaginary_phase():
    left = PauliTerm.from_pairs([("q0", "X")])
    right = PauliTerm.from_pairs([("q0", "Y")])
    with pytest.raises(ValueError, match="imaginary"):
        left.multiply_real(right)


def test_pauli_multiplication_allows_real_negative_phase():
    left = PauliTerm.from_pairs([("q0", "X"), ("q1", "Y")])
    right = PauliTerm.from_pairs([("q0", "Y"), ("q1", "Z")])
    product = left.multiply_real(right)
    assert product.sign == -1
    assert product.pairs == (("q0", "Z"), ("q1", "X"))


def test_commutation_uses_anticommutation_parity():
    assert (
        PauliTerm.from_pairs([("q0", "X")]).commutes_with(
            PauliTerm.from_pairs([("q0", "Y")])
        )
        is False
    )
    assert (
        PauliTerm.from_pairs([("q0", "X"), ("q1", "Z")]).commutes_with(
            PauliTerm.from_pairs([("q0", "Y"), ("q1", "X")])
        )
        is True
    )
