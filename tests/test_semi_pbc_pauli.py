from __future__ import annotations

import pytest

from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_pauli_term_canonicalizes_data_then_ancilla():
    term = PauliTerm.from_pairs([("a2", "X"), ("q10", "Z"), ("q2", "Y"), ("a0", "Z")])
    assert term.pairs == (("q2", "Y"), ("q10", "Z"), ("a0", "Z"), ("a2", "X"))
    assert term.weight == 4


def test_pauli_term_equality_ignores_source_id_but_preserves_metadata():
    left = PauliTerm.from_pairs([("q0", "Z")], source_id="source-a")
    right = PauliTerm.from_pairs([("q0", "Z")], source_id="source-b")

    assert left == right
    assert hash(left) == hash(right)
    assert left.source_id == "source-a"
    assert right.source_id == "source-b"


def test_direct_pauli_term_construction_canonicalizes_pairs():
    term = PauliTerm((("a1", "X"), ("q2", "Z"), ("q0", "I"), ("q1", "Y")))

    assert term.pairs == (("q1", "Y"), ("q2", "Z"), ("a1", "X"))
    assert term.sign == 1


def test_pauli_term_rejects_duplicate_non_identity_qubit():
    with pytest.raises(ValueError, match="duplicate"):
        PauliTerm.from_pairs([("q0", "X"), ("q0", "Z")])


@pytest.mark.parametrize("qubit", ["q01", "a01"])
def test_pauli_term_rejects_leading_zero_qubit_ids(qubit):
    with pytest.raises(ValueError, match="leading zero|unsupported qubit"):
        PauliTerm.from_pairs([(qubit, "Z")])


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PauliTerm.from_pairs([("q0", "Z")], sign=True), "sign"),
        (lambda: PauliTerm((("q0", "Z"),), sign=True), "sign"),
        (lambda: PauliTerm((("q0", "A"),), sign=1), "Pauli"),
        (lambda: PauliTerm(((0, "Z"),), sign=1), "qubit"),
        (lambda: PauliTerm((("q0", 0),), sign=1), "Pauli"),
        (lambda: PauliTerm((1,), sign=1), "pair"),
        (lambda: PauliTerm(None, sign=1), "terms"),
        (lambda: PauliTerm((("q0", "X"), ("q0", "Z")), sign=1), "duplicate"),
        (lambda: PauliTerm((("q01", "Z"),), sign=1), "leading zero|qubit"),
        (lambda: PauliTerm((("q0", "Z"),), sign=1, source_id=1), "source_id"),
    ],
)
def test_pauli_term_rejects_invalid_construction(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_full_width_conversion_omits_identity_in_sparse_form():
    term = PauliTerm.from_full_width("+IXYZ", source_id="line7")
    assert term.sign == 1
    assert term.pairs == (("q1", "X"), ("q2", "Y"), ("q3", "Z"))
    assert term.to_full_width(4) == "+IXYZ"


@pytest.mark.parametrize("data_qubits", [-1, True])
def test_to_full_width_rejects_invalid_data_qubit_width(data_qubits):
    with pytest.raises(ValueError, match="data_qubits"):
        PauliTerm.from_pairs([]).to_full_width(data_qubits)


def test_from_full_width_rejects_non_string_input():
    with pytest.raises(ValueError, match="signed Pauli string"):
        PauliTerm.from_full_width(123)


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
