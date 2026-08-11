import pytest

from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pbc_input import SourcePBCOp
from ftcircuitbench.semi_pbc.reducer import reduce_measurements


def src(idx, op, signed):
    return SourcePBCOp(
        id=idx,
        op=op,
        term=PauliTerm.from_full_width(signed, source_id=f"line{idx}"),
        source_id=f"line{idx}",
    )


def test_reducer_multiplies_by_prior_measurement_to_reduce_weight():
    ops = [
        src(0, "m_pauli", "+ZZII"),
        src(1, "m_pauli", "+ZZZI"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[1].term.pairs == (("q2", "Z"),)
    assert reduced[1].result_terms == ("src0",)
    assert reduced[1].result_const == 0


def test_reducer_normalizes_negative_replacement_sign_into_result_const():
    ops = [
        src(0, "m_pauli", "+ZI"),
        src(1, "m_pauli", "-ZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[1].term.sign == 1
    assert reduced[1].term.pairs == (("q1", "Z"),)
    assert reduced[1].result_terms == ("src0",)
    assert reduced[1].result_const == 1


def test_reducer_uses_deterministic_tie_breaking():
    ops = [
        src(0, "m_pauli", "+ZZI"),
        src(1, "m_pauli", "+ZIZ"),
        src(2, "m_pauli", "+ZZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[2].used_source_ids == ("line0",)


def test_reducer_drops_only_prior_measurements_invalidated_by_intervening_t_rotation():
    ops = [
        src(0, "m_pauli", "+ZZI"),
        src(1, "t_pauli", "+XII"),
        src(2, "m_pauli", "+IZI"),
        src(3, "m_pauli", "+ZZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[3].term.pairs == (("q0", "Z"), ("q2", "Z"))
    assert reduced[3].used_source_ids == ("line2",)
    assert reduced[3].result_terms == ("src2",)


def test_reducer_allows_prior_measurement_across_commuting_t_rotation():
    ops = [
        src(0, "m_pauli", "+ZZI"),
        src(1, "t_pauli", "+ZII"),
        src(2, "m_pauli", "+ZZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[2].term.pairs == (("q2", "Z"),)
    assert reduced[2].used_source_ids == ("line0",)


def test_greedy_order_zero_only_considers_most_recent_eligible_measurement():
    ops = [
        src(0, "m_pauli", "+ZZZI"),
        src(1, "m_pauli", "+ZZII"),
        src(2, "m_pauli", "+ZZZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=0)
    assert reduced[2].used_source_ids == ("line1",)
    assert reduced[2].term.pairs == (("q0", "Z"), ("q1", "Z"), ("q3", "Z"))


def test_greedy_order_two_can_use_pair_when_singletons_do_not_help():
    ops = [
        src(0, "m_pauli", "+ZIIZ"),
        src(1, "m_pauli", "+IZIZ"),
        src(2, "m_pauli", "+ZZZI"),
    ]
    reduced_one = reduce_measurements(ops, greedy_order=1)
    assert reduced_one[2].used_source_ids == ()
    reduced_two = reduce_measurements(ops, greedy_order=2)
    assert reduced_two[2].used_source_ids == ("line0", "line1")
    assert reduced_two[2].result_terms == ("src0", "src1")
    assert reduced_two[2].term.pairs == (("q2", "Z"),)


def test_reducer_rejects_invalid_greedy_order():
    with pytest.raises(ValueError, match="greedy_order"):
        reduce_measurements([src(0, "m_pauli", "+Z")], greedy_order=3)


def test_reducer_uses_prior_representative_for_candidate_multiplication():
    ops = [
        src(0, "m_pauli", "+ZZI"),
        src(1, "m_pauli", "+ZZZ"),
        src(2, "t_pauli", "+XII"),
        src(3, "m_pauli", "+ZZI"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[1].term.pairs == (("q2", "Z"),)
    assert reduced[3].used_source_ids == ()
    assert reduced[3].term.pairs == (("q0", "Z"), ("q1", "Z"))


def test_reducer_composes_result_terms_when_using_reduced_prior_representative():
    ops = [
        src(0, "m_pauli", "+IIX"),
        src(1, "m_pauli", "+IXX"),
        src(2, "m_pauli", "+XXI"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[1].term.pairs == (("q1", "X"),)
    assert reduced[1].result_terms == ("src0",)
    assert reduced[2].term.pairs == (("q0", "X"),)
    assert reduced[2].used_source_ids == ("line1",)
    assert reduced[2].result_terms == ("src0", "src1")


def test_reducer_composes_prior_result_const_when_mapping_representatives():
    ops = [
        src(0, "m_pauli", "+ZI"),
        src(1, "m_pauli", "-ZZ"),
        src(2, "t_pauli", "+XI"),
        src(3, "m_pauli", "+ZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[1].result_const == 1
    assert reduced[3].term.pairs == (("q0", "Z"),)
    assert reduced[3].used_source_ids == ("line1",)
    assert reduced[3].result_terms == ("src0", "src1")
    assert reduced[3].result_const == 1
