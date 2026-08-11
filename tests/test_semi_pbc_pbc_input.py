import pytest

from ftcircuitbench.semi_pbc.pbc_input import parse_pbc_file, parse_pbc_text


def test_parse_nwqec_pbc_text_to_source_ops():
    text = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    t_pauli +IXYZ;
    m_pauli -ZZII;
    """
    program = parse_pbc_text(text)
    assert program.data_qubits == 4
    assert [op.source_id for op in program.ops] == ["line5", "line6"]
    assert program.ops[0].op == "t_pauli"
    assert program.ops[0].term.pairs == (("q1", "X"), ("q2", "Y"), ("q3", "Z"))
    assert program.ops[1].op == "m_pauli"
    assert program.ops[1].term.sign == -1


def test_parse_rejects_malformed_pauli_length():
    text = "qreg q[2];\nt_pauli +XYZ;\n"
    with pytest.raises(ValueError, match="length"):
        parse_pbc_text(text)


def test_parse_infers_width_without_qreg_and_skips_comments():
    program = parse_pbc_text("// hi\n\nm_pauli +ZI;\n")
    assert program.data_qubits == 2
    assert program.ops[0].source_id == "line3"


def test_parse_rejects_malformed_sign():
    with pytest.raises(ValueError, match="unsupported"):
        parse_pbc_text("qreg q[1];\nt_pauli *Z;\n")


@pytest.mark.parametrize(
    "statement",
    [
        'include_bad "qelib1.inc";',
        "cregister c[1];",
        "OPENQASM_BAD 2.0;",
    ],
)
def test_parse_rejects_malformed_boilerplate_prefix_lookalikes(statement):
    text = f"qreg q[1];\n{statement}\nt_pauli +Z;\n"
    with pytest.raises(ValueError, match="unsupported"):
        parse_pbc_text(text)


def test_parse_rejects_generic_unsupported_operator():
    with pytest.raises(ValueError, match="unsupported"):
        parse_pbc_text("qreg q[1];\nh q[0];\n")


@pytest.mark.parametrize("statement", ["qreg r[1];", "qreg q[2]"])
def test_parse_rejects_malformed_qreg(statement):
    with pytest.raises(ValueError, match="qreg"):
        parse_pbc_text(f"{statement}\n")


def test_parse_pbc_file_reads_path(tmp_path):
    path = tmp_path / "toy.pbc"
    path.write_text("qreg q[1];\nt_pauli +Z;\n")

    assert parse_pbc_file(path) == parse_pbc_text(path.read_text())
