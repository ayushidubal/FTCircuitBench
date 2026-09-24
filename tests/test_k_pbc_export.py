from ftcircuitbench.k_pbc.export import semi_pbc_result_to_kpbc
from ftcircuitbench.k_pbc.ir import KPBCHeader
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_text


def test_naive_export_reuses_existing_capped_pipeline():
    result = compile_pbc_text("qreg q[2];\nt_pauli +ZZ;\nm_pauli +ZI;\n", k=1)
    header, ops = semi_pbc_result_to_kpbc(result)
    assert header == KPBCHeader(k=1, data_qubits=2)
    assert [op.op for op in ops] == ["cx", "t_pauli", "cx", "m_pauli", "xor"]
    assert ops[1].to_record()["angle_num"] == 1
    assert ops[1].to_record()["angle_den"] == 8
    assert ops[3].result == "c0"
    assert ops[4].target == "src1"
    assert ops[4].terms == ("c0",)
    assert all(op.term is None or op.term.weight <= 1 for op in ops)


def test_naive_export_accepts_expanded_kpbc_cliffords():
    class Header:
        k = 1
        data_qubits = 1

    class SourceOp:
        def __init__(self, id, op):
            self.id = id
            self.op = op
            self.qubits = ("q0",)
            self.source_id = None

    class Result:
        header = Header()
        ops = tuple(SourceOp(id, op) for id, op in enumerate(("i", "x", "y", "z")))

    header, ops = semi_pbc_result_to_kpbc(Result())

    assert header == KPBCHeader(k=1, data_qubits=1)
    assert [op.op for op in ops] == ["i", "x", "y", "z"]
