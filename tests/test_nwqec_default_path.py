"""
Smoke test: ensure default Gridsynth entrypoint runs with nwqec C++ when available.
Skips if nwqec is not installed.
"""

import pytest
from qiskit import QuantumCircuit

from ftcircuitbench.transpilers import transpile_to_gridsynth_clifford_t
from ftcircuitbench.transpilers.gs_transpiler import is_clifford_t_basis

nwqec = pytest.importorskip("nwqec")


def test_gridsynth_uses_cpp_when_available():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rz(0.3, 0)
    qc.cx(0, 1)

    ct = transpile_to_gridsynth_clifford_t(
        qc,
        gridsynth_precision=3,
        remove_final_measurements=True,
        force_python=False,
    )

    assert isinstance(ct, QuantumCircuit)
    assert is_clifford_t_basis(ct)
