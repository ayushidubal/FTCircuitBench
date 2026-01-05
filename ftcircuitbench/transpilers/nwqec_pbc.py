"""
Thin entrypoint to run nwqec PBC pipeline and adapt to PBM circuit.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps

from ftcircuitbench.analyzer import analyze_pbc_circuit
from ftcircuitbench.pbc_converter.nwqec_adapter import pbc_qasm_to_pbm


def is_nwqec_available() -> bool:
    try:
        import nwqec as _nq  # noqa: F401

        return True
    except Exception:
        return False


def transpile_to_pbc_cpp(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    epsilon: float | None = None,
    t_opt: bool = False,
    keep_cx: bool = False,
    forbid_python_fallback: bool = True,
) -> Tuple[QuantumCircuit, Dict]:
    import nwqec as nq
    fuse_supported = hasattr(nq, "fuse_t")
    fuse_applied = False
    pre_opt_rotation_ops = 0
    pre_opt_measurement_ops = 0
    post_opt_rotation_ops = 0
    post_opt_measurement_ops = 0
    pre_opt_stats: Dict = {}

    if forbid_python_fallback and not getattr(nq, "WITH_GRIDSYNTH_CPP", False):
        raise RuntimeError(
            "nwqec C++ gridsynth not available; refusing Python fallback"
        )

    # Load circuit to nwqec Circuit (using file-based load to avoid QASMParser dependency)
    tmp_path = None
    try:
        if isinstance(circuit_input, QuantumCircuit):
            qasm = qasm2_dumps(circuit_input)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".qasm", delete=False
            ) as tmp:
                tmp.write(qasm)
                tmp_path = tmp.name
            circ = nq.load_qasm(tmp_path)
        elif isinstance(circuit_input, str):
            if is_file:
                circ = nq.load_qasm(circuit_input)
            else:
                # treat as QASM source string
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".qasm", delete=False
                ) as tmp:
                    tmp.write(circuit_input)
                    tmp_path = tmp.name
                circ = nq.load_qasm(tmp_path)
        else:
            raise TypeError("circuit_input must be QuantumCircuit or str")

        # Use top-level to_pbc and optional opt_t
        kwargs = {}
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        circ = nq.to_pbc(circ, keep_cx=keep_cx, **kwargs)
        pre_opt_counts = circ.count_ops()
        pre_opt_rotation_ops = pre_opt_counts.get("t_pauli", 0)
        pre_opt_measurement_ops = pre_opt_counts.get("m_pauli", 0)
        # Analyze pre-optimization PBC circuit to populate pre_opt_* stats
        pre_qasm = circ.to_qasm()
        pre_pbc_qc, pre_basic_stats = pbc_qasm_to_pbm(pre_qasm)
        pre_analysis = analyze_pbc_circuit(
            pre_pbc_qc, pbc_conversion_stats=pre_basic_stats
        )
        # Normalize keys to pre_opt_* namespace
        for k, v in pre_basic_stats.items():
            pre_opt_stats[f"pre_opt_{k}"] = v
        for k, v in pre_analysis.items():
            if k.startswith("pbc_"):
                pre_opt_stats[f"pre_opt_{k[len('pbc_') : ]}"] = v
            else:
                pre_opt_stats[f"pre_opt_{k}"] = v
        if t_opt and fuse_supported:
            circ = nq.fuse_t(circ)
            fuse_applied = True
        elif t_opt and not fuse_supported:
            # Gracefully skip when the installed nwqec lacks fuse_t (renamed from opt_t).
            print("[nwqec] fuse_t not available; skipping T optimization.")
        post_counts = circ.count_ops()
        post_opt_rotation_ops = post_counts.get("t_pauli", 0)
        post_opt_measurement_ops = post_counts.get("m_pauli", 0)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Export to QASM and adapt to PBM circuit
    qasm = circ.to_qasm()
    pbc_qc, stats = pbc_qasm_to_pbm(qasm)
    # Analyze post-optimization PBC circuit to populate pbc_* stats
    post_analysis = analyze_pbc_circuit(pbc_qc, pbc_conversion_stats=stats)
    stats.update(post_analysis)
    stats["pbc_fuse_t_supported"] = fuse_supported
    stats["pbc_fuse_t_applied"] = fuse_applied
    stats["pre_opt_rotation_operators"] = pre_opt_rotation_ops
    stats["pbc_rotation_operators"] = post_opt_rotation_ops
    stats["pre_opt_measurement_operators"] = pre_opt_measurement_ops
    stats["pbc_measurement_operators"] = post_opt_measurement_ops
    stats.update(pre_opt_stats)

    # Normalize Pauli weight keys to match print_circuit_stats expectations
    # (expects *_avg_operator_pauli_weight, *_std_operator_pauli_weight, *_max_operator_pauli_weight)
    def _alias_weight(src_prefix: str, dst_prefix: str, dest: Dict[str, Any]) -> None:
        mappings = [
            ("avg_pauli_weight", "avg_operator_pauli_weight"),
            ("std_pauli_weight", "std_operator_pauli_weight"),
            ("max_pauli_weight", "max_operator_pauli_weight"),
        ]
        for src_suffix, dst_suffix in mappings:
            src_key = f"{src_prefix}{src_suffix}"
            dst_key = f"{dst_prefix}{dst_suffix}"
            if src_key in dest and dst_key not in dest:
                dest[dst_key] = dest[src_key]

    _alias_weight("pbc_", "pbc_", stats)
    _alias_weight("pre_opt_", "pre_opt_", stats)
    return pbc_qc, stats