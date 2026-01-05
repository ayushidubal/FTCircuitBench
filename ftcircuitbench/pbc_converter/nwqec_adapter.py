"""
Adapter to convert nwqec PBC QASM into FTCircuitBench PBM QuantumCircuit.

Parses lines like:
  t_pauli +XIZ;
  s_pauli -XYZI;
  z_pauli +ZII;
  m_pauli -ZII;

and emits PBM-style gates that FTCircuitBench analyzers expect:
  R<activePauli>(angle) on active qubits (non-I positions),
  Meas<sign><activePauli> on active qubits (sign preserved).

Angle mapping per user spec:
- t_pauli -> ±pi/8
- s_pauli -> ±pi/4
- z_pauli -> ±pi/2
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from ftcircuitbench.pbc_converter.pbm import PBM, Rotation

_QREG_RE = re.compile(r"^qreg\s+([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\];")
_T_RE = re.compile(r"^t_pauli\s+([+-][IXYZ]+)\s*;?")
_S_RE = re.compile(r"^s_pauli\s+([+-][IXYZ]+)\s*;?")
_Z_RE = re.compile(r"^z_pauli\s+([+-][IXYZ]+)\s*;?")
_M_RE = re.compile(r"^m_pauli\s+([+-][IXYZ]+)\s*;?")


def _extract_num_qubits_from_qasm(qasm: str) -> int:
    for line in qasm.splitlines():
        m = _QREG_RE.match(line.strip())
        if m:
            return int(m.group(2))
    # Fallback: try to infer from longest pauli string encountered
    max_len = 0
    for line in qasm.splitlines():
        for pat in (_T_RE, _S_RE, _Z_RE, _M_RE):
            m = pat.match(line.strip())
            if m:
                max_len = max(max_len, len(m.group(1)) - 1)  # minus sign
    if max_len > 0:
        return max_len
    raise ValueError("Could not determine number of qubits from QASM")


def _active_qubits_and_pauli(pauli_with_sign: str) -> Tuple[list[int], str, str]:
    """Return (active_indices, active_pauli_str, sign) from ±[IXYZ]+ string."""
    sign = pauli_with_sign[0]
    pauli = pauli_with_sign[1:]
    active_indices = [i for i, ch in enumerate(pauli) if ch != "I"]
    active_pauli = "".join(ch for ch in pauli if ch != "I")
    return active_indices, active_pauli, sign


def _angle_for(op: str, sign: str) -> str:
    if op == "t":
        return Rotation.PI_8.value if sign == "+" else Rotation.PI_m8.value
    if op == "s":
        return "pi/4" if sign == "+" else "-pi/4"
    if op == "z":
        return "pi/2" if sign == "+" else "-pi/2"
    raise ValueError(f"Unknown op kind for angle mapping: {op}")


def pbc_qasm_to_pbm(qasm: str) -> Tuple[QuantumCircuit, Dict]:
    """
    Convert nwqec PBC QASM text into a PBM QuantumCircuit and basic stats.

    Returns:
        (pbc_qc, stats)
    """
    num_qubits = _extract_num_qubits_from_qasm(qasm)
    qreg = QuantumRegister(num_qubits, "q")
    pbc_qc = QuantumCircuit(qreg)

    rotations = 0
    measurements = 0

    for raw in qasm.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        m = _T_RE.match(line)
        if m:
            idxs, active_pauli, sign = _active_qubits_and_pauli(m.group(1))
            if active_pauli:
                qargs = [qreg[i] for i in idxs]
                angle = _angle_for("t", sign)
                pbc_qc.append(PBM.generate_gate(active_pauli, angle), qargs)
                rotations += 1
            continue

        m = _S_RE.match(line)
        if m:
            raise RuntimeError(
                f"Encountered s_pauli in nwqec PBC output; unsupported for now: '{line}'"
            )

        m = _Z_RE.match(line)
        if m:
            raise RuntimeError(
                f"Encountered z_pauli in nwqec PBC output; unsupported for now: '{line}'"
            )

        m = _M_RE.match(line)
        if m:
            idxs, active_pauli, sign = _active_qubits_and_pauli(m.group(1))
            if active_pauli:
                qargs = [qreg[i] for i in idxs]
                # Preserve sign in measurement gate name: Meas+XYZ / Meas-XYZ
                pbc_qc.append(PBM.generate_measure(sign + active_pauli), qargs)
                measurements += 1
            continue

        # Ignore other QASM lines (includes qreg declaration, creg, etc.)

    stats: Dict = {
        "pbc_t_operators": rotations,  # rotation count (t/s/z combined)
        "pbc_measurement_operators": measurements,
    }
    return pbc_qc, stats
