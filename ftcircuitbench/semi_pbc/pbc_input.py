from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ftcircuitbench.semi_pbc.pauli import PauliTerm

_PBC_OP_RE = re.compile(r"\A(?P<op>t_pauli|m_pauli)\s+(?P<pauli>[+-][IXYZ]+)\s*;?\Z")
_QREG_RE = re.compile(r"\Aqreg\s+q\[(?P<data_qubits>0|[1-9][0-9]*)\]\s*;\Z")
_BOILERPLATE_RE = re.compile(
    r'\A(?:OPENQASM\s+[0-9]+(?:\.[0-9]+)?|include\s+"[^"]+"|'
    r"creg\s+[A-Za-z_][A-Za-z0-9_]*\[(?:0|[1-9][0-9]*)\])\s*;\Z"
)
_PBC_OP_START_RE = re.compile(r"\A(?P<op>t_pauli|m_pauli)\b")
_QREG_START_RE = re.compile(r"\Aqreg\b")


@dataclass(frozen=True)
class SourcePBCOp:
    id: int
    op: str
    term: PauliTerm
    source_id: str


@dataclass(frozen=True)
class PBCProgram:
    data_qubits: int
    ops: tuple[SourcePBCOp, ...]


def parse_pbc_text(text: str) -> PBCProgram:
    data_qubits: int | None = None
    ops: list[SourcePBCOp] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line)
        if not line or _is_skipped_statement(line):
            continue

        qreg_match = _QREG_RE.fullmatch(line)
        if qreg_match is not None:
            qreg_width = int(qreg_match.group("data_qubits"))
            if data_qubits is not None and qreg_width != data_qubits:
                raise ValueError(
                    f"line {line_number}: qreg width {qreg_width} conflicts with "
                    f"Pauli string length {data_qubits}"
                )
            data_qubits = qreg_width
            continue
        if _QREG_START_RE.match(line):
            raise ValueError(f"line {line_number}: unsupported qreg declaration")

        op_match = _PBC_OP_RE.fullmatch(line)
        if op_match is not None:
            signed_pauli = op_match.group("pauli")
            pauli_width = len(signed_pauli) - 1
            if data_qubits is None:
                data_qubits = pauli_width
            elif pauli_width != data_qubits:
                raise ValueError(
                    f"line {line_number}: Pauli string length {pauli_width} does "
                    f"not match data qubit length {data_qubits}"
                )
            source_id = f"line{line_number}"
            term = PauliTerm.from_full_width(signed_pauli, source_id=source_id)
            if term.weight < 1:
                raise ValueError(
                    f"line {line_number}: identity Pauli operation has weight 0"
                )
            ops.append(
                SourcePBCOp(
                    id=len(ops),
                    op=op_match.group("op"),
                    term=term,
                    source_id=source_id,
                )
            )
            continue
        if _PBC_OP_START_RE.match(line):
            raise ValueError(f"line {line_number}: unsupported PBC Pauli operation")

        op_name = line.split(maxsplit=1)[0].rstrip(";")
        raise ValueError(f"line {line_number}: unsupported PBC operator {op_name!r}")

    if data_qubits is None:
        raise ValueError("PBC input must contain qreg q[N] or a Pauli operation")
    return PBCProgram(data_qubits=data_qubits, ops=tuple(ops))


def parse_pbc_file(path: str | Path) -> PBCProgram:
    return parse_pbc_text(Path(path).read_text(encoding="utf-8"))


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _is_skipped_statement(line: str) -> bool:
    return _BOILERPLATE_RE.fullmatch(line) is not None
