# ruff: noqa: TRY004

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ftcircuitbench.semi_pbc.pauli import PauliTerm

_DATA_QUBIT_RE = re.compile(r"q([0-9]+)\Z")
_ANCILLA_QUBIT_RE = re.compile(r"a([0-9]+)\Z")
_CLASSICAL_RE = re.compile(r"(?:c|src)([0-9]+)\Z")
_CLIFFORD_OPS = {"h", "s", "sdg", "cx"}
_OP_FIELDS = {
    "h": {"id", "op", "qubits", "source_id"},
    "s": {"id", "op", "qubits", "source_id"},
    "sdg": {"id", "op", "qubits", "source_id"},
    "cx": {"id", "op", "qubits", "source_id"},
    "alloc": {"id", "op", "qubit", "basis", "source_id"},
    "release": {"id", "op", "qubit", "source_id"},
    "t_pauli": {
        "id",
        "op",
        "terms",
        "sign",
        "angle_num",
        "angle_den",
        "source_id",
    },
    "m_pauli": {"id", "op", "terms", "sign", "result", "source_id"},
    "xor": {"id", "op", "target", "terms", "const", "source_id"},
}


@dataclass(frozen=True)
class SemiPBCHeader:
    k: int
    data_qubits: int
    format: str = "semi-pbc"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be an integer >= 1")
        if type(self.data_qubits) is not int or self.data_qubits < 0:
            raise ValueError("data_qubits must be a non-negative integer")
        if self.format != "semi-pbc":
            raise ValueError(f"unsupported semi-PBC format {self.format!r}")
        if self.version != 1:
            raise ValueError(f"unsupported semi-PBC version {self.version!r}")

    def to_record(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "k": self.k,
            "data_qubits": self.data_qubits,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SemiPBCHeader:
        return cls(
            k=_required_int(record, "k"),
            data_qubits=_required_int(record, "data_qubits"),
            format=_required_str(record, "format"),
            version=_required_int(record, "version"),
        )


@dataclass(frozen=True)
class SemiPBCOp:
    id: int
    op: str
    qubits: tuple[str, ...] = ()
    qubit: str | None = None
    basis: str | None = None
    term: PauliTerm | None = None
    result: str | None = None
    target: str | None = None
    terms: tuple[str, ...] = ()
    const: int = 0
    angle_num: int | None = None
    angle_den: int | None = None
    source_id: str | None = None

    @classmethod
    def clifford(
        cls, id: int, op: str, qubits: Iterable[str], source_id: str | None = None
    ) -> SemiPBCOp:
        return cls(id=id, op=op, qubits=tuple(qubits), source_id=source_id)

    @classmethod
    def alloc(
        cls, id: int, qubit: str, basis: str = "zero", source_id: str | None = None
    ) -> SemiPBCOp:
        return cls(id=id, op="alloc", qubit=qubit, basis=basis, source_id=source_id)

    @classmethod
    def release(cls, id: int, qubit: str, source_id: str | None = None) -> SemiPBCOp:
        return cls(id=id, op="release", qubit=qubit, source_id=source_id)

    @classmethod
    def pauli_rotation(
        cls, id: int, term: PauliTerm, source_id: str | None = None
    ) -> SemiPBCOp:
        return cls(
            id=id,
            op="t_pauli",
            term=term,
            angle_num=1,
            angle_den=8,
            source_id=source_id,
        )

    @classmethod
    def measurement(
        cls,
        id: int,
        term: PauliTerm,
        result: str,
        source_id: str | None = None,
    ) -> SemiPBCOp:
        return cls(id=id, op="m_pauli", term=term, result=result, source_id=source_id)

    @classmethod
    def xor(
        cls,
        id: int,
        target: str,
        terms: Iterable[str],
        const: int = 0,
        source_id: str | None = None,
    ) -> SemiPBCOp:
        return cls(
            id=id,
            op="xor",
            target=target,
            terms=tuple(terms),
            const=const,
            source_id=source_id,
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"id": self.id, "op": self.op}
        if self.op in _CLIFFORD_OPS:
            record["qubits"] = list(self.qubits)
        elif self.op == "alloc":
            record["qubit"] = self.qubit
            record["basis"] = self.basis
        elif self.op == "release":
            record["qubit"] = self.qubit
        elif self.op == "t_pauli":
            if self.term is None:
                raise ValueError("t_pauli operation requires term")
            record["terms"] = [list(pair) for pair in self.term.pairs]
            record["sign"] = self.term.sign
            record["angle_num"] = self.angle_num
            record["angle_den"] = self.angle_den
        elif self.op == "m_pauli":
            if self.term is None:
                raise ValueError("m_pauli operation requires term")
            record["terms"] = [list(pair) for pair in self.term.pairs]
            record["sign"] = self.term.sign
            record["result"] = self.result
        elif self.op == "xor":
            record["target"] = self.target
            record["terms"] = list(self.terms)
            record["const"] = self.const
        else:
            raise ValueError(f"unsupported semi-PBC operation {self.op!r}")
        if self.source_id is not None:
            record["source_id"] = self.source_id
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SemiPBCOp:
        op = _required_str(record, "op")
        _reject_unknown_fields(record, op)
        id = _required_int(record, "id")
        source_id = _optional_str(record, "source_id")
        if op in _CLIFFORD_OPS:
            return cls.clifford(
                id, op, _required_str_list(record, "qubits"), source_id=source_id
            )
        if op == "alloc":
            return cls.alloc(
                id,
                _required_str(record, "qubit"),
                _required_str(record, "basis"),
                source_id=source_id,
            )
        if op == "release":
            return cls.release(id, _required_str(record, "qubit"), source_id=source_id)
        if op == "t_pauli":
            return cls(
                id=id,
                op=op,
                term=_pauli_term_from_record(record),
                angle_num=_required_int(record, "angle_num"),
                angle_den=_required_int(record, "angle_den"),
                source_id=source_id,
            )
        if op == "m_pauli":
            result = _required_str(record, "result")
            return cls.measurement(
                id,
                _pauli_term_from_record(record),
                result=result,
                source_id=source_id,
            )
        if op == "xor":
            return cls.xor(
                id,
                target=_required_str(record, "target"),
                terms=_required_str_list(record, "terms"),
                const=_required_int(record, "const"),
                source_id=source_id,
            )
        raise ValueError(f"unsupported semi-PBC operation {op!r}")

    def validate(self, header: SemiPBCHeader) -> None:
        _validate_op_id(self.id)
        if self.op in {"h", "s", "sdg"}:
            if len(self.qubits) != 1:
                raise ValueError(f"{self.op} requires exactly one qubit")
            _validate_any_qubit(self.qubits[0], header)
            return
        if self.op == "cx":
            if len(self.qubits) != 2:
                raise ValueError("cx requires exactly two qubits")
            for qubit in self.qubits:
                _validate_any_qubit(qubit, header)
            return
        if self.op == "alloc":
            if self.qubit is None:
                raise ValueError("alloc requires one ancilla qubit")
            _validate_ancilla_qubit(self.qubit)
            if self.basis != "zero":
                raise ValueError("alloc basis must be 'zero'")
            return
        if self.op == "release":
            if self.qubit is None:
                raise ValueError("release requires one ancilla qubit")
            _validate_ancilla_qubit(self.qubit)
            return
        if self.op == "t_pauli":
            if self.angle_num != 1 or self.angle_den != 8:
                raise ValueError("t_pauli requires angle_num=1 and angle_den=8")
            _validate_pauli_term(self.term, header, "t_pauli")
            return
        if self.op == "m_pauli":
            if self.result is None:
                raise ValueError("m_pauli requires result")
            _validate_classical_id(self.result, "result")
            _validate_pauli_term(self.term, header, "m_pauli")
            return
        if self.op == "xor":
            if self.target is None:
                raise ValueError("xor requires target")
            _validate_classical_id(self.target, "target")
            for term in self.terms:
                _validate_classical_id(term, "xor term")
            if self.const not in {0, 1}:
                raise ValueError("xor const must be 0 or 1")
            return
        raise ValueError(f"unsupported semi-PBC operation {self.op!r}")


def write_jsonl(
    path: str | Path, header: SemiPBCHeader, ops: Iterable[SemiPBCOp]
) -> None:
    output_path = Path(path)
    last_id: int | None = None
    with output_path.open("w") as output:
        output.write(_json_line(header.to_record()))
        for op in ops:
            op.validate(header)
            if last_id is not None and op.id <= last_id:
                raise ValueError(
                    "operation ids must be strictly monotonically increasing"
                )
            last_id = op.id
            output.write(_json_line(op.to_record()))


def read_jsonl(path: str | Path) -> tuple[SemiPBCHeader, list[SemiPBCOp]]:
    input_path = Path(path)
    lines = input_path.read_text().splitlines()
    if not lines:
        raise ValueError("semi-PBC JSONL file is empty")
    header = SemiPBCHeader.from_record(_json_record(lines[0], 1))
    ops: list[SemiPBCOp] = []
    last_id: int | None = None
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        op = SemiPBCOp.from_record(_json_record(line, line_number))
        op.validate(header)
        if last_id is not None and op.id <= last_id:
            raise ValueError("operation ids must be strictly monotonically increasing")
        last_id = op.id
        ops.append(op)
    return header, ops


def _json_record(line: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON on line {line_number}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"expected JSON object on line {line_number}")
    return record


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":")) + "\n"


def _reject_unknown_fields(record: dict[str, Any], op: str) -> None:
    allowed = _OP_FIELDS.get(op)
    if allowed is None:
        return
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"unknown field(s) for {op}: {', '.join(unknown)}")


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _required_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_str(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_str_list(record: dict[str, Any], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


def _pauli_term_from_record(record: dict[str, Any]) -> PauliTerm:
    raw_pairs = record.get("terms")
    if not isinstance(raw_pairs, list):
        raise ValueError("terms must be a list of Pauli pairs")
    pairs: list[tuple[str, str]] = []
    for pair in raw_pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
        ):
            raise ValueError("terms must contain [qubit, pauli] pairs")
        pairs.append((pair[0], pair[1]))
    return PauliTerm.from_pairs(pairs, sign=_required_int(record, "sign"))


def _validate_op_id(op_id: int) -> None:
    if type(op_id) is not int or op_id < 0:
        raise ValueError("operation id must be a non-negative integer")


def _validate_any_qubit(qubit: str, header: SemiPBCHeader) -> None:
    if _DATA_QUBIT_RE.fullmatch(qubit):
        _validate_data_qubit(qubit, header)
        return
    if _ANCILLA_QUBIT_RE.fullmatch(qubit):
        return
    raise ValueError(f"unsupported qubit id {qubit!r}")


def _validate_data_qubit(qubit: str, header: SemiPBCHeader) -> None:
    match = _DATA_QUBIT_RE.fullmatch(qubit)
    if match is None:
        raise ValueError(f"expected data qubit id q<N>, got {qubit!r}")
    idx = int(match.group(1))
    if idx >= header.data_qubits:
        raise ValueError(
            f"data qubit {qubit} outside header width {header.data_qubits}"
        )


def _validate_ancilla_qubit(qubit: str) -> None:
    if _ANCILLA_QUBIT_RE.fullmatch(qubit) is None:
        raise ValueError(f"expected ancilla qubit id a<N>, got {qubit!r}")


def _validate_classical_id(classical_id: str, field: str) -> None:
    if _CLASSICAL_RE.fullmatch(classical_id) is None:
        raise ValueError(f"{field} must be a classical id c<N> or src<N>")


def _validate_pauli_term(
    term: PauliTerm | None, header: SemiPBCHeader, op_name: str
) -> None:
    if term is None:
        raise ValueError(f"{op_name} requires term")
    if term.sign not in {1, -1}:
        raise ValueError(f"{op_name} sign must be 1 or -1")
    if term.weight < 1:
        raise ValueError(f"{op_name} Pauli term weight must be at least 1")
    if term.weight > header.k:
        raise ValueError(
            f"{op_name} Pauli term weight {term.weight} exceeds k={header.k}"
        )
    for qubit, _pauli in term.pairs:
        _validate_any_qubit(qubit, header)
