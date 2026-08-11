from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

PauliPair = tuple[str, str]
_QUBIT_ID_RE = re.compile(r"([qa])([0-9]+)\Z")
_PAULI_LABELS = {"X", "Y", "Z"}

_SINGLE_PRODUCT = {
    ("X", "X"): (None, 0),
    ("Y", "Y"): (None, 0),
    ("Z", "Z"): (None, 0),
    ("X", "Y"): ("Z", 1),
    ("Y", "Z"): ("X", 1),
    ("Z", "X"): ("Y", 1),
    ("Y", "X"): ("Z", 3),
    ("Z", "Y"): ("X", 3),
    ("X", "Z"): ("Y", 3),
}


def _parse_qubit_id(name: str) -> tuple[str, int]:
    try:
        match = _QUBIT_ID_RE.fullmatch(name)
    except TypeError as exc:
        raise ValueError(f"unsupported qubit id {name!r}") from exc
    if match is None:
        raise ValueError(f"unsupported qubit id {name!r}")
    prefix, suffix = match.groups()
    if len(suffix) > 1 and suffix.startswith("0"):
        raise ValueError(f"qubit id {name!r} has leading zero")
    return prefix, int(suffix)


def _qubit_sort_key(name: str) -> tuple[int, int]:
    prefix, idx = _parse_qubit_id(name)
    return (0 if prefix == "q" else 1, idx)


def _normalize_pairs(
    pairs: Iterable[PauliPair], *, sign: int, source_id: str | None
) -> tuple[PauliPair, ...]:
    if type(sign) is not int or sign not in {1, -1}:
        raise ValueError(f"unsupported sign {sign!r}")
    if source_id is not None and not isinstance(source_id, str):
        raise ValueError("source_id must be a string")
    if not isinstance(pairs, Iterable) or isinstance(pairs, (str, bytes)):
        raise ValueError(  # noqa: TRY004
            "Pauli terms must be an iterable of Pauli pairs"
        )

    seen: set[tuple[str, int]] = set()
    cleaned: list[PauliPair] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Pauli terms must contain 2-item Pauli pairs")
        qubit, pauli = pair
        if not isinstance(qubit, str):
            raise ValueError("qubit id must be a string")  # noqa: TRY004
        if not isinstance(pauli, str):
            raise ValueError("Pauli label must be a string")  # noqa: TRY004
        if pauli == "I":
            continue
        if pauli not in _PAULI_LABELS:
            raise ValueError(f"unsupported Pauli {pauli!r}")
        qubit_key = _parse_qubit_id(qubit)
        if qubit_key in seen:
            raise ValueError(f"duplicate Pauli entry for qubit {qubit!r}")
        seen.add(qubit_key)
        cleaned.append((qubit, pauli))
    return tuple(sorted(cleaned, key=lambda item: _qubit_sort_key(item[0])))


@dataclass(frozen=True)
class PauliTerm:
    pairs: tuple[PauliPair, ...]
    sign: int = 1
    source_id: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pairs",
            _normalize_pairs(self.pairs, sign=self.sign, source_id=self.source_id),
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[PauliPair],
        *,
        sign: int = 1,
        source_id: str | None = None,
    ) -> PauliTerm:
        return cls(pairs, sign, source_id)

    @classmethod
    def from_full_width(
        cls, signed_pauli: str, *, source_id: str | None = None
    ) -> PauliTerm:
        if not signed_pauli or signed_pauli[0] not in "+-":
            raise ValueError(f"expected signed Pauli string, got {signed_pauli!r}")
        sign = 1 if signed_pauli[0] == "+" else -1
        body = signed_pauli[1:]
        return cls.from_pairs(
            ((f"q{i}", p) for i, p in enumerate(body)),
            sign=sign,
            source_id=source_id,
        )

    @property
    def weight(self) -> int:
        return len(self.pairs)

    def to_full_width(self, data_qubits: int) -> str:
        chars = ["I"] * data_qubits
        for qubit, pauli in self.pairs:
            prefix, idx = _parse_qubit_id(qubit)
            if prefix != "q":
                raise ValueError(
                    "cannot emit ancilla term in full-width data-only format"
                )
            if idx >= data_qubits:
                raise ValueError(f"qubit {qubit} outside width {data_qubits}")
            chars[idx] = pauli
        return ("+" if self.sign == 1 else "-") + "".join(chars)

    def commutes_with(self, other: PauliTerm) -> bool:
        other_by_qubit = dict(other.pairs)
        count = 0
        for qubit, pauli in self.pairs:
            other_pauli = other_by_qubit.get(qubit)
            if other_pauli and pauli != other_pauli:
                count += 1
        return count % 2 == 0

    def multiply_real(self, other: PauliTerm) -> PauliTerm:
        merged: dict[str, str] = dict(self.pairs)
        sign = self.sign * other.sign
        phase = 0
        for qubit, pauli in other.pairs:
            current = merged.get(qubit)
            if current is None:
                merged[qubit] = pauli
            else:
                product, phase_delta = _SINGLE_PRODUCT[(current, pauli)]
                phase = (phase + phase_delta) % 4
                if product is None:
                    del merged[qubit]
                else:
                    merged[qubit] = product
        if phase % 2:
            raise ValueError("Pauli product has imaginary global phase")
        if phase == 2:
            sign *= -1
        return PauliTerm.from_pairs(merged.items(), sign=sign)
