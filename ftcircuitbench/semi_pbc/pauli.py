from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

PauliPair = Tuple[str, str]

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


def _qubit_sort_key(name: str) -> tuple[int, int]:
    prefix = name[0]
    if prefix not in {"q", "a"}:
        raise ValueError(f"unsupported qubit id {name!r}")
    try:
        idx = int(name[1:])
    except ValueError as exc:
        raise ValueError(f"unsupported qubit id {name!r}") from exc
    return (0 if prefix == "q" else 1, idx)


@dataclass(frozen=True)
class PauliTerm:
    pairs: tuple[PauliPair, ...]
    sign: int = 1
    source_id: str | None = None

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[PauliPair],
        *,
        sign: int = 1,
        source_id: str | None = None,
    ) -> "PauliTerm":
        if sign not in {1, -1}:
            raise ValueError(f"unsupported sign {sign!r}")
        seen: set[str] = set()
        cleaned: list[PauliPair] = []
        for qubit, pauli in pairs:
            if pauli == "I":
                continue
            if pauli not in {"X", "Y", "Z"}:
                raise ValueError(f"unsupported Pauli {pauli!r}")
            if qubit in seen:
                raise ValueError(f"duplicate Pauli entry for qubit {qubit!r}")
            _qubit_sort_key(qubit)
            seen.add(qubit)
            cleaned.append((qubit, pauli))
        return cls(
            tuple(sorted(cleaned, key=lambda item: _qubit_sort_key(item[0]))),
            sign,
            source_id,
        )

    @classmethod
    def from_full_width(
        cls, signed_pauli: str, *, source_id: str | None = None
    ) -> "PauliTerm":
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
            if not qubit.startswith("q"):
                raise ValueError("cannot emit ancilla term in full-width data-only format")
            idx = int(qubit[1:])
            if idx >= data_qubits:
                raise ValueError(f"qubit {qubit} outside width {data_qubits}")
            chars[idx] = pauli
        return ("+" if self.sign == 1 else "-") + "".join(chars)

    def commutes_with(self, other: "PauliTerm") -> bool:
        other_by_qubit = dict(other.pairs)
        count = 0
        for qubit, pauli in self.pairs:
            other_pauli = other_by_qubit.get(qubit)
            if other_pauli and pauli != other_pauli:
                count += 1
        return count % 2 == 0

    def multiply_real(self, other: "PauliTerm") -> "PauliTerm":
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
