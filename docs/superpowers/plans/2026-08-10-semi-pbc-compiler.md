# Semi-PBC Compiler Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 capped-weight PBC -> semi-PBC generation for existing `nwqec` PBC files.

**Architecture:** Add a focused `ftcircuitbench.semi_pbc` package with small modules for Pauli algebra, JSONL IR, PBC parsing, exact baseline lowering, guarded Peres/Galvao reduction, scheduling/reporting, and orchestration. Keep CLI behavior in a thin script that calls the package API. Direct C+T compilation and `dascot-rs` semi-PBC routing are out of scope for this plan.

**Tech Stack:** Python 3, dataclasses, stdlib `json`/`argparse`, pytest, existing repository scripts/tests.

**Spec:** `docs/superpowers/specs/2026-08-10-semi-pbc-compiler-design.md`

---

## File Structure

- Create `ftcircuitbench/semi_pbc/__init__.py`
  - Public exports for Phase 1 APIs.
- Create `ftcircuitbench/semi_pbc/pauli.py`
  - Sparse Pauli term representation, canonical ordering, multiplication, commutation, full-width PBC conversion.
- Create `ftcircuitbench/semi_pbc/ir.py`
  - Semi-PBC JSONL header/op dataclasses, validation, JSONL reader/writer.
- Create `ftcircuitbench/semi_pbc/pbc_input.py`
  - Parser for existing `nwqec` PBC text files.
- Create `ftcircuitbench/semi_pbc/lowering.py`
  - Basis-change helpers and exact Phase 1 max-`k` parity-network lowering for high-weight rotations/measurements.
- Create `ftcircuitbench/semi_pbc/reducer.py`
  - Guarded Peres/Galvao measurement representative reducer.
- Create `ftcircuitbench/semi_pbc/schedule.py`
  - Conservative sequential scheduling/reporting and default latency-depth summary.
- Create `ftcircuitbench/semi_pbc/pipeline.py`
  - End-to-end compile function and summary object.
- Create `compile_semi_pbc.py`
  - CLI wrapper.
- Create tests:
  - `tests/test_semi_pbc_pauli.py`
  - `tests/test_semi_pbc_ir.py`
  - `tests/test_semi_pbc_pbc_input.py`
  - `tests/test_semi_pbc_lowering.py`
  - `tests/test_semi_pbc_reducer.py`
  - `tests/test_semi_pbc_pipeline.py`
  - `tests/test_compile_semi_pbc_cli.py`

## Chunk 1: Core Pauli Algebra, IR, and PBC Input

### Task 1: Sparse Pauli Algebra

**Files:**
- Create: `ftcircuitbench/semi_pbc/__init__.py`
- Create: `ftcircuitbench/semi_pbc/pauli.py`
- Test: `tests/test_semi_pbc_pauli.py`

- [ ] **Step 1: Write failing tests for canonical sparse terms**

Create `tests/test_semi_pbc_pauli.py` with tests that assert:

```python
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_pauli_term_canonicalizes_data_then_ancilla():
    term = PauliTerm.from_pairs([("a2", "X"), ("q10", "Z"), ("q2", "Y"), ("a0", "Z")])
    assert term.pairs == (("q2", "Y"), ("q10", "Z"), ("a0", "Z"), ("a2", "X"))
    assert term.weight == 4


def test_pauli_term_rejects_duplicate_non_identity_qubit():
    with pytest.raises(ValueError, match="duplicate"):
        PauliTerm.from_pairs([("q0", "X"), ("q0", "Z")])


def test_full_width_conversion_omits_identity_in_sparse_form():
    term = PauliTerm.from_full_width("+IXYZ", source_id="line7")
    assert term.sign == 1
    assert term.pairs == (("q1", "X"), ("q2", "Y"), ("q3", "Z"))
    assert term.to_full_width(4) == "+IXYZ"
```

Include `import pytest`.

- [ ] **Step 2: Run the failing tests**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ftcircuitbench.semi_pbc'`.

- [ ] **Step 3: Implement `PauliTerm` and canonical ordering**

Create `ftcircuitbench/semi_pbc/__init__.py` with minimal exports.

Create `ftcircuitbench/semi_pbc/pauli.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

PauliPair = Tuple[str, str]


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
        return cls(tuple(sorted(cleaned, key=lambda item: _qubit_sort_key(item[0]))), sign, source_id)

    @classmethod
    def from_full_width(cls, signed_pauli: str, *, source_id: str | None = None) -> "PauliTerm":
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
```

- [ ] **Step 4: Run tests until they pass**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py -v`

Expected: PASS.

- [ ] **Step 5: Add Pauli multiplication and commutation tests**

Extend `tests/test_semi_pbc_pauli.py`:

```python
def test_pauli_multiplication_tracks_real_signed_result():
    left = PauliTerm.from_pairs([("q0", "X"), ("q1", "Z")])
    right = PauliTerm.from_pairs([("q0", "X"), ("q2", "Y")], sign=-1)
    product = left.multiply_real(right)
    assert product.sign == -1
    assert product.pairs == (("q1", "Z"), ("q2", "Y"))


def test_pauli_multiplication_rejects_imaginary_phase():
    left = PauliTerm.from_pairs([("q0", "X")])
    right = PauliTerm.from_pairs([("q0", "Y")])
    with pytest.raises(ValueError, match="imaginary"):
        left.multiply_real(right)


def test_pauli_multiplication_allows_real_negative_phase():
    left = PauliTerm.from_pairs([("q0", "X"), ("q1", "Y")])
    right = PauliTerm.from_pairs([("q0", "Y"), ("q1", "Z")])
    product = left.multiply_real(right)
    assert product.sign == -1
    assert product.pairs == (("q0", "Z"), ("q1", "X"))


def test_commutation_uses_anticommutation_parity():
    assert PauliTerm.from_pairs([("q0", "X")]).commutes_with(PauliTerm.from_pairs([("q0", "Y")])) is False
    assert PauliTerm.from_pairs([("q0", "X"), ("q1", "Z")]).commutes_with(
        PauliTerm.from_pairs([("q0", "Y"), ("q1", "X")])
    ) is True
```

- [ ] **Step 6: Run new tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py -v`

Expected: FAIL with `AttributeError` for missing `multiply_real` / `commutes_with`.

- [ ] **Step 7: Implement multiplication and commutation**

In `ftcircuitbench/semi_pbc/pauli.py`, add a signed single-qubit multiplication table:

```python
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
```

Then add methods to `PauliTerm`:

```python
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
```

This supports real signed products and rejects imaginary products, matching the Phase 1 reducer rule.

- [ ] **Step 8: Run tests**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ftcircuitbench/semi_pbc/__init__.py ftcircuitbench/semi_pbc/pauli.py tests/test_semi_pbc_pauli.py
git commit -m "feat: add semi-PBC Pauli algebra"
```

### Task 2: JSONL IR Schema

**Files:**
- Modify: `ftcircuitbench/semi_pbc/__init__.py`
- Create: `ftcircuitbench/semi_pbc/ir.py`
- Test: `tests/test_semi_pbc_ir.py`

- [ ] **Step 1: Write failing IR round-trip and validation tests**

Create `tests/test_semi_pbc_ir.py`:

```python
import json

import pytest

from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp, read_jsonl, write_jsonl
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_jsonl_round_trip_canonical_terms(tmp_path):
    path = tmp_path / "toy.semi_pbc.jsonl"
    header = SemiPBCHeader(k=2, data_qubits=3)
    ops = [
        SemiPBCOp.pauli_rotation(0, PauliTerm.from_pairs([("q2", "Z"), ("q0", "X")]), source_id="line1"),
        SemiPBCOp.measurement(1, PauliTerm.from_pairs([("q1", "Z")], sign=-1), result="c0", source_id="line2"),
        SemiPBCOp.xor(2, target="src1", terms=["c0"], const=1),
    ]
    write_jsonl(path, header, ops)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {"format": "semi-pbc", "version": 1, "k": 2, "data_qubits": 3}
    assert rows[1]["terms"] == [["q0", "X"], ["q2", "Z"]]
    loaded_header, loaded_ops = read_jsonl(path)
    assert loaded_header == header
    assert loaded_ops == ops


def test_write_jsonl_rejects_non_monotonic_ids(tmp_path):
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.clifford(1, "h", ("q0",)),
        SemiPBCOp.clifford(1, "s", ("q0",)),
    ]
    with pytest.raises(ValueError, match="monotonic"):
        write_jsonl(tmp_path / "bad.jsonl", header, ops)


def test_read_jsonl_rejects_non_monotonic_ids(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":1,"op":"h","qubits":["q0"]}\n'
        '{"id":1,"op":"s","qubits":["q0"]}\n'
    )
    with pytest.raises(ValueError, match="monotonic"):
        read_jsonl(path)


def test_read_jsonl_rejects_malformed_operation_record(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"format":"semi-pbc","version":1,"k":1,"data_qubits":1}\n'
        '{"id":0,"op":"m_pauli","terms":[["q0","Z"]]}\n'
    )
    with pytest.raises(ValueError, match="result"):
        read_jsonl(path)


def test_header_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        SemiPBCHeader(k=0, data_qubits=3)


def test_pauli_op_rejects_weight_above_k_when_validated():
    header = SemiPBCHeader(k=1, data_qubits=2)
    op = SemiPBCOp.pauli_rotation(0, PauliTerm.from_pairs([("q0", "Z"), ("q1", "Z")]))
    with pytest.raises(ValueError, match="weight"):
        op.validate(header)


def test_pauli_op_rejects_data_qubit_outside_header_width():
    header = SemiPBCHeader(k=1, data_qubits=1)
    op = SemiPBCOp.pauli_rotation(0, PauliTerm.from_pairs([("q1", "Z")]))
    with pytest.raises(ValueError, match="outside"):
        op.validate(header)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_ir.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing `ir`.

- [ ] **Step 3: Implement `SemiPBCHeader`, `SemiPBCOp`, JSONL IO**

Create `ftcircuitbench/semi_pbc/ir.py` with:

- `@dataclass(frozen=True) SemiPBCHeader(k: int, data_qubits: int, format: str = "semi-pbc", version: int = 1)`
- `@dataclass(frozen=True) SemiPBCOp(...)` with concrete fields:
  - `id: int`
  - `op: str`
  - `qubits: tuple[str, ...] = ()`
  - `qubit: str | None = None`
  - `basis: str | None = None`
  - `term: PauliTerm | None = None`
  - `result: str | None = None`
  - `target: str | None = None`
  - `terms: tuple[str, ...] = ()`
  - `const: int = 0`
  - `angle_num: int | None = None`
  - `angle_den: int | None = None`
  - `source_id: str | None = None`
- constructors:
  - `clifford(id, op, qubits)`
  - `alloc(id, qubit, basis="zero")`
  - `release(id, qubit)`
  - `pauli_rotation(id, term, source_id=None)`
  - `measurement(id, term, result, source_id=None)`
  - `xor(id, target, terms, const=0)`
- `to_record()` and `from_record()`
- `validate(header)`
- `write_jsonl(path, header, ops)`
- `read_jsonl(path)`

Validation must enforce:

- header `k >= 1`
- operation IDs are non-negative integers
- `write_jsonl` and `read_jsonl` validate operation IDs are strictly monotonically increasing
- `read_jsonl` calls `SemiPBCOp.from_record(...).validate(header)` for every operation record
- qubit IDs match `q<N>` or `a<N>` as appropriate
- data qubit IDs must satisfy `0 <= N < header.data_qubits`
- classical IDs match `c<N>` or `src<N>`
- `h`, `s`, and `sdg` have exactly one qubit
- `cx` has exactly two qubits
- `alloc` has one ancilla qubit and `basis == "zero"`
- `release` has one ancilla qubit
- `t_pauli` has `angle_num=1`, `angle_den=8`, `sign in {1,-1}`
- `m_pauli` has `sign in {1,-1}` and a result
- emitted Pauli ops have weight at least 1
- Pauli op weight `<= header.k` when `validate(header)` is called
- `xor.const in {0,1}`
- supported Clifford ops are `h`, `s`, `sdg`, `cx`

- [ ] **Step 4: Export IR API**

In `ftcircuitbench/semi_pbc/__init__.py`, export:

```python
from .ir import SemiPBCHeader, SemiPBCOp, read_jsonl, write_jsonl
from .pauli import PauliTerm
```

- [ ] **Step 5: Run IR tests**

Run: `python3 -m pytest tests/test_semi_pbc_ir.py -v`

Expected: PASS.

- [ ] **Step 6: Run Pauli tests again**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py tests/test_semi_pbc_ir.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ftcircuitbench/semi_pbc/__init__.py ftcircuitbench/semi_pbc/ir.py tests/test_semi_pbc_ir.py
git commit -m "feat: add semi-PBC JSONL IR"
```

### Task 3: Existing PBC Input Parser

**Files:**
- Create: `ftcircuitbench/semi_pbc/pbc_input.py`
- Test: `tests/test_semi_pbc_pbc_input.py`

- [ ] **Step 1: Write failing PBC parser tests**

Create `tests/test_semi_pbc_pbc_input.py`:

```python
import pytest

from ftcircuitbench.semi_pbc.pbc_input import parse_pbc_text


def test_parse_nwqec_pbc_text_to_source_ops():
    text = '''
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    t_pauli +IXYZ;
    m_pauli -ZZII;
    '''
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
```

The parser must not return `SemiPBCHeader`, because PBC input does not contain `k`. `k` is a compile option used later when creating the semi-PBC header.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_pbc_input.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement parser**

Create `ftcircuitbench/semi_pbc/pbc_input.py` with:

- `_PBC_OP_RE` recognizing `t_pauli` and `m_pauli` lines with signed full-width Pauli strings
- `_QREG_RE`
- `@dataclass(frozen=True) SourcePBCOp(id, op, term, source_id)`
- `@dataclass(frozen=True) PBCProgram(data_qubits, ops)`
- `parse_pbc_text(text: str) -> PBCProgram`
- `parse_pbc_file(path: str | Path) -> PBCProgram`

Parser behavior:

- skip blank lines, comments, `OPENQASM`, `include`, `creg`
- use `qreg q[N];` if present
- infer width from first Pauli if no `qreg` exists
- reject unsupported signs, unsupported operators, and length mismatch
- assign source IDs as physical file line numbers, e.g. `line5`

- [ ] **Step 3a: Add parser edge-case tests**

Extend `tests/test_semi_pbc_pbc_input.py` with tests for:

```python
def test_parse_infers_width_without_qreg_and_skips_comments():
    program = parse_pbc_text("// hi\n\nm_pauli +ZI;\n")
    assert program.data_qubits == 2
    assert program.ops[0].source_id == "line3"


def test_parse_rejects_malformed_sign():
    with pytest.raises(ValueError, match="unsupported"):
        parse_pbc_text("qreg q[1];\nt_pauli *Z;\n")
```

- [ ] **Step 4: Run parser tests**

Run: `python3 -m pytest tests/test_semi_pbc_pbc_input.py -v`

Expected: PASS.

- [ ] **Step 5: Run chunk tests**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py tests/test_semi_pbc_ir.py tests/test_semi_pbc_pbc_input.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ftcircuitbench/semi_pbc/pbc_input.py tests/test_semi_pbc_pbc_input.py
git commit -m "feat: parse PBC input for semi-PBC compiler"
```

## Chunk 2: Exact Baseline Lowering

### Task 4: Rotation and Measurement Lowering

**Files:**
- Create: `ftcircuitbench/semi_pbc/lowering.py`
- Test: `tests/test_semi_pbc_lowering.py`

- [ ] **Step 1: Write failing tests for basis changes and rotation lowering**

Create `tests/test_semi_pbc_lowering.py`:

```python
from ftcircuitbench.semi_pbc.lowering import lower_pauli_rotation
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_rotation_lowering_uses_basis_changes_and_max_k_rotation():
    term = PauliTerm.from_pairs([("q0", "X"), ("q1", "Y"), ("q2", "Z"), ("q3", "X")])
    ops = lower_pauli_rotation(start_id=0, term=term, k=2, source_id="line4")
    assert [op.op for op in ops] == ["h", "sdg", "h", "h", "cx", "cx", "t_pauli", "cx", "cx", "h", "h", "s", "h"]
    assert [op.qubits for op in ops if op.op == "cx"] == [("q2", "q0"), ("q3", "q0"), ("q3", "q0"), ("q2", "q0")]
    t_ops = [op for op in ops if op.op == "t_pauli"]
    assert len(t_ops) == 1
    assert t_ops[0].term.pairs == (("q0", "Z"), ("q1", "Z"))
    assert t_ops[0].source_id == "line4"
    assert max((op.term.weight for op in ops if op.op in {"t_pauli", "m_pauli"}), default=0) == 2


def test_rotation_passes_through_when_weight_within_k():
    term = PauliTerm.from_pairs([("q0", "Z"), ("q2", "Z")], sign=-1)
    ops = lower_pauli_rotation(start_id=10, term=term, k=2, source_id="line8")
    assert len(ops) == 1
    assert ops[0].op == "t_pauli"
    assert ops[0].id == 10
    assert ops[0].term == term


def test_lowering_rejects_invalid_k():
    term = PauliTerm.from_pairs([("q0", "Z")])
    with pytest.raises(ValueError, match="k"):
        lower_pauli_rotation(start_id=0, term=term, k=0)
```

Include `import pytest`.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement rotation lowering**

Create `ftcircuitbench/semi_pbc/lowering.py` with:

- `_basis_change_ops(term)`
- `_inverse_basis_change_ops(term)`
- `lower_pauli_rotation(start_id, term, k, source_id=None) -> list[SemiPBCOp]`

Implementation details:

- reject `k < 1`
- if `term.weight <= k`, emit one `t_pauli`
- else:
  - basis changes in canonical term order
  - keep the first `k` active qubits as the retained legal block
  - CNOT every remaining active qubit into the first retained qubit
  - one weight-`k` `t_pauli` on the retained block with original sign
  - reverse CNOT chain
  - inverse basis changes in reverse basis order
- use sequential IDs starting at `start_id`

- [ ] **Step 4: Run rotation tests**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: PASS for the rotation tests.

- [ ] **Step 5: Add failing measurement lowering tests**

Extend `tests/test_semi_pbc_lowering.py`:

```python
from ftcircuitbench.semi_pbc.lowering import lower_pauli_measurement


def test_measurement_lowering_uses_max_k_data_measurement_and_xor():
    term = PauliTerm.from_pairs([("q0", "X"), ("q1", "Z"), ("q2", "Z"), ("q3", "Z")], sign=-1)
    lowered = lower_pauli_measurement(
        start_id=0,
        term=term,
        k=2,
        result="src4",
        source_id="line4",
        next_ancilla=0,
        next_classical=0,
    )
    ops = lowered.ops
    assert lowered.next_ancilla == 0
    assert lowered.next_classical == 1
    assert [op.op for op in ops] == ["h", "cx", "cx", "m_pauli", "cx", "cx", "h", "xor"]
    assert [op.qubits for op in ops if op.op == "cx"] == [("q2", "q0"), ("q3", "q0"), ("q3", "q0"), ("q2", "q0")]
    measure = [op for op in ops if op.op == "m_pauli"][0]
    assert measure.result == "c0"
    assert measure.source_id == "line4"
    assert measure.term.sign == 1
    assert measure.term.pairs == (("q0", "Z"), ("q1", "Z"))
    xor = ops[-1]
    assert xor.target == "src4"
    assert xor.terms == ("c0",)
    assert xor.const == 1


def test_measurement_passes_through_when_weight_within_k():
    term = PauliTerm.from_pairs([("q1", "Z")])
    lowered = lower_pauli_measurement(
        start_id=3,
        term=term,
        k=1,
        result="src2",
        source_id="line2",
        next_ancilla=0,
        next_classical=0,
    )
    assert [op.op for op in lowered.ops] == ["m_pauli", "xor"]
    assert lowered.ops[0].result == "c0"
    assert lowered.ops[1].target == "src2"
    assert lowered.ops[1].terms == ("c0",)
    assert lowered.ops[1].const == 0


def test_negative_low_weight_measurement_normalizes_physical_sign():
    term = PauliTerm.from_pairs([("q1", "Z")], sign=-1)
    lowered = lower_pauli_measurement(
        start_id=0,
        term=term,
        k=1,
        result="src9",
        source_id="line9",
        next_ancilla=0,
        next_classical=0,
    )
    assert [op.op for op in lowered.ops] == ["m_pauli", "xor"]
    measure = lowered.ops[0]
    assert measure.term.sign == 1
    assert measure.term.pairs == (("q1", "Z"),)
    assert lowered.ops[1].terms == ("c0",)
    assert lowered.ops[1].const == 1


def test_measurement_lowering_rejects_invalid_k():
    term = PauliTerm.from_pairs([("q0", "Z")])
    with pytest.raises(ValueError, match="k"):
        lower_pauli_measurement(
            start_id=0,
            term=term,
            k=0,
            result="src0",
            source_id="line1",
            next_ancilla=0,
            next_classical=0,
        )
```

- [ ] **Step 6: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: FAIL with missing `lower_pauli_measurement`.

- [ ] **Step 7: Implement measurement lowering**

In `ftcircuitbench/semi_pbc/lowering.py`, add:

- `@dataclass(frozen=True) LoweringResult(ops, next_ancilla, next_classical)`
- `lower_pauli_measurement(start_id, term, k, result, source_id, next_ancilla, next_classical) -> LoweringResult`

Implementation details:

- reject `k < 1`
- allocate physical classical bit `c<N>` for every measurement
- normalize all emitted physical measurements to positive sign
- if `term.weight <= k`, emit physical `m_pauli +P` plus `xor` mapping to the source result
- if `term.weight > k`, basis-change data qubits, keep the first `k` active qubits as the retained legal block, CNOT every remaining active qubit into the first retained qubit, measure the retained positive Z block, uncompute, undo data basis changes, and emit source-result `xor`
- for negative signed source measurements, set `xor.const = 1`
- return updated ancilla/classical counters

- [ ] **Step 8: Run lowering tests**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: PASS.

- [ ] **Step 9: Add small equivalence tests**

Add dense helper functions inside `tests/test_semi_pbc_lowering.py`; keep them in the test file rather than production code.

Concrete tests:

```python
def test_rotation_lowering_matches_original_unitary_for_xyz():
    term = PauliTerm.from_pairs([("q0", "X"), ("q1", "Y"), ("q2", "Z")])
    lowered = lower_pauli_rotation(start_id=0, term=term, k=1, source_id="line1")
    original = pauli_rotation_matrix(term, data_qubits=3)
    compiled = semi_pbc_unitary(lowered, data_qubits=3)
    assert_allclose_up_to_global_phase(compiled, original)


def test_negative_measurement_lowering_matches_source_projectors():
    term = PauliTerm.from_pairs([("q0", "X"), ("q1", "Z")], sign=-1)
    lowered = lower_pauli_measurement(
        start_id=0,
        term=term,
        k=1,
        result="src0",
        source_id="line1",
        next_ancilla=0,
        next_classical=0,
    )
    expected_zero, expected_one = signed_pauli_projectors(term, data_qubits=2)
    actual_zero, actual_one = induced_source_projectors(lowered.ops, source="src0", data_qubits=2)
    assert np.allclose(actual_zero, expected_zero)
    assert np.allclose(actual_one, expected_one)
```

Helper semantics:

- Ancillas start in `|0>`.
- `induced_source_projectors` applies the unitary prefix, projects the physical measurement outcome, applies the emitted `xor` mapping to source bit `src0`, and traces/sums over ancilla outcomes after release.
- For signed Pauli term `sP`, source bit `0` projector is `(I + sP) / 2`, and source bit `1` projector is `(I - sP) / 2`.

- [ ] **Step 10: Run equivalence tests**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: PASS.

- [ ] **Step 11: Run all semi-PBC tests so far**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py tests/test_semi_pbc_ir.py tests/test_semi_pbc_pbc_input.py tests/test_semi_pbc_lowering.py -v`

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add ftcircuitbench/semi_pbc/lowering.py tests/test_semi_pbc_lowering.py
git commit -m "feat: add exact semi-PBC baseline lowering"
```

## Chunk 3: Guarded Peres/Galvao Reducer

### Task 5: Reducer Candidate Selection and Result Mapping

**Files:**
- Create: `ftcircuitbench/semi_pbc/reducer.py`
- Test: `tests/test_semi_pbc_reducer.py`

- [ ] **Step 1: Write failing tests for safe representative reduction**

Create `tests/test_semi_pbc_reducer.py`:

```python
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.pbc_input import SourcePBCOp
from ftcircuitbench.semi_pbc.reducer import reduce_measurements


def src(idx, op, signed):
    return SourcePBCOp(id=idx, op=op, term=PauliTerm.from_full_width(signed, source_id=f"line{idx}"), source_id=f"line{idx}")


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
        src(0, "m_pauli", "+XXII"),
        src(1, "m_pauli", "+YYII"),
        src(2, "m_pauli", "+XXZI"),
    ]
    reduced = reduce_measurements(ops, greedy_order=1)
    assert reduced[2].used_source_ids == ("line0",)
```

The reducer output can be a new dataclass that wraps the source op plus `result_terms`, `result_const`, and `used_source_ids`.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_reducer.py -v`

Expected: FAIL with missing reducer.

- [ ] **Step 3: Implement reducer dataclasses and safe basic reduction**

Create `ftcircuitbench/semi_pbc/reducer.py` with:

- `@dataclass(frozen=True) ReducedSourceOp`
  - `id`
  - `op`
  - `term`
  - `source_id`
  - `result_terms: tuple[str, ...] = ()`
  - `result_const: int = 0`
  - `used_source_ids: tuple[str, ...] = ()`
- `reduce_measurements(source_ops, greedy_order=1) -> list[ReducedSourceOp]`

Initial logic:

- pass through `t_pauli`
- for `m_pauli`, consider each previous measurement independently
- a prior measurement is eligible for the current measurement only if every intervening `t_pauli` commutes with that prior measurement's current representative
- generate candidates according to `greedy_order`
- use `PauliTerm.multiply_real`
- normalize the selected replacement term to positive sign; if the signed product is negative, toggle `result_const`
- `result_terms` stores source result names such as `src0`
- `used_source_ids` stores provenance names such as `line0`
- sort candidates by:
  - smaller weight
  - fewer prior measurements
  - prior source IDs tuple
  - candidate term pairs
- accept only if candidate weight is strictly lower than current weight
- store source result dependencies from selected prior measurements

- [ ] **Step 4: Run reducer tests**

Run: `python3 -m pytest tests/test_semi_pbc_reducer.py -v`

Expected: PASS.

- [ ] **Step 5: Add guarded unsafe-candidate tests**

Extend `tests/test_semi_pbc_reducer.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify behavior**

Run: `python3 -m pytest tests/test_semi_pbc_reducer.py -v`

Expected: PASS after implementation; if unsafe behavior fails, update reducer eligibility.

- [ ] **Step 7: Add `greedy_order=0` and `greedy_order=2` tests**

Add tests that verify:

Use explicit tests:

```python
def test_greedy_order_zero_only_considers_most_recent_eligible_measurement():
    ops = [
        src(0, "m_pauli", "+ZZZI"),
        src(1, "m_pauli", "+ZZII"),
        src(2, "m_pauli", "+ZZZZ"),
    ]
    reduced = reduce_measurements(ops, greedy_order=0)
    assert reduced[2].used_source_ids == ("line1",)
    assert reduced[2].term.pairs == (("q2", "Z"), ("q3", "Z"))


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
```

Include `import pytest`.

- [ ] **Step 8: Implement any missing order handling**

Update `reduce_measurements` so candidate enumeration exactly matches the spec.

- [ ] **Step 9: Run reducer tests**

Run: `python3 -m pytest tests/test_semi_pbc_reducer.py -v`

Expected: PASS.

- [ ] **Step 10: Run chunk tests**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py tests/test_semi_pbc_pbc_input.py tests/test_semi_pbc_reducer.py -v`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add ftcircuitbench/semi_pbc/reducer.py tests/test_semi_pbc_reducer.py
git commit -m "feat: add guarded Peres-Galvao reducer"
```

## Chunk 4: Pipeline, Scheduling, CLI, and Reporting

### Task 6: Scheduling and Summary Reporting

**Files:**
- Create: `ftcircuitbench/semi_pbc/schedule.py`
- Test: `tests/test_semi_pbc_pipeline.py`

- [ ] **Step 1: Write failing summary/depth tests**

Create `tests/test_semi_pbc_pipeline.py` with initial schedule tests:

```python
from ftcircuitbench.semi_pbc.ir import SemiPBCHeader, SemiPBCOp
from ftcircuitbench.semi_pbc.pauli import PauliTerm
from ftcircuitbench.semi_pbc.schedule import compute_summary


def test_compute_summary_uses_default_phase1_latencies():
    header = SemiPBCHeader(k=1, data_qubits=2)
    ops = [
        SemiPBCOp.clifford(0, "h", ("q0",)),
        SemiPBCOp.clifford(1, "cx", ("q0", "q1")),
        SemiPBCOp.pauli_rotation(2, PauliTerm.from_pairs([("q0", "Z")])),
        SemiPBCOp.xor(3, target="src0", terms=("c0",), const=0),
    ]
    summary = compute_summary(header, ops, input_op_count=1, max_input_weight=2)
    assert summary["output_op_count"] == 4
    assert summary["max_output_weight"] == 1
    assert summary["latency_weighted_depth"] == 3


def test_compute_summary_reports_unique_allocated_ancillas():
    header = SemiPBCHeader(k=1, data_qubits=1)
    ops = [
        SemiPBCOp.alloc(0, "a0"),
        SemiPBCOp.release(1, "a0"),
        SemiPBCOp.alloc(2, "a1"),
        SemiPBCOp.release(3, "a1"),
    ]
    summary = compute_summary(header, ops, input_op_count=1, max_input_weight=2)
    assert summary["ancilla_count"] == 2
    assert summary["max_live_ancillas"] == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py::test_compute_summary_uses_default_phase1_latencies -v`

Expected: FAIL with missing schedule module.

- [ ] **Step 3: Implement `compute_summary`**

Create `ftcircuitbench/semi_pbc/schedule.py`:

- constants:
  - 1q Clifford: `1`
  - 2q Clifford: `1`
  - `t_pauli`: `1`
  - `m_pauli`: `1`
  - `alloc`, `release`, `reset`: `0`
  - `xor`: `0`
- `compute_summary(header, ops, input_op_count, max_input_weight) -> dict`
- include:
  - `record_type: "summary"`
  - `k`
  - `input_op_count`
  - `output_op_count`
  - `max_input_weight`
  - `max_output_weight`
  - `ancilla_count`
  - `max_live_ancillas`
  - `latency_weighted_depth`
  - `latency_model: "phase1_default"`

`ancilla_count` means the number of unique allocated ancilla IDs in the emitted IR. `max_live_ancillas` means the peak simultaneously live ancillas and is the metric enforced by finite `ancilla_budget` values. Phase 1 computes conservative sequential latency-weighted depth; commutation-layer reconstruction is future work, not part of this task.

- [ ] **Step 4: Run summary tests**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ftcircuitbench/semi_pbc/schedule.py tests/test_semi_pbc_pipeline.py
git commit -m "feat: add semi-PBC summary reporting"
```

### Task 7: End-to-End Compiler Pipeline

**Files:**
- Create: `ftcircuitbench/semi_pbc/pipeline.py`
- Modify: `tests/test_semi_pbc_pipeline.py`

- [ ] **Step 1: Write failing end-to-end tests**

Extend `tests/test_semi_pbc_pipeline.py`:

```python
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_text


def test_compile_pbc_text_lowers_to_hard_cap_and_preserves_source_results():
    text = "qreg q[3];\nt_pauli +XYZ;\nm_pauli -ZZZ;\n"
    result = compile_pbc_text(text, k=1, measurement_reducer="none")
    assert result.summary["max_input_weight"] == 3
    assert result.summary["max_output_weight"] == 1
    assert result.summary["input_op_count"] == 2
    assert any(op.op == "xor" and op.target == "src1" and op.const == 1 for op in result.ops)
    for op in result.ops:
        if op.op in {"t_pauli", "m_pauli"}:
            op.validate(result.header)


def test_compile_pbc_text_uses_reducer_before_lowering():
    text = "qreg q[3];\nm_pauli +ZZI;\nm_pauli +ZZZ;\n"
    result = compile_pbc_text(text, k=1, measurement_reducer="peres-galvao-greedy", greedy_order=1)
    physical_measurements = [op for op in result.ops if op.op == "m_pauli"]
    assert any(op.source_id == "line3" and op.term.pairs == (("q2", "Z"),) for op in physical_measurements)
    assert any(op.op == "xor" and op.target == "src1" and op.terms == ("c1", "src0") and op.const == 0 for op in result.ops)


def test_compile_pbc_text_combines_reducer_sign_adjustment_into_final_xor():
    text = "qreg q[2];\nm_pauli +ZI;\nm_pauli -ZZ;\n"
    result = compile_pbc_text(text, k=1, measurement_reducer="peres-galvao-greedy", greedy_order=1)
    assert any(op.op == "m_pauli" and op.source_id == "line3" and op.term.pairs == (("q1", "Z"),) for op in result.ops)
    assert any(op.op == "xor" and op.target == "src1" and op.terms == ("c1", "src0") and op.const == 1 for op in result.ops)


def test_compile_pbc_text_is_deterministic():
    text = "qreg q[3];\nm_pauli +ZZI;\nm_pauli +ZZZ;\n"
    first = compile_pbc_text(text, k=1).jsonl_records()
    second = compile_pbc_text(text, k=1).jsonl_records()
    assert first == second
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: FAIL with missing `pipeline`.

- [ ] **Step 3: Implement `compile_pbc_text` and `compile_pbc_file`**

Create `ftcircuitbench/semi_pbc/pipeline.py`:

- `@dataclass(frozen=True) CompileResult(header, ops, summary, sidecar: dict | None = None)`
- `compile_pbc_text(text, *, k, objective="latency-depth", measurement_reducer="peres-galvao-greedy", greedy_order=1, rotation_lowering="parity-network", measurement_lowering="coherent-parity", ancilla_budget=None, emit_sidecar=True) -> CompileResult`
- `compile_pbc_file(path, **kwargs) -> CompileResult`

Pipeline:

1. parse source PBC into `PBCProgram`
2. replace header `k` with requested `k`
3. optionally run reducer
4. lower each source op in order
5. validate all emitted Pauli ops against `k`
6. compute summary

Counter rules:

- operation IDs increment monotonically across emitted quantum/classical ops
- measurement lowering owns `next_ancilla` and `next_classical`; the Phase 1 max-`k` data-compression strategy updates only the classical counter for high-weight measurements because it does not allocate ancillas
- pass-through measurements still emit physical `m_pauli -> cN` and `xor srcN = cN xor sign_adjust`
- source result names are `src<source op id>`
- when a `ReducedSourceOp` has `result_terms` or `result_const`, pipeline updates the final `xor` emitted by `lower_pauli_measurement` so that:

  ```text
  srcN = c_new xor reducer.result_terms xor lowering_sign_adjust xor reducer.result_const
  ```

Defined option behavior:

- `objective` must be `"latency-depth"`; reject anything else.
- `rotation_lowering` must be `"parity-network"`; reject anything else.
- `measurement_lowering` must be `"coherent-parity"`; reject anything else.
- `emit_sidecar=True` returns a `sidecar` dictionary in `CompileResult`; `False` returns `None`.
- `CompileResult.jsonl_records()` returns deterministic JSON-serializable records for header plus ops.
- `CompileResult.sidecar` should include `format: "semi-pbc-sidecar"`, `version`, `k`, and a per-output-op provenance list with `id`, `op`, `source_id`, and `gadget_id` when available.

- [ ] **Step 4: Run pipeline tests**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Add error behavior tests**

Extend `tests/test_semi_pbc_pipeline.py`:

```python
def test_compile_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=0)


def test_compile_rejects_unsupported_reducer_option():
    with pytest.raises(ValueError, match="measurement_reducer"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, measurement_reducer="bad")


def test_compile_allows_zero_ancilla_budget_for_data_compression():
    result = compile_pbc_text("qreg q[2];\nm_pauli +ZZ;\n", k=1, ancilla_budget=0)
    assert result.summary["max_output_weight"] == 1
    assert result.summary["ancilla_count"] == 0
    assert result.summary["max_live_ancillas"] == 0


def test_compile_rejects_unsupported_strategy_options():
    with pytest.raises(ValueError, match="objective"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, objective="space")
    with pytest.raises(ValueError, match="rotation_lowering"):
        compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, rotation_lowering="moflic-paler-k2")
    with pytest.raises(ValueError, match="measurement_lowering"):
        compile_pbc_text("qreg q[1];\nm_pauli +Z;\n", k=1, measurement_lowering="chunks")


def test_compile_result_sidecar_can_be_disabled():
    result = compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, emit_sidecar=False)
    assert result.sidecar is None


def test_compile_result_sidecar_records_output_provenance_when_enabled():
    result = compile_pbc_text("qreg q[1];\nt_pauli +Z;\n", k=1, emit_sidecar=True)
    assert result.sidecar["format"] == "semi-pbc-sidecar"
    assert result.sidecar["k"] == 1
    assert result.sidecar["provenance"][0]["id"] == 0
    assert result.sidecar["provenance"][0]["source_id"] == "line2"
```

Add `import pytest` if missing.

- [ ] **Step 6: Implement missing validation**

Update `pipeline.py` and lowering calls to enforce invalid options and finite ancilla budget.

- [ ] **Step 7: Run pipeline tests**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: PASS.

- [ ] **Step 8: Run all package tests**

Run: `python3 -m pytest tests/test_semi_pbc_pauli.py tests/test_semi_pbc_ir.py tests/test_semi_pbc_pbc_input.py tests/test_semi_pbc_lowering.py tests/test_semi_pbc_reducer.py tests/test_semi_pbc_pipeline.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ftcircuitbench/semi_pbc/pipeline.py tests/test_semi_pbc_pipeline.py
git commit -m "feat: add semi-PBC compile pipeline"
```

### Task 8: CLI Wrapper

**Files:**
- Create: `compile_semi_pbc.py`
- Test: `tests/test_compile_semi_pbc_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_compile_semi_pbc_cli.py`:

```python
import json
import subprocess
import sys


def test_compile_semi_pbc_cli_writes_jsonl_and_summary(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "toy.semi_pbc.jsonl"
    summary = tmp_path / "toy.summary.json"
    sidecar = tmp_path / "toy.sidecar.json"
    pbc.write_text("qreg q[2];\nt_pauli +ZZ;\nm_pauli -ZZ;\n")
    proc = subprocess.run(
        [
            sys.executable,
            "compile_semi_pbc.py",
            "--pbc",
            str(pbc),
            "--out",
            str(out),
            "--summary",
            str(summary),
            "--sidecar",
            str(sidecar),
            "--emit-sidecar",
            "--k",
            "1",
            "--measurement-reducer",
            "none",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["format"] == "semi-pbc"
    assert rows[0]["k"] == 1
    assert json.loads(summary.read_text())["max_output_weight"] == 1
    assert json.loads(sidecar.read_text())["format"] == "semi-pbc-sidecar"
    assert "max_output_weight=1" in proc.stdout
```

- [ ] **Step 2: Run CLI test to verify failure**

Run: `python3 -m pytest tests/test_compile_semi_pbc_cli.py -v`

Expected: FAIL because `compile_semi_pbc.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `compile_semi_pbc.py`:

- parse:
  - `--pbc`
  - `--out`
  - `--summary`
  - `--sidecar`
  - `--emit-sidecar`
  - `--k`
  - `--objective latency-depth`
  - `--measurement-reducer none|peres-galvao-greedy`
  - `--greedy-order 0|1|2`
  - `--rotation-lowering parity-network`
  - `--measurement-lowering coherent-parity`
  - `--ancilla-budget unlimited|N`
- call `compile_pbc_file`
- write JSONL via `write_jsonl`
- write summary JSON if requested
- write sidecar JSON by default; `--sidecar` overrides the destination and `--emit-sidecar` is accepted for explicitness
- if no `--sidecar` is provided, write `<out>.sidecar.json`
- print one concise line:

```text
wrote <out> ops=<N> max_output_weight=<W> latency_weighted_depth=<D>
```

- [ ] **Step 4: Run CLI tests**

Run: `python3 -m pytest tests/test_compile_semi_pbc_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Run end-to-end smoke test on an existing small PBC file**

Run:

```bash
printf 'qreg q[3];\nt_pauli +ZZZ;\nm_pauli -XZI;\n' > /tmp/semi_pbc_smoke_k2.pbc
python3 compile_semi_pbc.py \
  --pbc /tmp/semi_pbc_smoke_k2.pbc \
  --out /tmp/semi_pbc_smoke_k2.semi_pbc.jsonl \
  --summary /tmp/semi_pbc_smoke_k2.semi_pbc.summary.json \
  --k 2
```

Expected: command exits `0`, summary JSON has `"max_output_weight": 2`.

- [ ] **Step 6: Run all semi-PBC tests**

Run:

```bash
python3 -m pytest \
  tests/test_semi_pbc_pauli.py \
  tests/test_semi_pbc_ir.py \
  tests/test_semi_pbc_pbc_input.py \
  tests/test_semi_pbc_lowering.py \
  tests/test_semi_pbc_reducer.py \
  tests/test_semi_pbc_pipeline.py \
  tests/test_compile_semi_pbc_cli.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add compile_semi_pbc.py tests/test_compile_semi_pbc_cli.py
git commit -m "feat: add semi-PBC compiler CLI"
```

### Task 9: Final Integration Check

**Files:**
- Modify only if needed after verification failures.

- [ ] **Step 1: Check worktree scope**

Run: `git status --short`

Expected: only semi-PBC implementation files are changed or staged for this branch's work. Existing unrelated user changes may still be present; do not revert them.

- [ ] **Step 2: Run focused test suite**

Run:

```bash
python3 -m pytest \
  tests/test_semi_pbc_pauli.py \
  tests/test_semi_pbc_ir.py \
  tests/test_semi_pbc_pbc_input.py \
  tests/test_semi_pbc_lowering.py \
  tests/test_semi_pbc_reducer.py \
  tests/test_semi_pbc_pipeline.py \
  tests/test_compile_semi_pbc_cli.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run existing nwqec/PBC path tests for regression coverage**

Run: `python3 -m pytest tests/test_nwqec_ct.py tests/test_nwqec_default_path.py tests/test_api_pipeline.py -v`

Expected: PASS.

- [ ] **Step 4: Run CLI smoke test**

Run:

```bash
printf 'qreg q[3];\nt_pauli +XYZ;\nm_pauli -ZZZ;\n' > /tmp/semi_pbc_smoke_k1.pbc
python3 compile_semi_pbc.py \
  --pbc /tmp/semi_pbc_smoke_k1.pbc \
  --out /tmp/semi_pbc_smoke_k1.semi_pbc.jsonl \
  --summary /tmp/semi_pbc_smoke_k1.semi_pbc.summary.json \
  --k 1
```

Expected: exits `0`; `/tmp/semi_pbc_smoke_k1.semi_pbc.summary.json` reports `"max_output_weight": 1`.

- [ ] **Step 5: Commit any final integration fixes**

If verification required edits:

```bash
git add ftcircuitbench/semi_pbc tests/test_semi_pbc_*.py compile_semi_pbc.py tests/test_compile_semi_pbc_cli.py
git commit -m "test: finalize semi-PBC compiler validation"
```

If no edits were needed, do not create an empty commit.
