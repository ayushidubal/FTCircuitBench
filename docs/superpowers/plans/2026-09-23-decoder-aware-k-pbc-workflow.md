# Decoder-Aware k-PBC Workflow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end decoder-aware k-PBC workflow: generate k-PBC candidates locally, route every operation through fastkPBC, run tracegen with decoder-specific timing, and preserve server-scale artifacts for the 95-circuit suite.

**Architecture:** Keep candidate generation in `FTCircuitBench-semi-pbc`, build `fastkPBC` as the router for the `H/S/Sdg/CNOT/k-T_Pauli/k-M_Pauli` gateset, and add decoder-model/timed-trace support in `tracegen`. Keep FastLS and FastMQLSS as reference/tooling dependencies unless their interfaces need small adapters. Compiler and router remain decoder-neutral; tracegen is decoder-aware.

**Tech Stack:** Python/Qiskit/pytest for FTB candidate generation and orchestration, Rust/Cargo for fastkPBC/FastLS/FastMQLSS routing tools, Python/pandas/numpy or sklearn-compatible regression for tracegen decoder models, JSONL/JSON contracts for cross-tool artifacts.

---

## Scope Check

The approved spec spans multiple tools. This plan treats them as coordinated but separately testable chunks:

- Chunk 1: FTB k-PBC candidate format and naive candidate export.
- Chunk 2: FTB segmented capped Litinski candidate generator.
- Chunk 3: fastkPBC router contract and scaffold.
- Chunk 4: tracegen decoder timing model and routed-trace ingestion.
- Chunk 5: local orchestration and server handoff artifacts.

Do not start with the full 95-circuit server run. First make the smoke path deterministic on tiny circuits, then hand server instructions to run the full suite.

## File Structure

### FTB Candidate Generation

- Create `ftcircuitbench/k_pbc/__init__.py`: public k-PBC APIs.
- Create `ftcircuitbench/k_pbc/ir.py`: JSONL schema for `H/S/Sdg/CNOT/k-T_Pauli/k-M_Pauli` plus classical dependencies.
- Create `ftcircuitbench/k_pbc/segmented_litinski.py`: direct C+T to segmented k-PBC generator.
- Create `ftcircuitbench/k_pbc/export.py`: conversion helpers and file emitters.
- Create `generate_k_pbc.py`: CLI for candidate generation.
- Create `tests/test_k_pbc_ir.py`: schema and validation tests.
- Create `tests/test_segmented_litinski.py`: small semantic/cap tests.
- Create `tests/test_generate_k_pbc_cli.py`: CLI smoke tests.

### fastkPBC Router

- Create `/Users/ayushidubal/qmem/fastkPBC/Cargo.toml`.
- Create `/Users/ayushidubal/qmem/fastkPBC/src/main.rs`: CLI entry point.
- Create `/Users/ayushidubal/qmem/fastkPBC/src/parser.rs`: k-PBC JSONL parser.
- Create `/Users/ayushidubal/qmem/fastkPBC/src/structures.rs`: router input/output structs.
- Create `/Users/ayushidubal/qmem/fastkPBC/src/routing.rs`: first deterministic routing implementation or wrapper around existing FastMQLSS/FastLS logic.
- Create `/Users/ayushidubal/qmem/fastkPBC/tests/fixtures/toy.kpbc.jsonl`.
- Create `/Users/ayushidubal/qmem/fastkPBC/tests/cli.rs`.

### tracegen Decoder Timing

- Create `/Users/ayushidubal/qmem/.worktrees/tracegen/qmem_utils/decoder_model.py`: fitted decoder latency model classes.
- Modify `/Users/ayushidubal/qmem/.worktrees/tracegen/qmem_utils/trace.py`: preserve routed `L`, routed features, and k-PBC op metadata during normalization.
- Create `/Users/ayushidubal/qmem/.worktrees/tracegen/qmem_utils/kpbc_tracegen.py`: routed k-PBC trace ingestion and decoder-aware timing.
- Create `/Users/ayushidubal/qmem/.worktrees/tracegen/tests/test_decoder_model.py`.
- Create `/Users/ayushidubal/qmem/.worktrees/tracegen/tests/test_kpbc_tracegen.py`.

### Orchestration

- Create `scripts/run_kpbc_smoke.py`: local smoke orchestration.
- Create `scripts/prepare_kpbc_server_run.py`: emits server command manifests and run directories.
- Create `docs/kpbc-workflow-artifacts.md`: artifact schema and handoff conventions.

## Chunk 1: k-PBC Candidate Format In FTB

### Task 1: Add k-PBC IR Schema

**Files:**
- Create: `ftcircuitbench/k_pbc/__init__.py`
- Create: `ftcircuitbench/k_pbc/ir.py`
- Test: `tests/test_k_pbc_ir.py`

- [ ] **Step 1: Write failing schema tests**

Add `tests/test_k_pbc_ir.py`:

```python
import pytest

from ftcircuitbench.k_pbc.ir import KPBCHeader, KPBCOp, read_kpbc_jsonl, write_kpbc_jsonl
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_kpbc_round_trips_gate_set(tmp_path):
    path = tmp_path / "toy.kpbc.jsonl"
    header = KPBCHeader(k=2, data_qubits=3)
    ops = [
        KPBCOp.clifford(0, "h", ("q0",)),
        KPBCOp.clifford(1, "s", ("q1",)),
        KPBCOp.clifford(2, "sdg", ("q1",)),
        KPBCOp.clifford(3, "cx", ("q0", "q2")),
        KPBCOp.t_pauli(4, PauliTerm.from_pairs([("q0", "Z"), ("q2", "X")])),
        KPBCOp.m_pauli(5, PauliTerm.from_pairs([("q1", "Z")]), result="c0"),
    ]
    write_kpbc_jsonl(path, header, ops)
    loaded_header, loaded_ops = read_kpbc_jsonl(path)
    assert loaded_header == header
    assert loaded_ops == tuple(ops)


def test_kpbc_rejects_pauli_above_k():
    header = KPBCHeader(k=1, data_qubits=2)
    op = KPBCOp.t_pauli(0, PauliTerm.from_pairs([("q0", "Z"), ("q1", "Z")]))
    with pytest.raises(ValueError, match="exceeds k"):
        header.validate_ops([op])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_k_pbc_ir.py -v`

Expected: FAIL because `ftcircuitbench.k_pbc` does not exist.

- [ ] **Step 3: Implement minimal IR**

Create `ftcircuitbench/k_pbc/__init__.py`:

```python
from ftcircuitbench.k_pbc.ir import KPBCHeader, KPBCOp, read_kpbc_jsonl, write_kpbc_jsonl

__all__ = ["KPBCHeader", "KPBCOp", "read_kpbc_jsonl", "write_kpbc_jsonl"]
```

Create `ftcircuitbench/k_pbc/ir.py` using the existing semi-PBC schema style:

```python
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ftcircuitbench.semi_pbc.pauli import PauliTerm

_CLIFFORD_OPS = {"h", "s", "sdg", "cx"}
_ALLOWED_OPS = _CLIFFORD_OPS | {"t_pauli", "m_pauli", "xor"}


@dataclass(frozen=True)
class KPBCHeader:
    k: int
    data_qubits: int
    format: str = "k-pbc"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be an integer >= 1")
        if type(self.data_qubits) is not int or self.data_qubits < 0:
            raise ValueError("data_qubits must be non-negative")

    def to_record(self) -> dict:
        return {
            "format": self.format,
            "version": self.version,
            "k": self.k,
            "data_qubits": self.data_qubits,
        }

    def validate_ops(self, ops: Iterable["KPBCOp"]) -> None:
        for op in ops:
            if op.op in {"t_pauli", "m_pauli"} and op.term is not None:
                if op.term.weight > self.k:
                    raise ValueError(f"{op.op} Pauli term weight {op.term.weight} exceeds k={self.k}")


@dataclass(frozen=True)
class KPBCOp:
    id: int
    op: str
    qubits: tuple[str, ...] = ()
    term: PauliTerm | None = None
    result: str | None = None
    target: str | None = None
    terms: tuple[str, ...] = ()
    const: int = 0
    source_id: str | None = None

    @classmethod
    def clifford(cls, id: int, op: str, qubits: Iterable[str], source_id: str | None = None) -> "KPBCOp":
        if op not in _CLIFFORD_OPS:
            raise ValueError(f"unsupported Clifford op {op!r}")
        qubit_tuple = tuple(qubits)
        expected = 2 if op == "cx" else 1
        if len(qubit_tuple) != expected:
            raise ValueError(f"{op} expects {expected} qubits")
        return cls(id=id, op=op, qubits=qubit_tuple, source_id=source_id)

    @classmethod
    def t_pauli(cls, id: int, term: PauliTerm, source_id: str | None = None) -> "KPBCOp":
        return cls(id=id, op="t_pauli", term=term, source_id=source_id)

    @classmethod
    def m_pauli(cls, id: int, term: PauliTerm, result: str, source_id: str | None = None) -> "KPBCOp":
        return cls(id=id, op="m_pauli", term=term, result=result, source_id=source_id)
```

Include `to_record`, `from_record`, `read_kpbc_jsonl`, and `write_kpbc_jsonl` following `ftcircuitbench/semi_pbc/ir.py`.

- [ ] **Step 4: Run schema tests**

Run: `python3 -m pytest tests/test_k_pbc_ir.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ftcircuitbench/k_pbc tests/test_k_pbc_ir.py
git commit -m "feat: add k-PBC candidate IR"
```

### Task 2: Add Naive Candidate Export

**Files:**
- Create: `ftcircuitbench/k_pbc/export.py`
- Test: `tests/test_k_pbc_export.py`

- [ ] **Step 1: Write failing export tests**

Add `tests/test_k_pbc_export.py`:

```python
from ftcircuitbench.k_pbc.export import semi_pbc_result_to_kpbc
from ftcircuitbench.semi_pbc.pipeline import compile_pbc_text


def test_naive_export_reuses_existing_capped_pipeline():
    result = compile_pbc_text("qreg q[2];\nt_pauli +ZZ;\nm_pauli +ZI;\n", k=1)
    header, ops = semi_pbc_result_to_kpbc(result)
    assert header.k == 1
    assert all(op.op in {"h", "s", "sdg", "cx", "t_pauli", "m_pauli", "xor"} for op in ops)
    assert all(op.term is None or op.term.weight <= 1 for op in ops)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3 -m pytest tests/test_k_pbc_export.py -v`

Expected: FAIL because `semi_pbc_result_to_kpbc` does not exist.

- [ ] **Step 3: Implement exporter**

Create `ftcircuitbench/k_pbc/export.py`:

```python
from __future__ import annotations

from ftcircuitbench.k_pbc.ir import KPBCHeader, KPBCOp
from ftcircuitbench.semi_pbc.pipeline import CompileResult


def semi_pbc_result_to_kpbc(result: CompileResult) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    header = KPBCHeader(k=result.header.k, data_qubits=result.header.data_qubits)
    ops: list[KPBCOp] = []
    for op in result.ops:
        if op.op in {"alloc", "release"}:
            continue
        if op.op in {"h", "s", "sdg", "cx"}:
            ops.append(KPBCOp.clifford(op.id, op.op, op.qubits, source_id=op.source_id))
        elif op.op == "t_pauli":
            ops.append(KPBCOp.t_pauli(op.id, op.term, source_id=op.source_id))
        elif op.op == "m_pauli":
            ops.append(KPBCOp.m_pauli(op.id, op.term, op.result, source_id=op.source_id))
        elif op.op == "xor":
            ops.append(KPBCOp(id=op.id, op="xor", target=op.target, terms=op.terms, const=op.const, source_id=op.source_id))
        else:
            raise ValueError(f"unsupported semi-PBC op {op.op!r}")
    header.validate_ops(ops)
    return header, tuple(ops)
```

- [ ] **Step 4: Run export tests**

Run: `python3 -m pytest tests/test_k_pbc_export.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ftcircuitbench/k_pbc/export.py tests/test_k_pbc_export.py
git commit -m "feat: export naive k-PBC candidates"
```

## Chunk 2: Segmented Capped Litinski Candidate Generator

### Task 3: Extract Clifford Propagation Helpers

**Files:**
- Create/Modify: `ftcircuitbench/k_pbc/segmented_litinski.py`
- Reference only: `ftcircuitbench/pbc_converter/r_pauli_circ.py`
- Create: `tests/test_segmented_litinski.py`

- [ ] **Step 1: Write failing propagation tests**

Add to `tests/test_segmented_litinski.py`:

```python
from qiskit import QuantumCircuit

from ftcircuitbench.k_pbc.segmented_litinski import compile_clifford_t_to_kpbc


def test_segmented_litinski_keeps_support_at_k():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(0)
    header, ops = compile_clifford_t_to_kpbc(qc, k=1)
    assert header.k == 1
    assert all(op.term is None or op.term.weight <= 1 for op in ops)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3 -m pytest tests/test_segmented_litinski.py::test_segmented_litinski_keeps_support_at_k -v`

Expected: FAIL because the compiler does not exist.

- [ ] **Step 3: Implement tableau support helpers**

Prefer local helpers in `ftcircuitbench/k_pbc/segmented_litinski.py` instead of modifying `RotationPauliCirc` unless duplication becomes painful. Use `TableauPauliBasis` and `TableauForGate.apply_gate` from `ftcircuitbench/pbc_converter/tab_gate.py`.

Implement helpers:

```python
def _z_rotation_row(num_qubits: int, qubit: int, tdg: bool) -> np.ndarray: ...
def _row_weight(row: np.ndarray, num_qubits: int) -> int: ...
def _row_to_pauli_term(row: np.ndarray, num_qubits: int) -> PauliTerm: ...
def _prospective_rows_after_gate(rows: list[np.ndarray], gate_name: str, qubits: list[int], num_qubits: int) -> list[np.ndarray]: ...
```

- [ ] **Step 4: Run focused test**

Run: `python3 -m pytest tests/test_segmented_litinski.py::test_segmented_litinski_keeps_support_at_k -v`

Expected: still FAIL until the public compiler exists.

### Task 4: Implement Whole-Segment Cut Compiler

**Files:**
- Create: `ftcircuitbench/k_pbc/segmented_litinski.py`
- Test: `tests/test_segmented_litinski.py`

- [ ] **Step 1: Extend tests for segment cuts and k=n**

Add:

```python
def test_segmented_litinski_k_equals_n_matches_full_support_case():
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.cx(0, 1)
    header, ops = compile_clifford_t_to_kpbc(qc, k=2)
    assert header.k == 2
    assert any(op.op == "t_pauli" and op.term.weight == 2 for op in ops)


def test_segmented_litinski_emits_boundary_clifford_when_growth_would_exceed_k():
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.cx(0, 1)
    header, ops = compile_clifford_t_to_kpbc(qc, k=1)
    assert header.k == 1
    assert any(op.op == "cx" for op in ops)
    assert all(op.term is None or op.term.weight <= 1 for op in ops)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_segmented_litinski.py -v`

Expected: FAIL until compiler is implemented.

- [ ] **Step 3: Implement compiler**

Create public function:

```python
def compile_clifford_t_to_kpbc(qc: QuantumCircuit, *, k: int) -> tuple[KPBCHeader, tuple[KPBCOp, ...]]:
    ...
```

Implementation policy:

- Iterate over `qc.data` in reverse like `RotationPauliCirc.process`.
- Maintain pending Pauli rows for T/Tdg rotations.
- For Clifford gates, compute prospective pending rows.
- If all prospective weights are `<= k`, accept the propagation.
- If any prospective weight exceeds `k`, flush the current segment into `t_pauli` ops, emit the Clifford explicitly, clear or restart the segment according to the current Clifford frame, then continue.
- Emit final pending rows in source order.
- Preserve `h`, `s`, `sdg`, and `cx` explicitly when they are segment boundaries.
- Reject unsupported gates with `ValueError`.

- [ ] **Step 4: Run segmented tests**

Run: `python3 -m pytest tests/test_segmented_litinski.py -v`

Expected: PASS.

- [ ] **Step 5: Run relevant PBC tests**

Run: `python3 -m pytest tests/test_segmented_litinski.py tests/test_k_pbc_ir.py tests/test_k_pbc_export.py tests/test_semi_pbc_pauli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ftcircuitbench/k_pbc/segmented_litinski.py tests/test_segmented_litinski.py
git commit -m "feat: add segmented capped Litinski k-PBC generator"
```

### Task 5: Add Candidate Generation CLI

**Files:**
- Create: `generate_k_pbc.py`
- Test: `tests/test_generate_k_pbc_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add:

```python
import json
import subprocess
import sys


def test_generate_k_pbc_cli_writes_candidate(tmp_path):
    qasm = tmp_path / "toy.qasm"
    out = tmp_path / "toy.kpbc.jsonl"
    qasm.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nt q[0];\n')
    result = subprocess.run(
        [sys.executable, "generate_k_pbc.py", "--qasm", str(qasm), "--out", str(out), "--k", "1", "--mode", "segmented-litinski"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert out.exists()
    assert json.loads(out.read_text().splitlines()[0])["format"] == "k-pbc"
    assert "wrote" in result.stdout
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3 -m pytest tests/test_generate_k_pbc_cli.py -v`

Expected: FAIL because CLI does not exist.

- [ ] **Step 3: Implement CLI**

CLI options:

```text
--qasm PATH
--out PATH
--k N
--mode naive|segmented-litinski
--summary PATH optional
```

Use Qiskit to load QASM for segmented mode. For naive mode, use existing C+T -> PBC path only if the input path can run through existing `convert_to_pbc_circuit`; otherwise document the limitation and support segmented first.

- [ ] **Step 4: Run CLI tests**

Run: `python3 -m pytest tests/test_generate_k_pbc_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add generate_k_pbc.py tests/test_generate_k_pbc_cli.py
git commit -m "feat: add k-PBC candidate CLI"
```

## Chunk 3: fastkPBC Router

### Task 6: Scaffold fastkPBC Parser

**Files:**
- Create: `/Users/ayushidubal/qmem/fastkPBC/Cargo.toml`
- Create: `/Users/ayushidubal/qmem/fastkPBC/src/main.rs`
- Create: `/Users/ayushidubal/qmem/fastkPBC/src/parser.rs`
- Create: `/Users/ayushidubal/qmem/fastkPBC/src/structures.rs`
- Create: `/Users/ayushidubal/qmem/fastkPBC/tests/fixtures/toy.kpbc.jsonl`
- Create: `/Users/ayushidubal/qmem/fastkPBC/tests/cli.rs`

- [ ] **Step 1: Create router repo/worktree**

Run:

```bash
mkdir -p /Users/ayushidubal/qmem/fastkPBC/src /Users/ayushidubal/qmem/fastkPBC/tests/fixtures
cd /Users/ayushidubal/qmem/fastkPBC
git init
cargo init --bin .
```

Expected: new Rust binary crate.

- [ ] **Step 2: Write failing parser test**

Create `tests/fixtures/toy.kpbc.jsonl`:

```jsonl
{"format":"k-pbc","version":1,"k":2,"data_qubits":2}
{"id":0,"op":"h","qubits":["q0"]}
{"id":1,"op":"cx","qubits":["q0","q1"]}
{"id":2,"op":"t_pauli","terms":[["q0","Z"],["q1","Z"]],"sign":1,"angle_num":1,"angle_den":8}
{"id":3,"op":"m_pauli","terms":[["q1","X"]],"sign":1,"result":"c0"}
```

Create `tests/cli.rs`:

```rust
use std::process::Command;

#[test]
fn routes_toy_kpbc_file() {
    let output = Command::new(env!("CARGO_BIN_EXE_fastkpbc"))
        .args(["--input", "tests/fixtures/toy.kpbc.jsonl", "--output", "/tmp/toy.kpbc.routed.json"])
        .output()
        .expect("run fastkpbc");
    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
}
```

- [ ] **Step 3: Run test to verify failure**

Run: `cd /Users/ayushidubal/qmem/fastkPBC && cargo test`

Expected: FAIL until CLI/parser exists.

- [ ] **Step 4: Implement parser and identity-style router output**

Use structs:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct Header { pub format: String, pub version: u64, pub k: usize, pub data_qubits: usize }

#[derive(Debug, serde::Deserialize)]
pub struct Op {
    pub id: u64,
    pub op: String,
    pub qubits: Option<Vec<String>>,
    pub terms: Option<Vec<(String, String)>>,
    pub sign: Option<i8>,
    pub result: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct RoutedOp {
    pub id: u64,
    pub op: String,
    pub logical_qubits: Vec<usize>,
    pub routed_l: usize,
    pub routed_area: usize,
}
```

The first router can emit deterministic placeholder geometry with `routed_l = logical_qubits.len()` to unblock tracegen integration. Replace with real route search in the next task.

- [ ] **Step 5: Run parser tests**

Run: `cd /Users/ayushidubal/qmem/fastkPBC && cargo test`

Expected: PASS.

- [ ] **Step 6: Commit in fastkPBC**

```bash
cd /Users/ayushidubal/qmem/fastkPBC
git add .
git commit -m "feat: scaffold k-PBC router parser"
```

### Task 7: Add Real Routing Hook

**Files:**
- Modify: `/Users/ayushidubal/qmem/fastkPBC/src/routing.rs`
- Modify: `/Users/ayushidubal/qmem/fastkPBC/src/main.rs`
- Test: `/Users/ayushidubal/qmem/fastkPBC/tests/cli.rs`

- [ ] **Step 1: Write routed length assertion**

Extend `tests/cli.rs` to parse `/tmp/toy.kpbc.routed.json` and assert every routed op has `routed_l >= logical_qubits.len()`.

- [ ] **Step 2: Run test**

Run: `cd /Users/ayushidubal/qmem/fastkPBC && cargo test`

Expected: PASS with placeholder, then keep passing after real routing.

- [ ] **Step 3: Implement first real route adapter**

Choose the nearest reusable logic:

- If FastMQLSS already routes PBC-style multi-qubit operations well, wrap/adapt its parser structures.
- If FastLS has better Clifford routing for `H/S/Sdg/CNOT`, use its architecture/mapping pieces.
- Keep fastkPBC's external input/output format stable even if internals delegate to FastLS/FastMQLSS modules.

Record in code comments which existing tool is being wrapped and why.

- [ ] **Step 4: Run Rust tests**

Run: `cd /Users/ayushidubal/qmem/fastkPBC && cargo test`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/ayushidubal/qmem/fastkPBC
git add src tests
git commit -m "feat: route k-PBC operations"
```

## Chunk 4: tracegen Decoder-Aware Timing

### Task 8: Add Decoder Model Class

**Files:**
- Create: `/Users/ayushidubal/qmem/.worktrees/tracegen/qmem_utils/decoder_model.py`
- Test: `/Users/ayushidubal/qmem/.worktrees/tracegen/tests/test_decoder_model.py`

- [ ] **Step 1: Write failing decoder model tests**

Create:

```python
from qmem_utils.decoder_model import PowerLawDecoderModel


def test_power_law_decoder_model_scores_routed_op():
    model = PowerLawDecoderModel(name="toy", t0=1.0, alpha=2.0, beta_l=1.0, beta_d=2.0)
    assert model.latency({"routed_l": 3, "d": 5}) == 151.0


def test_decoder_model_flags_extrapolation():
    model = PowerLawDecoderModel(name="toy", t0=0.0, alpha=1.0, beta_l=1.0, beta_d=1.0, l_range=(1, 4), d_range=(3, 9))
    assert model.is_extrapolated({"routed_l": 5, "d": 5})
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd /Users/ayushidubal/qmem/.worktrees/tracegen && python3 -m pytest tests/test_decoder_model.py -v`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement model**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PowerLawDecoderModel:
    name: str
    t0: float
    alpha: float
    beta_l: float
    beta_d: float
    l_range: tuple[int, int] | None = None
    d_range: tuple[int, int] | None = None

    def latency(self, routed_op: Mapping[str, object]) -> float:
        l_value = float(routed_op["routed_l"])
        d_value = float(routed_op["d"])
        return self.t0 + self.alpha * (l_value ** self.beta_l) * (d_value ** self.beta_d)

    def is_extrapolated(self, routed_op: Mapping[str, object]) -> bool:
        l_value = int(routed_op["routed_l"])
        d_value = int(routed_op["d"])
        return _outside(l_value, self.l_range) or _outside(d_value, self.d_range)


def _outside(value: int, bounds: tuple[int, int] | None) -> bool:
    return bounds is not None and not (bounds[0] <= value <= bounds[1])
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/ayushidubal/qmem/.worktrees/tracegen && python3 -m pytest tests/test_decoder_model.py -v`

Expected: PASS.

- [ ] **Step 5: Commit in tracegen**

```bash
cd /Users/ayushidubal/qmem/.worktrees/tracegen
git add qmem_utils/decoder_model.py tests/test_decoder_model.py
git commit -m "feat: add decoder timing model"
```

### Task 9: Ingest Routed k-PBC Output

**Files:**
- Create: `/Users/ayushidubal/qmem/.worktrees/tracegen/qmem_utils/kpbc_tracegen.py`
- Modify: `/Users/ayushidubal/qmem/.worktrees/tracegen/qmem_utils/trace.py`
- Test: `/Users/ayushidubal/qmem/.worktrees/tracegen/tests/test_kpbc_tracegen.py`

- [ ] **Step 1: Write failing tracegen tests**

Create a small routed JSON fixture in the test file and assert durations come from `PowerLawDecoderModel`.

```python
from qmem_utils.decoder_model import PowerLawDecoderModel
from qmem_utils.kpbc_tracegen import time_routed_kpbc_ops


def test_time_routed_kpbc_ops_uses_decoder_model():
    routed = [{"id": 0, "op": "t_pauli", "logical_qubits": [0, 1], "routed_l": 3, "routed_area": 4}]
    model = PowerLawDecoderModel(name="toy", t0=1.0, alpha=1.0, beta_l=1.0, beta_d=1.0)
    events, summary = time_routed_kpbc_ops(routed, decoder_model=model, d=5)
    assert events[0]["op_time"] == 16
    assert summary["decoder_model"] == "toy"
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd /Users/ayushidubal/qmem/.worktrees/tracegen && python3 -m pytest tests/test_kpbc_tracegen.py -v`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement timing helper**

Implement:

```python
def time_routed_kpbc_ops(routed_ops, *, decoder_model, d: int):
    events = []
    t = 0
    extrapolations = 0
    for op in routed_ops:
        op_with_d = {**op, "d": d}
        duration = int(round(decoder_model.latency(op_with_d)))
        extrapolations += int(decoder_model.is_extrapolated(op_with_d))
        events.append({
            "t": t,
            "end": t + duration,
            "op_time": duration,
            "qubits": list(op.get("logical_qubits", [])),
            "op_name": op["op"],
            "op_type": op["op"],
            "routed_l": op.get("routed_l"),
            "routed_area": op.get("routed_area"),
        })
        t += duration
    return events, {"decoder_model": decoder_model.name, "makespan": t, "model_extrapolation_count": extrapolations}
```

This serial schedule is only the first smoke implementation. Replace with tracegen's full dependency/resource scheduler after the interface is stable.

- [ ] **Step 4: Run tracegen tests**

Run: `cd /Users/ayushidubal/qmem/.worktrees/tracegen && python3 -m pytest tests/test_decoder_model.py tests/test_kpbc_tracegen.py -v`

Expected: PASS.

- [ ] **Step 5: Commit in tracegen**

```bash
cd /Users/ayushidubal/qmem/.worktrees/tracegen
git add qmem_utils/kpbc_tracegen.py qmem_utils/trace.py tests/test_kpbc_tracegen.py
git commit -m "feat: time routed k-PBC traces"
```

## Chunk 5: Orchestration And Server Handoff

### Task 10: Add Local Smoke Runner

**Files:**
- Create: `scripts/run_kpbc_smoke.py`
- Test: `tests/test_kpbc_smoke_runner.py`

- [ ] **Step 1: Write failing smoke runner test**

Use monkeypatching to avoid requiring the Rust router in unit tests. Assert the runner creates an artifact directory with candidate, routed, trace, and summary files.

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/test_kpbc_smoke_runner.py -v`

Expected: FAIL because runner does not exist.

- [ ] **Step 3: Implement runner**

Runner inputs:

```text
--qasm PATH
--out-dir PATH
--k N
--compiler-mode segmented-litinski|naive
--router-bin PATH
--tracegen-python PATH optional
--decoder-model JSON
--d N
```

Runner outputs:

```text
candidate.kpbc.jsonl
routed.json
trace.jsonl
summary.json
run.log
```

- [ ] **Step 4: Run smoke runner tests**

Run: `python3 -m pytest tests/test_kpbc_smoke_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Run real local smoke**

Run:

```bash
python3 scripts/run_kpbc_smoke.py \
  --qasm qasm/small_example.qasm \
  --out-dir /tmp/kpbc_smoke \
  --k 1 \
  --compiler-mode segmented-litinski \
  --router-bin /Users/ayushidubal/qmem/fastkPBC/target/debug/fastkpbc \
  --d 5
```

Expected: writes all four artifacts. If no small QASM exists, create a temporary one in `/tmp`.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_kpbc_smoke.py tests/test_kpbc_smoke_runner.py
git commit -m "feat: add local k-PBC smoke runner"
```

### Task 11: Add Server Manifest Generator

**Files:**
- Create: `scripts/prepare_kpbc_server_run.py`
- Create: `docs/kpbc-workflow-artifacts.md`
- Test: `tests/test_prepare_kpbc_server_run.py`

- [ ] **Step 1: Write manifest test**

Test that a temporary input directory with two circuits produces a manifest with `(decoder, compiler_mode, k)` rows and no date/time in folder names.

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/test_prepare_kpbc_server_run.py -v`

Expected: FAIL because script does not exist.

- [ ] **Step 3: Implement manifest generator**

Inputs:

```text
--circuits-dir PATH
--out-dir PATH
--compiler-modes segmented-litinski naive
--k-values 1,n,mid
--decoders PATH_TO_DECODER_JSON
--d N
```

Output:

```text
manifest.jsonl
run_server.sh
README.md
```

The server script should parallelize runs but leave actual concurrency configurable:

```bash
JOBS="${JOBS:-8}"
xargs -P "$JOBS" -I {} bash -lc '{}'
```

- [ ] **Step 4: Run manifest tests**

Run: `python3 -m pytest tests/test_prepare_kpbc_server_run.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_kpbc_server_run.py docs/kpbc-workflow-artifacts.md tests/test_prepare_kpbc_server_run.py
git commit -m "feat: prepare k-PBC server run manifests"
```

### Task 12: Final Local Verification

**Files:**
- No new files.

- [ ] **Step 1: Run FTB tests**

Run:

```bash
python3 -m pytest \
  tests/test_k_pbc_ir.py \
  tests/test_k_pbc_export.py \
  tests/test_segmented_litinski.py \
  tests/test_generate_k_pbc_cli.py \
  tests/test_kpbc_smoke_runner.py \
  tests/test_prepare_kpbc_server_run.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run tracegen tests**

Run:

```bash
cd /Users/ayushidubal/qmem/.worktrees/tracegen
python3 -m pytest tests/test_decoder_model.py tests/test_kpbc_tracegen.py -v
```

Expected: PASS.

- [ ] **Step 3: Run fastkPBC tests**

Run:

```bash
cd /Users/ayushidubal/qmem/fastkPBC
cargo test
```

Expected: PASS.

- [ ] **Step 4: Record commit hashes**

Run:

```bash
git -C /Users/ayushidubal/qmem/FTCircuitBench-semi-pbc log -1 --oneline
git -C /Users/ayushidubal/qmem/.worktrees/tracegen log -1 --oneline
git -C /Users/ayushidubal/qmem/fastkPBC log -1 --oneline
```

Expected: three hashes to include in server handoff.

- [ ] **Step 5: Prepare first server handoff**

Create a message for the server agent with:

- repos/branches/hashes to clone or fetch;
- paths for FTB, tracegen, fastkPBC, FastLS, FastMQLSS;
- server input corpus path;
- output directory without time/date suffix;
- manifest generation command;
- run command with `JOBS`;
- expected artifacts to return.

Do not launch the 95-circuit server run until the local smoke path has produced all artifacts.
