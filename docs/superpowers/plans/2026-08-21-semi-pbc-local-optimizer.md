# Semi-PBC Local Optimizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact `local-window` optimizer for capped PBC -> semi-PBC lowering.

**Architecture:** Add a focused optimizer module that selects retained Pauli-block qubits from local source-term context and cancels adjacent inverse Clifford pairs after lowering. Keep baseline behavior available through `optimization="none"` and expose the new mode through the API, CLI, summary, and sidecar.

**Tech Stack:** Python 3, dataclasses, stdlib `argparse`/`math`/`itertools`, pytest, existing `ftcircuitbench.semi_pbc` modules.

**Spec:** `docs/superpowers/specs/2026-08-21-semi-pbc-local-optimizer-design.md`

---

## File Structure

- Create `ftcircuitbench/semi_pbc/optimizer.py`
  - Retained-block selection and adjacent inverse-Clifford cancellation.
- Modify `ftcircuitbench/semi_pbc/lowering.py`
  - Accept optional retained-qubit order for high-weight lowering.
- Modify `ftcircuitbench/semi_pbc/pipeline.py`
  - Add `optimization` validation, context-aware lowering, post-lowering cancellation, summary/sidecar metadata.
- Modify `compile_semi_pbc.py`
  - Add `--optimization none|local-window`.
- Test `tests/test_semi_pbc_optimizer.py`
  - Unit tests for selector and peephole.
- Modify `tests/test_semi_pbc_pipeline.py`
  - Integration tests for baseline preservation, optimized reduction, and sidecar filtering.
- Modify `tests/test_compile_semi_pbc_cli.py`
  - CLI test for `--optimization local-window`.

## Chunk 1: Optimizer Utilities

### Task 1: Retained-Block Selector

**Files:**
- Create: `ftcircuitbench/semi_pbc/optimizer.py`
- Test: `tests/test_semi_pbc_optimizer.py`

- [ ] **Step 1: Write failing selector tests**

Add tests asserting:

```python
from ftcircuitbench.semi_pbc.optimizer import choose_retained_block
from ftcircuitbench.semi_pbc.pauli import PauliTerm


def test_choose_retained_block_prefers_neighbor_overlap():
    term = PauliTerm.from_full_width("+ZZZZ")
    neighbor = PauliTerm.from_full_width("+IIZZ")

    block = choose_retained_block(term, k=2, neighbor_terms=(neighbor,))

    assert block.retained_qubits == ("q2", "q3")
    assert block.extra_qubits == ("q0", "q1")
    assert block.target == "q2"


def test_choose_retained_block_uses_canonical_tie_break():
    term = PauliTerm.from_full_width("+ZZZZ")

    block = choose_retained_block(term, k=2, neighbor_terms=())

    assert block.retained_qubits == ("q0", "q1")
    assert block.extra_qubits == ("q2", "q3")
    assert block.target == "q0"
```

- [ ] **Step 2: Run selector tests and verify RED**

Run: `python3 -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: FAIL because `ftcircuitbench.semi_pbc.optimizer` does not exist.

- [ ] **Step 3: Implement selector**

Create a small `RetainedBlock` dataclass and `choose_retained_block(term, k, neighbor_terms=(), enumeration_limit=2000)`.

Implementation rules:

- reject non-integer or `<1` `k`;
- reject identity terms;
- if `term.weight <= k`, retain the full term;
- compute neighbor-overlap counts from the supplied terms;
- enumerate combinations when `math.comb(weight, k) <= enumeration_limit`;
- otherwise use a deterministic greedy fallback;
- order retained qubits with the chosen target first, then the remaining retained qubits in source canonical order;
- order extra qubits in source canonical order.

- [ ] **Step 4: Run selector tests and verify GREEN**

Run: `python3 -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: PASS.

### Task 2: Adjacent Clifford Cancellation

**Files:**
- Modify: `ftcircuitbench/semi_pbc/optimizer.py`
- Test: `tests/test_semi_pbc_optimizer.py`

- [ ] **Step 1: Write failing peephole tests**

Add tests asserting:

```python
from ftcircuitbench.semi_pbc.ir import SemiPBCOp


def test_cancel_adjacent_inverse_cliffords_removes_only_exact_pairs():
    ops = [
        SemiPBCOp.clifford(0, "h", ("q0",)),
        SemiPBCOp.clifford(1, "h", ("q0",)),
        SemiPBCOp.clifford(2, "s", ("q1",)),
        SemiPBCOp.clifford(3, "sdg", ("q1",)),
        SemiPBCOp.clifford(4, "cx", ("q0", "q1")),
        SemiPBCOp.clifford(5, "cx", ("q0", "q1")),
        SemiPBCOp.clifford(6, "h", ("q2",)),
        SemiPBCOp.clifford(7, "h", ("q3",)),
    ]

    optimized = cancel_adjacent_inverse_cliffords(ops)

    assert [op.id for op in optimized] == [6, 7]
```

- [ ] **Step 2: Run peephole tests and verify RED**

Run: `python3 -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: FAIL because `cancel_adjacent_inverse_cliffords` is missing.

- [ ] **Step 3: Implement stack-style cancellation**

Add `cancel_adjacent_inverse_cliffords(ops)` that keeps operation IDs unchanged and removes only adjacent pairs accepted by `_are_inverse_cliffords(left, right)`.

- [ ] **Step 4: Run optimizer tests and verify GREEN**

Run: `python3 -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: PASS.

## Chunk 2: Lowering and Pipeline Wiring

### Task 3: Lowering Retained-Block Override

**Files:**
- Modify: `ftcircuitbench/semi_pbc/lowering.py`
- Test: `tests/test_semi_pbc_lowering.py`

- [ ] **Step 1: Write failing lowering tests**

Add tests proving `lower_pauli_rotation(..., retained_qubits=("q2", "q3"))` emits the bounded Pauli rotation on `q2,q3` and folds `q0,q1` into target `q2`.

- [ ] **Step 2: Run lowering test and verify RED**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: FAIL because the override parameter is unsupported.

- [ ] **Step 3: Implement retained override validation**

Change `_kept_and_extra_qubits(term, k, retained_qubits=None)` and thread the optional override through `lower_pauli_rotation` and `lower_pauli_measurement`.

Validation rules:

- retained override may only be supplied for active qubits in the term;
- retained count must be `min(k, term.weight)`;
- duplicates are rejected;
- the first retained qubit is the compression target.

- [ ] **Step 4: Run lowering tests and verify GREEN**

Run: `python3 -m pytest tests/test_semi_pbc_lowering.py -v`

Expected: PASS.

### Task 4: Pipeline Optimization Mode

**Files:**
- Modify: `ftcircuitbench/semi_pbc/pipeline.py`
- Test: `tests/test_semi_pbc_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Add tests asserting:

- `optimization="none"` preserves current op count on two adjacent high-weight rotations;
- `optimization="local-window"` reduces the op count for two adjacent identical `+ZZZZ` rotations at `k=2`;
- optimized sidecar provenance IDs match the retained output op IDs;
- unsupported optimization strings raise `ValueError`.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: FAIL because `compile_pbc_text` does not accept `optimization`.

- [ ] **Step 3: Implement pipeline wiring**

Add `optimization="none"` to `compile_pbc_text`.

For `local-window`:

- compute immediate non-identity neighbor terms from the reduced source-op stream;
- pass the selected retained block to lowering for high-weight `t_pauli` and `m_pauli`;
- run `cancel_adjacent_inverse_cliffords` over emitted ops;
- filter provenance to retained operation IDs;
- include `optimization` in summary and sidecar.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run: `python3 -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: PASS.

## Chunk 3: CLI and Full Verification

### Task 5: CLI Flag

**Files:**
- Modify: `compile_semi_pbc.py`
- Test: `tests/test_compile_semi_pbc_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add a CLI test that invokes:

```text
--optimization local-window
```

and asserts the summary file records `"optimization": "local-window"`.

- [ ] **Step 2: Run CLI test and verify RED**

Run: `python3 -m pytest tests/test_compile_semi_pbc_cli.py -v`

Expected: FAIL because the CLI option is unsupported.

- [ ] **Step 3: Implement CLI option**

Add parser choices `("none", "local-window")` and pass the value into `compile_pbc_file`.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run: `python3 -m pytest tests/test_compile_semi_pbc_cli.py -v`

Expected: PASS.

### Task 6: Verification, Review, Commit

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_semi_pbc_optimizer.py tests/test_semi_pbc_lowering.py tests/test_semi_pbc_pipeline.py tests/test_compile_semi_pbc_cli.py -v
```

Expected: PASS.

- [ ] **Step 2: Run formatting/lint verification**

Run:

```bash
python3 -m ruff check ftcircuitbench/semi_pbc compile_semi_pbc.py tests/test_semi_pbc_optimizer.py tests/test_semi_pbc_lowering.py tests/test_semi_pbc_pipeline.py tests/test_compile_semi_pbc_cli.py
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python3 -m pytest -q
```

Expected: existing suite remains green, with only known optional QASM3 skips if the optional dependency is absent.

- [ ] **Step 4: Inspect diff**

Run:

```bash
git diff --stat
git diff --check
git status --short
```

Expected: no whitespace errors and only intended files changed.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add docs/superpowers/plans/2026-08-21-semi-pbc-local-optimizer.md ftcircuitbench/semi_pbc/optimizer.py ftcircuitbench/semi_pbc/lowering.py ftcircuitbench/semi_pbc/pipeline.py compile_semi_pbc.py tests/test_semi_pbc_optimizer.py tests/test_semi_pbc_lowering.py tests/test_semi_pbc_pipeline.py tests/test_compile_semi_pbc_cli.py
git commit -m "feat: add semi-PBC local optimizer"
git push fork semi-pbc-compiler
```

Expected: branch `semi-pbc-compiler` updated on the fork.
