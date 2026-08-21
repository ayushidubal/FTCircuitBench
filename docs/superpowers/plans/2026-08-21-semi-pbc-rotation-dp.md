# Semi-PBC Rotation-DP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opportunity report and exact source-order rotation-DP optimizer for capped PBC -> semi-PBC lowering.

**Architecture:** Extend the existing `ftcircuitbench.semi_pbc` package. Keep optimization utilities in `optimizer.py`, compile orchestration in `pipeline.py`, reporting in a new `opportunities.py`, and CLI entry points thin. Use existing lowering functions to derive candidate frames so the DP optimizer shares the proven baseline gadget semantics.

**Tech Stack:** Python 3, dataclasses, stdlib `argparse`/`json`/`collections`, pytest, existing semi-PBC IR/lowering/parser modules.

**Spec:** `docs/superpowers/specs/2026-08-21-semi-pbc-rotation-dp-design.md`

---

## File Structure

- Modify `ftcircuitbench/semi_pbc/optimizer.py`
  - Add candidate retained-block generation, rotation-frame candidates, and run-level DP emission.
- Modify `ftcircuitbench/semi_pbc/pipeline.py`
  - Add `optimization="rotation-dp"` and route contiguous high-weight rotation runs through the DP optimizer.
- Create `ftcircuitbench/semi_pbc/opportunities.py`
  - Analyze PBC text/files and return compact opportunity reports.
- Modify `compile_semi_pbc.py`
  - Accept `--optimization rotation-dp`.
- Create `analyze_semi_pbc_opportunities.py`
  - CLI wrapper for opportunity reports.
- Modify tests:
  - `tests/test_semi_pbc_optimizer.py`
  - `tests/test_semi_pbc_pipeline.py`
  - `tests/test_compile_semi_pbc_cli.py`
- Create tests:
  - `tests/test_semi_pbc_opportunities.py`

## Chunk 1: Candidate Frames and Rotation-DP

### Task 1: Candidate Retained Blocks

**Files:**
- Modify: `ftcircuitbench/semi_pbc/optimizer.py`
- Test: `tests/test_semi_pbc_optimizer.py`

- [ ] **Step 1: Write failing candidate tests**

Add tests requiring `candidate_retained_blocks(term, k=2)` to include multiple target choices for the same retained subset and to include the existing `choose_retained_block` result first when neighbor terms are supplied.

- [ ] **Step 2: Run candidate tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: FAIL because `candidate_retained_blocks` is missing.

- [ ] **Step 3: Implement deterministic candidate generation**

Implementation rules:

- validate `k`;
- reject identity terms;
- for weight `<= k`, return the full active term as one candidate;
- when `comb(weight, min(k, weight)) * retained_count <= enumeration_limit`, enumerate every retained subset and every target inside it;
- otherwise return a deterministic unique list containing canonical and local-window candidates.

- [ ] **Step 4: Run candidate tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: PASS.

### Task 2: Rotation-DP Run Optimizer

**Files:**
- Modify: `ftcircuitbench/semi_pbc/optimizer.py`
- Test: `tests/test_semi_pbc_optimizer.py`

- [ ] **Step 1: Write failing DP utility tests**

Add a test for `optimize_rotation_run` using two high-weight rotations where the optimal frame sequence keeps the same parity frame and emits fewer CNOTs than independently lowered rotations.

- [ ] **Step 2: Run DP tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: FAIL because `optimize_rotation_run` is missing.

- [ ] **Step 3: Implement run optimizer**

Use `lower_pauli_rotation(..., retained_qubits=block.retained_qubits)` to derive each candidate's prefix, bounded rotation op, and suffix.

DP cost:

```text
len(prefix_0) + sum(len(cancel(suffix_i + prefix_j))) + len(suffix_last)
```

Emit selected candidates with fresh increasing IDs:

```text
prefix_0, t_0, transition_0_1, t_1, ..., suffix_last
```

Return a `RotationRunOptimization` dataclass containing `ops` and `source_ops_by_output_id` for provenance.

- [ ] **Step 4: Run optimizer tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_optimizer.py -v`

Expected: PASS.

## Chunk 2: Pipeline and CLI

### Task 3: Pipeline Integration

**Files:**
- Modify: `ftcircuitbench/semi_pbc/pipeline.py`
- Test: `tests/test_semi_pbc_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Add tests asserting:

- `compile_pbc_text(..., optimization="rotation-dp")` accepts the mode;
- the DP mode reduces a toy rotation run that `local-window` leaves unchanged;
- semantic unitary equivalence holds for that toy;
- sidecar provenance IDs match retained output IDs.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: FAIL because `rotation-dp` is unsupported.

- [ ] **Step 3: Implement pipeline routing**

In `_lower_source_ops`, detect contiguous runs of high-weight `t_pauli` operations only when `optimization == "rotation-dp"`. Lower such runs through `optimize_rotation_run`; lower all other operations through existing logic. Preserve `local-window` behavior for measurements and run final adjacent-Clifford cancellation.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_pipeline.py -v`

Expected: PASS.

### Task 4: Compiler CLI

**Files:**
- Modify: `compile_semi_pbc.py`
- Test: `tests/test_compile_semi_pbc_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add a test invoking `compile_semi_pbc.py --optimization rotation-dp` and asserting the summary records `"rotation-dp"`.

- [ ] **Step 2: Run CLI test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_compile_semi_pbc_cli.py -v`

Expected: FAIL because the CLI choices do not include `rotation-dp`.

- [ ] **Step 3: Add CLI choice**

Extend `--optimization` choices to `("none", "local-window", "rotation-dp")`.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_compile_semi_pbc_cli.py -v`

Expected: PASS.

## Chunk 3: Opportunity Reports

### Task 5: Report API

**Files:**
- Create: `ftcircuitbench/semi_pbc/opportunities.py`
- Test: `tests/test_semi_pbc_opportunities.py`

- [ ] **Step 1: Write failing report tests**

Add tests for `analyze_pbc_text_opportunities` on a toy PBC input. Assert expected histograms, high-weight counts, run lengths, repeated supports, and compile comparison keys for `none`, `local-window`, and `rotation-dp`.

- [ ] **Step 2: Run report tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_opportunities.py -v`

Expected: FAIL because the module is missing.

- [ ] **Step 3: Implement report API**

Parse and reduce source ops, compute report fields from the reduced source stream, and include compile comparisons by calling `compile_pbc_text` with each optimization mode.

- [ ] **Step 4: Run report tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_opportunities.py -v`

Expected: PASS.

### Task 6: Report CLI

**Files:**
- Create: `analyze_semi_pbc_opportunities.py`
- Test: `tests/test_semi_pbc_opportunities.py`

- [ ] **Step 1: Write failing report CLI test**

Invoke `analyze_semi_pbc_opportunities.py --pbc toy.pbc --k 2 --out report.json` and assert the output JSON has `format == "semi-pbc-opportunity-report"`.

- [ ] **Step 2: Run report CLI test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_opportunities.py -v`

Expected: FAIL because the script is missing.

- [ ] **Step 3: Implement report CLI**

Use `argparse`, `analyze_pbc_file_opportunities`, and atomic JSON writing.

- [ ] **Step 4: Run report tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_semi_pbc_opportunities.py -v`

Expected: PASS.

## Chunk 4: Verification and Bench Checks

### Task 7: Final Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_semi_pbc_optimizer.py tests/test_semi_pbc_pipeline.py tests/test_compile_semi_pbc_cli.py tests/test_semi_pbc_opportunities.py -v
```

Expected: PASS.

- [ ] **Step 2: Run lint**

Run:

```bash
.venv/bin/python -m ruff check ftcircuitbench/semi_pbc compile_semi_pbc.py analyze_semi_pbc_opportunities.py tests/test_semi_pbc_optimizer.py tests/test_semi_pbc_pipeline.py tests/test_compile_semi_pbc_cli.py tests/test_semi_pbc_opportunities.py
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS with only known optional-dependency skips.

- [ ] **Step 4: Compare adder/QFT/toy results**

Run current adder and QFT samples plus toy cases with `none`, `local-window`, and `rotation-dp`, and write disposable outputs under `/tmp/ftc_semi_pbc_rotation_dp_check`.

- [ ] **Step 5: Inspect diff and commit**

Run:

```bash
git diff --check
git status --short
git add <intended files>
git commit -m "feat: add semi-PBC rotation DP optimizer"
git push fork semi-pbc-compiler
```
