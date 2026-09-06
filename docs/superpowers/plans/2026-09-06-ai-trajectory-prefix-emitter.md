# AI Trajectory-Prefix Emitter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `ai-trajectory-prefix` semi-PBC compiler mode and richer benchmark outcome categories.

**Architecture:** Reuse the existing `SemiPBCOp` JSONL IR and pipeline lowering flow. Convert accepted AI trajectory prefixes into Clifford operations plus emitted/pending capped `t_pauli` operations, falling back to existing lowering when AI does not produce a usable prefix.

**Tech Stack:** Python, Qiskit optional AI stack, pytest, existing FTCircuitBench semi-PBC IR.

---

## Chunk 1: Rotation Window Emitter

### Task 1: Add Tests

**Files:**
- Modify: `tests/test_semi_pbc_pipeline.py`
- Modify: `tests/test_ai_pauli_network_experiment.py`

- [ ] Write a failing test showing `optimization="ai-trajectory-prefix"` emits a Clifford prefix and capped `t_pauli` tail from a mocked trajectory result.
- [ ] Write a failing equivalence test for a small rotation-only circuit.
- [ ] Write a failing test showing AI failure falls back to existing local lowering.
- [ ] Write a failing benchmark test for `result_category`.

### Task 2: Implement Emitter

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Modify: `ftcircuitbench/semi_pbc/pipeline.py`
- Modify: `benchmark_ai_pauli_network_synthesis.py`

- [ ] Add a helper that converts a replay prefix into `SemiPBCOp` records.
- [ ] Add `ai-trajectory-prefix` as an accepted pipeline optimization.
- [ ] Attempt AI replacement for high-weight rotation runs/windows and fall back on failure.
- [ ] Add benchmark result categorization from solver status and NaN diagnostics.

### Task 3: Verify

- [ ] Run targeted pytest files.
- [ ] Run full `.venv` pytest.
- [ ] Run full `.venv-ai` pytest.
- [ ] Run ruff.
- [ ] Commit and push.
