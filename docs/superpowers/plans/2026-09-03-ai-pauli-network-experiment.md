# AI Pauli-Network Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small optional harness for testing Qiskit's AI Pauli-network synthesis pass against toy Pauli-network circuits.

**Architecture:** Add one optional adapter module under `ftcircuitbench/semi_pbc/` and one CLI script at repo root. The adapter hides imports of `qiskit_ibm_transpiler` so normal FTCircuitBench usage does not require that package.

**Tech Stack:** Python 3.9+, Qiskit 2.0.2, optional `qiskit-ibm-transpiler`, pytest, ruff.

---

## Chunk 1: Optional Adapter And CLI

### Task 1: Dependency Detection

**Files:**
- Create: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Test: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write failing tests**

Test that `ai_pauli_network_dependency_status()` returns a dictionary with
`available`, `qiskit_version`, and `qiskit_ibm_transpiler_version`.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ai_pauli_network_experiment.py -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement optional dependency detection**

Add the module and use `importlib.util.find_spec` plus defensive imports.

- [ ] **Step 4: Re-run focused tests**

Expected: tests pass even when `qiskit_ibm_transpiler` is missing.

### Task 2: Toy Circuit Builder

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Test: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write failing tests**

Test that `build_pauli_network_circuit()` converts signed Pauli strings into a
Qiskit circuit with one rotation per input term.

- [ ] **Step 2: Confirm test failure**

Expected: function missing.

- [ ] **Step 3: Implement minimal builder**

Use basis changes, CX parity computation, `rz(pi/4)` for `+` and `rz(-pi/4)`
for `-`, then uncompute.

- [ ] **Step 4: Re-run focused tests**

Expected: tests pass.

### Task 3: AI Synthesis Runner

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Create: `try_ai_pauli_network_synthesis.py`
- Test: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write failing tests**

Test that metric collection works and that the runner returns a skipped result
when the optional dependency is missing.

- [ ] **Step 2: Confirm test failure**

Expected: function missing.

- [ ] **Step 3: Implement runner and CLI**

The runner should import `CollectPauliNetworks` and `AIPauliNetworkSynthesis`
only inside the function. The CLI should write JSON output and never fail just
because the optional dependency is absent.

- [ ] **Step 4: Re-run focused tests**

Expected: tests pass.

### Task 4: Optional Local Package Smoke

**Files:**
- No required source changes.

- [ ] **Step 1: Install optional package if approved**

Run:

```bash
.venv/bin/python -m pip install qiskit-ibm-transpiler
```

- [ ] **Step 2: Run the smoke CLI**

Run:

```bash
.venv/bin/python try_ai_pauli_network_synthesis.py --out /tmp/ai-pauli-smoke.json
```

Expected: JSON report with before/after metrics, or a clean skipped result if
the package cannot run in this environment.

- [ ] **Step 3: Run verification**

Run focused tests, ruff, and the smoke CLI.
