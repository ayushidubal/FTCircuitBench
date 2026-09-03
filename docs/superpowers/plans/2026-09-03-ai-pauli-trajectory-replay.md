# AI Pauli Trajectory Replay Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Option B scratch-probe claims into committed replay/truncation helpers and tests.

**Architecture:** Extend the optional AI Pauli adapter with pure-Python helpers that can decode/replay IBM Pauli-network solver trajectories without requiring `qiskit_ibm_transpiler` at import time. Keep this as experiment infrastructure first: prove conventions and k-terminal detection before integrating it into semi-PBC emission.

**Tech Stack:** Python 3.9+, Qiskit quantum-info Clifford/Pauli, pytest.

---

## Chunk 1: Replay Helpers And Tests

### Task 1: Decode IBM Pauli-Network Solution Markers

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Modify: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write a failing test**

Test that local marker decoding matches the known IBM encoding for one gate and one rotation.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_ai_pauli_network_experiment.py::test_decode_ai_pauli_solution_decodes_gate_and_rotation_markers -q
```

Expected: import failure because the helper does not exist.

- [ ] **Step 3: Implement minimal decoder**

Add `decode_ai_pauli_solution()` using the marker layout documented by `qiskit_gym.envs.synthesis.decode_pauli_solution`.

- [ ] **Step 4: Verify the test passes**

Run the same focused test.

### Task 2: Replay Gate Actions And Validate Rotation Emissions

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Modify: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write a failing test**

Use a fixed 2q trajectory where a CX action reduces `+ZZ` to a single-qubit `+IZ` before an `rz` emission.

- [ ] **Step 2: Run the focused test and verify it fails**

Expected: replay helper missing.

- [ ] **Step 3: Implement minimal replay**

Add `replay_ai_pauli_solution()` that evolves pending Paulis through decoded gate actions with reversed CX convention and `Pauli.evolve(..., frame="s")`, validates emitted rotation axis/qubit/sign, and records pending weights after every step.

- [ ] **Step 4: Verify the test passes**

Run the focused test.

### Task 3: Detect k-Terminal Prefixes

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Modify: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write a failing test**

Use a fixed trajectory where all pending terms become weight `<= 1` before the full trajectory ends.

- [ ] **Step 2: Run the focused test and verify it fails**

Expected: k-terminal helper missing or incorrect.

- [ ] **Step 3: Implement prefix detection**

Add `find_k_terminal_prefix()` over replay snapshots. Return the earliest step index and pending terms at that point.

- [ ] **Step 4: Verify the test passes**

Run the focused test.

### Task 4: Verification

**Files:**
- No additional source changes.

- [ ] **Step 1: Run focused AI tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pauli_network_experiment.py
```

- [ ] **Step 2: Run relevant semi-PBC tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pauli_network_experiment.py tests/test_semi_pbc_pipeline.py tests/test_semi_pbc_lowering.py
```

- [ ] **Step 3: Report remaining limitations**

Explicitly state that this proves replay mechanics on fixed toy trajectories, not full IBM-solver extraction on FTCircuitBench windows.

## Chunk 2: Real Solver Trajectory Extraction

### Task 5: Raw Action Extraction

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Modify: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write failing tests**

Test that the extractor skips cleanly when optional AI dependencies are missing and, with a fake model repository, calls `algorithm.solve(...)` and returns raw/decoded actions.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ai_pauli_network_experiment.py::test_extract_ai_pauli_trajectory_skips_when_dependency_missing tests/test_ai_pauli_network_experiment.py::test_extract_ai_pauli_trajectory_uses_model_algorithm_actions -q
```

Expected: import failure because the extractor does not exist.

- [ ] **Step 3: Implement minimal extractor**

Mirror IBM local synthesis model selection: ensure local Pauli-network models are loaded, hash the requested coupling map, prepare the input circuit using `AILocalPauliNetworkSynthesis._prepare_input`, call `model.env.get_state(prepared_input)`, then call `model.algorithm.solve(...)`.

- [ ] **Step 4: Verify focused tests pass**

Run the same focused tests.

### Task 6: k-Terminal Window Analysis

**Files:**
- Modify: `ftcircuitbench/semi_pbc/ai_pauli_network.py`
- Modify: `tests/test_ai_pauli_network_experiment.py`

- [ ] **Step 1: Write failing tests**

Test that a window trajectory analysis reports full trajectory length, k-terminal prefix length, pending terms, and pending weights.

- [ ] **Step 2: Run tests to verify failure**

Expected: analyzer missing.

- [ ] **Step 3: Implement minimal analyzer**

Compose existing window circuit construction, raw trajectory extraction, replay, and `find_k_terminal_prefix`.

- [ ] **Step 4: Verify focused tests pass**

Run the focused analyzer test.
