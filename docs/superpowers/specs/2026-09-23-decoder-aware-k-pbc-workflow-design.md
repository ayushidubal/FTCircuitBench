# Decoder-Aware k-PBC Workflow Design

**Date:** 2026-09-23
**Status:** Approved for implementation planning
**Scope:** Direct C+T to k-PBC candidate generation, compulsory routing, and decoder-aware tracegen evaluation. This spec defines the workflow and terminology before implementation.

## 1. Goal

Build an experimental pipeline that evaluates bounded-support PBC candidates using routed lattice-surgery geometry and decoder-specific latency models.

The core workflow is:

```text
C+T input
  -> k-PBC candidate generation
  -> route every quantum operation
  -> tracegen with decoder-specific timing
  -> cost, failure, and routed-footprint reports
```

The first implementation should keep compiler and router decisions decoder-neutral. Tracegen should be decoder-aware and should use fitted decoder timing models when assigning operation durations and building the timed trace.

## 2. Motivation

The current semi-PBC work uses `k` as a hard cap on exposed Pauli support. That is a useful compiler control, but it is not the physical quantity that a decoder sees.

A logical Pauli of small support can route poorly if its qubits are far apart, while a larger support can be compact if the participating patches are nearby. The decoder-cost variable should therefore be derived after routing.

This spec separates the two ideas:

```text
k = logical support cap used by the compiler to generate candidates
L = routed patch path length or routed footprint used by decoder models
```

The compiler emits candidates. The router maps candidates to geometry. Tracegen applies decoder-specific timing to the routed geometry.

## 3. Definitions

### 3.1 Compiler Support Cap k

`k` is the maximum logical Pauli support exposed by generated PBC operations.

For a `k-PBC` candidate:

- every `k-T_Pauli` operation has logical support at most `k`;
- every `k-M_Pauli` operation has logical support at most `k`;
- Clifford operations remain explicit and are not counted as Pauli support.

`k` is a candidate-generation knob. It is not a decoder latency variable.

### 3.2 Routed Length L

`L` is the primary routed geometry variable passed to the decoder model. Initially, use `L` as routed path length.

The pipeline may also record a small set of independent routed features when they are needed by a decoder model:

- routed area or active patch count;
- syndrome-round count;
- routed spacetime volume;
- window count or communication hops for distributed decoders.

Do not store large feature sets by default. Store only geometry quantities that are independent enough to support fitting or debugging.

### 3.3 Code Distance d

`d` is fixed for the initial experiments. Co-optimizing `d` with compiler mode or support cap is future work.

### 3.4 Decoder Model T_LS_D

For decoder `D`, tracegen uses a fitted model:

```text
T_LS_D = T0_D + alpha_D * f_D(L, d, routed_features) + T_comm_D(L, d, routed_features)
```

`T_comm_D` is decoder-specific. It may be nearly constant for a local serial model, but may depend on `L`, `d`, windowing, network hops, or geometry for distributed decoders.

## 4. Gate Set

The routed candidate gate set is:

```text
H
S
Sdg
CNOT
k-T_Pauli
k-M_Pauli
```

Do not add more primitive Clifford gates unless the router requires them. `H`, `S`, `Sdg`, and `CNOT` are sufficient for the intended Clifford structure. Other Clifford operations such as `X`, `Z`, `CZ`, and `SWAP` should be macros or router-internal operations unless a downstream interface requires them explicitly.

The candidate format must preserve classical dependencies and measurement-result postprocessing, even though those are not routed quantum operations.

## 5. Operation Semantics

### 5.1 k-T_Pauli

`k-T_Pauli` represents a signed `pi/8` Pauli rotation whose logical support is in `[1,k]`.

The implementation must define one unambiguous angle/sign convention for:

- `T` versus `Tdg`;
- positive versus negative Pauli terms;
- round-tripping from C+T to k-PBC and back to semantic checks.

The existing semi-PBC convention may be reused if it cleanly covers this path:

```text
angle_num = 1
angle_den = 8
sign in {+1, -1}
```

### 5.2 k-M_Pauli

`k-M_Pauli` represents a signed Pauli measurement whose logical support is in `[1,k]`.

The result bit convention must be explicit:

```text
bit 0 -> measured signed Pauli eigenvalue +1
bit 1 -> measured signed Pauli eigenvalue -1
```

Classical parity corrections, sign corrections, and source-result reconstruction should be represented as classical metadata or `xor`-style dependencies so tracegen can model feed-forward correctly.

## 6. Candidate Generation Modes

The workflow supports multiple compiler modes. The first comparison should include only modes that are active and interpretable. Deprecated AI and rotation-DP paths should not be part of the public comparison set.

### 6.1 Naive Baseline

The existing baseline is:

```text
C+T -> full PBC -> capped semi-PBC or k-PBC lowering
```

This path remains useful as a control because it starts from a full PBC representative and repairs operations that exceed the cap.

### 6.2 Segmented Capped Litinski

The new direct path is:

```text
C+T -> segmented k-PBC
```

It should follow the Litinski-style idea of propagating T-basis Paulis through a Clifford frame, but stop a segment before propagation would produce an invalid candidate.

The first implementation should use a conservative whole-segment cut:

1. Scan the C+T circuit through the Clifford frame.
2. Track pending Pauli rotations created by T/Tdg gates.
3. Before applying the next Clifford update, compute the prospective pending Pauli supports.
4. If all pending supports remain `<= k`, accept the update.
5. If any pending support would exceed `k`, close the current segment, emit the bounded PBC operations for that segment, keep the boundary Clifford explicitly, and start the next segment.

The pass should not flush individual offending Paulis while leaving other pending Paulis in the growing frame. Subset flushing is future work because it requires additional ordering and commutation proofs.

This mode should satisfy:

```text
k = 1 approximates an explicit Clifford+T-like representation
k = n can recover the full Litinski-style PBC behavior
middle k values expose bounded PBC blocks
```

## 7. Routing

Routing is compulsory and independent of the compiler pass.

For each emitted candidate:

```text
route every operation:
    H
    S
    Sdg
    CNOT
    k-T_Pauli
    k-M_Pauli
```

Routing produces the geometry that tracegen consumes. The router should emit enough information to recover:

- operation identity and source candidate operation;
- routed path length `L`;
- any additional routed features required by decoder models;
- resource conflicts or dependency edges used by tracegen;
- routing failures with explicit reasons.

Routing success must not be assumed from logical support. A support-`k` Pauli can still route badly.

## 8. Tracegen

Tracegen is decoder-aware.

Given a routed candidate and decoder model `T_LS_D`, tracegen should:

1. build the routed lattice-surgery operation trace;
2. assign decoder-aware durations using `T_LS_D`;
3. preserve feed-forward dependencies from measurements and classical postprocessing;
4. schedule operations under tracegen's resource and dependency model;
5. report critical-path latency and other cost metrics.

The first experiments should keep the compiler and router decoder-neutral, but run tracegen separately for each decoder model. This allows the same routed geometry to be evaluated under different decoder timing assumptions.

Later phases may allow decoder-aware routing or decoder-aware compilation. Those are intentionally out of scope for the first workflow.

## 9. Decoder Calibration

Before evaluating compiler candidates, fit `T_LS_D` for each decoder.

Calibration should use routed lattice-surgery jobs, not abstract Pauli weights. Sweep the variables the decoder model will actually receive:

- routed length `L`;
- code distance `d`;
- geometry features needed by that decoder;
- physical error rate `p`, if the decoder latency depends on it.

For each decoder, fit:

- mean latency;
- p50 latency;
- p95 latency;
- p99 latency.

Start with an interpretable model:

```text
T_LS_D = T0_D + alpha_D * L^beta_L * d^beta_d + T_comm_D(features)
```

Then use regression to improve calibration when the interpretable model leaves structured residuals. Suitable first regression tools include log-linear regression, ridge or lasso regression, and gradient-boosted trees. The interpretable fit should remain the main reported model unless the regression model is clearly needed and documented.

Every fitted model must record its validity range. Tracegen should flag any routed operation whose `(L, d, features)` falls outside the calibrated range.

## 10. Experiment Matrix

The evaluation loop is:

```text
for each decoder D:
    for each compiler mode:
        for each support cap k:
            emit k-PBC candidate
            route every Clifford and Pauli operation
            run tracegen with T_LS_D
            report cost, failures, and routed footprint distribution
```

Use fixed `d` initially.

For `k`, include:

- `k = 1`;
- `k = n`, where `n` is the circuit data-qubit count;
- selected middle values.

For small circuits, running all `k in [1,n]` is acceptable. For large circuits, use a sparse set until the pipeline cost is clear.

## 11. Metrics and Reports

Reports should keep logically distinct costs separate.

Required summary fields:

- circuit name and family;
- decoder name and decoder model version;
- compiler mode;
- support cap `k`;
- code distance `d`;
- emitted operation counts by gate type;
- routed operation counts by gate type;
- routing success or failure;
- tracegen success or failure;
- critical-path decoder-aware latency;
- total scheduled trace duration;
- total decoder work proxy, if available;
- routed `L` distribution;
- routed feature distribution for any non-`L` variables used by `T_LS_D`;
- model-extrapolation count.

Failure classes should be reported separately:

- candidate-generation failure;
- unsupported gate or dependency pattern;
- routing failure;
- tracegen resource conflict or schedule failure;
- decoder-model extrapolation;
- decoder-model evaluation failure.

Do not silently drop failed cases from averages.

## 12. Artifacts

Each experiment row should preserve enough data to debug and reproduce the result on the server:

- emitted k-PBC candidate;
- routing output;
- tracegen trace;
- decoder model configuration and fitted parameters;
- summary JSON;
- logs.

Router seeds, placement settings, and any randomized choices must be recorded. If a run uses deterministic settings, the artifact should still record that fact.

## 13. Tooling and Execution Boundaries

This workflow spans multiple tools and repositories. The implementation should not assume all code lives in this repository.

Known local tool roots at the time of this spec include:

- `FTCircuitBench-semi-pbc` for candidate generation and experiment orchestration;
- `tracegen` for routed trace extraction and decoder-aware timing;
- `FastLS` for lattice-surgery simulation or cost tooling;
- `FastMQLSS` for measurement/lattice-surgery related tooling;
- `fastkPBC`, to be built as the router for the `H/S/Sdg/CNOT/k-T_Pauli/k-M_Pauli` gateset.

Code should be written and tested on this machine, but the final 95-circuit execution at each major stage will run on the server. Server results will be handed back into this workflow step by step, so local code and server artifacts must stay easy to match.

Each tool should keep its own source changes in its own directory. Cross-tool interfaces should be explicit file formats or command-line contracts rather than hidden in ad hoc local state. When a stage consumes artifacts from another tool, it should record the producing tool version, command, input paths, and output paths.

Local validation should use smoke tests and small representative circuits. Server validation should run the full 95-circuit suite and return summaries, logs, and failed-case artifacts for follow-up.

## 14. Non-Goals

This spec does not require:

- implementing all future compiler passes;
- decoder-aware compiler candidate generation;
- decoder-aware routing;
- co-optimizing code distance `d`;
- subset flushing in segmented capped Litinski;
- public comparison against deprecated AI or rotation-DP paths;
- adding more primitive Clifford gates beyond the router's required interface.

## 15. First Implementation Scope

The first implementation should be deliberately narrow:

1. Define or reuse a k-PBC candidate format with the gateset in this spec.
2. Add a direct segmented capped Litinski candidate generator from C+T.
3. Keep the existing naive path as the baseline.
4. Route all candidate operations through the external router interface.
5. Run tracegen with a pluggable decoder timing model.
6. Produce server-friendly artifacts and summaries.

The first implementation should not try to optimize every segment boundary. Correctness, reproducibility, and comparable routed traces come first.
