# Semi-PBC Compiler Design

**Date:** 2026-08-10
**Status:** Approved for implementation planning
**Scope:** Phase 1 capped-weight PBC -> semi-PBC generation. Direct C+T -> semi-PBC and `dascot-rs` semi-PBC routing are future specs.

## 1. Goal

Build a compiler path that produces circuits equivalent to the source computation while enforcing a hard Pauli-block weight cap `k`.

The target gate set is:

- 1-qubit Clifford gates
- 2-qubit Clifford gates
- `[1,k]`-weight `pi/8` Pauli rotations
- `[1,k]`-weight Pauli measurements

The first implementation should support existing `nwqec`-generated PBC files. A later implementation should compile directly from Clifford+T to semi-PBC, because direct compilation can preserve structure that flat PBC files have already erased.

The implementation plan produced from this spec must cover Phase 1 only:

```text
nwqec PBC text -> deterministic capped semi-PBC IR + validation reports
```

## 2. Motivation

Pure PBC can produce high-weight logical Pauli rotations and measurements. Those operations are compact at the IR level, but difficult for routing and decoding because the merged patch size grows with Pauli weight.

Pure Clifford+T avoids high-weight Pauli blocks, but may over-expand structure that a router could handle more efficiently as bounded Pauli blocks.

Semi-PBC is the intended middle ground: keep bounded Pauli blocks when useful, but lower any operation whose weight would exceed the router/decoder-friendly cap.

## 3. Design Constraints

The compiler must satisfy:

1. **Equivalence:** The emitted semi-PBC circuit must be equivalent to the input computation, including measurement semantics and classical sign/outcome updates.
2. **Hard weight cap:** No emitted Pauli rotation or Pauli measurement may have support weight greater than `k`.
3. **Ancillas allowed:** The compiler may allocate ancillas to lower high-weight operations.
4. **Depth objective:** Optimize latency-weighted depth. During compiler scheduling, legal Pauli blocks have constant latency by op type. Routed decoder-aware timing may later use weight-dependent timing.
5. **Compact IR:** The semi-PBC IR should be quick to generate and read, and should not duplicate large full-width Pauli strings unless required by a downstream converter.
6. **Router compatibility:** The design must support a converter to the router-specific format. Existing fast `dascot-rs` support is C+T-shaped, so semi-PBC routing will need either a lowering mode or a router extension.

## 4. Literature Findings

### 4.1 Peres/Galvao 2025

[Peres and Galvao 2025](https://arxiv.org/abs/2408.04007) is the most directly relevant PBC source.

The method we should use first is their greedy measurement weight-reduction algorithm. If a PBC has already measured compatible Paulis `P1, ..., P_{r-1}`, then a later measurement `Pr` can be replaced by an equivalent representative:

```text
Pr * product(previous measured Paulis)
```

with classical correction from the previous outcomes. This is valid because multiplying by known stabilizers does not change the projected eigenspace. The algorithm searches candidate products and chooses a lower-weight representative.

This is useful as a safe preprocessing pass, but it is not a hard-cap mechanism. If no stabilizer-product representative has weight `<= k`, an explicit lowering gadget is still required.

The implementation must also be conservative about when this reducer is legal. Existing `nwqec` PBC files can contain both `t_pauli` rotations and `m_pauli` measurements. A previous measurement outcome can only be used as a stabilizer representative if the compiler can prove that intervening operations preserve the needed Pauli-stabilizer relation. If an intervening non-Clifford rotation could invalidate that relation, the reducer must skip that previous measurement or split the reduction into a smaller valid segment.

Their `incPBC` construction is also important because it gives a universal model with Pauli measurements of weight at most 2, but it is closer to a different computation model than to a post-pass over existing flat PBC files.

### 4.2 Peres/Galvao 2023

[Peres and Galvao 2023](https://quantum-journal.org/papers/q-2023-10-03-1126/) gives the practical Clifford+T-to-PBC construction. It is useful for the later direct C+T -> semi-PBC path because it exposes the binary symplectic machinery, dependence checks, and measurement back-propagation logic.

It does not impose a hard Pauli weight cap.

The companion repository [CompHybPBC](https://github.com/fcrperes/CompHybPBC) is useful as an implementation reference for the greedy algorithm and PBC side processing, but it is not a reusable semi-PBC file emitter.

### 4.3 Catalyst PBC

[Catalyst PBC passes](https://docs.pennylane.ai/projects/catalyst/en/latest/code/dialects/PBC/PBCPasses.html) provide a useful practical model:

- PPR operations for Pauli product rotations
- PPM operations for Pauli product measurements
- a `max-pauli-size` option in some passes
- `pbc.layer` regions for commuting/disjoint PBC layers

Catalyst's `max-pauli-size` is not our hard cap. It is a rewrite guard: if a commutation or merge would exceed the cap, Catalyst skips that rewrite. It does not split or lower existing high-weight PPR/PPM operations and does not guarantee all final Pauli blocks are bounded by `k`.

The useful part for us is the pass structure: bounded rewrites, explicit layers, and clean Pauli algebra utilities.

### 4.4 Pauli Rotation Lowering

[Moflic and Paler 2026](https://www.nature.com/articles/s41534-026-01226-x) gives a depth-oriented construction for Pauli exponentials using two-body interactions and `O(r)` ancillas. This is a strong candidate for optimized high-weight rotation lowering, especially for `k = 2`.

For a first correct implementation, the baseline lowering for a high-weight `pi/8` Pauli rotation should be a bounded parity-network construction that keeps the largest legal Pauli block:

1. Apply 1-qubit Cliffords to map the Pauli word to a Z-parity.
2. Keep `k` active qubits as the retained legal block.
3. Use CNOTs to coherently fold every remaining active qubit into one retained target qubit.
4. Apply one `pi/8` Pauli rotation on the retained weight-`k` Z block.
5. Uncompute the parity and undo basis changes.

This is exact and legal for any `k >= 1`. For an input weight `r > k`, it emits one weight-`k` Pauli rotation rather than always collapsing to weight 1. It may not be depth-optimal.

### 4.5 Pauli Measurement Lowering

For high-weight Pauli measurements, direct chunk measurement is not generally equivalent because it reveals extra partial-parity information.

The safe baseline is coherent parity compression followed by one bounded joint Pauli measurement:

1. Apply basis-change Cliffords on data qubits.
2. Keep `k` active qubits as the retained legal measurement block.
3. Use CNOTs to fold every remaining active qubit into one retained target qubit.
4. Measure the retained weight-`k` Z block as a single joint Pauli measurement.
5. Uncompute the CNOTs and undo basis changes.
6. Use an explicit classical `xor` to account for the original measurement sign.

This measures only the intended Pauli eigenvalue, does not reveal partial chunk parities, uses no ancilla in the local Phase 1 strategy, and is legal for any `k >= 1`.

Architecture-level sources such as [Parallel Logical Measurements via Quantum Code Surgery](https://arxiv.org/abs/2503.05003), [Homomorphic Logical Measurements](https://arxiv.org/abs/2211.03625), and [Extractors](https://arxiv.org/abs/2503.10390) are useful for future routing and layout choices, but they should not be the first compiler lowering algorithm.

## 5. Non-Uniqueness

Semi-PBC circuits are not unique. This is not a bug; it is a core property of the representation.

Even pure PBC is not mathematically unique. A fixed tool such as `nwqec.to_pbc(...); fuse_t(...)` may deterministically emit one PBC, but many equivalent PBCs exist because of:

- different Clifford propagation orders
- different T-gadget and fusion choices
- multiplying later measured Paulis by earlier measured Paulis
- different commutation and layering choices
- different ancilla gadgets
- different choices about what remains Clifford versus what is absorbed into Pauli blocks

The compiler should therefore be deterministic for a fixed configuration, but it should not claim to produce a canonical semi-PBC circuit.

This affects the two approaches differently:

- **PBC -> semi-PBC:** starts from one existing `nwqec` representative and performs local improvements plus hard-cap lowering. This is easier and immediately useful for the 95 existing PBC circuits, but it may miss better semi-PBC circuits available from other equivalent PBC representatives.
- **C+T -> semi-PBC:** has more freedom to preserve Clifford structure, choose bounded Pauli blocks earlier, and optimize depth/weight tradeoffs. It is likely to produce better circuits, but it is a larger optimization problem.

## 6. Architecture

The compiler should be structured as a small pass pipeline over an internal sparse semi-PBC IR.

```text
input
  -> parser/front-end
  -> sparse semi-PBC IR
  -> Pauli algebra simplification
  -> optional Peres/Galvao measurement reducer
  -> hard-cap lowering
  -> conservative sequential scheduling/reporting
  -> emit compact IR
  -> optional router-format converter
```

### 6.1 Internal IR

Phase 1 should emit a canonical line-oriented JSONL IR, with an optional human-readable text emitter later. JSONL is less pretty, but it gives parser, tests, and router converters a concrete schema.

The first line is a header:

```json
{"format":"semi-pbc","version":1,"k":4,"data_qubits":10}
```

Each following line is one operation. IDs are monotonically increasing integers. Data qubits are named `q0`, `q1`, ... from the input PBC width. Ancillas are named `a0`, `a1`, ... in allocation order. Classical bits are named `c0`, `c1`, ... in creation order. Source measurement results may also be named `src<N>` through `xor` operations.

Ancilla names are logical allocation IDs, not a direct physical resource count. Phase 1 emits a fresh `aN` for each allocation to keep lifetimes and provenance unambiguous. Resource budgeting is based on peak simultaneously live ancillas, so sequential gadgets that allocate `a0`, release it, then allocate `a1` require one live ancilla even though they use two logical allocation IDs.

Pauli terms are sparse arrays. Identity factors are omitted. Duplicate qubit entries in a term are invalid.

```text
terms: [["q3","X"], ["q5","Y"], ["a0","Z"]]
```

Term ordering is canonical: data qubits sort by numeric index, then ancillas sort by numeric index. Emitters must write Pauli factors in that canonical order.

Supported Phase 1 operation records:

```json
{"id":0,"op":"h","qubits":["q3"]}
{"id":1,"op":"s","qubits":["q3"]}
{"id":2,"op":"sdg","qubits":["q3"]}
{"id":3,"op":"cx","qubits":["q3","q7"]}
{"id":4,"op":"alloc","qubit":"a0","basis":"zero"}
{"id":5,"op":"release","qubit":"a0"}
{"id":6,"op":"t_pauli","sign":1,"angle_num":1,"angle_den":8,"terms":[["q3","X"],["q5","Y"]],"source_id":"line12"}
{"id":7,"op":"m_pauli","sign":-1,"terms":[["q1","Z"],["a0","Z"]],"result":"c0","source_id":"line13"}
{"id":8,"op":"xor","target":"src13","terms":["c0","c4"],"const":1}
```

For `t_pauli`, Phase 1 only supports `angle_num = 1`, `angle_den = 8`, and `sign in {1,-1}`. `sign = -1` means the inverse `pi/8` rotation.

For `m_pauli`, `result` is the physical measurement bit using the convention:

```text
bit 0 -> measured signed Pauli eigenvalue +1
bit 1 -> measured signed Pauli eigenvalue -1
```

Phase 1 lowerings should normalize emitted physical measurements to positive Pauli terms when possible, and preserve original source measurement signs through `xor.const`. The IR still supports signed `m_pauli` records so it can represent imported or future operations, but the Phase 1 compiler's own lowering path should use a uniform positive-measurement convention.

An `xor` operation is classical only. It defines:

```text
target = xor(terms) xor const
```

This is used to preserve source measurement-result semantics after Peres/Galvao replacement or sign changes in lowering gadgets.

Quantum operation signs are static in Phase 1. If an input or future pass requires a genuinely classically controlled quantum sign and it cannot be resolved into an `xor` result mapping, Phase 1 must reject that input with a clear error.

### 6.2 Optional Metadata Sidecar

The source IR should be sequential and sufficient for correctness. A sidecar may store:

- source operation IDs
- source PBC file line numbers
- gadget boundaries
- dependency edges
- commutation layers
- ancilla counts, including total logical allocations and peak live ancillas
- lowering strategy per operation
- cost-model settings

The sidecar is useful for tracegen and debugging, but should not be required by the router initially.

If Phase 1 emits a sidecar, it should be JSON with:

```json
{"format":"semi-pbc-sidecar","version":1,"k":4,"provenance":[]}
```

Each provenance record should include the output op ID, operation type, source ID when available, and gadget ID when available.

## 7. PBC -> Semi-PBC Pipeline

This is the first implementation target.

### 7.1 Input

Read existing `nwqec` PBC text files:

```text
t_pauli +XYZI...
m_pauli -IXYZ...
```

The FTCircuitBench adapter already emits this format through:

- [FTCircuitBench/ftcircuitbench/pbc_converter/nwqec_adapter.py](/Users/ayushidubal/qmem/FTCircuitBench-semi-pbc/ftcircuitbench/pbc_converter/nwqec_adapter.py)

### 7.2 Passes

Run these passes:

1. **Parse sparse PBC:** Convert full-width terms into sparse `PauliTerm` objects.
2. **Input metadata capture:** Preserve source line numbers and upstream layer sidecars when available.
3. **Measurement representative reduction:** Apply the Peres/Galvao greedy reducer to `m_pauli` operations only when prior measured Paulis are proven valid at that program point.
4. **Rotation lowering:** Any `t_pauli` with weight `> k` is lowered using a legal rotation gadget.
5. **Measurement lowering:** Any `m_pauli` with weight `> k` is lowered using a legal measurement gadget.
6. **Scheduling/reporting:** Preserve emitted operation order and compute conservative sequential latency-weighted depth. Commutation-layer reconstruction is future work.
7. **Emit:** Write compact semi-PBC IR plus optional metadata sidecar.

### 7.3 Baseline Lowering Gadgets

Phase 1 has two mandatory lowering gadgets. They are chosen for correctness and simplicity, not optimality.

#### 7.3.1 Basis Changes

For both rotation and measurement lowering, each Pauli factor is first mapped to a Z-basis parity by applying these gates before the parity network:

```text
Z: no gate
X: h q
Y: sdg q; h q
```

After the gadget, undo these basis changes in reverse:

```text
Z: no gate
X: h q
Y: h q; s q
```

This convention implements `U^dagger Z U = P` for each original Pauli factor.

#### 7.3.2 Rotation Lowering

For a high-weight operation:

```text
t_pauli sign P(q0, q1, ..., q{r-1})
```

where `r > k`, Phase 1 emits:

1. Basis changes on all active qubits.
2. A CNOT parity chain from active qubits beyond the retained block into the first retained qubit `q0`:

   ```text
   cx qk, q0
   cx q{k+1}, q0
   ...
   cx q{r-1}, q0
   ```

3. A rotation on the retained legal block:

   ```text
   t_pauli sign Z q0 Z q1 ... Z q{k-1}
   ```

4. The same CNOT chain in reverse order.
5. Inverse basis changes.

This is exact, uses no ancilla by default, and satisfies the hard cap for any `k >= 1`. When `r > k`, the emitted Pauli block has weight exactly `k`, so the compiler uses the largest legal retained block.

If an ancilla-budgeted tree strategy is added later, it must be a separate lowering option with its own equivalence tests.

#### 7.3.3 Measurement Lowering

For a source measurement operation:

```text
m_pauli sign P(q0, q1, ..., q{r-1}) -> src
```

where `r > k`, Phase 1 emits:

1. Basis changes on all active data qubits.
2. CNOTs from each active data qubit beyond the retained block into the first retained qubit `q0`:

   ```text
   cx qk, q0
   cx q{k+1}, q0
   ...
   cx q{r-1}, q0
   ```

3. A single bounded joint measurement on the retained legal block:

   ```text
   m_pauli + Z q0 Z q1 ... Z q{k-1} -> c_raw
   ```

4. The same CNOT chain in reverse order.
5. Inverse basis changes on data qubits.
6. Emit an `xor` mapping for the original source result:

   ```text
   src = c_raw xor (1 if original sign is -1 else 0)
   ```

This is the measurement analogue of the rotation lowering: it keeps the largest legal Pauli block and coherently compresses the rest into that block before measuring. It does not allocate an ancilla in Phase 1.

This measures only the intended full Pauli eigenvalue. It does not measure chunk parities, because chunk measurements would generally reveal extra information and change the computation.

For `r <= k`, Phase 1 should also normalize the physical measurement to positive sign:

```text
m_pauli + P -> c_raw
src = c_raw xor (1 if original sign is -1 else 0)
```

This convention makes source-result mapping uniform across pass-through, reduced, and lowered measurements.

### 7.4 Peres/Galvao Reducer Rules

The reducer is optional but enabled by default. It only rewrites measurements.

For a current measurement `M_r`, a prior measurement `M_j` is eligible only if:

1. `M_j` occurs before `M_r`.
2. The compiler has a physical result bit or source-result expression for `M_j`.
3. The propagated Pauli representative of `M_j` is still a Pauli stabilizer at `M_r`.
4. All intervening non-Clifford `t_pauli` rotations commute with that representative. If any intervening non-Clifford rotation anticommutes, `M_j` is ineligible in Phase 1.
5. Multiplying the candidate set produces a real signed Pauli word, not an imaginary phase.

For a selected candidate set `S`, the compiler replaces the physical measurement term by:

```text
M'_r = M_r * product(M_j for j in S)
```

and emits an `xor` expression mapping the original source result to the new physical result and the prior result bits:

```text
src_r = c_new xor c_j1 xor ... xor c_jm xor sign_adjust
```

`sign_adjust` is computed from signed Pauli multiplication, including the original signs and the replacement sign.

The implementation option `greedy-order` means:

- `0`: linear pass considering only the most recent eligible prior measurement
- `1`: consider all single eligible prior measurements
- `2`: consider all pairs of eligible prior measurements

Candidate tie-breaking is deterministic:

1. prefer candidates with smaller support weight
2. then prefer fewer prior measurements in the product
3. then prefer lexicographically smaller prior source IDs
4. then prefer lexicographically smaller canonical Pauli terms

For all modes, accept a replacement only if it strictly reduces the physical support weight. This includes the hard-cap case: a replacement is useful if it changes an operation from weight `> k` to weight `<= k`, or otherwise reduces the later lowering cost. If no safe improvement exists, leave the measurement unchanged for the lowering pass.

### 7.5 Strategy Parameters

Expose at least:

```text
--k
--ancilla-budget unlimited|N
--objective latency-depth
--measurement-reducer none|peres-galvao-greedy
--greedy-order 0|1|2
--rotation-lowering parity-network
--measurement-lowering coherent-parity
--emit-sidecar
```

The default should be conservative:

```text
measurement-reducer = peres-galvao-greedy
greedy-order = 1
rotation-lowering = parity-network
measurement-lowering = coherent-parity
emit-sidecar = true
```

The Phase 1 `"coherent-parity"` measurement lowering is data-qubit parity compression into a bounded joint measurement block. It is not an ancilla-only single-qubit extraction strategy, although such a strategy can be added later as a separate option.

`moflic-paler-k2` is not part of Phase 1. It should be specified separately after the baseline compiler is validated.

`--ancilla-budget N` constrains the maximum number of simultaneously live ancillas, not the total number of logical `aN` allocation IDs emitted over the program. The summary should report both:

- `ancilla_count`: total unique logical ancilla allocation IDs in the emitted IR
- `max_live_ancillas`: peak simultaneously live ancillas, which is the value checked against a finite budget

Error behavior:

- reject `k < 1`
- reject unsupported Pauli signs or rotation angles
- reject malformed input terms and duplicate non-identity terms on one qubit
- reject finite `--ancilla-budget` if the selected lowering's peak live ancilla requirement would exceed it
- fail after lowering if any emitted `t_pauli` or `m_pauli` still has weight `> k`
- reject incompatible options instead of silently changing strategy

## 8. Future C+T -> Semi-PBC Pipeline

This is not part of the Phase 1 implementation plan. It should get a separate design or plan after PBC -> semi-PBC is validated.

The direct compiler should not simply run C+T -> PBC -> semi-PBC. It should use the same lowerings and cost model, but make decisions earlier:

1. Parse Clifford+T.
2. Track Clifford frame / symplectic tableau.
3. Identify T-gadget and measurement dependencies.
4. Decide whether Clifford structure should remain as 1q/2q Cliffords or be absorbed into bounded Pauli blocks.
5. Preserve commutation layers from the source and from tableau dependence analysis.
6. Emit semi-PBC directly.

This path should produce better depth/weight tradeoffs than post-processing flat PBC, but it needs more careful validation.

## 9. Future Router Integration

This section is context for why the IR must be compact and converter-friendly. It is not part of the Phase 1 implementation plan, except for preserving enough metadata to make later conversion straightforward.

Existing `dascot-rs` is fast for C+T-shaped circuits and currently represents operations as `T`, `TDG`, and `CX`. Existing Amaro/QMR `mqlss` is the semantic reference for PBC-style routing, but it has not been fast enough for all 95 FTCircuitBench circuits.

The practical routing plan is:

1. **Short-term:** Emit semi-PBC and a fully lowered fallback form for small local correctness/routing smoke tests. The fallback may lower high-weight Pauli structure into Cliffords plus bounded Pauli rotations/measurements; it is for validation and format integration, not the final performance target.
2. **Medium-term:** Extend `dascot-rs` with bounded `PauliRot` and `PauliMeasurement` operation variants.
3. **Use Amaro/MQLSS as reference:** Port the useful bounded Pauli routing primitive into `dascot-rs` rather than writing a separate solver from scratch.

The router converter should consume the compact IR and produce the exact format required by the chosen backend. It should not require tracegen to infer gadget boundaries from scratch.

## 10. Validation

Validation should be staged:

1. **Parser tests:** Round-trip simple PBC and semi-PBC files.
2. **Pauli algebra tests:** Verify multiplication, commutation, signs, and support weights.
3. **Peres/Galvao reducer tests:** Verify that replacement representatives preserve measurement outcome relations, and that the reducer refuses prior measurements whose stabilizer validity cannot be proven across intervening operations.
4. **Hard-cap tests:** Assert every emitted `t_pauli` and `m_pauli` has weight `<= k`.
5. **Gadget equivalence tests:** For small random terms, compare original high-weight operation against lowered semi-PBC using statevector or stabilizer simulation where applicable.
6. **Pipeline smoke tests:** Convert small FTCircuitBench PBC files for several `k` values.
7. **Cost-model tests:** Verify deterministic scheduling and latency-weighted depth accounting.

The default Phase 1 depth report should use:

```text
1q Clifford: 1
2q Clifford: 1
t_pauli: 1
m_pauli: 1
alloc/release/reset: 0
classical xor: 0
```

These constants are only for compiler-side latency-weighted depth summaries. Phase 1 uses conservative sequential depth. Routed decoder-aware timing and commutation-layer scheduling are later models.

## 11. Phase 1 Acceptance Criteria

Given an existing `nwqec` PBC file and a valid `k`, Phase 1 is complete when it can:

1. Parse the PBC file into sparse operations.
2. Emit deterministic semi-PBC JSONL.
3. Prove by inspection that every emitted Pauli rotation and measurement has weight `<= k`.
4. Preserve source measurement semantics through explicit physical result bits and `xor` mappings.
5. Run small equivalence tests for high-weight rotation and measurement lowerings.
6. Run guarded reducer tests showing both accepted safe replacements and rejected unsafe candidates.
7. Produce a summary report with input op count, output op count, max input weight, max output weight, total logical ancilla count, peak live ancilla count, and latency-weighted depth.

## 12. Non-Goals

This design does not include:

- changing `nwqec` internals in the first implementation
- guaranteeing globally optimal semi-PBC circuits
- claiming semi-PBC output is canonical
- routing all 95 semi-PBC circuits before the `dascot-rs` extension exists
- implementing architecture-specific code-surgery or extractor layouts in the compiler pass
- implementing direct C+T -> semi-PBC in Phase 1
- extending `dascot-rs` in Phase 1

## 13. Open Research Questions

The implementation should leave room for:

1. Depth- and routing-optimized high-weight measurement lowerings, including ancilla-assisted trees and layout-aware retained-block choices.
2. Global `k > 2` strategies that choose retained blocks and reducer representatives to reduce CNOT count, depth, and routing cost.
3. A direct C+T -> semi-PBC optimizer that searches over Clifford-retention versus Pauli-block absorption.
4. A router-aware cost model that accounts for decoder latency after placement and routing.

## 14. Recommended Phase 1 Implementation Order

1. Define the sparse semi-PBC IR and parser/emitter.
2. Implement Pauli algebra utilities.
3. Implement PBC -> semi-PBC with no lowering except pass-through.
4. Add hard-cap checks.
5. Add safe baseline rotation and measurement lowering.
6. Add Peres/Galvao greedy measurement reduction.
7. Add scheduling/layer metadata.
8. Add summary reporting.
9. Add small-circuit equivalence tests and FTCircuitBench smoke tests.
10. Write separate specs/plans for router conversion, `dascot-rs` extension, and direct C+T -> semi-PBC.
