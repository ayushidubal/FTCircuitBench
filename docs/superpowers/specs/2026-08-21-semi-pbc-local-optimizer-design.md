# Semi-PBC Local Optimizer Design

**Date:** 2026-08-21
**Status:** Approved for implementation planning
**Scope:** First smarter-than-baseline optimizer for capped PBC -> semi-PBC lowering.

## 1. Goal

Add an exact local optimizer for the Phase 1 semi-PBC compiler. The optimizer should keep the hard Pauli-block cap `k`, preserve source PBC semantics, and reduce avoidable Clifford work when nearby high-weight Pauli gadgets have compatible structure.

This is the first step toward Paulihedral/tket/Rustiq-style synthesis, but it deliberately stays local and dependency-free.

## 2. Motivation

The baseline compiler is correct but mechanically lowers every high-weight Pauli independently:

```text
basis change
fold extra active qubits into one retained target
bounded Pauli rotation or measurement
unfold
undo basis change
```

That is a safe starting point, but it misses opportunities that matter for routing:

- adjacent gadgets may fold the same parity and then immediately uncompute/recompute it;
- arbitrary retained-qubit choices can make future local cancellation harder;
- choosing retained qubits without nearby context can preserve the wrong part of a Pauli block.

The local optimizer should make deterministic, context-aware choices before lowering and then remove exact adjacent inverse Clifford pairs after lowering.

## 3. Non-Goals

This pass will not implement full Paulihedral, tket, or Rustiq synthesis. In particular, it will not:

- reorder non-adjacent PBC operations globally;
- perform hardware-layout-aware routing;
- introduce new ancillas for tree or fanout parity networks;
- synthesize arbitrary Clifford normal forms;
- split high-weight measurements into independently measured chunks that reveal partial parities;
- depend on external optimizer packages.

Those remain future passes once we have a stable semi-PBC IR and router converter.

## 4. Inputs and Outputs

Input is the existing reduced source-op stream used by `compile_pbc_text`:

```text
nwqec PBC text -> parser -> optional Peres/Galvao reducer -> ReducedSourceOp stream
```

Output remains the same semi-PBC JSONL IR:

- 1-qubit Clifford ops;
- 2-qubit Clifford ops;
- `[1,k]`-weight `pi/8` Pauli rotations;
- `[1,k]`-weight Pauli measurements;
- classical `xor` result updates.

The sidecar remains optional. If optimization removes emitted operations, provenance records for removed operation IDs should also be removed.

## 5. Local Optimizer

The first optimizer mode is named `local-window`.

For each source Pauli term of weight `r > k`, it chooses the retained legal block using immediate-neighbor context:

1. Start from the term's active qubits in canonical order.
2. Look at the nearest previous and next non-identity source Pauli terms.
3. Score candidate retained subsets of size `k` by overlap with those neighboring supports.
4. Break ties deterministically using the source term's canonical qubit order.
5. Choose the retained target qubit from the retained block by the same neighbor-overlap score, again with canonical-order ties.

For large supports, full combination enumeration can be expensive. The pass should enumerate only when the number of `r choose k` candidates is below a fixed cap. Above that cap, use a deterministic greedy fallback: sort active qubits by descending neighbor-overlap score and canonical order, then keep the first `k`.

This choice does not change semantics. It only decides which legal weight-`k` block remains after parity compression.

## 6. Clifford Peephole

After local-window lowering, run a stack-style adjacent cancellation pass over emitted operations. It may remove only exact adjacent inverse Clifford pairs:

- `h(q)` followed by `h(q)`;
- `cx(control,target)` followed by the same `cx(control,target)`;
- `s(q)` followed by `sdg(q)`;
- `sdg(q)` followed by `s(q)`.

The pass must not commute operations, cross measurements, cross Pauli rotations, or reason about non-adjacent gates. This keeps it easy to audit and exact.

The cancellation pass is useful because local-window retained-block choices can align consecutive parity-compression gadgets. For example, two adjacent identical high-weight Z rotations can keep one folded parity live across the boundary after cancellation, reducing redundant CNOTs while preserving the same source operation order.

## 7. API and CLI

Add an `optimization` option:

```text
none
local-window
```

`none` preserves the previous baseline behavior for reproducibility. `local-window` enables context-aware retained-block selection and adjacent Clifford cancellation.

The CLI should expose:

```text
--optimization none|local-window
```

The summary and sidecar should record the selected optimization mode so benchmark artifacts are self-describing.

## 8. Testing

Tests should cover:

1. Retained-block selection prefers overlap with immediate neighbors.
2. The selector is deterministic and respects the hard cap.
3. Adjacent inverse Clifford pairs are removed exactly.
4. `compile_pbc_text(..., optimization="none")` preserves prior behavior.
5. `compile_pbc_text(..., optimization="local-window")` can reduce adjacent identical high-weight rotation gadgets.
6. The sidecar contains only retained output operation IDs after cancellation.
7. CLI accepts `--optimization local-window`.

Existing matrix/projector-style lowering tests should continue to provide semantic coverage for the underlying exact gadgets. The optimizer itself is conservative: it changes retained-qubit choice and deletes only adjacent inverse Clifford pairs.

## 9. Future Work

After this pass is stable, the next optimizers can target larger gains:

- windowed commuting-term grouping;
- Pauli-network synthesis with relaxed ordering;
- Clifford-frame tracking instead of immediate uncompute;
- layout-aware choice of retained block and target;
- direct C+T -> capped semi-PBC generation before flat PBC structure is lost.
