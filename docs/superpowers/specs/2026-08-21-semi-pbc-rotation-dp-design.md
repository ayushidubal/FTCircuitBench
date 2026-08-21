# Semi-PBC Rotation-DP Optimizer Design

**Date:** 2026-08-21
**Status:** Approved for implementation planning
**Scope:** Opportunity reporting plus the next exact optimizer after `local-window`.

## 1. Goal

Add two capabilities:

1. an opportunity report that tells us where capped-PBC optimization can plausibly win; and
2. a source-order dynamic-programming optimizer for contiguous high-weight Pauli rotation runs.

The optimizer must preserve the hard Pauli weight cap `k`, preserve source operation order, and preserve exact circuit semantics.

## 2. Motivation

The first `local-window` optimizer only improves op count when naive lowering creates adjacent inverse Clifford pairs. On the checked adder/QFT samples, it preserved the cap but did not reduce aggregate counts.

The next useful step is to optimize the Clifford frames between adjacent high-weight rotations, instead of choosing each lowered gadget independently.

Naive lowering emits each high-weight rotation as:

```text
C_i  R_i'  C_i^-1
```

where `C_i` maps the original Pauli word into a bounded retained Z-block, and `R_i'` is the legal `[1,k]`-weight `pi/8` Pauli rotation.

A run of high-weight rotations can instead be emitted as:

```text
C_0 R_0' (C_0^-1 C_1) R_1' ... (C_{n-2}^-1 C_{n-1}) R_{n-1}' C_{n-1}^-1
```

This is still source-order exact. It simply chooses the compression frame for each rotation jointly and cancels redundant Clifford work in the frame transitions.

## 3. Non-Goals

This phase will not:

- reorder rotations, even if they commute;
- optimize across measurements or classical `xor` operations;
- keep a frame across low-weight rotations;
- add ancillas or tree parity networks;
- implement arbitrary Clifford synthesis;
- use physical-layout distances;
- change high-weight measurement lowering beyond the existing `local-window` retained-block choice.

Those are future passes.

## 4. Opportunity Report

Add a report API and CLI that read PBC input and emit compact JSON.

The report should include:

- input operation count;
- rotation and measurement counts;
- max input Pauli weight;
- weight histograms by operation type;
- count of operations above `k`;
- contiguous high-weight rotation run count and largest run length;
- adjacent non-identity support-overlap summary;
- repeated Pauli support counts;
- compile-summary comparisons for `none`, `local-window`, and `rotation-dp`.

The report does not need to solve routing. It is a cheap compiler-side triage tool for deciding which circuits and `k` values deserve deeper optimization.

## 5. Rotation-DP Optimizer

Add a new optimization mode:

```text
rotation-dp
```

It should include all safe `local-window` behavior, plus source-order DP over contiguous runs of high-weight `t_pauli` operations.

For each high-weight rotation term:

1. Generate candidate retained blocks.
2. For each retained block, build the equivalent lowered rotation once:
   ```text
   prefix Clifford ops, bounded t_pauli block, suffix Clifford ops
   ```
3. Score transitions between candidate frames by applying the existing adjacent inverse-Clifford cancellation to:
   ```text
   previous suffix + next prefix
   ```
4. Use dynamic programming to choose one candidate per source rotation that minimizes:
   ```text
   prefix_cost(first) + sum(transition_costs) + suffix_cost(last)
   ```
   The Pauli rotations themselves are constant within the run, so the first cost model can optimize emitted Clifford count.
5. Emit:
   ```text
   first prefix, first bounded rotation, optimized transitions, remaining bounded rotations, last suffix
   ```

The first candidate set should be deterministic and bounded:

- enumerate all retained subsets and target choices when the candidate count is below a fixed limit;
- otherwise include a small deterministic fallback set containing the canonical candidate and the existing local-window candidate.

## 6. Correctness

The DP never changes source order. It only rewrites:

```text
C_i^-1 C_{i+1}
```

by deleting adjacent inverse Clifford pairs inside that transition.

This is exact because it is algebraically the same Clifford product as independent lowering. The emitted bounded Pauli rotation is applied in the selected frame for its source term.

The pass must break runs at:

- any `m_pauli`;
- any reduced identity measurement represented only by `xor`;
- any low-weight `t_pauli`;
- any unsupported source operation.

This avoids having to conjugate arbitrary lower-weight operations through a live frame in Phase 2.

## 7. API and CLI

Update the compiler API and CLI to allow:

```text
--optimization rotation-dp
```

Add a new report script:

```text
analyze_semi_pbc_opportunities.py --pbc <path> --k <k> --out <report.json>
```

The report CLI should default to the same measurement reducer settings as the compiler.

## 8. Testing

Tests should cover:

1. candidate block generation includes distinct targets for the same retained subset;
2. DP reduces a toy run where local-window cannot choose the optimal sequence;
3. DP preserves exact unitary semantics on the optimized toy run;
4. the compiler rejects unsupported optimization strings and accepts `rotation-dp`;
5. CLI accepts `--optimization rotation-dp`;
6. opportunity report JSON includes expected histograms and compile comparisons;
7. report CLI writes valid JSON;
8. existing `none` and `local-window` behavior remains unchanged.

## 9. Expected Limits

This pass may still show limited improvement on QFT/adder if useful rotations are not contiguous in source order. If the report shows short high-weight rotation runs or low repeated-support overlap, the next pass should be commuting-window reordering for rotations.
