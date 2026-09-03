# AI Pauli-Network Experiment Design

## Goal

Evaluate whether Qiskit's `AIPauliNetworkSynthesis` pass can help reduce the
Clifford/CX scaffolding produced by the capped semi-PBC lowering path.

## Scope

This is an experiment, not a compiler integration. The first implementation
should provide a small adapter and CLI that:

- detects whether `qiskit-ibm-transpiler` is importable;
- builds toy Qiskit Pauli-network circuits from Pauli rotation data;
- optionally runs `CollectPauliNetworks` and `AIPauliNetworkSynthesis`;
- reports total ops, two-qubit ops, depth, and whether the AI pass changed the
  circuit.

The adapter must be optional. FTCircuitBench should continue to install and
test without `qiskit-ibm-transpiler`.

## Approaches To Compare

1. **k=1 oracle comparison.** Lower PBC windows to ordinary Clifford plus
   single-qubit rotations, then compare our lowering against the IBM pass.
   This uses the pass as intended and gives a clear quality baseline.
2. **Frame-mining adapter.** Run the IBM pass, trace the emitted Clifford
   frames, and inspect whether intermediate frames make pending Paulis weight
   `<= k`. This can indicate whether the existing model discovers useful
   k-compatible frames before we modify any synthesis code.
3. **Clifford-frame postprocessing.** Keep semi-PBC Pauli blocks intact, but
   use `AICliffordSynthesis` or `AILinearFunctionSynthesis` on Clifford-only
   scaffolding between blocks.

The first local step should implement only the k=1 oracle comparison harness.
The other approaches require more design after we see package/API behavior and
small-circuit results.

## Success Criteria

- Missing optional dependency is reported cleanly.
- When the dependency is present, a toy Pauli-network circuit can be passed
  through the AI synthesis pass locally.
- The CLI writes a JSON report with before/after metrics.
- Existing semi-PBC tests continue to pass.

## Non-Goals

- Do not modify the semi-PBC compiler lowering path yet.
- Do not add `qiskit-ibm-transpiler` as a required package dependency.
- Do not modify IBM's synthesis implementation in this phase.
- Do not assume the pass output is directly valid semi-PBC IR.
