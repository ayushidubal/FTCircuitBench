# AI Trajectory-Prefix Emitter Design

## Goal

Turn the AI Pauli trajectory-prefix analysis into a first compiler path that emits semi-PBC JSONL operations for rotation windows.

## Scope

This slice handles consecutive `t_pauli` runs only. It does not alter measurement lowering, and it does not yet solve global window scheduling. The implementation adds an `ai-trajectory-prefix` optimization mode that attempts a bounded AI window replacement and falls back to the existing lowering path when the model fails, produces NaN diagnostics, cannot reach the cap, or would not be structurally valid.

## Architecture

The emitter reuses the existing semi-PBC IR:

- AI solver gate actions become `h`, `s`, or `cx` `SemiPBCOp` records.
- AI rotation emissions before the selected prefix become weight-1 `t_pauli` records.
- Remaining pending replay terms at the k-terminal prefix become capped `t_pauli` records.

The emitted operations are validated with the existing `validate_program` path. Benchmark output also records whether a trajectory result is `ok_clean`, `ok_with_solver_nan`, `failed_solver_nan`, or another status-derived category.

## Testing

Tests use mocked trajectory results for compiler emission so normal CI does not depend on the IBM model. Unit tests check that emitted semi-PBC operations match the intended Clifford-prefix plus capped-Pauli-tail shape and remain unitary-equivalent on small examples.
