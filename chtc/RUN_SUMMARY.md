# FTCircuitBench CHTC Run Summary (ARD-1)

- Generated on: 2026-03-06
- Output directory: `circuit_benchmarks_ard_1`
- Pipeline: `GS` only (`--skip-sk`)
- GS precisions: `3` and `10`
- PBC conversion: skipped (`--skip-pbc`)
- Fidelity computation: skipped (`--skip-fidelity`)
- Main shard count: `46` (`chtc/ftc_generate.sub`)
- Circuit filter used for main sweep: `chtc/remaining_gs_instances.txt`

## Coverage

- Total circuits discovered under `qasm/`: **95**
- Requested GS configurations: **190**
- Completed GS configurations: **190**
- Fully complete circuits (precisions 3 and 10): **95**
- Partially complete circuits: **0**
- Missing circuits: **0**

## Category Breakdown

| Category | Total | Complete | Partial | Missing | Prec3 done | Prec10 done |
|---|---:|---:|---:|---:|---:|---:|
| `adder` | 4 | 4 | 0 | 0 | 4 | 4 |
| `hamiltonians` | 33 | 33 | 0 | 0 | 33 | 33 |
| `hamiltonians_5trotter` | 36 | 36 | 0 | 0 | 36 | 36 |
| `hhl` | 4 | 4 | 0 | 0 | 4 | 4 |
| `qft` | 4 | 4 | 0 | 0 | 4 | 4 |
| `qpe` | 6 | 6 | 0 | 0 | 6 | 6 |
| `qsvt` | 8 | 8 | 0 | 0 | 8 | 8 |

## Missing Circuits

- None.

## Partial Circuits

- None.

Detailed per-instance status is in `chtc/run_coverage_summary.json`.
