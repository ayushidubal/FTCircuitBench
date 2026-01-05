# FTCircuitBench

Fault-tolerant circuit compilation and analysis with Gridsynth (GS), Solovay-Kitaev (SK), and Pauli-Based Computation (PBC).

- [`installation.md`](installation.md) — set up a `.venv` and install the package.
- [`api.md`](api.md) — public Python surface in `ftcircuitbench.api`.
- [`examples.md`](examples.md) — minimal CLI and programmatic recipes.

## Quick orientation

- Pipelines: GS or SK → Clifford+T → PBC conversion → optional fidelity + stats.
- Outputs: Clifford+T QASM under `clifford_t_output/`, PBC layers/basis under `pbc_output/`, stats JSON under `circuit_stats_output/`.
- Scripts: `analyze_circuit.py` (single circuit CLI), `generate_benchmarks.py` (batch), and the walkthrough notebook `FTCircuitBench_Pipeline_Demo.ipynb`.
- Library: import `PipelineConfig`, `run_pipeline`, or `run_analysis_for_file` from `ftcircuitbench.api`.
- Data: sample QASM circuits live in `qasm/`; plots are written as PDFs in `figs/`.

Start with installation, then try a single-circuit run with `analyze_circuit.py` or call `run_pipeline` directly from Python.
