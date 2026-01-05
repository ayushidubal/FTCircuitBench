# API Reference

Everything exposed for public use lives in `ftcircuitbench.api`.

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
```

## Data classes

### PipelineConfig
Settings for a single pipeline run (defaults shown):

- `pipeline`: `"gs"` or `"sk"`.
- `gridsynth_precision`: int (GS only, default 3).
- `sk_recursion`: int (SK only, default 1).
- `layering_method`: `"bare"`, `"v2"`, `"v3"`, `"singleton"` (default `"v2"`). If `layering_max_checks` is set, `"v2"` is treated as `"v3"`.
- `layering_max_checks`: optional int bound for PBC layering lookback.
- `optimize_pbc`: bool (Tfuse/T-merging), default `False`.
- `optimize_t_maxiter`: T-merging iterations (default 5; 0 disables).
- `prefer_cpp`: prefer the Gridsynth binary (default `True`).
- `calculate_fidelity`: compute fidelity when feasible (default `True`; SK fidelity is skipped above the internal qubit bound).
- `return_intermediate`: request intermediate circuits (default `True`).
- `max_workers`: optional worker cap for parallel PBC.
- `clifford_output_path`: save Clifford+T QASM if set.
- `pbc_output_prefix`: save PBC layer/measurement artifacts if set.

### PipelineResult
- `clifford_t_circuit`, `pbc_circuit`, `intermediate_circuit` (optional).
- `clifford_stats`, `pbc_stats`, `fidelity`, `timings`, `parameters`.
- `artifacts`: paths for any saved QASM/PBC files.
- `to_dict(include_circuits=False, include_artifacts=True)`.

### AnalysisResult
- `input_path`, `original_qubits`, `original_gates`.
- `pipelines`: `Dict[pipeline_name, PipelineResult]`.
- `to_dict(include_circuits=False, include_artifacts=True)`.

## Functions

- `run_pipeline(circuit, config)`: run GS or SK → PBC on a `QuantumCircuit`.
- `run_analysis(circuit, configs, source_path=None)`: run one or more `PipelineConfig` objects on an in-memory circuit.
- `run_analysis_for_file(qasm_file, configs)`: load QASM then delegate to `run_analysis`.

Example:

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file

cfgs = [
    PipelineConfig(pipeline="gs", gridsynth_precision=4, optimize_pbc=True),
    PipelineConfig(pipeline="sk", sk_recursion=2, calculate_fidelity=False),
]
analysis = run_analysis_for_file("qasm/hhl/hhl_7q.qasm", cfgs)
print(analysis.pipelines["gs"].clifford_stats["total_t_family_count"])
print(analysis.to_dict(include_artifacts=True))
```
