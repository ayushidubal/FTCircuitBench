# FTCircuitBench
Fault-tolerant circuit benchmarking with Clifford+T synthesis and Pauli-Based Computation (PBC).

## Install
```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional for local edits
pip install -e .
```
Requirements: Python 3.8+, `nwqec` (for fast Gridsynth/PBC, uses `fuse_t`), optional `gridsynth` binary in PATH for Python fallback.

## Entry points
- CLI (single circuit): `analyze_circuit.py`
- CLI (batch benchmarks): `generate_benchmarks.py`
- Notebook demo: `FTCircuitBench_Pipeline_Demo.ipynb`

### Examples
Analyze one circuit (GS pipeline, PBC on):
```bash
source .venv/bin/activate
python analyze_circuit.py qasm/qft/qft_18q.qasm \
  --pipeline gs \
  --gridsynth-precision 5 \
  --optimize-pbc \
  --optimize-t-maxiter 5 \
  --artifact-root circuit_outputs
```

Generate benchmarks:
```bash
source .venv/bin/activate
python generate_benchmarks.py --max-qubits 30 --output-dir circuit_benchmarks
```

Notebook:
Open `FTCircuitBench_Pipeline_Demo.ipynb` in Jupyter, select the project `.venv` kernel, run all cells.

Common CLI flags: `--pipeline {gs,sk,both}`, `--gridsynth-precision N`, `--sk-recursion N`, `--layering-method {bare,v2,v3,singleton}`, `--layering-max-checks K`, `--optimize-pbc`, `--optimize-t-maxiter N`, `--skip-fidelity`, `--max-workers N`.

## Repository structure (trimmed)
```
FTCircuitBench/
├── ftcircuitbench/            # Library code (API, analyzers, PBC converter, transpilers)
├── analyze_circuit.py         # CLI: analyze one circuit
├── generate_benchmarks.py     # CLI: batch benchmarks
├── FTCircuitBench_Pipeline_Demo.ipynb  # Notebook demo
├── qasm/                      # Input circuits
├── circuit_benchmarks/        # Benchmark outputs
├── circuit_stats_output/      # Stats JSON
├── clifford_t_output/         # Clifford+T QASM
├── pbc_output/                # PBC artifacts
├── requirements.txt
└── docs/
```