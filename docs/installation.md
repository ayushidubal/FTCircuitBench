# Installation

Tested on Python 3.8+. Use a fresh `.venv` so Qiskit and nwqec stay isolated.

## Prerequisites

- Python 3.8+.
- Virtual environment tool (`python -m venv` recommended). If a Conda env is active, deactivate it first.
- Gridsynth binary on your `PATH` (install via `cabal install gridsynth` and add the cabal bin dir). The package prefers the binary over the Python wrapper.

`requirements.txt` already includes runtime and dev/test deps (Qiskit, nwqec, pytest, ruff, etc.).

## Steps

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Quick checks:

```bash
python analyze_circuit.py --help
python - <<'PY'
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
print("OK: ftcircuitbench import")
PY
```

If the Gridsynth binary is missing, the Python path still works but runs slower; keep the binary available for best performance.
