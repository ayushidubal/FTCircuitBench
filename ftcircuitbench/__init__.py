"""
FTCircuitBench: fault-tolerant quantum circuit compilation benchmarks.

The public API names listed in ``__all__`` are loaded lazily so importing a
small submodule such as ``ftcircuitbench.semi_pbc`` does not import optional
analysis and visualization dependencies.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"numpy\.linalg\._linalg"
)
warnings.filterwarnings(
    "ignore",
    message=r".*(divide by zero|invalid value) encountered in det.*",
    category=RuntimeWarning,
)

_LAZY_EXPORTS = {
    "AnalysisResult": ("ftcircuitbench.api", "AnalysisResult"),
    "PipelineConfig": ("ftcircuitbench.api", "PipelineConfig"),
    "PipelineResult": ("ftcircuitbench.api", "PipelineResult"),
    "run_analysis": ("ftcircuitbench.api", "run_analysis"),
    "run_analysis_for_file": ("ftcircuitbench.api", "run_analysis_for_file"),
    "run_pipeline": ("ftcircuitbench.api", "run_pipeline"),
    "load_qasm_circuit": ("ftcircuitbench.parser", "load_qasm_circuit"),
    "transpile_qasm_to_target_basis": (
        "ftcircuitbench.parser",
        "transpile_qasm_to_target_basis",
    ),
    "decompose_rz_gates_gridsynth": (
        "ftcircuitbench.decomposer",
        "decompose_rz_gates_gridsynth",
    ),
    "transpile_to_solovay_kitaev_clifford_t": (
        "ftcircuitbench.transpilers",
        "transpile_to_solovay_kitaev_clifford_t",
    ),
    "transpile_to_gridsynth_clifford_t": (
        "ftcircuitbench.transpilers",
        "transpile_to_gridsynth_clifford_t",
    ),
    "analyze_clifford_t_circuit": (
        "ftcircuitbench.analyzer.clifford_t_analyzer",
        "analyze_clifford_t_circuit",
    ),
    "analyze_pbc_circuit": (
        "ftcircuitbench.analyzer.pbc_analyzer",
        "analyze_pbc_circuit",
    ),
    "convert_to_pbc_circuit": (
        "ftcircuitbench.pbc_converter",
        "convert_to_pbc_circuit",
    ),
    "rz_product_fidelity": ("ftcircuitbench.fidelity", "rz_product_fidelity"),
    "calculate_circuit_fidelity": (
        "ftcircuitbench.fidelity",
        "calculate_circuit_fidelity",
    ),
    "MAX_QUBITS_FOR_FIDELITY": (
        "ftcircuitbench.fidelity",
        "MAX_QUBITS_FOR_FIDELITY",
    ),
    "show_clifford_t_interaction_graph": (
        "ftcircuitbench.analyzer.visualization",
        "show_clifford_t_interaction_graph",
    ),
    "show_pbc_interaction_graph": (
        "ftcircuitbench.analyzer.visualization",
        "show_pbc_interaction_graph",
    ),
    "show_operator_weight_histogram": (
        "ftcircuitbench.analyzer.visualization",
        "show_operator_weight_histogram",
    ),
    "show_qubit_pbc_operations_plot": (
        "ftcircuitbench.analyzer.visualization",
        "show_qubit_pbc_operations_plot",
    ),
    "get_interaction_statistics": (
        "ftcircuitbench.analyzer.visualization",
        "get_interaction_statistics",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'ftcircuitbench' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])
