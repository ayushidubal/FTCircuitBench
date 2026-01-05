import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def _get_individual_fidelities_list(data: dict) -> Optional[List[float]]:
    """Try several keys to locate a list of individual fidelities."""
    for key in [
        "individual_fidelities",
        "fidelities",
        "fidelity_list",
        "per_gate_fidelities",
    ]:
        value = data.get(key)
        if isinstance(value, list) and len(value) > 0:
            return value
    return None


def collect_points(stats_files: List[Path]) -> Tuple[List[int], List[float]]:
    """Return x (len(individual_fidelities)) and y (fidelity) arrays from stats files.

    Skip files where an individual fidelities list is not present.
    """
    x_values: List[int] = []
    y_values: List[float] = []
    for stats_path in stats_files:
        try:
            with stats_path.open("r") as f:
                data = json.load(f)
            fidelity = data.get("fidelity")
            indiv_list = _get_individual_fidelities_list(data)
            if fidelity is None or indiv_list is None:
                continue
            x_values.append(int(len(indiv_list)))
            y_values.append(float(fidelity))
        except Exception:
            # Skip any malformed or unreadable files
            continue
    return x_values, y_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fidelity vs total gate count for GS stats"
    )
    parser.add_argument(
        "--precision",
        choices=["5", "8", "both"],
        default="both",
        help="Which precision set to plot",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    bench_root = root / "circuit_benchmarks"

    prec8_files = sorted(bench_root.rglob("*_gs_prec8_stats.json"))
    prec5_files = sorted(bench_root.rglob("*_gs_prec5_stats.json"))

    # Exclude special category directory
    prec8_files = [p for p in prec8_files if "hamiltonians_5trotter" not in str(p)]
    prec5_files = [p for p in prec5_files if "hamiltonians_5trotter" not in str(p)]

    fig, ax = plt.subplots(figsize=(7, 5))

    if args.precision in ("8", "both"):
        x8, y8 = collect_points(prec8_files)
        ax.scatter(x8, y8, s=20, c="#1f77b4", label="GS precision 8", alpha=0.9)

    if args.precision in ("5", "both"):
        x5, y5 = collect_points(prec5_files)
        ax.scatter(x5, y5, s=20, c="#aec7e8", label="GS precision 5", alpha=0.9)

    ax.set_xlabel("Number of individual fidelities")
    ax.set_ylabel("Fidelity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    if args.precision == "8":
        output_name = "allcircuits_gs_prec8_fidelity_vs_total_gate_count.pdf"
    elif args.precision == "5":
        output_name = "allcircuits_gs_prec5_fidelity_vs_total_gate_count.pdf"
    else:
        output_name = "allcircuits_gs_prec5_prec8_fidelity_vs_total_gate_count.pdf"

    output_path = root / output_name
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
