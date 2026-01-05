import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def find_stats_files_for_precision(base_dir: str, precision: int) -> List[str]:
    """
    Locate all stats.json files for GS precision level under circuit_benchmarks.
    """
    pattern = os.path.join(
        base_dir,
        "circuit_benchmarks",
        "*",
        "*",
        "GS",
        f"precision_level_{precision}",
        "*_gs_prec*_stats.json",
    )
    files = sorted(glob.glob(pattern))
    # Exclude special category directory
    files = [p for p in files if "hamiltonians_5trotter" not in p]
    return files


def extract_label_and_counts(stats_path: str) -> Tuple[str, int, int, int]:
    """
    Extract circuit label and counts from a stats JSON.

    Returns (label, clifford_count, t_family_count, total_gate_count)
    """
    with open(stats_path, "r") as f:
        data: Dict = json.load(f)

    # Label is the circuit directory name (e.g., adder_28q)
    circuit_dir = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(stats_path)))
    )
    label = circuit_dir.replace("_", " ")

    t_family = data.get("total_t_family_count")
    if t_family is None:
        t = data.get("t_count", 0) or 0
        tdg = data.get("tdg_count", 0) or 0
        t_family = int(t) + int(tdg)

    total_gate_count = data.get("total_gate_count")

    clifford = data.get("clifford_gate_count")
    if clifford is None:
        # Fallback: try detailed counts sum
        detailed = data.get("detailed_clifford_counts")
        if isinstance(detailed, dict):
            clifford = int(sum(detailed.values()))
        elif total_gate_count is not None:
            # As a last resort, approximate from total - T family
            clifford = max(0, int(total_gate_count) - int(t_family))
        else:
            clifford = 0

    # Ensure ints
    clifford = int(clifford)
    t_family = int(t_family)
    total_gate_count = (
        int(total_gate_count) if total_gate_count is not None else clifford + t_family
    )

    return label, clifford, t_family, total_gate_count


def build_dataset(stats_files: List[str]) -> List[Tuple[str, int, int, int]]:
    rows: List[Tuple[str, int, int, int]] = []
    for p in stats_files:
        try:
            rows.append(extract_label_and_counts(p))
        except Exception:
            # Skip malformed or unexpected files
            continue
    # Sort by total gate count ascending
    rows.sort(key=lambda r: r[3])
    return rows


def plot_stacked_counts(
    rows: List[Tuple[str, int, int, int]],
    title: str,
    output_pdf_path: str,
    font_size: int = 18,
    figsize: Tuple[float, float] = (16.0, 5.5),
) -> None:
    labels = [r[0] for r in rows]
    cliffords = [r[1] for r in rows]
    t_family = [r[2] for r in rows]

    plt.rcParams.update(
        {"font.size": font_size, "font.weight": "bold", "axes.labelweight": "bold"}
    )
    fig, ax = plt.subplots(figsize=figsize)

    x = list(range(len(labels)))
    ax.bar(x, cliffords, label="Clifford count", color="#1f77b4")  # blue
    ax.bar(
        x, t_family, bottom=cliffords, label="T-family count", color="#d62728"
    )  # red

    # No title per request
    # No x-axis label per request
    ax.set_xlabel("")
    ax.set_ylabel("Total Gate Count", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="right", rotation_mode="anchor")
    # Bold tick labels and slightly smaller x labels to reduce overlap
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
        try:
            tick.set_fontsize(max(10, font_size - 4))
        except Exception:
            pass
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Extra bottom margin to prevent overlap
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    precision = 8

    stats_files = find_stats_files_for_precision(base_dir, precision)
    rows = build_dataset(stats_files)

    if not rows:
        print("No stats files found for GS precision 8.")
        return

    output_pdf = os.path.join(
        base_dir,
        "allcircuits_gs_prec8_stacked_clifford_t_counts.pdf",
    )
    plot_stacked_counts(
        rows, "", output_pdf_path=output_pdf, font_size=18, figsize=(16.0, 5.5)
    )
    print(f"Saved plot to: {output_pdf}")


if __name__ == "__main__":
    main()
