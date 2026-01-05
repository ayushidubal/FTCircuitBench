import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_percent(value: str) -> float:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("%"):
            s = s[:-1]
        return float(s)
    except Exception:
        return None


def read_pbc_csv(
    csv_path: str, include_label: callable = None
) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = (r.get("label") or "").strip()
            if not label:
                continue
            if include_label is not None and not include_label(label):
                continue
            rot_red = parse_percent(r.get("rotation reduction"))
            pw_red = parse_percent(r.get("avg pauli weight reduction"))
            if rot_red is None and pw_red is None:
                continue
            rows.append(
                (
                    label,
                    rot_red if rot_red is not None else 0.0,
                    pw_red if pw_red is not None else 0.0,
                )
            )
    return rows


def wrap_axis_label(label: str, max_chars: int = 26) -> str:
    label = str(label)
    if len(label) <= max_chars:
        return label
    parts = label.split(" ")
    if len(parts) <= 1:
        return label
    target = len(label) // 2
    best_idx = 0
    best_delta = 10**9
    cur_len = 0
    for i, w in enumerate(parts[:-1], start=1):
        cur_len += len(w) + 1
        delta = abs(cur_len - 1 - target)
        if delta < best_delta:
            best_delta = delta
            best_idx = i
    return " ".join(parts[:best_idx]) + "\n" + " ".join(parts[best_idx:])


def plot_pbc_reductions(
    rows: List[Tuple[str, float, float]],
    output_pdf_path: str,
    font_size: int = 16,
    figsize: Tuple[float, float] = (16.0, 5.5),
    sort_ascending: bool = True,
    rotate_labels: int = 90,
):
    if not rows:
        print("No rows to plot.")
        return

    # Sort by rotation reduction (first series)
    rows = sorted(
        rows,
        key=lambda r: (r[1] if r[1] is not None else 0.0),
        reverse=not sort_ascending,
    )

    def format_label(label: str) -> str:
        lab = str(label)
        # Remove trailing pipeline suffix like -gs-8 or -gs-5 or -sk-#
        # Expected pattern: ...-gs-<digits> or ...-sk-<digits>
        for prefix in ("-gs-", "-sk-"):
            idx = lab.rfind(prefix)
            if idx != -1:
                # Ensure digits follow the prefix
                tail = lab[idx + len(prefix) :]
                if tail.isdigit():
                    lab = lab[:idx]
                    break
        # Replace hyphens with spaces to match other histogram styles
        return lab.replace("-", " ")

    labels = [format_label(r[0]) for r in rows]
    rot_vals = [r[1] for r in rows]
    pw_vals = [r[2] for r in rows]

    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=figsize)

    n_groups = len(labels)
    n_series = 2
    x = np.arange(n_groups)
    total_width = 0.85
    bar_width = total_width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

    # Colors: red then blue to match prior style
    palette = ["#d62728", "#1f77b4"]
    ax.bar(
        x + offsets[0],
        rot_vals,
        width=bar_width * 0.95,
        label="Rotation Reduction",
        color=palette[0],
    )
    ax.bar(
        x + offsets[1],
        pw_vals,
        width=bar_width * 0.95,
        label="Avg Pauli Weight Reduction",
        color=palette[1],
    )

    # Use symmetric log scale to handle positive/negative reductions
    try:
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
    except Exception:
        pass

    # No y-axis label in multi-stat plots
    ax.set_ylabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=rotate_labels, ha="right", rotation_mode="anchor"
    )

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
        try:
            tick.set_fontsize(max(10, font_size - 4))
        except Exception:
            pass

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.28)

    fig.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PBC optimizer reductions per circuit from pbc_stats.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to pbc_stats.csv (default: circuit_benchmarks/pbc_stats.csv)",
    )
    parser.add_argument(
        "--gs",
        type=str,
        choices=["5", "8"],
        default="8",
        help="GS precision suffix to include (-gs-5 or -gs-8)",
    )
    parser.add_argument(
        "--fontsize", type=int, default=16, help="Base font size (default: 16)"
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=16.0,
        help="Figure width in inches (default: 16.0)",
    )
    parser.add_argument(
        "--figheight",
        type=float,
        default=5.5,
        help="Figure height in inches (default: 5.5)",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order (default: ascending)",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.input or os.path.join(
        base_dir, "circuit_benchmarks", "pbc_stats.csv"
    )

    # Include only requested GS suffix and exclude qpe/qsvt. Include hamiltonians.
    def include_main(label: str) -> bool:
        lab = label.lower()
        if not lab.endswith(f"-gs-{args.gs}"):
            return False
        if lab.startswith("qpe-") or lab.startswith("qsvt-"):
            return False
        return True

    rows = read_pbc_csv(csv_path, include_label=include_main)

    if not rows:
        print(f"No usable rows found in: {csv_path}")
        return

    output_pdf = os.path.join(
        base_dir,
        f"allcircuits_pbc_rotation_and_pauli_weight_reduction_gs{args.gs}_histogram.pdf",
    )
    plot_pbc_reductions(
        rows,
        output_pdf_path=output_pdf,
        font_size=args.fontsize,
        figsize=(args.figwidth, args.figheight),
        sort_ascending=not args.desc,
    )
    print(f"Saved plot to: {output_pdf}")


if __name__ == "__main__":
    main()
