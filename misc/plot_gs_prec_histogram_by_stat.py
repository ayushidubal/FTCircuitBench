import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def find_stats_files_for_precision(base_dir: str, precision: int) -> List[str]:
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


def get_stat_value(data: Dict[str, Any], key: str) -> Any:
    """
    Retrieve a (possibly nested via dot-path) statistic from the JSON dict.
    Example: key="interaction_graph_density" or "pbc.pauli_weight_distribution.3".
    """
    if not key:
        return None
    parts = key.split(".")
    cur: Any = data
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            # Try integer key for numeric-string dicts
            if isinstance(cur, dict) and p.isdigit() and int(p) in cur:
                cur = cur[int(p)]
            else:
                return None
    return cur


def extract_label_and_stat(stats_path: str, stat_key: str) -> Tuple[str, float]:
    with open(stats_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    # Label is the circuit directory name (e.g., adder_28q)
    circuit_dir = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(stats_path)))
    )
    label = circuit_dir.replace("_", " ")

    value = get_stat_value(data, stat_key)
    if value is None:
        raise ValueError(f"Stat '{stat_key}' not found in {stats_path}")

    try:
        value_f = float(value)
    except Exception as e:
        raise ValueError(
            f"Stat '{stat_key}' not numeric in {stats_path}: {value}"
        ) from e

    return label, value_f


def build_dataset(stats_files: List[str], stat_key: str) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for p in stats_files:
        try:
            rows.append(extract_label_and_stat(p, stat_key))
        except Exception:
            # Skip files without the requested stat or non-numeric values
            continue
    return rows


def plot_stat_histogram(
    rows: List[Tuple[str, float]] = None,
    stat_key: str = "",
    output_pdf_path: str = "",
    font_size: int = 16,
    figsize: Tuple[float, float] = (16.0, 5.5),
    sort_ascending: bool = True,
    bar_color: str = "#1f77b4",
    rotate_labels: int = 90,
    log_scale: bool = False,
    multi_values: Dict[str, List[Tuple[str, float]]] = None,
    stat_keys: List[str] = None,
) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )

    def _wrap_axis_label(label: str, max_chars: int = 26) -> str:
        label = str(label)
        if len(label) <= max_chars:
            return label
        parts = label.split(" ")
        if len(parts) <= 1:
            return label
        # Find a split point closest to half the length
        target = len(label) // 2
        best_idx = 0
        best_delta = 10**9
        cur_len = 0
        for i, w in enumerate(parts[:-1], start=1):
            cur_len += len(w) + 1  # include a space
            delta = abs(cur_len - 1 - target)
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        return " ".join(parts[:best_idx]) + "\n" + " ".join(parts[best_idx:])

    def _format_stat_label(key: str) -> str:
        # Remove legacy 'louvain_' token if present, then prettify
        cleaned = str(key).replace("louvain_", "")
        return cleaned.replace("_", " ").title()

    # Single-series mode (backward compatible)
    if multi_values is None or not stat_keys or len(stat_keys) == 1:
        if not rows:
            print("No rows to plot.")
            return
        rows = sorted(rows, key=lambda r: r[1], reverse=not sort_ascending)
        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]

        fig, ax = plt.subplots(figsize=figsize)
        x = list(range(len(labels)))
        ax.bar(x, values, color=bar_color)

        y_label = _format_stat_label(stat_key or "Value")
        ax.set_ylabel(_wrap_axis_label(y_label), fontweight="bold")
        if log_scale:
            ax.set_yscale("log")
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
        fig.subplots_adjust(bottom=0.28)
        fig.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return

    # Multi-series grouped bars
    # Convert each stat's rows list to a dict: label -> value
    series_dicts: Dict[str, Dict[str, float]] = {}
    for k in stat_keys:
        d = {}
        for lbl, val in multi_values.get(k, []):
            d[lbl] = val
        series_dicts[k] = d

    # Intersect labels across all stats to ensure aligned groups
    common_labels = None
    for d in series_dicts.values():
        lbls = set(d.keys())
        common_labels = lbls if common_labels is None else (common_labels & lbls)
    if not common_labels:
        print("No common labels across requested stats; nothing to plot.")
        return
    common_labels = list(common_labels)

    # Sort by the first statistic
    primary = stat_keys[0]
    common_labels.sort(
        key=lambda lbl: series_dicts[primary][lbl], reverse=not sort_ascending
    )

    # Prepare values in consistent order
    values_per_series: List[List[float]] = []
    for k in stat_keys:
        vals = [series_dicts[k][lbl] for lbl in common_labels]
        values_per_series.append(vals)

    n_groups = len(common_labels)
    n_series = len(stat_keys)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)
    total_width = 0.85
    bar_width = total_width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

    # Use same primary colors as stacked counts: red then blue
    palette = ["#d62728", "#1f77b4"]  # red, blue
    cmap = plt.get_cmap("tab10")
    # Determine if we should use dual y-axes: modularity (0..1) vs num_communities (unbounded)
    use_dual_axes = False
    right_axis_indices: List[int] = []
    stat_keys_lower = [k.lower() for k in stat_keys]
    if any("modularity" in k for k in stat_keys_lower) and any(
        "num_communities" in k for k in stat_keys_lower
    ):
        use_dual_axes = True
        right_axis_indices = [
            i for i, k in enumerate(stat_keys_lower) if "modularity" in k
        ]

    ax_right = ax.twinx() if use_dual_axes else None

    for i, (k, vals) in enumerate(zip(stat_keys, values_per_series)):
        color = palette[i] if i < len(palette) else cmap(i % 10)
        target_ax = ax_right if (use_dual_axes and i in right_axis_indices) else ax
        target_ax.bar(
            x + offsets[i],
            vals,
            width=bar_width * 0.95,
            label=_format_stat_label(k),
            color=color,
        )

    # Axis labels and styles
    if use_dual_axes and ax_right is not None:
        ax.set_ylabel(_wrap_axis_label("Number Of Communities"), fontweight="bold")
        ax_right.set_ylabel(_wrap_axis_label("Modularity"), fontweight="bold")
        try:
            ax_right.set_ylim(0.0, 1.0)
        except Exception:
            pass
    else:
        if n_series > 1:
            ax.set_ylabel("")
        else:
            y_label = stat_keys[0].replace("_", " ").title()
            ax.set_ylabel(_wrap_axis_label(y_label), fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        common_labels, rotation=rotate_labels, ha="right", rotation_mode="anchor"
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
    if use_dual_axes and ax_right is not None:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False)
    else:
        ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.28)

    fig.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-circuit bar chart for a chosen GS stats key."
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="GS precision level to scan (default: 8)",
    )
    parser.add_argument(
        "--stat",
        action="append",
        required=True,
        help="Statistic key to plot (repeat for multiple). Supports comma-separated list within one flag.",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order (default: ascending)",
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
        "--color",
        type=str,
        default="#1f77b4",
        help="Bar color (default: matplotlib blue)",
    )
    parser.add_argument(
        "--log", action="store_true", help="Use logarithmic scale on the Y axis"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    stats_files = find_stats_files_for_precision(base_dir, args.precision)

    # Flatten stats list and support comma-separated values per flag
    raw_stats: List[str] = []
    for s in args.stat or []:
        raw_stats.extend([p.strip() for p in str(s).split(",") if p.strip()])
    stat_keys = list(dict.fromkeys(raw_stats))  # de-dup, preserve order

    if not stat_keys:
        print("No statistic keys provided.")
        return

    if len(stat_keys) == 1:
        rows = build_dataset(stats_files, stat_keys[0])
        if not rows:
            print(
                f"No usable data found for stat '{stat_keys[0]}' at precision {args.precision}."
            )
            return
        safe_stat = stat_keys[0].replace(".", "_")
        output_pdf = os.path.join(
            base_dir,
            f"allcircuits_gs_prec{args.precision}_{safe_stat}_histogram.pdf",
        )
        plot_stat_histogram(
            rows=rows,
            stat_key=stat_keys[0],
            output_pdf_path=output_pdf,
            font_size=args.fontsize,
            figsize=(args.figwidth, args.figheight),
            sort_ascending=not args.desc,
            bar_color=args.color,
            log_scale=args.log,
        )
        print(f"Saved plot to: {output_pdf}")
        return

    # Multi-stat mode
    multi_values: Dict[str, List[Tuple[str, float]]] = {}
    for k in stat_keys:
        rows_k = build_dataset(stats_files, k)
        if rows_k:
            multi_values[k] = rows_k
        else:
            multi_values[k] = []

    # Check any data exists
    if not any(multi_values.values()):
        print(
            f"No usable data found for requested stats at precision {args.precision}."
        )
        return

    safe_stats = "_and_".join(s.replace(".", "_") for s in stat_keys)
    output_pdf = os.path.join(
        base_dir,
        f"allcircuits_gs_prec{args.precision}_{safe_stats}_histogram.pdf",
    )

    plot_stat_histogram(
        output_pdf_path=output_pdf,
        font_size=args.fontsize,
        figsize=(args.figwidth, args.figheight),
        sort_ascending=not args.desc,
        rotate_labels=90,
        log_scale=args.log,
        multi_values=multi_values,
        stat_keys=stat_keys,
    )
    print(f"Saved plot to: {output_pdf}")
    print(f"Saved plot to: {output_pdf}")


if __name__ == "__main__":
    main()
