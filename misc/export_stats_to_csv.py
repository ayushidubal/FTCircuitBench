#!/usr/bin/env python3
import argparse
import csv
import json
import numbers
import os
import re
from typing import Any, Dict, List, Tuple


def find_stats_json_files(root_dir: str) -> List[str]:
    json_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("_stats.json"):
                json_paths.append(os.path.join(dirpath, filename))
    return json_paths


def extract_context_from_path(
    root_dir: str, file_path: str, stats: Dict[str, Any]
) -> Tuple[str, str, str, int]:
    rel = os.path.relpath(file_path, root_dir)
    parts = rel.split(os.sep)

    pipeline = (stats.get("pipeline") or "").upper()
    if not pipeline:
        if "GS" in parts:
            pipeline = "GS"
        elif "SK" in parts:
            pipeline = "SK"
        else:
            pipeline = ""  # unknown

    # Expect .../<category>/<instance>/<pipeline>/<param_dir>/<file>
    try:
        idx = parts.index(pipeline) if pipeline in parts else -1
    except ValueError:
        idx = -1

    instance = parts[idx - 1] if idx >= 1 else "unknown_instance"
    category = parts[idx - 2] if idx >= 2 else "unknown_category"

    # Parameter value
    param_value = stats.get("parameter_value")
    if not isinstance(param_value, int):
        # Fallback: parse from param dir name (precision_level_5 or recursion_degree_2)
        if idx >= 0 and len(parts) > idx + 1:
            m = re.search(r"(\d+)$", parts[idx + 1])
            if m:
                try:
                    param_value = int(m.group(1))
                except Exception:
                    param_value = None
    if not isinstance(param_value, int):
        # Fallback to gridsynth_precision or solovay_kitaev_recursion
        if pipeline == "GS" and isinstance(stats.get("gridsynth_precision"), int):
            param_value = int(stats["gridsynth_precision"])
        elif pipeline == "SK" and isinstance(
            stats.get("solovay_kitaev_recursion"), int
        ):
            param_value = int(stats["solovay_kitaev_recursion"])

    if not isinstance(param_value, int):
        param_value = -1

    return category, instance, pipeline, param_value


def sanitize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)
    return str(value)


def format_cell(field_name: str, value: Any) -> Any:
    # Special-case fidelity to use scientific notation with two decimals
    if field_name == "fidelity":
        try:
            if value is None or isinstance(value, str):
                return value if value is not None else ""
            return f"{float(value):.3f}"
        except Exception:
            return sanitize(value)
    # Default formatting
    return sanitize(value)


def combine_avg_std(avg_value: Any, std_value: Any) -> str:
    """Combine average and std values into a string of the form "avg ± std".
    Falls back gracefully if one is missing.
    """
    a = sanitize(avg_value)
    s = sanitize(std_value)
    a_str = "" if a is None else str(a)
    s_str = "" if s is None else str(s)
    if a_str and s_str:
        return f"{a_str} ± {s_str}"
    if a_str:
        return a_str
    if s_str:
        return f"± {s_str}"
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark JSON stats into CSVs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="circuit_benchmarks",
        help="Root directory to scan for *_stats.json",
    )
    parser.add_argument(
        "--ct-out",
        type=str,
        default=None,
        help="Output CSV path for Clifford+T stats (default: <input-dir>/ct_stats.csv)",
    )
    parser.add_argument(
        "--pbc-out",
        type=str,
        default=None,
        help="Output CSV path for PBC stats (default: <input-dir>/pbc_stats.csv)",
    )
    parser.add_argument(
        "--special-category",
        type=str,
        default="hamiltonians_5trotter",
        help="Category to isolate into a dedicated PBC-only CSV (excluded from ct/pbc)",
    )
    parser.add_argument(
        "--special-pbc-out",
        type=str,
        default=None,
        help="Output CSV path for the special category's PBC stats (default: <input-dir>/5trotter_hamiltonian_pbc_stats.csv)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    os.makedirs(input_dir, exist_ok=True)
    ct_out = args.ct_out or os.path.join(input_dir, "ct_stats.csv")
    pbc_out = args.pbc_out or os.path.join(input_dir, "pbc_stats.csv")
    special_category = str(args.special_category)
    special_pbc_out = args.special_pbc_out or os.path.join(
        input_dir, "5trotter_hamiltonian_pbc_stats.csv"
    )

    files = find_stats_json_files(input_dir)
    if not files:
        print(f"No *_stats.json files found under: {input_dir}")
        return

    # Define columns
    ct_columns = [
        "label",
        # General info
        "total gates",
        "depth",
        "clifford gates",
        "t gates",
        # "gate list",
        # "fidelity",
        # "fidelity method",
        # "transpilation clifford t time",
        # Interaction graph (CT)
        "I.G. graph density",
        "I.G. avg ± std degree",
        "I.G. modularity",
        "I.G. num communities",
    ]

    pbc_columns = [
        "label",
        # Pre/post key metrics
        "raw rotations",
        "optimized rotations",
        "rotation reduction",
        # Pauli weight
        "raw avg ± std pauli weight",
        "optimized avg ± std pauli weight",
        "avg pauli weight reduction",
        # Interaction degree
        # "raw avg ± std qubit interaction degree",
        # "optimized avg ± std qubit interaction degree",
        # "avg qubit interaction degree reduction",
        # PBC interaction graph (post)
        "I.G. avg ± std degree",
        "I.G. modularity",
        "I.G. num communities",
    ]

    ct_rows: List[Dict[str, Any]] = []
    pbc_rows: List[Dict[str, Any]] = []
    pbc_rows_special: List[Dict[str, Any]] = []

    for fp in files:
        try:
            with open(fp, "r") as f:
                stats = json.load(f)
        except Exception as e:
            print(f"Skipping unreadable JSON: {fp} ({e})")
            continue

        category, instance, pipeline, param_value = extract_context_from_path(
            input_dir, fp, stats
        )
        # Clean up instance for label brevity
        instance_clean = str(instance)
        instance_clean = instance_clean.replace("bandedcirculant", "")
        instance_clean = instance_clean.replace("triangular", "tri")
        # Normalize leftover separators after removals
        instance_clean = re.sub(r"__+", "_", instance_clean).strip("_")
        instance_part = instance_clean.replace("_", "-")
        instance_part = re.sub(r"-{2,}", "-", instance_part).strip("-")
        label = f"{instance_part}-{pipeline.lower()}-{param_value}"

        # CT row (skip if special category)
        if category != special_category:
            ct_row = {
                "label": label,
                # General info
                "total gates": stats.get("total_gate_count"),
                "depth": stats.get("depth"),
                "clifford gates": stats.get("clifford_gate_count"),
                "t gates": stats.get("total_t_family_count"),
                # "gate list": "",
                # "fidelity": stats.get("fidelity"),
                # "fidelity method": stats.get("fidelity_method"),
                # "transpilation clifford t time": stats.get("transpilation_clifford_t_time"),
                # CT interaction-graph (prefer canonical key; fallback to legacy for older JSONs)
                "I.G. graph density": (
                    stats.get("interaction_graph_density")
                    if "interaction_graph_density" in stats
                    else stats.get("interaction_graph_graph_density")
                ),
                "I.G. avg ± std degree": combine_avg_std(
                    stats.get("interaction_graph_avg_degree"),
                    stats.get("interaction_graph_std_degree"),
                ),
                "I.G. modularity": (
                    stats.get("interaction_graph_modularity")
                    if "interaction_graph_modularity" in stats
                    else stats.get("interaction_graph_louvain_modularity")
                ),
                "I.G. num communities": (
                    stats.get("interaction_graph_num_communities")
                    if "interaction_graph_num_communities" in stats
                    else stats.get("interaction_graph_louvain_num_communities")
                ),
            }
            # Populate gate list with counts, e.g., (h: 12, s: 8, cx: 4)
            gate_to_count = {}
            dcc = stats.get("detailed_clifford_counts") or {}
            if isinstance(dcc, dict):
                for g, c in dcc.items():
                    try:
                        c_int = int(c)
                        if c_int > 0:
                            gate_to_count[str(g)] = c_int
                    except Exception:
                        continue
            # Include T and Tdg counts if present
            try:
                t_c = int(stats.get("t_count") or 0)
                if t_c > 0:
                    gate_to_count["t"] = t_c
                tdg_c = int(stats.get("tdg_count") or 0)
                if tdg_c > 0:
                    gate_to_count["tdg"] = tdg_c
            except Exception:
                pass
            if gate_to_count:
                parts = [
                    f"{g}: {gate_to_count[g]}" for g in sorted(gate_to_count.keys())
                ]
                ct_row["gate list"] = f"({', '.join(parts)})"
            ct_rows.append(ct_row)

        # PBC row with derived percent reductions
        def pct(pre, post):
            try:
                if pre in (None, "N/A") or post in (None, "N/A"):
                    return ""
                pre_f = float(pre)
                post_f = float(post)
                if pre_f == 0:
                    return ""
                return f"{((pre_f - post_f) / pre_f) * 100:.2f}%"
            except Exception:
                return ""

        pre_rot_ops = stats.get("pre_opt_rotation_operators")
        post_rot_ops = stats.get("pbc_rotation_operators")
        pre_pw_avg = stats.get("pre_opt_avg_operator_pauli_weight")
        post_pw_avg = stats.get("pbc_avg_operator_pauli_weight")
        pre_pw_std = stats.get("pre_opt_std_operator_pauli_weight")
        post_pw_std = stats.get("pbc_std_operator_pauli_weight")
        # pre_deg_avg = stats.get("pre_opt_avg_qubit_interaction_degree")
        # post_deg_avg = stats.get("pbc_avg_qubit_interaction_degree")
        # pre_deg_std = stats.get("pre_opt_std_qubit_interaction_degree")
        # post_deg_std = stats.get("pbc_std_qubit_interaction_degree")

        pbc_row = {
            "label": label,
            "raw rotations": pre_rot_ops,
            "optimized rotations": post_rot_ops,
            "rotation reduction": pct(pre_rot_ops, post_rot_ops),
            # Pauli weight
            "raw avg ± std pauli weight": combine_avg_std(pre_pw_avg, pre_pw_std),
            "optimized avg ± std pauli weight": combine_avg_std(
                post_pw_avg, post_pw_std
            ),
            "avg pauli weight reduction": pct(pre_pw_avg, post_pw_avg),
            # Interaction degree
            # "raw avg ± std qubit interaction degree": combine_avg_std(pre_deg_avg, pre_deg_std),
            # "optimized avg ± std qubit interaction degree": combine_avg_std(post_deg_avg, post_deg_std),
            # "avg qubit interaction degree reduction": pct(pre_deg_avg, post_deg_avg),
            # PBC interaction-graph (pre)
            "I.G. avg ± std degree": combine_avg_std(
                stats.get("pbc_interaction_graph_avg_degree"),
                stats.get("pbc_interaction_graph_std_degree"),
            ),
            "I.G. modularity": (
                stats.get("pbc_interaction_graph_modularity")
                if "pbc_interaction_graph_modularity" in stats
                else stats.get("pbc_interaction_graph_louvain_modularity")
            ),
            "I.G. num communities": (
                stats.get("pbc_interaction_graph_num_communities")
                if "pbc_interaction_graph_num_communities" in stats
                else stats.get("pbc_interaction_graph_louvain_num_communities")
            ),
        }
        # Route PBC rows: special category goes to dedicated CSV; others to general PBC CSV
        if category == special_category:
            pbc_rows_special.append(pbc_row)
        else:
            pbc_rows.append(pbc_row)

    # Write CSVs
    with open(ct_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ct_columns)
        writer.writeheader()
        for row in ct_rows:
            writer.writerow({k: format_cell(k, row.get(k)) for k in ct_columns})

    with open(pbc_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pbc_columns)
        writer.writeheader()
        for row in pbc_rows:
            writer.writerow({k: sanitize(row.get(k)) for k in pbc_columns})

    print(f"Wrote CT stats: {ct_out}")
    print(f"Wrote PBC stats: {pbc_out}")

    # Write special-category PBC CSV
    with open(special_pbc_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pbc_columns)
        writer.writeheader()
        for row in pbc_rows_special:
            writer.writerow({k: sanitize(row.get(k)) for k in pbc_columns})
    print(f"Wrote special-category PBC stats ({special_category}): {special_pbc_out}")


if __name__ == "__main__":
    main()
