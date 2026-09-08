from __future__ import annotations

import json
import subprocess
import sys

from ftcircuitbench.semi_pbc.structure import compare_pbc_texts, summarize_pbc_text


def test_summarize_pbc_text_reports_ai_window_eligibility():
    text = (
        "qreg q[7];\n"
        "t_pauli +ZZZIIII;\n"
        "t_pauli +IZZZIII;\n"
        "t_pauli +IIZZZII;\n"
        "t_pauli +IIIZZZI;\n"
        "m_pauli +IIIIIZZ;\n"
    )

    summary = summarize_pbc_text(text, k=2)

    assert summary["source_op_count"] == 5
    assert summary["op_counts"] == {"m_pauli": 1, "t_pauli": 4}
    assert summary["high_weight_counts"] == {"m_pauli": 0, "t_pauli": 4}
    assert summary["rotation_windows"]["candidate_count"] == 1
    assert summary["rotation_windows"]["eligible_count"] == 1
    assert summary["rotation_windows"]["union_width_histogram"] == {"6": 1}
    assert summary["adjacent_support_overlap"]["nonzero_pairs"] == 4
    assert summary["adjacent_support_overlap"]["total_overlap"] == 7


def test_compare_pbc_texts_reports_metric_deltas():
    compact = (
        "qreg q[7];\n"
        "t_pauli +ZZZIIII;\n"
        "t_pauli +IZZZIII;\n"
        "t_pauli +IIZZZII;\n"
        "t_pauli +IIIZZZI;\n"
    )
    spread = (
        "qreg q[7];\n"
        "t_pauli +ZZZIIZZ;\n"
        "t_pauli +IZZIIZZ;\n"
        "t_pauli +ZIIZIZZ;\n"
        "t_pauli +IIZIZZZ;\n"
    )

    comparison = compare_pbc_texts(
        compact,
        spread,
        k=2,
        left_label="compact",
        right_label="spread",
    )

    assert comparison["left"]["label"] == "compact"
    assert comparison["right"]["label"] == "spread"
    assert comparison["left"]["rotation_windows"]["eligible_count"] == 1
    assert comparison["right"]["rotation_windows"]["eligible_count"] == 0
    assert comparison["deltas"]["ai_eligible_window_count"] == -1


def test_compare_pbc_structures_cli_writes_json(tmp_path):
    left = tmp_path / "left.pbc"
    right = tmp_path / "right.pbc"
    out = tmp_path / "comparison.json"
    left.write_text("qreg q[4];\nt_pauli +ZZZI;\nt_pauli +IZZZ;\n")
    right.write_text("qreg q[4];\nt_pauli +ZZZZ;\nt_pauli +ZZZZ;\n")

    subprocess.run(
        [
            sys.executable,
            "compare_pbc_structures.py",
            "--left",
            str(left),
            "--right",
            str(right),
            "--left-label",
            "left",
            "--right-label",
            "right",
            "--k",
            "2",
            "--out",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(out.read_text())
    assert report["format"] == "semi-pbc-structure-comparison"
    assert report["left"]["label"] == "left"
    assert report["right"]["label"] == "right"
