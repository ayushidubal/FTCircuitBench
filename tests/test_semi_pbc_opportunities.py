from __future__ import annotations

import json
import subprocess
import sys

from ftcircuitbench.semi_pbc.opportunities import analyze_pbc_text_opportunities


def test_analyze_pbc_text_opportunities_reports_weight_and_run_stats():
    text = (
        "qreg q[4];\n"
        "t_pauli +ZZZI;\n"
        "t_pauli +ZIZZ;\n"
        "m_pauli +ZZII;\n"
        "t_pauli +IIZZ;\n"
        "t_pauli +IIZZ;\n"
    )

    report = analyze_pbc_text_opportunities(
        text,
        k=2,
        measurement_reducer="none",
    )

    assert report["format"] == "semi-pbc-opportunity-report"
    assert report["k"] == 2
    assert report["input_op_count"] == 5
    assert report["op_counts"] == {"m_pauli": 1, "t_pauli": 4}
    assert report["high_weight_counts"] == {"m_pauli": 0, "t_pauli": 2}
    assert report["weight_histogram"]["t_pauli"] == {"2": 2, "3": 2}
    assert report["weight_histogram"]["m_pauli"] == {"2": 1}
    assert report["rotation_runs"]["count"] == 1
    assert report["rotation_runs"]["max_length"] == 2
    assert report["rotation_runs"]["length_histogram"] == {"2": 1}
    assert report["repeated_pauli_supports"]["count"] == 1
    assert report["repeated_pauli_supports"]["max_multiplicity"] == 2
    assert report["compile_comparisons"]["none"]["output_op_count"] == 10
    assert report["compile_comparisons"]["local-window"]["output_op_count"] == 10
    assert report["compile_comparisons"]["rotation-dp"]["output_op_count"] == 8


def test_opportunity_report_repeated_support_ignores_pauli_labels():
    report = analyze_pbc_text_opportunities(
        "qreg q[2];\nt_pauli +XX;\nt_pauli +ZZ;\n",
        k=1,
        measurement_reducer="none",
    )

    assert report["repeated_pauli_supports"] == {
        "count": 1,
        "max_multiplicity": 2,
    }


def test_opportunity_report_adjacent_overlap_skips_reduced_identity_ops():
    report = analyze_pbc_text_opportunities(
        "qreg q[2];\nm_pauli +ZI;\nm_pauli +ZI;\nt_pauli +IZ;\n",
        k=1,
    )

    assert report["adjacent_support_overlap"]["pairs"] == 1
    assert report["adjacent_support_overlap"]["max_overlap"] == 0


def test_analyze_semi_pbc_opportunities_cli_writes_json(tmp_path):
    pbc = tmp_path / "toy.pbc"
    out = tmp_path / "report.json"
    pbc.write_text("qreg q[4];\nt_pauli +ZZZI;\nt_pauli +ZIZZ;\n")

    subprocess.run(
        [
            sys.executable,
            "analyze_semi_pbc_opportunities.py",
            "--pbc",
            str(pbc),
            "--k",
            "2",
            "--out",
            str(out),
            "--measurement-reducer",
            "none",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(out.read_text())
    assert report["format"] == "semi-pbc-opportunity-report"
    assert report["compile_comparisons"]["rotation-dp"]["output_op_count"] == 4
