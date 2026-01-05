#!/usr/bin/env python3
"""
Comprehensive test runner for circuit fidelity calculations.
This script runs both rigorous and integration tests for fidelity calculations.
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

# Add the project root to the path
sys.path.append("..")


def run_tests(test_files: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Run pytest on the specified test files.

    Args:
        test_files: List of test file paths to run
        verbose: Whether to run tests in verbose mode

    Returns:
        Dictionary containing test results
    """
    results = {}

    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Running tests from: {test_file}")
        print(f"{'='*60}")

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", test_file]
        if verbose:
            cmd.append("-v")

        # Run the tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )
            end_time = time.time()

            # Parse results
            test_result = {
                "file": test_file,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0,
            }

            # Extract test summary if available
            if "passed" in result.stdout or "failed" in result.stdout:
                lines = result.stdout.split("\n")
                for line in lines:
                    if "passed" in line and "failed" in line:
                        test_result["summary"] = line.strip()
                        break

            results[test_file] = test_result

            # Print results
            if test_result["success"]:
                print(f"✅ Tests passed for {test_file}")
            else:
                print(f"❌ Tests failed for {test_file}")

            if verbose and result.stdout:
                print("\nTest output:")
                print(result.stdout)

            if result.stderr:
                print("\nTest errors:")
                print(result.stderr)

        except subprocess.TimeoutExpired:
            print(f"⏰ Tests timed out for {test_file}")
            results[test_file] = {
                "file": test_file,
                "return_code": -1,
                "stdout": "",
                "stderr": "Tests timed out after 5 minutes",
                "duration": 300,
                "success": False,
            }
        except Exception as e:
            print(f"💥 Error running tests for {test_file}: {e}")
            results[test_file] = {
                "file": test_file,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": 0,
                "success": False,
            }

    return results


def check_gridsynth_availability() -> bool:
    """Check if gridsynth is available for testing."""
    # Add cabal bin to PATH like in the notebook
    cabal_bin_path = os.path.expanduser("~/.cabal/bin")
    if cabal_bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cabal_bin_path + ":" + os.environ.get("PATH", "")

    try:
        result = subprocess.run(
            ["gridsynth", "--help"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate a summary report of test results."""
    report = []
    report.append("=" * 80)
    report.append("FIDELITY TEST SUMMARY REPORT")
    report.append("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["success"])
    failed_tests = total_tests - passed_tests

    report.append("\nOverall Results:")
    report.append(f"  Total test files: {total_tests}")
    report.append(f"  Passed: {passed_tests}")
    report.append(f"  Failed: {failed_tests}")
    report.append(f"  Success rate: {(passed_tests/total_tests)*100:.1f}%")

    # Detailed results
    report.append("\nDetailed Results:")
    for test_file, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = f"{result['duration']:.2f}s"
        report.append(f"  {status} {test_file} ({duration})")

        if not result["success"] and result["stderr"]:
            report.append(f"    Error: {result['stderr'][:200]}...")

    # Recommendations
    report.append("\nRecommendations:")
    if failed_tests == 0:
        report.append(
            "  🎉 All tests passed! Fidelity calculations are working correctly."
        )
    else:
        report.append("  ⚠️  Some tests failed. Please review the errors above.")
        report.append(
            "  💡 Check that gridsynth is installed and accessible if integration tests failed."
        )

    report.append("\nTest Coverage:")
    report.append("  - Rigorous unit tests with mocked decompositions")
    report.append("  - Integration tests with real gridsynth decompositions")
    report.append("  - Edge case testing (empty circuits, parameterized gates, etc.)")
    report.append("  - Error handling and boundary conditions")
    report.append("  - Performance and consistency checks")

    return "\n".join(report)


def main():
    """Main function to run all fidelity tests."""
    print("🚀 Starting comprehensive fidelity test suite...")

    # Check gridsynth availability
    gridsynth_available = check_gridsynth_availability()
    if gridsynth_available:
        print("✅ Gridsynth is available - integration tests will run")
    else:
        print("⚠️  Gridsynth not available - integration tests will be skipped")

    # Define test files
    test_files = [
        "tests/test_fidelity_rigorous.py",
        "tests/test_fidelity_integration.py",
    ]

    # Check if test files exist
    existing_tests = []
    for test_file in test_files:
        if os.path.exists(test_file):
            existing_tests.append(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")

    if not existing_tests:
        print("❌ No test files found!")
        return 1

    # Run tests
    print(f"\n📋 Running {len(existing_tests)} test files...")
    results = run_tests(existing_tests, verbose=True)

    # Generate and print summary report
    summary = generate_summary_report(results)
    print(f"\n{summary}")

    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"fidelity_test_results_{timestamp}.json"

    # Clean up results for JSON serialization
    json_results = {}
    for test_file, result in results.items():
        json_results[test_file] = {
            "file": result["file"],
            "return_code": result["return_code"],
            "duration": result["duration"],
            "success": result["success"],
            "summary": result.get("summary", ""),
            "error": result["stderr"] if result["stderr"] else None,
        }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {results_file}")

    # Return appropriate exit code
    failed_tests = sum(1 for r in results.values() if not r["success"])
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test file(s) failed!")
        return 1
    else:
        print("\n🎉 All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
