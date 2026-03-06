#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "ERROR: missing shard index argument"
  exit 1
fi

SHARD_INDEX="$1"
NUM_SHARDS="${NUM_SHARDS:-1}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-circuit_benchmarks_ard}"
USE_SHARD_OUTPUT="${USE_SHARD_OUTPUT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
INSTANCES_FILE="${INSTANCES_FILE:-}"
if [[ "${USE_SHARD_OUTPUT}" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR_BASE}_shard_${SHARD_INDEX}"
else
  OUTPUT_DIR="${OUTPUT_DIR_BASE}"
fi

echo "Starting FTCircuitBench benchmark generation on $(hostname)"
echo "Working directory: $(pwd)"
echo "Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Skip existing outputs: ${SKIP_EXISTING}"
echo "Instances file: ${INSTANCES_FILE:-<none>}"

if [[ ! -f "generate_benchmarks.py" ]]; then
  echo "ERROR: generate_benchmarks.py is missing from the job sandbox."
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: No python interpreter found in container."
  exit 1
fi

GEN_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SHARD_INDEX}"
  --skip-fidelity
  --skip-sk
  --skip-pbc
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  GEN_ARGS+=(--skip-existing)
fi

if [[ -n "${INSTANCES_FILE}" ]]; then
  GEN_ARGS+=(--instances-file "${INSTANCES_FILE}")
fi

set -x
"${PYTHON_BIN}" generate_benchmarks.py "${GEN_ARGS[@]}"
set +x

echo "Finished benchmark generation."
