Run from the `FTCircuitBench` repo root:

```bash
chmod +x chtc/run_generate_benchmarks_chtc.sh
condor_submit chtc/ftc_generate.sub
```

This queues `NUM_SHARDS` jobs from `chtc/ftc_generate.sub` (currently 46 shards).
Each shard runs:

```bash
python generate_benchmarks.py \
  --output-dir "circuit_benchmarks_ard_1" \
  --num-shards 46 \
  --shard-index <N> \
  --skip-fidelity \
  --skip-sk \
  --skip-pbc \
  --skip-existing \
  --instances-file chtc/remaining_gs_instances.txt
```

Outputs are merged into `circuit_benchmarks_ard_1/`.

To increase/decrease parallelism, change `NUM_SHARDS` in `chtc/ftc_generate.sub`.

Final run status and coverage summary are documented in `chtc/RUN_SUMMARY.md`.
