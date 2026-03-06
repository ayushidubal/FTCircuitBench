python generate_benchmarks.py \
  --output-dir "circuit_benchmarks_ard_1" \
  --skip-fidelity \
  --skip-sk \
  --skip-pbc

in generate_benchmarks.py
    set DEFAULT_GS_PRECISIONS = [3, 10]

for CHTC sharded execution, see:
    chtc/ftc_generate.sub
    chtc/run_generate_benchmarks_chtc.sh
