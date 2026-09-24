# k-PBC Workflow Artifacts

The local and server workflows use stable folder names with no date or time suffix.

- `candidate.kpbc.jsonl`: compiler output in the k-PBC JSONL format.
- `routed.json`: router output, one JSON array of routed operations with `logical_qubits`, `routed_l`, and `routed_area`.
- `trace.jsonl`: decoder-timed trace events plus a final summary record.
- `summary.json`: decoder model name, makespan, and extrapolation count.
- `run.log`: commands and artifact paths for the run.
- `manifest.jsonl`: server run rows, one row per `(circuit, decoder, compiler_mode, k)`.
- `commands.txt`: shell commands consumed by `run_server.sh`.
- `run_server.sh`: parallel runner controlled by `JOBS`, defaulting to 8.
