from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


_QREG_RE = re.compile(r"\bqreg\s+\w+\[(\d+)\]\s*;")
_SIZE_RE = re.compile(r"(.+?_(?:0|[1-9][0-9]*)q)(?:_|$)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a server k-PBC run manifest.")
    parser.add_argument("--pbc-dir", required=True, type=Path)
    parser.add_argument("--ct-dir", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--compiler-modes", nargs="+", required=True)
    parser.add_argument("--k-values", required=True)
    parser.add_argument("--decoders", required=True, type=Path)
    parser.add_argument("--d", required=True, type=_positive_int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir = args.out_dir / "decoder_models"
    decoder_dir.mkdir(exist_ok=True)
    decoders = _load_decoders(args.decoders)
    decoder_paths = _write_decoder_models(decoder_dir, decoders)
    rows = _build_manifest_rows(args, decoders, decoder_paths)
    _write_manifest(args.out_dir / "manifest.jsonl", rows)
    _write_server_script(args.out_dir / "run_server.sh", rows)
    _write_readme(args.out_dir / "README.md", rows)
    print(f"wrote {args.out_dir / 'manifest.jsonl'} rows={len(rows)}")
    return 0


def _build_manifest_rows(
    args: argparse.Namespace,
    decoders: list[dict[str, Any]],
    decoder_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = _smallest_pbc_per_family(args.pbc_dir)
    for family, pbc_path, n in selected:
        for decoder in decoders:
            decoder_name = decoder["name"]
            for compiler_mode in args.compiler_modes:
                input_path, input_kind = _input_for_mode(
                    compiler_mode=compiler_mode,
                    pbc_path=pbc_path,
                    ct_dir=args.ct_dir,
                    family=family,
                    n=n,
                )
                for k in _expand_k_values(args.k_values, n):
                    run_dir = (
                        args.out_dir
                        / "runs"
                        / family
                        / pbc_path.stem
                        / decoder_name
                        / compiler_mode
                        / f"k_{k}"
                    )
                    rows.append(
                        {
                            "family": family,
                            "input": str(input_path),
                            "input_kind": input_kind,
                            "pbc": str(pbc_path),
                            "circuit_name": pbc_path.stem,
                            "n": n,
                            "decoder": decoder_name,
                            "decoder_model": str(decoder_paths[decoder_name]),
                            "compiler_mode": compiler_mode,
                            "k": k,
                            "d": args.d,
                            "run_dir": str(run_dir),
                            "command": _run_command(
                                input_path=input_path,
                                input_kind=input_kind,
                                run_dir=run_dir,
                                compiler_mode=compiler_mode,
                                decoder_model=decoder_paths[decoder_name],
                                k=k,
                                d=args.d,
                            ),
                        }
                    )
    return rows


def _run_command(
    *,
    input_path: Path,
    input_kind: str,
    run_dir: Path,
    compiler_mode: str,
    decoder_model: Path,
    k: int,
    d: int,
) -> str:
    input_flag = "--qasm" if input_kind == "qasm" else "--pbc"
    return (
        "python3 scripts/run_kpbc_smoke.py "
        f"{input_flag} {shlex.quote(str(input_path))} "
        f"--out-dir {shlex.quote(str(run_dir))} "
        f"--k {k} "
        f"--compiler-mode {compiler_mode} "
        '--router-bin "$ROUTER_BIN" '
        '${TRACEGEN_REPO:+--tracegen-repo "$TRACEGEN_REPO"} '
        f"--decoder-model {shlex.quote(str(decoder_model))} "
        f"--d {d}"
    )


def _smallest_pbc_per_family(pbc_dir: Path) -> list[tuple[str, Path, int]]:
    best: dict[str, tuple[Path, int]] = {}
    for path in sorted(pbc_dir.rglob("*_pbc_post_opt.txt")):
        try:
            family = path.relative_to(pbc_dir).parts[0]
        except (IndexError, ValueError):
            continue
        n = _qreg_qubit_count(path)
        current = best.get(family)
        if current is None or (n, str(path)) < (current[1], str(current[0])):
            best[family] = (path, n)
    return [(family, path, n) for family, (path, n) in sorted(best.items())]


def _input_for_mode(
    *,
    compiler_mode: str,
    pbc_path: Path,
    ct_dir: Path | None,
    family: str,
    n: int,
) -> tuple[Path, str]:
    if compiler_mode == "pbc-naive-ladder":
        return pbc_path, "pbc"
    if compiler_mode == "ct-segmented-litinski":
        if ct_dir is None:
            raise ValueError("ct-segmented-litinski requires --ct-dir")
        return _find_ct_qasm(ct_dir=ct_dir, family=family, pbc_path=pbc_path, n=n), "qasm"
    raise ValueError(f"unsupported compiler mode {compiler_mode!r}")


def _find_ct_qasm(*, ct_dir: Path, family: str, pbc_path: Path, n: int) -> Path:
    wanted_stem = _ct_stem_from_pbc_stem(pbc_path.stem)
    candidates = sorted((ct_dir / family).rglob("*.qasm"))
    if not candidates:
        candidates = sorted(ct_dir.rglob("*.qasm"))
    for candidate in candidates:
        if candidate.stem == wanted_stem:
            return candidate
    same_width = [candidate for candidate in candidates if _qreg_qubit_count(candidate) == n]
    if len(same_width) == 1:
        return same_width[0]
    if not same_width:
        raise ValueError(f"could not find C+T QASM for {pbc_path}")
    raise ValueError(f"multiple C+T QASM candidates match {pbc_path}")


def _ct_stem_from_pbc_stem(stem: str) -> str:
    match = _SIZE_RE.match(stem)
    if match is None:
        return stem
    return match.group(1)


def _load_decoders(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    decoders = raw.get("decoders", raw) if isinstance(raw, dict) else raw
    if not isinstance(decoders, list):
        raise ValueError("decoder file must contain a list or {'decoders': [...]}")
    cleaned = []
    for decoder in decoders:
        if not isinstance(decoder, dict) or not decoder.get("name"):
            raise ValueError("each decoder must be an object with a name")
        cleaned.append(dict(decoder))
    return cleaned


def _write_decoder_models(
    decoder_dir: Path, decoders: list[dict[str, Any]]
) -> dict[str, Path]:
    paths = {}
    for decoder in decoders:
        path = decoder_dir / f"{_safe_name(decoder['name'])}.json"
        path.write_text(json.dumps(decoder, indent=2, sort_keys=True) + "\n")
        paths[decoder["name"]] = path
    return paths


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def _write_server_script(path: Path, rows: list[dict[str, Any]]) -> None:
    commands_path = path.with_name("commands.txt")
    commands_path.write_text("\n".join(row["command"] for row in rows) + "\n")
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'JOBS="${JOBS:-8}"\n'
        ': "${ROUTER_BIN:?set ROUTER_BIN to fastkpbc binary}"\n'
        f'cat "{commands_path}" | xargs -P "$JOBS" -I {{}} bash -lc {{}}\n'
    )
    path.chmod(0o755)


def _write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "# k-PBC Server Run\n\n"
        "Set `ROUTER_BIN` to the built fastkpbc binary, then run:\n\n"
        "```bash\n"
        "JOBS=8 ./run_server.sh\n"
        "```\n\n"
        f"Manifest rows: {len(rows)}\n",
        encoding="utf-8",
    )


def _expand_k_values(spec: str, n: int) -> list[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if token == "n":
            values.append(n)
        elif token == "mid":
            values.append(max(1, n // 2))
        else:
            values.append(_positive_int(token))
    return values


def _qasm_qubit_count(path: Path) -> int:
    return _qreg_qubit_count(path)


def _qreg_qubit_count(path: Path) -> int:
    match = _QREG_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"could not find qreg in {path}")
    return int(match.group(1))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected integer >= 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
