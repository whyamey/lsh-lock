#!/usr/bin/env python3
"""Run the official SP 800-90B non-IID suite on labeled binary readings.

Each real run selects ONE reading per identity. Full readings are concatenated
in sorted identity order, without changing coordinate order or thresholds.
The ten stream outputs remain separate from the other entropy estimators.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

TESTS = (
    "Most Common Value", "Collision Test (for bit strings only)",
    "Markov Test (for bit strings only)", "Compression Test (for bit strings only)",
    "T-Tuple Test", "LRS Test", "Multi Most Common in Window Test",
    "Lag Prediction Test", "Multi Markov Model with Counting Test (MultiMMC)", "LZ78Y Test",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_readings(root):
    """Return original labels, sorted file manifest and unchanged binary rows."""
    rows, manifest, names, groups = [], [], [], []
    for folder in sorted(root.iterdir()):
        if folder.name.startswith("."):
            continue
        if not folder.is_dir():
            raise ValueError(f"Expected identity subdirectories, found {folder}")
        names.append(folder.name)
        group = []
        for file in sorted(folder.iterdir()):
            if file.name.startswith("."):
                continue
            if not file.is_file():
                raise ValueError(f"Expected a reading file: {file}")
            content = file.read_bytes()
            fields = [s.strip() for s in content.decode("utf-8").strip().rstrip(",").split(",")]
            if not fields or any(s not in ("0", "1") for s in fields):
                raise ValueError(f"Expected comma-separated binary values: {file}")
            row = np.array(fields, dtype=np.uint8)
            if rows and len(row) != len(rows[0]):
                raise ValueError(f"Inconsistent reading length: {file}")
            group.append(len(rows))
            rows.append(row)
            manifest.append({"identity": folder.name, "reading": file.name,
                             "relative_path": file.relative_to(root).as_posix(),
                             "file_sha256": sha256(content)})
        if not group:
            raise ValueError(f"Identity has no reading files: {folder}")
        groups.append(np.array(group, dtype=np.int64))
    if len(names) < 2:
        raise ValueError("At least two identity labels are required")
    return np.stack(rows), manifest, names, groups


def select_rows(groups, seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(group) for group in groups], dtype=np.int64)


def control_bits(identities, width, seed):
    # Preserve the seed policy used in the completed face/iris/fingerprint runs.
    rng = np.random.default_rng(np.random.SeedSequence([20261015, seed]))
    return rng.integers(0, 2, size=(identities, width), dtype=np.uint8)


def parse_result(result):
    """Reject missing/non-finite tests instead of silently reporting a partial suite."""
    if result.get("errorLevel") != 0:
        raise ValueError("Official NIST tool reported an error; inspect the raw log/JSON")
    tests, overall, seen_overall = {}, None, False
    for case in result.get("testCases", []):
        name = case.get("testCaseDesc")
        if name == "Overall":
            if seen_overall:
                raise ValueError("Duplicate Overall result")
            seen_overall = True
            overall = case.get("hAssessed")
        elif name in TESTS:
            if name in tests:
                raise ValueError(f"Duplicate estimator: {name}")
            value = case.get("hOriginal", case.get("tTupleRes", case.get("lrsRes")))
            if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Missing/non-finite estimate: {name}; inspect the raw log/JSON")
            tests[name] = float(value)
        else:
            raise ValueError(f"Unexpected estimator name: {name!r}")
    if set(tests) != set(TESTS):
        raise ValueError(f"Not all ten tests returned estimates: missing {set(TESTS) - set(tests)}")
    if overall is None or not np.isfinite(overall) or abs(overall - min(tests.values())) > 1e-12:
        raise ValueError("Overall result disagrees with the individual estimates")
    return float(overall), {name: tests[name] for name in TESTS}


def summarize(records):
    result = {}
    for kind in sorted({r["kind"] for r in records}):
        runs = [r for r in records if r["kind"] == kind]
        def stats(values):
            return {"minimum": float(np.min(values)), "p05": float(np.quantile(values, .05)),
                    "median": float(np.median(values)), "maximum": float(np.max(values))}
        result[kind] = {"runs": len(runs), "suite_minimum": stats([r["suite_minimum"] for r in runs]),
            "per_estimator": {name: stats([r["estimates"][name] for r in runs]) for name in TESTS},
            "limiting_counts": dict(Counter(t for r in runs for t in r["limiting_estimators"]))}
    return result


def validate_options(args):
    if (args.trials < 1 or args.jobs < 1 or args.seed < 0
            or not np.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("Trials, jobs and timeout must be positive; seed must be nonnegative")
    binary = Path(shutil.which(args.nist_bin) or args.nist_bin).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Official ea_non_iid executable not found: {binary}")
    return binary


def run(args):
    root, out = args.templates.expanduser().resolve(), args.output.expanduser().resolve()
    if out.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {out}")
    binary = validate_options(args)
    binary_digest = sha256(binary.read_bytes())
    bits, manifest, names, groups = load_readings(root)
    width, samples = bits.shape[1], len(names) * bits.shape[1]
    selections = np.stack([select_rows(groups, seed) for seed in range(args.seed, args.seed + args.trials)])
    for choice in selections:
        if [manifest[int(i)]["identity"] for i in choice] != names:
            raise ValueError("Selection does not contain exactly one reading per identity")
    out.mkdir(parents=True)
    (out / "raw").mkdir()
    np.save(out / "selected_rows.npy", selections)
    write_json(out / "input_manifest.json", {"source_directory": str(root), "identities": names,
        "readings_per_identity": dict(zip(names, map(len, groups))), "rows": manifest,
        "bits_per_reading": width, "reading_count": len(bits), "ordered_bits_sha256": sha256(bits.tobytes())})
    warnings = ["Exploratory stream diagnostics, not validated template/fuzzy entropy or key-security bounds.",
        "Trials reuse identities; the observed minimum is not a bound over all possible readings.",
        "Serialization and coordinate order can affect NIST outputs; no reordering or score normalization is applied."]
    if samples < 1000000:
        warnings.append(f"Each run has {samples:,} binary samples, below NIST's one-million-sample requirement.")
    for warning in warnings:
        print(warning, file=sys.stderr, flush=True)
    kinds = ["real"] if args.no_controls else ["real", "iid_control"]
    records = []
    with tempfile.TemporaryDirectory(prefix="lsh-lock-nist-") as temporary:
        tmp = Path(temporary)
        def assess(kind, trial):
            seed = args.seed + trial
            raw = bits[selections[trial]] if kind == "real" else control_bits(len(names), width, seed)
            key = f"{kind}_{trial:04d}"
            source = (out if args.keep_inputs else tmp) / f"{key}.bin"
            raw.ravel(order="C").tofile(source)  # One byte per bit, NOT packed bytes.
            digest = sha256(source.read_bytes())
            native_path = out / "raw" / f"{key}.json"
            log_path = native_path.with_suffix(".log")
            command = [str(binary), "-i", "-a", "-vv", "-o", str(native_path), str(source), "1"]
            with log_path.open("x") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=args.timeout)
            if completed.returncode != 0:
                raise RuntimeError(f"NIST exited {completed.returncode}; inspect {log_path}")
            native = json.loads(native_path.read_text())
            minimum, estimates = parse_result(native)
            if native.get("sha256") != digest:
                raise ValueError(f"NIST input hash disagrees: {native_path}")
            if not args.keep_inputs:
                source.unlink()
            return {"kind": kind, "trial": trial, "seed": seed, "samples": samples,
                "input_sha256": digest, "tool_version": native.get("toolVersion"),
                "suite_minimum": minimum, "estimates": estimates,
                "limiting_estimators": [k for k, v in estimates.items() if v == minimum],
                "raw_json": native_path.relative_to(out).as_posix()}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(assess, kind, i) for i in range(args.trials) for kind in kinds]
            try:
                with (out / "completed_runs.jsonl").open("x") as checkpoint:
                    for future in as_completed(futures):
                        record = future.result()
                        records.append(record)
                        checkpoint.write(json.dumps(record, allow_nan=False) + "\n")
                        checkpoint.flush()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    records.sort(key=lambda r: (r["kind"], r["trial"]))
    if sha256(binary.read_bytes()) != binary_digest:
        raise ValueError("NIST executable changed during assessment; no success report written")
    summary = summarize(records)
    write_json(out / "report.json", {"schema_version": 1, "seed_start": args.seed,
        "trials_per_kind": args.trials, "bits_per_reading": width, "identities_per_run": len(names),
        "samples_per_run": samples, "unique_real_selections": len(np.unique(selections, axis=0)),
        "selection": "One reading per identity; sorted identity/filename order, original coordinate order.",
        "control_seed_policy": "Independent fair bits: NumPy default_rng(SeedSequence([20261015, selection_seed])).",
        "numpy_version": np.__version__, "nist_binary": str(binary), "nist_binary_sha256": binary_digest,
        "official_source": "https://github.com/usnistgov/SP800-90B_EntropyAssessment",
        "units": "bits per input bit; no conversion to template security bits", "warnings": warnings,
        "input_manifest": "input_manifest.json", "selection_indices": "selected_rows.npy",
        "summaries": summary, "runs": records})
    with (out / "all_scores.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "trial", "seed", "estimator", "bits_per_input_bit", "suite_minimum"])
        for r in records:
            for name, value in r["estimates"].items():
                writer.writerow([r["kind"], r["trial"], r["seed"], name, value, r["suite_minimum"]])
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", "-t", type=Path, required=True, help="<identity>/<reading> comma-separated 0/1 files")
    parser.add_argument("--output", "-o", type=Path, required=True, help="New directory; existing outputs are refused")
    parser.add_argument("--nist-bin", default="ea_non_iid", help="Official NIST executable, or ea_non_iid on PATH")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180, help="Timeout per native NIST run in seconds")
    parser.add_argument("--no-controls", action="store_true", help="Omit the default length-matched random-bit controls")
    parser.add_argument("--keep-inputs", action="store_true", help="Retain each byte-per-bit input as well as its hash")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"nist-entropy: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
