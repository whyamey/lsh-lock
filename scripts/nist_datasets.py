#!/usr/bin/env python3
"""Run the full-reading NIST protocol once each for face, iris test, and finger."""
import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from nist_entropy import load_readings, run, validate_options, write_json


def run_datasets(args):
    validate_options(args)
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {output}")
    root = args.data_root.expanduser().resolve()
    datasets = {
        "face": (args.face_templates or root / "face_embeddings_noah_nir").expanduser().resolve(),
        "iris": root / "iris_embeddings" / "test",
        "finger": root / "finger_embeddings",
    }
    # Validate EVERY input before launching any expensive native assessments.
    for source in datasets.values():
        load_readings(source)
    output.mkdir(parents=True)
    for name, source in datasets.items():
        run(SimpleNamespace(**{**vars(args), "templates": source, "output": output / name}))
    write_json(output / "complete.json", {
        "datasets": {name: {"source": str(source), "report": f"{name}/report.json"}
                     for name, source in datasets.items()},
        "trials_per_kind": args.trials, "controls": not args.no_controls,
        "note": "Separate full-reading diagnostics; no scores enter the per-subset security minimum.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-templates", type=Path,
                        help="Explicit face export, e.g. Ben's deduplicated face directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh directory")
    parser.add_argument("--nist-bin", default="ea_non_iid")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--keep-inputs", action="store_true")
    args = parser.parse_args()
    try:
        run_datasets(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"nist-datasets: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
