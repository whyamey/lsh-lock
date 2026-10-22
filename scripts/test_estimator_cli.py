"""Bounded native CLI integration tests; synthetic inputs, no lab machines or network."""
import csv
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("LSH_BIN", ROOT / "target/release/lsh-lock")).resolve()


def write_indices(path, subsets):
    path.write_bytes(struct.pack("<Q", len(subsets)) + b"".join(
        struct.pack("<Q", len(sub)) + struct.pack("<" + "Q" * len(sub), *sub)
        for sub in subsets))


def read_indices(path):
    data = memoryview(path.read_bytes()); offset = 0
    def number():
        nonlocal offset
        value, = struct.unpack_from("<Q", data, offset); offset += 8
        return value
    result = [[number() for _ in range(number())] for _ in range(number())]
    assert offset == len(data)
    return result


class EstimatorCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.templates = self.root / "readings"
        self.templates.mkdir(); rng = random.Random(123)
        for identity in reversed(range(16)):
            folder = self.templates / f"id_{identity:02d}"; folder.mkdir()
            base = [rng.randrange(2) for _ in range(64)]
            for reading in (2, 0, 1):
                bits = base.copy()
                if reading:
                    for i in rng.sample(range(64), 3 if reading == 1 else 5): bits[i] ^= 1
                (folder / f"{reading:02d}.bits").write_text(",".join(map(str, bits)) + "\n")
        self.subsets = [list(range(24)), list(range(20)) + list(range(24, 28)),
                        list(range(24, 48)), list(range(40, 64))]
        self.indices = self.root / "idx.bin"; write_indices(self.indices, self.subsets)

    def command(self, *args, success=True):
        result = subprocess.run([str(BINARY), *map(str, args)], text=True, capture_output=True,
                                timeout=30, env={**os.environ, "RAYON_NUM_THREADS": "2"})
        if success: self.assertEqual(result.returncode, 0, result.stderr)
        else: self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def suite(self, tag, *args, success=True):
        return self.command("entropy-suite", "-i", self.indices, "-t", self.templates, "-n", "4",
                            "--heavy", "2", "--samples", "1000", "--seed", "67", "--threads", "2",
                            "--csv", self.root / f"{tag}.csv", "--json", self.root / f"{tag}.json",
                            *args, success=success)

    def test_real_and_null_preserve_labels_subsets_and_report_sparse_heavy_rows(self):
        records = []
        for tag, extra in (("real", []), ("null", ["--null"])):
            summary = json.loads(self.suite(tag, *extra).stdout)
            record = json.loads((self.root / f"{tag}.json").read_text()); records.append(record)
            self.assertEqual(record["subsets"], self.subsets)
            self.assertEqual(record["synthetic_null"], tag == "null")
            self.assertEqual(record["identity_labels"], 16)
            self.assertEqual(record["readings"], 48)
            self.assertTrue(record["all_values_finite"])
            self.assertEqual(record["heavy_evaluated"], 2)
            self.assertEqual(summary["summary"]["daugman"]["evaluated"], 4)
            self.assertEqual(summary["summary"]["guess_tree"]["evaluated"], 2)
            self.assertFalse(set(record["train_identity_labels"]) & set(record["test_identity_labels"]))
            self.assertTrue(all(name.endswith("/00.bits") for name in record["representatives"]))
            with (self.root / f"{tag}.csv").open() as handle: rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["guess_tree"] != "" for row in rows), 2)
        self.assertEqual(records[0]["representatives"], records[1]["representatives"])
        self.assertEqual(records[0]["train_identity_labels"], records[1]["train_identity_labels"])

    def test_pair_union_binary_roundtrip_and_expected_output_reuse(self):
        target = self.root / "unions.bin"
        args = ("pair-unions", "-i", self.indices, "-o", target, "--pool", "4",
                "--top", "1", "--random", "2", "--seed", "67")
        self.command(*args); original = target.read_bytes()
        unions = read_indices(target)
        self.assertEqual(len(unions), 3)
        self.assertEqual(unions[0], sorted(set(self.subsets[0]) | set(self.subsets[1])))
        possible = {tuple(sorted(set(a) | set(b))) for i, a in enumerate(self.subsets) for b in self.subsets[i+1:]}
        self.assertTrue(all(tuple(u) in possible for u in unions))
        self.command(*args); self.assertEqual(original, target.read_bytes())
        result = self.command("entropy-suite", "-i", target, "-t", self.templates, "-n", "3",
                              "--heavy", "1", "--samples", "1000", "--threads", "2")
        self.assertEqual(json.loads(result.stdout)["subsets_evaluated"], 3)

    def test_invalid_index_count_and_out_of_range_indices_fail_before_output(self):
        self.command("entropy-suite", "-i", self.indices, "-t", self.templates, "-n", "5", success=False)
        write_indices(self.indices, [[0, 64]] * 4)
        self.suite("bad", success=False)
        self.assertFalse((self.root / "bad.csv").exists())

    def test_invalid_pair_counts_and_input_overwrite_are_rejected(self):
        target = self.root / "bad.bin"
        self.command("pair-unions", "-i", self.indices, "-o", target, "--pool", "2",
                     "--top", "2", "--random", "0", success=False)
        self.assertFalse(target.exists())
        before = self.indices.read_bytes()
        self.command("pair-unions", "-i", self.indices, "-o", self.indices, success=False)
        self.assertEqual(before, self.indices.read_bytes())
        write_indices(self.indices, [list(range(24))])
        self.command("pair-unions", "-i", self.indices, "-o", target, "--top", "0", "--random", "1", success=False)

    def test_nonbinary_readings_and_empty_training_split_are_rejected(self):
        (self.templates / "id_00/00.bits").write_text("2," + ",".join("0" for _ in range(63)))
        self.suite("bad_bits", success=False)
        self.assertFalse((self.root / "bad_bits.csv").exists())
        self.suite("bad_split", "--train-frac", "0", success=False)

    def test_nonfinite_upstream_values_are_reported_and_fail(self):
        for file in self.templates.glob("*/*"): file.write_text(",".join("0" for _ in range(64)))
        result = self.suite("constant", success=False)
        record = json.loads(result.stdout)
        self.assertFalse(record["all_values_finite"])
        self.assertGreater(record["summary"]["daugman"]["nonfinite"], 0)
        self.assertTrue((self.root / "constant.json").is_file())


if __name__ == "__main__": unittest.main()
