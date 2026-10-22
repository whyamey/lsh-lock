# Estimator commands used by the parent evaluation

The base snapshot `63376a9` contains `src/estimators.rs` but did not register it
in the executable. `entropy-suite` and `pair-unions` now call those existing
functions. The estimator source and its calculations are unchanged. This
restores the command forms used by `sample-ind-correct/eval/run_server.sh`.

## Per-subset suite

```bash
./target/release/lsh-lock entropy-suite \
  -i indices.bin -t /path/to/lab_templates -n 10000 --csv suite.csv
./target/release/lsh-lock entropy-suite \
  -i indices.bin -t /path/to/lab_templates -n 10000 --null
```

`-i` is the existing bincode `RandomIndices` file, including variable-width
unions. Exactly the first `-n` entries are evaluated. Coordinates must be
distinct and in range, with at least two coordinates per subset. Templates
are comma-separated binary readings in `<identity>/<reading>` folders;
identities and filenames are sorted lexicographically, preserving the prior
local adapter's representative selection. At least four labels are required.

Defaults are `--heavy 20`, `--samples 100000`, `--seed 67` and
`--train-frac 0.5`. The six cheap estimators run on every subset. The existing
three heavier estimators run on the requested number of subsets with lowest
Daugman values. `--heavy 0` selects cheap-only output. `--threads N` creates
a bounded Rayon pool; otherwise Rayon uses its normal environment defaults.
These CLI defaults are explicit adapter choices; they are not a recovered
record of an unpublished working CLI.

`--null` calls the existing `Dataset::null_like(seed)`: uniform latent class
bits with independent capture flips matched to mean genuine disagreement.
It retains the supplied class/reading structure and training split. Real
and null runs select heavy subsets independently. These nulls are not the
length-matched fair streams used by the separate full-reading NIST command.

The console prints JSON metadata and per-estimator summaries. The optional
CSV has one row per subset and columns:

```text
subset,size,daugman,daugman_id_min,daugman_id_p05,betabin_h2,mcv_sum,spectral,chowliu_hinf,guess_indep,guess_tree
```

Unevaluated heavy cells are blank. `--json details.json` additionally saves
exact subsets, individual values, representatives, training/test labels,
seed and heavy-selection policy. Output parents must exist. Requested
CSV/JSON output files are replaced on a rerun, consistent with the parent's
retry convention; the input index file cannot be overwritten. Nonfinite
upstream outputs are explicitly counted and return failure after retaining
the requested audit files. They are not silently replaced or dropped from
a successful assessment.

The unchanged estimates include fitted-model quantities, heuristics and
guess-rank diagnostics; their minimum is not established as a security
bound. In particular, upstream strictly-greater-probability rank handling
can give all equal-probability targets rank one. This integration preserves
that behavior and documents it; it does not implement a fuzzy-radius bound.

## Pair unions

```bash
./target/release/lsh-lock pair-unions \
  -i indices.bin -o unions.bin --pool 5000 --top 1000 --random 1000
```

This directly calls the existing overlap/random sampler and writes the same
bincode format as `random-sampling`. `--seed` defaults to 67. Top pairs are
selected from the first `pool` subsets; random pairs are selected from all
input subsets. There must be enough distinct top pairs and at least two
subsets when random pairs are requested. Overlapping/duplicate unions can
occur under the upstream sampling method. Output coordinates are sorted
and deduplicated by that method. The output file is replaced when reused,
as the parent intentionally reuses union paths across its groups; the input
cannot be overwritten.

The upstream top-overlap implementation materializes candidate pairs, so
large pools can be expensive. This wiring does not change that algorithm.

## Integration checks

The native CLI tests exercise real/null modes, sparse CSV reporting, labels
and split preservation, binary union round-trips, retry output reuse,
invalid input rejection and nonfinite failure reporting. A separate local
regression compares all values against the prior adapter with the identical
`estimators.rs`, both for real data and noise-matched nulls. These checks
establish command compatibility and unchanged calculations, not validity of
the estimators' source models or security of the cryptographic construction.

NIST remains a separate full-reading assessment, documented in
[nist-entropy.md](nist-entropy.md). The companion parent patch supplies its
evaluation group, checks these estimator commands before expensive entropy
jobs, and returns failure for counted foreground job failures.
