# NIST SP 800-90B integration

## Setup and execution

This wrapper invokes NIST's implementation; it does not reimplement the ten
estimators or modify `src/estimators.rs`. The reference experiments used
`ea_non_iid` 1.1.8 from official commit
`87c104d0ed4cbc96103e7b8b38d6f2c7e0a6b289`.

Build the tool separately using its
[official installation instructions](https://github.com/usnistgov/SP800-90B_EntropyAssessment).
For a new checkout:

```bash
git clone https://github.com/usnistgov/SP800-90B_EntropyAssessment
cd SP800-90B_EntropyAssessment
git checkout 87c104d0ed4cbc96103e7b8b38d6f2c7e0a6b289
cd cpp
make non_iid
```

The official README lists the required compiler and system libraries. Its
`selftest` directory provides reference-output checks. On the machine running
this wrapper, install NumPy in the chosen Python environment:

```bash
python3 -m pip install numpy
```

From this repository:

```bash
cargo build --release
./target/release/lsh-lock nist-entropy \
  -t /share/common/embeddings/synlock_embeddings/iris_embeddings/test \
  --nist-bin /path/to/ea_non_iid \
  -o nist-results/iris --trials 1000 --seed 67 --jobs 2
```

Alternatively call `python3 scripts/nist_entropy.py` with the same arguments.
`--python /path/to/python3` selects a Python environment for the Rust command.
`--keep-inputs` retains the individual byte-per-bit streams; by default they
are temporary and can be regenerated from the original readings and saved
selection indices. `--no-controls` disables the default random controls.
`--timeout` is a limit in seconds per native run (default 180).

For Ben's evaluation, run this **once per dataset/feature export**, alongside
the per-subset estimator suite. The command deliberately takes no locker
index file: a 60/70-bit subset would yield a different, much shorter stream
than the full-template protocol evaluated here. No NIST score is folded into
the minimum across Ben's other estimators. If per-subset NIST measurements
are needed, that requires an explicit separate protocol and controls.

### Hook for the full evaluation

`scripts/nist_datasets.py` runs this protocol once for face, iris **test** and
fingerprint, preflighting all three inputs before starting native jobs:

```bash
python3 scripts/nist_datasets.py \
  --data-root /share/common/embeddings/synlock_embeddings \
  --face-templates /path/to/face_embeddings_noah_nir_dedup \
  --nist-bin /path/to/ea_non_iid --output-root nist-results/evaluation-v1
```

The face override is explicit: without it the driver uses the original face
directory, preserving the previous experiments. It never removes or merges
labels. Ben's current evaluation uses a separately prepared face directory
without `0101`; use that same input for a matching evaluation. Its scores need
a fresh run and must not be presented as reproducing the original 196-label
results. Iris `conf` is never included. `complete.json` is written only after
all three assessments succeed. Failed runs retain evidence; choose a fresh
output directory to retry. There is no automatic resume.

The companion `docs/sample-ind-correct-nist.patch` adds a `nist` group to
`sample-ind-correct/eval/run_server.sh`, based on its main commit
`6445e89b103deeed98b07c2e84163a34770299ab`. It also includes that group in
the default evaluation. Review/apply it in that repository:

```bash
git -C ../sample-ind-correct apply --check ../lsh-lock/docs/sample-ind-correct-nist.patch
git -C ../sample-ind-correct apply ../lsh-lock/docs/sample-ind-correct-nist.patch
cd ../sample-ind-correct
DRY=1 bash eval/run_server.sh nist
NIST_EA_NON_IID=/path/to/ea_non_iid bash eval/run_server.sh nist
```

The `../lsh-lock` path above assumes the sibling checkout has that name.
The hook derives its source root from the selected `$LSH` binary path;
`NIST_LSH_ROOT` overrides that root, and `NIST_PYTHON` selects the Python
environment. The hook uses `$FACED` by default, so prepare that deduplicated
directory first (the existing `build` group does this), or explicitly set
`NIST_FACE`. `NIST_TRIALS`, `NIST_SEED`, `NIST_JOBS` (default 1) and
`NIST_TIMEOUT` configure execution. Every NIST invocation runs afresh, including
when its inputs and configuration are unchanged. It creates a unique
`$OUT/nist_<tag>_<suffix>/assessment` result root, with the parent's `.txt` or
`.part`/`.err` job logs alongside the attempt directory. `NIST_TAG` is an
optional human-readable prefix (default `full`); it is not a cache key. It
accepts letters, digits, dots, underscores and hyphens. Existing results,
including failed attempts and legacy `nist_full.txt` logs, are never reused
or deleted. Retrying allocates a fresh attempt automatically. Dry runs do
not allocate an attempt. Other evaluation groups retain their existing
skip behavior. NIST jobs use their own low concurrency limit, independent
of the parent's default 190 threads.

This source branch must be merged/cherry-picked into the `entropy-estimators`
checkout Ben actually builds, or used directly. Publishing this topic branch
alone will not change his existing checkout or evaluation script. The patch
is supplied for review; this addition does not modify the other repository.
The pinned LSH base `63376a9` contained `src/estimators.rs` but did not expose
the `entropy-suite`/`pair-unions` commands that the parent driver calls.
This branch now registers those commands through the existing functions,
without modifying estimator math; see [estimator-cli.md](estimator-cli.md)
for defaults, output formats and retained limitations. The companion patch
checks command availability before starting expensive entropy jobs, and
counted foreground failures return a nonzero evaluation status. NIST remains
its own full-reading group through the Python wrapper. Local compatibility
checks do not constitute execution of the full cryptographic evaluation;
Ben must build/adopt the reviewed branch and parent patch in his working
checkout before restarting it.

## Input and selection

Input layout is `<identity>/<reading>`, with comma-separated binary values.
Identity and reading filenames are sorted lexicographically. Hidden entries
are ignored; malformed readings, empty identities, or inconsistent widths
are errors. Identity labels and original reading values are preserved.

For each seed, NumPy `default_rng(seed)` chooses one reading uniformly from
each identity's saved readings. These full vectors are concatenated in sorted
identity order, with their original coordinate order. There is exactly one
reading per label in each real input; additional trials change the selected
readings, not the participant population. Seeds default to 67 through 1066.
Labels with different reading counts are supported.

NIST receives each bit as a separate byte valued 0 or 1, with
`bits_per_symbol=1`:

```text
ea_non_iid -i -a -vv -o native.json input.bin 1
```

Each trial also gets independent fair-bit control data with the same shape,
using `default_rng(SeedSequence([20261015, selection_seed]))`. This reproduces
the completed face/iris/fingerprint control policy. These controls differ
from Ben's capture-noise-matched latent-identity simulations.

The wrapper verifies all ten native estimates are present, finite and in
[0,1], verifies the reported overall value equals their minimum, and checks
the native input hash. Missing tests or native errors fail the command and
leave logs/checkpoints for inspection; they are never filled with zeros or
silently omitted. An existing output directory is refused.

## Outputs

- `input_manifest.json`: original identity/reading paths and file hashes,
  reading counts, dimensions and the hash of the ordered binary matrix.
- `selected_rows.npy`: the exact source-row choices in each real trial.
- `raw/`: original official JSON and logs for every real/control run.
- `completed_runs.jsonl`: checkpoint records, including input hashes.
- `report.json`: all ten outputs per trial, tool/NumPy versions, binary hash,
  selection policy and per-estimator minimum, 5th percentile, median and
  maximum. The suite minimum is a separately labeled descriptive statistic.
- `all_scores.csv`: one row per test per run, in bits per input bit.

## Tests

```bash
python3 -m unittest discover -s scripts -p 'test_nist*.py'
cargo test --locked
```

Protocol tests cover variable capture counts, preservation of original
labels, binary input validation, reproducibility, missing/non-finite native
estimates, inconsistent overall outputs, and summary semantics. Native-boundary
fixtures cover success, failures, hash mismatches, malformed JSON, timeout
propagation and executable changes. Driver checks cover all-input preflight,
configuration forwarding and failure markers. Real-tool
regression checks should additionally compare all ten estimates and input
hashes to the saved reference experiments; test fixtures alone do not
validate the native numerical implementation.

## Short paper-methods description

We supplement the biometric estimators with the ten non-IID estimators in
the official NIST SP 800-90B implementation: Most Common Value, Collision,
Markov, Compression, t-Tuple, Longest Repeated Substring (LRS), Multi Most
Common in Window, Lag Prediction, MultiMMC, and LZ78Y. For each dataset, we
select one saved reading per supplied identity label, concatenate the full
binary readings in fixed identity and coordinate order, and repeat for
1,000 seeded selections. Thresholds and saved bit values are unchanged.
We retain each estimator's output separately and compare with 1,000
length-matched independent fair-bit controls. Minima, percentiles and medians
describe the sampled selections; the minimum is not over all combinations.

| Estimator | Statistic or predictor examined |
|---|---|
| Most Common Value | Frequency of the most common symbol |
| Collision | Waiting times to repeated symbols |
| Markov | Binary symbol and transition probabilities |
| Compression | Recurrence of blocks in the compression statistic |
| t-Tuple | Frequencies of repeated length-t tuples |
| LRS | Repeated substrings and the associated collision statistic |
| Multi Most Common in Window | Predictions from several recent-window sizes |
| Lag Prediction | Predictions from previous symbols at several lags |
| MultiMMC | Predictions from several Markov context lengths |
| LZ78Y | Dictionary-based predictions from previous contexts |

These are exploratory stream diagnostics. The one-reading-per-identity
inputs in the current datasets contain fewer than the one million samples
specified by SP 800-90B. Increasing the trial count does not increase the
number of independent people or the input length of a run. A source model
justifying biometric vector concatenation as an entropy-source process has
not been established, and coordinate order can affect the outputs without
changing matching distances. We therefore do not interpret these rates,
their minimum, or their product with the template length as certified
template entropy, fuzzy min-entropy, or key-security bounds. The suite is
SP 800-90B, distinct from the SP 800-22 randomness suite cited in the original
paper.

Reference: [SP 800-90B](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90B.pdf),
sections 3.1.1, 3.2.2 and 6.3.1-6.3.10.
