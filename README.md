# LSH Then Lock

A Rust implementation of a fuzzy extractor using locality-sensitive hashing (LSH). This tool provides functionality for analyzing and implementing template protection through smart bit selection and correlation analysis.

## Warning

⚠️ **Research Use Only**

This implementation was built specifically to produce data for our research paper. While it's a decent implementation for other researchers to use in their academic work, it is **not** recommended for production use. The code has not been audited for security vulnerabilities and may not implement all necessary safeguards for real-world applications.

If you plan to use fuzzy extractors in a production environment, you should seek or create a properly audited implementation. If you create a production-ready version of this code and can verify its security, please let me know and I'll be happy to link to your repository.

## Installation

Ensure you have Rust and Cargo installed on your system. Then clone this repository:

```bash
git clone https://github.com/whyamey/lsh-lock
cd lsh-lock
cargo build --release
```

## Usage

You will need biometric embeddings to run this. Embeddings of synthetic irises are available in the [main repository](https://github.com/whyamey/feap). The tool provides several commands for different operations:

The parent evaluation's `entropy-suite` and `pair-unions` commands are
documented in [the estimator CLI guide](docs/estimator-cli.md). Separate
full-reading NIST diagnostics and the companion evaluation hook are in
[the NIST integration guide](docs/nist-entropy.md).

### Random Sampling

Generate lockers without considering zeta values:

```bash
cargo run -- random-sampling -o output.bin -c 250000 -s 80 -d 1024
```

Options:
- `-o, --output`: Output file path
- `-c, --count`: Number of samples (default: 250000)
- `-s, --size`: Size of each sample (default: 80)
- `-d, --dimensions`: Feature dimensions (default: 1024)

### Zeta Sampling

Generate lockers with respect to zeta values:

```bash
cargo run -- zeta-sampling -o output.bin -f confidence.txt -c 250000 -s 80 -a 1.0 --method exponent -d 1024
```

Options:
- `-o, --output`: Output file path
- `-f, --confidence`: Path to confidence file
- `-c, --count`: Number of samples (default: 250000)
- `-s, --size`: Size of each sample (default: 80)
- `-a, --alpha`: Alpha parameter (default: 1.0)
- `-d, --dimensions`: Feature dimensions (default: 1024)
- `--method`: Sampling method (gaps/like/ratio/exponent)
- `--bad-indices`: Comma-separated list of indices to exclude

### Correlation Analysis

Generate confidence values by analyzing correlations:

```bash
cargo run -- correlate -i input_dir -o output.txt -n 100 -m both -d 1024
```

Options:
- `-i, --input`: Input directory path
- `-o, --output`: Output file path
- `-n, --num-files`: Number of files to analyze (default: 100)
- `-m, --mode`: Analysis mode (single/pairs/both)
- `-d, --dimensions`: Feature dimensions (default: 1024)

### Entropy Analysis

Analyze the entropy of generated lockers:

```bash
cargo run -- analyze -i indices.bin -t templates_dir -n 1000
```

Options:
- `-i, --input`: Input file containing indices
- `-t, --templates`: Templates directory path
- `-n, --count`: Number of samples to analyze (default: 1000)

### NIST SP 800-90B stream diagnostics

The `nist-entropy` command runs all ten estimators in the **official NIST
non-IID executable**, with one saved reading per identity in each trial. It
retains labels, selection indices, input hashes, every native JSON/log, and
separate outputs for each test. Length-matched independent fair-bit controls
are enabled by default. Python 3 with NumPy and a separately built official
`ea_non_iid` executable are required.

```bash
cargo run --release -- nist-entropy \
  --templates /share/common/embeddings/synlock_embeddings/face_embeddings_noah_nir \
  --nist-bin /path/to/SP800-90B_EntropyAssessment/cpp/ea_non_iid \
  --output nist-results/face --trials 1000 --seed 67 --jobs 2
```

Use `iris_embeddings/test` for iris and `finger_embeddings` for fingerprints.
Do not pass the iris parent directory containing the separate `conf` split.
The width is read from the saved files, so the same command supports 512-bit,
1,024-bit and 1,280-bit readings without changing the wrapper.

See [setup, protocol, output schema and paper methods](docs/nist-entropy.md).
That page also covers the three-dataset runner and the companion patch for
Ben's evaluation script, including its explicit deduplicated face input.
This is a **full-reading stream diagnostic**, separate from the existing
per-subset estimators. Neither the minimum across NIST tests nor a conversion
of its rate into template bits is presented as a key-security bound.

### Cosine LSH Generation

Generate random projection LSH lockers using cosine distance:

```bash
cargo run -- cosine-generate -o output.bin -c 250000 -s 60
```

Options:
- `-o, --output`: Output file path
- `-c, --count`: Number of lockers to generate (default: 250000)
- `-s, --size`: Size of each locker (default: 60)

### Cosine LSH Analysis

Analyze entropy using cosine LSH:

```bash
cargo run -- analyze-cosine -i input.bin -t templates_dir -n 1000
```

Options:
- `-i, --input`: Input file containing cosine lockers
- `-t, --templates`: Templates directory path
- `-n, --count`: Number of samples to analyze (default: 1000)

### TAR Analysis

Calculate True Accept Rate for the lockers:

```bash
cargo run -- tar -i indices.bin -t templates_dir -n 250000
```

Options:
- `-i, --input`: Input file containing indices
- `-t, --templates`: Templates directory path
- `-n, --count`: Number of samples to analyze (default: 250000)

### Cosine TAR Analysis

Calculate True Accept Rate for cosine LSH lockers:

```bash
cargo run -- tar-cosine -i input.bin -t templates_dir -n 250000
```

Options:
- `-i, --input`: Input file containing cosine lockers
- `-t, --templates`: Templates directory path
- `-n, --count`: Number of samples to analyze (default: 250000)

### Multi-Template TAR Analysis

Find TAR with multiple template matching attempts:

```bash
cargo run -- tar-multi -i indices.bin -t templates_dir -n 250000 -r 10 -b 1 --input-selection strategy --output-selection strategy
```

Options:
- `-i, --input`: Input file containing indices
- `-t, --templates`: Templates directory path
- `-n, --count`: Number of samples to analyze (default: 250000)
- `-r, --tries`: Number of template attempts (default: 10)
- `-b, --base`: Number of base templates (default: 1)
- `--input-selection`: Strategy for input template selection
- `--output-selection`: Strategy for output template selection

## Citation
If you use our work, please cite:
```
@inproceedings{shukla2025fuzzy,
  title={Fuzzy Extractors are Practical: Cryptographic Strength Key Derivation from the Iris},
  author={Shukla, Amey and Demarest, Luke and Fuller, Benjamin and Ahmad, Sohaib and Manicke, Caleb and Russell, Alexander and Chen, Sixia},
  booktitle={Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  pages={3605--3619},
  year={2025}
}
```
