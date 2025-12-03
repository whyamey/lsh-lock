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

The tool provides several commands for different operations:

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
cargo run -- zeta-sampling -o output.bin -c confidence.txt -c 250000 -s 80 -a 1.0 --method exponent -d 1024
```

Options:
- `-o, --output`: Output file path
- `-c, --confidence`: Path to confidence file
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
cargo run -- tar-multi -i indices.bin -t templates_dir -n 250000 -t 10 -b 1 --input-selection strategy --output-selection strategy
```

Options:
- `-i, --input`: Input file containing indices
- `-t, --templates`: Templates directory path
- `-n, --count`: Number of samples to analyze (default: 250000)
- `-t, --tries`: Number of template attempts (default: 10)
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
