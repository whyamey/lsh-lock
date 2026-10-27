mod confidence;
mod entropy;
mod estimators;
mod sets;
mod suite;
mod tar;
mod types;

use confidence::{CorrelationAnalyzer, SamplingMethod, SmartSampler};
use entropy::{
    AnalysisTool, CosineLockerGenerator, FloatTemplateReader, RandomIndicesGenerator,
    TemplateReader,
};
use std::sync::Arc;
use structopt::StructOpt;
use tar::TARAnalyzer;
use types::AnalysisMode;

#[derive(StructOpt)]
#[structopt(
    name = "LSH then Lock",
    about = "Fuzzy Cryptography using locality-sensitive hashing."
)]
struct Opt {
    #[structopt(subcommand)]
    cmd: Command,
}

#[derive(StructOpt)]
enum Command {
    #[structopt(name = "random-sampling", about = "Generate lockers without zeta.")]
    RandomSampling {
        #[structopt(short, long)]
        output: String,
        #[structopt(short, long, default_value = "250000")]
        count: usize,
        #[structopt(short, long, default_value = "80")]
        size: usize,
        #[structopt(short, long, default_value = "1024")]
        dimensions: usize,
    },
    #[structopt(name = "zeta-sampling", about = "Generate lockers wrt zeta.")]
    ZetaSampling {
        #[structopt(short, long)]
        output: String,
        #[structopt(short, long)]
        confidence: String,
        #[structopt(short, long, default_value = "250000")]
        count: usize,
        #[structopt(short, long, default_value = "80")]
        size: usize,
        #[structopt(short, long, default_value = "1.0")]
        alpha: f64,
        #[structopt(long)]
        method: Option<String>,
        #[structopt(long = "bad-indices", use_delimiter = true)]
        bad_indices: Option<Vec<usize>>,
        #[structopt(short, long, default_value = "1024")]
        dimensions: usize,
    },
    #[structopt(
        name = "correlate",
        about = "Generate confidence by finding correlations for single/pair(s)."
    )]
    Correlate {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        output: String,
        #[structopt(short, long, default_value = "100")]
        num_files: usize,
        #[structopt(short, long, default_value = "single")]
        mode: AnalysisMode,
        #[structopt(short, long, default_value = "1024")]
        dimensions: usize,
    },
    #[structopt(name = "analyze", about = "Analyze the entropy of lockers.")]
    Analyze {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short = "n", long, default_value = "1000")]
        count: usize,
    },
    #[structopt(
        name = "analyze-sets",
        about = "Daugman entropy of many coordinate sets on sampled impostor pairs, with split halves and an independent-bit check; writes JSON."
    )]
    AnalyzeSets {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short, long)]
        output: String,
        /// Impostor pairs to sample (0 = all impostor pairs).
        #[structopt(long, default_value = "0")]
        pairs: usize,
        #[structopt(long, default_value = "1")]
        seed: u64,
        /// Classes to exclude (for example a known duplicate identity).
        #[structopt(long = "skip-classes", use_delimiter = true)]
        skip_classes: Option<Vec<String>>,
        /// File with one class per line used to fit the independent-bit prior (default: all classes).
        #[structopt(long = "prior-classes")]
        prior_classes: Option<String>,
    },
    #[structopt(
        name = "entropy-suite",
        about = "Every entropy estimator (Daugman, per-identity lower tail, beta-binomial, NIST MCV, spectral, Chow-Liu, guesswork) on many coordinate sets; writes JSON."
    )]
    EntropySuite {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short, long)]
        output: String,
        /// Classes to exclude (for example a known duplicate identity).
        #[structopt(long = "skip-classes", use_delimiter = true)]
        skip_classes: Option<Vec<String>>,
        /// File with one class per line on which model-based estimators are fitted (default: a seeded split).
        #[structopt(long = "train-classes")]
        train_classes: Option<String>,
        #[structopt(long = "train-frac", default_value = "0.5")]
        train_frac: f64,
        /// Run the model-based estimators on this many sets with the lowest Daugman value.
        #[structopt(long, default_value = "0")]
        heavy: usize,
        /// Model samples per Monte Carlo rank estimate.
        #[structopt(long, default_value = "2000")]
        samples: usize,
        #[structopt(long, default_value = "1")]
        seed: u64,
        /// Also run every estimator on a full-entropy dataset of the same shape.
        #[structopt(long)]
        null: bool,
    },
    #[structopt(
        name = "cosine-generate",
        about = "Generate random projection LSH lockers"
    )]
    CosineGenerate {
        #[structopt(short, long)]
        output: String,
        #[structopt(short, long, default_value = "250000")]
        count: usize,
        #[structopt(short, long, default_value = "60")]
        size: usize,
    },
    #[structopt(name = "analyze-cosine", about = "Analyze entropy using cosine LSH")]
    AnalyzeCosine {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short = "n", long, default_value = "1000")]
        count: usize,
    },
    #[structopt(name = "tar", about = "Find TAR/TPR of lockers.")]
    TAR {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short = "n", long, default_value = "250000")]
        count: usize,
    },
    #[structopt(name = "tar-cosine", about = "Find TAR of cosine LSH lockers.")]
    TARCosine {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short = "n", long, default_value = "250000")]
        count: usize,
    },
    #[structopt(
        name = "tar-multi",
        about = "Find TAR with multiple template matching attempts"
    )]
    TARMulti {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short = "n", long, default_value = "250000")]
        count: usize,
        #[structopt(short = "t", long, default_value = "10")]
        tries: usize,
        #[structopt(short = "b", long, default_value = "1")]
        base: usize,
        #[structopt(long = "input-selection")]
        input_selection: Option<String>,
        #[structopt(long = "output-selection")]
        output_selection: Option<String>,
    },
}

fn parse_sampling_method(method: Option<String>) -> SamplingMethod {
    match method.as_deref() {
        Some("like") => SamplingMethod::Like,
        Some("ratio") => SamplingMethod::Ratio,
        Some("exponent") => SamplingMethod::Exponent,
        Some(unknown) => {
            eprintln!(
                "Unknown sampling method '{}', defaulting to 'ratio'",
                unknown
            );
            SamplingMethod::default()
        }
        None => SamplingMethod::default(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    match opt.cmd {
        Command::RandomSampling {
            output,
            count,
            size,
            dimensions,
        } => {
            RandomIndicesGenerator::generate_and_store(&output, count, size, dimensions)?;
        }
        Command::ZetaSampling {
            output,
            confidence,
            count,
            size,
            alpha,
            method,
            bad_indices,
            dimensions,
        } => {
            let sampling_method = parse_sampling_method(method);
            SmartSampler::generate_and_store(
                &output,
                &confidence,
                count,
                size,
                alpha,
                sampling_method,
                bad_indices,
                dimensions,
            )?;
        }
        Command::Correlate {
            input,
            output,
            num_files,
            mode,
            dimensions,
        } => {
            let analyzer = CorrelationAnalyzer::new(&input, num_files, dimensions);
            analyzer.generate_correlation_report(&output, mode)?;
        }
        Command::Analyze {
            input,
            templates,
            count,
        } => {
            let random_indices = RandomIndicesGenerator::load(&input)?;
            let selected_indices = &random_indices.0[0..count];

            let templates = TemplateReader::read_templates(&templates)?;

            println!("Calculating entropies for each subset:");
            let (avg_diff_class_mean, avg_entropy, min_entropy, entropy_store) =
                AnalysisTool::calculate_class_based_fractional_hamming_mean_and_entropy(
                    &templates,
                    selected_indices,
                );

            let unwrap_entropy = Arc::try_unwrap(entropy_store)
                .unwrap()
                .into_inner()
                .unwrap();

            println!("\nSummary:");
            for (index, entropy) in unwrap_entropy.iter().enumerate() {
                println!("Entropy at subset {}: {}", index, entropy);
            }

            println!("Average Different Class Mean: {}", avg_diff_class_mean);
            println!(
                "Average Entropy: {}, Minimum Entropy: {}",
                avg_entropy, min_entropy
            );
        }
        Command::AnalyzeSets {
            input,
            templates,
            output,
            pairs,
            seed,
            skip_classes,
            prior_classes,
        } => {
            sets::run(
                &input,
                &templates,
                &output,
                pairs,
                seed,
                &skip_classes.unwrap_or_default(),
                prior_classes.as_deref(),
            )?;
        }
        Command::EntropySuite {
            input,
            templates,
            output,
            skip_classes,
            train_classes,
            train_frac,
            heavy,
            samples,
            seed,
            null,
        } => {
            suite::run(
                &input,
                &templates,
                &output,
                &skip_classes.unwrap_or_default(),
                train_classes.as_deref(),
                train_frac,
                heavy,
                samples,
                seed,
                null,
            )?;
        }
        Command::CosineGenerate {
            output,
            count,
            size,
        } => {
            println!("Generating {} cosine LSH lockers of size {}", count, size);
            CosineLockerGenerator::generate_and_store(&output, count, size)?;
        }
        Command::AnalyzeCosine {
            input,
            templates,
            count,
        } => {
            println!("Loading cosine lockers from {}", input);
            let lockers = CosineLockerGenerator::load(&input)?;
            let lockers = &lockers[0..count];

            println!("Loading float templates from {}", templates);
            let templates = FloatTemplateReader::read_templates(&templates)?;

            println!("Calculating cosine LSH entropies:");
            let (avg_diff_class_mean, avg_entropy, min_entropy, entropy_store) =
                AnalysisTool::calculate_cosine_entropy(&templates, &lockers);

            let unwrap_entropy = Arc::try_unwrap(entropy_store)
                .unwrap()
                .into_inner()
                .unwrap();

            println!("\nSummary:");
            println!("Average Different Class Mean: {}", avg_diff_class_mean);
            println!("Average Entropy: {}", avg_entropy);
            println!("Minimum Entropy: {}", min_entropy);

            println!("\nDetailed entropy by locker (first 10):");
            for (index, entropy) in unwrap_entropy.iter().enumerate().take(10) {
                println!("Locker {}: {}", index, entropy);
            }
        }
        Command::TAR {
            input,
            templates,
            count,
        } => {
            let random_indices = RandomIndicesGenerator::load(&input)?;
            let selected_indices = &random_indices.0[0..count];

            println!("Calculating True Accept Rate (TAR)...");
            let (tar, total_successes, total_comparisons) =
                TARAnalyzer::analyze_tar(&templates, selected_indices)?;

            println!("\nResults:");
            println!("True Accept Rate (TAR): {:.4}", tar);
            println!("Total Successes: {}", total_successes);
            println!("Total Comparisons: {}", total_comparisons);
        }
        Command::TARCosine {
            input,
            templates,
            count,
        } => {
            println!("Loading cosine lockers from {}", input);
            let lockers = CosineLockerGenerator::load(&input)?;
            let lockers = &lockers[0..count];

            println!("Calculating True Accept Rate (TAR)...");
            let (tar, total_successes, total_comparisons) =
                TARAnalyzer::analyze_cosine_tar(&templates, lockers)?;

            println!("\nResults:");
            println!("True Accept Rate (TAR): {:.4}", tar);
            println!("Total Successes: {}", total_successes);
            println!("Total Comparisons: {}", total_comparisons);
        }
        Command::TARMulti {
            input,
            templates,
            count,
            tries,
            base,
            input_selection,
            output_selection,
        } => {
            let random_indices = RandomIndicesGenerator::load(&input)?;
            let selected_indices = &random_indices.0[0..count];

            println!("Calculating Multi-Template True Accept Rate (TAR)...");
            println!(
                "Using {} base templates and {} comparison templates",
                base,
                tries - base
            );

            let (tar, successful_classes, total_classes, _) = TARAnalyzer::analyze_tar_multi(
                &templates,
                selected_indices,
                tries,
                base,
                input_selection.as_deref(),
                output_selection.as_deref(),
            )?;

            println!("\nResults:");
            println!("True Accept Rate (TAR): {:.4}", tar);
            println!("Successful Classes: {}", successful_classes);
            println!("Total Classes: {}", total_classes);
        }
    }

    Ok(())
}
