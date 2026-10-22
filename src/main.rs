mod confidence;
mod entropy;
mod estimator_cli;
mod estimators;
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
    #[structopt(
        name = "entropy-suite",
        about = "Run the existing per-subset estimators, optionally on a noise-matched null."
    )]
    EntropySuite(estimator_cli::SuiteArgs),
    #[structopt(
        name = "pair-unions",
        about = "Generate unions using the existing overlap/random pair sampler."
    )]
    PairUnions(estimator_cli::PairArgs),
    #[structopt(
        name = "nist-entropy",
        about = "Run official NIST stream diagnostics with one reading per identity."
    )]
    NistEntropy {
        #[structopt(short = "t", long)]
        templates: String,
        #[structopt(short = "o", long)]
        output: String,
        #[structopt(long, default_value = "ea_non_iid")]
        nist_bin: String,
        #[structopt(long, default_value = "1000")]
        trials: usize,
        #[structopt(long, default_value = "67")]
        seed: u64,
        #[structopt(long, default_value = "1")]
        jobs: usize,
        #[structopt(long, default_value = "180")]
        timeout: f64,
        #[structopt(long)]
        no_controls: bool,
        #[structopt(long)]
        keep_inputs: bool,
        #[structopt(long, default_value = "python3")]
        python: String,
    },
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
        #[structopt(short = "f", long)]
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
        #[structopt(short = "r", long, default_value = "10")]
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
        Command::EntropySuite(args) => estimator_cli::suite(args)?,
        Command::PairUnions(args) => estimator_cli::pairs(args)?,
        Command::NistEntropy {
            templates,
            output,
            nist_bin,
            trials,
            seed,
            jobs,
            timeout,
            no_controls,
            keep_inputs,
            python,
        } => {
            // Embed the same reviewed script so the binary also works outside
            // its original checkout. Python/NumPy and official NIST stay external.
            let mut command = std::process::Command::new(python);
            command.args(["-c", include_str!("../scripts/nist_entropy.py")]);
            command.args([
                "--templates",
                &templates,
                "--output",
                &output,
                "--nist-bin",
                &nist_bin,
            ]);
            command.args(["--trials", &trials.to_string(), "--seed", &seed.to_string()]);
            command.args([
                "--jobs",
                &jobs.to_string(),
                "--timeout",
                &timeout.to_string(),
            ]);
            if no_controls {
                command.arg("--no-controls");
            }
            if keep_inputs {
                command.arg("--keep-inputs");
            }
            let status = command.status()?;
            if !status.success() {
                return Err(format!(
                    "NIST wrapper failed ({status}); inspect its output and raw logs"
                )
                .into());
            }
        }
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

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn parent_estimator_command_forms_are_registered() {
        let opt = Opt::from_iter_safe([
            "lsh-lock",
            "entropy-suite",
            "-i",
            "idx.bin",
            "-t",
            "data",
            "-n",
            "10000",
            "--csv",
            "suite.csv",
        ])
        .unwrap();
        match opt.cmd {
            Command::EntropySuite(args) => {
                assert_eq!(args.count, 10000);
                assert_eq!((args.heavy, args.samples, args.seed), (20, 100000, 67));
                assert!(!args.synthetic_null);
            }
            _ => panic!("wrong command"),
        }
        let opt = Opt::from_iter_safe([
            "lsh-lock",
            "entropy-suite",
            "-i",
            "idx.bin",
            "-t",
            "data",
            "-n",
            "10000",
            "--null",
        ])
        .unwrap();
        assert!(matches!(opt.cmd, Command::EntropySuite(args) if args.synthetic_null));
        let opt = Opt::from_iter_safe([
            "lsh-lock",
            "pair-unions",
            "-i",
            "idx.bin",
            "-o",
            "unions.bin",
            "--pool",
            "5000",
            "--top",
            "1000",
            "--random",
            "1000",
        ])
        .unwrap();
        assert!(
            matches!(opt.cmd, Command::PairUnions(args) if args.pool == 5000 && args.top == 1000 && args.random == 1000)
        );
    }

    #[test]
    fn nist_command_parses_without_colliding_with_existing_short_flags() {
        let opt = Opt::from_iter_safe([
            "lsh-lock",
            "nist-entropy",
            "-t",
            "templates",
            "-o",
            "results",
        ])
        .expect("command parser should accept NIST with defaults");
        match opt.cmd {
            Command::NistEntropy {
                trials,
                seed,
                jobs,
                no_controls,
                ..
            } => {
                assert_eq!((trials, seed, jobs), (1000, 67, 1));
                assert!(!no_controls);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn zeta_confidence_and_count_have_distinct_short_flags() {
        let opt = Opt::from_iter_safe([
            "lsh-lock",
            "zeta-sampling",
            "-o",
            "out",
            "-f",
            "confidence",
            "-c",
            "42",
        ])
        .expect("confidence and count must not compete for -c");
        match opt.cmd {
            Command::ZetaSampling {
                confidence, count, ..
            } => {
                assert_eq!(confidence, "confidence");
                assert_eq!(count, 42);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn multi_tar_templates_and_tries_have_distinct_short_flags() {
        let opt = Opt::from_iter_safe([
            "lsh-lock",
            "tar-multi",
            "-i",
            "idx",
            "-t",
            "templates",
            "-r",
            "8",
        ])
        .expect("templates and tries must not compete for -t");
        match opt.cmd {
            Command::TARMulti {
                templates, tries, ..
            } => {
                assert_eq!(templates, "templates");
                assert_eq!(tries, 8);
            }
            _ => panic!("wrong command"),
        }
    }
}
