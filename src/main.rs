mod confidence;
mod entropy;
mod tar;
mod types;

use confidence::{CorrelationAnalyzer, SamplingMethod, SmartSampler};
use entropy::{AnalysisTool, RandomIndicesGenerator, TemplateReader};
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
    #[structopt(name = "tar", about = "Find TAR/TPR of lockers.")]
    TAR {
        #[structopt(short, long)]
        input: String,
        #[structopt(short, long)]
        templates: String,
        #[structopt(short = "n", long, default_value = "250000")]
        count: usize,
    },
}

fn parse_sampling_method(method: Option<String>) -> SamplingMethod {
    match method.as_deref() {
        Some("gaps") => SamplingMethod::Gaps,
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
        } => {
            RandomIndicesGenerator::generate_and_store(&output, count, size)?;
        }
        Command::ZetaSampling {
            output,
            confidence,
            count,
            size,
            alpha,
            method,
            bad_indices,
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
            )?;
        }
        Command::Correlate {
            input,
            output,
            num_files,
            mode,
        } => {
            let analyzer = CorrelationAnalyzer::new(&input, num_files);
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
            let (avg_diff_class_mean, avg_entropy, entropy_store) =
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
            println!("Average Entropy: {}", avg_entropy);
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
    }

    Ok(())
}
