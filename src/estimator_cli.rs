//! Command-line adapters only. The estimator and pair-sampling math stays in estimators.rs.
use crate::entropy::{RandomIndices, RandomIndicesGenerator, Template};
use crate::estimators::{self, Dataset, SuiteResult, CHEAP, HEAVY};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use structopt::StructOpt;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, StructOpt)]
pub struct SuiteArgs {
    #[structopt(short = "i", long)]
    pub input: PathBuf,
    #[structopt(short = "t", long)]
    pub templates: PathBuf,
    #[structopt(short = "n", long, default_value = "10000")]
    pub count: usize,
    #[structopt(long, default_value = "20")]
    pub heavy: usize,
    #[structopt(long, default_value = "100000")]
    pub samples: usize,
    #[structopt(long, default_value = "67")]
    pub seed: u64,
    #[structopt(long, default_value = "0.5")]
    pub train_frac: f64,
    #[structopt(long)]
    pub threads: Option<usize>,
    #[structopt(long = "null")]
    pub synthetic_null: bool,
    #[structopt(long)]
    pub csv: Option<PathBuf>,
    #[structopt(
        long,
        help = "Optional detailed JSON with exact subsets, values and selection metadata"
    )]
    pub json: Option<PathBuf>,
}

#[derive(Debug, StructOpt)]
pub struct PairArgs {
    #[structopt(short = "i", long)]
    pub input: PathBuf,
    #[structopt(short = "o", long)]
    pub output: PathBuf,
    #[structopt(long, default_value = "5000")]
    pub pool: usize,
    #[structopt(long, default_value = "1000")]
    pub top: usize,
    #[structopt(long, default_value = "1000")]
    pub random: usize,
    #[structopt(long, default_value = "67")]
    pub seed: u64,
}

fn load_templates(root: &Path) -> Result<(Vec<Template>, Vec<String>)> {
    let mut folders = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    folders.sort_by_key(|e| e.file_name());
    let (mut templates, mut paths) = (Vec::new(), Vec::new());
    for folder in folders {
        if folder.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if !folder.file_type()?.is_dir() {
            return Err("expected identity directories at the input root".into());
        }
        let class = folder
            .file_name()
            .into_string()
            .map_err(|_| "non-UTF8 identity")?;
        let mut files = fs::read_dir(folder.path())?.collect::<std::io::Result<Vec<_>>>()?;
        files.sort_by_key(|e| e.file_name());
        let before = templates.len();
        for file in files {
            if file.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if !file.file_type()?.is_file() {
                return Err("expected reading files inside each identity".into());
            }
            let text = fs::read_to_string(file.path())?;
            let data = text
                .trim()
                .trim_end_matches(',')
                .split(',')
                .map(str::parse::<u8>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if data.is_empty() || data.iter().any(|&v| v > 1) {
                return Err("nonbinary or empty reading".into());
            }
            paths.push(format!("{class}/{}", file.file_name().to_string_lossy()));
            templates.push(Template {
                class: class.clone(),
                data,
            });
        }
        if before == templates.len() {
            return Err(format!("identity has no readings: {class}").into());
        }
    }
    let width = templates.first().ok_or("empty dataset")?.data.len();
    if templates.iter().any(|t| t.data.len() != width) {
        return Err("inconsistent reading widths".into());
    }
    if templates
        .iter()
        .map(|t| &t.class)
        .collect::<BTreeSet<_>>()
        .len()
        < 4
    {
        return Err("entropy-suite requires at least four identity labels".into());
    }
    Ok((templates, paths))
}

fn validate_subsets(subsets: &[Vec<usize>], width: Option<usize>) -> Result<()> {
    if subsets.is_empty() {
        return Err("empty subset file".into());
    }
    for sub in subsets {
        if sub.len() < 2
            || sub.iter().collect::<BTreeSet<_>>().len() != sub.len()
            || width.is_some_and(|n| sub.iter().any(|&i| i >= n))
        {
            return Err("each subset needs at least two distinct, in-range coordinates".into());
        }
    }
    Ok(())
}

fn summarize(values: impl Iterator<Item = f64>) -> Value {
    let values: Vec<f64> = values.collect();
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_by(f64::total_cmp);
    let q = |fraction: f64| {
        finite
            .get(((finite.len().saturating_sub(1)) as f64 * fraction).round() as usize)
            .copied()
    };
    json!({"evaluated": values.len(), "finite": finite.len(), "nonfinite": values.len()-finite.len(),
        "minimum": finite.first(), "p05": q(0.05), "median": q(0.5), "maximum": finite.last()})
}

fn write_csv(path: &Path, subsets: &[Vec<usize>], result: &SuiteResult) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "subset,size,{},{}",
        CHEAP.join(","),
        HEAVY.join(",")
    )?;
    let heavy: BTreeMap<_, _> = result
        .heavy
        .iter()
        .map(|(index, values)| (*index, values))
        .collect();
    for (i, (sub, values)) in subsets.iter().zip(&result.cheap).enumerate() {
        write!(writer, "{i},{}", sub.len())?;
        for value in values {
            write!(writer, ",{value}")?;
        }
        for j in 0..HEAVY.len() {
            if let Some(values) = heavy.get(&i) {
                write!(writer, ",{}", values[j])?;
            } else {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn suite(args: SuiteArgs) -> Result<()> {
    if args.count == 0
        || args.samples == 0
        || args.threads == Some(0)
        || !args.train_frac.is_finite()
        || !(0.0..1.0).contains(&args.train_frac)
        || args.train_frac == 0.0
    {
        return Err("count/samples/threads must be positive and train-frac must be strictly between 0 and 1".into());
    }
    for output in [&args.csv, &args.json].into_iter().flatten() {
        if output.exists() && fs::canonicalize(output)? == fs::canonicalize(&args.input)? {
            return Err("output must not overwrite the input subset file".into());
        }
    }
    if args.csv.is_some() && args.csv == args.json {
        return Err("CSV and JSON outputs must differ".into());
    }
    let all = RandomIndicesGenerator::load(args.input.to_str().ok_or("non-UTF8 index path")?)?;
    if args.count > all.0.len() {
        return Err(format!(
            "requested {} subsets but input contains {}",
            args.count,
            all.0.len()
        )
        .into());
    }
    let subsets = &all.0[..args.count];
    let (templates, paths) = load_templates(&args.templates)?;
    let width = templates[0].data.len();
    validate_subsets(subsets, Some(width))?;
    let real = Dataset::new(templates, args.train_frac, args.seed);
    let n_train = real.train_class.iter().filter(|&&v| v).count();
    if n_train == 0 || n_train == real.class_names.len() {
        return Err("training and test identity splits must both be nonempty".into());
    }
    let synthetic;
    let data = if args.synthetic_null {
        synthetic = real.null_like(args.seed);
        &synthetic
    } else {
        &real
    };
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = args.threads {
        builder = builder.num_threads(threads);
    }
    let result = builder
        .build()?
        .install(|| estimators::run_suite(data, subsets, args.heavy, args.samples, args.seed));
    let mut summary = BTreeMap::new();
    for (j, name) in CHEAP.iter().enumerate() {
        summary.insert(*name, summarize(result.cheap.iter().map(|r| r[j])));
    }
    for (j, name) in HEAVY.iter().enumerate() {
        summary.insert(*name, summarize(result.heavy.iter().map(|(_, r)| r[j])));
    }
    let finite = result
        .cheap
        .iter()
        .flatten()
        .chain(result.heavy.iter().flat_map(|(_, r)| r.iter()))
        .all(|v| v.is_finite());
    let train_ids: Vec<_> = real
        .class_names
        .iter()
        .enumerate()
        .filter(|(i, _)| real.train_class[*i])
        .map(|(_, n)| n)
        .collect();
    let test_ids: Vec<_> = real
        .class_names
        .iter()
        .enumerate()
        .filter(|(i, _)| !real.train_class[*i])
        .map(|(_, n)| n)
        .collect();
    let representatives: Vec<_> = real.reps.iter().map(|&i| &paths[i]).collect();
    let record = json!({"command":"entropy-suite", "synthetic_null":args.synthetic_null, "seed":args.seed,
        "input":args.input, "templates":args.templates, "subsets_evaluated":subsets.len(),
        "readings":real.data.len(), "identity_labels":real.class_names.len(), "bits_per_reading":width,
        "train_frac":args.train_frac, "train_identity_labels":train_ids, "test_identity_labels":test_ids,
        "representative_policy":"lexicographically first saved reading per identity", "representatives":representatives,
        "heavy_requested":args.heavy, "heavy_evaluated":result.heavy.len(), "samples":args.samples,
        "heavy_selection":"lowest Daugman within each run; real and null can select different subsets",
        "all_values_finite":finite, "summary":summary,
        "limitations":["Unchanged upstream heuristics, fitted-model and guess-rank quantities; do not combine their minimum into a security bound.",
            "No fuzzy-radius bound is implemented. Null values are calibration diagnostics, not multiplicative corrections.",
            "Upstream strict-greater guess-rank handling can give rank one to all equal-probability targets; retained without alteration.",
            "A sampled subset minimum is only the observed minimum; NIST full-reading diagnostics are separate."]});
    if let Some(path) = &args.csv {
        write_csv(path, subsets, &result)?;
    }
    if let Some(path) = &args.json {
        let mut full = record.clone();
        full["subsets"] = json!(subsets);
        full["cheap_names"] = json!(CHEAP);
        full["heavy_names"] = json!(HEAVY);
        full["cheap"] = json!(result.cheap);
        full["heavy"] = json!(result.heavy);
        let mut file = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut file, &full)?;
        writeln!(file)?;
        file.flush()?;
    }
    println!("{}", serde_json::to_string_pretty(&record)?);
    if !finite {
        return Err("non-finite estimator output; see summary/raw CSV/JSON; result is not a successful assessment".into());
    }
    Ok(())
}

pub fn pairs(args: PairArgs) -> Result<()> {
    if args.input == args.output
        || (args.output.exists()
            && fs::canonicalize(&args.input)? == fs::canonicalize(&args.output)?)
    {
        return Err("pair-unions output must differ from its input".into());
    }
    let all = RandomIndicesGenerator::load(args.input.to_str().ok_or("non-UTF8 index path")?)?;
    validate_subsets(&all.0, None)?;
    let pool = args.pool.min(all.0.len());
    let available = pool
        .checked_mul(pool.saturating_sub(1))
        .ok_or("pair count overflow")?
        / 2;
    if args.top > available
        || (args.random > 0 && all.0.len() < 2)
        || (args.top == 0 && args.random == 0)
    {
        return Err(
            "invalid pair counts: need available top pairs and two subsets for random pairs".into(),
        );
    }
    let unions = estimators::pair_unions(&all.0, args.pool, args.top, args.random, args.seed);
    assert_eq!(unions.len(), args.top + args.random);
    // Parent runner intentionally reuses the same union path across groups; overwrite its output,
    // as random-sampling does. Inputs are never overwritten.
    let mut file = BufWriter::new(File::create(&args.output)?);
    bincode::serialize_into(&mut file, &RandomIndices(unions))?;
    file.flush()?;
    println!(
        "{}",
        json!({"command":"pair-unions", "input":args.input, "output":args.output,
        "pool":pool, "top":args.top, "random":args.random, "seed":args.seed,
        "unions":args.top+args.random, "math":"unchanged estimators::pair_unions"})
    );
    Ok(())
}
