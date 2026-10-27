//! `entropy-suite`: every estimator of `estimators.rs` on many coordinate sets, written as JSON.
//!
//! Input and template formats are those of `analyze-sets`. The cheap estimators run on every set; the heavy,
//! model-based ones on the `heavy` sets with the lowest Daugman value. Model-based estimators are fitted on the
//! training classes and evaluated on the others: by default a seeded split of `train_frac` of the classes, or the
//! classes listed in `--train-classes`. With `--null`, every estimator is also run on a dataset of the same shape
//! with known full entropy (`Dataset::null_like`), which measures each estimator's sample-size floor.
use crate::entropy::{RandomIndicesGenerator, TemplateReader};
use crate::estimators::{run_suite, Dataset, CHEAP, HEAVY};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::BufWriter;

#[derive(Serialize)]
struct Values {
    /// estimator name -> one value per input set
    cheap: BTreeMap<String, Vec<f64>>,
    /// indices (into the input sets) of the sets the heavy estimators ran on
    heavy_index: Vec<usize>,
    /// estimator name -> one value per entry of `heavy_index`
    heavy: BTreeMap<String, Vec<f64>>,
}

#[derive(Serialize)]
struct Report {
    templates: usize,
    classes: usize,
    train_classes: usize,
    seed: u64,
    samples: usize,
    size: Vec<usize>,
    values: Values,
    null: Option<Values>,
}

fn collect(ds: &Dataset, sets: &[Vec<usize>], heavy: usize, samples: usize, seed: u64) -> Values {
    let r = run_suite(ds, sets, heavy, samples, seed);
    let cheap = CHEAP
        .iter()
        .enumerate()
        .map(|(k, name)| (name.to_string(), r.cheap.iter().map(|v| v[k]).collect()))
        .collect();
    let heavy = HEAVY
        .iter()
        .enumerate()
        .map(|(k, name)| (name.to_string(), r.heavy.iter().map(|(_, v)| v[k]).collect()))
        .collect();
    Values { cheap, heavy_index: r.heavy.iter().map(|&(i, _)| i).collect(), heavy }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    input: &str,
    templates_dir: &str,
    output: &str,
    skip_classes: &[String],
    train_classes_file: Option<&str>,
    train_frac: f64,
    heavy: usize,
    samples: usize,
    seed: u64,
    null: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let sets = RandomIndicesGenerator::load(input)?.0;
    let mut templates = TemplateReader::read_templates(templates_dir)?;
    templates.retain(|t| !skip_classes.contains(&t.class));
    if templates.is_empty() {
        return Err("no templates".into());
    }
    let dims = templates[0].data.len();
    if templates.iter().any(|t| t.data.len() != dims) || sets.iter().flatten().any(|&j| j >= dims) {
        return Err("set coordinate out of range or templates of different lengths".into());
    }
    let mut ds = Dataset::new(templates, train_frac, seed);
    if let Some(path) = train_classes_file {
        let train: HashSet<String> = std::fs::read_to_string(path)?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        ds.train_class = ds.class_names.iter().map(|c| train.contains(c)).collect();
    }
    if !ds.train_class.iter().any(|&t| t) || ds.train_class.iter().all(|&t| t) {
        return Err("model-based estimators need both training and held-out classes".into());
    }
    let values = collect(&ds, &sets, heavy, samples, seed);
    let null = if null { Some(collect(&ds.null_like(seed), &sets, heavy, samples, seed)) } else { None };
    let report = Report {
        templates: ds.data.len(),
        classes: ds.class_names.len(),
        train_classes: ds.train_class.iter().filter(|&&t| t).count(),
        seed,
        samples,
        size: sets.iter().map(|s| s.len()).collect(),
        values,
        null,
    };
    serde_json::to_writer(BufWriter::new(File::create(output)?), &report)?;
    Ok(())
}
