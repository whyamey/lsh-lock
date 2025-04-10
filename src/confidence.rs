use crate::types::AnalysisMode;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::error::Error as StdError;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum SamplingMethod {
    Like,     // pair[0] ** alpha
    Ratio,    // pair[0]/max(pair[1], 1-pair[1]) ** alpha
    Exponent, // pair[0] ** (alpha/min_entropy(1 - pair[1]))
}

impl Default for SamplingMethod {
    fn default() -> Self {
        SamplingMethod::Exponent
    }
}

#[derive(Debug)]
struct ParseError {
    line_number: usize,
    message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Line {}: {}", self.line_number, self.message)
    }
}

impl StdError for ParseError {}

#[derive(Debug)]
pub struct ConfidenceEntry {
    pub predictability: f64,
    pub unlike_probabilities: Option<[f64; 4]>,
    pub unlike_probability: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SmartIndices(pub Vec<Vec<usize>>);

fn min_entropy(val: f64) -> f64 {
    if val <= 0.0 || val >= 1.0 {
        return 0.0;
    }
    -val.log2().min(-(1.0 - val).log2())
}

fn min_entropy_pairs(probabilities: &[f64; 4]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0 && p < 1.0)
        .map(|&p| -p.log2())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0)
}

pub struct ConfidenceReader;

impl ConfidenceReader {
    fn parse_float(s: &str, line_number: usize, field: &str) -> Result<f64, Box<dyn StdError>> {
        s.parse::<f64>().map_err(|e| -> Box<dyn StdError> {
            Box::new(ParseError {
                line_number,
                message: format!("Failed to parse {} '{}': {}", field, s, e),
            })
        })
    }

    pub fn read_confidence_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<ConfidenceEntry>, Box<dyn StdError>> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut line_number = 0;

        for line in reader.lines() {
            line_number += 1;
            let line = line?;

            if !line.starts_with("Hamming distance") {
                continue;
            }

            if line.contains('(') {
                let after_paren = line.split(')').nth(1).ok_or_else(|| {
                    Box::new(ParseError {
                        line_number,
                        message: "Malformed line: no closing parenthesis".to_string(),
                    })
                })?;

                let values: Vec<f64> = after_paren
                    .split_whitespace()
                    .map(|s| Self::parse_float(s, line_number, "probability"))
                    .collect::<Result<Vec<f64>, _>>()?;

                if values.len() < 8 {
                    println!(
                        "Warning: Skipping line {} - not enough values after parentheses",
                        line_number
                    );
                    continue;
                }

                let like_probabilities = [values[0], values[1], values[2], values[3]];
                let unlike_probabilities = [values[4], values[5], values[6], values[7]];

                entries.push(ConfidenceEntry {
                    predictability: 1.0 - like_probabilities[0],
                    unlike_probabilities: Some(unlike_probabilities),
                    unlike_probability: None,
                });
            } else if line.contains("Like/Unlike/Difference") {
                let parts: Vec<&str> = line[42..].trim().split(' ').collect();
                if parts.len() >= 4 {
                    let like_prob = Self::parse_float(parts[1], line_number, "like probability")?;
                    let unlike_prob =
                        Self::parse_float(parts[2], line_number, "unlike probability")?;

                    entries.push(ConfidenceEntry {
                        predictability: 1.0 - like_prob,
                        unlike_probabilities: None,
                        unlike_probability: Some(unlike_prob),
                    });
                }
            }
        }

        if entries.is_empty() {
            return Err(Box::new(ParseError {
                line_number: 0,
                message: "No valid entries found in file".to_string(),
            }));
        }

        println!("Successfully parsed {} entries", entries.len());
        Ok(entries)
    }
}

#[derive(Debug)]
struct GenerationError(String);

impl std::error::Error for GenerationError {}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct SmartSampler {
    confidence_entries: Vec<ConfidenceEntry>,
    bad_indices: HashSet<usize>,
    dimensions: usize,
}

impl SmartSampler {
    pub fn new(
        confidence_entries: Vec<ConfidenceEntry>,
        bad_indices: Option<Vec<usize>>,
        dimensions: usize,
    ) -> Self {
        let bad_indices = bad_indices
            .map(|indices| indices.into_iter().collect())
            .unwrap_or_default();

        Self {
            confidence_entries,
            bad_indices,
            dimensions,
        }
    }

    fn calculate_weights(&self, alpha: f64, method: SamplingMethod) -> Vec<f64> {
        self.confidence_entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                if self.bad_indices.contains(&idx) {
                    println!("Excluding index {} from sampling", idx);
                    return 0.0;
                }

                match method {
                    SamplingMethod::Ratio => {
                        let entropy_factor = if let Some(unlike_probs) = &entry.unlike_probabilities
                        {
                            let max_unlike_sum = unlike_probs
                                .iter()
                                .enumerate()
                                .map(|(i, _)| {
                                    let other_sum: f64 = unlike_probs
                                        .iter()
                                        .enumerate()
                                        .filter(|&(j, _)| j != i)
                                        .map(|(_, &p)| p)
                                        .sum();
                                    other_sum
                                })
                                .fold(0.0, f64::max);
                            max_unlike_sum
                        } else if let Some(unlike_prob) = entry.unlike_probability {
                            f64::max(unlike_prob, 1.0 - unlike_prob)
                        } else {
                            1.0
                        };

                        (entry.predictability / entropy_factor).powf(alpha)
                    }
                    SamplingMethod::Like => entry.predictability.powf(alpha),
                    SamplingMethod::Exponent => {
                        let min_ent = match (&entry.unlike_probabilities, entry.unlike_probability)
                        {
                            (Some(unlike_probs), _) => min_entropy_pairs(unlike_probs),
                            (_, Some(unlike_prob)) => min_entropy(unlike_prob),
                            _ => 0.0,
                        };

                        if min_ent <= 0.0 {
                            println!(
                                "Warning: Zero entropy at index {}, setting weight to 0",
                                idx
                            );
                            0.0
                        } else {
                            entry.predictability.powf(alpha / min_ent)
                        }
                    }
                }
            })
            .collect()
    }

    fn generate_single_subset(
        weights: &[f64],
        size: usize,
        bad_indices: &HashSet<usize>,
        dimensions: usize,
    ) -> Result<Vec<usize>, GenerationError> {
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::WeightedIndex::new(weights)
            .map_err(|_| GenerationError("Failed to create weight distribution".to_string()))?;

        let mut positions = HashSet::new();
        let mut attempts = 0;

        while positions.len() < size && attempts < 1_000_000 {
            let pair_index = dist.sample(&mut rng);

            let i = pair_index / dimensions;
            let j = pair_index % dimensions;

            if bad_indices.contains(&i) || bad_indices.contains(&j) {
                attempts += 1;
                continue;
            }

            if !positions.contains(&i) {
                positions.insert(i);
            }
            if !positions.contains(&j) && positions.len() < size {
                positions.insert(j);
            }

            attempts += 1;
        }

        if positions.len() != size {
            return Err(GenerationError(
                "Failed to generate non-duplicating subset".to_string(),
            ));
        }

        Ok(positions.into_iter().collect())
    }

    pub fn generate(
        &self,
        count: usize,
        size: usize,
        alpha: f64,
        method: SamplingMethod,
    ) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
        let weights = Arc::new(self.calculate_weights(alpha, method));
        let bad_indices = Arc::new(self.bad_indices.clone());
        let dimensions = self.dimensions;

        let indices: Vec<Result<Vec<usize>, GenerationError>> = (0..count)
            .into_par_iter()
            .map(|_| {
                let weights = &weights;
                let bad_indices = &bad_indices;
                Self::generate_single_subset(weights, size, bad_indices, dimensions)
            })
            .collect();

        let mut result = Vec::with_capacity(count);
        for index_result in indices {
            result.push(index_result?);
        }

        Ok(result)
    }

    pub fn generate_and_store(
        output_path: &str,
        confidence_path: &str,
        count: usize,
        size: usize,
        alpha: f64,
        method: SamplingMethod,
        bad_indices: Option<Vec<usize>>,
        dimensions: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let confidence_entries = ConfidenceReader::read_confidence_file(confidence_path)?;
        let sampler = Self::new(confidence_entries, bad_indices, dimensions);

        let indices = sampler.generate(count, size, alpha, method)?;
        let data = SmartIndices(indices);

        let file = File::create(output_path)?;
        bincode::serialize_into(file, &data)?;

        println!(
            "Indices generated and stored successfully using {:?} method.",
            method
        );
        Ok(())
    }
}

#[derive(Debug)]
struct BitString {
    data: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
struct PairResults {
    count_00: usize,
    count_01: usize,
    count_10: usize,
    count_11: usize,
}

impl PairResults {
    fn to_normalized_array(&self) -> [f64; 4] {
        let total = self.count_00 + self.count_01 + self.count_10 + self.count_11;
        if total == 0 {
            return [0.0; 4];
        }
        let total = total as f64;
        [
            self.count_00 as f64 / total,
            self.count_01 as f64 / total,
            self.count_10 as f64 / total,
            self.count_11 as f64 / total,
        ]
    }

    fn add_comparison(&mut self, b1_same: bool, b2_same: bool) {
        match (b1_same, b2_same) {
            (true, true) => self.count_00 += 1,
            (true, false) => self.count_01 += 1,
            (false, true) => self.count_10 += 1,
            (false, false) => self.count_11 += 1,
        }
    }
}

fn generate_bit_pairs(dimensions: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(dimensions * (dimensions - 1) / 2);
    for i in 0..dimensions {
        for j in (i + 1)..dimensions {
            pairs.push((i, j));
        }
    }
    pairs
}

pub struct CorrelationAnalyzer {
    base_path: PathBuf,
    num_files: usize,
    dimensions: usize,
}

impl CorrelationAnalyzer {
    pub fn new<P: AsRef<Path>>(base_path: P, num_files: usize, dimensions: usize) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            num_files,
            dimensions,
        }
    }

    fn read_bitstring<P: AsRef<Path>>(path: P) -> std::io::Result<BitString> {
        let content = fs::read_to_string(path)?;
        let data: Vec<u8> = content
            .trim_end_matches(',')
            .split(',')
            .filter_map(|s| s.parse::<u8>().ok())
            .collect();
        Ok(BitString { data })
    }

    fn get_folder_paths(&self) -> std::io::Result<Vec<PathBuf>> {
        Ok(fs::read_dir(&self.base_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect())
    }

    fn get_random_files<P: AsRef<Path>>(path: P, count: usize) -> std::io::Result<Vec<PathBuf>> {
        let mut rng = rand::thread_rng();
        let files: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();

        Ok(files
            .choose_multiple(&mut rng, count.min(files.len()))
            .cloned()
            .collect())
    }

    fn process_pair_single_bits(bitstring1: &[u8], bitstring2: &[u8]) -> Vec<f64> {
        let len = bitstring1.len().min(bitstring2.len());
        bitstring1
            .iter()
            .take(len)
            .zip(bitstring2.iter().take(len))
            .map(|(b1, b2)| if b1 != b2 { 1.0 } else { 0.0 })
            .collect()
    }

    fn process_pair_bit_pairs(
        bitstring1: &[u8],
        bitstring2: &[u8],
        pairs: &[(usize, usize)],
    ) -> Vec<PairResults> {
        let mut results = vec![PairResults::default(); pairs.len()];
        let len = bitstring1.len().min(bitstring2.len());

        for (idx, &(pos1, pos2)) in pairs.iter().enumerate() {
            if pos1 >= len || pos2 >= len {
                continue;
            }

            let str1_first = bitstring1[pos1];
            let str1_second = bitstring1[pos2];
            let str2_first = bitstring2[pos1];
            let str2_second = bitstring2[pos2];

            results[idx].add_comparison(str1_first == str2_first, str1_second == str2_second);
        }

        results
    }

    fn process_folder_pair(
        folder_pair: (&PathBuf, &PathBuf),
        num_files: usize,
        pairs: &[(usize, usize)],
        progress_counter: &Arc<AtomicUsize>,
    ) -> std::io::Result<(Vec<f64>, Vec<PairResults>)> {
        let (folder1, folder2) = folder_pair;
        let files1 = Self::get_random_files(folder1, num_files)?;
        let files2 = Self::get_random_files(folder2, num_files)?;

        let mut single_bit_results = Vec::new();
        let mut pair_results = Vec::new();
        let mut comparison_count = 0;

        for (f1, f2) in files1.iter().zip(files2.iter()) {
            let bitstring1 = Self::read_bitstring(f1)?;
            let bitstring2 = Self::read_bitstring(f2)?;

            if single_bit_results.is_empty() {
                single_bit_results = vec![0.0; bitstring1.data.len().min(bitstring2.data.len())];
                pair_results = vec![PairResults::default(); pairs.len()];
            }

            let pair_single_results =
                Self::process_pair_single_bits(&bitstring1.data, &bitstring2.data);
            let pair_pair_results =
                Self::process_pair_bit_pairs(&bitstring1.data, &bitstring2.data, pairs);

            for (total, current) in single_bit_results
                .iter_mut()
                .zip(pair_single_results.iter())
            {
                *total += current;
            }

            for (total, current) in pair_results.iter_mut().zip(pair_pair_results.iter()) {
                total.count_00 += current.count_00;
                total.count_01 += current.count_01;
                total.count_10 += current.count_10;
                total.count_11 += current.count_11;
            }

            comparison_count += 1;
        }

        if comparison_count > 0 {
            for total in single_bit_results.iter_mut() {
                *total /= comparison_count as f64;
            }
        }

        progress_counter.fetch_add(1, Ordering::Relaxed);
        Ok((single_bit_results, pair_results))
    }

    pub fn generate_correlation_report<P: AsRef<Path>>(
        &self,
        output: P,
        mode: AnalysisMode,
    ) -> std::io::Result<()> {
        let folders = self.get_folder_paths()?;
        let pairs = generate_bit_pairs(self.dimensions);
        println!("Generated {} bit pairs", pairs.len());

        let multi_progress = MultiProgress::new();
        let progress_style = ProgressStyle::default_bar()
            .template("{prefix:.bold.dim} [{wide_bar:.cyan/blue}] {pos}/{len} folder pairs ({eta})")
            .unwrap()
            .progress_chars("█▇▆▅▄▃▂▁  ");

        let same_class_pairs: Vec<_> = folders.iter().map(|f| (f, f)).collect();
        let diff_class_pairs: Vec<_> = folders
            .iter()
            .enumerate()
            .flat_map(|(i, f1)| folders[i + 1..].iter().map(move |f2| (f1, f2)))
            .collect();

        println!(
            "Processing {} same-class and {} different-class folder pairs",
            same_class_pairs.len(),
            diff_class_pairs.len()
        );

        let same_class_counter = Arc::new(AtomicUsize::new(0));
        let diff_class_counter = Arc::new(AtomicUsize::new(0));

        let same_class_pb = multi_progress.add(ProgressBar::new(same_class_pairs.len() as u64));
        same_class_pb.set_style(progress_style.clone());
        same_class_pb.set_prefix("Same Class");

        let diff_class_pb = multi_progress.add(ProgressBar::new(diff_class_pairs.len() as u64));
        diff_class_pb.set_style(progress_style.clone());
        diff_class_pb.set_prefix("Diff Class");

        let same_class_results: Vec<(Vec<f64>, Vec<PairResults>)> = same_class_pairs
            .par_iter()
            .filter_map(|pair| {
                let result =
                    Self::process_folder_pair(*pair, self.num_files, &pairs, &same_class_counter);
                same_class_pb.set_position(same_class_counter.load(Ordering::Relaxed) as u64);
                result.ok()
            })
            .collect();

        same_class_pb.finish_with_message("Complete");

        let diff_class_results: Vec<(Vec<f64>, Vec<PairResults>)> = diff_class_pairs
            .par_iter()
            .filter_map(|pair| {
                let result =
                    Self::process_folder_pair(*pair, self.num_files, &pairs, &diff_class_counter);
                diff_class_pb.set_position(diff_class_counter.load(Ordering::Relaxed) as u64);
                result.ok()
            })
            .collect();

        diff_class_pb.finish_with_message("Complete");

        let results_pb = multi_progress.add(ProgressBar::new(pairs.len() as u64));
        results_pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:.bold.dim} [{wide_bar:.cyan/blue}] {pos}/{len} pairs ({eta})")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );
        results_pb.set_prefix("Processing");

        let file = File::create(output)?;
        let mut writer = BufWriter::new(file);

        match mode {
            AnalysisMode::Single | AnalysisMode::Both => {
                let single_bit_len = same_class_results.first().map_or(0, |(v, _)| v.len());
                let mut same_class_avg = vec![0.0; single_bit_len];
                let mut diff_class_avg = vec![0.0; single_bit_len];

                for idx in 0..single_bit_len {
                    same_class_avg[idx] =
                        same_class_results.iter().map(|(v, _)| v[idx]).sum::<f64>()
                            / same_class_results.len() as f64;

                    diff_class_avg[idx] =
                        diff_class_results.iter().map(|(v, _)| v[idx]).sum::<f64>()
                            / diff_class_results.len() as f64;

                    writeln!(
                        writer,
                        "Hamming distance, Like/Unlike/Difference = {} {} {} {}",
                        idx,
                        same_class_avg[idx],
                        diff_class_avg[idx],
                        diff_class_avg[idx] - same_class_avg[idx]
                    )?;
                }
            }
            _ => {}
        }

        match mode {
            AnalysisMode::Pairs | AnalysisMode::Both => {
                let pair_len = pairs.len();
                let mut same_class_pair_results = vec![PairResults::default(); pair_len];
                let mut diff_class_pair_results = vec![PairResults::default(); pair_len];

                for idx in 0..pair_len {
                    for (_, pair_results) in &same_class_results {
                        same_class_pair_results[idx].count_00 += pair_results[idx].count_00;
                        same_class_pair_results[idx].count_01 += pair_results[idx].count_01;
                        same_class_pair_results[idx].count_10 += pair_results[idx].count_10;
                        same_class_pair_results[idx].count_11 += pair_results[idx].count_11;
                    }

                    for (_, pair_results) in &diff_class_results {
                        diff_class_pair_results[idx].count_00 += pair_results[idx].count_00;
                        diff_class_pair_results[idx].count_01 += pair_results[idx].count_01;
                        diff_class_pair_results[idx].count_10 += pair_results[idx].count_10;
                        diff_class_pair_results[idx].count_11 += pair_results[idx].count_11;
                    }

                    let same_normalized = same_class_pair_results[idx].to_normalized_array();
                    let diff_normalized = diff_class_pair_results[idx].to_normalized_array();

                    let same_sum: f64 = same_normalized.iter().sum();
                    let diff_sum: f64 = diff_normalized.iter().sum();
                    assert!(
                        (same_sum - 1.0).abs() < 1e-10,
                        "Same class probabilities sum to {}",
                        same_sum
                    );
                    assert!(
                        (diff_sum - 1.0).abs() < 1e-10,
                        "Different class probabilities sum to {}",
                        diff_sum
                    );

                    writeln!(
                        writer,
                        "Hamming distance, Like/Unlike/Difference ({}, {}) {} {} {} {} {} {} {} {}",
                        pairs[idx].0,
                        pairs[idx].1,
                        1.0 - same_normalized[0],
                        same_normalized[1],
                        same_normalized[2],
                        same_normalized[3],
                        diff_normalized[0],
                        diff_normalized[1],
                        diff_normalized[2],
                        diff_normalized[3]
                    )?;

                    results_pb.inc(1);
                }
            }
            _ => {}
        }

        results_pb.finish_with_message("Complete");
        Ok(())
    }
}
