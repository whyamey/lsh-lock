use bincode;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::BinarySimilarity;
use std::fs::{self, File};
use std::io::{BufWriter, Read};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug)]
pub struct Template {
    pub class: String,
    pub data: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct RandomIndices(pub Vec<Vec<usize>>);

pub struct TemplateReader;

impl TemplateReader {
    pub fn read_templates(
        iris_fat_path: &str,
    ) -> Result<Vec<Template>, Box<dyn std::error::Error>> {
        let mut templates = Vec::new();
        for entry in fs::read_dir(iris_fat_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let class = path.file_name().unwrap().to_str().unwrap().to_string();
                for file_entry in fs::read_dir(path)? {
                    let file_entry = file_entry?;
                    let file_path = file_entry.path();
                    if file_path.is_file() {
                        let mut content = String::new();
                        File::open(&file_path)?.read_to_string(&mut content)?;
                        let data: Vec<u8> = content
                            .trim_end_matches(',')
                            .split(',')
                            .map(|s| s.parse::<u8>().unwrap())
                            .collect();
                        templates.push(Template {
                            class: class.clone(),
                            data,
                        });
                    }
                }
            }
        }
        Ok(templates)
    }
}

pub struct RandomIndicesGenerator;

impl RandomIndicesGenerator {
    pub fn generate_and_store(
        file_path: &str,
        count: usize,
        size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        let random_indices: Vec<Vec<usize>> = (0..count)
            .map(|_| (0..size).map(|_| rng.gen_range(0..1024)).collect())
            .collect();

        let data = RandomIndices(random_indices);
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data)?;

        println!("Random indices generated and stored successfully.");
        Ok(())
    }

    pub fn load(file_path: &str) -> Result<RandomIndices, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let random_indices: RandomIndices = bincode::deserialize_from(file)?;
        Ok(random_indices)
    }
}

pub struct AnalysisTool;

impl AnalysisTool {
    #[inline(always)]
    fn calc_hamming(a: &[u8], b: &[u8]) -> f64 {
        u8::hamming(a, b).unwrap_or(0.0)
    }

    pub fn calculate_class_based_fractional_hamming_mean_and_entropy(
        templates: &[Template],
        indices: &[Vec<usize>],
    ) -> (f64, f64, Arc<Mutex<Vec<f64>>>) {
        let total_indices = indices.len();
        let progress = Arc::new(AtomicUsize::new(0));
        let entropy_store = Arc::new(Mutex::new(vec![0.0; total_indices]));
        let template_count = templates.len();

        let class_diff: Vec<Vec<bool>> = templates
            .iter()
            .map(|t1| templates.iter().map(|t2| t1.class != t2.class).collect())
            .collect();

        let selected_templates: Vec<Vec<Vec<u8>>> = indices
            .par_iter()
            .map(|index_set| {
                templates
                    .iter()
                    .map(|t| index_set.iter().map(|&i| t.data[i]).collect())
                    .collect()
            })
            .collect();

        let results: Vec<(f64, f64)> = (0..total_indices)
            .into_par_iter()
            .map(|index| {
                let mut diff_class_sum = 0.0;
                let mut diff_class_count = 0;
                let mut variance_sum = 0.0;

                for i in 0..template_count {
                    for j in (i + 1)..template_count {
                        if class_diff[i][j] {
                            let distance = Self::calc_hamming(
                                &selected_templates[index][i],
                                &selected_templates[index][j],
                            );
                            let normalized_distance = distance / (indices[index].len() as f64);
                            diff_class_sum += normalized_distance;
                            diff_class_count += 1;
                            variance_sum += normalized_distance * normalized_distance;
                        }
                    }
                }

                let diff_class_mean = if diff_class_count > 0 {
                    diff_class_sum / diff_class_count as f64
                } else {
                    0.0
                };

                let variance = if diff_class_count > 0 {
                    (variance_sum / diff_class_count as f64) - diff_class_mean * diff_class_mean
                } else {
                    0.0
                };

                let degrees_freedom = if variance != 0.0 {
                    (diff_class_mean * (1.0 - diff_class_mean)) / variance
                } else {
                    0.0
                };
                let min_entropy = if diff_class_mean > 0.0 && diff_class_mean < 1.0 {
                    f64::min(-diff_class_mean.log2(), -(1.0 - diff_class_mean).log2())
                } else {
                    0.0
                };
                let entropy = degrees_freedom * min_entropy;

                entropy_store.lock().unwrap()[index] = entropy;

                let current_progress = progress.fetch_add(1, Ordering::SeqCst) + 1;
                if current_progress % (total_indices / 10) == 0 {
                    println!("Progress: {}%", (current_progress * 100) / total_indices);
                }

                (diff_class_mean, entropy)
            })
            .collect();

        let (diff_class_mean_sum, entropy_sum): (f64, f64) = results
            .iter()
            .fold((0.0, 0.0), |acc, &(diff_class_mean, entropy)| {
                (acc.0 + diff_class_mean, acc.1 + entropy)
            });

        let avg_diff_class_mean = diff_class_mean_sum / total_indices as f64;
        let avg_entropy = entropy_sum / total_indices as f64;

        (avg_diff_class_mean, avg_entropy, entropy_store)
    }
}
