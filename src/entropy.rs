use bincode;
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::BinarySimilarity;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read};
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

#[derive(Debug)]
pub struct FloatTemplate {
    pub class: String,
    pub data: Vec<f32>,
}

pub struct FloatTemplateReader;

impl FloatTemplateReader {
    pub fn read_templates(
        embeddings_path: &str,
    ) -> Result<Vec<FloatTemplate>, Box<dyn std::error::Error>> {
        let mut templates = Vec::new();
        for entry in fs::read_dir(embeddings_path)? {
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
                        let data: Vec<f32> = content
                            .trim_end_matches(',')
                            .split(',')
                            .map(|s| s.parse::<f32>().unwrap())
                            .collect();
                        templates.push(FloatTemplate {
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
        dimensions: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        println!(
            "Generating {} lockers with {} unique positions each...",
            count, size
        );
        let progress_chunk = count / 10;

        let random_indices: Vec<Vec<usize>> = (0..count)
            .map(|i| {
                let mut indices = HashSet::with_capacity(size);
                let mut attempts = 0;
                const MAX_ATTEMPTS: usize = 1_000_000;

                while indices.len() < size && attempts < MAX_ATTEMPTS {
                    indices.insert(rng.gen_range(0..dimensions));
                    attempts += 1;
                }

                if indices.len() < size {
                    return Err(format!(
                        "Failed to generate enough unique indices after {} attempts",
                        MAX_ATTEMPTS
                    ));
                }

                // Print progress every 10%
                if progress_chunk > 0 && i % progress_chunk == 0 {
                    println!("Progress: {}%", (i * 100) / count);
                }

                Ok(indices.into_iter().collect())
            })
            .collect::<Result<Vec<Vec<usize>>, String>>()?;

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

#[derive(Serialize, Deserialize)]
pub struct CosineLocker {
    seeds: Vec<u64>,
}

impl CosineLocker {
    pub fn new(k: usize) -> Self {
        let mut rng = rand::thread_rng();
        let seeds: Vec<u64> = (0..k).map(|_| rng.gen()).collect();
        Self { seeds }
    }

    pub fn get_projection_vectors(&self) -> Vec<Vec<f32>> {
        self.seeds
            .iter()
            .map(|&seed| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut proj: Vec<f32> = (0..1024).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let norm = (proj.iter().map(|x| x * x).sum::<f32>()).sqrt();
                proj.iter_mut().for_each(|x| *x /= norm);
                proj
            })
            .collect()
    }
}

pub struct CosineLockerGenerator;

impl CosineLockerGenerator {
    pub fn generate_and_store(
        file_path: &str,
        count: usize,
        size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "Generating {} lockers with {} projections each...",
            count, size
        );

        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        bincode::serialize_into(&mut writer, &count)?;

        for i in 0..count {
            let locker = CosineLocker::new(size);
            bincode::serialize_into(&mut writer, &locker.seeds)?;

            if (i + 1) % (count / 10).max(1) == 0 {
                println!("Progress: {}%", ((i + 1) * 100) / count);
            }
        }

        Ok(())
    }

    pub fn load(file_path: &str) -> Result<Vec<CosineLocker>, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        let count: usize = bincode::deserialize_from(&mut reader)?;
        let mut lockers = Vec::with_capacity(count);

        for _ in 0..count {
            let seeds: Vec<u64> = bincode::deserialize_from(&mut reader)?;
            lockers.push(CosineLocker { seeds });
        }

        Ok(lockers)
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
    ) -> (f64, f64, f64, Arc<Mutex<Vec<f64>>>) {
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

        let min_entropy = entropy_store
            .lock()
            .unwrap()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        let avg_diff_class_mean = diff_class_mean_sum / total_indices as f64;
        let avg_entropy = entropy_sum / total_indices as f64;

        (avg_diff_class_mean, avg_entropy, min_entropy, entropy_store)
    }

    pub fn calculate_cosine_entropy(
        templates: &[FloatTemplate],
        lockers: &[CosineLocker],
    ) -> (f64, f64, f64, Arc<Mutex<Vec<f64>>>) {
        let total_lockers = lockers.len();
        let progress = Arc::new(AtomicUsize::new(0));
        let entropy_store = Arc::new(Mutex::new(vec![0.0; total_lockers]));

        let template_matrix: Array2<f32> = Array2::from_shape_vec(
            (templates.len(), templates[0].data.len()),
            templates.iter().flat_map(|t| t.data.clone()).collect(),
        )
        .unwrap();

        let class_pairs: Array2<bool> =
            Array2::from_shape_fn((templates.len(), templates.len()), |(i, j)| {
                templates[i].class != templates[j].class
            });

        let results: Vec<(f64, f64)> = lockers
            .par_iter()
            .enumerate()
            .map(|(locker_idx, locker)| {
                let proj_matrix = Array2::from_shape_vec(
                    (template_matrix.ncols(), locker.seeds.len()),
                    locker
                        .get_projection_vectors()
                        .into_iter()
                        .flat_map(|v| v)
                        .collect(),
                )
                .unwrap();

                let projections = template_matrix.dot(&proj_matrix);
                let signs = projections.mapv(|x| x > 0.0);

                let mut diff_class_distances = Vec::new();

                for i in 0..templates.len() {
                    for j in (i + 1)..templates.len() {
                        if class_pairs[[i, j]] {
                            let distance = signs
                                .row(i)
                                .iter()
                                .zip(signs.row(j).iter())
                                .filter(|(&a, &b)| a != b)
                                .count() as f64
                                / signs.ncols() as f64;
                            diff_class_distances.push(distance);
                        }
                    }
                }

                let diff_class_count = diff_class_distances.len();
                let (diff_class_mean, variance) = if diff_class_count > 0 {
                    let mean = diff_class_distances.iter().sum::<f64>() / diff_class_count as f64;
                    let var = diff_class_distances
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .sum::<f64>()
                        / diff_class_count as f64;
                    (mean, var)
                } else {
                    (0.0, 0.0)
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
                entropy_store.lock().unwrap()[locker_idx] = entropy;

                let current_progress = progress.fetch_add(1, Ordering::SeqCst) + 1;
                if current_progress % (total_lockers / 10).max(1) == 0 {
                    println!("Progress: {}%", (current_progress * 100) / total_lockers);
                }

                (diff_class_mean, entropy)
            })
            .collect();

        let (diff_class_mean_sum, entropy_sum) = results
            .iter()
            .fold((0.0, 0.0), |acc, &(diff_class_mean, entropy)| {
                (acc.0 + diff_class_mean, acc.1 + entropy)
            });

        let min_entropy = entropy_store
            .lock()
            .unwrap()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        (
            diff_class_mean_sum / total_lockers as f64,
            entropy_sum / total_lockers as f64,
            min_entropy,
            entropy_store,
        )
    }
}
