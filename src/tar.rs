use crate::entropy::CosineLocker;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use simsimd::BinarySimilarity;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug)]
struct Template {
    data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClassSelection {
    pub class_name: String,
    pub base_files: Vec<String>,
    pub comparison_files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SelectionData {
    pub selections: HashMap<String, ClassSelection>,
}

pub struct TARAnalyzer;

impl TARAnalyzer {
    fn parse_binary_file<P: AsRef<Path>>(path: P) -> std::io::Result<Template> {
        let mut content = String::new();
        File::open(path)?.read_to_string(&mut content)?;

        let data: Vec<u8> = content
            .trim_end_matches(',')
            .split(',')
            .filter_map(|s| s.parse::<u8>().ok())
            .collect();

        Ok(Template { data })
    }

    #[inline(always)]
    fn calc_hamming(a: &[u8], b: &[u8]) -> f64 {
        u8::hamming(a, b).unwrap_or(f64::MAX)
    }

    #[inline(always)]
    fn compare_permutations(base_perm: &[Vec<u8>], target_perm: &[Vec<u8>]) -> bool {
        base_perm
            .iter()
            .zip(target_perm.iter())
            .any(|(base, target)| Self::calc_hamming(base, target) == 0.0)
    }

    fn create_permutations_batch(
        templates: &[Template],
        positions: &[Vec<usize>],
    ) -> Vec<Vec<Vec<u8>>> {
        templates
            .par_iter()
            .map(|template| {
                positions
                    .iter()
                    .map(|pos_set| {
                        pos_set
                            .iter()
                            .map(|&pos| template.data[pos])
                            .collect::<Vec<u8>>()
                    })
                    .collect()
            })
            .collect()
    }

    fn process_single_class<P: AsRef<Path>>(
        class_path: P,
        positions: &[Vec<usize>],
    ) -> std::io::Result<(usize, usize)> {
        let class_path = class_path.as_ref();
        let files: Vec<_> = fs::read_dir(class_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();

        if files.len() < 2 {
            return Ok((0, 0));
        }

        let mut rng = rand::thread_rng();
        let files = if files.len() > 11 {
            files
                .choose_multiple(&mut rng, 11)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            files
        };

        let templates: Vec<Template> = files
            .par_iter()
            .filter_map(|file| Self::parse_binary_file(file).ok())
            .collect();

        if templates.is_empty() {
            return Ok((0, 0));
        }

        let all_permutations = Self::create_permutations_batch(&templates, positions);

        let base_idx = rand::thread_rng().gen_range(0..templates.len());
        let base_permutations = &all_permutations[base_idx];

        let success_count = all_permutations
            .par_iter()
            .enumerate()
            .filter(|(idx, _)| *idx != base_idx)
            .filter(|(_, target_permutations)| {
                Self::compare_permutations(base_permutations, target_permutations)
            })
            .count();

        Ok((success_count, templates.len() - 1))
    }

    pub fn analyze_tar<P: AsRef<Path>>(
        feature_directory: P,
        positions: &[Vec<usize>],
    ) -> std::io::Result<(f64, usize, usize)> {
        let progress = Arc::new(AtomicUsize::new(0));
        let feature_directory = feature_directory.as_ref();

        let class_dirs: Vec<_> = fs::read_dir(feature_directory)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();

        let total_dirs = class_dirs.len();

        let results: Vec<(usize, usize)> = class_dirs
            .par_iter()
            .map(|class_path| {
                let result = Self::process_single_class(class_path, positions).unwrap_or((0, 0));

                let current_progress = progress.fetch_add(1, Ordering::SeqCst) + 1;
                if current_progress % (total_dirs / 10).max(1) == 0 {
                    println!("Progress: {}%", (current_progress * 100) / total_dirs);
                }

                result
            })
            .collect();

        let total_successes: usize = results.iter().map(|(success, _)| success).sum();
        let total_comparisons: usize = results.iter().map(|(_, comparisons)| comparisons).sum();

        let tar = if total_comparisons > 0 {
            total_successes as f64 / total_comparisons as f64
        } else {
            0.0
        };

        Ok((tar, total_successes, total_comparisons))
    }

    fn read_single_template<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<f32>> {
        let mut content = String::new();
        File::open(path)?.read_to_string(&mut content)?;

        Ok(content
            .trim_end_matches(',')
            .split(',')
            .filter_map(|s| s.parse::<f32>().ok())
            .collect())
    }

    pub fn analyze_tar_multi<P: AsRef<Path>>(
        feature_directory: P,
        positions: &[Vec<usize>],
        tries: usize,
        base_count: usize,
        input_selection: Option<&str>,
        output_selection: Option<&str>,
    ) -> std::io::Result<(f64, usize, usize, Option<SelectionData>)> {
        if base_count >= tries {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Base count must be less than the number of tries",
            ));
        }

        let progress = Arc::new(AtomicUsize::new(0));
        let feature_directory = feature_directory.as_ref();

        let class_dirs: Vec<PathBuf> = fs::read_dir(feature_directory)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();

        let total_dirs = class_dirs.len();

        let input_selections = if let Some(input_path) = input_selection {
            let file = File::open(input_path)?;
            let reader = BufReader::new(file);
            let selection_data: SelectionData = serde_json::from_reader(reader)?;
            Some(selection_data)
        } else {
            None
        };

        let output_selections = Arc::new(Mutex::new(SelectionData {
            selections: HashMap::new(),
        }));
        let should_output = output_selection.is_some();

        let results: Vec<(usize, usize)> = class_dirs
            .par_iter()
            .filter_map(|class_path| {
                let class_name = class_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let result = if let Some(ref input_sel) = input_selections {
                    if let Some(class_sel) = input_sel.selections.get(&class_name) {
                        Self::process_with_selection(class_path, positions, &class_sel)
                    } else {
                        None
                    }
                } else {
                    let result = Self::process_single_class_multi_with_indices(
                        class_path, positions, tries, base_count,
                    )
                    .ok()
                    .flatten();

                    if should_output {
                        if let Some((_, _, base_files, comparison_files)) = &result {
                            let mut selections = output_selections.lock().unwrap();
                            selections.selections.insert(
                                class_name.clone(),
                                ClassSelection {
                                    class_name: class_name.clone(),
                                    base_files: base_files.clone(),
                                    comparison_files: comparison_files.clone(),
                                },
                            );
                        }
                    }

                    result
                };

                let current_progress = progress.fetch_add(1, Ordering::SeqCst) + 1;
                if current_progress % (total_dirs / 10).max(1) == 0 {
                    println!("Progress: {}%", (current_progress * 100) / total_dirs);
                }

                result.map(|(success, total, _, _)| (success, total))
            })
            .collect();

        let classes_passed: usize = results.iter().map(|(success, _)| *success).sum();
        let total_classes = results.len();

        let tar = if total_classes > 0 {
            classes_passed as f64 / total_classes as f64
        } else {
            0.0
        };

        let selection_data = if should_output {
            let selections = Arc::try_unwrap(output_selections)
                .unwrap()
                .into_inner()
                .unwrap();

            if let Some(output_path) = output_selection {
                let file = File::create(output_path)?;
                let writer = BufWriter::new(file);
                serde_json::to_writer_pretty(writer, &selections)?;
                println!("Selection data saved to {}", output_path);
            }

            Some(selections)
        } else {
            None
        };

        Ok((tar, classes_passed, total_classes, selection_data))
    }

    fn process_single_class_multi_with_indices<P: AsRef<Path>>(
        class_path: P,
        positions: &[Vec<usize>],
        tries: usize,
        base_count: usize,
    ) -> std::io::Result<Option<(usize, usize, Vec<String>, Vec<String>)>> {
        let class_path = class_path.as_ref();
        let files: Vec<PathBuf> = fs::read_dir(class_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();

        if files.len() < tries {
            return Ok(None);
        }

        let mut rng = rand::thread_rng();
        let selected_files: Vec<PathBuf> =
            files.choose_multiple(&mut rng, tries).cloned().collect();

        let templates: Vec<Template> = selected_files
            .par_iter()
            .filter_map(|file| Self::parse_binary_file(file).ok())
            .collect();

        if templates.len() < tries {
            return Ok(None);
        }

        let all_permutations = Self::create_permutations_batch(&templates, positions);

        let mut indices: Vec<usize> = (0..templates.len()).collect();
        indices.shuffle(&mut rng);

        let base_indices: Vec<usize> = indices.iter().take(base_count).cloned().collect();
        let comparison_indices: Vec<usize> = indices
            .iter()
            .skip(base_count)
            .take(tries - base_count)
            .cloned()
            .collect();

        let base_files: Vec<String> = base_indices
            .iter()
            .map(|&idx| {
                selected_files[idx]
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            })
            .collect();

        let comparison_files: Vec<String> = comparison_indices
            .iter()
            .map(|&idx| {
                selected_files[idx]
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            })
            .collect();

        let base_permutations: Vec<&Vec<Vec<u8>>> = base_indices
            .iter()
            .map(|&idx| &all_permutations[idx])
            .collect();

        let found_match = comparison_indices.par_iter().any(|&idx| {
            let target_permutations = &all_permutations[idx];
            base_permutations
                .iter()
                .any(|base_perm| Self::compare_permutations(base_perm, target_permutations))
        });

        Ok(Some((
            if found_match { 1 } else { 0 },
            1,
            base_files,
            comparison_files,
        )))
    }

    fn process_with_selection<P: AsRef<Path>>(
        class_path: P,
        positions: &[Vec<usize>],
        selection: &ClassSelection,
    ) -> Option<(usize, usize, Vec<String>, Vec<String>)> {
        let class_path = class_path.as_ref();

        let files: HashMap<String, PathBuf> = match fs::read_dir(class_path) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let path = entry.path();
                    if path.is_file() {
                        let filename = path.file_name().and_then(|n| n.to_str()).map(String::from);
                        filename.map(|name| (name, path))
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => return None,
        };

        let base_files: Vec<PathBuf> = selection
            .base_files
            .iter()
            .filter_map(|name| files.get(name).cloned())
            .collect();

        let comparison_files: Vec<PathBuf> = selection
            .comparison_files
            .iter()
            .filter_map(|name| files.get(name).cloned())
            .collect();

        if base_files.len() != selection.base_files.len()
            || comparison_files.len() != selection.comparison_files.len()
        {
            return None;
        }

        let all_files: Vec<PathBuf> = base_files
            .iter()
            .chain(comparison_files.iter())
            .cloned()
            .collect();

        let templates: Vec<Template> = match all_files
            .iter()
            .map(|file| Self::parse_binary_file(file))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(t) => t,
            Err(_) => return None,
        };

        let all_permutations = Self::create_permutations_batch(&templates, positions);

        let base_count = base_files.len();
        let base_permutations: Vec<&Vec<Vec<u8>>> =
            (0..base_count).map(|idx| &all_permutations[idx]).collect();

        let found_match = (base_count..templates.len()).any(|idx| {
            let target_permutations = &all_permutations[idx];
            base_permutations
                .iter()
                .any(|base_perm| Self::compare_permutations(base_perm, target_permutations))
        });

        Some((
            if found_match { 1 } else { 0 },
            1,
            selection.base_files.clone(),
            selection.comparison_files.clone(),
        ))
    }

    pub fn analyze_cosine_tar<P: AsRef<Path>>(
        feature_directory: P,
        lockers: &[CosineLocker],
    ) -> std::io::Result<(f64, usize, usize)> {
        let progress = Arc::new(AtomicUsize::new(0));
        let feature_directory = feature_directory.as_ref();

        let class_dirs: Vec<_> = fs::read_dir(feature_directory)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();

        let total_dirs = class_dirs.len();
        let results = Arc::new(Mutex::new((0, 0)));

        let projection_matrices: Vec<Array2<f32>> = lockers
            .par_iter()
            .map(|locker| {
                let vectors = locker.get_projection_vectors();
                Array2::from_shape_vec(
                    (vectors.len(), vectors[0].len()),
                    vectors.into_iter().flat_map(|v| v).collect(),
                )
                .unwrap()
            })
            .collect();

        class_dirs.into_par_iter().for_each(|class_dir| {
            let files: Vec<_> = fs::read_dir(&class_dir)
                .unwrap()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .collect();

            if files.len() < 2 {
                return;
            }

            let templates: Vec<Vec<f32>> = files
                .par_iter()
                .filter_map(|file| Self::read_single_template(file).ok())
                .collect();

            let mut local_successes = 0;
            let mut local_comparisons = 0;

            let templates_array: Vec<Array1<f32>> = templates
                .iter()
                .map(|t| Array1::from_vec(t.clone()))
                .collect();

            for i in 0..templates.len() {
                for j in (i + 1)..templates.len() {
                    local_comparisons += 1;
                    let mut found_match = false;

                    for proj_matrix in &projection_matrices {
                        let proj1 = proj_matrix.dot(&templates_array[i]);
                        let proj2 = proj_matrix.dot(&templates_array[j]);

                        let signs1: Vec<bool> = proj1.iter().map(|&x| x > 0.0).collect();
                        let signs2: Vec<bool> = proj2.iter().map(|&x| x > 0.0).collect();

                        if signs1 == signs2 {
                            found_match = true;
                            break;
                        }
                    }

                    if found_match {
                        local_successes += 1;
                    }
                }
            }

            let mut results = results.lock().unwrap();
            results.0 += local_successes;
            results.1 += local_comparisons;

            let current_progress = progress.fetch_add(1, Ordering::SeqCst) + 1;
            if current_progress % (total_dirs / 10).max(1) == 0 {
                println!("Progress: {}%", (current_progress * 100) / total_dirs);
            }
        });

        let (total_successes, total_comparisons) = *results.lock().unwrap();
        let tar = if total_comparisons > 0 {
            total_successes as f64 / total_comparisons as f64
        } else {
            0.0
        };

        Ok((tar, total_successes, total_comparisons))
    }
}
