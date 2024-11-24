use crate::entropy::CosineLocker;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use simsimd::BinarySimilarity;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct Template {
    data: Vec<u8>,
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

        let mut total_successes = 0;
        let mut total_comparisons = 0;
        let total_dirs = class_dirs.len();

        for class_dir in class_dirs {
            let files: Vec<_> = fs::read_dir(class_dir)?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .collect();

            if files.len() < 2 {
                continue;
            }

            let templates: Vec<Vec<f32>> = files
                .par_iter()
                .filter_map(|file| Self::read_single_template(file).ok())
                .collect();

            let all_hashes: Vec<Vec<Vec<u8>>> = templates
                .par_iter()
                .map(|template| lockers.iter().map(|locker| locker.hash(template)).collect())
                .collect();

            for i in 0..templates.len() {
                for j in (i + 1)..templates.len() {
                    total_comparisons += 1;
                    if all_hashes[i]
                        .iter()
                        .zip(all_hashes[j].iter())
                        .any(|(hash1, hash2)| hash1 == hash2)
                    {
                        total_successes += 1;
                    }
                }
            }

            let current_progress = progress.fetch_add(1, Ordering::SeqCst) + 1;
            if current_progress % (total_dirs / 10).max(1) == 0 {
                println!("Progress: {}%", (current_progress * 100) / total_dirs);
            }
        }

        let tar = if total_comparisons > 0 {
            total_successes as f64 / total_comparisons as f64
        } else {
            0.0
        };

        Ok((tar, total_successes, total_comparisons))
    }
}
