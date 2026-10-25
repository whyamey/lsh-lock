//! Entropy of many coordinate sets, with an independent-bit statistical check.
//!
//! For each set S this computes Daugman's degrees-of-freedom estimate, exactly as
//! `AnalysisTool::calculate_class_based_fractional_hamming_mean_and_entropy` does
//! (impostor fractional Hamming distances on S, dof = p(1-p)/var, entropy =
//! dof * min(-log2 p, -log2(1-p))), but
//!   * over a fixed seeded sample of impostor pairs (or all of them),
//!   * also on the two halves of that sample (for a split-half noise estimate), and
//!   * next to the independent-bit min-entropy sum_{j in S} -log2 max(p_j, 1-p_j),
//!     where p_j is the class-balanced frequency of bit j on the prior classes.
//! The independent-bit value is a statistical check, not a second security claim:
//! it ignores correlations between coordinates, which Daugman's test captures.
use crate::entropy::{RandomIndicesGenerator, Template, TemplateReader};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::BufWriter;

#[derive(Serialize)]
pub struct SetReport {
    pub templates: usize,
    pub classes: usize,
    pub prior_classes: usize,
    pub impostor_pairs: usize,
    pub all_pairs: bool,
    pub seed: u64,
    pub size: Vec<usize>,
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub dof: Vec<f64>,
    pub daugman: Vec<f64>,
    pub daugman_half_a: Vec<f64>,
    pub daugman_half_b: Vec<f64>,
    pub independent_bit: Vec<f64>,
    pub prior: Vec<f64>,
    pub summary: BTreeMap<String, f64>,
}

fn daugman(sum: f64, sq: f64, count: usize) -> (f64, f64, f64, f64) {
    if count == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = sum / count as f64;
    let var = (sq / count as f64 - mean * mean).max(0.0);
    let dof = if var > 0.0 { mean * (1.0 - mean) / var } else { 0.0 };
    let h = if mean > 0.0 && mean < 1.0 {
        f64::min(-mean.log2(), -(1.0 - mean).log2())
    } else {
        0.0
    };
    (mean, var, dof, dof * h)
}

fn pack(t: &Template, set: &[usize], out: &mut [u8]) {
    out.iter_mut().for_each(|b| *b = 0);
    for (k, &j) in set.iter().enumerate() {
        if t.data[j] != 0 {
            out[k >> 3] |= 1 << (k & 7);
        }
    }
}

/// Seeded impostor pairs (i < j, different classes); `count == 0` means all of them.
pub fn impostor_pairs(templates: &[Template], count: usize, seed: u64) -> (Vec<(u32, u32)>, bool) {
    let n = templates.len();
    if count == 0 {
        let mut all = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if templates[i].class != templates[j].class {
                    all.push((i as u32, j as u32));
                }
            }
        }
        return (all, true);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut seen = HashSet::with_capacity(count * 2);
    let mut pairs = Vec::with_capacity(count);
    let mut attempts: u64 = 0;
    while pairs.len() < count {
        attempts += 1;
        if attempts > 1000 * count as u64 + 1_000_000 {
            break; // fewer distinct impostor pairs than requested
        }
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        if i == j || templates[i].class == templates[j].class {
            continue;
        }
        if seen.insert((i, j)) {
            pairs.push((i as u32, j as u32));
        }
    }
    (pairs, false)
}

/// Class-balanced bit frequencies over the given classes (all classes if `None`).
pub fn prior(templates: &[Template], classes: Option<&HashSet<String>>) -> (Vec<f64>, usize) {
    let dims = templates[0].data.len();
    let mut per: BTreeMap<&str, (Vec<f64>, usize)> = BTreeMap::new();
    for t in templates {
        if let Some(c) = classes {
            if !c.contains(&t.class) {
                continue;
            }
        }
        let e = per.entry(&t.class).or_insert_with(|| (vec![0.0; dims], 0));
        for (s, &b) in e.0.iter_mut().zip(&t.data) {
            *s += b as f64;
        }
        e.1 += 1;
    }
    let mut p = vec![0.0; dims];
    for (sum, count) in per.values() {
        for (pj, s) in p.iter_mut().zip(sum) {
            *pj += s / *count as f64;
        }
    }
    let k = per.len().max(1) as f64;
    p.iter_mut().for_each(|v| *v /= k);
    (p, per.len())
}

pub fn independent_bit(p: &[f64], set: &[usize]) -> f64 {
    set.iter()
        .map(|&j| {
            let q = p[j].clamp(1e-6, 1.0 - 1e-6);
            -(q.max(1.0 - q)).log2()
        })
        .sum()
}

pub fn analyze(
    templates: &[Template],
    sets: &[Vec<usize>],
    pairs: usize,
    seed: u64,
    prior_classes: Option<&HashSet<String>>,
) -> SetReport {
    let (pairs_list, all) = impostor_pairs(templates, pairs, seed);
    let (p, used) = prior(templates, prior_classes);
    let classes: HashSet<&str> = templates.iter().map(|t| t.class.as_str()).collect();
    let stats: Vec<[f64; 8]> = sets
        .par_iter()
        .map(|set| {
            let bytes = (set.len() + 7) / 8;
            let mut packed = vec![0u8; bytes * templates.len()];
            for (t, chunk) in templates.iter().zip(packed.chunks_mut(bytes)) {
                pack(t, set, chunk);
            }
            let m = set.len() as f64;
            let (mut s, mut q, mut sa, mut qa, mut sb, mut qb) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut na, mut nb) = (0usize, 0usize);
            for (k, &(i, j)) in pairs_list.iter().enumerate() {
                let a = &packed[i as usize * bytes..(i as usize + 1) * bytes];
                let b = &packed[j as usize * bytes..(j as usize + 1) * bytes];
                let d: u32 = a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
                let f = d as f64 / m;
                s += f;
                q += f * f;
                if k % 2 == 0 {
                    sa += f;
                    qa += f * f;
                    na += 1;
                } else {
                    sb += f;
                    qb += f * f;
                    nb += 1;
                }
            }
            let (mean, var, dof, h) = daugman(s, q, pairs_list.len());
            let ha = daugman(sa, qa, na).3;
            let hb = daugman(sb, qb, nb).3;
            [set.len() as f64, mean, var, dof, h, ha, hb, independent_bit(&p, set)]
        })
        .collect();
    let col = |k: usize| stats.iter().map(|r| r[k]).collect::<Vec<f64>>();
    let daugman_col = col(4);
    let ind = col(7);
    let mut summary = BTreeMap::new();
    let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    summary.insert("daugman_min".into(), min(&daugman_col));
    summary.insert("daugman_mean".into(), avg(&daugman_col));
    summary.insert("independent_bit_min".into(), min(&ind));
    summary.insert("independent_bit_mean".into(), avg(&ind));
    summary.insert("rank_correlation".into(), spearman(&daugman_col, &ind));
    SetReport {
        templates: templates.len(),
        classes: classes.len(),
        prior_classes: used,
        impostor_pairs: pairs_list.len(),
        all_pairs: all,
        seed,
        size: stats.iter().map(|r| r[0] as usize).collect(),
        mean: col(1),
        variance: col(2),
        dof: col(3),
        daugman: daugman_col,
        daugman_half_a: col(5),
        daugman_half_b: col(6),
        independent_bit: ind,
        prior: p,
        summary,
    }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation (NaN if fewer than three sets or a constant column).
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 3 {
        return f64::NAN;
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        c += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    if va == 0.0 || vb == 0.0 {
        f64::NAN
    } else {
        c / (va * vb).sqrt()
    }
}

pub fn run(
    input: &str,
    templates_dir: &str,
    output: &str,
    pairs: usize,
    seed: u64,
    skip_classes: &[String],
    prior_classes_file: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let sets = RandomIndicesGenerator::load(input)?.0;
    let mut templates = TemplateReader::read_templates(templates_dir)?;
    templates.retain(|t| !skip_classes.contains(&t.class));
    templates.sort_by(|a, b| a.class.cmp(&b.class));
    if templates.is_empty() {
        return Err("no templates".into());
    }
    let dims = templates[0].data.len();
    if templates.iter().any(|t| t.data.len() != dims) {
        return Err("templates differ in length".into());
    }
    if sets.iter().flatten().any(|&j| j >= dims) {
        return Err("set coordinate out of range".into());
    }
    let prior_classes: Option<HashSet<String>> = match prior_classes_file {
        Some(path) => Some(
            std::fs::read_to_string(path)?
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
        ),
        None => None,
    };
    eprintln!("analyze-sets: {} sets, {} templates, pairs {}", sets.len(), templates.len(), if pairs == 0 { "all".to_string() } else { pairs.to_string() });
    let report = analyze(&templates, &sets, pairs, seed, prior_classes.as_ref());
    serde_json::to_writer(BufWriter::new(File::create(output)?), &report)?;
    for (k, v) in &report.summary {
        println!("{}: {}", k, v);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::AnalysisTool;

    fn synthetic() -> Vec<Template> {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut out = Vec::new();
        for c in 0..12 {
            let base: Vec<u8> = (0..64).map(|_| rng.gen_range(0..2)).collect();
            for _ in 0..4 {
                let data = base.iter().map(|&b| if rng.gen_bool(0.1) { 1 - b } else { b }).collect();
                out.push(Template { class: format!("c{c}"), data });
            }
        }
        out
    }

    #[test]
    fn all_pairs_match_existing_analysis() {
        let t = synthetic();
        // The existing function prints progress every tenth set, so it needs at least ten.
        let sets: Vec<Vec<usize>> = (0..12).map(|k| (k..k + 5 + 4 * k).collect()).collect();
        let r = analyze(&t, &sets, 0, 1, None);
        let (_, _, _, store) = AnalysisTool::calculate_class_based_fractional_hamming_mean_and_entropy(&t, &sets);
        for (a, b) in r.daugman.iter().zip(store.lock().unwrap().iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        assert!(r.all_pairs);
    }

    #[test]
    fn independent_bit_is_sum_of_coordinate_min_entropies() {
        let p = vec![0.5, 0.75, 0.1];
        let h = independent_bit(&p, &[0, 1, 2]);
        let expect = 1.0 - (0.75f64).log2() - (0.9f64).log2();
        assert!((h - expect).abs() < 1e-9);
    }

    #[test]
    fn sampled_pairs_are_distinct_impostors() {
        let t = synthetic();
        let (pairs, all) = impostor_pairs(&t, 500, 3);
        assert!(!all && pairs.len() == 500);
        let set: HashSet<_> = pairs.iter().collect();
        assert_eq!(set.len(), 500);
        assert!(pairs.iter().all(|&(i, j)| i < j && t[i as usize].class != t[j as usize].class));
    }

    #[test]
    fn spearman_basic() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 6.0, 9.0]) - 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
    }
}
