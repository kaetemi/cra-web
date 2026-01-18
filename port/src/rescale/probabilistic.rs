//! Probabilistic scatter-based rescaling (experimental)
//!
//! This module implements rescaling using probabilistic scattering from kernel weights.
//! Instead of deterministic weighted blending, each source pixel probabilistically
//! chooses where to "land" in the destination, preserving total energy.
//!
//! For the two-sample signed approach:
//! 1. Each source pixel samples a destination from positive kernel weights
//! 2. Each source pixel also samples a destination for its negative contribution
//! 3. Positive contributions add, negative contributions subtract
//! 4. Normalize by total weight that landed at each destination
//!
//! This preserves the sharpening effect of negative Lanczos/Sinc lobes while
//! producing fundamentally different artifacts (noise instead of ringing).

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::eval_kernel;

/// Simple deterministic RNG based on pixel position
/// Uses a variant of the PCG algorithm for good statistical properties
#[derive(Clone, Copy)]
struct PositionRng {
    state: u64,
}

impl PositionRng {
    /// Create RNG seeded by pixel position
    #[inline]
    fn new(x: usize, y: usize, seed: u64) -> Self {
        // Mix position into seed using a simple hash
        let pos_hash = (x as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (y as u64).wrapping_mul(0x6C8E9CF570932BD5);
        Self {
            state: seed.wrapping_add(pos_hash).wrapping_mul(0x5851F42D4C957F2D),
        }
    }

    /// Generate next random u32
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // PCG-XSH-RR
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Generate random f32 in [0, 1)
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }
}

/// Sample an index from weights treated as unnormalized probabilities
/// Returns None if all weights are zero
#[inline]
fn sample_from_weights(weights: &[f32], total: f32, rng: &mut PositionRng) -> Option<usize> {
    if total.abs() < 1e-10 || weights.is_empty() {
        return None;
    }

    let r = rng.next_f32() * total;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if r < cumulative {
            return Some(i);
        }
    }
    // Fallback for floating point edge case
    Some(weights.len() - 1)
}

/// Precomputed scatter weights for probabilistic sampling
/// Maps one source pixel to its potential destination pixels
#[derive(Clone)]
pub struct ProbScatterWeights {
    /// First destination index this source can affect
    pub start_idx: usize,
    /// Positive weights (destinations where kernel > 0)
    pub pos_weights: Vec<f32>,
    /// Negative weights as positive values (destinations where kernel < 0)
    pub neg_weights: Vec<f32>,
    /// Sum of positive weights
    pub pos_sum: f32,
    /// Sum of negative weights (as positive)
    pub neg_sum: f32,
    /// Suggested number of samples (capped for full-extent kernels like Sinc)
    pub suggested_samples: usize,
}

/// Precompute scatter weights for 1D probabilistic resampling
/// Returns weights for each SOURCE position (maps src -> affected dst range)
pub fn precompute_prob_scatter_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
    radius: i32,
    method: RescaleMethod,
) -> Vec<ProbScatterWeights> {
    let mut all_weights = Vec::with_capacity(src_len);
    let kernel_method = method.kernel_method();

    // For full-extent kernels (like Sinc), use Lanczos3-equivalent sample count
    // Lanczos3 has radius 3, so footprint is roughly 2 * 3 * filter_scale / scale
    let lanczos3_samples = ((2.0 * 3.0 * filter_scale) / scale).ceil() as usize;

    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;
    let inv_scale = 1.0 / scale;

    for src_i in 0..src_len {
        // Find which destination pixels this source pixel affects
        let dst_center = (src_i as f32 - offset + 0.5) * inv_scale - 0.5;
        let dst_radius = (radius as f32 * filter_scale) * inv_scale;

        let start = ((dst_center - dst_radius).floor() as i32).max(0) as usize;
        let end = ((dst_center + dst_radius).ceil() as i32).min(dst_len as i32 - 1) as usize;

        let mut pos_weights = Vec::with_capacity(end.saturating_sub(start) + 1);
        let mut neg_weights = Vec::with_capacity(end.saturating_sub(start) + 1);
        let mut pos_sum = 0.0f32;
        let mut neg_sum = 0.0f32;

        for dst_i in start..=end {
            // Calculate where this destination samples in source space
            let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
            let d = (src_i as f32 - src_pos) / filter_scale;
            let weight = eval_kernel(kernel_method, d);

            if weight >= 0.0 {
                pos_weights.push(weight);
                neg_weights.push(0.0);
                pos_sum += weight;
            } else {
                pos_weights.push(0.0);
                neg_weights.push(-weight);
                neg_sum += -weight;
            }
        }

        // Use actual footprint for normal kernels, Lanczos3-equivalent for full-extent
        let footprint = pos_weights.len().max(1);
        let suggested_samples = if kernel_method.is_full_extent() {
            lanczos3_samples.max(1)
        } else {
            footprint
        };

        all_weights.push(ProbScatterWeights {
            start_idx: start,
            pos_weights,
            neg_weights,
            pos_sum,
            neg_sum,
            suggested_samples,
        });
    }

    all_weights
}

/// Scatter-based probabilistic 1D resample
/// Each source pixel scatters multiple times based on its footprint size
#[inline]
fn prob_scatter_row(
    src: &[Pixel4],
    scatter_weights: &[ProbScatterWeights],
    dst_len: usize,
    y: usize,
    seed: u64,
) -> Vec<Pixel4> {
    // Accumulators
    let mut acc = vec![Pixel4::default(); dst_len];
    let mut weight_acc = vec![0.0f32; dst_len];

    // Scatter phase: each source pixel scatters based on suggested sample count
    for (src_i, sw) in scatter_weights.iter().enumerate() {
        let pixel = src[src_i];
        let num_samples = sw.suggested_samples;

        for sample_idx in 0..num_samples {
            // Re-seed RNG for each sample to maintain determinism
            let mut sample_rng = PositionRng::new(src_i * 1000 + sample_idx, y, seed);

            // Scatter positive contribution
            if let Some(rel_idx) = sample_from_weights(&sw.pos_weights, sw.pos_sum, &mut sample_rng) {
                let dst_i = sw.start_idx + rel_idx;
                if dst_i < dst_len {
                    // Each sample contributes 1/num_samples of the source's weight
                    let sample_weight = sw.pos_sum / num_samples as f32;
                    acc[dst_i] = acc[dst_i] + pixel * sample_weight;
                    weight_acc[dst_i] += sample_weight;
                }
            }

            // Scatter negative contribution (subtractive)
            if sw.neg_sum > 1e-10 {
                if let Some(rel_idx) = sample_from_weights(&sw.neg_weights, sw.neg_sum, &mut sample_rng) {
                    let dst_i = sw.start_idx + rel_idx;
                    if dst_i < dst_len {
                        let sample_weight = sw.neg_sum / num_samples as f32;
                        acc[dst_i] = acc[dst_i] - pixel * sample_weight;
                        // Negative contribution subtracts from normalizer too,
                        // matching the signed kernel weight sum
                        weight_acc[dst_i] -= sample_weight;
                    }
                }
            }
        }
    }

    // Normalize
    let mut dst = vec![Pixel4::default(); dst_len];
    for i in 0..dst_len {
        if weight_acc[i].abs() > 1e-10 {
            dst[i] = acc[i] * (1.0 / weight_acc[i]);
        }
    }

    dst
}

/// Probabilistic scatter-based separable kernel rescale (2-pass)
pub fn rescale_prob_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    seed: u64,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);

    let kernel_method = method.kernel_method();

    let (radius_x, radius_y) = if kernel_method.is_full_extent() {
        (src_width as i32, src_height as i32)
    } else {
        let base_radius = kernel_method.base_radius();
        (
            (base_radius * filter_scale_x).ceil() as i32,
            (base_radius * filter_scale_y).ceil() as i32,
        )
    };

    // Precompute scatter weights (source -> destination mapping)
    let h_weights = precompute_prob_scatter_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method);
    let v_weights = precompute_prob_scatter_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method);

    // Pass 1: Horizontal scatter
    let h_seed = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = prob_scatter_row(src_row, &h_weights, dst_width, y, h_seed);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Vertical scatter (process columns)
    let v_seed = seed.wrapping_mul(0x6C8E9CF570932BD5);
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for x in 0..dst_width {
        // Extract column
        let col: Vec<Pixel4> = (0..src_height).map(|y| temp[y * dst_width + x]).collect();

        // Scatter this column
        let resampled = prob_scatter_row(&col, &v_weights, dst_height, x, v_seed);

        // Write back
        for (y, pixel) in resampled.into_iter().enumerate() {
            dst[y * dst_width + x] = pixel;
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (x + 1) as f32 / dst_width as f32 * 0.5);
        }
    }

    dst
}

/// Alpha-aware probabilistic scatter rescale
pub fn rescale_prob_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    seed: u64,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);

    let kernel_method = method.kernel_method();

    let (radius_x, radius_y) = if kernel_method.is_full_extent() {
        (src_width as i32, src_height as i32)
    } else {
        let base_radius = kernel_method.base_radius();
        (
            (base_radius * filter_scale_x).ceil() as i32,
            (base_radius * filter_scale_y).ceil() as i32,
        )
    };

    let h_weights = precompute_prob_scatter_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method);
    let v_weights = precompute_prob_scatter_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method);

    // Pass 1: Alpha-aware horizontal scatter
    let h_seed = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = prob_scatter_row_alpha(src_row, &h_weights, dst_width, y, h_seed);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Alpha-aware vertical scatter
    let v_seed = seed.wrapping_mul(0x6C8E9CF570932BD5);
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for x in 0..dst_width {
        let col: Vec<Pixel4> = (0..src_height).map(|y| temp[y * dst_width + x]).collect();
        let resampled = prob_scatter_row_alpha(&col, &v_weights, dst_height, x, v_seed);

        for (y, pixel) in resampled.into_iter().enumerate() {
            dst[y * dst_width + x] = pixel;
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (x + 1) as f32 / dst_width as f32 * 0.5);
        }
    }

    dst
}

/// Alpha-aware scatter row - weights scatter probability by alpha
fn prob_scatter_row_alpha(
    src: &[Pixel4],
    scatter_weights: &[ProbScatterWeights],
    dst_len: usize,
    y: usize,
    seed: u64,
) -> Vec<Pixel4> {
    // Accumulators
    let mut rgb_pos_acc = vec![[0.0f32; 3]; dst_len];
    let mut rgb_neg_acc = vec![[0.0f32; 3]; dst_len];
    let mut alpha_acc = vec![0.0f32; dst_len];
    let mut pos_weight_acc = vec![0.0f32; dst_len];
    let mut neg_weight_acc = vec![0.0f32; dst_len];
    let mut alpha_weight_acc = vec![0.0f32; dst_len];

    for (src_i, sw) in scatter_weights.iter().enumerate() {
        let pixel = src[src_i];
        let alpha = pixel.a();

        // Weight scatter probabilities by alpha for RGB
        let alpha_pos_weights: Vec<f32> = sw.pos_weights.iter().map(|&w| w * alpha).collect();
        let alpha_pos_sum: f32 = alpha_pos_weights.iter().sum();
        let alpha_neg_weights: Vec<f32> = sw.neg_weights.iter().map(|&w| w * alpha).collect();
        let alpha_neg_sum: f32 = alpha_neg_weights.iter().sum();

        let num_samples = sw.suggested_samples;

        for sample_idx in 0..num_samples {
            let mut sample_rng = PositionRng::new(src_i * 1000 + sample_idx, y, seed);

            // Scatter positive RGB (weighted by alpha)
            let (pos_weights_to_use, pos_sum_to_use) = if alpha_pos_sum.abs() > 1e-10 {
                (&alpha_pos_weights as &[f32], alpha_pos_sum)
            } else {
                (&sw.pos_weights as &[f32], sw.pos_sum)
            };

            if let Some(rel_idx) = sample_from_weights(pos_weights_to_use, pos_sum_to_use, &mut sample_rng) {
                let dst_i = sw.start_idx + rel_idx;
                if dst_i < dst_len {
                    let sample_weight = sw.pos_sum * alpha / num_samples as f32;
                    rgb_pos_acc[dst_i][0] += pixel.r() * sample_weight;
                    rgb_pos_acc[dst_i][1] += pixel.g() * sample_weight;
                    rgb_pos_acc[dst_i][2] += pixel.b() * sample_weight;
                    pos_weight_acc[dst_i] += sample_weight;
                }
            }

            // Scatter negative RGB (weighted by alpha)
            if sw.neg_sum > 1e-10 {
                let (neg_weights_to_use, neg_sum_to_use) = if alpha_neg_sum.abs() > 1e-10 {
                    (&alpha_neg_weights as &[f32], alpha_neg_sum)
                } else {
                    (&sw.neg_weights as &[f32], sw.neg_sum)
                };

                if let Some(rel_idx) = sample_from_weights(neg_weights_to_use, neg_sum_to_use, &mut sample_rng) {
                    let dst_i = sw.start_idx + rel_idx;
                    if dst_i < dst_len {
                        let sample_weight = sw.neg_sum * alpha / num_samples as f32;
                        rgb_neg_acc[dst_i][0] += pixel.r() * sample_weight;
                        rgb_neg_acc[dst_i][1] += pixel.g() * sample_weight;
                        rgb_neg_acc[dst_i][2] += pixel.b() * sample_weight;
                        neg_weight_acc[dst_i] += sample_weight;
                    }
                }
            }

            // Alpha scatters based on positive weights only
            if let Some(rel_idx) = sample_from_weights(&sw.pos_weights, sw.pos_sum, &mut sample_rng) {
                let dst_i = sw.start_idx + rel_idx;
                if dst_i < dst_len {
                    let sample_weight = sw.pos_sum / num_samples as f32;
                    alpha_acc[dst_i] += alpha * sample_weight;
                    alpha_weight_acc[dst_i] += sample_weight;
                }
            }
        }
    }

    // Normalize and build output
    let mut dst = vec![Pixel4::default(); dst_len];
    for i in 0..dst_len {
        // Subtract negative weights to match signed kernel weight sum
        let rgb_total = pos_weight_acc[i] - neg_weight_acc[i];
        let (r, g, b) = if rgb_total > 1e-10 {
            let inv = 1.0 / rgb_total;
            (
                (rgb_pos_acc[i][0] - rgb_neg_acc[i][0]) * inv,
                (rgb_pos_acc[i][1] - rgb_neg_acc[i][1]) * inv,
                (rgb_pos_acc[i][2] - rgb_neg_acc[i][2]) * inv,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let a = if alpha_weight_acc[i].abs() > 1e-10 {
            (alpha_acc[i] / alpha_weight_acc[i]).clamp(0.0, 1.0)
        } else {
            0.0
        };

        dst[i] = Pixel4::new(r, g, b, a);
    }

    dst
}
