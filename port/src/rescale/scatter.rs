//! Scatter-based separable rescaling (experimental)
//!
//! This module implements rescaling using the scatter approach:
//! instead of asking "what source pixels contribute to this destination pixel?" (gather),
//! we ask "what destination pixels does this source pixel affect?" (scatter).
//!
//! The loop structure is inverted:
//! - Gather: for each dst { for each affecting src { dst += src * weight } }
//! - Scatter: for each src { for each affected dst { dst += src * weight } }; normalize
//!
//! For deterministic weighted blending, this produces identical results to gather.
//! The difference becomes meaningful for probabilistic variants.

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::{eval_kernel, eval_kernel_mixed, select_kernel_for_source};

/// Precomputed scatter weights for a single source position
/// Maps one source pixel to the destination pixels it affects
#[derive(Clone)]
pub struct ScatterWeights {
    /// First destination index this source affects
    pub start_idx: usize,
    /// Weights for each affected destination pixel
    /// NOT normalized - normalization happens after accumulation
    pub weights: Vec<f32>,
}

/// Precompute scatter weights for 1D resampling
/// Returns weights for each SOURCE position (inverse of gather weights)
pub fn precompute_scatter_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
    radius: i32,
    method: RescaleMethod,
) -> Vec<ScatterWeights> {
    let mut all_weights = Vec::with_capacity(src_len);

    // Use the actual kernel method (not the scatter variant)
    let kernel_method = method.kernel_method();

    // Center offset for uniform scaling
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    // Inverse scale: how destination positions map to source positions
    let inv_scale = 1.0 / scale;

    for src_i in 0..src_len {
        // Find which destination pixels this source pixel affects
        // A source at src_i affects destination pixels where:
        //   |src_i - src_pos(dst_i)| < radius * filter_scale
        // where src_pos(dst_i) = (dst_i + 0.5) * scale - 0.5 + offset
        //
        // Solving for dst_i:
        //   dst_i = (src_i - offset + 0.5) / scale - 0.5
        // with a range of ± radius * filter_scale / scale

        let dst_center = (src_i as f32 - offset + 0.5) * inv_scale - 0.5;
        let dst_radius = (radius as f32 * filter_scale) * inv_scale;

        let start = ((dst_center - dst_radius).floor() as i32).max(0) as usize;
        let end = ((dst_center + dst_radius).ceil() as i32).min(dst_len as i32 - 1) as usize;

        let mut weights = Vec::with_capacity(end.saturating_sub(start) + 1);

        for dst_i in start..=end {
            // Calculate where this destination samples in source space
            let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
            let d = (src_i as f32 - src_pos) / filter_scale;
            let weight = eval_kernel(kernel_method, d);
            weights.push(weight);
        }

        all_weights.push(ScatterWeights {
            start_idx: start,
            weights,
        });
    }

    all_weights
}

/// Scatter-based 1D resample for Pixel4 row
/// Scatters each source pixel to its affected destinations, then normalizes
#[inline]
fn scatter_row_pixel4(
    src: &[Pixel4],
    scatter_weights: &[ScatterWeights],
    dst_len: usize,
) -> Vec<Pixel4> {
    // Accumulator for destination pixels
    let mut dst = vec![Pixel4::default(); dst_len];
    // Weight accumulator for normalization
    let mut weight_sums = vec![0.0f32; dst_len];

    // Scatter phase: each source pixel contributes to destination pixels
    for (src_i, sw) in scatter_weights.iter().enumerate() {
        let pixel = src[src_i];
        for (i, &weight) in sw.weights.iter().enumerate() {
            let dst_i = sw.start_idx + i;
            if dst_i < dst_len {
                dst[dst_i] = dst[dst_i] + pixel * weight;
                weight_sums[dst_i] += weight;
            }
        }
    }

    // Normalize phase
    for (i, sum) in weight_sums.iter().enumerate() {
        if sum.abs() > 1e-8 {
            dst[i] = dst[i] * (1.0 / sum);
        }
    }

    dst
}

/// Scatter-based separable kernel rescale (2-pass)
pub fn rescale_scatter_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);

    let kernel_method = method.kernel_method();

    // For Sinc, use full image extent; otherwise use kernel's base radius
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
    let h_weights = precompute_scatter_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method);
    let v_weights = precompute_scatter_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method);

    // Pass 1: Horizontal scatter resample (src_width -> dst_width)
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = scatter_row_pixel4(src_row, &h_weights, dst_width);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Vertical scatter resample
    // We need to process columns, but for cache efficiency we'll process by output rows
    // and do the scatter accumulation differently
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];
    let mut weight_sums = vec![0.0f32; dst_width * dst_height];

    // Scatter from each source row to affected destination rows
    for src_y in 0..src_height {
        let sw = &v_weights[src_y];
        let src_row_start = src_y * dst_width;

        for (i, &weight) in sw.weights.iter().enumerate() {
            let dst_y = sw.start_idx + i;
            if dst_y < dst_height {
                let dst_row_start = dst_y * dst_width;
                for x in 0..dst_width {
                    dst[dst_row_start + x] = dst[dst_row_start + x] + temp[src_row_start + x] * weight;
                    weight_sums[dst_row_start + x] += weight;
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Normalize
    for (i, sum) in weight_sums.iter().enumerate() {
        if sum.abs() > 1e-8 {
            dst[i] = dst[i] * (1.0 / sum);
        }
    }

    dst
}

/// Alpha-aware scatter-based separable kernel rescale (2-pass)
pub fn rescale_scatter_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
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

    let h_weights = precompute_scatter_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method);
    let v_weights = precompute_scatter_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method);

    // Pass 1: Alpha-aware horizontal scatter
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = scatter_row_alpha_pixel4(src_row, &h_weights, dst_width);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Alpha-aware vertical scatter
    let mut dst_rgb = vec![[0.0f32; 3]; dst_width * dst_height];
    let mut dst_alpha = vec![0.0f32; dst_width * dst_height];
    let mut alpha_weight_sums = vec![0.0f32; dst_width * dst_height];
    let mut weight_sums = vec![0.0f32; dst_width * dst_height];
    // For fallback when all alpha is zero
    let mut rgb_unweighted = vec![[0.0f32; 3]; dst_width * dst_height];

    for src_y in 0..src_height {
        let sw = &v_weights[src_y];
        let src_row_start = src_y * dst_width;

        for (i, &weight) in sw.weights.iter().enumerate() {
            let dst_y = sw.start_idx + i;
            if dst_y < dst_height {
                let dst_row_start = dst_y * dst_width;
                for x in 0..dst_width {
                    let p = temp[src_row_start + x];
                    let alpha = p.a();
                    let aw = weight * alpha;
                    let idx = dst_row_start + x;

                    dst_rgb[idx][0] += aw * p.r();
                    dst_rgb[idx][1] += aw * p.g();
                    dst_rgb[idx][2] += aw * p.b();
                    dst_alpha[idx] += weight * alpha;
                    alpha_weight_sums[idx] += aw;
                    weight_sums[idx] += weight;

                    rgb_unweighted[idx][0] += weight * p.r();
                    rgb_unweighted[idx][1] += weight * p.g();
                    rgb_unweighted[idx][2] += weight * p.b();
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Normalize and build final output
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];
    for i in 0..dst.len() {
        let out_a = if weight_sums[i].abs() > 1e-8 {
            dst_alpha[i] / weight_sums[i]
        } else {
            0.0
        };

        let (out_r, out_g, out_b) = if alpha_weight_sums[i].abs() > 1e-8 {
            let inv_aw = 1.0 / alpha_weight_sums[i];
            (dst_rgb[i][0] * inv_aw, dst_rgb[i][1] * inv_aw, dst_rgb[i][2] * inv_aw)
        } else if weight_sums[i].abs() > 1e-8 {
            let inv_w = 1.0 / weight_sums[i];
            (rgb_unweighted[i][0] * inv_w, rgb_unweighted[i][1] * inv_w, rgb_unweighted[i][2] * inv_w)
        } else {
            (0.0, 0.0, 0.0)
        };

        dst[i] = Pixel4::new(out_r, out_g, out_b, out_a);
    }

    dst
}

/// Alpha-aware scatter-based 1D resample
fn scatter_row_alpha_pixel4(
    src: &[Pixel4],
    scatter_weights: &[ScatterWeights],
    dst_len: usize,
) -> Vec<Pixel4> {
    // Accumulators
    let mut dst_rgb = vec![[0.0f32; 3]; dst_len];
    let mut dst_alpha = vec![0.0f32; dst_len];
    let mut alpha_weight_sums = vec![0.0f32; dst_len];
    let mut weight_sums = vec![0.0f32; dst_len];
    let mut rgb_unweighted = vec![[0.0f32; 3]; dst_len];

    // Scatter phase
    for (src_i, sw) in scatter_weights.iter().enumerate() {
        let p = src[src_i];
        let alpha = p.a();

        for (i, &weight) in sw.weights.iter().enumerate() {
            let dst_i = sw.start_idx + i;
            if dst_i < dst_len {
                let aw = weight * alpha;
                dst_rgb[dst_i][0] += aw * p.r();
                dst_rgb[dst_i][1] += aw * p.g();
                dst_rgb[dst_i][2] += aw * p.b();
                dst_alpha[dst_i] += weight * alpha;
                alpha_weight_sums[dst_i] += aw;
                weight_sums[dst_i] += weight;

                rgb_unweighted[dst_i][0] += weight * p.r();
                rgb_unweighted[dst_i][1] += weight * p.g();
                rgb_unweighted[dst_i][2] += weight * p.b();
            }
        }
    }

    // Normalize phase
    let mut dst = vec![Pixel4::default(); dst_len];
    for i in 0..dst_len {
        let out_a = if weight_sums[i].abs() > 1e-8 {
            dst_alpha[i] / weight_sums[i]
        } else {
            0.0
        };

        let (out_r, out_g, out_b) = if alpha_weight_sums[i].abs() > 1e-8 {
            let inv_aw = 1.0 / alpha_weight_sums[i];
            (dst_rgb[i][0] * inv_aw, dst_rgb[i][1] * inv_aw, dst_rgb[i][2] * inv_aw)
        } else if weight_sums[i].abs() > 1e-8 {
            let inv_w = 1.0 / weight_sums[i];
            (rgb_unweighted[i][0] * inv_w, rgb_unweighted[i][1] * inv_w, rgb_unweighted[i][2] * inv_w)
        } else {
            (0.0, 0.0, 0.0)
        };

        dst[i] = Pixel4::new(out_r, out_g, out_b, out_a);
    }

    dst
}

// =============================================================================
// Mixed kernel scatter implementation (Lanczos2/Lanczos3 per-source selection)
// =============================================================================

/// Mixed scatter-based separable kernel rescale (2-pass)
/// Uses per-source-pixel kernel selection (Lanczos2 or Lanczos3) via Wang hash.
pub fn rescale_mixed_scatter_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);

    // Center offset for uniform scaling
    let mapped_src_len_x = dst_width as f32 * scale_x;
    let offset_x = (src_width as f32 - mapped_src_len_x) / 2.0;

    let mapped_src_len_y = dst_height as f32 * scale_y;
    let offset_y = (src_height as f32 - mapped_src_len_y) / 2.0;

    let inv_scale_x = 1.0 / scale_x;
    let inv_scale_y = 1.0 / scale_y;

    // Pass 1: Horizontal mixed scatter (src_width -> dst_width)
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    let mut h_weight_sums = vec![0.0f32; dst_width * src_height];

    for src_y in 0..src_height {
        let row_start = src_y * src_width;
        let temp_row_start = src_y * dst_width;

        for src_x in 0..src_width {
            let pixel = src[row_start + src_x];

            // Select kernel based on source pixel position
            let use_lanczos3 = select_kernel_for_source(src_x, src_y, seed);
            let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };

            // Find affected destination range
            let dst_center = (src_x as f32 - offset_x + 0.5) * inv_scale_x - 0.5;
            let dst_radius = (effective_radius * filter_scale_x) * inv_scale_x;

            let start = ((dst_center - dst_radius).floor() as i32).max(0) as usize;
            let end = ((dst_center + dst_radius).ceil() as i32).min(dst_width as i32 - 1) as usize;

            for dst_x in start..=end {
                let src_pos = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;
                let d = (src_x as f32 - src_pos) / filter_scale_x;
                let weight = eval_kernel_mixed(d, use_lanczos3);

                if weight.abs() > 1e-10 {
                    let idx = temp_row_start + dst_x;
                    temp[idx] = temp[idx] + pixel * weight;
                    h_weight_sums[idx] += weight;
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb((src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Normalize horizontal pass
    for i in 0..temp.len() {
        if h_weight_sums[i].abs() > 1e-8 {
            temp[i] = temp[i] * (1.0 / h_weight_sums[i]);
        }
    }

    // Pass 2: Vertical mixed scatter
    // For vertical pass, we use a different seed component to vary the pattern
    // The intermediate pixels are at (dst_x, src_y), so we use src_y for selection
    let v_seed = seed ^ 0x9E3779B9; // Golden ratio bits for variation

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];
    let mut v_weight_sums = vec![0.0f32; dst_width * dst_height];

    for src_y in 0..src_height {
        let temp_row_start = src_y * dst_width;

        // Select kernel for this row (using src_y with varied seed)
        // All pixels in this row use the same kernel for coherence
        let use_lanczos3 = select_kernel_for_source(0, src_y, v_seed);
        let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };

        // Find affected destination range
        let dst_center = (src_y as f32 - offset_y + 0.5) * inv_scale_y - 0.5;
        let dst_radius = (effective_radius * filter_scale_y) * inv_scale_y;

        let start = ((dst_center - dst_radius).floor() as i32).max(0) as usize;
        let end = ((dst_center + dst_radius).ceil() as i32).min(dst_height as i32 - 1) as usize;

        for dst_y in start..=end {
            let src_pos = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;
            let d = (src_y as f32 - src_pos) / filter_scale_y;
            let weight = eval_kernel_mixed(d, use_lanczos3);

            if weight.abs() > 1e-10 {
                let dst_row_start = dst_y * dst_width;
                for x in 0..dst_width {
                    dst[dst_row_start + x] = dst[dst_row_start + x] + temp[temp_row_start + x] * weight;
                    v_weight_sums[dst_row_start + x] += weight;
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Normalize vertical pass
    for i in 0..dst.len() {
        if v_weight_sums[i].abs() > 1e-8 {
            dst[i] = dst[i] * (1.0 / v_weight_sums[i]);
        }
    }

    dst
}

/// Alpha-aware mixed scatter-based separable kernel rescale (2-pass)
pub fn rescale_mixed_scatter_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);

    let mapped_src_len_x = dst_width as f32 * scale_x;
    let offset_x = (src_width as f32 - mapped_src_len_x) / 2.0;

    let mapped_src_len_y = dst_height as f32 * scale_y;
    let offset_y = (src_height as f32 - mapped_src_len_y) / 2.0;

    let inv_scale_x = 1.0 / scale_x;
    let inv_scale_y = 1.0 / scale_y;

    // Pass 1: Alpha-aware horizontal mixed scatter
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    let mut h_rgb = vec![[0.0f32; 3]; dst_width * src_height];
    let mut h_alpha = vec![0.0f32; dst_width * src_height];
    let mut h_alpha_weight_sums = vec![0.0f32; dst_width * src_height];
    let mut h_weight_sums = vec![0.0f32; dst_width * src_height];
    let mut h_rgb_unweighted = vec![[0.0f32; 3]; dst_width * src_height];

    for src_y in 0..src_height {
        let row_start = src_y * src_width;
        let temp_row_start = src_y * dst_width;

        for src_x in 0..src_width {
            let p = src[row_start + src_x];
            let alpha = p.a();

            let use_lanczos3 = select_kernel_for_source(src_x, src_y, seed);
            let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };

            let dst_center = (src_x as f32 - offset_x + 0.5) * inv_scale_x - 0.5;
            let dst_radius = (effective_radius * filter_scale_x) * inv_scale_x;

            let start = ((dst_center - dst_radius).floor() as i32).max(0) as usize;
            let end = ((dst_center + dst_radius).ceil() as i32).min(dst_width as i32 - 1) as usize;

            for dst_x in start..=end {
                let src_pos = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;
                let d = (src_x as f32 - src_pos) / filter_scale_x;
                let weight = eval_kernel_mixed(d, use_lanczos3);

                if weight.abs() > 1e-10 {
                    let idx = temp_row_start + dst_x;
                    let aw = weight * alpha;

                    h_rgb[idx][0] += aw * p.r();
                    h_rgb[idx][1] += aw * p.g();
                    h_rgb[idx][2] += aw * p.b();
                    h_alpha[idx] += weight * alpha;
                    h_alpha_weight_sums[idx] += aw;
                    h_weight_sums[idx] += weight;

                    h_rgb_unweighted[idx][0] += weight * p.r();
                    h_rgb_unweighted[idx][1] += weight * p.g();
                    h_rgb_unweighted[idx][2] += weight * p.b();
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb((src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Normalize horizontal pass into temp buffer
    for i in 0..temp.len() {
        let out_a = if h_weight_sums[i].abs() > 1e-8 {
            h_alpha[i] / h_weight_sums[i]
        } else {
            0.0
        };

        let (out_r, out_g, out_b) = if h_alpha_weight_sums[i].abs() > 1e-8 {
            let inv_aw = 1.0 / h_alpha_weight_sums[i];
            (h_rgb[i][0] * inv_aw, h_rgb[i][1] * inv_aw, h_rgb[i][2] * inv_aw)
        } else if h_weight_sums[i].abs() > 1e-8 {
            let inv_w = 1.0 / h_weight_sums[i];
            (h_rgb_unweighted[i][0] * inv_w, h_rgb_unweighted[i][1] * inv_w, h_rgb_unweighted[i][2] * inv_w)
        } else {
            (0.0, 0.0, 0.0)
        };

        temp[i] = Pixel4::new(out_r, out_g, out_b, out_a);
    }

    // Pass 2: Alpha-aware vertical mixed scatter
    let v_seed = seed ^ 0x9E3779B9;

    let mut dst_rgb = vec![[0.0f32; 3]; dst_width * dst_height];
    let mut dst_alpha = vec![0.0f32; dst_width * dst_height];
    let mut v_alpha_weight_sums = vec![0.0f32; dst_width * dst_height];
    let mut v_weight_sums = vec![0.0f32; dst_width * dst_height];
    let mut v_rgb_unweighted = vec![[0.0f32; 3]; dst_width * dst_height];

    for src_y in 0..src_height {
        let temp_row_start = src_y * dst_width;

        let use_lanczos3 = select_kernel_for_source(0, src_y, v_seed);
        let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };

        let dst_center = (src_y as f32 - offset_y + 0.5) * inv_scale_y - 0.5;
        let dst_radius = (effective_radius * filter_scale_y) * inv_scale_y;

        let start = ((dst_center - dst_radius).floor() as i32).max(0) as usize;
        let end = ((dst_center + dst_radius).ceil() as i32).min(dst_height as i32 - 1) as usize;

        for dst_y in start..=end {
            let src_pos = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;
            let d = (src_y as f32 - src_pos) / filter_scale_y;
            let weight = eval_kernel_mixed(d, use_lanczos3);

            if weight.abs() > 1e-10 {
                let dst_row_start = dst_y * dst_width;
                for x in 0..dst_width {
                    let p = temp[temp_row_start + x];
                    let alpha = p.a();
                    let aw = weight * alpha;
                    let idx = dst_row_start + x;

                    dst_rgb[idx][0] += aw * p.r();
                    dst_rgb[idx][1] += aw * p.g();
                    dst_rgb[idx][2] += aw * p.b();
                    dst_alpha[idx] += weight * alpha;
                    v_alpha_weight_sums[idx] += aw;
                    v_weight_sums[idx] += weight;

                    v_rgb_unweighted[idx][0] += weight * p.r();
                    v_rgb_unweighted[idx][1] += weight * p.g();
                    v_rgb_unweighted[idx][2] += weight * p.b();
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Final normalization
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];
    for i in 0..dst.len() {
        let out_a = if v_weight_sums[i].abs() > 1e-8 {
            dst_alpha[i] / v_weight_sums[i]
        } else {
            0.0
        };

        let (out_r, out_g, out_b) = if v_alpha_weight_sums[i].abs() > 1e-8 {
            let inv_aw = 1.0 / v_alpha_weight_sums[i];
            (dst_rgb[i][0] * inv_aw, dst_rgb[i][1] * inv_aw, dst_rgb[i][2] * inv_aw)
        } else if v_weight_sums[i].abs() > 1e-8 {
            let inv_w = 1.0 / v_weight_sums[i];
            (v_rgb_unweighted[i][0] * inv_w, v_rgb_unweighted[i][1] * inv_w, v_rgb_unweighted[i][2] * inv_w)
        } else {
            (0.0, 0.0, 0.0)
        };

        dst[i] = Pixel4::new(out_r, out_g, out_b, out_a);
    }

    dst
}
