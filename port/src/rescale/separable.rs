//! Separable 2-pass kernel rescaling
//!
//! Implements high-quality rescaling using separable convolution kernels
//! (Mitchell, Catmull-Rom, Lanczos3, Sinc) with precomputed weights.

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::{KernelWeights, precompute_kernel_weights, eval_kernel_mixed, select_kernel_for_source};

/// Generic 1D resample for Pixel4 row using precomputed weights
/// Uses SIMD-friendly Pixel4 scalar multiply for better vectorization
#[inline]
fn resample_row_pixel4_precomputed(
    src: &[Pixel4],
    kernel_weights: &[KernelWeights],
) -> Vec<Pixel4> {
    let dst_len = kernel_weights.len();
    let mut dst = vec![Pixel4::default(); dst_len];

    for (dst_i, kw) in kernel_weights.iter().enumerate() {
        if kw.weights.is_empty() {
            dst[dst_i] = src[kw.fallback_idx];
        } else {
            let mut sum = Pixel4::default();
            for (i, &weight) in kw.weights.iter().enumerate() {
                // SIMD-friendly: scalar multiply broadcasts weight to all 4 channels
                sum = sum + src[kw.start_idx + i] * weight;
            }
            dst[dst_i] = sum;
        }
    }

    dst
}

/// Alpha-aware 1D resample for Pixel4 row using precomputed weights
/// RGB is weighted by alpha; alpha is interpolated normally.
/// Falls back to unweighted RGB if all contributing pixels are transparent.
#[inline]
fn resample_row_alpha_precomputed(
    src: &[Pixel4],
    kernel_weights: &[KernelWeights],
) -> Vec<Pixel4> {
    let dst_len = kernel_weights.len();
    let mut dst = vec![Pixel4::default(); dst_len];

    for (dst_i, kw) in kernel_weights.iter().enumerate() {
        if kw.weights.is_empty() {
            dst[dst_i] = src[kw.fallback_idx];
        } else {
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_a = 0.0f32;
            let mut sum_alpha_weight = 0.0f32;
            let mut sum_weight = 0.0f32;
            // For fallback: unweighted RGB sum
            let mut sum_r_unweighted = 0.0f32;
            let mut sum_g_unweighted = 0.0f32;
            let mut sum_b_unweighted = 0.0f32;

            for (i, &weight) in kw.weights.iter().enumerate() {
                let p = src[kw.start_idx + i];
                let alpha = p.a();
                let aw = weight * alpha;

                sum_r += aw * p.r();
                sum_g += aw * p.g();
                sum_b += aw * p.b();
                sum_a += weight * alpha;
                sum_alpha_weight += aw;
                sum_weight += weight;

                sum_r_unweighted += weight * p.r();
                sum_g_unweighted += weight * p.g();
                sum_b_unweighted += weight * p.b();
            }

            // Normalize alpha (normal interpolation)
            let out_a = if sum_weight.abs() > 1e-8 { sum_a / sum_weight } else { 0.0 };

            // RGB: alpha-weighted or fallback to unweighted
            let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                let inv_aw = 1.0 / sum_alpha_weight;
                (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
            } else if sum_weight.abs() > 1e-8 {
                let inv_w = 1.0 / sum_weight;
                (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
            } else {
                (0.0, 0.0, 0.0)
            };

            dst[dst_i] = Pixel4::new(out_r, out_g, out_b, out_a);
        }
    }

    dst
}

/// Rescale Pixel4 array using separable kernel interpolation (2-pass)
/// Uses precomputed kernel weights for efficiency
/// Progress callback is optional - receives 0.0-1.0 after each row
pub fn rescale_kernel_pixels(
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

    // For Sinc, use full image extent; otherwise use kernel's base radius
    let (radius_x, radius_y) = if method.is_full_extent() {
        // Full extent: radius covers entire source dimension
        (src_width as i32, src_height as i32)
    } else {
        let base_radius = method.base_radius();
        (
            (base_radius * filter_scale_x).ceil() as i32,
            (base_radius * filter_scale_y).ceil() as i32,
        )
    };

    // Precompute weights for horizontal and vertical passes (reused across all rows/columns)
    let h_weights = precompute_kernel_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method);
    let v_weights = precompute_kernel_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method);

    // Pass 1: Horizontal resample each row (src_width -> dst_width)
    // Progress: 0% to 50%
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = resample_row_pixel4_precomputed(src_row, &h_weights);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Vertical resample - process by output row for better cache locality
    // Each output row reads from multiple input rows (determined by kernel weights)
    // Progress: 50% to 100%
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let kw = &v_weights[dst_y];
        let dst_row_start = dst_y * dst_width;

        if kw.weights.is_empty() {
            // Fallback: copy entire row
            let src_row_start = kw.fallback_idx * dst_width;
            dst[dst_row_start..dst_row_start + dst_width]
                .copy_from_slice(&temp[src_row_start..src_row_start + dst_width]);
        } else {
            // Convolve: for each output pixel in this row, accumulate from source rows
            for x in 0..dst_width {
                let mut sum = Pixel4::default();
                for (i, &weight) in kw.weights.iter().enumerate() {
                    let src_y = kw.start_idx + i;
                    sum = sum + temp[src_y * dst_width + x] * weight;
                }
                dst[dst_row_start + x] = sum;
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (dst_y + 1) as f32 / dst_height as f32 * 0.5);
        }
    }

    dst
}

/// Alpha-aware separable kernel rescale (2-pass)
/// RGB channels are weighted by alpha to prevent transparent pixel color bleeding.
pub fn rescale_kernel_alpha_pixels(
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

    // For Sinc, use full image extent; otherwise use kernel's base radius
    let (radius_x, radius_y) = if method.is_full_extent() {
        (src_width as i32, src_height as i32)
    } else {
        let base_radius = method.base_radius();
        (
            (base_radius * filter_scale_x).ceil() as i32,
            (base_radius * filter_scale_y).ceil() as i32,
        )
    };

    let h_weights = precompute_kernel_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method);
    let v_weights = precompute_kernel_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method);

    // Pass 1: Alpha-aware horizontal resample
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = resample_row_alpha_precomputed(src_row, &h_weights);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Alpha-aware vertical resample
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let kw = &v_weights[dst_y];
        let dst_row_start = dst_y * dst_width;

        if kw.weights.is_empty() {
            let src_row_start = kw.fallback_idx * dst_width;
            dst[dst_row_start..dst_row_start + dst_width]
                .copy_from_slice(&temp[src_row_start..src_row_start + dst_width]);
        } else {
            for x in 0..dst_width {
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut sum_a = 0.0f32;
                let mut sum_alpha_weight = 0.0f32;
                let mut sum_weight = 0.0f32;
                let mut sum_r_unweighted = 0.0f32;
                let mut sum_g_unweighted = 0.0f32;
                let mut sum_b_unweighted = 0.0f32;

                for (i, &weight) in kw.weights.iter().enumerate() {
                    let src_y = kw.start_idx + i;
                    let p = temp[src_y * dst_width + x];
                    let alpha = p.a();
                    let aw = weight * alpha;

                    sum_r += aw * p.r();
                    sum_g += aw * p.g();
                    sum_b += aw * p.b();
                    sum_a += weight * alpha;
                    sum_alpha_weight += aw;
                    sum_weight += weight;

                    sum_r_unweighted += weight * p.r();
                    sum_g_unweighted += weight * p.g();
                    sum_b_unweighted += weight * p.b();
                }

                let out_a = if sum_weight.abs() > 1e-8 { sum_a / sum_weight } else { 0.0 };

                let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                    let inv_aw = 1.0 / sum_alpha_weight;
                    (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
                } else if sum_weight.abs() > 1e-8 {
                    let inv_w = 1.0 / sum_weight;
                    (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
                } else {
                    (0.0, 0.0, 0.0)
                };

                dst[dst_row_start + x] = Pixel4::new(out_r, out_g, out_b, out_a);
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (dst_y + 1) as f32 / dst_height as f32 * 0.5);
        }
    }

    dst
}

// =============================================================================
// Mixed kernel gather implementation (Lanczos2/Lanczos3 per-source selection)
// =============================================================================

/// Mixed kernel gather-based separable rescale (2-pass)
/// Precomputes kernel selection for each source pixel and uses that during gathering.
/// This produces results identical to mixed scatter.
pub fn rescale_mixed_kernel_pixels(
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

    // Use Lanczos3 radius to cover both kernels
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    // Center offset
    let mapped_src_len_x = dst_width as f32 * scale_x;
    let offset_x = (src_width as f32 - mapped_src_len_x) / 2.0;

    let mapped_src_len_y = dst_height as f32 * scale_y;
    let offset_y = (src_height as f32 - mapped_src_len_y) / 2.0;

    // Pass 1: Mixed horizontal gather
    let mut temp = vec![Pixel4::default(); dst_width * src_height];

    for src_y in 0..src_height {
        let src_row = &src[src_y * src_width..(src_y + 1) * src_width];
        let temp_row_start = src_y * dst_width;

        // Precompute kernel selection for all source pixels in this row
        let kernel_choices: Vec<bool> = (0..src_width)
            .map(|src_x| select_kernel_for_source(src_x, src_y, seed))
            .collect();

        for dst_x in 0..dst_width {
            let src_pos = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;
            let center = src_pos.floor() as i32;

            let start = (center - radius_x).max(0) as usize;
            let end = ((center + radius_x) as usize).min(src_width - 1);

            let mut sum = Pixel4::default();
            let mut weight_sum = 0.0f32;

            for src_x in start..=end {
                let use_lanczos3 = kernel_choices[src_x];
                let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };
                let d = (src_pos - src_x as f32) / filter_scale_x;

                // Skip if outside this kernel's effective radius
                if d.abs() > effective_radius {
                    continue;
                }

                let weight = eval_kernel_mixed(d, use_lanczos3);
                sum = sum + src_row[src_x] * weight;
                weight_sum += weight;
            }

            // Normalize
            if weight_sum.abs() > 1e-8 {
                temp[temp_row_start + dst_x] = sum * (1.0 / weight_sum);
            } else {
                let fallback = src_pos.round().clamp(0.0, (src_width - 1) as f32) as usize;
                temp[temp_row_start + dst_x] = src_row[fallback];
            }
        }

        if let Some(ref mut cb) = progress {
            cb((src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Mixed vertical gather
    // Use different seed for vertical to vary pattern
    let v_seed = seed ^ 0x9E3779B9;

    // Precompute kernel selection for all source rows
    let v_kernel_choices: Vec<bool> = (0..src_height)
        .map(|src_y| select_kernel_for_source(0, src_y, v_seed))
        .collect();

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let src_pos = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;
        let center = src_pos.floor() as i32;

        let start = (center - radius_y).max(0) as usize;
        let end = ((center + radius_y) as usize).min(src_height - 1);

        let dst_row_start = dst_y * dst_width;

        for x in 0..dst_width {
            let mut sum = Pixel4::default();
            let mut weight_sum = 0.0f32;

            for src_y in start..=end {
                let use_lanczos3 = v_kernel_choices[src_y];
                let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };
                let d = (src_pos - src_y as f32) / filter_scale_y;

                if d.abs() > effective_radius {
                    continue;
                }

                let weight = eval_kernel_mixed(d, use_lanczos3);
                sum = sum + temp[src_y * dst_width + x] * weight;
                weight_sum += weight;
            }

            if weight_sum.abs() > 1e-8 {
                dst[dst_row_start + x] = sum * (1.0 / weight_sum);
            } else {
                let fallback = src_pos.round().clamp(0.0, (src_height - 1) as f32) as usize;
                dst[dst_row_start + x] = temp[fallback * dst_width + x];
            }
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (dst_y + 1) as f32 / dst_height as f32 * 0.5);
        }
    }

    dst
}

/// Alpha-aware mixed kernel gather-based separable rescale (2-pass)
pub fn rescale_mixed_kernel_alpha_pixels(
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

    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    let mapped_src_len_x = dst_width as f32 * scale_x;
    let offset_x = (src_width as f32 - mapped_src_len_x) / 2.0;

    let mapped_src_len_y = dst_height as f32 * scale_y;
    let offset_y = (src_height as f32 - mapped_src_len_y) / 2.0;

    // Pass 1: Alpha-aware mixed horizontal gather
    let mut temp = vec![Pixel4::default(); dst_width * src_height];

    for src_y in 0..src_height {
        let src_row = &src[src_y * src_width..(src_y + 1) * src_width];
        let temp_row_start = src_y * dst_width;

        let kernel_choices: Vec<bool> = (0..src_width)
            .map(|src_x| select_kernel_for_source(src_x, src_y, seed))
            .collect();

        for dst_x in 0..dst_width {
            let src_pos = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;
            let center = src_pos.floor() as i32;

            let start = (center - radius_x).max(0) as usize;
            let end = ((center + radius_x) as usize).min(src_width - 1);

            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_a = 0.0f32;
            let mut sum_alpha_weight = 0.0f32;
            let mut sum_weight = 0.0f32;
            let mut sum_r_unweighted = 0.0f32;
            let mut sum_g_unweighted = 0.0f32;
            let mut sum_b_unweighted = 0.0f32;

            for src_x in start..=end {
                let use_lanczos3 = kernel_choices[src_x];
                let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };
                let d = (src_pos - src_x as f32) / filter_scale_x;

                if d.abs() > effective_radius {
                    continue;
                }

                let weight = eval_kernel_mixed(d, use_lanczos3);
                let p = src_row[src_x];
                let alpha = p.a();
                let aw = weight * alpha;

                sum_r += aw * p.r();
                sum_g += aw * p.g();
                sum_b += aw * p.b();
                sum_a += weight * alpha;
                sum_alpha_weight += aw;
                sum_weight += weight;

                sum_r_unweighted += weight * p.r();
                sum_g_unweighted += weight * p.g();
                sum_b_unweighted += weight * p.b();
            }

            let out_a = if sum_weight.abs() > 1e-8 { sum_a / sum_weight } else { 0.0 };

            let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                let inv_aw = 1.0 / sum_alpha_weight;
                (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
            } else if sum_weight.abs() > 1e-8 {
                let inv_w = 1.0 / sum_weight;
                (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
            } else {
                (0.0, 0.0, 0.0)
            };

            temp[temp_row_start + dst_x] = Pixel4::new(out_r, out_g, out_b, out_a);
        }

        if let Some(ref mut cb) = progress {
            cb((src_y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Alpha-aware mixed vertical gather
    let v_seed = seed ^ 0x9E3779B9;

    let v_kernel_choices: Vec<bool> = (0..src_height)
        .map(|src_y| select_kernel_for_source(0, src_y, v_seed))
        .collect();

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let src_pos = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;
        let center = src_pos.floor() as i32;

        let start = (center - radius_y).max(0) as usize;
        let end = ((center + radius_y) as usize).min(src_height - 1);

        let dst_row_start = dst_y * dst_width;

        for x in 0..dst_width {
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_a = 0.0f32;
            let mut sum_alpha_weight = 0.0f32;
            let mut sum_weight = 0.0f32;
            let mut sum_r_unweighted = 0.0f32;
            let mut sum_g_unweighted = 0.0f32;
            let mut sum_b_unweighted = 0.0f32;

            for src_y in start..=end {
                let use_lanczos3 = v_kernel_choices[src_y];
                let effective_radius = if use_lanczos3 { 3.0 } else { 2.0 };
                let d = (src_pos - src_y as f32) / filter_scale_y;

                if d.abs() > effective_radius {
                    continue;
                }

                let weight = eval_kernel_mixed(d, use_lanczos3);
                let p = temp[src_y * dst_width + x];
                let alpha = p.a();
                let aw = weight * alpha;

                sum_r += aw * p.r();
                sum_g += aw * p.g();
                sum_b += aw * p.b();
                sum_a += weight * alpha;
                sum_alpha_weight += aw;
                sum_weight += weight;

                sum_r_unweighted += weight * p.r();
                sum_g_unweighted += weight * p.g();
                sum_b_unweighted += weight * p.b();
            }

            let out_a = if sum_weight.abs() > 1e-8 { sum_a / sum_weight } else { 0.0 };

            let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                let inv_aw = 1.0 / sum_alpha_weight;
                (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
            } else if sum_weight.abs() > 1e-8 {
                let inv_w = 1.0 / sum_weight;
                (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
            } else {
                (0.0, 0.0, 0.0)
            };

            dst[dst_row_start + x] = Pixel4::new(out_r, out_g, out_b, out_a);
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (dst_y + 1) as f32 / dst_height as f32 * 0.5);
        }
    }

    dst
}
