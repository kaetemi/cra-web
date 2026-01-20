//! Separable 2-pass kernel rescaling
//!
//! Implements high-quality rescaling using separable convolution kernels
//! (Mitchell, Catmull-Rom, Lanczos3, Sinc) with precomputed weights.

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::{KernelWeights, precompute_kernel_weights, precompute_box_weights, precompute_tent_kernel_weights};

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
///
/// When `tent_mode` is true, uses sample-to-sample mapping for tent-space coordinates:
/// - Maps edge samples to edge samples (position 0→0, position N-1→M-1)
/// - Used for tent-volume supersampling where samples represent a bilinear surface
pub fn rescale_kernel_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    tent_mode: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Compute scales - in tent mode, use sample-to-sample mapping with uniform scaling support
    let (scale_x, scale_y) = if tent_mode {
        // Compute natural tent scales for each dimension
        let natural_tent_scale_x = if dst_width > 1 {
            (src_width - 1) as f32 / (dst_width - 1) as f32
        } else {
            1.0
        };
        let natural_tent_scale_y = if dst_height > 1 {
            (src_height - 1) as f32 / (dst_height - 1) as f32
        } else {
            1.0
        };

        // Apply uniform scaling if requested
        match scale_mode {
            ScaleMode::Independent => (natural_tent_scale_x, natural_tent_scale_y),
            ScaleMode::UniformWidth => (natural_tent_scale_x, natural_tent_scale_x),
            ScaleMode::UniformHeight => (natural_tent_scale_y, natural_tent_scale_y),
        }
    } else {
        calculate_scales(src_width, src_height, dst_width, dst_height, scale_mode)
    };

    // Filter scale controls kernel support
    // In tent mode: expand kernel when upscaling (scale < 1.0), multiplier goes from 1.0 to 2.0
    let multiplier_x = if tent_mode { (1.0 / scale_x).clamp(1.0, 2.0) } else { 1.0 };
    let multiplier_y = if tent_mode { (1.0 / scale_y).clamp(1.0, 2.0) } else { 1.0 };
    let filter_scale_x = scale_x.max(1.0) * multiplier_x;
    let filter_scale_y = scale_y.max(1.0) * multiplier_y;

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
    // Box uses true area integration for physically correct averaging
    // Tent-space methods use precomputed equivalent direct kernels
    let (h_weights, v_weights) = match method {
        RescaleMethod::Box => (
            // Box filter uses raw scale (not clamped) for true area integration
            // scale = src/dst, which is exactly the dest pixel footprint in source space
            // In tent mode, apply the same multiplier for box as other kernels
            precompute_box_weights(src_width, dst_width, scale_x, scale_x * multiplier_x, tent_mode),
            precompute_box_weights(src_height, dst_height, scale_y, scale_y * multiplier_y, tent_mode),
        ),
        RescaleMethod::TentBox => (
            // Tent-space box pipeline as equivalent direct kernel
            // Pass computed scale for uniform scaling support
            precompute_tent_kernel_weights(src_width, dst_width, scale_x, RescaleMethod::Box),
            precompute_tent_kernel_weights(src_height, dst_height, scale_y, RescaleMethod::Box),
        ),
        RescaleMethod::TentLanczos3 => (
            // Tent-space lanczos3 pipeline as equivalent direct kernel
            // Pass computed scale for uniform scaling support
            precompute_tent_kernel_weights(src_width, dst_width, scale_x, RescaleMethod::Lanczos3),
            precompute_tent_kernel_weights(src_height, dst_height, scale_y, RescaleMethod::Lanczos3),
        ),
        _ => (
            precompute_kernel_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method, tent_mode),
            precompute_kernel_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method, tent_mode),
        ),
    };

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
    tent_mode: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Compute scales - in tent mode, use sample-to-sample mapping with uniform scaling support
    let (scale_x, scale_y) = if tent_mode {
        // Compute natural tent scales for each dimension
        let natural_tent_scale_x = if dst_width > 1 {
            (src_width - 1) as f32 / (dst_width - 1) as f32
        } else {
            1.0
        };
        let natural_tent_scale_y = if dst_height > 1 {
            (src_height - 1) as f32 / (dst_height - 1) as f32
        } else {
            1.0
        };

        // Apply uniform scaling if requested
        match scale_mode {
            ScaleMode::Independent => (natural_tent_scale_x, natural_tent_scale_y),
            ScaleMode::UniformWidth => (natural_tent_scale_x, natural_tent_scale_x),
            ScaleMode::UniformHeight => (natural_tent_scale_y, natural_tent_scale_y),
        }
    } else {
        calculate_scales(src_width, src_height, dst_width, dst_height, scale_mode)
    };

    // Filter scale controls kernel support
    // In tent mode: expand kernel when upscaling (scale < 1.0), multiplier goes from 1.0 to 2.0
    let multiplier_x = if tent_mode { (1.0 / scale_x).clamp(1.0, 2.0) } else { 1.0 };
    let multiplier_y = if tent_mode { (1.0 / scale_y).clamp(1.0, 2.0) } else { 1.0 };
    let filter_scale_x = scale_x.max(1.0) * multiplier_x;
    let filter_scale_y = scale_y.max(1.0) * multiplier_y;

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

    // Box uses true area integration for physically correct averaging
    // Tent-space methods use precomputed equivalent direct kernels
    let (h_weights, v_weights) = match method {
        RescaleMethod::Box => (
            // Box filter uses raw scale (not clamped) for true area integration
            // In tent mode, apply the same multiplier for box as other kernels
            precompute_box_weights(src_width, dst_width, scale_x, scale_x * multiplier_x, tent_mode),
            precompute_box_weights(src_height, dst_height, scale_y, scale_y * multiplier_y, tent_mode),
        ),
        RescaleMethod::TentBox => (
            // Tent-space box pipeline as equivalent direct kernel
            // Pass computed scale for uniform scaling support
            precompute_tent_kernel_weights(src_width, dst_width, scale_x, RescaleMethod::Box),
            precompute_tent_kernel_weights(src_height, dst_height, scale_y, RescaleMethod::Box),
        ),
        RescaleMethod::TentLanczos3 => (
            // Tent-space lanczos3 pipeline as equivalent direct kernel
            // Pass computed scale for uniform scaling support
            precompute_tent_kernel_weights(src_width, dst_width, scale_x, RescaleMethod::Lanczos3),
            precompute_tent_kernel_weights(src_height, dst_height, scale_y, RescaleMethod::Lanczos3),
        ),
        _ => (
            precompute_kernel_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x, method, tent_mode),
            precompute_kernel_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y, method, tent_mode),
        ),
    };

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
