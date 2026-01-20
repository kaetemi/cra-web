//! 2D Tent-Space EWA resampling
//!
//! This module implements 2D (non-separable) tent-space resampling using
//! precomputed kernel weights. Unlike the separable TentBox/TentLanczos3
//! methods, these use the full 2D tent-space kernel.
//!
//! Available methods:
//! - **Tent2DBox**: 2D tent-space with box filter integration
//! - **Tent2DLanczos3Jinc**: 2D tent-space with EWA Lanczos3-jinc kernel

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::precompute_tent_2d_kernel_weights;

/// 2D Tent-space resampling for Pixel4 images using precomputed kernel weights.
///
/// This applies the full 2D tent-space kernel (expand → resample → contract)
/// in a single pass using precomputed weights.
pub fn rescale_tent_2d_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate scales
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Determine if using EWA or box kernel
    let is_ewa = matches!(method, RescaleMethod::Tent2DLanczos3Jinc);

    // Precompute all 2D kernel weights
    let all_weights = precompute_tent_2d_kernel_weights(
        src_height, src_width, dst_height, dst_width,
        scale_y, scale_x, is_ewa
    );

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let row_weights = &all_weights[dst_y];

        for dst_x in 0..dst_width {
            let kw = &row_weights[dst_x];

            let mut sum = Pixel4::default();

            if kw.weights.is_empty() || kw.weights.iter().all(|&w| w.abs() < 1e-10) {
                // Fallback: use nearest pixel
                let idx = kw.fallback_y * src_width + kw.fallback_x;
                dst[dst_y * dst_width + dst_x] = src[idx];
                continue;
            }

            // Apply 2D kernel weights
            for ky in 0..kw.height {
                let src_y = kw.start_y + ky;
                if src_y >= src_height {
                    continue;
                }

                for kx in 0..kw.width {
                    let src_x = kw.start_x + kx;
                    if src_x >= src_width {
                        continue;
                    }

                    let weight = kw.weights[ky * kw.width + kx];
                    if weight.abs() < 1e-10 {
                        continue;
                    }

                    let pixel = src[src_y * src_width + src_x];
                    sum = sum + pixel * weight;
                }
            }

            dst[dst_y * dst_width + dst_x] = sum;
        }

        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

/// Alpha-aware 2D tent-space resampling for Pixel4 images.
/// RGB channels are weighted by alpha to prevent transparent pixel color bleeding.
pub fn rescale_tent_2d_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate scales
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Determine if using EWA or box kernel
    let is_ewa = matches!(method, RescaleMethod::Tent2DLanczos3Jinc);

    // Precompute all 2D kernel weights
    let all_weights = precompute_tent_2d_kernel_weights(
        src_height, src_width, dst_height, dst_width,
        scale_y, scale_x, is_ewa
    );

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let row_weights = &all_weights[dst_y];

        for dst_x in 0..dst_width {
            let kw = &row_weights[dst_x];

            if kw.weights.is_empty() || kw.weights.iter().all(|&w| w.abs() < 1e-10) {
                // Fallback: use nearest pixel
                let idx = kw.fallback_y * src_width + kw.fallback_x;
                dst[dst_y * dst_width + dst_x] = src[idx];
                continue;
            }

            // Alpha-aware accumulation
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_a = 0.0f32;
            let mut sum_alpha_weight = 0.0f32;
            let mut weight_sum = 0.0f32;
            // Fallback: unweighted RGB sum
            let mut sum_r_unweighted = 0.0f32;
            let mut sum_g_unweighted = 0.0f32;
            let mut sum_b_unweighted = 0.0f32;

            // Apply 2D kernel weights
            for ky in 0..kw.height {
                let src_y = kw.start_y + ky;
                if src_y >= src_height {
                    continue;
                }

                for kx in 0..kw.width {
                    let src_x = kw.start_x + kx;
                    if src_x >= src_width {
                        continue;
                    }

                    let weight = kw.weights[ky * kw.width + kx];
                    if weight.abs() < 1e-10 {
                        continue;
                    }

                    let p = src[src_y * src_width + src_x];
                    let alpha = p.a();
                    let aw = weight * alpha;

                    sum_r += aw * p.r();
                    sum_g += aw * p.g();
                    sum_b += aw * p.b();
                    sum_a += weight * alpha;
                    sum_alpha_weight += aw;
                    weight_sum += weight;

                    sum_r_unweighted += weight * p.r();
                    sum_g_unweighted += weight * p.g();
                    sum_b_unweighted += weight * p.b();
                }
            }

            // Normalize
            let out_a = if weight_sum.abs() > 1e-8 {
                sum_a / weight_sum
            } else {
                0.0
            };

            let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                let inv_aw = 1.0 / sum_alpha_weight;
                (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
            } else if weight_sum.abs() > 1e-8 {
                let inv_w = 1.0 / weight_sum;
                (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
            } else {
                // Fallback: nearest neighbor
                let p = src[kw.fallback_y * src_width + kw.fallback_x];
                (p.r(), p.g(), p.b())
            };

            dst[dst_y * dst_width + dst_x] = Pixel4::new(out_r, out_g, out_b, out_a);
        }

        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}
