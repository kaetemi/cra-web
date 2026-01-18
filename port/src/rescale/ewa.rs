//! EWA (Elliptical Weighted Average) resampling with radial sinc-Lanczos kernels
//!
//! This module implements EWA resampling using 1D sinc-Lanczos applied radially.
//! Unlike separable 2-pass convolution, EWA uses a true 2D filter footprint which
//! provides better quality for diagonal edges and non-uniform scaling.
//!
//! NOTE: This is "radial sinc-Lanczos", not true jinc-based EWA Lanczos.
//! True EWA Lanczos uses jinc (J1 Bessel) as the 2D analog of sinc.
//! The current implementation uses sinc(r)*sinc(r/a) on the radial distance,
//! which is a common simplification but not mathematically optimal for 2D.

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::{lanczos2, lanczos3};

/// Evaluate the radial sinc-Lanczos kernel
/// Uses the 1D sinc-Lanczos function on the radial distance
#[inline]
fn eval_ewa_kernel(method: RescaleMethod, r: f32) -> f32 {
    match method {
        RescaleMethod::EWASincLanczos2 => lanczos2(r),
        RescaleMethod::EWASincLanczos3 => lanczos3(r),
        // For the base methods, also support them
        RescaleMethod::Lanczos2 => lanczos2(r),
        RescaleMethod::Lanczos3 => lanczos3(r),
        _ => 0.0,
    }
}

/// Get the kernel radius for EWA methods
#[inline]
fn ewa_radius(method: RescaleMethod) -> f32 {
    match method {
        RescaleMethod::EWASincLanczos2 | RescaleMethod::Lanczos2 => 2.0,
        RescaleMethod::EWASincLanczos3 | RescaleMethod::Lanczos3 => 3.0,
        _ => 3.0,
    }
}

/// EWA resampling for Pixel4 images
/// Uses 2D radially symmetric Lanczos kernel
pub fn rescale_ewa_pixels(
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

    // Filter scale: expand kernel when downscaling
    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);

    // Use the larger filter scale for the radially symmetric kernel
    // This ensures we don't miss any source pixels when scales differ
    let filter_scale = filter_scale_x.max(filter_scale_y);

    let base_radius = ewa_radius(method);
    let radius = base_radius * filter_scale;

    // Center offsets for uniform scaling
    let mapped_src_width = dst_width as f32 * scale_x;
    let mapped_src_height = dst_height as f32 * scale_y;
    let offset_x = (src_width as f32 - mapped_src_width) / 2.0;
    let offset_y = (src_height as f32 - mapped_src_height) / 2.0;

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        // Map destination pixel center to source coordinates
        let src_pos_y = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;

        // Find vertical range of source pixels that could contribute
        let src_y_min = ((src_pos_y - radius).floor() as i32).max(0) as usize;
        let src_y_max = ((src_pos_y + radius).ceil() as i32).min(src_height as i32 - 1) as usize;

        for dst_x in 0..dst_width {
            // Map destination pixel center to source coordinates
            let src_pos_x = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;

            // Find horizontal range of source pixels that could contribute
            let src_x_min = ((src_pos_x - radius).floor() as i32).max(0) as usize;
            let src_x_max = ((src_pos_x + radius).ceil() as i32).min(src_width as i32 - 1) as usize;

            let mut sum = Pixel4::default();
            let mut weight_sum = 0.0f32;

            // Sample all source pixels within the circular kernel footprint
            for sy in src_y_min..=src_y_max {
                let dy = (sy as f32 - src_pos_y) / filter_scale_y;
                let dy2 = dy * dy;

                for sx in src_x_min..=src_x_max {
                    let dx = (sx as f32 - src_pos_x) / filter_scale_x;

                    // Compute 2D radial distance
                    let r = (dx * dx + dy2).sqrt();

                    // Early out if outside kernel radius
                    if r >= base_radius {
                        continue;
                    }

                    let weight = eval_ewa_kernel(method, r);
                    if weight.abs() > 1e-8 {
                        let pixel = src[sy * src_width + sx];
                        sum = sum + pixel * weight;
                        weight_sum += weight;
                    }
                }
            }

            // Normalize
            let dst_idx = dst_y * dst_width + dst_x;
            if weight_sum.abs() > 1e-8 {
                dst[dst_idx] = sum * (1.0 / weight_sum);
            } else {
                // Fallback: nearest neighbor
                let nx = src_pos_x.round().clamp(0.0, (src_width - 1) as f32) as usize;
                let ny = src_pos_y.round().clamp(0.0, (src_height - 1) as f32) as usize;
                dst[dst_idx] = src[ny * src_width + nx];
            }
        }

        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

/// Alpha-aware EWA resampling for Pixel4 images
/// RGB channels are weighted by alpha to prevent transparent pixel color bleeding
pub fn rescale_ewa_alpha_pixels(
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
    let filter_scale = filter_scale_x.max(filter_scale_y);

    let base_radius = ewa_radius(method);
    let radius = base_radius * filter_scale;

    let mapped_src_width = dst_width as f32 * scale_x;
    let mapped_src_height = dst_height as f32 * scale_y;
    let offset_x = (src_width as f32 - mapped_src_width) / 2.0;
    let offset_y = (src_height as f32 - mapped_src_height) / 2.0;

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let src_pos_y = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;

        let src_y_min = ((src_pos_y - radius).floor() as i32).max(0) as usize;
        let src_y_max = ((src_pos_y + radius).ceil() as i32).min(src_height as i32 - 1) as usize;

        for dst_x in 0..dst_width {
            let src_pos_x = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;

            let src_x_min = ((src_pos_x - radius).floor() as i32).max(0) as usize;
            let src_x_max = ((src_pos_x + radius).ceil() as i32).min(src_width as i32 - 1) as usize;

            // Accumulators for alpha-aware blending
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

            for sy in src_y_min..=src_y_max {
                let dy = (sy as f32 - src_pos_y) / filter_scale_y;
                let dy2 = dy * dy;

                for sx in src_x_min..=src_x_max {
                    let dx = (sx as f32 - src_pos_x) / filter_scale_x;
                    let r = (dx * dx + dy2).sqrt();

                    if r >= base_radius {
                        continue;
                    }

                    let weight = eval_ewa_kernel(method, r);
                    if weight.abs() > 1e-8 {
                        let p = src[sy * src_width + sx];
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
            }

            // Normalize
            let dst_idx = dst_y * dst_width + dst_x;

            // Alpha: normal interpolation
            let out_a = if weight_sum.abs() > 1e-8 {
                sum_a / weight_sum
            } else {
                0.0
            };

            // RGB: alpha-weighted or fallback to unweighted
            let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                let inv_aw = 1.0 / sum_alpha_weight;
                (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
            } else if weight_sum.abs() > 1e-8 {
                let inv_w = 1.0 / weight_sum;
                (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
            } else {
                // Fallback: nearest neighbor
                let nx = src_pos_x.round().clamp(0.0, (src_width - 1) as f32) as usize;
                let ny = src_pos_y.round().clamp(0.0, (src_height - 1) as f32) as usize;
                let p = src[ny * src_width + nx];
                (p.r(), p.g(), p.b())
            };

            dst[dst_idx] = Pixel4::new(out_r, out_g, out_b, out_a);
        }

        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}
