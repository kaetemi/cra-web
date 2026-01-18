//! EWA (Elliptical Weighted Average) resampling
//!
//! This module implements EWA resampling using 2D radially symmetric kernels.
//! Unlike separable 2-pass convolution, EWA uses a true 2D filter footprint which
//! provides better quality for diagonal edges and non-uniform scaling.
//!
//! Available kernel types:
//! - **EWASincLanczos2/3**: Uses 1D sinc-Lanczos applied radially (sinc(r)*sinc(r/a)).
//! - **EWALanczos2/3**: Uses proper jinc-based kernel (jinc(r)*jinc(r/a)).
//! - **EWAMitchell**: Mitchell-Netravali cubic (B=C=1/3) applied radially.
//! - **EWACatmullRom**: Catmull-Rom cubic (B=0, C=0.5) applied radially.
//! - **StochasticJinc**: Probabilistic jinc with Gaussian sampling as implicit window.

use crate::pixel::Pixel4;
use super::{RescaleMethod, ScaleMode, calculate_scales};
use super::kernels::{lanczos2, lanczos3, ewa_lanczos, mitchell, catmull_rom, jinc};

/// Fast deterministic hash for stochastic sampling
/// Returns a value in [0, 1)
#[inline]
fn hash_to_uniform(seed: u32) -> f32 {
    // PCG-like hash for good distribution
    let mut h = seed;
    h = h.wrapping_mul(747796405).wrapping_add(2891336453);
    h = ((h >> ((h >> 28).wrapping_add(4))) ^ h).wrapping_mul(277803737);
    h = (h >> 22) ^ h;
    (h as f32) / (u32::MAX as f32)
}

/// Generate two uniform random values from pixel coordinates and sample index
#[inline]
fn hash_uniform_pair(dst_x: usize, dst_y: usize, sample_idx: usize) -> (f32, f32) {
    let seed1 = (dst_x as u32)
        .wrapping_mul(1597334677)
        .wrapping_add((dst_y as u32).wrapping_mul(3812015801))
        .wrapping_add((sample_idx as u32).wrapping_mul(2798796415));
    let seed2 = seed1.wrapping_mul(1103515245).wrapping_add(12345);
    (hash_to_uniform(seed1), hash_to_uniform(seed2))
}

/// Generate a 2D Gaussian-distributed sample using Box-Muller transform
/// Returns (dx, dy) offset scaled by sigma
#[inline]
fn gaussian_sample_2d(dst_x: usize, dst_y: usize, sample_idx: usize, sigma: f32) -> (f32, f32) {
    let (u1, u2) = hash_uniform_pair(dst_x, dst_y, sample_idx);
    // Clamp u1 away from 0 to avoid ln(0)
    let u1 = u1.max(1e-10);
    let r = (-2.0 * u1.ln()).sqrt() * sigma;
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Evaluate the EWA kernel at radial distance r
#[inline]
fn eval_ewa_kernel(method: RescaleMethod, r: f32) -> f32 {
    match method {
        // Radial sinc-Lanczos (1D kernel applied radially)
        RescaleMethod::EWASincLanczos2 | RescaleMethod::Lanczos2 => lanczos2(r),
        RescaleMethod::EWASincLanczos3 | RescaleMethod::Lanczos3 => lanczos3(r),
        // Proper jinc-based EWA Lanczos (true 2D kernel)
        RescaleMethod::EWALanczos2 => ewa_lanczos(r, 2.0),
        RescaleMethod::EWALanczos3 => ewa_lanczos(r, 3.0),
        // Cubic kernels applied radially
        RescaleMethod::EWAMitchell | RescaleMethod::Mitchell => mitchell(r),
        RescaleMethod::EWACatmullRom | RescaleMethod::CatmullRom => catmull_rom(r),
        // Pure jinc (unwindowed, full extent)
        RescaleMethod::Jinc => jinc(r),
        _ => 0.0,
    }
}

/// Get the kernel radius for EWA methods
/// Returns 0.0 for full-extent methods (Jinc)
#[inline]
fn ewa_radius(method: RescaleMethod) -> f32 {
    match method {
        RescaleMethod::EWASincLanczos2 | RescaleMethod::EWALanczos2 | RescaleMethod::Lanczos2 => 2.0,
        RescaleMethod::EWASincLanczos3 | RescaleMethod::EWALanczos3 | RescaleMethod::Lanczos3 => 3.0,
        RescaleMethod::EWAMitchell | RescaleMethod::Mitchell => 2.0,
        RescaleMethod::EWACatmullRom | RescaleMethod::CatmullRom => 2.0,
        RescaleMethod::Jinc => 0.0, // Full extent
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
    // For full extent methods (radius 0), use max dimension; otherwise scale the radius
    let radius = if base_radius == 0.0 {
        (src_width.max(src_height) as f32) * filter_scale
    } else {
        base_radius * filter_scale
    };

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

                    // Early out if outside kernel radius (skip for full-extent methods where base_radius == 0)
                    if base_radius > 0.0 && r >= base_radius {
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
    // For full extent methods (radius 0), use max dimension; otherwise scale the radius
    let radius = if base_radius == 0.0 {
        (src_width.max(src_height) as f32) * filter_scale
    } else {
        base_radius * filter_scale
    };

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

                    // Early out if outside kernel radius (skip for full-extent methods where base_radius == 0)
                    if base_radius > 0.0 && r >= base_radius {
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

/// Stochastic EWA Jinc resampling with adaptive sample count
///
/// Uses Monte Carlo sampling with Gaussian-distributed sample positions.
/// Instead of evaluating all pixels, takes samples drawn from a 2D Gaussian.
/// Sample count matches what an EWA Lanczos5 would use (π × radius²), scaling
/// with filter_scale² for proper coverage when downscaling.
///
/// Parameters:
/// - Base radius: 5.0 (Lanczos5 equivalent)
/// - Sample count: π × (radius × filter_scale)², clamped to [64, 4096]
/// - Sigma: radius × filter_scale (Gaussian matches kernel support)
pub fn rescale_stochastic_jinc_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Filter scale: expand kernel when downscaling
    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let filter_scale = filter_scale_x.max(filter_scale_y);

    // Match Lanczos5 sample count: π × r² where r = base_radius × filter_scale
    const BASE_RADIUS: f32 = 5.0;
    let scaled_radius = BASE_RADIUS * filter_scale;
    let num_samples = (std::f32::consts::PI * scaled_radius * scaled_radius)
        .round()
        .clamp(64.0, 4096.0) as usize;

    // Sigma for Gaussian sampling - matches the kernel support radius
    let sigma = scaled_radius;

    // Center offsets for uniform scaling
    let mapped_src_width = dst_width as f32 * scale_x;
    let mapped_src_height = dst_height as f32 * scale_y;
    let offset_x = (src_width as f32 - mapped_src_width) / 2.0;
    let offset_y = (src_height as f32 - mapped_src_height) / 2.0;

    let src_w = src_width as i32;
    let src_h = src_height as i32;

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let src_pos_y = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;

        for dst_x in 0..dst_width {
            let src_pos_x = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;

            let mut sum = Pixel4::default();
            let mut weight_sum = 0.0f32;

            for sample_idx in 0..num_samples {
                // Generate Gaussian-distributed offset
                let (dx, dy) = gaussian_sample_2d(dst_x, dst_y, sample_idx, sigma);

                // Map to source pixel coordinates
                let sx = (src_pos_x + dx).round() as i32;
                let sy = (src_pos_y + dy).round() as i32;

                // Bounds check
                if sx < 0 || sx >= src_w || sy < 0 || sy >= src_h {
                    continue;
                }

                // Compute radial distance in filter-scaled space
                let r = ((dx / filter_scale_x).powi(2) + (dy / filter_scale_y).powi(2)).sqrt();

                // Weight by jinc - no importance sampling correction needed
                // since we're sampling FROM the Gaussian (it's our window)
                let weight = jinc(r);

                if weight.abs() > 1e-8 {
                    let pixel = src[sy as usize * src_width + sx as usize];
                    sum = sum + pixel * weight;
                    weight_sum += weight;
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

/// Stochastic Jinc scatter resampling - normalizes emission per source pixel
///
/// Unlike the gather variant which normalizes intake per destination, this version
/// normalizes emission per source pixel. Each source pixel distributes its value
/// across destinations with weights summing to 1.0, ensuring energy conservation.
///
/// Parameters:
/// - Base radius: 5.0 (Lanczos5 equivalent)
/// - Sample count: scales with destination/source area ratio for proper coverage
/// - Sigma: radius × filter_scale (Gaussian matches kernel support)
pub fn rescale_stochastic_jinc_scatter_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Filter scale: expand kernel when downscaling
    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let filter_scale = filter_scale_x.max(filter_scale_y);

    // Base sample count from Lanczos5: π × r²
    const BASE_RADIUS: f32 = 5.0;
    let scaled_radius = BASE_RADIUS * filter_scale;
    let base_samples = std::f32::consts::PI * scaled_radius * scaled_radius;

    // Scale by area ratio to match gather's total sample count
    // Upscaling: more samples per source (each covers more dest pixels)
    // Downscaling: fewer samples per source (many sources per dest)
    let area_ratio = 1.0 / (scale_x * scale_y);
    let num_samples = (base_samples * area_ratio)
        .round()
        .clamp(64.0, 16384.0) as usize;

    // Emission scale: also compensate for area ratio
    let emission_scale = area_ratio;

    // Sigma for Gaussian sampling - matches first jinc zero (r ≈ 1.22)
    const JINC_FIRST_ZERO: f32 = 1.2197;
    let sigma = JINC_FIRST_ZERO * filter_scale;

    // Center offsets for uniform scaling
    let mapped_src_width = dst_width as f32 * scale_x;
    let mapped_src_height = dst_height as f32 * scale_y;
    let offset_x = (src_width as f32 - mapped_src_width) / 2.0;
    let offset_y = (src_height as f32 - mapped_src_height) / 2.0;

    let dst_w = dst_width as i32;
    let dst_h = dst_height as i32;

    // Accumulator for destination pixels (no per-destination normalization)
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for src_y in 0..src_height {
        for src_x in 0..src_width {
            let src_pixel = src[src_y * src_width + src_x];

            // Map source pixel center to destination coordinates
            // Inverse of: src_pos = (dst + 0.5) * scale - 0.5 + offset
            // => dst = (src_pos + 0.5 - offset) / scale - 0.5
            let dst_pos_x = (src_x as f32 + 0.5 - offset_x) / scale_x - 0.5;
            let dst_pos_y = (src_y as f32 + 0.5 - offset_y) / scale_y - 0.5;

            // Collect samples and weights for this source pixel
            let mut samples: Vec<(usize, usize, f32)> = Vec::with_capacity(num_samples);
            let mut weight_sum = 0.0f32;

            for sample_idx in 0..num_samples {
                // Generate Gaussian-distributed offset in source space
                let (dx_src, dy_src) = gaussian_sample_2d(src_x, src_y, sample_idx, sigma);

                // Compute radial distance in filter-scaled space (source space)
                let r = ((dx_src / filter_scale_x).powi(2) + (dy_src / filter_scale_y).powi(2)).sqrt();

                // Weight by jinc - always count towards weight_sum
                let weight = jinc(r);
                if weight.abs() > 1e-8 {
                    weight_sum += weight;

                    // Convert offset to destination space
                    let dx_dst = dx_src / scale_x;
                    let dy_dst = dy_src / scale_y;

                    // Map to destination pixel coordinates
                    let dx = (dst_pos_x + dx_dst).round() as i32;
                    let dy = (dst_pos_y + dy_dst).round() as i32;

                    // Only emit if in bounds
                    if dx >= 0 && dx < dst_w && dy >= 0 && dy < dst_h {
                        samples.push((dx as usize, dy as usize, weight));
                    }
                }
            }

            // Normalize emission per source pixel, scaled by area ratio
            // Each source emits emission_scale total (1.0 at 1:1, 4.0 at 2x upscale, 0.25 at 2x downscale)
            if weight_sum.abs() > 1e-8 {
                let inv_weight_sum = emission_scale / weight_sum;
                for (dx, dy, weight) in samples {
                    let normalized_weight = weight * inv_weight_sum;
                    let dst_idx = dy * dst_width + dx;
                    dst[dst_idx] = dst[dst_idx] + src_pixel * normalized_weight;
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb((src_y + 1) as f32 / src_height as f32);
        }
    }

    dst
}

/// Alpha-aware stochastic Jinc scatter resampling
/// RGB channels are weighted by alpha, emission normalized per source pixel
pub fn rescale_stochastic_jinc_scatter_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let filter_scale = filter_scale_x.max(filter_scale_y);

    // Base sample count from Lanczos5: π × r²
    const BASE_RADIUS: f32 = 5.0;
    let scaled_radius = BASE_RADIUS * filter_scale;
    let base_samples = std::f32::consts::PI * scaled_radius * scaled_radius;

    // Scale by area ratio to match gather's total sample count
    let area_ratio = 1.0 / (scale_x * scale_y);
    let num_samples = (base_samples * area_ratio)
        .round()
        .clamp(64.0, 16384.0) as usize;

    // Emission scale: also compensate for area ratio
    let emission_scale = area_ratio;

    // Sigma for Gaussian sampling - matches first jinc zero (r ≈ 1.22)
    const JINC_FIRST_ZERO: f32 = 1.2197;
    let sigma = JINC_FIRST_ZERO * filter_scale;

    let mapped_src_width = dst_width as f32 * scale_x;
    let mapped_src_height = dst_height as f32 * scale_y;
    let offset_x = (src_width as f32 - mapped_src_width) / 2.0;
    let offset_y = (src_height as f32 - mapped_src_height) / 2.0;

    let dst_w = dst_width as i32;
    let dst_h = dst_height as i32;

    // Accumulators for alpha-aware blending
    let mut dst_r = vec![0.0f32; dst_width * dst_height];
    let mut dst_g = vec![0.0f32; dst_width * dst_height];
    let mut dst_b = vec![0.0f32; dst_width * dst_height];
    let mut dst_a = vec![0.0f32; dst_width * dst_height];
    let mut dst_alpha_weight = vec![0.0f32; dst_width * dst_height];

    for src_y in 0..src_height {
        for src_x in 0..src_width {
            let p = src[src_y * src_width + src_x];
            let alpha = p.a();

            let dst_pos_x = (src_x as f32 + 0.5 - offset_x) / scale_x - 0.5;
            let dst_pos_y = (src_y as f32 + 0.5 - offset_y) / scale_y - 0.5;

            // Collect samples and weights
            let mut samples: Vec<(usize, usize, f32)> = Vec::with_capacity(num_samples);
            let mut weight_sum = 0.0f32;

            for sample_idx in 0..num_samples {
                let (dx_src, dy_src) = gaussian_sample_2d(src_x, src_y, sample_idx, sigma);

                // Compute radial distance in filter-scaled space (source space)
                let r = ((dx_src / filter_scale_x).powi(2) + (dy_src / filter_scale_y).powi(2)).sqrt();

                // Weight by jinc - always count towards weight_sum
                let weight = jinc(r);
                if weight.abs() > 1e-8 {
                    weight_sum += weight;

                    // Convert offset to destination space
                    let dx_dst = dx_src / scale_x;
                    let dy_dst = dy_src / scale_y;

                    // Map to destination pixel coordinates
                    let dx = (dst_pos_x + dx_dst).round() as i32;
                    let dy = (dst_pos_y + dy_dst).round() as i32;

                    // Only emit if in bounds
                    if dx >= 0 && dx < dst_w && dy >= 0 && dy < dst_h {
                        samples.push((dx as usize, dy as usize, weight));
                    }
                }
            }

            // Normalize emission and scatter, scaled by area ratio
            if weight_sum.abs() > 1e-8 {
                let inv_weight_sum = emission_scale / weight_sum;
                for (dx, dy, weight) in samples {
                    let normalized_weight = weight * inv_weight_sum;
                    let aw = normalized_weight * alpha;
                    let dst_idx = dy * dst_width + dx;

                    dst_r[dst_idx] += aw * p.r();
                    dst_g[dst_idx] += aw * p.g();
                    dst_b[dst_idx] += aw * p.b();
                    dst_a[dst_idx] += normalized_weight * alpha;
                    dst_alpha_weight[dst_idx] += aw;
                }
            }
        }

        if let Some(ref mut cb) = progress {
            cb((src_y + 1) as f32 / src_height as f32);
        }
    }

    // Finalize: divide RGB by alpha weight
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];
    for i in 0..dst.len() {
        let out_a = dst_a[i];
        let (out_r, out_g, out_b) = if dst_alpha_weight[i].abs() > 1e-8 {
            let inv_aw = 1.0 / dst_alpha_weight[i];
            (dst_r[i] * inv_aw, dst_g[i] * inv_aw, dst_b[i] * inv_aw)
        } else {
            (0.0, 0.0, 0.0)
        };
        dst[i] = Pixel4::new(out_r, out_g, out_b, out_a);
    }

    dst
}

/// Alpha-aware stochastic EWA Jinc resampling with adaptive sample count
/// RGB channels are weighted by alpha to prevent transparent pixel color bleeding
pub fn rescale_stochastic_jinc_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let filter_scale = filter_scale_x.max(filter_scale_y);

    // Match Lanczos5 sample count: π × r² where r = base_radius × filter_scale
    const BASE_RADIUS: f32 = 5.0;
    let scaled_radius = BASE_RADIUS * filter_scale;
    let num_samples = (std::f32::consts::PI * scaled_radius * scaled_radius)
        .round()
        .clamp(64.0, 4096.0) as usize;

    // Sigma for Gaussian sampling - matches the kernel support radius
    let sigma = scaled_radius;

    let mapped_src_width = dst_width as f32 * scale_x;
    let mapped_src_height = dst_height as f32 * scale_y;
    let offset_x = (src_width as f32 - mapped_src_width) / 2.0;
    let offset_y = (src_height as f32 - mapped_src_height) / 2.0;

    let src_w = src_width as i32;
    let src_h = src_height as i32;

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let src_pos_y = (dst_y as f32 + 0.5) * scale_y - 0.5 + offset_y;

        for dst_x in 0..dst_width {
            let src_pos_x = (dst_x as f32 + 0.5) * scale_x - 0.5 + offset_x;

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

            for sample_idx in 0..num_samples {
                // Generate Gaussian-distributed offset
                let (dx, dy) = gaussian_sample_2d(dst_x, dst_y, sample_idx, sigma);

                // Map to source pixel coordinates
                let sx = (src_pos_x + dx).round() as i32;
                let sy = (src_pos_y + dy).round() as i32;

                // Bounds check
                if sx < 0 || sx >= src_w || sy < 0 || sy >= src_h {
                    continue;
                }

                // Compute radial distance in filter-scaled space
                let r = ((dx / filter_scale_x).powi(2) + (dy / filter_scale_y).powi(2)).sqrt();

                let weight = jinc(r);

                if weight.abs() > 1e-8 {
                    let p = src[sy as usize * src_width + sx as usize];
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
            let dst_idx = dst_y * dst_width + dst_x;

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
