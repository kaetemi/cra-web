/// Image rescaling module with Bilinear and Lanczos support
///
/// Operates in linear RGB space for correct color blending during interpolation.

use std::f32::consts::PI;

/// Rescaling method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RescaleMethod {
    /// Bilinear interpolation - fast, good for moderate scaling
    Bilinear,
    /// Lanczos3 - high quality, better for significant downscaling
    Lanczos3,
}

/// Scale mode for aspect ratio preservation
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScaleMode {
    /// Independent X/Y scaling (default, can cause slight AR distortion)
    #[default]
    Independent,
    /// Uniform scaling based on width (width is primary dimension)
    UniformWidth,
    /// Uniform scaling based on height (height is primary dimension)
    UniformHeight,
}

impl RescaleMethod {
    pub fn from_str(s: &str) -> Option<RescaleMethod> {
        match s.to_lowercase().as_str() {
            "bilinear" | "linear" => Some(RescaleMethod::Bilinear),
            "lanczos" | "lanczos3" => Some(RescaleMethod::Lanczos3),
            _ => None,
        }
    }
}

/// Lanczos kernel with a=3
#[inline]
fn lanczos3(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else if x.abs() >= 3.0 {
        0.0
    } else {
        let pi_x = PI * x;
        let pi_x_3 = pi_x / 3.0;
        (pi_x.sin() / pi_x) * (pi_x_3.sin() / pi_x_3)
    }
}

/// Precomputed kernel weights for a single output position
/// Weights are normalized (sum to 1.0) and include source index range
#[derive(Clone)]
struct KernelWeights {
    /// First source index to sample from
    start_idx: usize,
    /// Normalized weights for each source sample (length = end_idx - start_idx + 1)
    weights: Vec<f32>,
    /// Fallback source index (when no weights available, e.g., at edges)
    fallback_idx: usize,
}

/// Precompute all kernel weights for 1D Lanczos resampling
/// Returns exact weights for each destination position
fn precompute_lanczos_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
    radius: i32,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Center offset: if scale doesn't match src_len/dst_len (uniform scaling),
    // center the mapping so edges are equally cropped/extended
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
        let center = src_pos.floor() as i32;

        // Find the valid source index range
        let start = (center - radius).max(0) as usize;
        let end = ((center + radius) as usize).min(src_len - 1);

        // Collect ALL weights in the range (no skipping - maintains index correspondence)
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            let d = (src_pos - si as f32) / filter_scale;
            let weight = lanczos3(d);
            weights.push(weight);
            weight_sum += weight;
        }

        // Normalize weights (exact normalization)
        if weight_sum.abs() > 1e-8 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        let fallback = src_pos.round().clamp(0.0, (src_len - 1) as f32) as usize;

        all_weights.push(KernelWeights {
            start_idx: start,
            weights,
            fallback_idx: fallback,
        });
    }

    all_weights
}

/// Calculate scale factors based on scale mode
fn calculate_scales(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
) -> (f32, f32) {
    match scale_mode {
        ScaleMode::Independent => {
            (src_width as f32 / dst_width as f32,
             src_height as f32 / dst_height as f32)
        }
        ScaleMode::UniformWidth => {
            let scale = src_width as f32 / dst_width as f32;
            (scale, scale)
        }
        ScaleMode::UniformHeight => {
            let scale = src_height as f32 / dst_height as f32;
            (scale, scale)
        }
    }
}

/// Check if a number is a power of 2
#[inline]
fn is_power_of_2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Calculate target dimensions preserving aspect ratio
///
/// When both width and height are specified, checks if they're within 1 pixel
/// of uniform aspect ratio. If so, picks the "best" primary dimension:
/// 1. Power of 2 takes precedence
/// 2. Clean division of source dimension takes precedence
/// 3. Otherwise, largest dimension is primary
///
/// Set `force_exact` to true to skip the automatic uniform scaling detection
/// and use the exact dimensions provided (even if they cause slight distortion).
pub fn calculate_target_dimensions(
    src_width: usize,
    src_height: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
) -> (usize, usize) {
    calculate_target_dimensions_exact(src_width, src_height, target_width, target_height, false)
}

/// Calculate target dimensions with explicit control over uniform scaling
///
/// When `force_exact` is false (default), automatically adjusts dimensions to
/// preserve aspect ratio if they're within 1 pixel of uniform scaling.
///
/// When `force_exact` is true, uses the exact dimensions provided without
/// any automatic adjustment, allowing intentional non-uniform scaling.
pub fn calculate_target_dimensions_exact(
    src_width: usize,
    src_height: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
    force_exact: bool,
) -> (usize, usize) {
    match (target_width, target_height) {
        (Some(w), Some(h)) => {
            // If force_exact, skip all automatic adjustment
            if force_exact {
                return (w, h);
            }

            // Calculate what uniform AR would give us from each dimension
            let h_from_w = (w as f64 * src_height as f64 / src_width as f64).round() as usize;
            let w_from_h = (h as f64 * src_width as f64 / src_height as f64).round() as usize;

            // Check if both dimensions are within 1 pixel of uniform AR
            let h_close = (h as isize - h_from_w as isize).abs() <= 1;
            let w_close = (w as isize - w_from_h as isize).abs() <= 1;

            if h_close || w_close {
                // Pick the best primary dimension
                let width_is_pow2 = is_power_of_2(w);
                let height_is_pow2 = is_power_of_2(h);
                let width_divides = src_width % w == 0;
                let height_divides = src_height % h == 0;

                let use_width_as_primary = if width_is_pow2 && !height_is_pow2 {
                    true
                } else if height_is_pow2 && !width_is_pow2 {
                    false
                } else if width_divides && !height_divides {
                    true
                } else if height_divides && !width_divides {
                    false
                } else {
                    // Default: use larger dimension as primary
                    w >= h
                };

                if use_width_as_primary {
                    let aspect = src_height as f64 / src_width as f64;
                    (w, (w as f64 * aspect).round() as usize)
                } else {
                    let aspect = src_width as f64 / src_height as f64;
                    ((h as f64 * aspect).round() as usize, h)
                }
            } else {
                // Dimensions are too different from AR - use exact values (intentional distortion)
                (w, h)
            }
        }
        (Some(w), None) => {
            let aspect = src_height as f64 / src_width as f64;
            (w, (w as f64 * aspect).round() as usize)
        }
        (None, Some(h)) => {
            let aspect = src_width as f64 / src_height as f64;
            ((h as f64 * aspect).round() as usize, h)
        }
        (None, None) => (src_width, src_height),
    }
}

// ============================================================================
// SIMD-friendly Pixel4 rescaling
// ============================================================================

use crate::pixel::{Pixel4, lerp};

/// Rescale Pixel4 array using bilinear interpolation
/// Progress callback is optional - receives 0.0-1.0 after each row
fn rescale_bilinear_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let max_x = (src_width - 1) as f32;
    let max_y = (src_height - 1) as f32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = ((dst_x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, max_x);
            let src_y = ((dst_y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, max_y);

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            // Sample four corners
            let p00 = src[y0 * src_width + x0];
            let p10 = src[y0 * src_width + x1];
            let p01 = src[y1 * src_width + x0];
            let p11 = src[y1 * src_width + x1];

            // Bilinear interpolation using SIMD-friendly lerp
            let top = lerp(p00, p10, fx);
            let bottom = lerp(p01, p11, fx);
            dst[dst_y * dst_width + dst_x] = lerp(top, bottom, fy);
        }
        // Report progress after each row
        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

/// Alpha-aware bilinear interpolation helper
/// Weights RGB by alpha, falls back to unweighted if total alpha ≈ 0
#[inline]
fn bilinear_alpha_aware(
    p00: Pixel4, p10: Pixel4, p01: Pixel4, p11: Pixel4,
    fx: f32, fy: f32,
) -> Pixel4 {
    // Bilinear weights for each corner
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    // Alpha values
    let a00 = p00.a();
    let a10 = p10.a();
    let a01 = p01.a();
    let a11 = p11.a();

    // Alpha-weighted sum for RGB
    let aw00 = w00 * a00;
    let aw10 = w10 * a10;
    let aw01 = w01 * a01;
    let aw11 = w11 * a11;
    let total_alpha_weight = aw00 + aw10 + aw01 + aw11;

    // Interpolate alpha normally
    let out_alpha = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11;

    // RGB: use alpha-weighted interpolation, fall back to normal if all transparent
    let (out_r, out_g, out_b) = if total_alpha_weight > 1e-8 {
        let inv_aw = 1.0 / total_alpha_weight;
        (
            (aw00 * p00.r() + aw10 * p10.r() + aw01 * p01.r() + aw11 * p11.r()) * inv_aw,
            (aw00 * p00.g() + aw10 * p10.g() + aw01 * p01.g() + aw11 * p11.g()) * inv_aw,
            (aw00 * p00.b() + aw10 * p10.b() + aw01 * p01.b() + aw11 * p11.b()) * inv_aw,
        )
    } else {
        // All pixels transparent: use unweighted interpolation to preserve RGB
        (
            w00 * p00.r() + w10 * p10.r() + w01 * p01.r() + w11 * p11.r(),
            w00 * p00.g() + w10 * p10.g() + w01 * p01.g() + w11 * p11.g(),
            w00 * p00.b() + w10 * p10.b() + w01 * p01.b() + w11 * p11.b(),
        )
    };

    Pixel4::new(out_r, out_g, out_b, out_alpha)
}

/// Rescale Pixel4 array using alpha-aware bilinear interpolation
/// RGB channels are weighted by alpha during interpolation to prevent
/// transparent pixels from bleeding color into opaque regions.
/// Fully transparent regions preserve their underlying RGB values.
fn rescale_bilinear_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let max_x = (src_width - 1) as f32;
    let max_y = (src_height - 1) as f32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = ((dst_x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, max_x);
            let src_y = ((dst_y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, max_y);

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let p00 = src[y0 * src_width + x0];
            let p10 = src[y0 * src_width + x1];
            let p01 = src[y1 * src_width + x0];
            let p11 = src[y1 * src_width + x1];

            dst[dst_y * dst_width + dst_x] = bilinear_alpha_aware(p00, p10, p01, p11, fx, fy);
        }
        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

/// Lanczos3 1D resample for Pixel4 row using precomputed weights
/// Uses SIMD-friendly Pixel4 scalar multiply for better vectorization
#[inline]
fn lanczos3_resample_row_pixel4_precomputed(
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

/// Alpha-aware Lanczos3 1D resample for Pixel4 row
/// RGB is weighted by alpha; alpha is interpolated normally.
/// Falls back to unweighted RGB if all contributing pixels are transparent.
#[inline]
fn lanczos3_resample_row_alpha_precomputed(
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

/// Rescale Pixel4 array using Lanczos3 interpolation (separable, 2-pass)
/// Uses precomputed kernel weights for efficiency
/// Progress callback is optional - receives 0.0-1.0 after each row
fn rescale_lanczos3_pixels(
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
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    // Precompute weights for horizontal and vertical passes (reused across all rows/columns)
    let h_weights = precompute_lanczos_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x);
    let v_weights = precompute_lanczos_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y);

    // Pass 1: Horizontal resample each row (src_width -> dst_width)
    // Progress: 0% to 50%
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = lanczos3_resample_row_pixel4_precomputed(src_row, &h_weights);
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

/// Alpha-aware Lanczos3 rescale (separable, 2-pass)
/// RGB channels are weighted by alpha to prevent transparent pixel color bleeding.
fn rescale_lanczos3_alpha_pixels(
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
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    let h_weights = precompute_lanczos_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x);
    let v_weights = precompute_lanczos_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y);

    // Pass 1: Alpha-aware horizontal resample
    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = lanczos3_resample_row_alpha_precomputed(src_row, &h_weights);
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

/// Rescale Pixel4 image (SIMD-friendly, linear space)
pub fn rescale(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> Vec<Pixel4> {
    rescale_with_progress(src, src_width, src_height, dst_width, dst_height, method, scale_mode, None)
}

/// Rescale Pixel4 image with optional progress callback (SIMD-friendly, linear space)
/// Progress callback receives 0.0-1.0 (0.0 before first row, 1.0 after last row)
pub fn rescale_with_progress(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    if src_width == dst_width && src_height == dst_height {
        if let Some(ref mut cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => rescale_bilinear_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, progress),
        RescaleMethod::Lanczos3 => rescale_lanczos3_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, progress),
    }
}

/// Alpha-aware rescale for RGBA images
///
/// Unlike regular rescaling which treats all 4 channels equally, this function
/// weights RGB contributions by alpha during interpolation. This prevents
/// transparent pixels (which often have arbitrary RGB values, e.g., black in
/// dithered images) from bleeding their color into opaque regions.
///
/// Behavior:
/// - Opaque regions: RGB interpolated normally (alpha weights are ~1.0)
/// - Mixed regions: opaque pixels dominate RGB (weighted by their alpha)
/// - Fully transparent regions: RGB interpolated normally (fallback preserves underlying color)
/// - Alpha channel: always interpolated normally
pub fn rescale_with_alpha(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> Vec<Pixel4> {
    rescale_with_alpha_progress(src, src_width, src_height, dst_width, dst_height, method, scale_mode, None)
}

/// Alpha-aware rescale with progress callback
pub fn rescale_with_alpha_progress(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    if src_width == dst_width && src_height == dst_height {
        if let Some(ref mut cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => rescale_bilinear_alpha_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, progress),
        RescaleMethod::Lanczos3 => rescale_lanczos3_alpha_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, progress),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_identity() {
        // 2x2 image using Pixel4
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.25, 0.25, 0.25, 0.0),
            Pixel4::new(0.5, 0.5, 0.5, 0.0),
            Pixel4::new(0.75, 0.75, 0.75, 0.0),
        ];
        let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_bilinear_upscale() {
        // 2x2 -> 4x4
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        ];
        let dst = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);
        assert_eq!(dst.len(), 16);
        // Output should be in valid range and contain intermediate values
        for p in &dst {
            assert!(p[0] >= 0.0 && p[0] <= 1.0);
        }
        // Should have some variation
        let min = dst.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max = dst.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        assert!(max > min);
    }

    #[test]
    fn test_lanczos_identity() {
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.25, 0.25, 0.25, 0.0),
            Pixel4::new(0.5, 0.5, 0.5, 0.0),
            Pixel4::new(0.75, 0.75, 0.75, 0.0),
        ];
        let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_lanczos_downscale() {
        // 4x4 -> 2x2
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        ];
        let dst = rescale(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);
        assert_eq!(dst.len(), 4);
        // Left half should be ~0, right half ~1
        assert!(dst[0][0] < 0.5);
        assert!(dst[1][0] > 0.5);
    }

    #[test]
    fn test_calculate_dimensions() {
        // Width only
        let (w, h) = calculate_target_dimensions(100, 50, Some(50), None);
        assert_eq!(w, 50);
        assert_eq!(h, 25);

        // Height only
        let (w, h) = calculate_target_dimensions(100, 50, None, Some(25));
        assert_eq!(w, 50);
        assert_eq!(h, 25);

        // Both - exact AR match, larger dimension (width) is primary
        let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(100));
        assert_eq!(w, 200);
        assert_eq!(h, 100);

        // None
        let (w, h) = calculate_target_dimensions(100, 50, None, None);
        assert_eq!(w, 100);
        assert_eq!(h, 50);
    }

    #[test]
    fn test_calculate_dimensions_smart_primary() {
        // Power of 2 takes precedence: height=256 is pow2, width=512 is also pow2
        // Both pow2 -> larger wins, so width=512 is primary
        let (w, h) = calculate_target_dimensions(1024, 512, Some(512), Some(256));
        assert_eq!(w, 512);
        assert_eq!(h, 256);

        // Power of 2 takes precedence: only height=256 is pow2
        // 1920x1080 -> 455x256 (height is pow2, width is not)
        let (w, h) = calculate_target_dimensions(1920, 1080, Some(455), Some(256));
        assert_eq!(h, 256); // Height is primary (pow2)
        assert_eq!(w, (256.0_f64 * 1920.0 / 1080.0).round() as usize); // Width calculated from height

        // Power of 2 takes precedence: only width=512 is pow2
        // 1920x1080 -> 512x288
        let (w, h) = calculate_target_dimensions(1920, 1080, Some(512), Some(288));
        assert_eq!(w, 512); // Width is primary (pow2)
        assert_eq!(h, (512.0_f64 * 1080.0 / 1920.0).round() as usize); // Height calculated from width

        // Clean division: 1000x500 -> 250x125 (250 divides 1000, 125 divides 500)
        // Both divide cleanly, larger wins
        let (w, h) = calculate_target_dimensions(1000, 500, Some(250), Some(125));
        assert_eq!(w, 250);
        assert_eq!(h, 125);

        // Clean division: 1000x500 -> 200x100 (200 divides 1000, 100 divides 500)
        // Width is larger, so primary
        let (w, h) = calculate_target_dimensions(1000, 500, Some(200), Some(100));
        assert_eq!(w, 200);
        assert_eq!(h, 100);

        // Clean division wins over larger: 999x500 -> 200x100
        // 200 doesn't divide 999, but 100 divides 500 -> height is primary
        let (w, h) = calculate_target_dimensions(999, 500, Some(200), Some(100));
        assert_eq!(h, 100); // Height is primary (clean division)
        assert_eq!(w, (100.0_f64 * 999.0 / 500.0).round() as usize);

        // Intentional distortion: dimensions far from AR are kept exact
        // 100x50 (2:1) -> 200x200 (1:1) - very different AR
        let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(200));
        assert_eq!(w, 200);
        assert_eq!(h, 200); // Kept exact - intentional squish

        // Within 1 pixel tolerance: 100x50 -> 200x99
        // h_from_w = 200 * 50 / 100 = 100, diff = |99-100| = 1 (within tolerance)
        let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(99));
        assert_eq!(w, 200); // Width is primary (larger)
        assert_eq!(h, 100); // Corrected to proper AR
    }

    #[test]
    fn test_bilinear_roundtrip_2x() {
        // Test that 2x upscale then 2x downscale returns approximately original
        // 4x4 -> 8x8 -> 4x4
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.1, 0.1, 0.1, 0.0), Pixel4::new(0.4, 0.4, 0.4, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0),
            Pixel4::new(0.2, 0.2, 0.2, 0.0), Pixel4::new(0.5, 0.5, 0.5, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0),
            Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0),
        ];
        let up = rescale(&src, 4, 4, 8, 8, RescaleMethod::Bilinear, ScaleMode::Independent);
        let down = rescale(&up, 8, 8, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

        for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
            let diff = (orig[0] - result[0]).abs();
            assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig[0], result[0], diff);
        }
    }

    #[test]
    fn test_lanczos_roundtrip_2x() {
        // Test that 2x upscale then 2x downscale returns approximately original
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.1, 0.1, 0.1, 0.0), Pixel4::new(0.4, 0.4, 0.4, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0),
            Pixel4::new(0.2, 0.2, 0.2, 0.0), Pixel4::new(0.5, 0.5, 0.5, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0),
            Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0),
        ];
        let up = rescale(&src, 4, 4, 8, 8, RescaleMethod::Lanczos3, ScaleMode::Independent);
        let down = rescale(&up, 8, 8, 4, 4, RescaleMethod::Lanczos3, ScaleMode::Independent);

        for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
            let diff = (orig[0] - result[0]).abs();
            assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig[0], result[0], diff);
        }
    }

    #[test]
    fn test_no_shift_on_upscale() {
        // A single white pixel in center should stay centered after upscale
        // 3x3 with center pixel white -> 6x6
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        ];
        let dst = rescale(&src, 3, 3, 6, 6, RescaleMethod::Bilinear, ScaleMode::Independent);

        // The brightest area should still be in the center region
        let center_sum = dst[2 * 6 + 2][0] + dst[2 * 6 + 3][0] + dst[3 * 6 + 2][0] + dst[3 * 6 + 3][0];
        let corner_sum = dst[0][0] + dst[5][0] + dst[30][0] + dst[35][0];

        assert!(center_sum > corner_sum, "Center should be brighter than corners");
    }

    #[test]
    fn test_edge_pixels_preserved() {
        // Edge pixels shouldn't expand or shift weirdly
        // Left column black, right column white
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        ];
        let up = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

        // Left edge (x=0) should still be darkest
        // Right edge (x=3) should still be brightest
        let left_avg = (up[0][0] + up[4][0] + up[8][0] + up[12][0]) / 4.0;
        let right_avg = (up[3][0] + up[7][0] + up[11][0] + up[15][0]) / 4.0;

        assert!(left_avg < 0.5, "Left edge should be dark: {}", left_avg);
        assert!(right_avg > 0.5, "Right edge should be bright: {}", right_avg);
    }

    #[test]
    fn test_rgb_bilinear_identity() {
        let src = vec![
            Pixel4::new(0.0, 0.1, 0.2, 0.0),
            Pixel4::new(0.3, 0.4, 0.5, 0.0),
            Pixel4::new(0.6, 0.7, 0.8, 0.0),
            Pixel4::new(0.9, 1.0, 0.5, 0.0),
        ];
        let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_rgb_bilinear_upscale() {
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        ];
        let dst = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);
        assert_eq!(dst.len(), 16);

        // All values should be in valid range
        for p in &dst {
            for c in 0..3 {
                assert!(p[c] >= 0.0 && p[c] <= 1.0);
            }
        }
    }

    #[test]
    fn test_rgb_lanczos_roundtrip() {
        let src = vec![
            Pixel4::new(0.1, 0.2, 0.3, 0.0), Pixel4::new(0.4, 0.5, 0.6, 0.0),
            Pixel4::new(0.7, 0.8, 0.9, 0.0), Pixel4::new(0.2, 0.3, 0.4, 0.0),
        ];
        let up = rescale(&src, 2, 2, 4, 4, RescaleMethod::Lanczos3, ScaleMode::Independent);
        let down = rescale(&up, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);

        for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
            for c in 0..3 {
                let diff = (orig[c] - result[c]).abs();
                assert!(diff < 0.2, "Pixel {} channel {} drifted: {} -> {}", i, c, orig[c], result[c]);
            }
        }
    }

    #[test]
    fn test_uniform_scale_mode() {
        // Test that uniform scale modes produce identical scale factors
        // 100x50 -> 200x100 should be exactly 2x in both directions with uniform mode
        let (sx1, sy1) = calculate_scales(100, 50, 200, 100, ScaleMode::Independent);
        assert_eq!(sx1, 0.5);
        assert_eq!(sy1, 0.5);

        // 100x50 -> 200x99 with independent: different scales
        let (sx2, sy2) = calculate_scales(100, 50, 200, 99, ScaleMode::Independent);
        assert!((sx2 - 0.5).abs() < 0.001);
        assert!((sy2 - 0.505).abs() < 0.01); // 50/99 ≈ 0.505

        // With UniformWidth: both use width scale
        let (sx3, sy3) = calculate_scales(100, 50, 200, 99, ScaleMode::UniformWidth);
        assert_eq!(sx3, sy3);
        assert_eq!(sx3, 0.5);

        // With UniformHeight: both use height scale
        let (sx4, sy4) = calculate_scales(100, 50, 200, 99, ScaleMode::UniformHeight);
        assert_eq!(sx4, sy4);
        assert!((sx4 - 0.505).abs() < 0.01);
    }

    #[test]
    fn test_lanczos_prime_dimensions() {
        // Test with prime number dimensions to stress-test kernel weight computation
        // Prime numbers create non-repeating scale factors that can hit edge cases
        // 97x89 -> 53x47 (all primes)
        let src_w = 97;
        let src_h = 89;
        let dst_w = 53;
        let dst_h = 47;

        // Create a gradient test pattern
        let mut src = Vec::with_capacity(src_w * src_h);
        for y in 0..src_h {
            for x in 0..src_w {
                let r = x as f32 / (src_w - 1) as f32;
                let g = y as f32 / (src_h - 1) as f32;
                let b = ((x + y) as f32 / (src_w + src_h - 2) as f32).min(1.0);
                src.push(Pixel4::new(r, g, b, 0.0));
            }
        }

        let dst = rescale(&src, src_w, src_h, dst_w, dst_h, RescaleMethod::Lanczos3, ScaleMode::Independent);
        assert_eq!(dst.len(), dst_w * dst_h);

        // Verify output gradient is roughly preserved (corners should match)
        // Top-left should be dark
        assert!(dst[0][0] < 0.1, "Top-left R should be dark: {}", dst[0][0]);
        assert!(dst[0][1] < 0.1, "Top-left G should be dark: {}", dst[0][1]);

        // Bottom-right should be bright
        let br = &dst[(dst_h - 1) * dst_w + (dst_w - 1)];
        assert!(br[0] > 0.9, "Bottom-right R should be bright: {}", br[0]);
        assert!(br[1] > 0.9, "Bottom-right G should be bright: {}", br[1]);

        // All values should be finite and reasonable
        for (i, p) in dst.iter().enumerate() {
            for c in 0..3 {
                assert!(p[c].is_finite(), "Pixel {} channel {} is not finite: {}", i, c, p[c]);
            }
        }
    }

    #[test]
    fn test_lanczos_extreme_downscale_primes() {
        // Extreme downscale with primes: 1009x1013 -> 7x11
        // This creates a huge filter radius and tests weight accumulation
        let src_w = 127; // Smaller primes for faster test
        let src_h = 131;
        let dst_w = 7;
        let dst_h = 11;

        // Checkerboard pattern
        let mut src = Vec::with_capacity(src_w * src_h);
        for y in 0..src_h {
            for x in 0..src_w {
                let v = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
                src.push(Pixel4::new(v, v, v, 0.0));
            }
        }

        let dst = rescale(&src, src_w, src_h, dst_w, dst_h, RescaleMethod::Lanczos3, ScaleMode::Independent);
        assert_eq!(dst.len(), dst_w * dst_h);

        // Checkerboard should average to ~0.5 after extreme downscale
        for (i, p) in dst.iter().enumerate() {
            for c in 0..3 {
                assert!(p[c].is_finite(), "Pixel {} channel {} is not finite", i, c);
                // Should be somewhere around 0.5 (averaged checkerboard)
                assert!(p[c] > 0.3 && p[c] < 0.7,
                    "Pixel {} channel {} should be ~0.5 (averaged): {}", i, c, p[c]);
            }
        }
    }

    // ========================================================================
    // Alpha-aware rescaling tests
    // ========================================================================

    #[test]
    fn test_alpha_aware_no_bleed_bilinear() {
        // Test that black transparent pixels don't bleed into white opaque pixels
        // 2x2: top-left is white opaque, others are black transparent
        let src = vec![
            Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        ];

        // Regular rescale: black pixels bleed into the result
        let regular = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

        // Alpha-aware rescale: black transparent pixels should not affect RGB
        let alpha_aware = rescale_with_alpha(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

        // The top-left region (where opaque pixel was) should be brighter in alpha-aware
        // In regular mode, black transparent pixels darken the result
        let regular_tl = regular[0].r();
        let alpha_tl = alpha_aware[0].r();

        // Alpha-aware should preserve white better (closer to 1.0)
        assert!(alpha_tl >= regular_tl,
            "Alpha-aware top-left ({}) should be >= regular ({})", alpha_tl, regular_tl);
    }

    #[test]
    fn test_alpha_aware_preserves_transparent_rgb() {
        // Test that fully transparent regions preserve their underlying RGB
        // All pixels transparent, but with different RGB values
        let src = vec![
            Pixel4::new(1.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 1.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 0.0, 0.0),
        ];

        let dst = rescale_with_alpha(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);

        // Identity transform should preserve values exactly
        assert_eq!(src, dst);
    }

    #[test]
    fn test_alpha_aware_lanczos_no_bleed() {
        // Test that black transparent pixels don't darken opaque regions
        // Use a larger image to avoid edge effects where Lanczos overshoots
        // 4x4 with opaque white center, black transparent border
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        ];

        // Downscale 4x4 -> 2x2 (where black border would blend in)
        let regular = rescale(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);
        let alpha_aware = rescale_with_alpha(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);

        // In regular mode, the black border darkens everything
        // In alpha-aware mode, transparent pixels don't affect RGB
        // Check center pixel brightness
        let regular_avg: f32 = regular.iter().map(|p| p.r()).sum::<f32>() / 4.0;
        let alpha_avg: f32 = alpha_aware.iter().map(|p| p.r()).sum::<f32>() / 4.0;

        assert!(alpha_avg > regular_avg,
            "Alpha-aware avg brightness ({}) should be > regular ({})", alpha_avg, regular_avg);
    }

    #[test]
    fn test_alpha_channel_interpolated_normally() {
        // Alpha should be interpolated like any other channel
        // 2x2 with varying alpha
        let src = vec![
            Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        ];

        let dst = rescale_with_alpha(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

        // Top-left corner should have highest alpha (near 1.0)
        // Bottom-right corner should have lowest alpha (near 0.0)
        assert!(dst[0].a() > 0.5, "Top-left alpha should be high: {}", dst[0].a());
        assert!(dst[15].a() < 0.5, "Bottom-right alpha should be low: {}", dst[15].a());
    }

    #[test]
    fn test_alpha_aware_dithered_pattern() {
        // Simulate a dithered image: alternating opaque colored and black transparent
        // This is the problematic case for naive scaling
        let src = vec![
            Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0),
            Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0),
        ];

        // Downscale 4x4 -> 2x2
        let dst = rescale_with_alpha(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);

        // The RGB should still be approximately orange (1.0, 0.5, 0.0), not darkened by black
        for p in &dst {
            // Allow some tolerance due to Lanczos ringing
            assert!(p.r() > 0.7, "Red should stay high (not darkened by black): {}", p.r());
            assert!(p.g() > 0.3 && p.g() < 0.7, "Green should be mid-range: {}", p.g());
            assert!(p.b() < 0.3, "Blue should stay low: {}", p.b());
        }
    }
}
