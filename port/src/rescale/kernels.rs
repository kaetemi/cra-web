//! Interpolation kernel functions and weight precomputation
//!
//! Contains the mathematical kernel functions (Mitchell, Catmull-Rom, Lanczos3, Sinc)
//! and the weight precomputation logic for separable 2-pass rescaling.

use std::f32::consts::PI;
use super::RescaleMethod;

/// Mitchell-Netravali kernel with B=C=1/3
/// This setting minimizes both blur and ringing artifacts.
/// Support is [-2, 2], overshoot is typically <1%
#[inline]
pub fn mitchell(x: f32) -> f32 {
    let x = x.abs();
    if x >= 2.0 {
        0.0
    } else if x >= 1.0 {
        // (-B - 6C)|x|³ + (6B + 30C)|x|² + (-12B - 48C)|x| + (8B + 24C)
        // With B=C=1/3: -7/3 x³ + 12x² - 20x + 32/3, divided by 6
        (-7.0/18.0) * x * x * x + 2.0 * x * x - (10.0/3.0) * x + 16.0/9.0
    } else {
        // (12 - 9B - 6C)|x|³ + (-18 + 12B + 6C)|x|² + (6 - 2B)
        // With B=C=1/3: 7x³ - 12x² + 16/3, divided by 6
        (7.0/6.0) * x * x * x - 2.0 * x * x + 8.0/9.0
    }
}

/// Catmull-Rom spline kernel (B=0, C=0.5)
/// Sharper than Mitchell, less ringing than Lanczos.
/// This is an interpolating spline (passes through original sample points).
/// Support is [-2, 2]
#[inline]
pub fn catmull_rom(x: f32) -> f32 {
    let x = x.abs();
    if x >= 2.0 {
        0.0
    } else if x >= 1.0 {
        // (-B - 6C)|x|³ + (6B + 30C)|x|² + (-12B - 48C)|x| + (8B + 24C)
        // With B=0, C=0.5: -3x³ + 15x² - 24x + 12, divided by 6
        -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
    } else {
        // (12 - 9B - 6C)|x|³ + (-18 + 12B + 6C)|x|² + (6 - 2B)
        // With B=0, C=0.5: 9x³ - 15x² + 6, divided by 6
        1.5 * x * x * x - 2.5 * x * x + 1.0
    }
}

/// Lanczos kernel with a=2
/// Smaller window than Lanczos3, less ringing but also less sharp.
#[inline]
pub fn lanczos2(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else if x.abs() >= 2.0 {
        0.0
    } else {
        let pi_x = PI * x;
        let pi_x_2 = pi_x / 2.0;
        (pi_x.sin() / pi_x) * (pi_x_2.sin() / pi_x_2)
    }
}

/// Lanczos kernel with a=3
#[inline]
pub fn lanczos3(x: f32) -> f32 {
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

/// Pure sinc kernel (non-windowed)
/// This is the theoretically ideal interpolation kernel for band-limited signals.
/// Unlike Lanczos, it has no window function and extends to infinity (full image).
/// WARNING: Causes severe Gibbs phenomenon (ringing) at sharp edges.
#[inline]
pub fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let pi_x = PI * x;
        pi_x.sin() / pi_x
    }
}

/// Bessel function J1(x) - polynomial approximation (Numerical Recipes)
/// Used for the jinc function which is the 2D analog of sinc.
#[inline]
pub fn bessel_j1(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
            + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
            + y * (99447.43394 + y * (376.9991397 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
            + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995 + y * (-0.2002690873e-3
            + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let ans = (0.636619772 / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 { -ans } else { ans }
    }
}

/// Jinc function: J1(π*x) / (π*x)
/// The 2D radially symmetric analog of sinc - ideal lowpass for circular apertures.
/// Used in proper EWA (Elliptical Weighted Average) resampling.
#[inline]
pub fn jinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        0.5  // limit as x -> 0
    } else {
        let pi_x = PI * x;
        bessel_j1(pi_x) / pi_x
    }
}

/// EWA Lanczos kernel: jinc windowed by jinc
/// The proper 2D radially symmetric Lanczos kernel for EWA resampling.
/// Uses jinc(r) * jinc(r/lobes) where jinc is the 2D analog of sinc.
#[inline]
pub fn ewa_lanczos(r: f32, lobes: f32) -> f32 {
    if r >= lobes {
        0.0
    } else if r < 1e-8 {
        1.0
    } else {
        jinc(r) * jinc(r / lobes)
    }
}

/// Box filter (rectangular window) - point sampled version
/// Returns 1.0 for |x| <= 0.5, 0.0 otherwise.
/// Note: For proper area-averaged box filtering, use box_integrated instead.
#[inline]
pub fn box_filter(x: f32) -> f32 {
    if x.abs() <= 0.5 {
        1.0
    } else {
        0.0
    }
}

/// Compute the exact overlap between a destination pixel's footprint and a source pixel.
///
/// For a destination pixel centered at `src_pos` (in source coordinates):
/// - Its footprint spans: [src_pos - half_width, src_pos + half_width]
/// - where half_width = 0.5 * filter_scale, and filter_scale = src_size / dst_size
///
/// For source pixel `si` (centered at integer position si):
/// - Its area spans: [si - 0.5, si + 0.5]
///
/// Returns the length of the intersection (overlap) between these two intervals.
///
/// Behavior by scale:
/// - Upscaling (filter_scale < 1): dest footprint < 1 source pixel, so most dest pixels
///   fall entirely within one source pixel (nearest-neighbor), with boundary pixels blended
/// - Downscaling (filter_scale > 1): dest footprint > 1 source pixel, so each dest pixel
///   properly averages multiple source pixels weighted by their overlap area
#[inline]
pub fn box_integrated(src_pos: f32, si: f32, filter_scale: f32) -> f32 {
    // Destination pixel's footprint in source coordinates
    let half_width = 0.5 * filter_scale;
    let dst_start = src_pos - half_width;
    let dst_end = src_pos + half_width;

    // Source pixel's area
    let src_start = si - 0.5;
    let src_end = si + 0.5;

    // Compute overlap
    let overlap_start = dst_start.max(src_start);
    let overlap_end = dst_end.min(src_end);

    // Return the overlap length (0 if no overlap)
    (overlap_end - overlap_start).max(0.0)
}

/// Generic kernel evaluation
#[inline]
pub fn eval_kernel(method: RescaleMethod, x: f32) -> f32 {
    match method {
        RescaleMethod::Bilinear => {
            let x = x.abs();
            if x < 1.0 { 1.0 - x } else { 0.0 }
        }
        RescaleMethod::Mitchell | RescaleMethod::EWAMitchell => mitchell(x),
        RescaleMethod::CatmullRom | RescaleMethod::EWACatmullRom => catmull_rom(x),
        RescaleMethod::Lanczos2 | RescaleMethod::EWASincLanczos2 | RescaleMethod::EWALanczos2 => lanczos2(x),
        RescaleMethod::Lanczos3 | RescaleMethod::Lanczos3Scatter | RescaleMethod::EWASincLanczos3 | RescaleMethod::EWALanczos3 => lanczos3(x),
        RescaleMethod::Sinc | RescaleMethod::SincScatter => sinc(x),
        RescaleMethod::Jinc => jinc(x),  // 2D radial, but fallback for 1D context
        RescaleMethod::Box => box_filter(x),
    }
}

/// Precomputed kernel weights for a single output position
/// Weights are normalized (sum to 1.0) and include source index range
#[derive(Clone)]
pub struct KernelWeights {
    /// First source index to sample from
    pub start_idx: usize,
    /// Normalized weights for each source sample (length = end_idx - start_idx + 1)
    pub weights: Vec<f32>,
    /// Fallback source index (when no weights available, e.g., at edges)
    pub fallback_idx: usize,
}

/// Precompute all kernel weights for 1D resampling
/// Returns exact weights for each destination position
pub fn precompute_kernel_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
    radius: i32,
    method: RescaleMethod,
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
            let weight = eval_kernel(method, d);
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

/// Precompute box filter weights using true area integration.
///
/// Unlike other kernels, box filter should receive the raw scale (src/dst) as filter_scale,
/// NOT the clamped max(scale, 1.0) value. This ensures:
/// - Upscaling: dest footprint < source pixel → nearest-neighbor behavior
/// - Downscaling: dest footprint > source pixel → proper area averaging
pub fn precompute_box_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Center offset
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    // Box filter radius depends on filter_scale:
    // The destination pixel footprint is filter_scale wide in source space
    // We need to sample all source pixels that could overlap this footprint
    let box_radius = (0.5 * filter_scale).ceil() as i32 + 1;

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
        let center = src_pos.floor() as i32;

        // Find the valid source index range
        let start = (center - box_radius).max(0) as usize;
        let end = ((center + box_radius) as usize).min(src_len - 1);

        // Collect box integrated weights for each source pixel
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            let weight = box_integrated(src_pos, si as f32, filter_scale);
            weights.push(weight);
            weight_sum += weight;
        }

        // Normalize weights
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
