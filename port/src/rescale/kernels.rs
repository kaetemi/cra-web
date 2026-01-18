//! Interpolation kernel functions and weight precomputation
//!
//! Contains the mathematical kernel functions (Mitchell, Catmull-Rom, Lanczos3, Sinc)
//! and the weight precomputation logic for separable 2-pass rescaling.

use std::f32::consts::PI;
use super::RescaleMethod;

/// Wang hash for deterministic per-pixel randomization
/// Same as used in mixed dithering for consistency
#[inline]
pub fn wang_hash(mut x: u32) -> u32 {
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_mul(9);
    x = x ^ (x >> 4);
    x = x.wrapping_mul(0x27d4eb2d);
    x = x ^ (x >> 15);
    x
}

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

/// Lanczos kernel with a=4
/// Sharper than Lanczos3, more ringing artifacts.
#[inline]
pub fn lanczos4(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else if x.abs() >= 4.0 {
        0.0
    } else {
        let pi_x = PI * x;
        let pi_x_4 = pi_x / 4.0;
        (pi_x.sin() / pi_x) * (pi_x_4.sin() / pi_x_4)
    }
}

/// Lanczos kernel with a=5
/// Very sharp, noticeable ringing on edges.
#[inline]
pub fn lanczos5(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else if x.abs() >= 5.0 {
        0.0
    } else {
        let pi_x = PI * x;
        let pi_x_5 = pi_x / 5.0;
        (pi_x.sin() / pi_x) * (pi_x_5.sin() / pi_x_5)
    }
}

/// Lanczos kernel with a=6
/// Extremely sharp, approaching sinc behavior. Significant ringing.
#[inline]
pub fn lanczos6(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else if x.abs() >= 6.0 {
        0.0
    } else {
        let pi_x = PI * x;
        let pi_x_6 = pi_x / 6.0;
        (pi_x.sin() / pi_x) * (pi_x_6.sin() / pi_x_6)
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
/// This computes the true box integral: the fraction of source pixel `si` that falls
/// within the destination pixel's footprint.
///
/// For a destination pixel at position `dst_pos`:
/// - Its footprint in source space is: [src_pos - half_width, src_pos + half_width]
/// - where half_width = 0.5 * filter_scale (filter_scale = src_size / dst_size)
///
/// For source pixel `si`:
/// - Its area spans: [si - 0.5, si + 0.5]
///
/// The overlap is the intersection of these two intervals.
///
/// This gives physically correct area averaging:
/// - For upscaling (filter_scale < 1): partial source pixel coverage
/// - For downscaling (filter_scale > 1): multiple source pixels averaged proportionally
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
        RescaleMethod::Mitchell => mitchell(x),
        RescaleMethod::CatmullRom => catmull_rom(x),
        RescaleMethod::Lanczos2 => lanczos2(x),
        RescaleMethod::Lanczos3 | RescaleMethod::Lanczos3Scatter | RescaleMethod::Lanczos3Integrated => lanczos3(x),
        RescaleMethod::Lanczos4 => lanczos4(x),
        RescaleMethod::Lanczos5 => lanczos5(x),
        RescaleMethod::Lanczos6 => lanczos6(x),
        RescaleMethod::Sinc | RescaleMethod::SincScatter | RescaleMethod::SincIntegrated => sinc(x),
        RescaleMethod::Box => box_filter(x),
        // Mixed methods should not call eval_kernel directly - they use specialized eval functions
        RescaleMethod::LanczosMixed | RescaleMethod::LanczosMixedScatter => lanczos3(x),
        RescaleMethod::Lanczos24Mixed => lanczos4(x),
        RescaleMethod::Lanczos35Mixed => lanczos5(x),
    }
}

/// Evaluate kernel for mixed 2+3 mode, selecting Lanczos2 or Lanczos3 based on source pixel
#[inline]
pub fn eval_kernel_mixed(x: f32, use_larger: bool) -> f32 {
    if use_larger {
        lanczos3(x)
    } else {
        lanczos2(x)
    }
}

/// Evaluate kernel for mixed 2+4 mode, selecting Lanczos2 or Lanczos4 based on source pixel
#[inline]
pub fn eval_kernel_mixed_24(x: f32, use_larger: bool) -> f32 {
    if use_larger {
        lanczos4(x)
    } else {
        lanczos2(x)
    }
}

/// Evaluate kernel for mixed 3+5 mode, selecting Lanczos3 or Lanczos5 based on source pixel
#[inline]
pub fn eval_kernel_mixed_35(x: f32, use_larger: bool) -> f32 {
    if use_larger {
        lanczos5(x)
    } else {
        lanczos3(x)
    }
}

/// Determine which kernel to use for a source pixel based on Wang hash
#[inline]
pub fn select_kernel_for_source(src_x: usize, src_y: usize, seed: u32) -> bool {
    let hashed_seed = wang_hash(seed);
    let pixel_hash = wang_hash((src_x as u32) ^ ((src_y as u32) << 16) ^ hashed_seed);
    // Use bit 0 to select kernel: true = larger kernel, false = smaller kernel
    (pixel_hash & 1) != 0
}

/// Integrate Lanczos3 kernel over a pixel's area using 5-point Gaussian quadrature
///
/// For a source pixel at index `si` (spanning si-0.5 to si+0.5), computes:
/// ∫[si-0.5 to si+0.5] lanczos3((src_pos - t) / filter_scale) dt
///
/// This is more accurate than point sampling, especially for downscaling
/// where each destination pixel covers multiple source pixels.
#[inline]
pub fn lanczos3_integrated(src_pos: f32, si: f32, filter_scale: f32) -> f32 {
    // 5-point Gauss-Legendre quadrature nodes and weights for [-1, 1]
    // Transformed to [-0.5, 0.5] by multiplying nodes by 0.5 and weights by 0.5
    const NODES: [f32; 5] = [
        0.0,
        -0.538469310105683 * 0.5,
         0.538469310105683 * 0.5,
        -0.906179845938664 * 0.5,
         0.906179845938664 * 0.5,
    ];
    const WEIGHTS: [f32; 5] = [
        0.568888888888889 * 0.5,
        0.478628670499366 * 0.5,
        0.478628670499366 * 0.5,
        0.236926885056189 * 0.5,
        0.236926885056189 * 0.5,
    ];

    let mut sum = 0.0f32;
    for i in 0..5 {
        // Sample point within the pixel: t = si + node
        // Kernel argument: (src_pos - t) / filter_scale = (src_pos - si - node) / filter_scale
        let d = (src_pos - si - NODES[i]) / filter_scale;
        sum += WEIGHTS[i] * lanczos3(d);
    }
    sum
}

/// Integrate Sinc kernel over a pixel's area using 5-point Gaussian quadrature
///
/// For a source pixel at index `si` (spanning si-0.5 to si+0.5), computes:
/// ∫[si-0.5 to si+0.5] sinc((src_pos - t) / filter_scale) dt
///
/// This is more accurate than point sampling for the pure sinc kernel.
#[inline]
pub fn sinc_integrated(src_pos: f32, si: f32, filter_scale: f32) -> f32 {
    // 5-point Gauss-Legendre quadrature nodes and weights for [-1, 1]
    // Transformed to [-0.5, 0.5] by multiplying nodes by 0.5 and weights by 0.5
    const NODES: [f32; 5] = [
        0.0,
        -0.538469310105683 * 0.5,
         0.538469310105683 * 0.5,
        -0.906179845938664 * 0.5,
         0.906179845938664 * 0.5,
    ];
    const WEIGHTS: [f32; 5] = [
        0.568888888888889 * 0.5,
        0.478628670499366 * 0.5,
        0.478628670499366 * 0.5,
        0.236926885056189 * 0.5,
        0.236926885056189 * 0.5,
    ];

    let mut sum = 0.0f32;
    for i in 0..5 {
        let d = (src_pos - si - NODES[i]) / filter_scale;
        sum += WEIGHTS[i] * sinc(d);
    }
    sum
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

/// Precompute integrated Lanczos3 kernel weights for 1D resampling
/// Uses pixel area integration instead of point sampling for more accurate weights
pub fn precompute_integrated_kernel_weights(
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

        // Collect integrated weights for each source pixel
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            let weight = lanczos3_integrated(src_pos, si as f32, filter_scale);
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

/// Precompute box filter weights using true area integration
/// Computes exact pixel overlap between destination and source pixels
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

/// Precompute integrated Sinc kernel weights for 1D resampling (full extent)
/// Uses pixel area integration instead of point sampling for more accurate weights
pub fn precompute_integrated_sinc_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Center offset
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;

        // Sinc uses full extent - all source pixels
        let start = 0;
        let end = src_len - 1;

        // Collect integrated weights for each source pixel
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            let weight = sinc_integrated(src_pos, si as f32, filter_scale);
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
