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
        RescaleMethod::Lanczos2 | RescaleMethod::Lanczos2Scatter => lanczos2(x),
        RescaleMethod::Lanczos3 | RescaleMethod::Lanczos3Scatter => lanczos3(x),
        RescaleMethod::Sinc | RescaleMethod::SincScatter => sinc(x),
        // Mixed methods should not call eval_kernel directly - they use eval_kernel_mixed
        RescaleMethod::LanczosMixed | RescaleMethod::LanczosMixedScatter => lanczos3(x),
    }
}

/// Evaluate kernel for mixed mode, selecting Lanczos2 or Lanczos3 based on source pixel
#[inline]
pub fn eval_kernel_mixed(x: f32, use_lanczos3: bool) -> f32 {
    if use_lanczos3 {
        lanczos3(x)
    } else {
        lanczos2(x)
    }
}

/// Determine which kernel to use for a source pixel based on Wang hash
#[inline]
pub fn select_kernel_for_source(src_x: usize, src_y: usize, seed: u32) -> bool {
    let hashed_seed = wang_hash(seed);
    let pixel_hash = wang_hash((src_x as u32) ^ ((src_y as u32) << 16) ^ hashed_seed);
    // Use bit 0 to select kernel: true = Lanczos3, false = Lanczos2
    (pixel_hash & 1) != 0
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
