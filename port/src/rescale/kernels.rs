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
        RescaleMethod::Jinc | RescaleMethod::StochasticJinc | RescaleMethod::StochasticJincScatter | RescaleMethod::StochasticJincScatterNormalized |
        RescaleMethod::EWALanczos3Sharp | RescaleMethod::EWALanczos4Sharpest => jinc(x),  // 2D radial, but fallback for 1D context
        RescaleMethod::Box | RescaleMethod::TentBox | RescaleMethod::Tent2DBox => box_filter(x),
        RescaleMethod::TentLanczos3 => lanczos3(x),
        // 2D methods use EWA/radial kernels, but provide fallback for 1D context
        RescaleMethod::Tent2DLanczos3Jinc => ewa_lanczos(x, 3.0),
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

/// Precomputed 2D kernel weights for a single output position
/// Weights are stored as a 2D grid representing source pixel contributions
#[derive(Clone)]
pub struct KernelWeights2D {
    /// Top-left source Y index
    pub start_y: usize,
    /// Top-left source X index
    pub start_x: usize,
    /// Kernel height (number of source rows)
    pub height: usize,
    /// Kernel width (number of source columns)
    pub width: usize,
    /// Normalized weights stored row-major: weights[y * width + x]
    pub weights: Vec<f32>,
    /// Fallback source (y, x) index
    pub fallback_y: usize,
    pub fallback_x: usize,
}

/// Precompute all kernel weights for 1D resampling
/// Returns exact weights for each destination position
///
/// When `tent_mode` is true, uses sample-to-sample mapping for tent-space coordinates:
/// - Scale becomes (src_len-1)/(dst_len-1) to map edge samples to edge samples
/// - Offset is adjusted to align integer sample positions rather than pixel centers
pub fn precompute_kernel_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
    radius: i32,
    method: RescaleMethod,
    tent_mode: bool,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Coordinate mapping:
    // - In tent mode, caller provides the effective tent scale (potentially uniform across dimensions)
    // - In non-tent mode, use the passed-in scale directly
    // Both cases use the same centering formula: offset = (src - dst * scale) / 2
    let _ = tent_mode; // tent_mode affects how caller computes scale; here we just use the passed scale
    let center_offset = (src_len as f32 - dst_len as f32 * scale) / 2.0;
    let (effective_scale, offset) = (scale, center_offset);

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * effective_scale - 0.5 + offset;
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
///
/// When `tent_mode` is true, uses sample-to-sample mapping for tent-space coordinates.
pub fn precompute_box_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    filter_scale: f32,
    tent_mode: bool,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Coordinate mapping: use passed-in scale with centering offset
    // In tent mode, caller provides the effective tent scale (potentially uniform across dimensions)
    let _ = tent_mode; // tent_mode affects how caller computes scale; here we just use the passed scale
    let center_offset = (src_len as f32 - dst_len as f32 * scale) / 2.0;
    let (effective_scale, offset) = (scale, center_offset);

    // Box filter radius depends on filter_scale:
    // The destination pixel footprint is filter_scale wide in source space
    // We need to sample all source pixels that could overlap this footprint
    let box_radius = (0.5 * filter_scale).ceil() as i32 + 1;

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * effective_scale - 0.5 + offset;
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

// =============================================================================
// Tent-Space Kernel Computation
// =============================================================================
//
// Implements the equivalent direct kernel for tent-space downsampling pipeline:
// 1. Box → Tent expansion (with volume-preserving sharpening)
// 2. Resampling with a 1D kernel in tent space
// 3. Tent → Box contraction
//
// The key insight is that this entire pipeline can be collapsed into a single
// set of weights per output pixel, derived analytically.
//
// Example: 2× downsample with box kernel produces [-1, 7, 26, 26, 7, -1]/64

use std::collections::HashMap;

/// Tent expansion coefficients for a corner position (even tent index).
/// Corner C_i at tent position 2*i = (V_{i-1} + V_i) / 2
#[inline]
fn tent_corner_coeffs(box_idx: i32, src_len: usize) -> Vec<(usize, f32)> {
    let mut coeffs = Vec::with_capacity(2);
    let i_minus_1 = box_idx - 1;
    let i = box_idx;

    // Clamp to valid range with edge extension
    let i_minus_1_clamped = i_minus_1.clamp(0, src_len as i32 - 1) as usize;
    let i_clamped = i.clamp(0, src_len as i32 - 1) as usize;

    coeffs.push((i_minus_1_clamped, 0.5));
    coeffs.push((i_clamped, 0.5));
    coeffs
}

/// Tent expansion coefficients for a center position (odd tent index).
/// Center M_i at tent position 2*i+1 = 3/2*V_i - 1/4*V_{i-1} - 1/4*V_{i+1}
#[inline]
fn tent_center_coeffs(box_idx: i32, src_len: usize) -> Vec<(usize, f32)> {
    let mut coeffs = Vec::with_capacity(3);
    let i = box_idx;
    let i_minus_1 = box_idx - 1;
    let i_plus_1 = box_idx + 1;

    // Clamp to valid range with edge extension
    let i_clamped = i.clamp(0, src_len as i32 - 1) as usize;
    let i_minus_1_clamped = i_minus_1.clamp(0, src_len as i32 - 1) as usize;
    let i_plus_1_clamped = i_plus_1.clamp(0, src_len as i32 - 1) as usize;

    coeffs.push((i_minus_1_clamped, -0.25));
    coeffs.push((i_clamped, 1.5));
    coeffs.push((i_plus_1_clamped, -0.25));
    coeffs
}

/// Get tent expansion coefficients for a tent position (integer).
/// Even positions are corners, odd positions are centers.
#[inline]
fn tent_value_coeffs(tent_pos: i32, src_len: usize) -> Vec<(usize, f32)> {
    if tent_pos % 2 == 0 {
        tent_corner_coeffs(tent_pos / 2, src_len)
    } else {
        tent_center_coeffs(tent_pos / 2, src_len)
    }
}

/// Sample the tent surface at a position using box-integrated weighting.
/// Returns coefficients for input pixels that contribute to this sample.
fn sample_tent_surface_box(
    center_pos: f32,
    half_width: f32,
    src_len: usize,
) -> HashMap<usize, f32> {
    let mut combined_coeffs: HashMap<usize, f32> = HashMap::new();

    let interval_start = center_pos - half_width;
    let interval_end = center_pos + half_width;

    // Tent space has positions 0, 1, 2, ... where each owns [pos-0.5, pos+0.5]
    // We need to find tent positions that overlap with [interval_start, interval_end]
    let start_tent = (interval_start + 0.5).floor() as i32 - 1;
    let end_tent = (interval_end + 0.5).ceil() as i32 + 1;

    // Maximum tent position is 2*(src_len-1)+1 = 2*src_len - 1
    let max_tent_pos = (2 * src_len) as i32 - 1;

    let mut weight_sum = 0.0f32;

    for tent_pos in start_tent..=end_tent {
        // Clamp tent position
        let tent_pos_clamped = tent_pos.clamp(0, max_tent_pos);

        // Compute overlap between [interval_start, interval_end] and [tent_pos - 0.5, tent_pos + 0.5]
        let tent_start = tent_pos_clamped as f32 - 0.5;
        let tent_end = tent_pos_clamped as f32 + 0.5;
        let overlap_start = interval_start.max(tent_start);
        let overlap_end = interval_end.min(tent_end);
        let overlap = (overlap_end - overlap_start).max(0.0);

        if overlap < 1e-10 {
            continue;
        }

        // Get input pixel coefficients for this tent position
        let tent_coeffs = tent_value_coeffs(tent_pos_clamped, src_len);

        // Add weighted contributions
        for (idx, coeff) in tent_coeffs {
            *combined_coeffs.entry(idx).or_insert(0.0) += overlap * coeff;
        }
        weight_sum += overlap;
    }

    // Normalize by weight sum
    if weight_sum.abs() > 1e-10 {
        for (_, coeff) in combined_coeffs.iter_mut() {
            *coeff /= weight_sum;
        }
    }

    combined_coeffs
}

/// Sample the tent surface at a position using a general kernel.
/// Returns coefficients for input pixels that contribute to this sample.
fn sample_tent_surface_kernel(
    center_pos: f32,
    half_width: f32,
    kernel_func: fn(f32) -> f32,
    kernel_radius: f32,
    src_len: usize,
) -> HashMap<usize, f32> {
    let mut combined_coeffs: HashMap<usize, f32> = HashMap::new();

    // Find tent positions within kernel radius
    let start_tent = (center_pos - half_width).floor() as i32 - 1;
    let end_tent = (center_pos + half_width).ceil() as i32 + 1;

    // Maximum tent position is 2*(src_len-1)+1 = 2*src_len - 1
    let max_tent_pos = (2 * src_len) as i32 - 1;

    let mut weight_sum = 0.0f32;

    for tent_pos in start_tent..=end_tent {
        // Clamp tent position
        let tent_pos_clamped = tent_pos.clamp(0, max_tent_pos);

        // Map position to kernel's domain
        let kernel_arg = if half_width > 1e-10 {
            (tent_pos_clamped as f32 - center_pos) / half_width * kernel_radius
        } else {
            0.0
        };

        let kernel_weight = kernel_func(kernel_arg);
        if kernel_weight.abs() < 1e-10 {
            continue;
        }

        // Get input pixel coefficients for this tent position
        let tent_coeffs = tent_value_coeffs(tent_pos_clamped, src_len);

        // Add weighted contributions
        for (idx, coeff) in tent_coeffs {
            *combined_coeffs.entry(idx).or_insert(0.0) += kernel_weight * coeff;
        }
        weight_sum += kernel_weight;
    }

    // Normalize by weight sum
    if weight_sum.abs() > 1e-10 {
        for (_, coeff) in combined_coeffs.iter_mut() {
            *coeff /= weight_sum;
        }
    }

    combined_coeffs
}

/// Apply tent contraction weights: 1/4 * left + 1/2 * center + 1/4 * right
fn contract_tent_coeffs(
    left_coeffs: &HashMap<usize, f32>,
    center_coeffs: &HashMap<usize, f32>,
    right_coeffs: &HashMap<usize, f32>,
) -> HashMap<usize, f32> {
    let mut combined: HashMap<usize, f32> = HashMap::new();

    for (&idx, &coeff) in left_coeffs {
        *combined.entry(idx).or_insert(0.0) += 0.25 * coeff;
    }
    for (&idx, &coeff) in center_coeffs {
        *combined.entry(idx).or_insert(0.0) += 0.5 * coeff;
    }
    for (&idx, &coeff) in right_coeffs {
        *combined.entry(idx).or_insert(0.0) += 0.25 * coeff;
    }

    combined
}

/// Precompute tent-space kernel weights for 1D resampling.
///
/// This implements the tent-space pipeline as direct kernel weights:
/// 1. Expansion: box → tent (with volume-preserving sharpening)
/// 2. Resampling: sample tent surface with specified kernel
/// 3. Contraction: tent → box (1/4, 1/2, 1/4 weighting)
///
/// The `inner_method` specifies the resampling kernel used in tent space.
/// For TentBox, use RescaleMethod::Box.
/// For TentLanczos3, use RescaleMethod::Lanczos3.
///
/// The `scale` parameter allows uniform scaling support - pass the desired scale
/// factor (which may differ from src_len/dst_len for uniform scaling modes).
pub fn precompute_tent_kernel_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
    inner_method: RescaleMethod,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Kernel half-width in tent space
    // Following tent_kernel.py: for box, width_tent = ratio, half_width = ratio/2
    // For other kernels: radius * scale
    let (kernel_radius, half_width) = match inner_method {
        RescaleMethod::Box | RescaleMethod::TentBox => {
            // Box filter: width in tent space = scale, half_width = scale/2
            // This matches tent_kernel.py's default: width_box = ratio/2, width_tent = ratio
            (0.5_f32, scale / 2.0)
        }
        RescaleMethod::Lanczos3 | RescaleMethod::TentLanczos3 => (3.0_f32, scale * 3.0 / 2.0),
        RescaleMethod::Lanczos2 => (2.0_f32, scale * 2.0 / 2.0),
        RescaleMethod::Mitchell => (2.0_f32, scale * 2.0 / 2.0),
        RescaleMethod::CatmullRom => (2.0_f32, scale * 2.0 / 2.0),
        _ => (3.0_f32, scale * 3.0 / 2.0), // Default to Lanczos3-like
    };

    // Center offset for standard centering: output pixel 0 centered at input (scale-1)/2
    let offset = (scale - 1.0) / 2.0;

    for dst_i in 0..dst_len {
        // Output pixel center in input (box) space
        let box_center = dst_i as f32 * scale + offset;

        // Map to tent space: box position i → tent position 2*i + 1 (center)
        let tent_center = 2.0 * box_center + 1.0;

        // Spacing in tent space for contraction = scale (matches tent_kernel.py)
        let spacing = scale;

        // Three positions for contraction
        let left_pos = tent_center - spacing;
        let center_pos = tent_center;
        let right_pos = tent_center + spacing;

        // Sample tent surface at each position
        let (left_coeffs, center_coeffs, right_coeffs) = match inner_method {
            RescaleMethod::Box | RescaleMethod::TentBox => {
                // Box uses area-integrated sampling
                (
                    sample_tent_surface_box(left_pos, half_width, src_len),
                    sample_tent_surface_box(center_pos, half_width, src_len),
                    sample_tent_surface_box(right_pos, half_width, src_len),
                )
            }
            _ => {
                // Other kernels use point-weighted sampling
                let kernel_func: fn(f32) -> f32 = match inner_method {
                    RescaleMethod::Lanczos3 | RescaleMethod::TentLanczos3 => lanczos3,
                    RescaleMethod::Lanczos2 => lanczos2,
                    RescaleMethod::Mitchell => mitchell,
                    RescaleMethod::CatmullRom => catmull_rom,
                    _ => lanczos3,
                };
                (
                    sample_tent_surface_kernel(left_pos, half_width, kernel_func, kernel_radius, src_len),
                    sample_tent_surface_kernel(center_pos, half_width, kernel_func, kernel_radius, src_len),
                    sample_tent_surface_kernel(right_pos, half_width, kernel_func, kernel_radius, src_len),
                )
            }
        };

        // Apply contraction
        let contracted = contract_tent_coeffs(&left_coeffs, &center_coeffs, &right_coeffs);

        // Convert to KernelWeights format
        if contracted.is_empty() {
            let fallback = box_center.round().clamp(0.0, (src_len - 1) as f32) as usize;
            all_weights.push(KernelWeights {
                start_idx: fallback,
                weights: vec![1.0],
                fallback_idx: fallback,
            });
        } else {
            // Find the range of indices
            let min_idx = contracted.keys().copied().min().unwrap();
            let max_idx = contracted.keys().copied().max().unwrap();

            // Build contiguous weight array
            let mut weights = vec![0.0f32; max_idx - min_idx + 1];
            for (&idx, &coeff) in &contracted {
                weights[idx - min_idx] = coeff;
            }

            // Normalize weights
            let sum: f32 = weights.iter().sum();
            if sum.abs() > 1e-8 {
                for w in &mut weights {
                    *w /= sum;
                }
            }

            let fallback = box_center.round().clamp(0.0, (src_len - 1) as f32) as usize;
            all_weights.push(KernelWeights {
                start_idx: min_idx,
                weights,
                fallback_idx: fallback,
            });
        }
    }

    all_weights
}

// =============================================================================
// 2D Tent-Space Kernel Computation
// =============================================================================
//
// Implements the equivalent direct 2D kernel for tent-space downsampling pipeline:
// 1. Box → Tent expansion (with volume-preserving sharpening)
// 2. Resampling with a 2D kernel in tent space
// 3. Tent → Box contraction
//
// 2D Tent-space layout:
// - Positions where both indices are even: corners (4-way average)
// - Positions where one index is even, one odd: edges (2-way average)
// - Positions where both indices are odd: centers (volume-preserving)
//
// Integration weights for contraction (3×3 neighborhood):
//     1/16  1/8  1/16
//     1/8   1/4  1/8
//     1/16  1/8  1/16
//
// Example: 2× downsample with 2D box kernel produces 6×6 kernel:
// [-1, -9, -22, -22, -9, -1]
// [-9, 47, 186, 186, 47, -9]
// [-22, 186, 668, 668, 186, -22]
// [-22, 186, 668, 668, 186, -22]
// [-9, 47, 186, 186, 47, -9]
// [-1, -9, -22, -22, -9, -1]
// / 4096

/// 2D tent expansion coefficients for a corner position (both indices even).
/// Corner at tent position (2*y_idx, 2*x_idx) = 4-way average of surrounding pixels.
fn tent_corner_coeffs_2d(y_idx: i32, x_idx: i32, src_h: usize, src_w: usize) -> Vec<((usize, usize), f32)> {
    let mut coeffs = Vec::with_capacity(4);

    // Four contributing pixels (with edge clamping)
    let y0 = (y_idx - 1).clamp(0, src_h as i32 - 1) as usize;
    let y1 = y_idx.clamp(0, src_h as i32 - 1) as usize;
    let x0 = (x_idx - 1).clamp(0, src_w as i32 - 1) as usize;
    let x1 = x_idx.clamp(0, src_w as i32 - 1) as usize;

    coeffs.push(((y0, x0), 0.25));
    coeffs.push(((y0, x1), 0.25));
    coeffs.push(((y1, x0), 0.25));
    coeffs.push(((y1, x1), 0.25));
    coeffs
}

/// 2D tent expansion coefficients for a vertical edge (y odd, x even).
/// Edge at tent position (2*y_idx+1, 2*x_idx) = 2-way horizontal average.
fn tent_vedge_coeffs_2d(y_idx: i32, x_idx: i32, src_h: usize, src_w: usize) -> Vec<((usize, usize), f32)> {
    let mut coeffs = Vec::with_capacity(2);

    let y = y_idx.clamp(0, src_h as i32 - 1) as usize;
    let x0 = (x_idx - 1).clamp(0, src_w as i32 - 1) as usize;
    let x1 = x_idx.clamp(0, src_w as i32 - 1) as usize;

    coeffs.push(((y, x0), 0.5));
    coeffs.push(((y, x1), 0.5));
    coeffs
}

/// 2D tent expansion coefficients for a horizontal edge (y even, x odd).
/// Edge at tent position (2*y_idx, 2*x_idx+1) = 2-way vertical average.
fn tent_hedge_coeffs_2d(y_idx: i32, x_idx: i32, src_h: usize, src_w: usize) -> Vec<((usize, usize), f32)> {
    let mut coeffs = Vec::with_capacity(2);

    let y0 = (y_idx - 1).clamp(0, src_h as i32 - 1) as usize;
    let y1 = y_idx.clamp(0, src_h as i32 - 1) as usize;
    let x = x_idx.clamp(0, src_w as i32 - 1) as usize;

    coeffs.push(((y0, x), 0.5));
    coeffs.push(((y1, x), 0.5));
    coeffs
}

/// 2D tent expansion coefficients for a center position (both indices odd).
/// Center at tent position (2*y_idx+1, 2*x_idx+1) = volume-preserving.
///
/// For 2D volume preservation with integration weights:
///     1/16 * corners + 1/8 * edges + 1/4 * center = V (original pixel value)
///
/// Center M = 4*V - 1/4*(corner_sum) - 1/2*(edge_sum)
fn tent_center_coeffs_2d(y_idx: i32, x_idx: i32, src_h: usize, src_w: usize) -> Vec<((usize, usize), f32)> {
    // Collect all contributions
    let mut coeffs_map: HashMap<(usize, usize), f32> = HashMap::new();

    // The center pixel V_ij itself contributes 4.0
    let y = y_idx.clamp(0, src_h as i32 - 1) as usize;
    let x = x_idx.clamp(0, src_w as i32 - 1) as usize;
    *coeffs_map.entry((y, x)).or_insert(0.0) += 4.0;

    // The 4 corners of the 3×3 tent neighborhood contribute -1/4 each
    // Corners: (2y, 2x), (2y, 2x+2), (2y+2, 2x), (2y+2, 2x+2)
    for corner_coeffs in [
        tent_corner_coeffs_2d(y_idx, x_idx, src_h, src_w),
        tent_corner_coeffs_2d(y_idx, x_idx + 1, src_h, src_w),
        tent_corner_coeffs_2d(y_idx + 1, x_idx, src_h, src_w),
        tent_corner_coeffs_2d(y_idx + 1, x_idx + 1, src_h, src_w),
    ] {
        for ((cy, cx), coeff) in corner_coeffs {
            *coeffs_map.entry((cy, cx)).or_insert(0.0) -= 0.25 * coeff;
        }
    }

    // The 4 edges of the 3×3 tent neighborhood contribute -1/2 each
    // Edges: (2y, 2x+1), (2y+2, 2x+1), (2y+1, 2x), (2y+1, 2x+2)
    for edge_coeffs in [
        tent_hedge_coeffs_2d(y_idx, x_idx, src_h, src_w),     // top edge
        tent_hedge_coeffs_2d(y_idx + 1, x_idx, src_h, src_w), // bottom edge
        tent_vedge_coeffs_2d(y_idx, x_idx, src_h, src_w),     // left edge
        tent_vedge_coeffs_2d(y_idx, x_idx + 1, src_h, src_w), // right edge
    ] {
        for ((ey, ex), coeff) in edge_coeffs {
            *coeffs_map.entry((ey, ex)).or_insert(0.0) -= 0.5 * coeff;
        }
    }

    coeffs_map.into_iter().filter(|(_, c)| c.abs() > 1e-10).collect()
}

/// Get 2D tent expansion coefficients for a tent position (integer coordinates).
fn tent_value_coeffs_2d(tent_y: i32, tent_x: i32, src_h: usize, src_w: usize) -> Vec<((usize, usize), f32)> {
    let y_even = tent_y % 2 == 0;
    let x_even = tent_x % 2 == 0;

    if y_even && x_even {
        // Corner
        tent_corner_coeffs_2d(tent_y / 2, tent_x / 2, src_h, src_w)
    } else if !y_even && x_even {
        // Vertical edge
        tent_vedge_coeffs_2d((tent_y - 1) / 2, tent_x / 2, src_h, src_w)
    } else if y_even && !x_even {
        // Horizontal edge
        tent_hedge_coeffs_2d(tent_y / 2, (tent_x - 1) / 2, src_h, src_w)
    } else {
        // Center
        tent_center_coeffs_2d((tent_y - 1) / 2, (tent_x - 1) / 2, src_h, src_w)
    }
}

/// Sample the 2D tent surface at a position using box-integrated weighting.
/// Returns coefficients for input pixels that contribute to this sample.
fn sample_tent_surface_2d_box(
    center_y: f32,
    center_x: f32,
    half_width: f32,
    src_h: usize,
    src_w: usize,
) -> HashMap<(usize, usize), f32> {
    let mut combined_coeffs: HashMap<(usize, usize), f32> = HashMap::new();

    let interval_y_start = center_y - half_width;
    let interval_y_end = center_y + half_width;
    let interval_x_start = center_x - half_width;
    let interval_x_end = center_x + half_width;

    // Find tent positions that overlap with the interval
    let start_tent_y = (interval_y_start + 0.5).floor() as i32 - 1;
    let end_tent_y = (interval_y_end + 0.5).ceil() as i32 + 1;
    let start_tent_x = (interval_x_start + 0.5).floor() as i32 - 1;
    let end_tent_x = (interval_x_end + 0.5).ceil() as i32 + 1;

    // Maximum tent positions
    let max_tent_y = (2 * src_h) as i32 - 1;
    let max_tent_x = (2 * src_w) as i32 - 1;

    let mut weight_sum = 0.0f32;

    for tent_y in start_tent_y..=end_tent_y {
        let tent_y_clamped = tent_y.clamp(0, max_tent_y);

        // Compute Y overlap
        let tent_y_start = tent_y_clamped as f32 - 0.5;
        let tent_y_end = tent_y_clamped as f32 + 0.5;
        let overlap_y_start = interval_y_start.max(tent_y_start);
        let overlap_y_end = interval_y_end.min(tent_y_end);
        let overlap_y = (overlap_y_end - overlap_y_start).max(0.0);

        if overlap_y < 1e-10 {
            continue;
        }

        for tent_x in start_tent_x..=end_tent_x {
            let tent_x_clamped = tent_x.clamp(0, max_tent_x);

            // Compute X overlap
            let tent_x_start = tent_x_clamped as f32 - 0.5;
            let tent_x_end = tent_x_clamped as f32 + 0.5;
            let overlap_x_start = interval_x_start.max(tent_x_start);
            let overlap_x_end = interval_x_end.min(tent_x_end);
            let overlap_x = (overlap_x_end - overlap_x_start).max(0.0);

            if overlap_x < 1e-10 {
                continue;
            }

            // 2D overlap area
            let overlap_area = overlap_y * overlap_x;

            // Get input pixel coefficients for this tent position
            let tent_coeffs = tent_value_coeffs_2d(tent_y_clamped, tent_x_clamped, src_h, src_w);

            // Add weighted contributions
            for ((y, x), coeff) in tent_coeffs {
                *combined_coeffs.entry((y, x)).or_insert(0.0) += overlap_area * coeff;
            }
            weight_sum += overlap_area;
        }
    }

    // Normalize by weight sum
    if weight_sum.abs() > 1e-10 {
        for (_, coeff) in combined_coeffs.iter_mut() {
            *coeff /= weight_sum;
        }
    }

    combined_coeffs
}

/// Sample the 2D tent surface using an EWA (radial) kernel.
/// Returns coefficients for input pixels that contribute to this sample.
fn sample_tent_surface_2d_ewa(
    center_y: f32,
    center_x: f32,
    half_width: f32,
    kernel_radius: f32,
    src_h: usize,
    src_w: usize,
) -> HashMap<(usize, usize), f32> {
    let mut combined_coeffs: HashMap<(usize, usize), f32> = HashMap::new();

    // Find tent positions within kernel radius
    let start_tent_y = (center_y - half_width).floor() as i32 - 1;
    let end_tent_y = (center_y + half_width).ceil() as i32 + 1;
    let start_tent_x = (center_x - half_width).floor() as i32 - 1;
    let end_tent_x = (center_x + half_width).ceil() as i32 + 1;

    // Maximum tent positions
    let max_tent_y = (2 * src_h) as i32 - 1;
    let max_tent_x = (2 * src_w) as i32 - 1;

    let mut weight_sum = 0.0f32;

    for tent_y in start_tent_y..=end_tent_y {
        let tent_y_clamped = tent_y.clamp(0, max_tent_y);

        for tent_x in start_tent_x..=end_tent_x {
            let tent_x_clamped = tent_x.clamp(0, max_tent_x);

            // Map position to kernel's domain (radial distance)
            let ky = if half_width > 1e-10 {
                (tent_y_clamped as f32 - center_y) / half_width * kernel_radius
            } else {
                0.0
            };
            let kx = if half_width > 1e-10 {
                (tent_x_clamped as f32 - center_x) / half_width * kernel_radius
            } else {
                0.0
            };
            let r = (ky * ky + kx * kx).sqrt();

            // EWA Lanczos3 with jinc
            let kernel_weight = ewa_lanczos(r, kernel_radius);

            if kernel_weight.abs() < 1e-10 {
                continue;
            }

            // Get input pixel coefficients for this tent position
            let tent_coeffs = tent_value_coeffs_2d(tent_y_clamped, tent_x_clamped, src_h, src_w);

            // Add weighted contributions
            for ((y, x), coeff) in tent_coeffs {
                *combined_coeffs.entry((y, x)).or_insert(0.0) += kernel_weight * coeff;
            }
            weight_sum += kernel_weight;
        }
    }

    // Normalize by weight sum
    if weight_sum.abs() > 1e-10 {
        for (_, coeff) in combined_coeffs.iter_mut() {
            *coeff /= weight_sum;
        }
    }

    combined_coeffs
}

/// Apply 2D tent contraction weights to the 3×3 neighborhood of samples.
/// Weights: corners=1/16, edges=1/8, center=1/4
fn contract_tent_coeffs_2d(
    samples: &[HashMap<(usize, usize), f32>; 9],
) -> HashMap<(usize, usize), f32> {
    let mut combined: HashMap<(usize, usize), f32> = HashMap::new();

    // Contraction weights for 3×3 neighborhood:
    // [0] 1/16  [1] 1/8  [2] 1/16
    // [3] 1/8   [4] 1/4  [5] 1/8
    // [6] 1/16  [7] 1/8  [8] 1/16
    let weights = [
        1.0/16.0, 1.0/8.0, 1.0/16.0,
        1.0/8.0,  1.0/4.0, 1.0/8.0,
        1.0/16.0, 1.0/8.0, 1.0/16.0,
    ];

    for (i, sample) in samples.iter().enumerate() {
        let w = weights[i];
        for (&(y, x), &coeff) in sample {
            *combined.entry((y, x)).or_insert(0.0) += w * coeff;
        }
    }

    combined
}

/// Precompute 2D tent-space kernel weights for a single output pixel.
///
/// This implements the 2D tent-space pipeline as direct kernel weights:
/// 1. Expansion: box → tent (with volume-preserving sharpening)
/// 2. Resampling: sample 2D tent surface with specified kernel (box or EWA)
/// 3. Contraction: tent → box (3×3 integration weights)
pub fn precompute_tent_2d_kernel_weights_single(
    dst_y: usize,
    dst_x: usize,
    src_h: usize,
    src_w: usize,
    scale_y: f32,
    scale_x: f32,
    is_ewa: bool,
) -> KernelWeights2D {
    // Output pixel center in input (box) space
    let offset_y = (scale_y - 1.0) / 2.0;
    let offset_x = (scale_x - 1.0) / 2.0;
    let box_center_y = dst_y as f32 * scale_y + offset_y;
    let box_center_x = dst_x as f32 * scale_x + offset_x;

    // Map to tent space: box position i → tent position 2*i + 1 (center)
    let tent_center_y = 2.0 * box_center_y + 1.0;
    let tent_center_x = 2.0 * box_center_x + 1.0;

    // Spacing in tent space for contraction = scale
    let spacing_y = scale_y;
    let spacing_x = scale_x;

    // Kernel half-width in tent space
    let (kernel_radius, half_width_y, half_width_x) = if is_ewa {
        // EWA Lanczos3: radius = 3.0, half_width = scale * radius / 2
        (3.0_f32, scale_y * 3.0 / 2.0, scale_x * 3.0 / 2.0)
    } else {
        // Box: half_width = scale / 2
        (0.5_f32, scale_y / 2.0, scale_x / 2.0)
    };

    // Use the larger half-width for symmetric sampling (could use asymmetric for non-uniform)
    let half_width = half_width_y.max(half_width_x);

    // 9 positions for 3×3 contraction neighborhood
    let positions: [(f32, f32); 9] = [
        (tent_center_y - spacing_y, tent_center_x - spacing_x), // TL
        (tent_center_y - spacing_y, tent_center_x),             // T
        (tent_center_y - spacing_y, tent_center_x + spacing_x), // TR
        (tent_center_y, tent_center_x - spacing_x),             // L
        (tent_center_y, tent_center_x),                         // C
        (tent_center_y, tent_center_x + spacing_x),             // R
        (tent_center_y + spacing_y, tent_center_x - spacing_x), // BL
        (tent_center_y + spacing_y, tent_center_x),             // B
        (tent_center_y + spacing_y, tent_center_x + spacing_x), // BR
    ];

    // Sample tent surface at each position
    let samples: [HashMap<(usize, usize), f32>; 9] = if is_ewa {
        [
            sample_tent_surface_2d_ewa(positions[0].0, positions[0].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[1].0, positions[1].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[2].0, positions[2].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[3].0, positions[3].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[4].0, positions[4].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[5].0, positions[5].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[6].0, positions[6].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[7].0, positions[7].1, half_width, kernel_radius, src_h, src_w),
            sample_tent_surface_2d_ewa(positions[8].0, positions[8].1, half_width, kernel_radius, src_h, src_w),
        ]
    } else {
        [
            sample_tent_surface_2d_box(positions[0].0, positions[0].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[1].0, positions[1].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[2].0, positions[2].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[3].0, positions[3].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[4].0, positions[4].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[5].0, positions[5].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[6].0, positions[6].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[7].0, positions[7].1, half_width, src_h, src_w),
            sample_tent_surface_2d_box(positions[8].0, positions[8].1, half_width, src_h, src_w),
        ]
    };

    // Apply contraction
    let contracted = contract_tent_coeffs_2d(&samples);

    // Convert to KernelWeights2D format
    if contracted.is_empty() {
        let fallback_y = box_center_y.round().clamp(0.0, (src_h - 1) as f32) as usize;
        let fallback_x = box_center_x.round().clamp(0.0, (src_w - 1) as f32) as usize;
        return KernelWeights2D {
            start_y: fallback_y,
            start_x: fallback_x,
            height: 1,
            width: 1,
            weights: vec![1.0],
            fallback_y,
            fallback_x,
        };
    }

    // Find the bounding box of non-zero weights
    let min_y = contracted.keys().map(|(y, _)| *y).min().unwrap();
    let max_y = contracted.keys().map(|(y, _)| *y).max().unwrap();
    let min_x = contracted.keys().map(|(_, x)| *x).min().unwrap();
    let max_x = contracted.keys().map(|(_, x)| *x).max().unwrap();

    let height = max_y - min_y + 1;
    let width = max_x - min_x + 1;

    // Build 2D weight array
    let mut weights = vec![0.0f32; height * width];
    for (&(y, x), &coeff) in &contracted {
        let idx = (y - min_y) * width + (x - min_x);
        weights[idx] = coeff;
    }

    // Normalize weights
    let sum: f32 = weights.iter().sum();
    if sum.abs() > 1e-8 {
        for w in &mut weights {
            *w /= sum;
        }
    }

    let fallback_y = box_center_y.round().clamp(0.0, (src_h - 1) as f32) as usize;
    let fallback_x = box_center_x.round().clamp(0.0, (src_w - 1) as f32) as usize;

    KernelWeights2D {
        start_y: min_y,
        start_x: min_x,
        height,
        width,
        weights,
        fallback_y,
        fallback_x,
    }
}

/// Precompute all 2D tent-space kernel weights for the entire output image.
/// Returns a 2D vector indexed by [dst_y][dst_x].
pub fn precompute_tent_2d_kernel_weights(
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    scale_y: f32,
    scale_x: f32,
    is_ewa: bool,
) -> Vec<Vec<KernelWeights2D>> {
    let mut all_weights = Vec::with_capacity(dst_h);

    for dst_y in 0..dst_h {
        let mut row_weights = Vec::with_capacity(dst_w);
        for dst_x in 0..dst_w {
            row_weights.push(precompute_tent_2d_kernel_weights_single(
                dst_y, dst_x, src_h, src_w, scale_y, scale_x, is_ewa
            ));
        }
        all_weights.push(row_weights);
    }

    all_weights
}
