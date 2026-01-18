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
        RescaleMethod::Mitchell => mitchell(x),
        RescaleMethod::CatmullRom => catmull_rom(x),
        RescaleMethod::Lanczos2 | RescaleMethod::EWALanczos2 => lanczos2(x),
        RescaleMethod::Lanczos3 | RescaleMethod::Lanczos3Scatter | RescaleMethod::EWALanczos3 => lanczos3(x),
        RescaleMethod::Sinc | RescaleMethod::SincScatter => sinc(x),
        RescaleMethod::Box => box_filter(x),
        // PeakedCosine uses its own specialized precomputation with scale-dependent parameters
        RescaleMethod::PeakedCosine | RescaleMethod::PeakedCosineCorrected => sinc(x), // Fallback; actual implementation uses peaked_cosine_sinc
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

// ============================================================================
// Peaked Cosine (AVIR-style) Windowed Sinc
// ============================================================================

/// AVIR-style Peaked Cosine window function parameters for lowpass filtering
pub struct PeakedCosineParams {
    /// Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
    pub cutoff: f32,
    /// Filter half-length in samples
    pub half_length: f32,
    /// Alpha shape parameter for the window
    pub alpha: f32,
}

/// Default AVIR lowpass filter parameters
const AVIR_LPF_BASE_LEN: f32 = 7.56;
const AVIR_LPF_CUTOFF_MULT: f32 = 0.79285;
const AVIR_LPF_ALPHA: f32 = 4.76449;

/// Default AVIR interpolation filter parameters (for upscaling)
const AVIR_INT_LEN: f32 = 18.0;
const AVIR_INT_CUTOFF: f32 = 0.7372;
const AVIR_INT_ALPHA: f32 = 6.41341;

/// Calculate Peaked Cosine parameters based on scale factor
///
/// For downscaling (scale > 1): uses lowpass filter parameters
/// For upscaling (scale < 1): uses interpolation filter parameters
pub fn calculate_peaked_cosine_params(scale: f32) -> PeakedCosineParams {
    if scale > 1.0 {
        // Downscaling: use lowpass filter
        let cutoff = AVIR_LPF_CUTOFF_MULT / scale;
        let half_length = (AVIR_LPF_BASE_LEN / 2.0) / cutoff;
        PeakedCosineParams {
            cutoff,
            half_length,
            alpha: AVIR_LPF_ALPHA,
        }
    } else {
        // Upscaling: use interpolation filter
        // For upscaling, the filter is applied at output rate
        // Scale the half_length by the inverse scale factor
        let half_length = (AVIR_INT_LEN / 2.0) * scale;
        PeakedCosineParams {
            cutoff: AVIR_INT_CUTOFF,
            half_length: half_length.max(2.0), // Minimum reasonable filter size
            alpha: AVIR_INT_ALPHA,
        }
    }
}

/// Peaked Cosine window function
/// w(n) = cos(π * n / (2 * L)) * (1 - (n / L)^α)
///
/// Returns 0 if n >= L (outside window support)
#[inline]
pub fn peaked_cosine_window(n: f32, half_length: f32, alpha: f32) -> f32 {
    if n.abs() >= half_length {
        return 0.0;
    }

    let t = n / half_length;
    let cos_term = (PI * n / (2.0 * half_length)).cos();
    let shape_term = 1.0 - t.abs().powf(alpha);

    cos_term * shape_term
}

/// Windowed sinc kernel with Peaked Cosine window
/// h(n) = sinc(n * fc) * w(n)
#[inline]
pub fn peaked_cosine_sinc(n: f32, params: &PeakedCosineParams) -> f32 {
    let sinc_val = if n.abs() < 1e-8 {
        1.0
    } else {
        let x = PI * n * params.cutoff;
        x.sin() / x
    };

    sinc_val * peaked_cosine_window(n, params.half_length, params.alpha)
}

/// Precompute Peaked Cosine filter weights for 1D resampling
///
/// Unlike fixed-kernel methods, the filter parameters depend on the scale factor,
/// making this filter adaptive to the scaling operation.
pub fn precompute_peaked_cosine_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
) -> Vec<KernelWeights> {
    let mut all_weights = Vec::with_capacity(dst_len);

    // Calculate filter parameters based on scale
    let params = calculate_peaked_cosine_params(scale);

    // The filter radius in source pixels
    let radius = params.half_length.ceil() as i32;

    // Center offset for uniform scaling
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    for dst_i in 0..dst_len {
        // Map destination pixel center to source coordinates
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
        let center = src_pos.floor() as i32;

        // Find the valid source index range
        let start = (center - radius).max(0) as usize;
        let end = ((center + radius) as usize).min(src_len - 1);

        // Collect weights
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            // Distance from destination position in source space
            let d = src_pos - si as f32;
            let weight = peaked_cosine_sinc(d, &params);
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

// ============================================================================
// Peaked Cosine with Frequency Response Correction
// ============================================================================

/// Number of frequency bins for measuring response (AVIR uses 65)
const CORRECTION_NUM_BINS: usize = 65;

/// Correction filter length (AVIR uses 5.5-8 taps; 7 is the sweet spot)
const CORRECTION_FILTER_LEN: usize = 7;

/// Alpha for the correction filter window
const CORRECTION_ALPHA: f32 = 0.98;

/// Measure the frequency response of a filter at specified frequency bins
///
/// Returns array of magnitude responses from DC to the target bandwidth.
/// The filter coefficients are centered (index 0 is the center tap).
fn measure_frequency_response(
    weights: &[f32],
    num_bins: usize,
    bandwidth: f32,
) -> Vec<f32> {
    let mut response = Vec::with_capacity(num_bins);
    let half_len = weights.len() / 2;

    for i in 0..num_bins {
        let freq = (i as f32 / (num_bins - 1) as f32) * bandwidth * PI;

        let mut re = 0.0f32;
        let mut im = 0.0f32;

        for (j, &w) in weights.iter().enumerate() {
            let n = j as f32 - half_len as f32;
            re += w * (freq * n).cos();
            im += w * (freq * n).sin();
        }

        let magnitude = (re * re + im * im).sqrt();
        response.push(magnitude);
    }

    response
}

/// Design a short FIR correction filter using least-squares to match target gains
///
/// Uses a simplified least-squares approach with Peaked Cosine windowing.
/// The target gains should be the inverse of the measured frequency response
/// (normalized to achieve flat passband).
fn design_correction_filter(target_gains: &[f32]) -> Vec<f32> {
    let num_bins = target_gains.len();
    let half_len = CORRECTION_FILTER_LEN / 2;
    let filter_len = 2 * half_len + 1;

    // Build frequency points
    let freqs: Vec<f32> = (0..num_bins)
        .map(|i| (i as f32 / (num_bins - 1) as f32) * PI)
        .collect();

    // Build matrix: A[i][j] = cos(freq[i] * n[j]) where n = [-half_len, ..., half_len]
    // Then solve A * coeffs = target_gains in least-squares sense

    // Simple least-squares via normal equations: (A^T * A) * x = A^T * b
    // For a symmetric filter, we only need to compute the cosine terms

    let mut ata = vec![vec![0.0f32; filter_len]; filter_len];
    let mut atb = vec![0.0f32; filter_len];

    for (i, &freq) in freqs.iter().enumerate() {
        // Compute row of A
        let mut row = Vec::with_capacity(filter_len);
        for j in 0..filter_len {
            let n = j as f32 - half_len as f32;
            row.push((freq * n).cos());
        }

        // Accumulate A^T * A
        for j in 0..filter_len {
            for k in 0..filter_len {
                ata[j][k] += row[j] * row[k];
            }
        }

        // Accumulate A^T * b
        for j in 0..filter_len {
            atb[j] += row[j] * target_gains[i];
        }
    }

    // Solve using simple Gaussian elimination with partial pivoting
    let mut coeffs = solve_linear_system(&mut ata, &mut atb);

    // Enforce symmetry (the least-squares problem has redundant columns due to cos(-x)=cos(x),
    // so the solver might return an asymmetric solution which causes phase shift)
    for j in 0..half_len {
        let avg = (coeffs[j] + coeffs[filter_len - 1 - j]) / 2.0;
        coeffs[j] = avg;
        coeffs[filter_len - 1 - j] = avg;
    }

    // Apply Peaked Cosine window
    let window_l = half_len as f32 + 0.5;
    let mut windowed_coeffs = Vec::with_capacity(filter_len);
    for j in 0..filter_len {
        let n = (j as f32 - half_len as f32).abs();
        let window = peaked_cosine_window(n, window_l, CORRECTION_ALPHA);
        windowed_coeffs.push(coeffs[j] * window);
    }

    // Normalize to sum to 1
    let sum: f32 = windowed_coeffs.iter().sum();
    if sum.abs() > 1e-8 {
        for c in &mut windowed_coeffs {
            *c /= sum;
        }
    }

    windowed_coeffs
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &mut [Vec<f32>], b: &mut [f32]) -> Vec<f32> {
    let n = b.len();

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_val = a[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        // Check for singularity
        if a[i][i].abs() < 1e-10 {
            continue;
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-10 {
            x[i] = 0.0;
            continue;
        }
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }

    x
}

/// Convolve two 1D kernels
fn convolve_kernels(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let result_len = a.len() + b.len() - 1;
    let mut result = vec![0.0f32; result_len];

    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i + j] += av * bv;
        }
    }

    result
}

/// Precompute Peaked Cosine filter weights with frequency response correction
///
/// This builds the primary Peaked Cosine filter, measures its frequency response,
/// designs a correction filter to flatten the passband, and convolves them together.
pub fn precompute_peaked_cosine_corrected_weights(
    src_len: usize,
    dst_len: usize,
    scale: f32,
) -> Vec<KernelWeights> {
    // Calculate filter parameters
    let params = calculate_peaked_cosine_params(scale);
    let primary_radius = params.half_length.ceil() as i32;

    // Build the primary filter kernel (centered, for frequency analysis)
    let primary_len = 2 * primary_radius as usize + 1;
    let mut primary_kernel = Vec::with_capacity(primary_len);
    for i in 0..primary_len {
        let n = i as f32 - primary_radius as f32;
        primary_kernel.push(peaked_cosine_sinc(n, &params));
    }

    // Normalize primary kernel
    let primary_sum: f32 = primary_kernel.iter().sum();
    if primary_sum.abs() > 1e-8 {
        for w in &mut primary_kernel {
            *w /= primary_sum;
        }
    }

    // Measure frequency response
    // Bandwidth is relative to Nyquist: for downscaling, bandwidth = 1/scale
    let bandwidth = if scale > 1.0 { 1.0 / scale } else { 1.0 };
    let response = measure_frequency_response(&primary_kernel, CORRECTION_NUM_BINS, bandwidth);

    // Calculate target gains (inverse of measured response, clamped)
    // This aims to flatten the passband
    let mut target_gains = Vec::with_capacity(CORRECTION_NUM_BINS);
    for r in &response {
        // Clamp to avoid extreme correction
        let inv = if *r > 0.1 { 1.0 / r } else { 10.0 };
        // Smooth attenuation at band edges
        target_gains.push(inv.min(2.0));
    }

    // Design correction filter
    let correction_kernel = design_correction_filter(&target_gains);

    // Convolve primary and correction kernels
    let combined_kernel = convolve_kernels(&primary_kernel, &correction_kernel);
    let combined_half_len = combined_kernel.len() / 2;

    // Now precompute weights using the combined kernel
    let mut all_weights = Vec::with_capacity(dst_len);

    // Center offset for uniform scaling
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    let combined_radius = combined_half_len as i32;

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
        let center = src_pos.floor() as i32;

        // Find the valid source index range
        let start = (center - combined_radius).max(0) as usize;
        let end = ((center + combined_radius) as usize).min(src_len - 1);

        // Collect weights by sampling the combined kernel
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            let d = src_pos - si as f32;
            // Sample the combined kernel (interpolating between taps)
            let kernel_pos = d + combined_half_len as f32;

            let weight = if kernel_pos < 0.0 || kernel_pos > combined_kernel.len() as f32 - 1.0 {
                // Outside kernel support
                0.0
            } else {
                let kernel_idx = kernel_pos.floor() as usize;
                let frac = kernel_pos - kernel_idx as f32;

                if kernel_idx >= combined_kernel.len() - 1 {
                    // At the last index, can't interpolate right
                    combined_kernel[kernel_idx]
                } else {
                    // Normal linear interpolation
                    let w0 = combined_kernel[kernel_idx];
                    let w1 = combined_kernel[kernel_idx + 1];
                    w0 * (1.0 - frac) + w1 * frac
                }
            };

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
