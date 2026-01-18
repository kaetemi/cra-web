//! Variance-adaptive rescaling with energy-conserving error scatter
//!
//! This algorithm adapts kernel sharpness based on local image variance:
//! - Low variance (smooth regions): use gentler kernels (Lanczos2) to minimize ringing
//! - High variance (edges): use sharper kernels (Lanczos6) where ringing is hidden
//!
//! The algorithm tracks energy conservation and scatters any error back to the image,
//! weighted by destination variance so error flows to high-variance regions.
//!
//! ## Algorithm Overview
//!
//! 1. **Variance calculation**: Compute inbetween variance (|pixel[i+1] - pixel[i]|),
//!    then per-pixel variance using RMS of neighboring edge variances.
//!
//! 2. **Variance rescaling**: Rescale variance buffer using Catmull-Rom to get
//!    per-destination-pixel variance values.
//!
//! 3. **Adaptive gather**: For each destination pixel, select Lanczos2-6 based on
//!    local variance + wang hash for probabilistic interpolation between adjacent kernels.
//!    Simultaneously track how much each source pixel contributed.
//!
//! 4. **Error calculation**: error[i] = source[i] * (1.0 - contribution[i])
//!
//! 5. **Error scatter**: Scatter error using Lanczos7 window weighted by dest variance,
//!    normalized to 1 for energy conservation.

use std::f32::consts::PI;
use crate::pixel::Pixel4;
use crate::color::linear_rgb_to_oklab;
use super::{ScaleMode, calculate_scales};
use super::kernels::{wang_hash, lanczos2, lanczos3, lanczos4, lanczos5, lanczos6, catmull_rom};

// ============================================================================
// Lanczos7 window for error scatter (wide, smooth, non-negative in support)
// ============================================================================

/// Lanczos7 window function: sinc(x/7) for |x| < 7
/// This is just the window part, NOT the full Lanczos kernel (sinc(x) * sinc(x/7)).
/// The window is smooth, wide, and non-negative within its support, making it
/// ideal for error scatter where we don't want to introduce new artifacts.
#[inline]
fn lanczos7_window(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else if x.abs() >= 7.0 {
        0.0
    } else {
        // Just the window: sinc(x/7) = sin(pi*x/7) / (pi*x/7)
        let pi_x_7 = PI * x / 7.0;
        pi_x_7.sin() / pi_x_7
    }
}

// ============================================================================
// Variance calculation
// ============================================================================

/// Calculate inbetween variance for a row of pixels using OkLab distance
/// Returns N-1 values for N pixels, representing perceptual distance between adjacent pixels.
///
/// Uses OkLab color space for perceptually uniform distance measurement.
/// Normalized so that black-to-white edge ≈ 1.0 variance.
fn calculate_inbetween_variance(row: &[Pixel4]) -> Vec<f32> {
    if row.len() < 2 {
        return vec![];
    }

    let mut variances = Vec::with_capacity(row.len() - 1);
    for i in 0..row.len() - 1 {
        let p1 = row[i];
        let p2 = row[i + 1];

        // Convert to OkLab for perceptually uniform distance
        let (l1, a1, b1) = linear_rgb_to_oklab(p1.r(), p1.g(), p1.b());
        let (l2, a2, b2) = linear_rgb_to_oklab(p2.r(), p2.g(), p2.b());

        // Euclidean distance in OkLab space
        let dl = l2 - l1;
        let da = a2 - a1;
        let db = b2 - b1;
        let diff = (dl * dl + da * da + db * db).sqrt();

        // Normalize: black (0,0,0) to white (1,1,1) in OkLab is distance 1.0
        // (both have a=b=0, only L differs from 0 to 1)
        // So OkLab distance is already normalized for luminance edges.
        // Saturated color differences can exceed 1.0, which is fine - they're high variance.
        variances.push(diff);
    }

    variances
}

/// Calculate per-pixel variance using RMS of left and right inbetween variances
/// Returns N values for N pixels
fn calculate_pixel_variance(inbetween: &[f32], num_pixels: usize) -> Vec<f32> {
    if num_pixels == 0 {
        return vec![];
    }
    if num_pixels == 1 {
        return vec![0.0];
    }

    let mut pixel_var = Vec::with_capacity(num_pixels);

    for i in 0..num_pixels {
        let var = if i == 0 {
            // First pixel: only has right neighbor
            inbetween[0]
        } else if i == num_pixels - 1 {
            // Last pixel: only has left neighbor
            inbetween[inbetween.len() - 1]
        } else {
            // Middle pixels: RMS of left and right
            let left = inbetween[i - 1];
            let right = inbetween[i];
            ((left * left + right * right) / 2.0).sqrt()
        };
        pixel_var.push(var);
    }

    pixel_var
}

// ============================================================================
// Catmull-Rom variance rescaling
// ============================================================================

/// Rescale variance buffer using Catmull-Rom interpolation
/// Catmull-Rom is chosen for smooth, non-ringing interpolation
fn rescale_variance_catmull_rom(
    src_var: &[f32],
    dst_len: usize,
) -> Vec<f32> {
    if dst_len == 0 {
        return vec![];
    }

    let src_len = src_var.len();
    if src_len == 0 {
        return vec![0.0; dst_len];
    }
    if src_len == 1 {
        return vec![src_var[0]; dst_len];
    }

    let scale = src_len as f32 / dst_len as f32;
    let mut dst_var = Vec::with_capacity(dst_len);

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5;
        let center = src_pos.floor() as i32;

        // Catmull-Rom uses 4 samples: center-1, center, center+1, center+2
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for offset in -1..=2i32 {
            let si = center + offset;
            if si >= 0 && si < src_len as i32 {
                let d = src_pos - si as f32;
                let weight = catmull_rom(d);
                sum += weight * src_var[si as usize];
                weight_sum += weight;
            }
        }

        // Normalize and clamp to non-negative (variance can't be negative)
        let var = if weight_sum.abs() > 1e-8 {
            (sum / weight_sum).max(0.0)
        } else {
            src_var[src_pos.round().clamp(0.0, (src_len - 1) as f32) as usize]
        };

        dst_var.push(var);
    }

    dst_var
}

// ============================================================================
// Kernel selection based on variance
// ============================================================================

/// Kernel thresholds for variance-based selection
/// Variance 0.0 -> Lanczos2, 1.0 -> Lanczos6
/// Intermediate values interpolate between adjacent kernels using wang hash
const VARIANCE_THRESHOLDS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Select kernel index (2-6) based on variance and wang hash
/// Returns (kernel_lobe_count, interpolation_weight) where weight is for probabilistic selection
#[inline]
fn select_kernel_for_variance(variance: f32, pixel_hash: u32) -> usize {
    // Normalize variance to [0, 1] range
    // Assuming max meaningful variance is around 1.0 (full black to white edge)
    // Values > 1.0 just use Lanczos6
    let norm_var = variance.clamp(0.0, 1.0);

    // Find which threshold range we're in
    let mut kernel_idx = 0usize; // Will be 0-4, mapping to Lanczos 2-6
    for i in 0..4 {
        if norm_var >= VARIANCE_THRESHOLDS[i] && norm_var < VARIANCE_THRESHOLDS[i + 1] {
            // Interpolate between kernel i and i+1
            let t = (norm_var - VARIANCE_THRESHOLDS[i]) / (VARIANCE_THRESHOLDS[i + 1] - VARIANCE_THRESHOLDS[i]);
            // Use wang hash to probabilistically select between the two kernels
            let hash_float = (pixel_hash & 0xFFFF) as f32 / 65535.0;
            kernel_idx = if hash_float < t { i + 1 } else { i };
            break;
        }
        kernel_idx = i + 1;
    }

    // kernel_idx 0-4 maps to Lanczos 2-6
    kernel_idx.min(4) + 2
}

/// Evaluate Lanczos kernel with given lobe count
#[inline]
fn eval_lanczos(x: f32, lobes: usize) -> f32 {
    match lobes {
        2 => lanczos2(x),
        3 => lanczos3(x),
        4 => lanczos4(x),
        5 => lanczos5(x),
        6 => lanczos6(x),
        _ => lanczos3(x), // fallback
    }
}

// ============================================================================
// 1D variance-adaptive resample with contribution tracking
// ============================================================================

/// Result of 1D variance-adaptive resample
struct ResampleResult {
    /// Resampled pixel values
    pixels: Vec<Pixel4>,
    /// Contribution sum for each source pixel (how much it was sampled)
    contributions: Vec<f32>,
}

/// Variance-adaptive 1D resample with contribution tracking
fn resample_row_variance_adaptive(
    src: &[Pixel4],
    src_var: &[f32],
    dst_len: usize,
    scale: f32,
    seed: u32,
    row_idx: usize,
) -> ResampleResult {
    let src_len = src.len();
    let filter_scale = scale.max(1.0);

    // Rescale variance to destination space using Catmull-Rom
    let dst_var = rescale_variance_catmull_rom(src_var, dst_len);

    // Maximum radius needed (Lanczos6) - used implicitly via kernel_lobes selection
    let _max_radius = (6.0 * filter_scale).ceil() as i32;

    // Center offset for mapping
    let mapped_src_len = dst_len as f32 * scale;
    let offset = (src_len as f32 - mapped_src_len) / 2.0;

    let mut dst_pixels = Vec::with_capacity(dst_len);
    let mut contributions = vec![0.0f32; src_len];

    for dst_i in 0..dst_len {
        let src_pos = (dst_i as f32 + 0.5) * scale - 0.5 + offset;
        let center = src_pos.floor() as i32;

        // Get variance at this destination pixel
        let variance = dst_var[dst_i];

        // Select kernel based on variance and pixel-specific hash
        let pixel_hash = wang_hash((dst_i as u32) ^ ((row_idx as u32) << 16) ^ seed);
        let kernel_lobes = select_kernel_for_variance(variance, pixel_hash);
        let kernel_radius = kernel_lobes as f32;
        let effective_radius = (kernel_radius * filter_scale).ceil() as i32;

        // Sample range
        let start = (center - effective_radius).max(0) as usize;
        let end = ((center + effective_radius) as usize).min(src_len - 1);

        let mut sum = Pixel4::default();
        let mut weight_sum = 0.0f32;

        for si in start..=end {
            let d = (src_pos - si as f32) / filter_scale;
            if d.abs() > kernel_radius {
                continue;
            }

            let weight = eval_lanczos(d, kernel_lobes);
            sum = sum + src[si] * weight;
            weight_sum += weight;

            // Track contribution (unnormalized - we'll normalize later)
            contributions[si] += weight;
        }

        // Normalize output pixel
        if weight_sum.abs() > 1e-8 {
            dst_pixels.push(sum * (1.0 / weight_sum));
            // Note: contributions are accumulated as raw weights, then normalized
            // globally after all destination pixels are processed
        } else {
            let fallback = src_pos.round().clamp(0.0, (src_len - 1) as f32) as usize;
            dst_pixels.push(src[fallback]);
            contributions[fallback] += 1.0;
        }
    }

    // Normalize contributions: each source pixel should have total contribution
    // equal to (dst_len / src_len) for perfect energy conservation
    // Actually, we want contribution[i] to represent "how much of source[i] was used"
    // If contribution[i] == 1.0, source[i] was fully accounted for
    // We need to scale by the expected contribution which is dst_len / src_len
    let expected_total = dst_len as f32 / src_len as f32;
    for c in &mut contributions {
        *c /= expected_total.max(1e-8);
    }

    ResampleResult {
        pixels: dst_pixels,
        contributions,
    }
}

// ============================================================================
// Error calculation and scatter
// ============================================================================

/// Calculate error per source pixel
/// error[i] = source[i] * (1.0 - contribution[i])
fn calculate_error(src: &[Pixel4], contributions: &[f32]) -> Vec<Pixel4> {
    src.iter()
        .zip(contributions.iter())
        .map(|(pixel, &contrib)| {
            let error_factor = 1.0 - contrib;
            Pixel4::new(
                pixel.r() * error_factor,
                pixel.g() * error_factor,
                pixel.b() * error_factor,
                0.0, // Alpha is not part of error calculation
            )
        })
        .collect()
}

/// Scatter error from source pixels to destination pixels
/// Uses Lanczos7 window weighted by destination variance
fn scatter_error(
    dst: &mut [Pixel4],
    error: &[Pixel4],
    dst_var: &[f32],
    scale: f32,
) {
    let src_len = error.len();
    let dst_len = dst.len();

    if src_len == 0 || dst_len == 0 {
        return;
    }

    let filter_scale = scale.max(1.0);
    let scatter_radius = (7.0 * filter_scale).ceil() as i32;

    // Inverse scale for scatter: map source positions to destination
    let inv_scale = dst_len as f32 / src_len as f32;
    let mapped_dst_len = src_len as f32 * inv_scale;
    let offset = (dst_len as f32 - mapped_dst_len) / 2.0;

    // For each source pixel with error, scatter to destination
    for (si, err) in error.iter().enumerate() {
        // Skip if error is negligible
        let err_mag = err.r().abs() + err.g().abs() + err.b().abs();
        if err_mag < 1e-8 {
            continue;
        }

        // Map source position to destination space
        let dst_pos = (si as f32 + 0.5) * inv_scale - 0.5 + offset;
        let center = dst_pos.floor() as i32;

        let start = (center - scatter_radius).max(0) as usize;
        let end = ((center + scatter_radius) as usize).min(dst_len - 1);

        // Calculate scatter weights: Lanczos7 window * destination variance
        let mut weights = Vec::with_capacity(end - start + 1);
        let mut weight_sum = 0.0f32;

        for di in start..=end {
            let d = (dst_pos - di as f32) / filter_scale;
            if d.abs() > 7.0 {
                weights.push(0.0);
                continue;
            }

            // Lanczos7 window weight * destination variance
            // Window is already non-negative within support, no abs needed
            let base_weight = lanczos7_window(d);
            let var_weight = dst_var[di].max(0.001); // Minimum variance to avoid zero
            let weight = base_weight * var_weight;
            weights.push(weight);
            weight_sum += weight;
        }

        // Normalize and scatter
        if weight_sum.abs() > 1e-8 {
            let inv_sum = 1.0 / weight_sum;
            for (i, di) in (start..=end).enumerate() {
                let w = weights[i] * inv_sum;
                dst[di] = Pixel4::new(
                    dst[di].r() + err.r() * w,
                    dst[di].g() + err.g() * w,
                    dst[di].b() + err.b() * w,
                    dst[di].a(),
                );
            }
        }
    }
}

// ============================================================================
// Full variance-adaptive separable rescale
// ============================================================================

/// Variance-adaptive separable rescale (2-pass)
///
/// This algorithm:
/// 1. Calculates per-pixel variance from the source image
/// 2. Adapts kernel selection (Lanczos2-6) based on local variance
/// 3. Tracks energy contribution from each source pixel
/// 4. Scatters any energy error back, weighted by destination variance
pub fn rescale_variance_adaptive(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    if src_width == dst_width && src_height == dst_height {
        if let Some(ref mut cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Deterministic seed
    let seed = (src_width as u32).wrapping_mul(31) ^ (src_height as u32).wrapping_mul(17)
        ^ (dst_width as u32).wrapping_mul(13) ^ (dst_height as u32).wrapping_mul(7);

    // ========================================================================
    // Pass 1: Horizontal resample with variance adaptation
    // ========================================================================

    let mut temp = vec![Pixel4::default(); dst_width * src_height];
    let mut h_contributions = vec![0.0f32; src_width * src_height];

    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];

        // Calculate variance for this row
        let inbetween_var = calculate_inbetween_variance(src_row);
        let pixel_var = calculate_pixel_variance(&inbetween_var, src_width);

        // Resample with variance adaptation
        let result = resample_row_variance_adaptive(
            src_row,
            &pixel_var,
            dst_width,
            scale_x,
            seed,
            y,
        );

        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&result.pixels);
        h_contributions[y * src_width..(y + 1) * src_width].copy_from_slice(&result.contributions);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.25);
        }
    }

    // ========================================================================
    // Horizontal error scatter
    // ========================================================================

    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let contrib_row = &h_contributions[y * src_width..(y + 1) * src_width];

        // Calculate error
        let error = calculate_error(src_row, contrib_row);

        // Get destination variance for scatter weighting
        let inbetween_var = calculate_inbetween_variance(&temp[y * dst_width..(y + 1) * dst_width]);
        let dst_var = calculate_pixel_variance(&inbetween_var, dst_width);

        // Scatter error to destination
        scatter_error(
            &mut temp[y * dst_width..(y + 1) * dst_width],
            &error,
            &dst_var,
            scale_x,
        );

        if let Some(ref mut cb) = progress {
            cb(0.25 + (y + 1) as f32 / src_height as f32 * 0.25);
        }
    }

    // ========================================================================
    // Pass 2: Vertical resample with variance adaptation
    // ========================================================================

    let v_seed = seed ^ 0x9E3779B9;
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];
    let mut v_contributions = vec![0.0f32; dst_width * src_height];

    // Process column by column
    for x in 0..dst_width {
        // Extract column from temp
        let src_col: Vec<Pixel4> = (0..src_height)
            .map(|y| temp[y * dst_width + x])
            .collect();

        // Calculate variance for this column
        let inbetween_var = calculate_inbetween_variance(&src_col);
        let pixel_var = calculate_pixel_variance(&inbetween_var, src_height);

        // Resample with variance adaptation
        let result = resample_row_variance_adaptive(
            &src_col,
            &pixel_var,
            dst_height,
            scale_y,
            v_seed,
            x,
        );

        // Store result column
        for (y, pixel) in result.pixels.into_iter().enumerate() {
            dst[y * dst_width + x] = pixel;
        }

        // Store contributions
        for (y, &contrib) in result.contributions.iter().enumerate() {
            v_contributions[y * dst_width + x] = contrib;
        }
    }

    if let Some(ref mut cb) = progress {
        cb(0.75);
    }

    // ========================================================================
    // Vertical error scatter
    // ========================================================================

    for x in 0..dst_width {
        // Extract column from temp (source for vertical pass)
        let src_col: Vec<Pixel4> = (0..src_height)
            .map(|y| temp[y * dst_width + x])
            .collect();

        // Extract contributions for this column
        let contrib_col: Vec<f32> = (0..src_height)
            .map(|y| v_contributions[y * dst_width + x])
            .collect();

        // Calculate error
        let error = calculate_error(&src_col, &contrib_col);

        // Extract destination column for variance calculation
        let dst_col: Vec<Pixel4> = (0..dst_height)
            .map(|y| dst[y * dst_width + x])
            .collect();

        // Get destination variance for scatter weighting
        let inbetween_var = calculate_inbetween_variance(&dst_col);
        let dst_var = calculate_pixel_variance(&inbetween_var, dst_height);

        // Scatter error to destination column
        let mut dst_col_mut = dst_col;
        scatter_error(
            &mut dst_col_mut,
            &error,
            &dst_var,
            scale_y,
        );

        // Write back to destination
        for (y, pixel) in dst_col_mut.into_iter().enumerate() {
            dst[y * dst_width + x] = pixel;
        }
    }

    if let Some(ref mut cb) = progress {
        cb(1.0);
    }

    dst
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inbetween_variance() {
        // Test black to white edge - should give distance ≈ 1.0 in OkLab
        let row_bw = vec![
            Pixel4::new(0.0, 0.0, 0.0, 1.0),
            Pixel4::new(1.0, 1.0, 1.0, 1.0),
        ];
        let var_bw = calculate_inbetween_variance(&row_bw);
        assert_eq!(var_bw.len(), 1);
        // Black (L=0) to white (L=1) in OkLab: distance = 1.0
        assert!((var_bw[0] - 1.0).abs() < 0.01, "Black to white should be ~1.0, got {}", var_bw[0]);

        // Test gray ramp - OkLab distances won't be equal since L is perceptual
        let row = vec![
            Pixel4::new(0.0, 0.0, 0.0, 1.0),
            Pixel4::new(0.5, 0.5, 0.5, 1.0),
            Pixel4::new(1.0, 1.0, 1.0, 1.0),
        ];
        let var = calculate_inbetween_variance(&row);
        assert_eq!(var.len(), 2);
        // Both steps should be positive and sum to ~1.0 (total black-to-white distance)
        assert!(var[0] > 0.0 && var[0] < 1.0);
        assert!(var[1] > 0.0 && var[1] < 1.0);
        assert!((var[0] + var[1] - 1.0).abs() < 0.01, "Sum should be ~1.0, got {}", var[0] + var[1]);
    }

    #[test]
    fn test_pixel_variance_rms() {
        let inbetween = vec![0.5, 1.0];
        let pixel_var = calculate_pixel_variance(&inbetween, 3);
        assert_eq!(pixel_var.len(), 3);
        // First pixel: only right neighbor (0.5)
        assert!((pixel_var[0] - 0.5).abs() < 1e-6);
        // Middle pixel: RMS of 0.5 and 1.0 = sqrt((0.25 + 1.0) / 2) = sqrt(0.625) ≈ 0.79
        assert!((pixel_var[1] - 0.79).abs() < 0.01);
        // Last pixel: only left neighbor (1.0)
        assert!((pixel_var[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_selection_low_variance() {
        // Low variance should select Lanczos2 (lobes=2)
        let lobes = select_kernel_for_variance(0.0, 0);
        assert_eq!(lobes, 2);
    }

    #[test]
    fn test_kernel_selection_high_variance() {
        // High variance should select Lanczos6 (lobes=6)
        let lobes = select_kernel_for_variance(1.0, 0);
        assert_eq!(lobes, 6);
    }

    #[test]
    fn test_rescale_identity() {
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 1.0),
            Pixel4::new(0.5, 0.5, 0.5, 1.0),
            Pixel4::new(1.0, 1.0, 1.0, 1.0),
            Pixel4::new(0.5, 0.5, 0.5, 1.0),
        ];
        let dst = rescale_variance_adaptive(&src, 2, 2, 2, 2, ScaleMode::Independent, None);
        assert_eq!(dst.len(), 4);
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s.r() - d.r()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rescale_downscale() {
        // 4x4 -> 2x2 downscale
        let mut src = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                let val = (x + y) as f32 / 6.0;
                src.push(Pixel4::new(val, val, val, 1.0));
            }
        }
        let dst = rescale_variance_adaptive(&src, 4, 4, 2, 2, ScaleMode::Independent, None);
        assert_eq!(dst.len(), 4);

        // All values should be finite and reasonable
        for p in &dst {
            assert!(p.r().is_finite());
            assert!(p.r() >= -0.5 && p.r() <= 1.5); // Allow some overshoot
        }
    }

    #[test]
    fn test_rescale_upscale() {
        // 2x2 -> 4x4 upscale
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 1.0),
            Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 1.0),
        ];
        let dst = rescale_variance_adaptive(&src, 2, 2, 4, 4, ScaleMode::Independent, None);
        assert_eq!(dst.len(), 16);

        // All values should be finite
        for p in &dst {
            assert!(p.r().is_finite());
        }
    }
}
