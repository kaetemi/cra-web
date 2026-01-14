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

/// Rescale a single channel using bilinear interpolation
fn rescale_bilinear(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Maximum valid coordinate (for clamping)
    let max_x = (src_width - 1) as f32;
    let max_y = (src_height - 1) as f32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // Map destination pixel center to source coordinates
            // Using half-pixel offset: center of dst pixel maps to corresponding position in src
            let src_x = ((dst_x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, max_x);
            let src_y = ((dst_y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, max_y);

            // Get integer and fractional parts (after clamping for correct edge behavior)
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

            // Bilinear interpolation
            let top = p00 * (1.0 - fx) + p10 * fx;
            let bottom = p01 * (1.0 - fx) + p11 * fx;
            dst[dst_y * dst_width + dst_x] = top * (1.0 - fy) + bottom * fy;
        }
    }

    dst
}

/// Lanczos3 1D resample using precomputed weights - resamples a contiguous slice
#[inline]
fn lanczos3_resample_1d_precomputed(
    src: &[f32],
    kernel_weights: &[KernelWeights],
) -> Vec<f32> {
    let dst_len = kernel_weights.len();
    let mut dst = vec![0.0f32; dst_len];

    for (dst_i, kw) in kernel_weights.iter().enumerate() {
        if kw.weights.is_empty() {
            dst[dst_i] = src[kw.fallback_idx];
        } else {
            let mut sum = 0.0f32;
            for (i, &weight) in kw.weights.iter().enumerate() {
                sum += src[kw.start_idx + i] * weight;
            }
            dst[dst_i] = sum;
        }
    }

    dst
}

/// Rescale a single channel using Lanczos3 interpolation (separable, 2-pass)
/// Uses precomputed kernel weights for efficiency
fn rescale_lanczos3(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
) -> Vec<f32> {
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // For downscaling, increase filter radius
    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    // Precompute weights for horizontal and vertical passes
    let h_weights = precompute_lanczos_weights(src_width, dst_width, scale_x, filter_scale_x, radius_x);
    let v_weights = precompute_lanczos_weights(src_height, dst_height, scale_y, filter_scale_y, radius_y);

    // Pass 1: Horizontal resample each row (src_width -> dst_width)
    let mut temp = vec![0.0f32; dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = lanczos3_resample_1d_precomputed(src_row, &h_weights);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);
    }

    // Pass 2: Vertical resample each column (src_height -> dst_height)
    let mut dst = vec![0.0f32; dst_width * dst_height];
    for x in 0..dst_width {
        // Extract column
        let col: Vec<f32> = (0..src_height).map(|y| temp[y * dst_width + x]).collect();
        let dst_col = lanczos3_resample_1d_precomputed(&col, &v_weights);
        for (y, &val) in dst_col.iter().enumerate() {
            dst[y * dst_width + x] = val;
        }
    }

    dst
}

/// Rescale a single channel
pub fn rescale_channel(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
) -> Vec<f32> {
    rescale_channel_uniform(src, src_width, src_height, dst_width, dst_height, method, ScaleMode::Independent)
}

/// Rescale a single channel with uniform scale mode for perfect AR preservation
pub fn rescale_channel_uniform(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> Vec<f32> {
    if src_width == dst_width && src_height == dst_height {
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => rescale_bilinear(src, src_width, src_height, dst_width, dst_height, scale_mode),
        RescaleMethod::Lanczos3 => rescale_lanczos3(src, src_width, src_height, dst_width, dst_height, scale_mode),
    }
}

/// Rescale RGB image (separate channels, linear space)
pub fn rescale_rgb(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let r_out = rescale_channel_uniform(r, src_width, src_height, dst_width, dst_height, method, scale_mode);
    let g_out = rescale_channel_uniform(g, src_width, src_height, dst_width, dst_height, method, scale_mode);
    let b_out = rescale_channel_uniform(b, src_width, src_height, dst_width, dst_height, method, scale_mode);
    (r_out, g_out, b_out)
}

/// Rescale interleaved RGB image (linear space, f32 values in 0-1 range)
pub fn rescale_rgb_interleaved(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> Vec<f32> {
    rescale_rgb_interleaved_with_progress(src, src_width, src_height, dst_width, dst_height, method, scale_mode, None)
}

/// Calculate target dimensions preserving aspect ratio
pub fn calculate_target_dimensions(
    src_width: usize,
    src_height: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
) -> (usize, usize) {
    match (target_width, target_height) {
        (Some(w), Some(h)) => (w, h),
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
    let mut dst = vec![[0.0f32; 4]; dst_width * dst_height];

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

/// Lanczos3 1D resample for Pixel4 row using precomputed weights
#[inline]
fn lanczos3_resample_row_pixel4_precomputed(
    src: &[Pixel4],
    kernel_weights: &[KernelWeights],
) -> Vec<Pixel4> {
    let dst_len = kernel_weights.len();
    let mut dst = vec![[0.0f32; 4]; dst_len];

    for (dst_i, kw) in kernel_weights.iter().enumerate() {
        if kw.weights.is_empty() {
            dst[dst_i] = src[kw.fallback_idx];
        } else {
            let mut sum = [0.0f32; 4];
            for (i, &weight) in kw.weights.iter().enumerate() {
                let sample = src[kw.start_idx + i];
                sum[0] += sample[0] * weight;
                sum[1] += sample[1] * weight;
                sum[2] += sample[2] * weight;
                sum[3] += sample[3] * weight;
            }
            dst[dst_i] = sum;
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
    let mut temp = vec![[0.0f32; 4]; dst_width * src_height];
    for y in 0..src_height {
        let src_row = &src[y * src_width..(y + 1) * src_width];
        let dst_row = lanczos3_resample_row_pixel4_precomputed(src_row, &h_weights);
        temp[y * dst_width..(y + 1) * dst_width].copy_from_slice(&dst_row);

        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / src_height as f32 * 0.5);
        }
    }

    // Pass 2: Vertical resample each column (src_height -> dst_height)
    // Progress: 50% to 100%
    let mut dst = vec![[0.0f32; 4]; dst_width * dst_height];
    for x in 0..dst_width {
        // Extract column
        let col: Vec<Pixel4> = (0..src_height).map(|y| temp[y * dst_width + x]).collect();
        let dst_col = lanczos3_resample_row_pixel4_precomputed(&col, &v_weights);
        for (y, &val) in dst_col.iter().enumerate() {
            dst[y * dst_width + x] = val;
        }

        if let Some(ref mut cb) = progress {
            cb(0.5 + (x + 1) as f32 / dst_width as f32 * 0.5);
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

/// Rescale interleaved RGB image with optional progress callback (linear space, f32 values in 0-1 range)
pub fn rescale_rgb_interleaved_with_progress(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<f32> {
    let src_pixels = src_width * src_height;

    // Convert interleaved to Pixel4
    let mut pixels: Vec<Pixel4> = Vec::with_capacity(src_pixels);
    for i in 0..src_pixels {
        pixels.push([src[i * 3], src[i * 3 + 1], src[i * 3 + 2], 0.0]);
    }

    // Rescale with progress
    let result_pixels = rescale_with_progress(
        &pixels, src_width, src_height, dst_width, dst_height, method, scale_mode, progress
    );

    // Convert back to interleaved
    let dst_pixels = dst_width * dst_height;
    let mut dst = Vec::with_capacity(dst_pixels * 3);
    for p in result_pixels {
        dst.push(p[0]);
        dst.push(p[1]);
        dst.push(p[2]);
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_identity() {
        // 2x2 image
        let src = vec![0.0, 0.25, 0.5, 0.75];
        let dst = rescale_channel(&src, 2, 2, 2, 2, RescaleMethod::Bilinear);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_bilinear_upscale() {
        // 2x2 -> 4x4
        // Source: [0, 1]
        //         [0, 1]
        let src = vec![0.0, 1.0, 0.0, 1.0];
        let dst = rescale_channel(&src, 2, 2, 4, 4, RescaleMethod::Bilinear);
        assert_eq!(dst.len(), 16);
        // Output should be in valid range and contain intermediate values
        for val in &dst {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
        // Should have some variation (not all same value)
        let min = dst.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = dst.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > min); // There should be some variation
    }

    #[test]
    fn test_lanczos_identity() {
        let src = vec![0.0, 0.25, 0.5, 0.75];
        let dst = rescale_channel(&src, 2, 2, 2, 2, RescaleMethod::Lanczos3);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_lanczos_downscale() {
        // 4x4 -> 2x2
        let src = vec![
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        let dst = rescale_channel(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3);
        assert_eq!(dst.len(), 4);
        // Left half should be ~0, right half ~1
        assert!(dst[0] < 0.5);
        assert!(dst[1] > 0.5);
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

        // Both
        let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(100));
        assert_eq!(w, 200);
        assert_eq!(h, 100);

        // None
        let (w, h) = calculate_target_dimensions(100, 50, None, None);
        assert_eq!(w, 100);
        assert_eq!(h, 50);
    }

    #[test]
    fn test_bilinear_roundtrip_2x() {
        // Test that 2x upscale then 2x downscale returns approximately original
        // 4x4 -> 8x8 -> 4x4
        let src = vec![
            0.0, 0.3, 0.6, 1.0,
            0.1, 0.4, 0.7, 0.9,
            0.2, 0.5, 0.8, 0.8,
            0.3, 0.6, 0.9, 0.7,
        ];
        let up = rescale_channel(&src, 4, 4, 8, 8, RescaleMethod::Bilinear);
        let down = rescale_channel(&up, 8, 8, 4, 4, RescaleMethod::Bilinear);

        // Should be close to original (within reasonable tolerance for interpolation)
        for (i, (&orig, &result)) in src.iter().zip(down.iter()).enumerate() {
            let diff = (orig - result).abs();
            assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig, result, diff);
        }
    }

    #[test]
    fn test_lanczos_roundtrip_2x() {
        // Test that 2x upscale then 2x downscale returns approximately original
        let src = vec![
            0.0, 0.3, 0.6, 1.0,
            0.1, 0.4, 0.7, 0.9,
            0.2, 0.5, 0.8, 0.8,
            0.3, 0.6, 0.9, 0.7,
        ];
        let up = rescale_channel(&src, 4, 4, 8, 8, RescaleMethod::Lanczos3);
        let down = rescale_channel(&up, 8, 8, 4, 4, RescaleMethod::Lanczos3);

        for (i, (&orig, &result)) in src.iter().zip(down.iter()).enumerate() {
            let diff = (orig - result).abs();
            assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig, result, diff);
        }
    }

    #[test]
    fn test_no_shift_on_upscale() {
        // A single white pixel in center should stay centered after upscale
        // 3x3 with center pixel white -> 6x6
        let src = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let dst = rescale_channel(&src, 3, 3, 6, 6, RescaleMethod::Bilinear);

        // The brightest area should still be in the center region
        // Center of 6x6 is around pixels (2,2), (2,3), (3,2), (3,3)
        let center_sum = dst[2 * 6 + 2] + dst[2 * 6 + 3] + dst[3 * 6 + 2] + dst[3 * 6 + 3];
        let corner_sum = dst[0] + dst[5] + dst[30] + dst[35]; // corners

        assert!(center_sum > corner_sum, "Center should be brighter than corners");
    }

    #[test]
    fn test_edge_pixels_preserved() {
        // Edge pixels shouldn't expand or shift weirdly
        // Left column black, right column white
        let src = vec![
            0.0, 1.0,
            0.0, 1.0,
        ];
        let up = rescale_channel(&src, 2, 2, 4, 4, RescaleMethod::Bilinear);

        // Left edge (x=0) should still be darkest
        // Right edge (x=3) should still be brightest
        let left_avg = (up[0] + up[4] + up[8] + up[12]) / 4.0;
        let right_avg = (up[3] + up[7] + up[11] + up[15]) / 4.0;

        assert!(left_avg < 0.5, "Left edge should be dark: {}", left_avg);
        assert!(right_avg > 0.5, "Right edge should be bright: {}", right_avg);
    }

    // RGB pixel rescale tests
    #[test]
    fn test_rgb_bilinear_identity() {
        let src = vec![
            [0.0, 0.1, 0.2, 0.0],
            [0.3, 0.4, 0.5, 0.0],
            [0.6, 0.7, 0.8, 0.0],
            [0.9, 1.0, 0.5, 0.0],
        ];
        let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_rgb_bilinear_upscale() {
        let src = vec![
            [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0],
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
            [0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 0.0],
            [0.7, 0.8, 0.9, 0.0], [0.2, 0.3, 0.4, 0.0],
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
                src.push([r, g, b, 0.0]);
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
                src.push([v, v, v, 0.0]);
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
}
