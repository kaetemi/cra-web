/// Image rescaling module with Bilinear and Lanczos support
///
/// Operates in linear RGB space for correct color blending during interpolation.

use std::f32::consts::PI;

/// Progress callback type: receives progress as f32 in 0.0-1.0 range
/// 0.0 = before first row, 1.0 = after last row
pub type ProgressCallback<'a> = Option<&'a mut dyn FnMut(f32)>;

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

/// Rescale a single channel using Lanczos3 interpolation
fn rescale_lanczos3(
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

    // For downscaling, we need to increase the filter radius
    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // Map destination pixel to source coordinates
            let src_x = (dst_x as f32 + 0.5) * scale_x - 0.5;
            let src_y = (dst_y as f32 + 0.5) * scale_y - 0.5;

            let center_x = src_x.floor() as i32;
            let center_y = src_y.floor() as i32;

            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            // Sample in the kernel window
            for ky in -radius_y..=radius_y {
                let sy = center_y + ky;
                if sy < 0 || sy >= src_height as i32 {
                    continue;
                }

                let dy = (src_y - sy as f32) / filter_scale_y;
                let wy = lanczos3(dy);

                if wy.abs() < 1e-8 {
                    continue;
                }

                for kx in -radius_x..=radius_x {
                    let sx = center_x + kx;
                    if sx < 0 || sx >= src_width as i32 {
                        continue;
                    }

                    let dx = (src_x - sx as f32) / filter_scale_x;
                    let wx = lanczos3(dx);

                    let weight = wx * wy;
                    if weight.abs() < 1e-8 {
                        continue;
                    }

                    let sample = src[sy as usize * src_width + sx as usize];
                    sum += sample * weight;
                    weight_sum += weight;
                }
            }

            // Normalize
            dst[dst_y * dst_width + dst_x] = if weight_sum.abs() > 1e-8 {
                (sum / weight_sum).clamp(0.0, 1.0)
            } else {
                // Fallback to nearest neighbor
                let sx = src_x.round().clamp(0.0, (src_width - 1) as f32) as usize;
                let sy = src_y.round().clamp(0.0, (src_height - 1) as f32) as usize;
                src[sy * src_width + sx]
            };
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
    let src_pixels = src_width * src_height;

    // Split into channels
    let mut r = Vec::with_capacity(src_pixels);
    let mut g = Vec::with_capacity(src_pixels);
    let mut b = Vec::with_capacity(src_pixels);

    for i in 0..src_pixels {
        r.push(src[i * 3]);
        g.push(src[i * 3 + 1]);
        b.push(src[i * 3 + 2]);
    }

    // Rescale each channel
    let (r_out, g_out, b_out) = rescale_rgb(&r, &g, &b, src_width, src_height, dst_width, dst_height, method, scale_mode);

    // Interleave
    let dst_pixels = dst_width * dst_height;
    let mut dst = Vec::with_capacity(dst_pixels * 3);
    for i in 0..dst_pixels {
        dst.push(r_out[i]);
        dst.push(g_out[i]);
        dst.push(b_out[i]);
    }

    dst
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
fn rescale_bilinear_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
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
    }

    dst
}

/// Rescale Pixel4 array using Lanczos3 interpolation
fn rescale_lanczos3_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
) -> Vec<Pixel4> {
    let mut dst = vec![[0.0f32; 4]; dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = (dst_x as f32 + 0.5) * scale_x - 0.5;
            let src_y = (dst_y as f32 + 0.5) * scale_y - 0.5;

            let center_x = src_x.floor() as i32;
            let center_y = src_y.floor() as i32;

            let mut sum = [0.0f32; 4];
            let mut weight_sum = 0.0f32;

            for ky in -radius_y..=radius_y {
                let sy = center_y + ky;
                if sy < 0 || sy >= src_height as i32 {
                    continue;
                }

                let dy = (src_y - sy as f32) / filter_scale_y;
                let wy = lanczos3(dy);

                if wy.abs() < 1e-8 {
                    continue;
                }

                for kx in -radius_x..=radius_x {
                    let sx = center_x + kx;
                    if sx < 0 || sx >= src_width as i32 {
                        continue;
                    }

                    let dx = (src_x - sx as f32) / filter_scale_x;
                    let wx = lanczos3(dx);

                    let weight = wx * wy;
                    if weight.abs() < 1e-8 {
                        continue;
                    }

                    let sample = src[sy as usize * src_width + sx as usize];
                    sum[0] += sample[0] * weight;
                    sum[1] += sample[1] * weight;
                    sum[2] += sample[2] * weight;
                    sum[3] += sample[3] * weight;
                    weight_sum += weight;
                }
            }

            dst[dst_y * dst_width + dst_x] = if weight_sum.abs() > 1e-8 {
                [
                    (sum[0] / weight_sum).clamp(0.0, 1.0),
                    (sum[1] / weight_sum).clamp(0.0, 1.0),
                    (sum[2] / weight_sum).clamp(0.0, 1.0),
                    (sum[3] / weight_sum).clamp(0.0, 1.0),
                ]
            } else {
                let sx = src_x.round().clamp(0.0, (src_width - 1) as f32) as usize;
                let sy = src_y.round().clamp(0.0, (src_height - 1) as f32) as usize;
                src[sy * src_width + sx]
            };
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
    if src_width == dst_width && src_height == dst_height {
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => rescale_bilinear_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode),
        RescaleMethod::Lanczos3 => rescale_lanczos3_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode),
    }
}

// ============================================================================
// Progress-enabled rescaling functions
// ============================================================================

/// Rescale Pixel4 array using bilinear interpolation with progress callback
fn rescale_bilinear_pixels_progress<F: FnMut(f32)>(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: F,
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
        // Report progress after each row (1.0 after last row)
        progress((dst_y + 1) as f32 / dst_height as f32);
    }

    dst
}

/// Rescale Pixel4 array using Lanczos3 interpolation with progress callback
fn rescale_lanczos3_pixels_progress<F: FnMut(f32)>(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: F,
) -> Vec<Pixel4> {
    let mut dst = vec![[0.0f32; 4]; dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    let filter_scale_x = scale_x.max(1.0);
    let filter_scale_y = scale_y.max(1.0);
    let radius_x = (3.0 * filter_scale_x).ceil() as i32;
    let radius_y = (3.0 * filter_scale_y).ceil() as i32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = (dst_x as f32 + 0.5) * scale_x - 0.5;
            let src_y = (dst_y as f32 + 0.5) * scale_y - 0.5;

            let center_x = src_x.floor() as i32;
            let center_y = src_y.floor() as i32;

            let mut sum = [0.0f32; 4];
            let mut weight_sum = 0.0f32;

            for ky in -radius_y..=radius_y {
                let sy = center_y + ky;
                if sy < 0 || sy >= src_height as i32 {
                    continue;
                }

                let dy = (src_y - sy as f32) / filter_scale_y;
                let wy = lanczos3(dy);

                if wy.abs() < 1e-8 {
                    continue;
                }

                for kx in -radius_x..=radius_x {
                    let sx = center_x + kx;
                    if sx < 0 || sx >= src_width as i32 {
                        continue;
                    }

                    let dx = (src_x - sx as f32) / filter_scale_x;
                    let wx = lanczos3(dx);

                    let weight = wx * wy;
                    if weight.abs() < 1e-8 {
                        continue;
                    }

                    let sample = src[sy as usize * src_width + sx as usize];
                    sum[0] += sample[0] * weight;
                    sum[1] += sample[1] * weight;
                    sum[2] += sample[2] * weight;
                    sum[3] += sample[3] * weight;
                    weight_sum += weight;
                }
            }

            dst[dst_y * dst_width + dst_x] = if weight_sum.abs() > 1e-8 {
                [
                    (sum[0] / weight_sum).clamp(0.0, 1.0),
                    (sum[1] / weight_sum).clamp(0.0, 1.0),
                    (sum[2] / weight_sum).clamp(0.0, 1.0),
                    (sum[3] / weight_sum).clamp(0.0, 1.0),
                ]
            } else {
                let sx = src_x.round().clamp(0.0, (src_width - 1) as f32) as usize;
                let sy = src_y.round().clamp(0.0, (src_height - 1) as f32) as usize;
                src[sy * src_width + sx]
            };
        }
        // Report progress after each row (1.0 after last row)
        progress((dst_y + 1) as f32 / dst_height as f32);
    }

    dst
}

/// Rescale Pixel4 image with progress callback (SIMD-friendly, linear space)
pub fn rescale_with_progress<F: FnMut(f32)>(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: F,
) -> Vec<Pixel4> {
    if src_width == dst_width && src_height == dst_height {
        progress(1.0);
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => rescale_bilinear_pixels_progress(src, src_width, src_height, dst_width, dst_height, scale_mode, progress),
        RescaleMethod::Lanczos3 => rescale_lanczos3_pixels_progress(src, src_width, src_height, dst_width, dst_height, scale_mode, progress),
    }
}

/// Rescale interleaved RGB image with progress callback (linear space, f32 values in 0-1 range)
pub fn rescale_rgb_interleaved_with_progress<F: FnMut(f32)>(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    progress: F,
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
}
