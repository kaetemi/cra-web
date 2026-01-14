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

/// Rescale a single channel using bilinear interpolation
fn rescale_bilinear(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height];

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // Map destination pixel to source coordinates (center of pixel)
            let src_x = (dst_x as f32 + 0.5) * scale_x - 0.5;
            let src_y = (dst_y as f32 + 0.5) * scale_y - 0.5;

            // Get integer and fractional parts
            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            // Clamp coordinates
            let x0 = x0.max(0).min(src_width as i32 - 1) as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let y0 = y0.max(0).min(src_height as i32 - 1) as usize;
            let y1 = (y0 + 1).min(src_height - 1);

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
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height];

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

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
    if src_width == dst_width && src_height == dst_height {
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => rescale_bilinear(src, src_width, src_height, dst_width, dst_height),
        RescaleMethod::Lanczos3 => rescale_lanczos3(src, src_width, src_height, dst_width, dst_height),
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
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let r_out = rescale_channel(r, src_width, src_height, dst_width, dst_height, method);
    let g_out = rescale_channel(g, src_width, src_height, dst_width, dst_height, method);
    let b_out = rescale_channel(b, src_width, src_height, dst_width, dst_height, method);
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
    let (r_out, g_out, b_out) = rescale_rgb(&r, &g, &b, src_width, src_height, dst_width, dst_height, method);

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
}
