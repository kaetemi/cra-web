/// Basic RGB histogram matching algorithm.
/// Corresponds to color_correction_basic_rgb.py

use crate::dither::{dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, AlignmentMode, InterpolationMode};
use crate::pixel::Pixel4;

/// Basic RGB histogram matching - returns linear RGB as Pixel4 array
///
/// This is the core algorithm that performs histogram matching in linear RGB space
/// and returns the result as linear RGB Pixel4 array (0-1 range).
///
/// Args:
///     input: Input image as linear RGB Pixel4 array (0-1 range)
///     reference: Reference image as linear RGB Pixel4 array (0-1 range)
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram quantization
///
/// Returns: Linear RGB Pixel4 array
pub fn color_correct_basic_rgb_linear(
    input: &[Pixel4],
    reference: &[Pixel4],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let pixel_count = input.len();

    // Extract and scale channels to 0-255 range
    let mut in_r_scaled = Vec::with_capacity(pixel_count);
    let mut in_g_scaled = Vec::with_capacity(pixel_count);
    let mut in_b_scaled = Vec::with_capacity(pixel_count);

    for &p in input {
        in_r_scaled.push((p[0] * 255.0).clamp(0.0, 255.0));
        in_g_scaled.push((p[1] * 255.0).clamp(0.0, 255.0));
        in_b_scaled.push((p[2] * 255.0).clamp(0.0, 255.0));
    }

    let mut ref_r_scaled = Vec::with_capacity(reference.len());
    let mut ref_g_scaled = Vec::with_capacity(reference.len());
    let mut ref_b_scaled = Vec::with_capacity(reference.len());

    for &p in reference {
        ref_r_scaled.push((p[0] * 255.0).clamp(0.0, 255.0));
        ref_g_scaled.push((p[1] * 255.0).clamp(0.0, 255.0));
        ref_b_scaled.push((p[2] * 255.0).clamp(0.0, 255.0));
    }

    if let Some(ref mut cb) = progress {
        cb(0.1);
    }

    // Match histograms
    // histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
    let (matched_r, matched_g, matched_b): (Vec<f32>, Vec<f32>, Vec<f32>) = if histogram_mode > 0 {
        // Use f32 histogram matching directly (no dithering/quantization)
        let align_mode = if histogram_mode == 2 {
            AlignmentMode::Midpoint
        } else {
            AlignmentMode::Endpoint
        };
        let r = match_histogram_f32(&in_r_scaled, &ref_r_scaled, InterpolationMode::Linear, align_mode, 0);
        let g = match_histogram_f32(&in_g_scaled, &ref_g_scaled, InterpolationMode::Linear, align_mode, 1);
        let b = match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 2);
        (r, g, b)
    } else {
        // Use binned histogram matching with dithering
        let in_r_u8 = dither_with_mode(&in_r_scaled, input_width, input_height, histogram_dither_mode, 0);
        let in_g_u8 = dither_with_mode(&in_g_scaled, input_width, input_height, histogram_dither_mode, 1);
        let in_b_u8 = dither_with_mode(&in_b_scaled, input_width, input_height, histogram_dither_mode, 2);

        let ref_r_u8 = dither_with_mode(&ref_r_scaled, ref_width, ref_height, histogram_dither_mode, 3);
        let ref_g_u8 = dither_with_mode(&ref_g_scaled, ref_width, ref_height, histogram_dither_mode, 4);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, 5);

        let r = match_histogram(&in_r_u8, &ref_r_u8);
        let g = match_histogram(&in_g_u8, &ref_g_u8);
        let b = match_histogram(&in_b_u8, &ref_b_u8);

        // Convert u8 to f32
        (
            r.iter().map(|&v| v as f32).collect(),
            g.iter().map(|&v| v as f32).collect(),
            b.iter().map(|&v| v as f32).collect(),
        )
    };

    if let Some(ref mut cb) = progress {
        cb(0.8);
    }

    // Reconstruct Pixel4 array, scaling back to 0-1 range
    let mut result = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        result.push(Pixel4::new(
            matched_r[i] / 255.0,
            matched_g[i] / 255.0,
            matched_b[i] / 255.0,
            0.0,
        ));
    }

    if let Some(ref mut cb) = progress {
        cb(1.0);
    }

    result
}
