/// Basic RGB histogram matching algorithm.
/// Corresponds to color_correction_basic_rgb.py

use crate::dither::{dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, AlignmentMode, InterpolationMode};

/// Basic RGB histogram matching - returns linear RGB channels
///
/// This is the core algorithm that performs histogram matching in linear RGB space
/// and returns the result as linear RGB channels (f32, 0-1 range).
///
/// Args:
///     in_r, in_g, in_b: Input image as linear RGB channels (0-1 range)
///     ref_r, ref_g, ref_b: Reference image as linear RGB channels (0-1 range)
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram quantization
///
/// Returns: (R, G, B) linear RGB channels
pub fn color_correct_basic_rgb_linear(
    in_r: &[f32],
    in_g: &[f32],
    in_b: &[f32],
    ref_r: &[f32],
    ref_g: &[f32],
    ref_b: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Scale to 0-255 range
    let in_r_scaled: Vec<f32> = in_r.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let in_g_scaled: Vec<f32> = in_g.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let in_b_scaled: Vec<f32> = in_b.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();

    let ref_r_scaled: Vec<f32> = ref_r.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let ref_g_scaled: Vec<f32> = ref_g.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let ref_b_scaled: Vec<f32> = ref_b.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();

    // Match histograms
    // histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
    if histogram_mode > 0 {
        // Use f32 histogram matching directly (no dithering/quantization)
        let align_mode = if histogram_mode == 2 {
            AlignmentMode::Midpoint
        } else {
            AlignmentMode::Endpoint
        };
        let matched_r =
            match_histogram_f32(&in_r_scaled, &ref_r_scaled, InterpolationMode::Linear, align_mode, 0);
        let matched_g =
            match_histogram_f32(&in_g_scaled, &ref_g_scaled, InterpolationMode::Linear, align_mode, 1);
        let matched_b =
            match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 2);

        // Scale back to linear 0-1 range
        let r_linear: Vec<f32> = matched_r.iter().map(|&v| v / 255.0).collect();
        let g_linear: Vec<f32> = matched_g.iter().map(|&v| v / 255.0).collect();
        let b_linear: Vec<f32> = matched_b.iter().map(|&v| v / 255.0).collect();
        (r_linear, g_linear, b_linear)
    } else {
        // Use binned histogram matching with dithering
        // Each dither call gets a unique seed for deterministic but varied randomization
        let in_r_u8 = dither_with_mode(&in_r_scaled, input_width, input_height, histogram_dither_mode, 0);
        let in_g_u8 = dither_with_mode(&in_g_scaled, input_width, input_height, histogram_dither_mode, 1);
        let in_b_u8 = dither_with_mode(&in_b_scaled, input_width, input_height, histogram_dither_mode, 2);

        let ref_r_u8 = dither_with_mode(&ref_r_scaled, ref_width, ref_height, histogram_dither_mode, 3);
        let ref_g_u8 = dither_with_mode(&ref_g_scaled, ref_width, ref_height, histogram_dither_mode, 4);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, 5);

        let matched_r = match_histogram(&in_r_u8, &ref_r_u8);
        let matched_g = match_histogram(&in_g_u8, &ref_g_u8);
        let matched_b = match_histogram(&in_b_u8, &ref_b_u8);

        // Scale back to linear 0-1 range
        let r_linear: Vec<f32> = matched_r.iter().map(|&v| v as f32 / 255.0).collect();
        let g_linear: Vec<f32> = matched_g.iter().map(|&v| v as f32 / 255.0).collect();
        let b_linear: Vec<f32> = matched_b.iter().map(|&v| v as f32 / 255.0).collect();
        (r_linear, g_linear, b_linear)
    }
}
