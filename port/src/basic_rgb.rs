/// Basic RGB histogram matching algorithm.
/// Corresponds to color_correction_basic_rgb.py

use crate::color::{
    interleave_rgb_u8, linear_to_srgb_scaled_channels, srgb_to_linear_channels,
};
use crate::dither::{floyd_steinberg_dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, InterpolationMode};

/// Basic RGB histogram matching
///
/// Args:
///     input_srgb: Input image as sRGB values (0-1), flat array HxWx3
///     ref_srgb: Reference image as sRGB values (0-1), flat array HxWx3
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
///
/// Returns:
///     Output image as sRGB uint8, flat array HxWx3
pub fn color_correct_basic_rgb(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    use_f32_histogram: bool,
    dither_mode: DitherMode,
) -> Vec<u8> {
    // Convert to separate linear RGB channels
    let (in_r, in_g, in_b) = srgb_to_linear_channels(input_srgb, input_width, input_height);
    let (ref_r, ref_g, ref_b) = srgb_to_linear_channels(ref_srgb, ref_width, ref_height);

    // Scale to 0-255 range
    let in_r_scaled: Vec<f32> = in_r.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let in_g_scaled: Vec<f32> = in_g.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let in_b_scaled: Vec<f32> = in_b.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();

    let ref_r_scaled: Vec<f32> = ref_r.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let ref_g_scaled: Vec<f32> = ref_g.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();
    let ref_b_scaled: Vec<f32> = ref_b.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0)).collect();

    // Match histograms
    let (matched_r_linear, matched_g_linear, matched_b_linear) = if use_f32_histogram {
        // Use f32 histogram matching directly (no dithering/quantization)
        let matched_r =
            match_histogram_f32(&in_r_scaled, &ref_r_scaled, InterpolationMode::Linear, 0);
        let matched_g =
            match_histogram_f32(&in_g_scaled, &ref_g_scaled, InterpolationMode::Linear, 1);
        let matched_b =
            match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, 2);

        // Scale back to linear 0-1 range
        let r_linear: Vec<f32> = matched_r.iter().map(|&v| v / 255.0).collect();
        let g_linear: Vec<f32> = matched_g.iter().map(|&v| v / 255.0).collect();
        let b_linear: Vec<f32> = matched_b.iter().map(|&v| v / 255.0).collect();
        (r_linear, g_linear, b_linear)
    } else {
        // Use binned histogram matching with dithering
        // Each dither call gets a unique seed for deterministic but varied randomization
        let in_r_u8 = floyd_steinberg_dither_with_mode(&in_r_scaled, input_width, input_height, dither_mode, 0);
        let in_g_u8 = floyd_steinberg_dither_with_mode(&in_g_scaled, input_width, input_height, dither_mode, 1);
        let in_b_u8 = floyd_steinberg_dither_with_mode(&in_b_scaled, input_width, input_height, dither_mode, 2);

        let ref_r_u8 = floyd_steinberg_dither_with_mode(&ref_r_scaled, ref_width, ref_height, dither_mode, 3);
        let ref_g_u8 = floyd_steinberg_dither_with_mode(&ref_g_scaled, ref_width, ref_height, dither_mode, 4);
        let ref_b_u8 = floyd_steinberg_dither_with_mode(&ref_b_scaled, ref_width, ref_height, dither_mode, 5);

        let matched_r = match_histogram(&in_r_u8, &ref_r_u8);
        let matched_g = match_histogram(&in_g_u8, &ref_g_u8);
        let matched_b = match_histogram(&in_b_u8, &ref_b_u8);

        // Scale back to linear 0-1 range
        let r_linear: Vec<f32> = matched_r.iter().map(|&v| v as f32 / 255.0).collect();
        let g_linear: Vec<f32> = matched_g.iter().map(|&v| v as f32 / 255.0).collect();
        let b_linear: Vec<f32> = matched_b.iter().map(|&v| v as f32 / 255.0).collect();
        (r_linear, g_linear, b_linear)
    };

    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) =
        linear_to_srgb_scaled_channels(&matched_r_linear, &matched_g_linear, &matched_b_linear);

    // Dither each channel for final output
    let r_u8 = floyd_steinberg_dither_with_mode(&r_scaled, input_width, input_height, dither_mode, 6);
    let g_u8 = floyd_steinberg_dither_with_mode(&g_scaled, input_width, input_height, dither_mode, 7);
    let b_u8 = floyd_steinberg_dither_with_mode(&b_scaled, input_width, input_height, dither_mode, 8);

    // Interleave only at the very end
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}
