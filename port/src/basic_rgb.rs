/// Basic RGB histogram matching algorithm.
/// Corresponds to color_correction_basic_rgb.py

use crate::color::{linear_to_srgb, srgb_to_linear};
use crate::dither::{dither_rgb, floyd_steinberg_dither};
use crate::histogram::match_histogram;

/// Basic RGB histogram matching
///
/// Args:
///     input_srgb: Input image as sRGB values (0-1), flat array HxWx3
///     ref_srgb: Reference image as sRGB values (0-1), flat array HxWx3
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
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
) -> Vec<u8> {
    let input_pixels = input_width * input_height;
    let ref_pixels = ref_width * ref_height;

    // Convert to linear RGB
    let mut input_linear = input_srgb.to_vec();
    let mut ref_linear = ref_srgb.to_vec();
    srgb_to_linear(&mut input_linear);
    srgb_to_linear(&mut ref_linear);

    // Scale to 0-255 range
    let input_scaled: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            (0..input_pixels)
                .map(|i| (input_linear[i * 3 + ch] * 255.0).clamp(0.0, 255.0))
                .collect()
        })
        .collect();

    let ref_scaled: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            (0..ref_pixels)
                .map(|i| (ref_linear[i * 3 + ch] * 255.0).clamp(0.0, 255.0))
                .collect()
        })
        .collect();

    // Dither each channel directly
    let input_channels: Vec<Vec<u8>> = input_scaled
        .iter()
        .map(|ch| floyd_steinberg_dither(ch, input_width, input_height))
        .collect();

    let ref_channels: Vec<Vec<u8>> = ref_scaled
        .iter()
        .map(|ch| floyd_steinberg_dither(ch, ref_width, ref_height))
        .collect();

    // Match histograms
    let matched_channels: Vec<Vec<u8>> = (0..3)
        .map(|ch| match_histogram(&input_channels[ch], &ref_channels[ch]))
        .collect();

    // Scale back to 0-1 range
    let mut matched_linear = vec![0.0f32; input_pixels * 3];
    for i in 0..input_pixels {
        for ch in 0..3 {
            matched_linear[i * 3 + ch] = matched_channels[ch][i] as f32 / 255.0;
        }
    }

    // Convert back to sRGB
    linear_to_srgb(&mut matched_linear);

    // Final dither to uint8
    dither_rgb(&matched_linear, input_width, input_height)
}
