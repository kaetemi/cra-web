/// Basic LAB histogram matching algorithm.
/// Corresponds to color_correction_basic.py

use crate::color::{
    interleave_rgb_u8, lab_to_linear_rgb_channels, linear_rgb_to_lab_channels,
    linear_to_srgb_scaled_channels, srgb_to_linear_channels,
};
use crate::dither::floyd_steinberg_dither;
use crate::histogram::match_histogram;

/// Scale L channel: 0-100 -> 0-255
fn scale_l_to_uint8(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0 / 100.0).collect()
}

/// Scale AB channels: -127..127 -> 0-255
fn scale_ab_to_uint8(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let a_scaled: Vec<f32> = a.iter().map(|&v| (v + 127.0) * 255.0 / 254.0).collect();
    let b_scaled: Vec<f32> = b.iter().map(|&v| (v + 127.0) * 255.0 / 254.0).collect();
    (a_scaled, b_scaled)
}

/// Reverse L scaling: uint8 -> 0-100
fn scale_uint8_to_l(l: &[u8]) -> Vec<f32> {
    l.iter().map(|&v| v as f32 * 100.0 / 255.0).collect()
}

/// Reverse AB scaling: uint8 -> -127..127
fn scale_uint8_to_ab(a: &[u8], b: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let a_lab: Vec<f32> = a.iter().map(|&v| v as f32 * 254.0 / 255.0 - 127.0).collect();
    let b_lab: Vec<f32> = b.iter().map(|&v| v as f32 * 254.0 / 255.0 - 127.0).collect();
    (a_lab, b_lab)
}

/// Basic LAB histogram matching
pub fn color_correct_basic_lab(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
) -> Vec<u8> {
    // Convert to separate linear RGB channels
    let (in_r, in_g, in_b) = srgb_to_linear_channels(input_srgb, input_width, input_height);
    let (ref_r, ref_g, ref_b) = srgb_to_linear_channels(ref_srgb, ref_width, ref_height);

    // Convert to separate LAB channels
    let (in_l, in_a, in_b_ch) = linear_rgb_to_lab_channels(&in_r, &in_g, &in_b);
    let (ref_l, ref_a, ref_b_ch) = linear_rgb_to_lab_channels(&ref_r, &ref_g, &ref_b);

    // Store original L if preserving luminosity
    let original_l = if keep_luminosity { in_l.clone() } else { Vec::new() };

    // Scale to 0-255 range
    let in_l_scaled = scale_l_to_uint8(&in_l);
    let (in_a_scaled, in_b_scaled) = scale_ab_to_uint8(&in_a, &in_b_ch);
    let ref_l_scaled = scale_l_to_uint8(&ref_l);
    let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a, &ref_b_ch);

    // Dither each channel
    let in_l_u8 = floyd_steinberg_dither(&in_l_scaled, input_width, input_height);
    let in_a_u8 = floyd_steinberg_dither(&in_a_scaled, input_width, input_height);
    let in_b_u8 = floyd_steinberg_dither(&in_b_scaled, input_width, input_height);
    let ref_l_u8 = floyd_steinberg_dither(&ref_l_scaled, ref_width, ref_height);
    let ref_a_u8 = floyd_steinberg_dither(&ref_a_scaled, ref_width, ref_height);
    let ref_b_u8 = floyd_steinberg_dither(&ref_b_scaled, ref_width, ref_height);

    // Match histograms
    let (final_l, final_a, final_b) = if keep_luminosity {
        let matched_a = match_histogram(&in_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&in_b_u8, &ref_b_u8);
        let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b);
        (original_l, a_lab, b_lab)
    } else {
        let matched_l = match_histogram(&in_l_u8, &ref_l_u8);
        let matched_a = match_histogram(&in_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&in_b_u8, &ref_b_u8);
        let l_lab = scale_uint8_to_l(&matched_l);
        let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b);
        (l_lab, a_lab, b_lab)
    };

    // Convert LAB back to linear RGB (separate channels)
    let (out_r, out_g, out_b) = lab_to_linear_rgb_channels(&final_l, &final_a, &final_b);

    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) = linear_to_srgb_scaled_channels(&out_r, &out_g, &out_b);

    // Dither each channel
    let r_u8 = floyd_steinberg_dither(&r_scaled, input_width, input_height);
    let g_u8 = floyd_steinberg_dither(&g_scaled, input_width, input_height);
    let b_u8 = floyd_steinberg_dither(&b_scaled, input_width, input_height);

    // Interleave only at the very end
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_roundtrip() {
        let l = vec![50.0];
        let a = vec![0.0];
        let b = vec![0.0];

        let l_scaled = scale_l_to_uint8(&l);
        let (a_scaled, b_scaled) = scale_ab_to_uint8(&a, &b);

        let l_u8 = vec![l_scaled[0].round() as u8];
        let a_u8 = vec![a_scaled[0].round() as u8];
        let b_u8 = vec![b_scaled[0].round() as u8];

        let l2 = scale_uint8_to_l(&l_u8);
        let (a2, b2) = scale_uint8_to_ab(&a_u8, &b_u8);

        assert!((l2[0] - 50.0).abs() < 1.0);
        assert!((a2[0] - 0.0).abs() < 1.0);
        assert!((b2[0] - 0.0).abs() < 1.0);
    }
}
