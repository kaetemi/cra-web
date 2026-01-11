/// Basic LAB histogram matching algorithm.
/// Corresponds to color_correction_basic.py

use crate::color::{
    interleave_rgb_u8, lab_to_linear_rgb_channels, linear_rgb_to_lab_channels,
    linear_to_srgb_scaled_channels, srgb_to_linear_channels,
};
use crate::dither::{dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, InterpolationMode};

/// Scale L channel: 0-100 -> 0-255
fn scale_l_to_255(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0 / 100.0).collect()
}

/// Scale AB channels: -127..127 -> 0-255
fn scale_ab_to_255(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let a_scaled: Vec<f32> = a.iter().map(|&v| (v + 127.0) * 255.0 / 254.0).collect();
    let b_scaled: Vec<f32> = b.iter().map(|&v| (v + 127.0) * 255.0 / 254.0).collect();
    (a_scaled, b_scaled)
}

/// Reverse L scaling: 0-255 -> 0-100
fn scale_255_to_l(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 100.0 / 255.0).collect()
}

/// Reverse L scaling: uint8 -> 0-100
fn scale_uint8_to_l(l: &[u8]) -> Vec<f32> {
    l.iter().map(|&v| v as f32 * 100.0 / 255.0).collect()
}

/// Reverse AB scaling: 0-255 -> -127..127
fn scale_255_to_ab(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let a_lab: Vec<f32> = a.iter().map(|&v| v * 254.0 / 255.0 - 127.0).collect();
    let b_lab: Vec<f32> = b.iter().map(|&v| v * 254.0 / 255.0 - 127.0).collect();
    (a_lab, b_lab)
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
    use_f32_histogram: bool,
    dither_mode: DitherMode,
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
    let in_l_scaled = scale_l_to_255(&in_l);
    let (in_a_scaled, in_b_scaled) = scale_ab_to_255(&in_a, &in_b_ch);
    let ref_l_scaled = scale_l_to_255(&ref_l);
    let (ref_a_scaled, ref_b_scaled) = scale_ab_to_255(&ref_a, &ref_b_ch);

    // Match histograms
    let (final_l, final_a, final_b) = if use_f32_histogram {
        // Use f32 histogram matching directly (no dithering/quantization)
        if keep_luminosity {
            let matched_a =
                match_histogram_f32(&in_a_scaled, &ref_a_scaled, InterpolationMode::Linear, 0);
            let matched_b =
                match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, 1);
            let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b);
            (original_l, a_lab, b_lab)
        } else {
            let matched_l =
                match_histogram_f32(&in_l_scaled, &ref_l_scaled, InterpolationMode::Linear, 0);
            let matched_a =
                match_histogram_f32(&in_a_scaled, &ref_a_scaled, InterpolationMode::Linear, 1);
            let matched_b =
                match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, 2);
            let l_lab = scale_255_to_l(&matched_l);
            let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b);
            (l_lab, a_lab, b_lab)
        }
    } else {
        // Use binned histogram matching with dithering
        // Each dither call gets a unique seed for deterministic but varied randomization
        let in_l_u8 = dither_with_mode(&in_l_scaled, input_width, input_height, dither_mode, 0);
        let in_a_u8 = dither_with_mode(&in_a_scaled, input_width, input_height, dither_mode, 1);
        let in_b_u8 = dither_with_mode(&in_b_scaled, input_width, input_height, dither_mode, 2);
        let ref_l_u8 = dither_with_mode(&ref_l_scaled, ref_width, ref_height, dither_mode, 3);
        let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, dither_mode, 4);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, dither_mode, 5);

        if keep_luminosity {
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
        }
    };

    // Convert LAB back to linear RGB (separate channels)
    let (out_r, out_g, out_b) = lab_to_linear_rgb_channels(&final_l, &final_a, &final_b);

    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) = linear_to_srgb_scaled_channels(&out_r, &out_g, &out_b);

    // Dither each channel for final output
    let r_u8 = dither_with_mode(&r_scaled, input_width, input_height, dither_mode, 6);
    let g_u8 = dither_with_mode(&g_scaled, input_width, input_height, dither_mode, 7);
    let b_u8 = dither_with_mode(&b_scaled, input_width, input_height, dither_mode, 8);

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

        let l_scaled = scale_l_to_255(&l);
        let (a_scaled, b_scaled) = scale_ab_to_255(&a, &b);

        // Test roundtrip through f32 scaling
        let l2 = scale_255_to_l(&l_scaled);
        let (a2, b2) = scale_255_to_ab(&a_scaled, &b_scaled);

        assert!((l2[0] - 50.0).abs() < 0.01);
        assert!((a2[0] - 0.0).abs() < 0.01);
        assert!((b2[0] - 0.0).abs() < 0.01);

        // Also test roundtrip through uint8 (via rounding)
        let l_u8 = vec![l_scaled[0].round() as u8];
        let a_u8 = vec![a_scaled[0].round() as u8];
        let b_u8 = vec![b_scaled[0].round() as u8];

        let l3 = scale_uint8_to_l(&l_u8);
        let (a3, b3) = scale_uint8_to_ab(&a_u8, &b_u8);

        assert!((l3[0] - 50.0).abs() < 1.0);
        assert!((a3[0] - 0.0).abs() < 1.0);
        assert!((b3[0] - 0.0).abs() < 1.0);
    }
}
