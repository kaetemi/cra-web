/// Basic LAB histogram matching algorithm.
/// Corresponds to color_correction_basic.py

use crate::color::{
    extract_channel, image_lab_to_rgb, image_rgb_to_lab, linear_to_srgb, srgb_to_linear,
};
use crate::dither::{dither_channel_stack, dither_rgb};
use crate::histogram::match_histogram;

/// Scale LAB values to uint8 range for histogram matching
/// L: 0-100 -> 0-255
/// A, B: -127 to 127 -> 0-255
fn scale_lab_to_uint8(lab: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = width * height;
    let mut l_scaled = vec![0.0f32; pixels];
    let mut a_scaled = vec![0.0f32; pixels];
    let mut b_scaled = vec![0.0f32; pixels];

    for i in 0..pixels {
        let idx = i * 3;
        l_scaled[i] = lab[idx] * 255.0 / 100.0;
        a_scaled[i] = (lab[idx + 1] + 127.0) * 255.0 / 254.0;
        b_scaled[i] = (lab[idx + 2] + 127.0) * 255.0 / 254.0;
    }

    (l_scaled, a_scaled, b_scaled)
}

/// Reverse the uint8 scaling back to LAB range
fn scale_uint8_to_lab(l: &[u8], a: &[u8], b: &[u8]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = l.len();
    let mut l_lab = vec![0.0f32; pixels];
    let mut a_lab = vec![0.0f32; pixels];
    let mut b_lab = vec![0.0f32; pixels];

    for i in 0..pixels {
        l_lab[i] = l[i] as f32 * 100.0 / 255.0;
        a_lab[i] = a[i] as f32 * 254.0 / 255.0 - 127.0;
        b_lab[i] = b[i] as f32 * 254.0 / 255.0 - 127.0;
    }

    (l_lab, a_lab, b_lab)
}

/// Basic LAB histogram matching
///
/// Args:
///     input_srgb: Input image as sRGB values (0-1), flat array HxWx3
///     ref_srgb: Reference image as sRGB values (0-1), flat array HxWx3
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///
/// Returns:
///     Output image as sRGB uint8, flat array HxWx3
pub fn color_correct_basic_lab(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
) -> Vec<u8> {
    let input_pixels = input_width * input_height;

    // Convert to linear RGB
    let mut input_linear = input_srgb.to_vec();
    let mut ref_linear = ref_srgb.to_vec();
    srgb_to_linear(&mut input_linear);
    srgb_to_linear(&mut ref_linear);

    // Convert to LAB
    let input_lab = image_rgb_to_lab(&input_linear, input_width, input_height);
    let ref_lab = image_rgb_to_lab(&ref_linear, ref_width, ref_height);

    // Store original L channel if preserving luminosity
    let original_l: Vec<f32> = if keep_luminosity {
        extract_channel(&input_lab, input_width, input_height, 0)
    } else {
        Vec::new()
    };

    // Scale to uint8 range
    let (input_l, input_a, input_b) = scale_lab_to_uint8(&input_lab, input_width, input_height);
    let (ref_l, ref_a, ref_b) = scale_lab_to_uint8(&ref_lab, ref_width, ref_height);

    // Dither channels
    let input_uint8 =
        dither_channel_stack(&[input_l, input_a, input_b], input_width, input_height);
    let ref_uint8 = dither_channel_stack(&[ref_l, ref_a, ref_b], ref_width, ref_height);

    // Extract dithered channels
    let input_l_u8: Vec<u8> = (0..input_pixels).map(|i| input_uint8[i * 3]).collect();
    let input_a_u8: Vec<u8> = (0..input_pixels).map(|i| input_uint8[i * 3 + 1]).collect();
    let input_b_u8: Vec<u8> = (0..input_pixels).map(|i| input_uint8[i * 3 + 2]).collect();

    let ref_pixels = ref_width * ref_height;
    let ref_l_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 3]).collect();
    let ref_a_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 3 + 1]).collect();
    let ref_b_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 3 + 2]).collect();

    // Match histograms
    let (final_l, final_a, final_b) = if keep_luminosity {
        let matched_a = match_histogram(&input_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&input_b_u8, &ref_b_u8);
        let (_, a_lab, b_lab) = scale_uint8_to_lab(&vec![0u8; input_pixels], &matched_a, &matched_b);
        (original_l, a_lab, b_lab)
    } else {
        let matched_l = match_histogram(&input_l_u8, &ref_l_u8);
        let matched_a = match_histogram(&input_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&input_b_u8, &ref_b_u8);
        scale_uint8_to_lab(&matched_l, &matched_a, &matched_b)
    };

    // Reconstruct LAB image
    let mut final_lab = vec![0.0f32; input_pixels * 3];
    for i in 0..input_pixels {
        final_lab[i * 3] = final_l[i];
        final_lab[i * 3 + 1] = final_a[i];
        final_lab[i * 3 + 2] = final_b[i];
    }

    // Convert back to linear RGB, then sRGB
    let mut final_linear = image_lab_to_rgb(&final_lab, input_width, input_height);
    linear_to_srgb(&mut final_linear);

    // Final dither to uint8
    dither_rgb(&final_linear, input_width, input_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_roundtrip() {
        let lab = vec![50.0, 0.0, 0.0]; // Middle gray in LAB
        let (l, a, b) = scale_lab_to_uint8(&lab, 1, 1);

        // Simulate dithering (just round for test)
        let l_u8 = vec![l[0].round() as u8];
        let a_u8 = vec![a[0].round() as u8];
        let b_u8 = vec![b[0].round() as u8];

        let (l2, a2, b2) = scale_uint8_to_lab(&l_u8, &a_u8, &b_u8);

        assert!((l2[0] - 50.0).abs() < 1.0);
        assert!((a2[0] - 0.0).abs() < 1.0);
        assert!((b2[0] - 0.0).abs() < 1.0);
    }
}
