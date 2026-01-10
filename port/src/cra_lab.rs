/// CRA (Chroma Rotation Averaging) in LAB color space.
/// Corresponds to color_correction_cra.py

use crate::color::{
    extract_channel, image_lab_to_rgb, image_rgb_to_lab, linear_to_srgb, srgb_to_linear,
};
use crate::dither::{dither_channel_stack, dither_rgb};
use crate::histogram::match_histogram;
use crate::rotation::{compute_ab_ranges, deg_to_rad, rotate_ab};

/// Default rotation angles for CRA
const ROTATION_ANGLES: [f32; 3] = [0.0, 30.0, 60.0];

/// Default blend factors for iterative refinement
const BLEND_FACTORS: [f32; 3] = [0.25, 0.5, 1.0];

/// Scale L channel to uint8 range: L (0-100) -> 0-255
fn scale_l_to_uint8(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0 / 100.0).collect()
}

/// Reverse L scaling: uint8 -> L (0-100)
fn scale_uint8_to_l(l: &[u8]) -> Vec<f32> {
    l.iter().map(|&v| v as f32 * 100.0 / 255.0).collect()
}

/// Scale AB values to uint8 range based on precomputed ranges
fn scale_ab_to_uint8(a: &[f32], b: &[f32], ab_ranges: [[f32; 2]; 2]) -> (Vec<f32>, Vec<f32>) {
    let [a_min, a_max] = ab_ranges[0];
    let [b_min, b_max] = ab_ranges[1];

    let a_scaled: Vec<f32> = a
        .iter()
        .map(|&v| (v - a_min) / (a_max - a_min) * 255.0)
        .collect();
    let b_scaled: Vec<f32> = b
        .iter()
        .map(|&v| (v - b_min) / (b_max - b_min) * 255.0)
        .collect();

    (a_scaled, b_scaled)
}

/// Reverse uint8 to AB scaling
fn scale_uint8_to_ab(a: &[u8], b: &[u8], ab_ranges: [[f32; 2]; 2]) -> (Vec<f32>, Vec<f32>) {
    let [a_min, a_max] = ab_ranges[0];
    let [b_min, b_max] = ab_ranges[1];

    let a_lab: Vec<f32> = a
        .iter()
        .map(|&v| v as f32 / 255.0 * (a_max - a_min) + a_min)
        .collect();
    let b_lab: Vec<f32> = b
        .iter()
        .map(|&v| v as f32 / 255.0 * (b_max - b_min) + b_min)
        .collect();

    (a_lab, b_lab)
}

/// Process one iteration of LAB-space histogram matching
fn process_lab_iteration(
    current_a: &[f32],
    current_b: &[f32],
    ref_a: &[f32],
    ref_b: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    rotation_angles: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let input_pixels = input_width * input_height;
    let ref_pixels = ref_width * ref_height;

    let mut all_corrected_a: Vec<Vec<f32>> = Vec::new();
    let mut all_corrected_b: Vec<Vec<f32>> = Vec::new();

    for &theta_deg in rotation_angles {
        let theta_rad = deg_to_rad(theta_deg);
        let ab_ranges = compute_ab_ranges(theta_deg);

        // Rotate AB
        let (a_rot, b_rot) = rotate_ab(current_a, current_b, theta_rad);
        let (ref_a_rot, ref_b_rot) = rotate_ab(ref_a, ref_b, theta_rad);

        // Scale and dither
        let (a_scaled, b_scaled) = scale_ab_to_uint8(&a_rot, &b_rot, ab_ranges);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a_rot, &ref_b_rot, ab_ranges);

        let input_uint8 = dither_channel_stack(&[a_scaled, b_scaled], input_width, input_height);
        let ref_uint8 = dither_channel_stack(&[ref_a_scaled, ref_b_scaled], ref_width, ref_height);

        // Extract channels
        let input_a_u8: Vec<u8> = (0..input_pixels).map(|i| input_uint8[i * 2]).collect();
        let input_b_u8: Vec<u8> = (0..input_pixels).map(|i| input_uint8[i * 2 + 1]).collect();
        let ref_a_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 2]).collect();
        let ref_b_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 2 + 1]).collect();

        // Match histograms
        let matched_a = match_histogram(&input_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&input_b_u8, &ref_b_u8);

        // Scale back
        let (a_matched, b_matched) = scale_uint8_to_ab(&matched_a, &matched_b, ab_ranges);

        // Rotate back
        let (a_back, b_back) = rotate_ab(&a_matched, &b_matched, -theta_rad);

        all_corrected_a.push(a_back);
        all_corrected_b.push(b_back);
    }

    // Average all corrections
    let num_angles = all_corrected_a.len() as f32;
    let avg_a: Vec<f32> = (0..input_pixels)
        .map(|i| all_corrected_a.iter().map(|v| v[i]).sum::<f32>() / num_angles)
        .collect();
    let avg_b: Vec<f32> = (0..input_pixels)
        .map(|i| all_corrected_b.iter().map(|v| v[i]).sum::<f32>() / num_angles)
        .collect();

    (avg_a, avg_b)
}

/// CRA LAB color correction
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
pub fn color_correct_cra_lab(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
) -> Vec<u8> {
    let input_pixels = input_width * input_height;
    let ref_pixels = ref_width * ref_height;

    // Convert to linear RGB
    let mut input_linear = input_srgb.to_vec();
    let mut ref_linear = ref_srgb.to_vec();
    srgb_to_linear(&mut input_linear);
    srgb_to_linear(&mut ref_linear);

    // Convert to LAB
    let input_lab = image_rgb_to_lab(&input_linear, input_width, input_height);
    let ref_lab = image_rgb_to_lab(&ref_linear, ref_width, ref_height);

    // Extract channels
    let original_l = extract_channel(&input_lab, input_width, input_height, 0);
    let mut current_a = extract_channel(&input_lab, input_width, input_height, 1);
    let mut current_b = extract_channel(&input_lab, input_width, input_height, 2);

    let ref_l = extract_channel(&ref_lab, ref_width, ref_height, 0);
    let ref_a = extract_channel(&ref_lab, ref_width, ref_height, 1);
    let ref_b = extract_channel(&ref_lab, ref_width, ref_height, 2);

    // Store current L (may be modified)
    let current_l = original_l.clone();

    // Iterative refinement
    for &blend_factor in &BLEND_FACTORS {
        let (avg_a, avg_b) = process_lab_iteration(
            &current_a,
            &current_b,
            &ref_a,
            &ref_b,
            input_width,
            input_height,
            ref_width,
            ref_height,
            &ROTATION_ANGLES,
        );

        // Blend with current
        for i in 0..input_pixels {
            current_a[i] = current_a[i] * (1.0 - blend_factor) + avg_a[i] * blend_factor;
            current_b[i] = current_b[i] * (1.0 - blend_factor) + avg_b[i] * blend_factor;
        }
    }

    // Final LAB histogram match
    let final_ab_ranges = compute_ab_ranges(0.0);

    let l_scaled = scale_l_to_uint8(&current_l);
    let (a_scaled, b_scaled) = scale_ab_to_uint8(&current_a, &current_b, final_ab_ranges);
    let current_uint8 = dither_channel_stack(&[l_scaled, a_scaled, b_scaled], input_width, input_height);

    let ref_l_scaled = scale_l_to_uint8(&ref_l);
    let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a, &ref_b, final_ab_ranges);
    let ref_uint8 = dither_channel_stack(&[ref_l_scaled, ref_a_scaled, ref_b_scaled], ref_width, ref_height);

    // Extract channels
    let current_l_u8: Vec<u8> = (0..input_pixels).map(|i| current_uint8[i * 3]).collect();
    let current_a_u8: Vec<u8> = (0..input_pixels).map(|i| current_uint8[i * 3 + 1]).collect();
    let current_b_u8: Vec<u8> = (0..input_pixels).map(|i| current_uint8[i * 3 + 2]).collect();

    let ref_l_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 3]).collect();
    let ref_a_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 3 + 1]).collect();
    let ref_b_u8: Vec<u8> = (0..ref_pixels).map(|i| ref_uint8[i * 3 + 2]).collect();

    // Match histograms
    let (final_l, final_a, final_b) = if keep_luminosity {
        let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&current_b_u8, &ref_b_u8);
        let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);
        (original_l, a_lab, b_lab)
    } else {
        let matched_l = match_histogram(&current_l_u8, &ref_l_u8);
        let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&current_b_u8, &ref_b_u8);
        let l_lab = scale_uint8_to_l(&matched_l);
        let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);
        (l_lab, a_lab, b_lab)
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
