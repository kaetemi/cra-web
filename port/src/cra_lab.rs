/// CRA (Chroma Rotation Averaging) in LAB color space.
/// Corresponds to color_correction_cra.py

use crate::color::{
    interleave_rgb_u8, lab_to_linear_rgb_channels, linear_rgb_to_lab_channels,
    linear_to_srgb_scaled_channels, srgb_to_linear_channels,
};
use crate::dither::{dither_with_mode, DitherMode};
use crate::dither_colorspace_lab::{
    lab_space_dither_with_mode, LabQuantParams, LabQuantSpace,
};
use crate::dither_common::PerceptualSpace;
use crate::histogram::{match_histogram, match_histogram_f32, AlignmentMode, InterpolationMode};
use crate::rotation::{compute_ab_ranges, deg_to_rad, rotate_ab};

/// Default rotation angles for CRA
const ROTATION_ANGLES: [f32; 3] = [0.0, 30.0, 60.0];

/// Default blend factors for iterative refinement
const BLEND_FACTORS: [f32; 3] = [0.25, 0.5, 1.0];

/// Scale L channel to 0-255 range: L (0-100) -> 0-255
fn scale_l_to_255(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0 / 100.0).collect()
}

/// Reverse L scaling: 0-255 -> L (0-100)
fn scale_255_to_l(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 100.0 / 255.0).collect()
}

/// Reverse L scaling from uint8: uint8 -> L (0-100)
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

/// Reverse f32 (0-255 range) to AB scaling
fn scale_255_to_ab(a: &[f32], b: &[f32], ab_ranges: [[f32; 2]; 2]) -> (Vec<f32>, Vec<f32>) {
    let [a_min, a_max] = ab_ranges[0];
    let [b_min, b_max] = ab_ranges[1];

    let a_lab: Vec<f32> = a
        .iter()
        .map(|&v| v / 255.0 * (a_max - a_min) + a_min)
        .collect();
    let b_lab: Vec<f32> = b
        .iter()
        .map(|&v| v / 255.0 * (b_max - b_min) + b_min)
        .collect();

    (a_lab, b_lab)
}

/// Build LabQuantParams from ab_ranges for CIELAB
/// Maps L: 0-100 -> 0-255, a/b: [min,max] -> 0-255
fn lab_quant_params_from_ranges(ab_ranges: [[f32; 2]; 2], quantize_l: bool, rotation_deg: f32) -> LabQuantParams {
    let [a_min, a_max] = ab_ranges[0];
    let [b_min, b_max] = ab_ranges[1];

    LabQuantParams {
        quantize_l,
        rotation_deg,
        scale_l: 255.0 / 100.0,
        offset_l: 0.0,
        scale_a: 255.0 / (a_max - a_min),
        offset_a: -a_min * 255.0 / (a_max - a_min),
        scale_b: 255.0 / (b_max - b_min),
        offset_b: -b_min * 255.0 / (b_max - b_min),
    }
}

/// Process one iteration of LAB-space histogram matching
///
/// histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
/// color_aware: if true and histogram_mode == 0, use color-aware Lab dithering
/// distance_space: perceptual space for color-aware dithering distance metric
#[allow(clippy::too_many_arguments)]
fn process_lab_iteration(
    current_l: &[f32],
    current_a: &[f32],
    current_b: &[f32],
    ref_l: &[f32],
    ref_a: &[f32],
    ref_b: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    rotation_angles: &[f32],
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    color_aware: bool,
    distance_space: PerceptualSpace,
    dither_seed_base: u32,
) -> (Vec<f32>, Vec<f32>) {
    let input_pixels = input_width * input_height;

    let mut all_corrected_a: Vec<Vec<f32>> = Vec::new();
    let mut all_corrected_b: Vec<Vec<f32>> = Vec::new();

    for (pass_idx, &theta_deg) in rotation_angles.iter().enumerate() {
        let theta_rad = deg_to_rad(theta_deg);
        let ab_ranges = compute_ab_ranges(theta_deg);

        let (a_matched, b_matched) = if histogram_mode > 0 {
            // Use f32 histogram matching directly (no dithering/quantization)
            // Rotate AB externally for f32 mode
            let (a_rot, b_rot) = rotate_ab(current_a, current_b, theta_rad);
            let (ref_a_rot, ref_b_rot) = rotate_ab(ref_a, ref_b, theta_rad);

            // Scale to 0-255 range
            let (a_scaled, b_scaled) = scale_ab_to_uint8(&a_rot, &b_rot, ab_ranges);
            let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a_rot, &ref_b_rot, ab_ranges);

            let align_mode = if histogram_mode == 2 {
                AlignmentMode::Midpoint
            } else {
                AlignmentMode::Endpoint
            };
            // Use different seeds per pass and channel for noise averaging
            let seed_a = (pass_idx * 2) as u32;
            let seed_b = (pass_idx * 2 + 1) as u32;
            let matched_a =
                match_histogram_f32(&a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, seed_a);
            let matched_b =
                match_histogram_f32(&b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, seed_b);
            let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b, ab_ranges);

            // Rotate back
            rotate_ab(&a_lab, &b_lab, -theta_rad)
        } else if color_aware {
            // Use color-aware Lab dithering - rotation handled internally
            let params = lab_quant_params_from_ranges(ab_ranges, false, theta_deg);
            let pass_seed = dither_seed_base + (pass_idx as u32) * 2;

            // Dither input (L not quantized, but used for distance)
            // Pass unrotated values - rotation happens inside dither function
            let (_, input_a_u8, input_b_u8) = lab_space_dither_with_mode(
                current_l, current_a, current_b,
                input_width, input_height,
                &params,
                LabQuantSpace::CIELab,
                distance_space,
                histogram_dither_mode.into(),
                pass_seed,
            );

            // Dither reference
            let (_, ref_a_u8, ref_b_u8) = lab_space_dither_with_mode(
                ref_l, ref_a, ref_b,
                ref_width, ref_height,
                &params,
                LabQuantSpace::CIELab,
                distance_space,
                histogram_dither_mode.into(),
                pass_seed + 1,
            );

            // Histogram match in rotated u8 space
            let matched_a = match_histogram(&input_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&input_b_u8, &ref_b_u8);

            // Convert back to Lab and unrotate
            let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b, ab_ranges);
            rotate_ab(&a_lab, &b_lab, -theta_rad)
        } else {
            // Use channel-independent binned histogram matching with dithering
            // Rotate AB externally
            let (a_rot, b_rot) = rotate_ab(current_a, current_b, theta_rad);
            let (ref_a_rot, ref_b_rot) = rotate_ab(ref_a, ref_b, theta_rad);

            // Scale to 0-255 range
            let (a_scaled, b_scaled) = scale_ab_to_uint8(&a_rot, &b_rot, ab_ranges);
            let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a_rot, &ref_b_rot, ab_ranges);

            // Each pass gets unique seeds: base + pass_idx * 4 + channel offset
            let pass_seed = dither_seed_base + (pass_idx as u32) * 4;
            let input_a_u8 = dither_with_mode(&a_scaled, input_width, input_height, histogram_dither_mode, pass_seed);
            let input_b_u8 = dither_with_mode(&b_scaled, input_width, input_height, histogram_dither_mode, pass_seed + 1);
            let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, histogram_dither_mode, pass_seed + 2);
            let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, pass_seed + 3);

            let matched_a = match_histogram(&input_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&input_b_u8, &ref_b_u8);
            let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b, ab_ranges);

            // Rotate back
            rotate_ab(&a_lab, &b_lab, -theta_rad)
        };

        all_corrected_a.push(a_matched);
        all_corrected_b.push(b_matched);
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
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram preparation
///     color_aware_histogram: If true and histogram_mode == 0, use color-aware Lab dithering
///     histogram_distance_space: Perceptual space for color-aware histogram dithering
///     output_dither_mode: Dither mode for final RGB output
///
/// Returns:
///     Output image as sRGB uint8, flat array HxWx3
#[allow(clippy::too_many_arguments)]
pub fn color_correct_cra_lab(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    color_aware_histogram: bool,
    histogram_distance_space: PerceptualSpace,
    output_dither_mode: DitherMode,
) -> Vec<u8> {
    let input_pixels = input_width * input_height;

    // Convert to separate linear RGB channels
    let (in_r, in_g, in_b) = srgb_to_linear_channels(input_srgb, input_width, input_height);
    let (ref_r, ref_g, ref_b) = srgb_to_linear_channels(ref_srgb, ref_width, ref_height);

    // Convert to separate LAB channels
    let (original_l, in_a, in_b_ch) = linear_rgb_to_lab_channels(&in_r, &in_g, &in_b);
    let (ref_l, ref_a, ref_b_ch) = linear_rgb_to_lab_channels(&ref_r, &ref_g, &ref_b);

    // Initialize current AB channels
    let mut current_a = in_a;
    let mut current_b = in_b_ch;

    // Store current L (may be modified)
    let current_l = original_l.clone();

    // Iterative refinement
    // Each iteration gets a unique seed range: iteration_idx * 100 + pass seeds
    for (iter_idx, &blend_factor) in BLEND_FACTORS.iter().enumerate() {
        let (avg_a, avg_b) = process_lab_iteration(
            &current_l,
            &current_a,
            &current_b,
            &ref_l,
            &ref_a,
            &ref_b_ch,
            input_width,
            input_height,
            ref_width,
            ref_height,
            &ROTATION_ANGLES,
            histogram_mode,
            histogram_dither_mode,
            color_aware_histogram,
            histogram_distance_space,
            (iter_idx as u32) * 100,
        );

        // Blend with current
        for i in 0..input_pixels {
            current_a[i] = current_a[i] * (1.0 - blend_factor) + avg_a[i] * blend_factor;
            current_b[i] = current_b[i] * (1.0 - blend_factor) + avg_b[i] * blend_factor;
        }
    }

    // Final LAB histogram match
    let final_ab_ranges = compute_ab_ranges(0.0);

    // Match histograms (final pass uses high seed values to avoid collision with rotation passes)
    let (final_l, final_a, final_b) = if histogram_mode > 0 {
        // Use f32 histogram matching directly (no dithering needed)
        let l_scaled = scale_l_to_255(&current_l);
        let (a_scaled, b_scaled) = scale_ab_to_uint8(&current_a, &current_b, final_ab_ranges);
        let ref_l_scaled = scale_l_to_255(&ref_l);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a, &ref_b_ch, final_ab_ranges);

        let align_mode = if histogram_mode == 2 {
            AlignmentMode::Midpoint
        } else {
            AlignmentMode::Endpoint
        };
        if keep_luminosity {
            let matched_a =
                match_histogram_f32(&a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, 100);
            let matched_b =
                match_histogram_f32(&b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 101);
            let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b, final_ab_ranges);
            (original_l, a_lab, b_lab)
        } else {
            let matched_l =
                match_histogram_f32(&l_scaled, &ref_l_scaled, InterpolationMode::Linear, align_mode, 100);
            let matched_a =
                match_histogram_f32(&a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, 101);
            let matched_b =
                match_histogram_f32(&b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 102);
            let l_lab = scale_255_to_l(&matched_l);
            let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b, final_ab_ranges);
            (l_lab, a_lab, b_lab)
        }
    } else if color_aware_histogram {
        // Use color-aware Lab dithering for final histogram matching
        // Final pass has no rotation (0 degrees)
        let params = lab_quant_params_from_ranges(final_ab_ranges, !keep_luminosity, 0.0);

        // Dither input
        let (current_l_u8, current_a_u8, current_b_u8) = lab_space_dither_with_mode(
            &current_l, &current_a, &current_b,
            input_width, input_height,
            &params,
            LabQuantSpace::CIELab,
            histogram_distance_space,
            histogram_dither_mode.into(),
            1000,
        );

        // Dither reference
        let (ref_l_u8, ref_a_u8, ref_b_u8) = lab_space_dither_with_mode(
            &ref_l, &ref_a, &ref_b_ch,
            ref_width, ref_height,
            &params,
            LabQuantSpace::CIELab,
            histogram_distance_space,
            histogram_dither_mode.into(),
            1001,
        );

        if keep_luminosity {
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
        }
    } else {
        // Use channel-independent binned histogram matching with dithering
        // Final pass uses high seed values (1000+) to avoid collision with iteration passes
        let l_scaled = scale_l_to_255(&current_l);
        let (a_scaled, b_scaled) = scale_ab_to_uint8(&current_a, &current_b, final_ab_ranges);
        let ref_l_scaled = scale_l_to_255(&ref_l);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a, &ref_b_ch, final_ab_ranges);

        let current_l_u8 = dither_with_mode(&l_scaled, input_width, input_height, histogram_dither_mode, 1000);
        let current_a_u8 = dither_with_mode(&a_scaled, input_width, input_height, histogram_dither_mode, 1001);
        let current_b_u8 = dither_with_mode(&b_scaled, input_width, input_height, histogram_dither_mode, 1002);
        let ref_l_u8 = dither_with_mode(&ref_l_scaled, ref_width, ref_height, histogram_dither_mode, 1003);
        let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, histogram_dither_mode, 1004);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, 1005);

        if keep_luminosity {
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
        }
    };

    // Convert LAB back to linear RGB (separate channels)
    let (out_r, out_g, out_b) = lab_to_linear_rgb_channels(&final_l, &final_a, &final_b);

    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) = linear_to_srgb_scaled_channels(&out_r, &out_g, &out_b);

    // Dither each channel for final output
    let r_u8 = dither_with_mode(&r_scaled, input_width, input_height, output_dither_mode, 1006);
    let g_u8 = dither_with_mode(&g_scaled, input_width, input_height, output_dither_mode, 1007);
    let b_u8 = dither_with_mode(&b_scaled, input_width, input_height, output_dither_mode, 1008);

    // Interleave only at the very end
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}
