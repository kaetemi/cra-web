/// CRA (Chroma Rotation Averaging) in Oklab color space.
/// Analogous to cra_lab.rs but using the Oklab color space.

use crate::color::{
    interleave_rgb_u8, linear_rgb_to_oklab_channels, linear_to_srgb_scaled_channels,
    oklab_to_linear_rgb_channels, srgb_to_linear_channels,
};
use crate::dither::{dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, AlignmentMode, InterpolationMode};
use crate::rotation::{compute_oklab_ab_ranges, deg_to_rad, rotate_ab};

/// Default rotation angles for CRA
const ROTATION_ANGLES: [f32; 3] = [0.0, 30.0, 60.0];

/// Default blend factors for iterative refinement
const BLEND_FACTORS: [f32; 3] = [0.25, 0.5, 1.0];

/// Scale L channel to 0-255 range: L (0-1) -> 0-255
fn scale_l_to_255(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0).collect()
}

/// Reverse L scaling: 0-255 -> L (0-1)
fn scale_255_to_l(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v / 255.0).collect()
}

/// Reverse L scaling from uint8: uint8 -> L (0-1)
fn scale_uint8_to_l(l: &[u8]) -> Vec<f32> {
    l.iter().map(|&v| v as f32 / 255.0).collect()
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

    let a_oklab: Vec<f32> = a
        .iter()
        .map(|&v| v as f32 / 255.0 * (a_max - a_min) + a_min)
        .collect();
    let b_oklab: Vec<f32> = b
        .iter()
        .map(|&v| v as f32 / 255.0 * (b_max - b_min) + b_min)
        .collect();

    (a_oklab, b_oklab)
}

/// Reverse f32 (0-255 range) to AB scaling
fn scale_255_to_ab(a: &[f32], b: &[f32], ab_ranges: [[f32; 2]; 2]) -> (Vec<f32>, Vec<f32>) {
    let [a_min, a_max] = ab_ranges[0];
    let [b_min, b_max] = ab_ranges[1];

    let a_oklab: Vec<f32> = a
        .iter()
        .map(|&v| v / 255.0 * (a_max - a_min) + a_min)
        .collect();
    let b_oklab: Vec<f32> = b
        .iter()
        .map(|&v| v / 255.0 * (b_max - b_min) + b_min)
        .collect();

    (a_oklab, b_oklab)
}

/// Process one iteration of Oklab-space histogram matching
///
/// histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
fn process_oklab_iteration(
    current_a: &[f32],
    current_b: &[f32],
    ref_a: &[f32],
    ref_b: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    rotation_angles: &[f32],
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    dither_seed_base: u32,
) -> (Vec<f32>, Vec<f32>) {
    let input_pixels = input_width * input_height;

    let mut all_corrected_a: Vec<Vec<f32>> = Vec::new();
    let mut all_corrected_b: Vec<Vec<f32>> = Vec::new();

    for (pass_idx, &theta_deg) in rotation_angles.iter().enumerate() {
        let theta_rad = deg_to_rad(theta_deg);
        let ab_ranges = compute_oklab_ab_ranges(theta_deg);

        // Rotate AB
        let (a_rot, b_rot) = rotate_ab(current_a, current_b, theta_rad);
        let (ref_a_rot, ref_b_rot) = rotate_ab(ref_a, ref_b, theta_rad);

        // Scale to 0-255 range
        let (a_scaled, b_scaled) = scale_ab_to_uint8(&a_rot, &b_rot, ab_ranges);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a_rot, &ref_b_rot, ab_ranges);

        let (a_matched, b_matched) = if histogram_mode > 0 {
            // Use f32 histogram matching directly (no dithering/quantization)
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
            scale_255_to_ab(&matched_a, &matched_b, ab_ranges)
        } else {
            // Use binned histogram matching with dithering
            // Each pass gets unique seeds: base + pass_idx * 4 + channel offset
            let pass_seed = dither_seed_base + (pass_idx as u32) * 4;
            let input_a_u8 = dither_with_mode(&a_scaled, input_width, input_height, histogram_dither_mode, pass_seed);
            let input_b_u8 = dither_with_mode(&b_scaled, input_width, input_height, histogram_dither_mode, pass_seed + 1);
            let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, histogram_dither_mode, pass_seed + 2);
            let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, pass_seed + 3);

            let matched_a = match_histogram(&input_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&input_b_u8, &ref_b_u8);
            scale_uint8_to_ab(&matched_a, &matched_b, ab_ranges)
        };

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

/// CRA Oklab color correction
///
/// Args:
///     input_srgb: Input image as sRGB values (0-1), flat array HxWx3
///     ref_srgb: Reference image as sRGB values (0-1), flat array HxWx3
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///
/// Returns:
///     Output image as sRGB uint8, flat array HxWx3
pub fn color_correct_cra_oklab(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    output_dither_mode: DitherMode,
) -> Vec<u8> {
    let input_pixels = input_width * input_height;

    // Convert to separate linear RGB channels
    let (in_r, in_g, in_b) = srgb_to_linear_channels(input_srgb, input_width, input_height);
    let (ref_r, ref_g, ref_b) = srgb_to_linear_channels(ref_srgb, ref_width, ref_height);

    // Convert to separate Oklab channels
    let (original_l, in_a, in_b_ch) = linear_rgb_to_oklab_channels(&in_r, &in_g, &in_b);
    let (ref_l, ref_a, ref_b_ch) = linear_rgb_to_oklab_channels(&ref_r, &ref_g, &ref_b);

    // Initialize current AB channels
    let mut current_a = in_a;
    let mut current_b = in_b_ch;

    // Store current L (may be modified)
    let current_l = original_l.clone();

    // Iterative refinement
    // Each iteration gets a unique seed range: iteration_idx * 100 + pass seeds
    for (iter_idx, &blend_factor) in BLEND_FACTORS.iter().enumerate() {
        let (avg_a, avg_b) = process_oklab_iteration(
            &current_a,
            &current_b,
            &ref_a,
            &ref_b_ch,
            input_width,
            input_height,
            ref_width,
            ref_height,
            &ROTATION_ANGLES,
            histogram_mode,
            histogram_dither_mode,
            (iter_idx as u32) * 100,
        );

        // Blend with current
        for i in 0..input_pixels {
            current_a[i] = current_a[i] * (1.0 - blend_factor) + avg_a[i] * blend_factor;
            current_b[i] = current_b[i] * (1.0 - blend_factor) + avg_b[i] * blend_factor;
        }
    }

    // Final Oklab histogram match
    let final_ab_ranges = compute_oklab_ab_ranges(0.0);

    // Scale to 0-255 range
    let l_scaled = scale_l_to_255(&current_l);
    let (a_scaled, b_scaled) = scale_ab_to_uint8(&current_a, &current_b, final_ab_ranges);
    let ref_l_scaled = scale_l_to_255(&ref_l);
    let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a, &ref_b_ch, final_ab_ranges);

    // Match histograms (final pass uses high seed values to avoid collision with rotation passes)
    let (final_l, final_a, final_b) = if histogram_mode > 0 {
        // Use f32 histogram matching directly
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
            let (a_oklab, b_oklab) = scale_255_to_ab(&matched_a, &matched_b, final_ab_ranges);
            (original_l, a_oklab, b_oklab)
        } else {
            let matched_l =
                match_histogram_f32(&l_scaled, &ref_l_scaled, InterpolationMode::Linear, align_mode, 100);
            let matched_a =
                match_histogram_f32(&a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, 101);
            let matched_b =
                match_histogram_f32(&b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 102);
            let l_oklab = scale_255_to_l(&matched_l);
            let (a_oklab, b_oklab) = scale_255_to_ab(&matched_a, &matched_b, final_ab_ranges);
            (l_oklab, a_oklab, b_oklab)
        }
    } else {
        // Use binned histogram matching with dithering
        // Final pass uses high seed values (1000+) to avoid collision with iteration passes
        let current_l_u8 = dither_with_mode(&l_scaled, input_width, input_height, histogram_dither_mode, 1000);
        let current_a_u8 = dither_with_mode(&a_scaled, input_width, input_height, histogram_dither_mode, 1001);
        let current_b_u8 = dither_with_mode(&b_scaled, input_width, input_height, histogram_dither_mode, 1002);
        let ref_l_u8 = dither_with_mode(&ref_l_scaled, ref_width, ref_height, histogram_dither_mode, 1003);
        let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, histogram_dither_mode, 1004);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, 1005);

        if keep_luminosity {
            let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&current_b_u8, &ref_b_u8);
            let (a_oklab, b_oklab) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);
            (original_l, a_oklab, b_oklab)
        } else {
            let matched_l = match_histogram(&current_l_u8, &ref_l_u8);
            let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&current_b_u8, &ref_b_u8);
            let l_oklab = scale_uint8_to_l(&matched_l);
            let (a_oklab, b_oklab) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);
            (l_oklab, a_oklab, b_oklab)
        }
    };

    // Convert Oklab back to linear RGB (separate channels)
    let (out_r, out_g, out_b) = oklab_to_linear_rgb_channels(&final_l, &final_a, &final_b);

    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) = linear_to_srgb_scaled_channels(&out_r, &out_g, &out_b);

    // Dither each channel for final output
    let r_u8 = dither_with_mode(&r_scaled, input_width, input_height, output_dither_mode, 1006);
    let g_u8 = dither_with_mode(&g_scaled, input_width, input_height, output_dither_mode, 1007);
    let b_u8 = dither_with_mode(&b_scaled, input_width, input_height, output_dither_mode, 1008);

    // Interleave only at the very end
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}
