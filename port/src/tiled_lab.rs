/// Tiled CRA (Chroma Rotation Averaging) in LAB color space.
/// Corresponds to color_correction_tiled.py

use crate::color::{
    extract_channel, image_lab_to_rgb, image_rgb_to_lab, linear_to_srgb, srgb_to_linear,
};
use crate::dither::{dither_rgb, floyd_steinberg_dither};
use crate::histogram::match_histogram;
use crate::rotation::{compute_ab_ranges, deg_to_rad, rotate_ab};
use crate::tiling::{
    accumulate_block_single, create_hamming_weights, extract_block_single, generate_tile_blocks,
    normalize_accumulated,
};

/// Default rotation angles for CRA
const ROTATION_ANGLES: [f32; 3] = [0.0, 30.0, 60.0];

/// Default blend factors for iterative refinement
const BLEND_FACTORS: [f32; 3] = [0.25, 0.5, 1.0];

/// Scale L channel to uint8 range
fn scale_l_to_uint8(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0 / 100.0).collect()
}

/// Reverse L scaling
fn scale_uint8_to_l(l: &[u8]) -> Vec<f32> {
    l.iter().map(|&v| v as f32 * 100.0 / 255.0).collect()
}

/// Scale AB values to uint8 range
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

/// Process one iteration for a single block (AB only)
fn process_block_iteration(
    current_a: &[f32],
    current_b: &[f32],
    ref_a: &[f32],
    ref_b: &[f32],
    block_width: usize,
    block_height: usize,
    ref_block_width: usize,
    ref_block_height: usize,
    rotation_angles: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let block_pixels = block_width * block_height;
    let ref_block_pixels = ref_block_width * ref_block_height;

    let mut all_corrected_a: Vec<Vec<f32>> = Vec::new();
    let mut all_corrected_b: Vec<Vec<f32>> = Vec::new();

    for &theta_deg in rotation_angles {
        let theta_rad = deg_to_rad(theta_deg);
        let ab_ranges = compute_ab_ranges(theta_deg);

        // Rotate AB
        let (a_rot, b_rot) = rotate_ab(current_a, current_b, theta_rad);
        let (ref_a_rot, ref_b_rot) = rotate_ab(ref_a, ref_b, theta_rad);

        // Scale and dither each channel directly
        let (a_scaled, b_scaled) = scale_ab_to_uint8(&a_rot, &b_rot, ab_ranges);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a_rot, &ref_b_rot, ab_ranges);

        let input_a_u8 = floyd_steinberg_dither(&a_scaled, block_width, block_height);
        let input_b_u8 = floyd_steinberg_dither(&b_scaled, block_width, block_height);
        let ref_a_u8 = floyd_steinberg_dither(&ref_a_scaled, ref_block_width, ref_block_height);
        let ref_b_u8 = floyd_steinberg_dither(&ref_b_scaled, ref_block_width, ref_block_height);

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
    let avg_a: Vec<f32> = (0..block_pixels)
        .map(|i| all_corrected_a.iter().map(|v| v[i]).sum::<f32>() / num_angles)
        .collect();
    let avg_b: Vec<f32> = (0..block_pixels)
        .map(|i| all_corrected_b.iter().map(|v| v[i]).sum::<f32>() / num_angles)
        .collect();

    (avg_a, avg_b)
}

/// Process one iteration for a single block (with L channel)
fn process_block_iteration_with_l(
    current_l: &[f32],
    current_a: &[f32],
    current_b: &[f32],
    ref_l: &[f32],
    ref_a: &[f32],
    ref_b: &[f32],
    block_width: usize,
    block_height: usize,
    ref_block_width: usize,
    ref_block_height: usize,
    rotation_angles: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let block_pixels = block_width * block_height;
    let ref_block_pixels = ref_block_width * ref_block_height;

    // Process AB channels
    let (avg_a, avg_b) = process_block_iteration(
        current_a,
        current_b,
        ref_a,
        ref_b,
        block_width,
        block_height,
        ref_block_width,
        ref_block_height,
        rotation_angles,
    );

    // Process L channel
    let l_scaled = scale_l_to_uint8(current_l);
    let ref_l_scaled = scale_l_to_uint8(ref_l);

    let l_uint8 = floyd_steinberg_dither(&l_scaled, block_width, block_height);
    let ref_l_uint8 = floyd_steinberg_dither(&ref_l_scaled, ref_block_width, ref_block_height);

    let matched_l = match_histogram(&l_uint8, &ref_l_uint8);
    let avg_l = scale_uint8_to_l(&matched_l);

    (avg_l, avg_a, avg_b)
}

/// Tiled CRA LAB color correction
///
/// Args:
///     input_srgb: Input image as sRGB values (0-1), flat array HxWx3
///     ref_srgb: Reference image as sRGB values (0-1), flat array HxWx3
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     tiled_luminosity: If true, process L channel per-tile before global match
///
/// Returns:
///     Output image as sRGB uint8, flat array HxWx3
pub fn color_correct_tiled_lab(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    tiled_luminosity: bool,
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
    let input_l = extract_channel(&input_lab, input_width, input_height, 0);
    let input_a = extract_channel(&input_lab, input_width, input_height, 1);
    let input_b = extract_channel(&input_lab, input_width, input_height, 2);

    let ref_l = extract_channel(&ref_lab, ref_width, ref_height, 0);
    let ref_a = extract_channel(&ref_lab, ref_width, ref_height, 1);
    let ref_b = extract_channel(&ref_lab, ref_width, ref_height, 2);

    // Store original L for use when tiled_luminosity is disabled
    let original_l = input_l.clone();

    // Generate tile blocks
    let blocks = generate_tile_blocks(input_height, input_width, ref_height, ref_width);

    // Initialize accumulators
    let mut a_acc = vec![0.0f32; input_pixels];
    let mut b_acc = vec![0.0f32; input_pixels];
    let mut weight_acc = vec![0.0f32; input_pixels];
    let mut l_acc = if tiled_luminosity {
        vec![0.0f32; input_pixels]
    } else {
        Vec::new()
    };

    // Process each block
    for block in &blocks {
        let block_height = block.y_end - block.y_start;
        let block_width = block.x_end - block.x_start;
        let ref_block_height = block.ref_y_end - block.ref_y_start;
        let ref_block_width = block.ref_x_end - block.ref_x_start;

        // Extract block data
        let mut current_a = extract_block_single(
            &input_a,
            input_width,
            block.y_start,
            block.y_end,
            block.x_start,
            block.x_end,
        );
        let mut current_b = extract_block_single(
            &input_b,
            input_width,
            block.y_start,
            block.y_end,
            block.x_start,
            block.x_end,
        );
        let ref_a_block = extract_block_single(
            &ref_a,
            ref_width,
            block.ref_y_start,
            block.ref_y_end,
            block.ref_x_start,
            block.ref_x_end,
        );
        let ref_b_block = extract_block_single(
            &ref_b,
            ref_width,
            block.ref_y_start,
            block.ref_y_end,
            block.ref_x_start,
            block.ref_x_end,
        );

        let (mut current_l, ref_l_block) = if tiled_luminosity {
            (
                extract_block_single(
                    &input_l,
                    input_width,
                    block.y_start,
                    block.y_end,
                    block.x_start,
                    block.x_end,
                ),
                extract_block_single(
                    &ref_l,
                    ref_width,
                    block.ref_y_start,
                    block.ref_y_end,
                    block.ref_x_start,
                    block.ref_x_end,
                ),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Iterative correction
        for &blend_factor in &BLEND_FACTORS {
            if tiled_luminosity {
                let (avg_l, avg_a, avg_b) = process_block_iteration_with_l(
                    &current_l,
                    &current_a,
                    &current_b,
                    &ref_l_block,
                    &ref_a_block,
                    &ref_b_block,
                    block_width,
                    block_height,
                    ref_block_width,
                    ref_block_height,
                    &ROTATION_ANGLES,
                );
                let block_pixels = block_width * block_height;
                for i in 0..block_pixels {
                    current_l[i] = current_l[i] * (1.0 - blend_factor) + avg_l[i] * blend_factor;
                }
                for i in 0..block_pixels {
                    current_a[i] = current_a[i] * (1.0 - blend_factor) + avg_a[i] * blend_factor;
                    current_b[i] = current_b[i] * (1.0 - blend_factor) + avg_b[i] * blend_factor;
                }
            } else {
                let (avg_a, avg_b) = process_block_iteration(
                    &current_a,
                    &current_b,
                    &ref_a_block,
                    &ref_b_block,
                    block_width,
                    block_height,
                    ref_block_width,
                    ref_block_height,
                    &ROTATION_ANGLES,
                );
                let block_pixels = block_width * block_height;
                for i in 0..block_pixels {
                    current_a[i] = current_a[i] * (1.0 - blend_factor) + avg_a[i] * blend_factor;
                    current_b[i] = current_b[i] * (1.0 - blend_factor) + avg_b[i] * blend_factor;
                }
            }
        }

        // Create Hamming weights for smooth blending
        let weights = create_hamming_weights(block_height, block_width);

        // Accumulate weighted results
        accumulate_block_single(
            &mut a_acc,
            &mut weight_acc,
            input_width,
            &current_a,
            &weights,
            block.y_start,
            block.y_end,
            block.x_start,
            block.x_end,
        );

        // Need separate weight accumulator for B (but same weights)
        let mut b_weight_acc = vec![0.0f32; input_pixels];
        accumulate_block_single(
            &mut b_acc,
            &mut b_weight_acc,
            input_width,
            &current_b,
            &weights,
            block.y_start,
            block.y_end,
            block.x_start,
            block.x_end,
        );

        if tiled_luminosity {
            let mut l_weight_acc = vec![0.0f32; input_pixels];
            accumulate_block_single(
                &mut l_acc,
                &mut l_weight_acc,
                input_width,
                &current_l,
                &weights,
                block.y_start,
                block.y_end,
                block.x_start,
                block.x_end,
            );
        }
    }

    // Normalize accumulated results
    normalize_accumulated(&mut a_acc, &weight_acc);
    normalize_accumulated(&mut b_acc, &weight_acc);

    let final_l_input = if tiled_luminosity {
        normalize_accumulated(&mut l_acc, &weight_acc);
        l_acc
    } else {
        original_l.clone()
    };

    // Final global histogram match
    let final_ab_ranges = compute_ab_ranges(0.0);

    // Scale and dither each channel directly
    let l_scaled = scale_l_to_uint8(&final_l_input);
    let (a_scaled, b_scaled) = scale_ab_to_uint8(&a_acc, &b_acc, final_ab_ranges);
    let current_l_u8 = floyd_steinberg_dither(&l_scaled, input_width, input_height);
    let current_a_u8 = floyd_steinberg_dither(&a_scaled, input_width, input_height);
    let current_b_u8 = floyd_steinberg_dither(&b_scaled, input_width, input_height);

    let ref_l_scaled = scale_l_to_uint8(&ref_l);
    let (ref_a_scaled, ref_b_scaled) = scale_ab_to_uint8(&ref_a, &ref_b, final_ab_ranges);
    let ref_l_u8 = floyd_steinberg_dither(&ref_l_scaled, ref_width, ref_height);
    let ref_a_u8 = floyd_steinberg_dither(&ref_a_scaled, ref_width, ref_height);
    let ref_b_u8 = floyd_steinberg_dither(&ref_b_scaled, ref_width, ref_height);

    // Match histograms for all channels
    let matched_l = match_histogram(&current_l_u8, &ref_l_u8);
    let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
    let matched_b = match_histogram(&current_b_u8, &ref_b_u8);

    let final_l = scale_uint8_to_l(&matched_l);
    let (final_a, final_b) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);

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
