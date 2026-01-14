/// Tiled CRA (Chroma Rotation Averaging) in LAB/OkLab color space.
/// Supports both CIELAB and OkLab color spaces via the colorspace parameter.

use crate::color::{
    lab_to_linear_rgb_channels, linear_rgb_to_lab_channels, linear_rgb_to_oklab_channels,
    oklab_to_linear_rgb_channels,
};
use crate::dither::{dither_with_mode, DitherMode};
use crate::dither_colorspace_lab::{
    lab_space_dither_with_mode, LabQuantParams, LabQuantSpace,
};
use crate::dither_common::PerceptualSpace;
use crate::histogram::{match_histogram, match_histogram_f32, AlignmentMode, InterpolationMode};
use crate::rotation::{compute_ab_ranges, compute_oklab_ab_ranges, deg_to_rad, rotate_ab};
use crate::tiling::{
    accumulate_block_single, create_hamming_weights, extract_block_single, generate_tile_blocks,
    normalize_accumulated,
};

/// Default rotation angles for CRA
const ROTATION_ANGLES: [f32; 3] = [0.0, 30.0, 60.0];

/// Default blend factors for iterative refinement
const BLEND_FACTORS: [f32; 3] = [0.25, 0.5, 1.0];

/// Get L scale factor based on colorspace
/// CIELAB L: 0-100, OkLab L: 0-1
fn l_scale_factor(colorspace: LabQuantSpace) -> f32 {
    match colorspace {
        LabQuantSpace::CIELab => 100.0,
        LabQuantSpace::OkLab => 1.0,
    }
}

/// Scale L channel to 0-255 range
fn scale_l_to_255(l: &[f32], colorspace: LabQuantSpace) -> Vec<f32> {
    let scale = l_scale_factor(colorspace);
    l.iter().map(|&v| v * 255.0 / scale).collect()
}

/// Reverse L scaling: 0-255 -> native range
fn scale_255_to_l(l: &[f32], colorspace: LabQuantSpace) -> Vec<f32> {
    let scale = l_scale_factor(colorspace);
    l.iter().map(|&v| v * scale / 255.0).collect()
}

/// Reverse L scaling: uint8 -> native range
fn scale_uint8_to_l(l: &[u8], colorspace: LabQuantSpace) -> Vec<f32> {
    let scale = l_scale_factor(colorspace);
    l.iter().map(|&v| v as f32 * scale / 255.0).collect()
}

/// Get AB ranges based on colorspace and rotation angle
fn get_ab_ranges(theta_deg: f32, colorspace: LabQuantSpace) -> [[f32; 2]; 2] {
    match colorspace {
        LabQuantSpace::CIELab => compute_ab_ranges(theta_deg),
        LabQuantSpace::OkLab => compute_oklab_ab_ranges(theta_deg),
    }
}

/// Scale AB values to 0-255 range
fn scale_ab_to_255(a: &[f32], b: &[f32], ab_ranges: [[f32; 2]; 2]) -> (Vec<f32>, Vec<f32>) {
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

/// Build LabQuantParams from ab_ranges for the given colorspace
fn lab_quant_params_from_ranges(
    ab_ranges: [[f32; 2]; 2],
    quantize_l: bool,
    rotation_deg: f32,
    colorspace: LabQuantSpace,
) -> LabQuantParams {
    let [a_min, a_max] = ab_ranges[0];
    let [b_min, b_max] = ab_ranges[1];

    let scale_l = match colorspace {
        LabQuantSpace::CIELab => 255.0 / 100.0, // L: 0-100 -> 0-255
        LabQuantSpace::OkLab => 255.0,          // L: 0-1 -> 0-255
    };

    LabQuantParams {
        quantize_l,
        rotation_deg,
        scale_l,
        offset_l: 0.0,
        scale_a: 255.0 / (a_max - a_min),
        offset_a: -a_min * 255.0 / (a_max - a_min),
        scale_b: 255.0 / (b_max - b_min),
        offset_b: -b_min * 255.0 / (b_max - b_min),
    }
}

/// Convert linear RGB to Lab channels based on colorspace
fn linear_rgb_to_lab(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    colorspace: LabQuantSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    match colorspace {
        LabQuantSpace::CIELab => linear_rgb_to_lab_channels(r, g, b),
        LabQuantSpace::OkLab => linear_rgb_to_oklab_channels(r, g, b),
    }
}

/// Convert Lab channels to linear RGB based on colorspace
fn lab_to_linear_rgb(
    l: &[f32],
    a: &[f32],
    b: &[f32],
    colorspace: LabQuantSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    match colorspace {
        LabQuantSpace::CIELab => lab_to_linear_rgb_channels(l, a, b),
        LabQuantSpace::OkLab => oklab_to_linear_rgb_channels(l, a, b),
    }
}

/// Process one iteration for a single block (AB only)
///
/// histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
/// color_aware: if true and histogram_mode == 0, use color-aware Lab dithering
/// distance_space: perceptual space for color-aware dithering distance metric
#[allow(clippy::too_many_arguments)]
fn process_block_iteration(
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
    histogram_mode: u8,
    block_seed: u32,
    histogram_dither_mode: DitherMode,
    color_aware: bool,
    distance_space: PerceptualSpace,
    colorspace: LabQuantSpace,
) -> (Vec<f32>, Vec<f32>) {
    let block_pixels = block_width * block_height;

    let mut all_corrected_a: Vec<Vec<f32>> = Vec::new();
    let mut all_corrected_b: Vec<Vec<f32>> = Vec::new();

    for (pass_idx, &theta_deg) in rotation_angles.iter().enumerate() {
        let theta_rad = deg_to_rad(theta_deg);
        let ab_ranges = get_ab_ranges(theta_deg, colorspace);

        let (a_matched, b_matched) = if histogram_mode > 0 {
            // Use f32 histogram matching directly (no dithering/quantization)
            // Rotate AB externally for f32 mode
            let (a_rot, b_rot) = rotate_ab(current_a, current_b, theta_rad);
            let (ref_a_rot, ref_b_rot) = rotate_ab(ref_a, ref_b, theta_rad);

            // Scale to 0-255 range
            let (a_scaled, b_scaled) = scale_ab_to_255(&a_rot, &b_rot, ab_ranges);
            let (ref_a_scaled, ref_b_scaled) = scale_ab_to_255(&ref_a_rot, &ref_b_rot, ab_ranges);

            let align_mode = if histogram_mode == 2 {
                AlignmentMode::Midpoint
            } else {
                AlignmentMode::Endpoint
            };
            // Different seeds per block, pass, and channel for noise averaging
            let seed_a = block_seed.wrapping_mul(10) + (pass_idx as u32 * 2);
            let seed_b = block_seed.wrapping_mul(10) + (pass_idx as u32 * 2 + 1);
            let matched_a =
                match_histogram_f32(&a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, seed_a);
            let matched_b =
                match_histogram_f32(&b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, seed_b);
            let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b, ab_ranges);

            // Rotate back
            rotate_ab(&a_lab, &b_lab, -theta_rad)
        } else if color_aware {
            // Use color-aware Lab dithering - rotation handled internally
            let params = lab_quant_params_from_ranges(ab_ranges, false, theta_deg, colorspace);
            let pass_seed = block_seed.wrapping_mul(1000) + (pass_idx as u32) * 2;

            // Dither input (L not quantized, but used for distance)
            // Pass unrotated values - rotation happens inside dither function
            let (_, input_a_u8, input_b_u8) = lab_space_dither_with_mode(
                current_l, current_a, current_b,
                block_width, block_height,
                &params,
                colorspace,
                distance_space,
                histogram_dither_mode.into(),
                pass_seed,
            );

            // Dither reference
            let (_, ref_a_u8, ref_b_u8) = lab_space_dither_with_mode(
                ref_l, ref_a, ref_b,
                ref_block_width, ref_block_height,
                &params,
                colorspace,
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
            let (a_scaled, b_scaled) = scale_ab_to_255(&a_rot, &b_rot, ab_ranges);
            let (ref_a_scaled, ref_b_scaled) = scale_ab_to_255(&ref_a_rot, &ref_b_rot, ab_ranges);

            // Each pass gets unique seeds: block_seed * 1000 + pass_idx * 4 + channel offset
            let pass_seed = block_seed.wrapping_mul(1000) + (pass_idx as u32) * 4;
            let input_a_u8 = dither_with_mode(&a_scaled, block_width, block_height, histogram_dither_mode, pass_seed);
            let input_b_u8 = dither_with_mode(&b_scaled, block_width, block_height, histogram_dither_mode, pass_seed + 1);
            let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_block_width, ref_block_height, histogram_dither_mode, pass_seed + 2);
            let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_block_width, ref_block_height, histogram_dither_mode, pass_seed + 3);

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
    let avg_a: Vec<f32> = (0..block_pixels)
        .map(|i| all_corrected_a.iter().map(|v| v[i]).sum::<f32>() / num_angles)
        .collect();
    let avg_b: Vec<f32> = (0..block_pixels)
        .map(|i| all_corrected_b.iter().map(|v| v[i]).sum::<f32>() / num_angles)
        .collect();

    (avg_a, avg_b)
}

/// Process one iteration for a single block (with L channel)
///
/// histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
/// color_aware: if true and histogram_mode == 0, use color-aware Lab dithering
/// distance_space: perceptual space for color-aware dithering distance metric
#[allow(clippy::too_many_arguments)]
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
    histogram_mode: u8,
    block_seed: u32,
    histogram_dither_mode: DitherMode,
    color_aware: bool,
    distance_space: PerceptualSpace,
    colorspace: LabQuantSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Process AB channels
    let (avg_a, avg_b) = process_block_iteration(
        current_l,
        current_a,
        current_b,
        ref_l,
        ref_a,
        ref_b,
        block_width,
        block_height,
        ref_block_width,
        ref_block_height,
        rotation_angles,
        histogram_mode,
        block_seed,
        histogram_dither_mode,
        color_aware,
        distance_space,
        colorspace,
    );

    // Process L channel
    let l_scaled = scale_l_to_255(current_l, colorspace);
    let ref_l_scaled = scale_l_to_255(ref_l, colorspace);

    let avg_l = if histogram_mode > 0 {
        // Use a distinct seed for L channel (offset by 100)
        let align_mode = if histogram_mode == 2 {
            AlignmentMode::Midpoint
        } else {
            AlignmentMode::Endpoint
        };
        let l_seed = block_seed.wrapping_mul(10) + 100;
        let matched_l = match_histogram_f32(&l_scaled, &ref_l_scaled, InterpolationMode::Linear, align_mode, l_seed);
        scale_255_to_l(&matched_l, colorspace)
    } else if color_aware {
        // Use color-aware Lab dithering for L channel (no rotation for L-only)
        let ab_ranges = get_ab_ranges(0.0, colorspace);
        let params = lab_quant_params_from_ranges(ab_ranges, true, 0.0, colorspace);
        let l_seed = block_seed.wrapping_mul(1000) + 500;

        let (l_uint8, _, _) = lab_space_dither_with_mode(
            current_l, current_a, current_b,
            block_width, block_height,
            &params,
            colorspace,
            distance_space,
            histogram_dither_mode.into(),
            l_seed,
        );

        let (ref_l_uint8, _, _) = lab_space_dither_with_mode(
            ref_l, ref_a, ref_b,
            ref_block_width, ref_block_height,
            &params,
            colorspace,
            distance_space,
            histogram_dither_mode.into(),
            l_seed + 1,
        );

        let matched_l = match_histogram(&l_uint8, &ref_l_uint8);
        scale_uint8_to_l(&matched_l, colorspace)
    } else {
        // L channel uses distinct seeds based on block_seed
        let l_seed = block_seed.wrapping_mul(1000) + 500;
        let l_uint8 = dither_with_mode(&l_scaled, block_width, block_height, histogram_dither_mode, l_seed);
        let ref_l_uint8 = dither_with_mode(&ref_l_scaled, ref_block_width, ref_block_height, histogram_dither_mode, l_seed + 1);
        let matched_l = match_histogram(&l_uint8, &ref_l_uint8);
        scale_uint8_to_l(&matched_l, colorspace)
    };

    (avg_l, avg_a, avg_b)
}

/// Tiled CRA color correction with configurable colorspace - returns linear RGB
///
/// Args:
///     in_r, in_g, in_b: Input image as linear RGB channels (0-1 range)
///     ref_r, ref_g, ref_b: Reference image as linear RGB channels (0-1 range)
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     colorspace: Color space to use (CIELAB or OkLab)
///     tiled_luminosity: If true, process L channel per-tile before global match
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram preparation
///     color_aware_histogram: If true and histogram_mode == 0, use color-aware Lab dithering
///     histogram_distance_space: Perceptual space for color-aware histogram dithering
///
/// Returns:
///     (R, G, B) linear RGB channels (f32, 0-1 range)
#[allow(clippy::too_many_arguments)]
pub fn color_correct_tiled_linear(
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
    colorspace: LabQuantSpace,
    tiled_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    color_aware_histogram: bool,
    histogram_distance_space: PerceptualSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input_pixels = input_width * input_height;

    // Convert to separate Lab channels
    let (input_l, input_a, input_b) = linear_rgb_to_lab(in_r, in_g, in_b, colorspace);
    let (ref_l, ref_a, ref_b) = linear_rgb_to_lab(ref_r, ref_g, ref_b, colorspace);

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
    for (block_idx, block) in blocks.iter().enumerate() {
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

        // Extract L blocks - always needed for color-aware distance or tiled_luminosity
        let needs_l_blocks = tiled_luminosity || color_aware_histogram;
        let (mut current_l, ref_l_block) = if needs_l_blocks {
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
                    histogram_mode,
                    block_idx as u32,
                    histogram_dither_mode,
                    color_aware_histogram,
                    histogram_distance_space,
                    colorspace,
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
                    histogram_mode,
                    block_idx as u32,
                    histogram_dither_mode,
                    color_aware_histogram,
                    histogram_distance_space,
                    colorspace,
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
    let final_ab_ranges = get_ab_ranges(0.0, colorspace);

    // Match histograms for all channels (final pass uses high seeds to avoid collision with block passes)
    let (final_l, final_a, final_b) = if histogram_mode > 0 {
        // Use f32 histogram matching directly (no dithering/quantization)
        let l_scaled = scale_l_to_255(&final_l_input, colorspace);
        let (a_scaled, b_scaled) = scale_ab_to_255(&a_acc, &b_acc, final_ab_ranges);
        let ref_l_scaled = scale_l_to_255(&ref_l, colorspace);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_255(&ref_a, &ref_b, final_ab_ranges);

        let align_mode = if histogram_mode == 2 {
            AlignmentMode::Midpoint
        } else {
            AlignmentMode::Endpoint
        };
        let matched_l = match_histogram_f32(&l_scaled, &ref_l_scaled, InterpolationMode::Linear, align_mode, 1000);
        let matched_a = match_histogram_f32(&a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, 1001);
        let matched_b = match_histogram_f32(&b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 1002);
        let l_lab = scale_255_to_l(&matched_l, colorspace);
        let (a_lab, b_lab) = scale_255_to_ab(&matched_a, &matched_b, final_ab_ranges);
        (l_lab, a_lab, b_lab)
    } else if color_aware_histogram {
        // Use color-aware Lab dithering for final histogram matching
        // Final pass has no rotation (0 degrees)
        let params = lab_quant_params_from_ranges(final_ab_ranges, true, 0.0, colorspace);

        // Dither input
        let (current_l_u8, current_a_u8, current_b_u8) = lab_space_dither_with_mode(
            &final_l_input, &a_acc, &b_acc,
            input_width, input_height,
            &params,
            colorspace,
            histogram_distance_space,
            histogram_dither_mode.into(),
            10000,
        );

        // Dither reference
        let (ref_l_u8, ref_a_u8, ref_b_u8) = lab_space_dither_with_mode(
            &ref_l, &ref_a, &ref_b,
            ref_width, ref_height,
            &params,
            colorspace,
            histogram_distance_space,
            histogram_dither_mode.into(),
            10001,
        );

        let matched_l = match_histogram(&current_l_u8, &ref_l_u8);
        let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&current_b_u8, &ref_b_u8);
        let l_lab = scale_uint8_to_l(&matched_l, colorspace);
        let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);
        (l_lab, a_lab, b_lab)
    } else {
        // Use channel-independent binned histogram matching with dithering
        // Final global pass uses very high seed values to avoid collision with block passes
        let l_scaled = scale_l_to_255(&final_l_input, colorspace);
        let (a_scaled, b_scaled) = scale_ab_to_255(&a_acc, &b_acc, final_ab_ranges);
        let ref_l_scaled = scale_l_to_255(&ref_l, colorspace);
        let (ref_a_scaled, ref_b_scaled) = scale_ab_to_255(&ref_a, &ref_b, final_ab_ranges);

        let current_l_u8 = dither_with_mode(&l_scaled, input_width, input_height, histogram_dither_mode, 10000);
        let current_a_u8 = dither_with_mode(&a_scaled, input_width, input_height, histogram_dither_mode, 10001);
        let current_b_u8 = dither_with_mode(&b_scaled, input_width, input_height, histogram_dither_mode, 10002);
        let ref_l_u8 = dither_with_mode(&ref_l_scaled, ref_width, ref_height, histogram_dither_mode, 10003);
        let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, histogram_dither_mode, 10004);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, 10005);

        let matched_l = match_histogram(&current_l_u8, &ref_l_u8);
        let matched_a = match_histogram(&current_a_u8, &ref_a_u8);
        let matched_b = match_histogram(&current_b_u8, &ref_b_u8);
        let l_lab = scale_uint8_to_l(&matched_l, colorspace);
        let (a_lab, b_lab) = scale_uint8_to_ab(&matched_a, &matched_b, final_ab_ranges);
        (l_lab, a_lab, b_lab)
    };

    // Convert LAB back to linear RGB (separate channels)
    lab_to_linear_rgb(&final_l, &final_a, &final_b, colorspace)
}

/// Convenience wrapper: Tiled CRA LAB color correction (CIELAB colorspace) - returns linear RGB
#[allow(clippy::too_many_arguments)]
pub fn color_correct_tiled_lab_linear(
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
    tiled_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    color_aware_histogram: bool,
    histogram_distance_space: PerceptualSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    color_correct_tiled_linear(
        in_r, in_g, in_b,
        ref_r, ref_g, ref_b,
        input_width,
        input_height,
        ref_width,
        ref_height,
        LabQuantSpace::CIELab,
        tiled_luminosity,
        histogram_mode,
        histogram_dither_mode,
        color_aware_histogram,
        histogram_distance_space,
    )
}

/// Convenience wrapper: Tiled CRA OkLab color correction (OkLab colorspace) - returns linear RGB
#[allow(clippy::too_many_arguments)]
pub fn color_correct_tiled_oklab_linear(
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
    tiled_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    color_aware_histogram: bool,
    histogram_distance_space: PerceptualSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    color_correct_tiled_linear(
        in_r, in_g, in_b,
        ref_r, ref_g, ref_b,
        input_width,
        input_height,
        ref_width,
        ref_height,
        LabQuantSpace::OkLab,
        tiled_luminosity,
        histogram_mode,
        histogram_dither_mode,
        color_aware_histogram,
        histogram_distance_space,
    )
}
