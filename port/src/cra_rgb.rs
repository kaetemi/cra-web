/// CRA (Chroma Rotation Averaging) in RGB color space.
/// Corresponds to color_correction_cra_rgb.py

use crate::color::{interleave_rgb_u8, linear_to_srgb_scaled_channels, srgb_to_linear};
use crate::dither::{dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, InterpolationMode};
use crate::rotation::{
    compute_rgb_channel_ranges, deg_to_rad, perceptual_scale_factors, perceptual_scale_rgb,
    perceptual_unscale_rgb, rotate_rgb,
};

/// Default rotation angles for CRA RGB (evenly spaced across 120° period)
const ROTATION_ANGLES: [f32; 3] = [0.0, 40.0, 80.0];

/// Default blend factors for iterative refinement
const BLEND_FACTORS: [f32; 3] = [0.25, 0.5, 1.0];

/// Scale RGB to uint8 range based on channel ranges
fn scale_rgb_to_uint8(rgb: &[f32], channel_ranges: [[f32; 2]; 3]) -> Vec<f32> {
    let pixels = rgb.len() / 3;
    let mut result = vec![0.0f32; rgb.len()];

    for i in 0..pixels {
        for c in 0..3 {
            let [min_val, max_val] = channel_ranges[c];
            result[i * 3 + c] = (rgb[i * 3 + c] - min_val) / (max_val - min_val) * 255.0;
        }
    }

    result
}

/// Reverse uint8 to RGB scaling
fn scale_uint8_to_rgb(rgb_uint8: &[u8], channel_ranges: [[f32; 2]; 3]) -> Vec<f32> {
    let pixels = rgb_uint8.len() / 3;
    let mut result = vec![0.0f32; rgb_uint8.len()];

    for i in 0..pixels {
        for c in 0..3 {
            let [min_val, max_val] = channel_ranges[c];
            result[i * 3 + c] =
                rgb_uint8[i * 3 + c] as f32 / 255.0 * (max_val - min_val) + min_val;
        }
    }

    result
}

/// Reverse f32 (0-255 range) to RGB scaling
fn scale_255_to_rgb(rgb_255: &[f32], channel_ranges: [[f32; 2]; 3]) -> Vec<f32> {
    let pixels = rgb_255.len() / 3;
    let mut result = vec![0.0f32; rgb_255.len()];

    for i in 0..pixels {
        for c in 0..3 {
            let [min_val, max_val] = channel_ranges[c];
            result[i * 3 + c] = rgb_255[i * 3 + c] / 255.0 * (max_val - min_val) + min_val;
        }
    }

    result
}

/// Process one iteration of RGB-space histogram matching
fn process_rgb_iteration(
    current_rgb: &[f32],
    ref_rgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    rotation_angles: &[f32],
    perceptual_scale: Option<[f32; 3]>,
    use_f32_histogram: bool,
    dither_mode: DitherMode,
    dither_seed_base: u32,
) -> Vec<f32> {
    let input_pixels = input_width * input_height;
    let ref_pixels = ref_width * ref_height;

    let mut all_corrected: Vec<Vec<f32>> = Vec::new();

    for (pass_idx, &theta_deg) in rotation_angles.iter().enumerate() {
        let theta_rad = deg_to_rad(theta_deg);
        let channel_ranges = compute_rgb_channel_ranges(theta_deg, perceptual_scale);

        // Rotate RGB
        let current_rot = rotate_rgb(current_rgb, theta_rad);
        let ref_rot = rotate_rgb(ref_rgb, theta_rad);

        // Scale to 0-255 range
        let current_scaled = scale_rgb_to_uint8(&current_rot, channel_ranges);
        let ref_scaled = scale_rgb_to_uint8(&ref_rot, channel_ranges);

        let matched_scaled = if use_f32_histogram {
            // Use f32 histogram matching directly (no dithering/quantization)
            // Extract each channel
            let current_chs: Vec<Vec<f32>> = (0..3)
                .map(|c| (0..input_pixels).map(|i| current_scaled[i * 3 + c]).collect())
                .collect();
            let ref_chs: Vec<Vec<f32>> = (0..3)
                .map(|c| (0..ref_pixels).map(|i| ref_scaled[i * 3 + c]).collect())
                .collect();

            // Match histograms with different seeds per pass and channel for noise averaging
            let matched: Vec<Vec<f32>> = (0..3)
                .map(|c| {
                    let seed = (pass_idx * 3 + c) as u32;
                    match_histogram_f32(&current_chs[c], &ref_chs[c], InterpolationMode::Linear, seed)
                })
                .collect();

            // Reconstruct interleaved
            let mut matched_255 = vec![0.0f32; input_pixels * 3];
            for i in 0..input_pixels {
                for c in 0..3 {
                    matched_255[i * 3 + c] = matched[c][i];
                }
            }
            scale_255_to_rgb(&matched_255, channel_ranges)
        } else {
            // Use binned histogram matching with dithering
            // Each pass gets unique seeds: base + pass_idx * 6 + channel offset
            let pass_seed = dither_seed_base + (pass_idx as u32) * 6;
            let current_u8: Vec<Vec<u8>> = (0..3)
                .map(|c| {
                    let ch: Vec<f32> = (0..input_pixels)
                        .map(|i| current_scaled[i * 3 + c])
                        .collect();
                    dither_with_mode(&ch, input_width, input_height, dither_mode, pass_seed + c as u32)
                })
                .collect();
            let ref_u8: Vec<Vec<u8>> = (0..3)
                .map(|c| {
                    let ch: Vec<f32> = (0..ref_pixels)
                        .map(|i| ref_scaled[i * 3 + c])
                        .collect();
                    dither_with_mode(&ch, ref_width, ref_height, dither_mode, pass_seed + 3 + c as u32)
                })
                .collect();

            // Match histograms
            let matched: Vec<Vec<u8>> = (0..3)
                .map(|c| match_histogram(&current_u8[c], &ref_u8[c]))
                .collect();

            // Reconstruct interleaved
            let mut matched_uint8 = vec![0u8; input_pixels * 3];
            for i in 0..input_pixels {
                for c in 0..3 {
                    matched_uint8[i * 3 + c] = matched[c][i];
                }
            }
            scale_uint8_to_rgb(&matched_uint8, channel_ranges)
        };

        // Rotate back
        let matched_rgb = rotate_rgb(&matched_scaled, -theta_rad);

        all_corrected.push(matched_rgb);
    }

    // Average all corrections
    let num_angles = all_corrected.len() as f32;
    let mut avg_rgb = vec![0.0f32; input_pixels * 3];
    for i in 0..(input_pixels * 3) {
        avg_rgb[i] = all_corrected.iter().map(|v| v[i]).sum::<f32>() / num_angles;
    }

    avg_rgb
}

/// CRA RGB color correction
///
/// Args:
///     input_srgb: Input image as sRGB values (0-1), flat array HxWx3
///     ref_srgb: Reference image as sRGB values (0-1), flat array HxWx3
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     use_perceptual: If true, use perceptual weighting
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
///
/// Returns:
///     Output image as sRGB uint8, flat array HxWx3
pub fn color_correct_cra_rgb(
    input_srgb: &[f32],
    ref_srgb: &[f32],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    use_perceptual: bool,
    use_f32_histogram: bool,
    dither_mode: DitherMode,
) -> Vec<u8> {
    let input_pixels = input_width * input_height;
    let ref_pixels = ref_width * ref_height;

    // Determine scale factors
    let perceptual_scale = if use_perceptual {
        Some(perceptual_scale_factors())
    } else {
        None
    };

    // Convert to linear RGB
    let mut input_linear = input_srgb.to_vec();
    let mut ref_linear = ref_srgb.to_vec();
    srgb_to_linear(&mut input_linear);
    srgb_to_linear(&mut ref_linear);

    // Apply perceptual scaling if enabled
    let (mut current, ref_scaled) = if let Some(scale) = perceptual_scale {
        (
            perceptual_scale_rgb(&input_linear, scale),
            perceptual_scale_rgb(&ref_linear, scale),
        )
    } else {
        (input_linear.clone(), ref_linear.clone())
    };

    // Iterative refinement
    // Each iteration gets a unique seed range: iteration_idx * 100 + pass seeds
    for (iter_idx, &blend_factor) in BLEND_FACTORS.iter().enumerate() {
        let rgb_result = process_rgb_iteration(
            &current,
            &ref_scaled,
            input_width,
            input_height,
            ref_width,
            ref_height,
            &ROTATION_ANGLES,
            perceptual_scale,
            use_f32_histogram,
            dither_mode,
            (iter_idx as u32) * 100,
        );

        // Blend with current
        for i in 0..(input_pixels * 3) {
            current[i] = current[i] * (1.0 - blend_factor) + rgb_result[i] * blend_factor;
        }
    }

    // Final histogram match
    let final_ranges = compute_rgb_channel_ranges(0.0, perceptual_scale);

    let current_scaled = scale_rgb_to_uint8(&current, final_ranges);
    let ref_scaled_final = scale_rgb_to_uint8(&ref_scaled, final_ranges);

    let mut final_scaled = if use_f32_histogram {
        // Use f32 histogram matching directly (no dithering/quantization)
        let current_chs: Vec<Vec<f32>> = (0..3)
            .map(|c| (0..input_pixels).map(|i| current_scaled[i * 3 + c]).collect())
            .collect();
        let ref_chs: Vec<Vec<f32>> = (0..3)
            .map(|c| (0..ref_pixels).map(|i| ref_scaled_final[i * 3 + c]).collect())
            .collect();

        // Final pass uses high seed values to avoid collision with rotation passes
        let matched: Vec<Vec<f32>> = (0..3)
            .map(|c| {
                let seed = (100 + c) as u32;
                match_histogram_f32(&current_chs[c], &ref_chs[c], InterpolationMode::Linear, seed)
            })
            .collect();

        let mut matched_255 = vec![0.0f32; input_pixels * 3];
        for i in 0..input_pixels {
            for c in 0..3 {
                matched_255[i * 3 + c] = matched[c][i];
            }
        }
        scale_255_to_rgb(&matched_255, final_ranges)
    } else {
        // Use binned histogram matching with dithering
        // Final pass uses high seed values (1000+) to avoid collision with iteration passes
        let current_u8: Vec<Vec<u8>> = (0..3)
            .map(|c| {
                let ch: Vec<f32> = (0..input_pixels)
                    .map(|i| current_scaled[i * 3 + c])
                    .collect();
                dither_with_mode(&ch, input_width, input_height, dither_mode, 1000 + c as u32)
            })
            .collect();
        let ref_u8: Vec<Vec<u8>> = (0..3)
            .map(|c| {
                let ch: Vec<f32> = (0..ref_pixels)
                    .map(|i| ref_scaled_final[i * 3 + c])
                    .collect();
                dither_with_mode(&ch, ref_width, ref_height, dither_mode, 1003 + c as u32)
            })
            .collect();

        let matched: Vec<Vec<u8>> = (0..3)
            .map(|c| match_histogram(&current_u8[c], &ref_u8[c]))
            .collect();

        let mut matched_uint8 = vec![0u8; input_pixels * 3];
        for i in 0..input_pixels {
            for c in 0..3 {
                matched_uint8[i * 3 + c] = matched[c][i];
            }
        }
        scale_uint8_to_rgb(&matched_uint8, final_ranges)
    };

    // Remove perceptual scaling if it was applied
    if let Some(scale) = perceptual_scale {
        final_scaled = perceptual_unscale_rgb(&final_scaled, scale);
    }

    // Extract separate linear RGB channels
    let final_r: Vec<f32> = (0..input_pixels).map(|i| final_scaled[i * 3]).collect();
    let final_g: Vec<f32> = (0..input_pixels).map(|i| final_scaled[i * 3 + 1]).collect();
    let final_b: Vec<f32> = (0..input_pixels).map(|i| final_scaled[i * 3 + 2]).collect();

    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) =
        linear_to_srgb_scaled_channels(&final_r, &final_g, &final_b);

    // Dither each channel for final output
    let r_u8 = dither_with_mode(&r_scaled, input_width, input_height, dither_mode, 1006);
    let g_u8 = dither_with_mode(&g_scaled, input_width, input_height, dither_mode, 1007);
    let b_u8 = dither_with_mode(&b_scaled, input_width, input_height, dither_mode, 1008);

    // Interleave only at the very end
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}
