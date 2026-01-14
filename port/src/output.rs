/// Output finalization module.
///
/// Converts linear RGB to final sRGB uint8 output with configurable dithering.
/// This module provides the final step in the color correction pipeline,
/// separating histogram matching (which outputs linear RGB) from output quantization.

use crate::color::{interleave_rgb_u8, linear_to_srgb_scaled_channels};
use crate::dither::{dither_with_mode, DitherMode};
use crate::dither_colorspace_aware::colorspace_aware_dither_rgb_with_mode;
use crate::dither_common::PerceptualSpace;

/// Finalize linear RGB to sRGB uint8 with channel-independent dithering.
///
/// This is the standard output path that processes each RGB channel independently.
///
/// Args:
///     r, g, b: Linear RGB channels (0-1 range)
///     width, height: Image dimensions
///     dither_mode: Error diffusion algorithm to use
///     seed: Random seed for mixed dithering modes
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_linear_to_srgb_u8(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    dither_mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) = linear_to_srgb_scaled_channels(r, g, b);

    // Dither each channel independently
    let r_u8 = dither_with_mode(&r_scaled, width, height, dither_mode, seed);
    let g_u8 = dither_with_mode(&g_scaled, width, height, dither_mode, seed.wrapping_add(1));
    let b_u8 = dither_with_mode(&b_scaled, width, height, dither_mode, seed.wrapping_add(2));

    // Interleave channels
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}

/// Finalize linear RGB to sRGB uint8 with color-aware dithering.
///
/// Uses joint RGB processing with perceptual distance metrics for improved
/// color accuracy at the cost of some computational overhead.
///
/// Args:
///     r, g, b: Linear RGB channels (0-1 range)
///     width, height: Image dimensions
///     distance_space: Perceptual space for distance calculations
///     dither_mode: Error diffusion algorithm to use
///     seed: Random seed for mixed dithering modes
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_linear_to_srgb_u8_color_aware(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    distance_space: PerceptualSpace,
    dither_mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    // Convert to sRGB and scale to 0-255
    let (r_scaled, g_scaled, b_scaled) = linear_to_srgb_scaled_channels(r, g, b);

    // Color-aware dithering (joint RGB processing)
    let (r_u8, g_u8, b_u8) = colorspace_aware_dither_rgb_with_mode(
        &r_scaled,
        &g_scaled,
        &b_scaled,
        width,
        height,
        8, 8, 8, // Full 8-bit output
        distance_space,
        dither_mode.into(),
        seed,
    );

    // Interleave channels
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}

/// Finalize linear RGB to sRGB uint8 with configurable dithering.
///
/// Convenience function that selects between channel-independent and color-aware
/// dithering based on the `color_aware` parameter.
///
/// Args:
///     r, g, b: Linear RGB channels (0-1 range)
///     width, height: Image dimensions
///     dither_mode: Error diffusion algorithm to use
///     color_aware: If true, use color-aware dithering
///     distance_space: Perceptual space for color-aware dithering (ignored if color_aware is false)
///     seed: Random seed for mixed dithering modes
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_linear_to_srgb_u8_with_options(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    dither_mode: DitherMode,
    color_aware: bool,
    distance_space: PerceptualSpace,
    seed: u32,
) -> Vec<u8> {
    if color_aware {
        finalize_linear_to_srgb_u8_color_aware(
            r, g, b, width, height, distance_space, dither_mode, seed,
        )
    } else {
        finalize_linear_to_srgb_u8(r, g, b, width, height, dither_mode, seed)
    }
}
