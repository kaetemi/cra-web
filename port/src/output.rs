/// Output finalization module.
///
/// Converts linear RGB to final sRGB uint8 output with configurable dithering.
/// This module provides the final step in the color correction pipeline,
/// separating histogram matching (which outputs linear RGB) from output quantization.
///
/// The primary API is `finalize_output` which takes linear RGB and an `OutputTechnique`
/// enum to select the dithering method.

use crate::color::{denormalize_inplace, interleave_rgb_u8, linear_to_srgb_inplace};
use crate::dither::{dither_with_mode, dither_with_mode_bits, DitherMode};
use crate::dither_colorspace_aware::colorspace_aware_dither_rgb_channels;
use crate::dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode;
use crate::dither_common::{OutputTechnique, PerceptualSpace};
use crate::pixel::{pixels_to_channels, pixels_to_srgb_u8, Pixel4};

/// Finalize linear RGB to sRGB uint8 without dithering.
///
/// This is the fastest path for when no dithering is needed.
///
/// Args:
///     pixels: Linear RGB Pixel4 array (0-1 range) - modified in place
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_to_srgb_u8(pixels: &mut [Pixel4]) -> Vec<u8> {
    // Convert linear to sRGB 0-255 in place
    linear_to_srgb_inplace(pixels);
    denormalize_inplace(pixels);

    // Convert to u8 output
    pixels_to_srgb_u8(pixels)
}

/// Finalize linear RGB to sRGB uint8 with channel-independent dithering.
///
/// Args:
///     pixels: Linear RGB Pixel4 array (0-1 range) - modified in place
///     width, height: Image dimensions
///     dither_mode: Error diffusion algorithm to use
///     seed: Random seed for mixed dithering modes
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_to_srgb_u8_dithered(
    pixels: &mut [Pixel4],
    width: usize,
    height: usize,
    dither_mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    // Convert linear to sRGB 0-255 in place
    linear_to_srgb_inplace(pixels);
    denormalize_inplace(pixels);

    // Extract channels for dithering
    let (r_scaled, g_scaled, b_scaled) = pixels_to_channels(pixels);

    // Dither each channel independently
    let r_u8 = dither_with_mode(&r_scaled, width, height, dither_mode, seed);
    let g_u8 = dither_with_mode(&g_scaled, width, height, dither_mode, seed.wrapping_add(1));
    let b_u8 = dither_with_mode(&b_scaled, width, height, dither_mode, seed.wrapping_add(2));

    // Interleave channels
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}

/// Finalize linear RGB to sRGB uint8 with color-aware dithering.
///
/// Uses joint RGB processing with perceptual distance metrics.
///
/// Args:
///     pixels: Linear RGB Pixel4 array (0-1 range) - modified in place
///     width, height: Image dimensions
///     distance_space: Perceptual space for distance calculations
///     dither_mode: Error diffusion algorithm to use
///     seed: Random seed for mixed dithering modes
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_to_srgb_u8_color_aware(
    pixels: &mut [Pixel4],
    width: usize,
    height: usize,
    distance_space: PerceptualSpace,
    dither_mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    // Convert linear to sRGB 0-255 in place
    linear_to_srgb_inplace(pixels);
    denormalize_inplace(pixels);

    // Color-aware dithering (joint RGB processing)
    let (r_u8, g_u8, b_u8) = colorspace_aware_dither_rgb_channels(
        pixels,
        width,
        height,
        8, 8, 8, // Full 8-bit output
        distance_space,
        dither_mode.into(),
        seed,
        None,
    );

    // Interleave channels
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}

/// Finalize linear RGB to sRGB uint8 with configurable options.
///
/// Args:
///     pixels: Linear RGB Pixel4 array (0-1 range) - modified in place
///     width, height: Image dimensions
///     dither_mode: Error diffusion algorithm to use (use None option for no dithering)
///     color_aware: If true, use color-aware dithering
///     distance_space: Perceptual space for color-aware dithering
///     seed: Random seed for mixed dithering modes
///
/// Returns:
///     Interleaved sRGB uint8 data (RGBRGB...)
pub fn finalize_to_srgb_u8_with_options(
    pixels: &mut [Pixel4],
    width: usize,
    height: usize,
    dither_mode: Option<DitherMode>,
    color_aware: bool,
    distance_space: PerceptualSpace,
    seed: u32,
) -> Vec<u8> {
    match dither_mode {
        None => finalize_to_srgb_u8(pixels),
        Some(mode) if color_aware => finalize_to_srgb_u8_color_aware(
            pixels,
            width,
            height,
            distance_space,
            mode,
            seed,
        ),
        Some(mode) => finalize_to_srgb_u8_dithered(pixels, width, height, mode, seed),
    }
}

// ============================================================================
// Unified Output API
// ============================================================================

/// Dither sRGB 0-255 Pixel4 data to the specified bit depth.
///
/// This is the core dithering function that takes already-converted sRGB data.
/// Use this when you have sRGB 0-255 values (e.g., from web canvas after
/// linear_to_srgb conversion).
///
/// Args:
///     srgb_pixels: sRGB Pixel4 array (0-255 range)
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per channel (1-8)
///     technique: Dithering technique selection
///     seed: Random seed for mixed dithering modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Tuple of (R, G, B) channel vectors as u8
pub fn dither_output(
    srgb_pixels: &[Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: OutputTechnique,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    match technique {
        OutputTechnique::None => {
            // Simple quantization without dithering - no progress needed
            let (r, g, b) = pixels_to_channels(srgb_pixels);
            let r_u8: Vec<u8> = r.iter().map(|v| quantize_no_dither(*v, bits_r)).collect();
            let g_u8: Vec<u8> = g.iter().map(|v| quantize_no_dither(*v, bits_g)).collect();
            let b_u8: Vec<u8> = b.iter().map(|v| quantize_no_dither(*v, bits_b)).collect();
            (r_u8, g_u8, b_u8)
        }
        OutputTechnique::PerChannel { mode } => {
            // Per-channel error diffusion in sRGB space
            // Note: progress callback is ignored for per-channel (not practical to report)
            let (r, g, b) = pixels_to_channels(srgb_pixels);
            let r_u8 = dither_with_mode_bits(&r, width, height, mode, seed, bits_r, None);
            let g_u8 = dither_with_mode_bits(&g, width, height, mode, seed.wrapping_add(1), bits_g, None);
            let b_u8 = dither_with_mode_bits(&b, width, height, mode, seed.wrapping_add(2), bits_b, None);
            (r_u8, g_u8, b_u8)
        }
        OutputTechnique::ColorspaceAware { mode, space } => {
            // Joint RGB color-aware dithering with progress
            colorspace_aware_dither_rgb_channels(
                srgb_pixels,
                width,
                height,
                bits_r,
                bits_g,
                bits_b,
                space,
                mode,
                seed,
                progress,
            )
        }
    }
}

/// Dither sRGB 0-255 Pixel4 data to interleaved u8 output.
///
/// Convenience wrapper around `dither_output` that returns interleaved RGB.
///
/// Args:
///     srgb_pixels: sRGB Pixel4 array (0-255 range)
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per channel (1-8)
///     technique: Dithering technique selection
///     seed: Random seed for mixed dithering modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Interleaved RGB u8 data (RGBRGB...)
pub fn dither_output_interleaved(
    srgb_pixels: &[Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: OutputTechnique,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let (r, g, b) = dither_output(srgb_pixels, width, height, bits_r, bits_g, bits_b, technique, seed, progress);
    interleave_rgb_u8(&r, &g, &b)
}

/// Finalize linear RGB to dithered sRGB u8 output (unified API).
///
/// This is the primary output function that handles the complete pipeline:
/// 1. Convert linear RGB (0-1) to sRGB (0-255)
/// 2. Apply the selected dithering technique
/// 3. Quantize to the target bit depth
///
/// Args:
///     pixels: Linear RGB Pixel4 array (0-1 range) - modified in place
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per channel (1-8, use 8 for RGB888)
///     technique: Dithering technique selection
///     seed: Random seed for mixed dithering modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Tuple of (R, G, B) channel vectors as u8
pub fn finalize_output(
    pixels: &mut [Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: OutputTechnique,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Convert linear RGB to sRGB 0-255 in place
    linear_to_srgb_inplace(pixels);
    denormalize_inplace(pixels);

    // Apply dithering
    dither_output(pixels, width, height, bits_r, bits_g, bits_b, technique, seed, progress)
}

/// Finalize linear RGB to interleaved sRGB u8 output (unified API).
///
/// Convenience wrapper around `finalize_output` that returns interleaved RGB.
///
/// Args:
///     pixels: Linear RGB Pixel4 array (0-1 range) - modified in place
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per channel (1-8, use 8 for RGB888)
///     technique: Dithering technique selection
///     seed: Random seed for mixed dithering modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Interleaved sRGB u8 data (RGBRGB...)
pub fn finalize_output_interleaved(
    pixels: &mut [Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: OutputTechnique,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let (r, g, b) = finalize_output(pixels, width, height, bits_r, bits_g, bits_b, technique, seed, progress);
    interleave_rgb_u8(&r, &g, &b)
}

/// Simple quantization without dithering
#[inline]
fn quantize_no_dither(value: f32, bits: u8) -> u8 {
    let max_level = (1u32 << bits) - 1;
    let scaled = (value / 255.0 * max_level as f32).round() as u8;
    // Bit-replicate to 8 bits
    crate::dither_common::bit_replicate(scaled, bits)
}

// ============================================================================
// Unified Grayscale Output API
// ============================================================================

/// Dither sRGB 0-255 grayscale data to the specified bit depth.
///
/// This is the core grayscale dithering function that takes already-converted sRGB data.
/// Use this when you have sRGB 0-255 grayscale values.
///
/// Args:
///     gray_channel: Grayscale input as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits: Output bit depth (1-8)
///     technique: Dithering technique selection
///     seed: Random seed for mixed dithering modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Grayscale output as u8
pub fn dither_output_gray(
    gray_channel: &[f32],
    width: usize,
    height: usize,
    bits: u8,
    technique: OutputTechnique,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    match technique {
        OutputTechnique::None => {
            // Simple quantization without dithering - no progress needed
            gray_channel.iter().map(|v| quantize_no_dither(*v, bits)).collect()
        }
        OutputTechnique::PerChannel { mode } => {
            // Per-channel error diffusion in sRGB space (no linear/perceptual conversion)
            // Note: progress callback is ignored for per-channel
            crate::dither::dither_with_mode_bits(gray_channel, width, height, mode, seed, bits, None)
        }
        OutputTechnique::ColorspaceAware { mode, space } => {
            // Color-aware dithering for grayscale
            colorspace_aware_dither_gray_with_mode(
                gray_channel,
                width,
                height,
                bits,
                space,
                mode,
                seed,
                progress,
            )
        }
    }
}
