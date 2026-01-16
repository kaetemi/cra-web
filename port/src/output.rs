/// Output dithering module.
///
/// Provides dithering functions that take sRGB 0-255 input and produce
/// quantized output at the specified bit depth.
///
/// The primary API is `dither_output_rgb` for RGB and `dither_output_luminosity`
/// for grayscale.

use crate::color::{interleave_rgb_u8, interleave_rgba_u8};
use crate::dither::dither_with_mode_bits;
use crate::dither_rgb::colorspace_aware_dither_rgb_channels;
use crate::dither_rgba::colorspace_aware_dither_rgba_channels;
use crate::dither_luminosity::colorspace_aware_dither_gray_with_mode;
use crate::dither_common::OutputTechnique;
use crate::pixel::{pixels_to_channels, pixels_to_channels_rgba, Pixel4};

// ============================================================================
// RGB Output API
// ============================================================================

/// Dither sRGB 0-255 Pixel4 data to interleaved u8 output.
///
/// Takes already-converted sRGB data and applies the specified dithering technique.
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
///     Interleaved RGB u8 data (RGBRGB...)
pub fn dither_output_rgb(
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
    match technique {
        OutputTechnique::None => {
            // Simple quantization without dithering - no progress needed
            let (r, g, b) = pixels_to_channels(srgb_pixels);
            let r_u8: Vec<u8> = r.iter().map(|v| quantize_no_dither(*v, bits_r)).collect();
            let g_u8: Vec<u8> = g.iter().map(|v| quantize_no_dither(*v, bits_g)).collect();
            let b_u8: Vec<u8> = b.iter().map(|v| quantize_no_dither(*v, bits_b)).collect();
            interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
        }
        OutputTechnique::PerChannel { mode } => {
            // Per-channel error diffusion in sRGB space
            // Note: progress callback is ignored for per-channel (not practical to report)
            let (r, g, b) = pixels_to_channels(srgb_pixels);
            let r_u8 = dither_with_mode_bits(&r, width, height, mode, seed, bits_r, None);
            let g_u8 = dither_with_mode_bits(&g, width, height, mode, seed.wrapping_add(1), bits_g, None);
            let b_u8 = dither_with_mode_bits(&b, width, height, mode, seed.wrapping_add(2), bits_b, None);
            interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
        }
        OutputTechnique::ColorspaceAware { mode, space } => {
            // Joint RGB color-aware dithering with progress
            let (r, g, b) = colorspace_aware_dither_rgb_channels(
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
            );
            interleave_rgb_u8(&r, &g, &b)
        }
    }
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
// RGBA Output API
// ============================================================================

/// Dither sRGB 0-255 Pixel4 data to interleaved u8 RGBA output.
///
/// When bits_a > 0: All channels (RGB + alpha) are dithered with alpha-aware error propagation:
/// - Alpha is dithered first using standard single-channel dithering
/// - RGB channels are dithered with error weighted by alpha visibility
/// - Fully transparent pixels pass error to neighbors without absorbing it
/// - Fully opaque pixels behave like standard RGB dithering
///
/// When bits_a == 0: Alpha is stripped and RGB-only dithering is used.
/// Output alpha channel is set to 255 (fully opaque).
///
/// Args:
///     srgb_pixels: sRGB Pixel4 array (all channels in 0-255 range)
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per RGB channel (1-8)
///     bits_a: Output bit depth for alpha (1-8), or 0 to strip alpha
///     technique: Dithering technique selection
///     seed: Random seed for mixed dithering modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Interleaved RGBA u8 data (RGBARGBA...)
pub fn dither_output_rgba(
    srgb_pixels: &[Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    bits_a: u8,
    technique: OutputTechnique,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    // When bits_a == 0, strip alpha and use RGB-only dithering
    if bits_a == 0 {
        let rgb_data = dither_output_rgb(
            srgb_pixels,
            width,
            height,
            bits_r,
            bits_g,
            bits_b,
            technique,
            seed,
            progress,
        );
        // Convert RGB to RGBA with alpha=255
        let pixels = rgb_data.len() / 3;
        let mut rgba_data = Vec::with_capacity(pixels * 4);
        for i in 0..pixels {
            rgba_data.push(rgb_data[i * 3]);
            rgba_data.push(rgb_data[i * 3 + 1]);
            rgba_data.push(rgb_data[i * 3 + 2]);
            rgba_data.push(255);
        }
        return rgba_data;
    }

    match technique {
        OutputTechnique::None => {
            // Simple quantization without dithering - no progress needed
            let (r, g, b, a) = pixels_to_channels_rgba(srgb_pixels);
            let r_u8: Vec<u8> = r.iter().map(|v| quantize_no_dither(*v, bits_r)).collect();
            let g_u8: Vec<u8> = g.iter().map(|v| quantize_no_dither(*v, bits_g)).collect();
            let b_u8: Vec<u8> = b.iter().map(|v| quantize_no_dither(*v, bits_b)).collect();
            let a_u8: Vec<u8> = a.iter().map(|v| quantize_no_dither(*v, bits_a)).collect();
            interleave_rgba_u8(&r_u8, &g_u8, &b_u8, &a_u8)
        }
        OutputTechnique::PerChannel { mode } => {
            // Per-channel error diffusion in sRGB space
            let (r, g, b, a) = pixels_to_channels_rgba(srgb_pixels);
            let r_u8 = dither_with_mode_bits(&r, width, height, mode, seed, bits_r, None);
            let g_u8 = dither_with_mode_bits(&g, width, height, mode, seed.wrapping_add(1), bits_g, None);
            let b_u8 = dither_with_mode_bits(&b, width, height, mode, seed.wrapping_add(2), bits_b, None);
            let a_u8 = dither_with_mode_bits(&a, width, height, mode, seed.wrapping_add(3), bits_a, None);
            interleave_rgba_u8(&r_u8, &g_u8, &b_u8, &a_u8)
        }
        OutputTechnique::ColorspaceAware { mode, space } => {
            // Alpha-aware color-space dithering with progress
            // Uses alpha visibility weighting for RGB error propagation
            let (r, g, b, a) = colorspace_aware_dither_rgba_channels(
                srgb_pixels,
                width,
                height,
                bits_r,
                bits_g,
                bits_b,
                bits_a,
                space,
                mode,
                seed,
                progress,
            );
            interleave_rgba_u8(&r, &g, &b, &a)
        }
    }
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
pub fn dither_output_luminosity(
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
