//! GIF output for palettized formats.
//!
//! GIF supports up to 256 colors in a global color table, making it ideal for
//! palette-based dithering output. Unlike PNG, GIF:
//! - Only supports 1-bit transparency (fully opaque or fully transparent)
//! - Has a maximum of 256 colors
//! - Uses LZW compression (good for flat colors and patterns)

use super::color_format::ColorFormat;
use super::binary::{
    decode_rgb_pixel, decode_argb_pixel, decode_gray_pixel, decode_la_pixel,
    encode_rgb_pixel, encode_argb_pixel, encode_gray_pixel, encode_la_pixel,
};
use gif::{Encoder, Frame, Repeat};
use std::io::Cursor;

/// Check if format can be output as a GIF (≤8 bits per pixel, ≤256 colors)
pub fn supports_gif(format: &ColorFormat) -> bool {
    format.total_bits <= 8
}

/// Encode interleaved u8 data to GIF format.
///
/// This function takes already-dithered interleaved u8 data and produces a
/// GIF where the palette contains bit-replicated colors for each possible
/// packed value.
///
/// The input data format depends on the color format:
/// - Grayscale (L): 1 byte per pixel (grayscale value)
/// - Grayscale+Alpha (LA): 2 bytes per pixel (L, A)
/// - RGB: 3 bytes per pixel (R, G, B)
/// - RGBA/ARGB: 4 bytes per pixel (R, G, B, A)
///
/// Returns the GIF file bytes.
pub fn encode_palettized_gif(
    interleaved_data: &[u8],
    width: usize,
    height: usize,
    format: &ColorFormat,
) -> Result<Vec<u8>, String> {
    if !supports_gif(format) {
        return Err(format!(
            "Format {} has {} bits per pixel, GIF requires ≤8 bits (≤256 colors)",
            format.name, format.total_bits
        ));
    }

    let num_entries = 1usize << format.total_bits;

    // Generate the palette (RGB only - GIF uses separate transparency handling)
    let mut rgb_palette: Vec<u8> = Vec::with_capacity(num_entries * 3);
    let mut transparent_index: Option<u8> = None;

    for packed in 0..num_entries {
        let packed_u32 = packed as u32;

        if format.is_grayscale && format.has_alpha {
            // LA format
            let (l, a) = decode_la_pixel(packed_u32, format.bits_r, format.bits_a);
            rgb_palette.extend_from_slice(&[l, l, l]);
            // GIF only supports 1-bit transparency - use first fully transparent color
            if a == 0 && transparent_index.is_none() {
                transparent_index = Some(packed as u8);
            }
        } else if format.is_grayscale {
            // Pure grayscale (L)
            let l = decode_gray_pixel(packed_u32, format.bits_r);
            rgb_palette.extend_from_slice(&[l, l, l]);
        } else if format.has_alpha {
            // ARGB format
            let (r, g, b, a) = decode_argb_pixel(
                packed_u32,
                format.bits_a,
                format.bits_r,
                format.bits_g,
                format.bits_b,
            );
            rgb_palette.extend_from_slice(&[r, g, b]);
            // GIF only supports 1-bit transparency
            if a == 0 && transparent_index.is_none() {
                transparent_index = Some(packed as u8);
            }
        } else {
            // RGB format
            let (r, g, b) = decode_rgb_pixel(packed_u32, format.bits_r, format.bits_g, format.bits_b);
            rgb_palette.extend_from_slice(&[r, g, b]);
        }
    }

    // Convert interleaved data to palette indices
    let pixel_count = width * height;
    let bytes_per_input_pixel = if format.is_grayscale && format.has_alpha {
        2 // LA
    } else if format.is_grayscale {
        1 // L
    } else if format.has_alpha {
        4 // RGBA
    } else {
        3 // RGB
    };

    if interleaved_data.len() < pixel_count * bytes_per_input_pixel {
        return Err(format!(
            "Input data too small: got {} bytes, expected {} bytes for {}x{} image",
            interleaved_data.len(),
            pixel_count * bytes_per_input_pixel,
            width,
            height
        ));
    }

    // Convert each pixel to a palette index (packed value)
    let mut indices = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        let offset = i * bytes_per_input_pixel;
        let index = if format.is_grayscale && format.has_alpha {
            let l = interleaved_data[offset];
            let a = interleaved_data[offset + 1];
            encode_la_pixel(l, a, format.bits_r, format.bits_a) as u8
        } else if format.is_grayscale {
            let l = interleaved_data[offset];
            encode_gray_pixel(l, format.bits_r) as u8
        } else if format.has_alpha {
            let r = interleaved_data[offset];
            let g = interleaved_data[offset + 1];
            let b = interleaved_data[offset + 2];
            let a = interleaved_data[offset + 3];
            encode_argb_pixel(a, r, g, b, format.bits_a, format.bits_r, format.bits_g, format.bits_b) as u8
        } else {
            let r = interleaved_data[offset];
            let g = interleaved_data[offset + 1];
            let b = interleaved_data[offset + 2];
            encode_rgb_pixel(r, g, b, format.bits_r, format.bits_g, format.bits_b) as u8
        };
        indices.push(index);
    }

    // Encode to GIF
    encode_gif_with_palette(&indices, width, height, &rgb_palette, transparent_index)
}

/// Encode a GIF using explicit palette indices and palette colors.
///
/// This is used for true paletted output modes (like CGA palettes) where we have:
/// - `indices`: One byte per pixel, indexing into the palette
/// - `palette`: Explicit RGBA colors (R, G, B, A) tuples
///
/// The output GIF uses the specified palette. Fully transparent colors (A=0)
/// will use GIF's transparency extension.
pub fn encode_explicit_palette_gif(
    indices: &[u8],
    width: usize,
    height: usize,
    palette: &[(u8, u8, u8, u8)],
) -> Result<Vec<u8>, String> {
    let pixel_count = width * height;
    if indices.len() < pixel_count {
        return Err(format!(
            "Indices array too small: got {} bytes, expected {} for {}x{} image",
            indices.len(), pixel_count, width, height
        ));
    }

    if palette.is_empty() || palette.len() > 256 {
        return Err(format!(
            "Palette must have 1-256 colors, got {}",
            palette.len()
        ));
    }

    // Build RGB palette and find transparent index
    let mut rgb_palette: Vec<u8> = Vec::with_capacity(palette.len() * 3);
    let mut transparent_index: Option<u8> = None;

    for (i, &(r, g, b, a)) in palette.iter().enumerate() {
        rgb_palette.extend_from_slice(&[r, g, b]);
        // GIF only supports 1-bit transparency - use first fully transparent color
        if a == 0 && transparent_index.is_none() {
            transparent_index = Some(i as u8);
        }
    }

    encode_gif_with_palette(&indices[..pixel_count], width, height, &rgb_palette, transparent_index)
}

/// Internal function to encode GIF with a given palette.
fn encode_gif_with_palette(
    indices: &[u8],
    width: usize,
    height: usize,
    rgb_palette: &[u8],
    transparent_index: Option<u8>,
) -> Result<Vec<u8>, String> {
    let mut output = Cursor::new(Vec::new());

    // GIF palette must be a power of 2 in size (up to 256)
    let num_colors = rgb_palette.len() / 3;
    let palette_bits = (num_colors as f64).log2().ceil() as u8;
    let padded_palette_size = 1usize << palette_bits;

    // Pad palette to power of 2 if needed
    let mut padded_palette = rgb_palette.to_vec();
    while padded_palette.len() < padded_palette_size * 3 {
        padded_palette.extend_from_slice(&[0, 0, 0]);
    }

    {
        let mut encoder = Encoder::new(
            &mut output,
            width as u16,
            height as u16,
            &padded_palette,
        ).map_err(|e| format!("GIF encoder creation failed: {}", e))?;

        encoder.set_repeat(Repeat::Finite(0)).map_err(|e| format!("GIF repeat setting failed: {}", e))?;

        // Create frame from indices
        let mut frame = Frame::from_palette_pixels(
            width as u16,
            height as u16,
            indices,
            padded_palette.clone(),
            transparent_index,
        );

        // Set frame delay to 0 (static image)
        frame.delay = 0;

        encoder.write_frame(&frame).map_err(|e| format!("GIF frame write failed: {}", e))?;
    }

    Ok(output.into_inner())
}
