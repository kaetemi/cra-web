//! Palettized PNG output for low-bit-depth formats.
//!
//! For formats with ≤8 bits per pixel, we can represent all possible packed values
//! (at most 256) in a PNG palette. The raw binary data becomes palette indices
//! directly, and the palette entries contain the decoded RGB(A) colors.

use super::color_format::ColorFormat;
use super::binary::{
    decode_rgb_pixel, decode_argb_pixel, decode_gray_pixel, decode_la_pixel,
    encode_rgb_pixel, encode_argb_pixel, encode_gray_pixel, encode_la_pixel,
};

/// Check if format can be output as a palettized PNG (≤8 bits per pixel)
///
/// For formats with total_bits ≤ 8, we can represent all possible packed values
/// (at most 256) in a PNG palette. The raw binary data becomes palette indices
/// directly, and the palette entries contain the decoded RGB(A) colors.
pub fn supports_palettized_png(format: &ColorFormat) -> bool {
    format.total_bits <= 8
}

/// Generate an identity palette for a color format.
///
/// For each possible packed value (0 to 2^total_bits - 1), decode it to its
/// bit-replicated RGB(A) color. This palette acts as an identity function:
/// the raw binary data becomes the palette indices directly.
///
/// Returns (rgb_palette, optional_alpha_palette) where:
/// - rgb_palette: Vec of [R, G, B] entries
/// - alpha_palette: Some(Vec<u8>) if format has alpha, None otherwise
pub fn generate_palette(format: &ColorFormat) -> Result<(Vec<[u8; 3]>, Option<Vec<u8>>), String> {
    if !supports_palettized_png(format) {
        return Err(format!(
            "Format {} has {} bits per pixel, palettized PNG requires ≤8 bits",
            format.name, format.total_bits
        ));
    }

    let num_entries = 1usize << format.total_bits;
    let mut rgb_palette = Vec::with_capacity(num_entries);
    let mut alpha_palette = if format.has_alpha {
        Some(Vec::with_capacity(num_entries))
    } else {
        None
    };

    for packed in 0..num_entries {
        let packed_u32 = packed as u32;

        if format.is_grayscale && format.has_alpha {
            // LA format
            let (l, a) = decode_la_pixel(packed_u32, format.bits_r, format.bits_a);
            rgb_palette.push([l, l, l]);
            if let Some(ref mut alpha) = alpha_palette {
                alpha.push(a);
            }
        } else if format.is_grayscale {
            // Pure grayscale (L)
            let l = decode_gray_pixel(packed_u32, format.bits_r);
            rgb_palette.push([l, l, l]);
        } else if format.has_alpha {
            // ARGB format
            let (r, g, b, a) = decode_argb_pixel(
                packed_u32,
                format.bits_a,
                format.bits_r,
                format.bits_g,
                format.bits_b,
            );
            rgb_palette.push([r, g, b]);
            if let Some(ref mut alpha) = alpha_palette {
                alpha.push(a);
            }
        } else {
            // RGB format
            let (r, g, b) = decode_rgb_pixel(packed_u32, format.bits_r, format.bits_g, format.bits_b);
            rgb_palette.push([r, g, b]);
        }
    }

    Ok((rgb_palette, alpha_palette))
}

/// Encode interleaved u8 data to palettized PNG format.
///
/// This function takes already-dithered interleaved u8 data and produces a
/// palettized PNG where the palette contains bit-replicated colors for each
/// possible packed value.
///
/// The input data format depends on the color format:
/// - Grayscale (L): 1 byte per pixel (grayscale value)
/// - Grayscale+Alpha (LA): 2 bytes per pixel (L, A)
/// - RGB: 3 bytes per pixel (R, G, B)
/// - RGBA/ARGB: 4 bytes per pixel (R, G, B, A)
///
/// Returns the PNG file bytes.
pub fn encode_palettized_png(
    interleaved_data: &[u8],
    width: usize,
    height: usize,
    format: &ColorFormat,
) -> Result<Vec<u8>, String> {
    if !supports_palettized_png(format) {
        return Err(format!(
            "Format {} has {} bits per pixel, palettized PNG requires ≤8 bits",
            format.name, format.total_bits
        ));
    }

    // Generate the palette
    let (rgb_palette, alpha_palette) = generate_palette(format)?;

    // Determine the PNG bit depth (1, 2, 4, or 8)
    let png_bit_depth = match format.total_bits {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        5..=8 => 8,
        _ => unreachable!(),
    };

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
            // LA format: encode L and A
            let l = interleaved_data[offset];
            let a = interleaved_data[offset + 1];
            encode_la_pixel(l, a, format.bits_r, format.bits_a) as u8
        } else if format.is_grayscale {
            // L format
            let l = interleaved_data[offset];
            encode_gray_pixel(l, format.bits_r) as u8
        } else if format.has_alpha {
            // RGBA -> ARGB packed format
            let r = interleaved_data[offset];
            let g = interleaved_data[offset + 1];
            let b = interleaved_data[offset + 2];
            let a = interleaved_data[offset + 3];
            encode_argb_pixel(a, r, g, b, format.bits_a, format.bits_r, format.bits_g, format.bits_b) as u8
        } else {
            // RGB format
            let r = interleaved_data[offset];
            let g = interleaved_data[offset + 1];
            let b = interleaved_data[offset + 2];
            encode_rgb_pixel(r, g, b, format.bits_r, format.bits_g, format.bits_b) as u8
        };
        indices.push(index);
    }

    // Create PNG encoder
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, width as u32, height as u32);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::from_u8(png_bit_depth).ok_or("Invalid bit depth")?);

        // Set the palette (flatten [R, G, B] entries)
        let flat_palette: Vec<u8> = rgb_palette.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        encoder.set_palette(flat_palette);

        // Set tRNS chunk for alpha if needed
        if let Some(ref alpha) = alpha_palette {
            encoder.set_trns(alpha.clone());
        }

        let mut writer = encoder.write_header().map_err(|e| format!("PNG header error: {}", e))?;

        // Pack indices according to bit depth
        let row_bytes = match png_bit_depth {
            1 => (width + 7) / 8,
            2 => (width + 3) / 4,
            4 => (width + 1) / 2,
            8 => width,
            _ => unreachable!(),
        };

        let mut packed_data = vec![0u8; row_bytes * height];

        for y in 0..height {
            let row_start = y * width;
            let packed_row_start = y * row_bytes;

            match png_bit_depth {
                8 => {
                    // 1 index per byte
                    for x in 0..width {
                        packed_data[packed_row_start + x] = indices[row_start + x];
                    }
                }
                4 => {
                    // 2 indices per byte (high nibble first)
                    for x in 0..width {
                        let byte_idx = x / 2;
                        let nibble_pos = 1 - (x % 2); // 0 = low nibble, 1 = high nibble
                        let idx = indices[row_start + x] & 0x0F;
                        packed_data[packed_row_start + byte_idx] |= idx << (nibble_pos * 4);
                    }
                }
                2 => {
                    // 4 indices per byte (MSB first)
                    for x in 0..width {
                        let byte_idx = x / 4;
                        let bit_pos = 3 - (x % 4); // 3, 2, 1, 0 for positions 0, 1, 2, 3
                        let idx = indices[row_start + x] & 0x03;
                        packed_data[packed_row_start + byte_idx] |= idx << (bit_pos * 2);
                    }
                }
                1 => {
                    // 8 indices per byte (MSB first)
                    for x in 0..width {
                        let byte_idx = x / 8;
                        let bit_pos = 7 - (x % 8);
                        let idx = indices[row_start + x] & 0x01;
                        packed_data[packed_row_start + byte_idx] |= idx << bit_pos;
                    }
                }
                _ => unreachable!(),
            }
        }

        writer.write_image_data(&packed_data).map_err(|e| format!("PNG write error: {}", e))?;
    }

    Ok(output)
}
