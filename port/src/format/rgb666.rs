//! RGB666 format encoding and decoding.
//!
//! RGB666 is a special 18-bit format that packs 4 pixels into 9 bytes.
//! Each pixel has 6 bits per channel (R6 G6 B6).

use super::color_format::ColorFormat;
use super::binary::{align_to_stride, StrideFill, bit_replicate_to_u8};

/// Encode RGB666 data with row alignment and configurable stride
///
/// Args:
///     rgb_data: Interleaved RGB data (RGBRGB..., 3 bytes per pixel)
///     stride: Row stride alignment in bytes (must be power of 2, 1-128)
///     fill: How to fill padding bytes (Black = zeros, Repeat = repeat last pixel)
pub fn encode_rgb666_row_aligned_stride(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    // Each row: groups of 4 pixels -> 9 bytes
    let groups_per_row = (width + 3) / 4;
    let raw_bytes_per_row = groups_per_row * 9;
    let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
    let padding = aligned_bytes_per_row - raw_bytes_per_row;
    let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

    for y in 0..height {
        let row_start = y * width;
        let mut x = 0;

        // Get last pixel for repeat fill
        let last_idx = row_start + width - 1;
        let last_r = (rgb_data[last_idx * 3] >> 2) as u32;
        let last_g = (rgb_data[last_idx * 3 + 1] >> 2) as u32;
        let last_b = (rgb_data[last_idx * 3 + 2] >> 2) as u32;
        let last_rgb6: u32 = last_b | (last_g << 6) | (last_r << 12);

        while x < width {
            let mut block = [0u8; 9];

            for i in 0..4 {
                let rgb6 = if x + i >= width {
                    // Past end of row: use fill mode
                    match fill {
                        StrideFill::Black => continue, // Leave as zeros
                        StrideFill::Repeat => last_rgb6, // Use last pixel
                    }
                } else {
                    let idx = row_start + x + i;
                    let r = (rgb_data[idx * 3] >> 2) as u32;
                    let g = (rgb_data[idx * 3 + 1] >> 2) as u32;
                    let b = (rgb_data[idx * 3 + 2] >> 2) as u32;
                    b | (g << 6) | (r << 12)
                };

                let first_byte = i * 2;
                let bit_offset = i * 2;
                let rgb6_shifted: u32 = rgb6 << bit_offset;

                block[first_byte] |= (rgb6_shifted & 0xFF) as u8;
                block[first_byte + 1] = ((rgb6_shifted >> 8) & 0xFF) as u8;
                block[first_byte + 2] |= ((rgb6_shifted >> 16) & 0xFF) as u8;
            }

            output.extend_from_slice(&block);
            x += 4;
        }

        // Add padding bytes to align row to stride
        match fill {
            StrideFill::Black => {
                for _ in 0..padding {
                    output.push(0);
                }
            }
            StrideFill::Repeat => {
                // For RGB666, we need to repeat in 9-byte blocks (4 pixels)
                // For simplicity, we'll repeat the encoded last pixel value byte-wise
                let fill_bytes = [
                    (last_rgb6 & 0xFF) as u8,
                    ((last_rgb6 >> 8) & 0xFF) as u8,
                    ((last_rgb6 >> 16) & 0x03) as u8,
                ];
                for i in 0..padding {
                    output.push(fill_bytes[i % 3]);
                }
            }
        }
    }

    output
}

/// Encode RGB666 data with row alignment (each row padded to 9-byte boundary for 4-pixel groups)
/// This is a convenience wrapper with stride=1 (no additional alignment) and black fill
///
/// Args:
///     rgb_data: Interleaved RGB data (RGBRGB..., 3 bytes per pixel)
pub fn encode_rgb666_row_aligned(
    rgb_data: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    encode_rgb666_row_aligned_stride(rgb_data, width, height, 1, StrideFill::Black)
}

/// Decode RGB666 raw data (special 4-pixels-per-9-bytes packing)
pub fn decode_rgb666_raw(
    data: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
    output: &mut [u8],
) -> Result<(), String> {
    // RGB666 packs 4 pixels into 9 bytes
    // Each pixel is 18 bits: R6 G6 B6 (R in MSB, B in LSB)
    // Pixels are packed sequentially into the 72-bit block

    let groups_per_row = (width + 3) / 4;

    for y in 0..height {
        let row_start = y * row_stride;

        for group in 0..groups_per_row {
            let group_start = row_start + group * 9;

            // Read 9 bytes (72 bits)
            let mut bits72 = [0u8; 9];
            for i in 0..9 {
                if group_start + i < data.len() {
                    bits72[i] = data[group_start + i];
                }
            }

            // Extract 4 pixels from the 72-bit block
            for i in 0..4 {
                let x = group * 4 + i;
                if x >= width {
                    break;
                }

                // Each pixel is at bit offset i * 18
                // Extract 18 bits for this pixel
                let bit_offset = i * 2; // Byte offset in the 9-byte block
                let first_byte = i * 2;

                // Read 3 bytes containing this pixel's 18 bits
                let b0 = bits72[first_byte] as u32;
                let b1 = bits72[first_byte + 1] as u32;
                let b2 = bits72[first_byte + 2] as u32;

                let shifted = b0 | (b1 << 8) | (b2 << 16);
                let rgb18 = (shifted >> bit_offset) & 0x3FFFF; // 18 bits

                // Extract B6, G6, R6 (B in LSB, R in MSB)
                let b6 = (rgb18 & 0x3F) as u8;
                let g6 = ((rgb18 >> 6) & 0x3F) as u8;
                let r6 = ((rgb18 >> 12) & 0x3F) as u8;

                // Bit-replicate 6 bits to 8 bits
                let r = bit_replicate_to_u8(r6 as u32, 6);
                let g = bit_replicate_to_u8(g6 as u32, 6);
                let b = bit_replicate_to_u8(b6 as u32, 6);

                let out_idx = (y * width + x) * 4;
                output[out_idx] = r;
                output[out_idx + 1] = g;
                output[out_idx + 2] = b;
                output[out_idx + 3] = 255;
            }
        }
    }

    Ok(())
}

/// Calculate expected row stride for RGB666 format
pub fn rgb666_row_stride(width: usize) -> usize {
    let groups_per_row = (width + 3) / 4;
    groups_per_row * 9
}

/// Check if a ColorFormat is RGB666
pub fn is_rgb666(format: &ColorFormat) -> bool {
    format.is_rgb666()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb666_row_aligned() {
        // Test that partial groups use fill mode correctly
        // 5 pixels wide, 1 row: last group has 1 real pixel + 3 padding pixels

        // Use a distinctive last pixel: R=252 (6-bit: 63), G=0, B=0
        // First 4 pixels are black (0,0,0), last pixel is bright red (252,0,0)
        let mut rgb = vec![0u8; 5 * 3];
        rgb[4 * 3] = 252; // Last pixel R = 252

        // With Black fill, partial group positions should be zeros
        let black_fill = encode_rgb666_row_aligned_stride(&rgb, 5, 1, 1, StrideFill::Black);
        assert_eq!(black_fill.len(), 18); // 2 groups * 9 bytes

        // With Repeat fill, partial group positions should repeat the last pixel
        let repeat_fill = encode_rgb666_row_aligned_stride(&rgb, 5, 1, 1, StrideFill::Repeat);
        assert_eq!(repeat_fill.len(), 18);

        // The outputs should differ (partial pixels are filled differently)
        assert_ne!(black_fill, repeat_fill);

        // In repeat mode, positions 1, 2, 3 in the second group should have red pixel data
        // Second group starts at byte 9
        // With repeat, the last 3 positions should have the red pixel encoded
        // The second group should not be all zeros like in black mode
        let second_group_black = &black_fill[9..18];
        let second_group_repeat = &repeat_fill[9..18];

        // Black mode: positions after first pixel should be zero
        // First pixel (position 0) at bit offset 0 uses bytes 9, 10, 11 (partially)
        // Positions 1, 2, 3 should leave remaining bytes mostly zero in black mode

        // Repeat mode: all 4 positions should have the red pixel
        // So the byte pattern should be different and non-zero in more positions
        assert_ne!(second_group_black, second_group_repeat);
    }
}
