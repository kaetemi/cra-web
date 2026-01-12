//! Binary format encoding for dithered images.
//!
//! Provides functions to encode dithered pixel data into packed binary formats
//! suitable for embedded systems and other applications requiring compact storage.
//!
//! Supports:
//! - RGB formats with configurable bit depths (RGB565, RGB332, RGB888, etc.)
//! - Grayscale formats (L1, L2, L4, L8)
//! - Packed binary output (continuous bit stream)
//! - Row-aligned binary output (each row padded to byte boundary)

use crate::dither_common::bit_replicate;

/// Parsed color format with bit depths per channel
#[derive(Debug, Clone)]
pub struct ColorFormat {
    /// Format name (e.g., "RGB565", "L4")
    pub name: String,
    /// Whether this is a grayscale format
    pub is_grayscale: bool,
    /// Bits per red channel (or grayscale)
    pub bits_r: u8,
    /// Bits per green channel (0 for grayscale)
    pub bits_g: u8,
    /// Bits per blue channel (0 for grayscale)
    pub bits_b: u8,
    /// Total bits per pixel
    pub total_bits: u8,
}

impl ColorFormat {
    /// Parse a format string like "RGB565", "RGB111", "L4", "L8", etc.
    pub fn parse(format: &str) -> Result<Self, String> {
        let format_upper = format.to_uppercase();

        // Grayscale formats: L1, L2, L4, L8
        if format_upper.starts_with('L') {
            let bits_str = &format_upper[1..];
            let bits: u8 = bits_str
                .parse()
                .map_err(|_| format!("Invalid grayscale format '{}': expected L followed by bit count (1-8)", format))?;

            if bits < 1 || bits > 8 {
                return Err(format!("Grayscale bits must be 1-8, got {}", bits));
            }

            return Ok(ColorFormat {
                name: format_upper,
                is_grayscale: true,
                bits_r: bits,
                bits_g: 0,
                bits_b: 0,
                total_bits: bits,
            });
        }

        // RGB formats: RGB565, RGB111, RGB332, RGB888, etc.
        if format_upper.starts_with("RGB") {
            let bits_str = &format_upper[3..];

            if bits_str.len() != 3 {
                return Err(format!(
                    "Invalid RGB format '{}': expected RGB followed by 3 digits (e.g., RGB565)",
                    format
                ));
            }

            let bits_r: u8 = bits_str[0..1]
                .parse()
                .map_err(|_| format!("Invalid red bit count in '{}'", format))?;
            let bits_g: u8 = bits_str[1..2]
                .parse()
                .map_err(|_| format!("Invalid green bit count in '{}'", format))?;
            let bits_b: u8 = bits_str[2..3]
                .parse()
                .map_err(|_| format!("Invalid blue bit count in '{}'", format))?;

            if bits_r < 1 || bits_r > 8 {
                return Err(format!("Red bits must be 1-8, got {}", bits_r));
            }
            if bits_g < 1 || bits_g > 8 {
                return Err(format!("Green bits must be 1-8, got {}", bits_g));
            }
            if bits_b < 1 || bits_b > 8 {
                return Err(format!("Blue bits must be 1-8, got {}", bits_b));
            }

            let total_bits = bits_r + bits_g + bits_b;

            return Ok(ColorFormat {
                name: format_upper,
                is_grayscale: false,
                bits_r,
                bits_g,
                bits_b,
                total_bits,
            });
        }

        Err(format!(
            "Unknown format '{}': expected RGB### (e.g., RGB565) or L# (e.g., L4)",
            format
        ))
    }

    /// Create a format directly from bit depths
    pub fn from_bits(bits_r: u8, bits_g: u8, bits_b: u8) -> Self {
        let is_grayscale = bits_g == 0 && bits_b == 0;
        let total_bits = if is_grayscale {
            bits_r
        } else {
            bits_r + bits_g + bits_b
        };
        let name = if is_grayscale {
            format!("L{}", bits_r)
        } else {
            format!("RGB{}{}{}", bits_r, bits_g, bits_b)
        };

        ColorFormat {
            name,
            is_grayscale,
            bits_r,
            bits_g,
            bits_b,
            total_bits,
        }
    }

    /// Check if this format can be represented in a standard binary output
    /// Binary output is supported for formats that fit within power-of-2 sizes
    pub fn supports_binary(&self) -> bool {
        // Supported: formats where total bits is 1, 2, 4, 8, 16, 24, or 32
        matches!(self.total_bits, 1 | 2 | 4 | 8 | 16 | 24 | 32)
    }

    /// Get the number of bytes per pixel for binary output (rounded up)
    pub fn bytes_per_pixel(&self) -> usize {
        ((self.total_bits as usize) + 7) / 8
    }

    /// Get the number of pixels that fit in one byte (for sub-byte formats)
    pub fn pixels_per_byte(&self) -> usize {
        if self.total_bits >= 8 {
            1
        } else {
            8 / (self.total_bits as usize)
        }
    }
}

/// Encode RGB pixel to packed binary format
/// Returns the raw bits as a u32 (caller should mask to appropriate bit width)
#[inline]
pub fn encode_rgb_pixel(r: u8, g: u8, b: u8, bits_r: u8, bits_g: u8, bits_b: u8) -> u32 {
    // Extract the significant bits from each channel
    // The dithered values are bit-replicated, so we need to extract the original N bits
    let r_val = (r >> (8 - bits_r)) as u32;
    let g_val = (g >> (8 - bits_g)) as u32;
    let b_val = (b >> (8 - bits_b)) as u32;

    // Pack as R,G,B from MSB to LSB
    let g_shift = bits_b;
    let r_shift = bits_b + bits_g;

    (r_val << r_shift) | (g_val << g_shift) | b_val
}

/// Encode grayscale pixel to packed binary format
#[inline]
pub fn encode_gray_pixel(l: u8, bits: u8) -> u32 {
    (l >> (8 - bits)) as u32
}

/// Encode a single channel to packed binary format
#[inline]
pub fn encode_channel_pixel(value: u8, bits: u8) -> u32 {
    (value >> (8 - bits)) as u32
}

/// Write packed binary output for RGB data (continuous bit stream, no row padding)
pub fn encode_rgb_packed(
    r_data: &[u8],
    g_data: &[u8],
    b_data: &[u8],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    let total_bits = (bits_r + bits_g + bits_b) as usize;
    let total_pixels = width * height;
    let mut output = Vec::new();

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = (total_bits + 7) / 8;
        output.reserve(total_pixels * bytes_per_pixel);

        for i in 0..total_pixels {
            let val = encode_rgb_pixel(r_data[i], g_data[i], b_data[i], bits_r, bits_g, bits_b);
            // Write little-endian (LSB first)
            for byte_idx in 0..bytes_per_pixel {
                output.push(((val >> (byte_idx * 8)) & 0xFF) as u8);
            }
        }
    } else {
        // Sub-byte pixels - pack multiple pixels per byte
        let pixels_per_byte = 8 / total_bits;
        let total_bytes = (total_pixels + pixels_per_byte - 1) / pixels_per_byte;
        output.reserve(total_bytes);

        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        for i in 0..total_pixels {
            let val = encode_rgb_pixel(r_data[i], g_data[i], b_data[i], bits_r, bits_g, bits_b);

            // Pack from MSB to LSB
            let shift = 8 - bits_in_byte - total_bits;
            current_byte |= (val as u8) << shift;
            bits_in_byte += total_bits;

            if bits_in_byte == 8 {
                output.push(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }

        // Flush remaining bits
        if bits_in_byte > 0 {
            output.push(current_byte);
        }
    }

    output
}

/// Write row-aligned binary output for RGB data (each row padded to byte boundary)
pub fn encode_rgb_row_aligned(
    r_data: &[u8],
    g_data: &[u8],
    b_data: &[u8],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    let total_bits = (bits_r + bits_g + bits_b) as usize;

    if total_bits >= 8 {
        // For byte-aligned formats, row alignment is automatic
        return encode_rgb_packed(r_data, g_data, b_data, width, height, bits_r, bits_g, bits_b);
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = 8 / total_bits;
    let bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    let mut output = Vec::with_capacity(bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        for x in 0..width {
            let i = y * width + x;
            let val = encode_rgb_pixel(r_data[i], g_data[i], b_data[i], bits_r, bits_g, bits_b);

            // Pack from MSB to LSB
            let shift = 8 - bits_in_byte - total_bits;
            current_byte |= (val as u8) << shift;
            bits_in_byte += total_bits;

            if bits_in_byte == 8 {
                output.push(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }

        // Flush remaining bits at end of row
        if bits_in_byte > 0 {
            output.push(current_byte);
        }
    }

    output
}

/// Write packed binary output for grayscale data
pub fn encode_gray_packed(
    gray_data: &[u8],
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    let total_bits = bits as usize;
    let total_pixels = width * height;
    let mut output = Vec::new();

    if total_bits >= 8 {
        // Byte-aligned pixels (L8)
        output.reserve(total_pixels);
        for i in 0..total_pixels {
            output.push(gray_data[i]);
        }
    } else {
        // Sub-byte pixels - pack multiple pixels per byte
        let pixels_per_byte = 8 / total_bits;
        let total_bytes = (total_pixels + pixels_per_byte - 1) / pixels_per_byte;
        output.reserve(total_bytes);

        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        for i in 0..total_pixels {
            let val = encode_gray_pixel(gray_data[i], bits);

            // Pack from MSB to LSB
            let shift = 8 - bits_in_byte - total_bits;
            current_byte |= (val as u8) << shift;
            bits_in_byte += total_bits;

            if bits_in_byte == 8 {
                output.push(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }

        // Flush remaining bits
        if bits_in_byte > 0 {
            output.push(current_byte);
        }
    }

    output
}

/// Write row-aligned binary output for grayscale data
pub fn encode_gray_row_aligned(
    gray_data: &[u8],
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    let total_bits = bits as usize;

    if total_bits >= 8 {
        // For byte-aligned formats, row alignment is automatic
        return encode_gray_packed(gray_data, width, height, bits);
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = 8 / total_bits;
    let bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    let mut output = Vec::with_capacity(bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        for x in 0..width {
            let i = y * width + x;
            let val = encode_gray_pixel(gray_data[i], bits);

            // Pack from MSB to LSB
            let shift = 8 - bits_in_byte - total_bits;
            current_byte |= (val as u8) << shift;
            bits_in_byte += total_bits;

            if bits_in_byte == 8 {
                output.push(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }

        // Flush remaining bits at end of row
        if bits_in_byte > 0 {
            output.push(current_byte);
        }
    }

    output
}

/// Write packed binary output for a single channel
pub fn encode_channel_packed(
    channel_data: &[u8],
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    // Single channel encoding is identical to grayscale
    encode_gray_packed(channel_data, width, height, bits)
}

/// Write row-aligned binary output for a single channel
pub fn encode_channel_row_aligned(
    channel_data: &[u8],
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    // Single channel encoding is identical to grayscale
    encode_gray_row_aligned(channel_data, width, height, bits)
}

/// Check if a format string is valid
pub fn is_valid_format(format: &str) -> bool {
    ColorFormat::parse(format).is_ok()
}

/// Check if a format supports binary output
pub fn format_supports_binary(format: &str) -> bool {
    ColorFormat::parse(format)
        .map(|f| f.supports_binary())
        .unwrap_or(false)
}

/// Get total bits per pixel for a format
pub fn format_total_bits(format: &str) -> Option<u8> {
    ColorFormat::parse(format).ok().map(|f| f.total_bits)
}

/// Check if format is grayscale
pub fn format_is_grayscale(format: &str) -> bool {
    ColorFormat::parse(format)
        .map(|f| f.is_grayscale)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_parsing() {
        let rgb565 = ColorFormat::parse("RGB565").unwrap();
        assert_eq!(rgb565.bits_r, 5);
        assert_eq!(rgb565.bits_g, 6);
        assert_eq!(rgb565.bits_b, 5);
        assert_eq!(rgb565.total_bits, 16);
        assert!(!rgb565.is_grayscale);
        assert!(rgb565.supports_binary());

        let l4 = ColorFormat::parse("L4").unwrap();
        assert_eq!(l4.bits_r, 4);
        assert_eq!(l4.total_bits, 4);
        assert!(l4.is_grayscale);
        assert!(l4.supports_binary());

        let rgb111 = ColorFormat::parse("RGB111").unwrap();
        assert_eq!(rgb111.total_bits, 3);
        assert!(!rgb111.supports_binary()); // 3 bits doesn't fit neatly
    }

    #[test]
    fn test_rgb565_encoding() {
        // Test encoding a single pixel
        // R=255 (5 bits = 31), G=255 (6 bits = 63), B=255 (5 bits = 31)
        // Packed: 31 << 11 | 63 << 5 | 31 = 0xFFFF
        let val = encode_rgb_pixel(255, 255, 255, 5, 6, 5);
        assert_eq!(val, 0xFFFF);

        // R=0, G=0, B=0
        let val = encode_rgb_pixel(0, 0, 0, 5, 6, 5);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_gray_encoding() {
        // L4: 255 -> 15
        let val = encode_gray_pixel(255, 4);
        assert_eq!(val, 15);

        // L1: 255 -> 1
        let val = encode_gray_pixel(255, 1);
        assert_eq!(val, 1);

        // L1: 0 -> 0
        let val = encode_gray_pixel(0, 1);
        assert_eq!(val, 0);
    }
}
