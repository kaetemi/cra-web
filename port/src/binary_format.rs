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


/// Parsed color format with bit depths per channel
#[derive(Debug, Clone)]
pub struct ColorFormat {
    /// Format name (e.g., "RGB565", "ARGB8888", "L4")
    pub name: String,
    /// Whether this is a grayscale format
    pub is_grayscale: bool,
    /// Whether this format includes an alpha channel
    pub has_alpha: bool,
    /// Bits per alpha channel (0 if no alpha)
    pub bits_a: u8,
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
    /// Parse a format string like "RGB565", "ARGB8888", "ARGB1555", "L4", etc.
    ///
    /// Supported formats:
    /// - Grayscale: L1, L2, L4, L8
    /// - RGB: RGB8 (=RGB888), RGB565, RGB332, RGB666, etc.
    /// - ARGB: ARGB8 (=ARGB8888), ARGB4 (=ARGB4444), ARGB1555, ARGB4444, etc.
    ///
    /// ARGB format uses hardware ordering: A in MSB, then R, G, B in LSB.
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
                has_alpha: false,
                bits_a: 0,
                bits_r: bits,
                bits_g: 0,
                bits_b: 0,
                total_bits: bits,
            });
        }

        // ARGB formats: ARGB8 (=ARGB8888), ARGB4 (=ARGB4444), ARGB1555, ARGB4444, etc.
        // Uses hardware ordering: A in MSB, R, G, B toward LSB
        if format_upper.starts_with("ARGB") {
            let bits_str = &format_upper[4..];

            let (bits_a, bits_r, bits_g, bits_b): (u8, u8, u8, u8) = match bits_str.len() {
                // Single digit: same bits for all channels (ARGB8 = ARGB8888)
                1 => {
                    let bits: u8 = bits_str
                        .parse()
                        .map_err(|_| format!("Invalid bit count in '{}'", format))?;
                    (bits, bits, bits, bits)
                }
                // Four digits: individual channel bits (ARGB1555, ARGB4444, ARGB8888)
                4 => {
                    let a: u8 = bits_str[0..1]
                        .parse()
                        .map_err(|_| format!("Invalid alpha bit count in '{}'", format))?;
                    let r: u8 = bits_str[1..2]
                        .parse()
                        .map_err(|_| format!("Invalid red bit count in '{}'", format))?;
                    let g: u8 = bits_str[2..3]
                        .parse()
                        .map_err(|_| format!("Invalid green bit count in '{}'", format))?;
                    let b: u8 = bits_str[3..4]
                        .parse()
                        .map_err(|_| format!("Invalid blue bit count in '{}'", format))?;
                    (a, r, g, b)
                }
                _ => {
                    return Err(format!(
                        "Invalid ARGB format '{}': expected ARGB followed by 1 digit (e.g., ARGB8) or 4 digits (e.g., ARGB1555)",
                        format
                    ));
                }
            };

            if bits_a < 1 || bits_a > 8 {
                return Err(format!("Alpha bits must be 1-8, got {}", bits_a));
            }
            if bits_r < 1 || bits_r > 8 {
                return Err(format!("Red bits must be 1-8, got {}", bits_r));
            }
            if bits_g < 1 || bits_g > 8 {
                return Err(format!("Green bits must be 1-8, got {}", bits_g));
            }
            if bits_b < 1 || bits_b > 8 {
                return Err(format!("Blue bits must be 1-8, got {}", bits_b));
            }

            let total_bits = bits_a + bits_r + bits_g + bits_b;

            return Ok(ColorFormat {
                name: format_upper,
                is_grayscale: false,
                has_alpha: true,
                bits_a,
                bits_r,
                bits_g,
                bits_b,
                total_bits,
            });
        }

        // RGB formats: RGB8 (shorthand for RGB888), RGB565, RGB111, RGB332, RGB888, etc.
        if format_upper.starts_with("RGB") {
            let bits_str = &format_upper[3..];

            let (bits_r, bits_g, bits_b): (u8, u8, u8) = match bits_str.len() {
                // Single digit: same bits for all channels (RGB8 = RGB888)
                1 => {
                    let bits: u8 = bits_str
                        .parse()
                        .map_err(|_| format!("Invalid bit count in '{}'", format))?;
                    (bits, bits, bits)
                }
                // Three digits: individual channel bits (RGB565)
                3 => {
                    let r: u8 = bits_str[0..1]
                        .parse()
                        .map_err(|_| format!("Invalid red bit count in '{}'", format))?;
                    let g: u8 = bits_str[1..2]
                        .parse()
                        .map_err(|_| format!("Invalid green bit count in '{}'", format))?;
                    let b: u8 = bits_str[2..3]
                        .parse()
                        .map_err(|_| format!("Invalid blue bit count in '{}'", format))?;
                    (r, g, b)
                }
                _ => {
                    return Err(format!(
                        "Invalid RGB format '{}': expected RGB followed by 1 digit (e.g., RGB8) or 3 digits (e.g., RGB565)",
                        format
                    ));
                }
            };

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
                has_alpha: false,
                bits_a: 0,
                bits_r,
                bits_g,
                bits_b,
                total_bits,
            });
        }

        Err(format!(
            "Unknown format '{}': expected ARGB#### (e.g., ARGB1555), RGB### (e.g., RGB565), or L# (e.g., L4)",
            format
        ))
    }

    /// Create a format directly from bit depths (RGB only, no alpha)
    pub fn from_bits(bits_r: u8, bits_g: u8, bits_b: u8) -> Self {
        Self::from_bits_rgba(0, bits_r, bits_g, bits_b)
    }

    /// Create a format directly from bit depths including alpha
    pub fn from_bits_rgba(bits_a: u8, bits_r: u8, bits_g: u8, bits_b: u8) -> Self {
        let is_grayscale = bits_g == 0 && bits_b == 0 && bits_a == 0;
        let has_alpha = bits_a > 0;
        let total_bits = if is_grayscale {
            bits_r
        } else {
            bits_a + bits_r + bits_g + bits_b
        };
        let name = if is_grayscale {
            format!("L{}", bits_r)
        } else if has_alpha {
            format!("ARGB{}{}{}{}", bits_a, bits_r, bits_g, bits_b)
        } else {
            format!("RGB{}{}{}", bits_r, bits_g, bits_b)
        };

        ColorFormat {
            name,
            is_grayscale,
            has_alpha,
            bits_a,
            bits_r,
            bits_g,
            bits_b,
            total_bits,
        }
    }

    /// Check if this format can be represented in a standard binary output
    /// Binary output is supported for formats that fit within power-of-2 sizes,
    /// plus special support for 18-bit RGB666 (4 pixels packed into 9 bytes)
    pub fn supports_binary(&self) -> bool {
        // Supported: formats where total bits is 1, 2, 4, 8, 16, 18 (RGB666), 24, or 32
        matches!(self.total_bits, 1 | 2 | 4 | 8 | 16 | 18 | 24 | 32)
    }

    /// Check if this format uses the special RGB666 packing (4 pixels -> 9 bytes)
    pub fn is_rgb666(&self) -> bool {
        !self.is_grayscale && !self.has_alpha && self.bits_r == 6 && self.bits_g == 6 && self.bits_b == 6
    }

    /// Check if this is an ARGB format (has alpha channel)
    pub fn is_argb(&self) -> bool {
        self.has_alpha
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

/// Align a byte count up to the specified stride (must be power of 2)
#[inline]
pub fn align_to_stride(bytes: usize, stride: usize) -> usize {
    debug_assert!(stride.is_power_of_two(), "stride must be power of 2");
    (bytes + stride - 1) & !(stride - 1)
}

/// Validate stride alignment value (must be power of 2, 1-128)
pub fn is_valid_stride(stride: usize) -> bool {
    stride.is_power_of_two() && stride >= 1 && stride <= 128
}

/// Stride padding fill mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrideFill {
    /// Fill padding with zeros (black)
    #[default]
    Black,
    /// Repeat the last pixel to fill padding
    Repeat,
}

impl StrideFill {
    /// Convert from u8 for WASM interface (0 = Black, 1 = Repeat)
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => StrideFill::Repeat,
            _ => StrideFill::Black,
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

/// Encode ARGB pixel to packed binary format (hardware ordering)
/// Returns the raw bits as a u32 (caller should mask to appropriate bit width)
/// Layout: A in MSB, then R, G, B toward LSB
/// Example ARGB1555: A[15] R[14:10] G[9:5] B[4:0]
/// Example ARGB8888: A[31:24] R[23:16] G[15:8] B[7:0]
#[inline]
pub fn encode_argb_pixel(a: u8, r: u8, g: u8, b: u8, bits_a: u8, bits_r: u8, bits_g: u8, bits_b: u8) -> u32 {
    // Extract the significant bits from each channel
    let a_val = (a >> (8 - bits_a)) as u32;
    let r_val = (r >> (8 - bits_r)) as u32;
    let g_val = (g >> (8 - bits_g)) as u32;
    let b_val = (b >> (8 - bits_b)) as u32;

    // Pack as A,R,G,B from MSB to LSB
    let g_shift = bits_b;
    let r_shift = bits_b + bits_g;
    let a_shift = bits_b + bits_g + bits_r;

    (a_val << a_shift) | (r_val << r_shift) | (g_val << g_shift) | b_val
}

/// Encode RGB666 data using 4-pixel-to-9-byte packing
///
/// This matches the reference implementation where 4 pixels (4 × 18 = 72 bits)
/// are packed into 9 bytes as a continuous bit stream.
///
/// Each pixel is encoded as R6 G6 B6 (R in MSB, B in LSB = 18 bits)
/// The 4 pixels are packed sequentially into the 72-bit block.
///
/// Args:
///     rgb_data: Interleaved RGB data (RGBRGB..., 3 bytes per pixel)
pub fn encode_rgb666_packed(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    fill: StrideFill,
) -> Vec<u8> {
    let total_pixels = width * height;
    // Calculate output size: each group of 4 pixels -> 9 bytes
    // Round up to handle partial groups at end of image
    let num_groups = (total_pixels + 3) / 4;
    let mut output = Vec::with_capacity(num_groups * 9);

    // Get last pixel for repeat fill
    let last_idx = total_pixels - 1;
    let last_r = (rgb_data[last_idx * 3] >> 2) as u32;
    let last_g = (rgb_data[last_idx * 3 + 1] >> 2) as u32;
    let last_b = (rgb_data[last_idx * 3 + 2] >> 2) as u32;
    let last_rgb6: u32 = last_b | (last_g << 6) | (last_r << 12);

    let mut pixel_idx = 0;
    while pixel_idx < total_pixels {
        // Process 4 pixels at a time into a 9-byte block
        let mut block = [0u8; 9];

        for i in 0..4 {
            let rgb6 = if pixel_idx + i >= total_pixels {
                // Past end of image: use fill mode
                match fill {
                    StrideFill::Black => continue, // Leave as zeros
                    StrideFill::Repeat => last_rgb6, // Use last pixel
                }
            } else {
                let idx = pixel_idx + i;
                let r = (rgb_data[idx * 3] >> 2) as u32; // 6 bits
                let g = (rgb_data[idx * 3 + 1] >> 2) as u32; // 6 bits
                let b = (rgb_data[idx * 3 + 2] >> 2) as u32; // 6 bits
                b | (g << 6) | (r << 12)
            };

            // Pack as B6 | G6<<6 | R6<<12 = 18 bits total
            let first_byte = i * 2;
            let bit_offset = i * 2; // 0, 2, 4, 6
            let rgb6_shifted: u32 = rgb6 << bit_offset;

            block[first_byte] |= (rgb6_shifted & 0xFF) as u8;
            block[first_byte + 1] = ((rgb6_shifted >> 8) & 0xFF) as u8;
            block[first_byte + 2] |= ((rgb6_shifted >> 16) & 0xFF) as u8;
        }

        output.extend_from_slice(&block);
        pixel_idx += 4;
    }

    output
}

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

/// Write packed binary output for RGB data (continuous bit stream, no row padding)
///
/// Args:
///     rgb_data: Interleaved RGB data (RGBRGB..., 3 bytes per pixel)
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per channel
///     fill: How to fill partial groups (only applies to RGB666's 4-pixel grouping)
pub fn encode_rgb_packed(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    fill: StrideFill,
) -> Vec<u8> {
    // Special case: RGB666 uses 4-pixel-to-9-byte packing
    if bits_r == 6 && bits_g == 6 && bits_b == 6 {
        return encode_rgb666_packed(rgb_data, width, height, fill);
    }

    let total_bits = (bits_r + bits_g + bits_b) as usize;
    let total_pixels = width * height;
    let mut output = Vec::new();

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = (total_bits + 7) / 8;
        output.reserve(total_pixels * bytes_per_pixel);

        for i in 0..total_pixels {
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];
            let val = encode_rgb_pixel(r, g, b, bits_r, bits_g, bits_b);
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
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];
            let val = encode_rgb_pixel(r, g, b, bits_r, bits_g, bits_b);

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

/// Write row-aligned binary output for RGB data with configurable stride
///
/// Args:
///     rgb_data: Interleaved RGB data (RGBRGB..., 3 bytes per pixel)
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Output bit depth per channel
///     stride: Row stride alignment in bytes (must be power of 2, 1-128)
///     fill: How to fill padding bytes (Black = zeros, Repeat = repeat last pixel)
pub fn encode_rgb_row_aligned_stride(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    // Special case: RGB666 uses 4-pixel-to-9-byte packing
    if bits_r == 6 && bits_g == 6 && bits_b == 6 {
        return encode_rgb666_row_aligned_stride(rgb_data, width, height, stride, fill);
    }

    let total_bits = (bits_r + bits_g + bits_b) as usize;

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = (total_bits + 7) / 8;
        let raw_bytes_per_row = width * bytes_per_pixel;
        let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
        let padding = aligned_bytes_per_row - raw_bytes_per_row;
        let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

        for y in 0..height {
            // Get last pixel for repeat fill
            let last_i = y * width + width - 1;
            let last_r = rgb_data[last_i * 3];
            let last_g = rgb_data[last_i * 3 + 1];
            let last_b = rgb_data[last_i * 3 + 2];
            let last_val = encode_rgb_pixel(last_r, last_g, last_b, bits_r, bits_g, bits_b);

            for x in 0..width {
                let i = y * width + x;
                let r = rgb_data[i * 3];
                let g = rgb_data[i * 3 + 1];
                let b = rgb_data[i * 3 + 2];
                let val = encode_rgb_pixel(r, g, b, bits_r, bits_g, bits_b);
                // Write little-endian (LSB first)
                for byte_idx in 0..bytes_per_pixel {
                    output.push(((val >> (byte_idx * 8)) & 0xFF) as u8);
                }
            }
            // Add padding bytes to align row to stride
            match fill {
                StrideFill::Black => {
                    for _ in 0..padding {
                        output.push(0);
                    }
                }
                StrideFill::Repeat => {
                    // Repeat the encoded last pixel bytes
                    for i in 0..padding {
                        let byte_idx = i % bytes_per_pixel;
                        output.push(((last_val >> (byte_idx * 8)) & 0xFF) as u8);
                    }
                }
            }
        }
        return output;
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = 8 / total_bits;
    let raw_bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
    let padding = aligned_bytes_per_row - raw_bytes_per_row;
    let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        // Get last pixel for repeat fill
        let last_i = y * width + width - 1;
        let last_r = rgb_data[last_i * 3];
        let last_g = rgb_data[last_i * 3 + 1];
        let last_b = rgb_data[last_i * 3 + 2];
        let last_val = encode_rgb_pixel(last_r, last_g, last_b, bits_r, bits_g, bits_b) as u8;

        for x in 0..width {
            let i = y * width + x;
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];
            let val = encode_rgb_pixel(r, g, b, bits_r, bits_g, bits_b);

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

        // Add padding bytes to align row to stride
        match fill {
            StrideFill::Black => {
                for _ in 0..padding {
                    output.push(0);
                }
            }
            StrideFill::Repeat => {
                // Fill with repeated last pixel value
                for _ in 0..padding {
                    output.push(last_val);
                }
            }
        }
    }

    output
}

/// Write row-aligned binary output for RGB data (each row padded to byte boundary)
/// This is a convenience wrapper with stride=1 (no additional alignment beyond byte boundary) and black fill
///
/// Args:
///     rgb_data: Interleaved RGB data (RGBRGB..., 3 bytes per pixel)
pub fn encode_rgb_row_aligned(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    encode_rgb_row_aligned_stride(rgb_data, width, height, bits_r, bits_g, bits_b, 1, StrideFill::Black)
}

// ============================================================================
// ARGB Encoding Functions (hardware ordering: A in MSB, R, G, B toward LSB)
// ============================================================================

/// Write packed binary output for ARGB data (continuous bit stream, no row padding)
///
/// Args:
///     argb_data: Interleaved ARGB data (RGBARGBA..., 4 bytes per pixel, standard RGBA order in memory)
///     width, height: Image dimensions
///     bits_a, bits_r, bits_g, bits_b: Output bit depth per channel
pub fn encode_argb_packed(
    rgba_data: &[u8],
    width: usize,
    height: usize,
    bits_a: u8,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    let total_bits = (bits_a + bits_r + bits_g + bits_b) as usize;
    let total_pixels = width * height;
    let mut output = Vec::new();

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = (total_bits + 7) / 8;
        output.reserve(total_pixels * bytes_per_pixel);

        for i in 0..total_pixels {
            let r = rgba_data[i * 4];
            let g = rgba_data[i * 4 + 1];
            let b = rgba_data[i * 4 + 2];
            let a = rgba_data[i * 4 + 3];
            let val = encode_argb_pixel(a, r, g, b, bits_a, bits_r, bits_g, bits_b);
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
            let r = rgba_data[i * 4];
            let g = rgba_data[i * 4 + 1];
            let b = rgba_data[i * 4 + 2];
            let a = rgba_data[i * 4 + 3];
            let val = encode_argb_pixel(a, r, g, b, bits_a, bits_r, bits_g, bits_b);

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

/// Write row-aligned binary output for ARGB data with configurable stride
///
/// Args:
///     rgba_data: Interleaved RGBA data (RGBARGBA..., 4 bytes per pixel)
///     width, height: Image dimensions
///     bits_a, bits_r, bits_g, bits_b: Output bit depth per channel
///     stride: Row stride alignment in bytes (must be power of 2, 1-128)
///     fill: How to fill padding bytes (Black = zeros, Repeat = repeat last pixel)
pub fn encode_argb_row_aligned_stride(
    rgba_data: &[u8],
    width: usize,
    height: usize,
    bits_a: u8,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    let total_bits = (bits_a + bits_r + bits_g + bits_b) as usize;

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = (total_bits + 7) / 8;
        let raw_bytes_per_row = width * bytes_per_pixel;
        let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
        let padding = aligned_bytes_per_row - raw_bytes_per_row;
        let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

        for y in 0..height {
            // Get last pixel for repeat fill
            let last_i = y * width + width - 1;
            let last_r = rgba_data[last_i * 4];
            let last_g = rgba_data[last_i * 4 + 1];
            let last_b = rgba_data[last_i * 4 + 2];
            let last_a = rgba_data[last_i * 4 + 3];
            let last_val = encode_argb_pixel(last_a, last_r, last_g, last_b, bits_a, bits_r, bits_g, bits_b);

            for x in 0..width {
                let i = y * width + x;
                let r = rgba_data[i * 4];
                let g = rgba_data[i * 4 + 1];
                let b = rgba_data[i * 4 + 2];
                let a = rgba_data[i * 4 + 3];
                let val = encode_argb_pixel(a, r, g, b, bits_a, bits_r, bits_g, bits_b);
                // Write little-endian (LSB first)
                for byte_idx in 0..bytes_per_pixel {
                    output.push(((val >> (byte_idx * 8)) & 0xFF) as u8);
                }
            }
            // Add padding bytes to align row to stride
            match fill {
                StrideFill::Black => {
                    for _ in 0..padding {
                        output.push(0);
                    }
                }
                StrideFill::Repeat => {
                    // Repeat the encoded last pixel bytes
                    for i in 0..padding {
                        let byte_idx = i % bytes_per_pixel;
                        output.push(((last_val >> (byte_idx * 8)) & 0xFF) as u8);
                    }
                }
            }
        }
        return output;
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = 8 / total_bits;
    let raw_bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
    let padding = aligned_bytes_per_row - raw_bytes_per_row;
    let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        // Get last pixel for repeat fill
        let last_i = y * width + width - 1;
        let last_r = rgba_data[last_i * 4];
        let last_g = rgba_data[last_i * 4 + 1];
        let last_b = rgba_data[last_i * 4 + 2];
        let last_a = rgba_data[last_i * 4 + 3];
        let last_val = encode_argb_pixel(last_a, last_r, last_g, last_b, bits_a, bits_r, bits_g, bits_b) as u8;

        for x in 0..width {
            let i = y * width + x;
            let r = rgba_data[i * 4];
            let g = rgba_data[i * 4 + 1];
            let b = rgba_data[i * 4 + 2];
            let a = rgba_data[i * 4 + 3];
            let val = encode_argb_pixel(a, r, g, b, bits_a, bits_r, bits_g, bits_b);

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

        // Add padding bytes to align row to stride
        match fill {
            StrideFill::Black => {
                for _ in 0..padding {
                    output.push(0);
                }
            }
            StrideFill::Repeat => {
                // Fill with repeated last pixel value
                for _ in 0..padding {
                    output.push(last_val);
                }
            }
        }
    }

    output
}

/// Write row-aligned binary output for ARGB data (each row padded to byte boundary)
/// This is a convenience wrapper with stride=1 and black fill
pub fn encode_argb_row_aligned(
    rgba_data: &[u8],
    width: usize,
    height: usize,
    bits_a: u8,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    encode_argb_row_aligned_stride(rgba_data, width, height, bits_a, bits_r, bits_g, bits_b, 1, StrideFill::Black)
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

/// Write row-aligned binary output for grayscale data with configurable stride
///
/// stride: Row stride alignment in bytes (must be power of 2, 1-128)
/// fill: How to fill padding bytes (Black = zeros, Repeat = repeat last pixel)
pub fn encode_gray_row_aligned_stride(
    gray_data: &[u8],
    width: usize,
    height: usize,
    bits: u8,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    let total_bits = bits as usize;

    if total_bits >= 8 {
        // Byte-aligned pixels (L8)
        let raw_bytes_per_row = width;
        let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
        let padding = aligned_bytes_per_row - raw_bytes_per_row;
        let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

        for y in 0..height {
            let last_val = gray_data[y * width + width - 1];
            for x in 0..width {
                output.push(gray_data[y * width + x]);
            }
            match fill {
                StrideFill::Black => {
                    for _ in 0..padding {
                        output.push(0);
                    }
                }
                StrideFill::Repeat => {
                    for _ in 0..padding {
                        output.push(last_val);
                    }
                }
            }
        }
        return output;
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = 8 / total_bits;
    let raw_bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
    let padding = aligned_bytes_per_row - raw_bytes_per_row;
    let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        // Get last pixel for repeat fill
        let last_val = encode_gray_pixel(gray_data[y * width + width - 1], bits) as u8;

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

        // Add padding bytes to align row to stride
        match fill {
            StrideFill::Black => {
                for _ in 0..padding {
                    output.push(0);
                }
            }
            StrideFill::Repeat => {
                for _ in 0..padding {
                    output.push(last_val);
                }
            }
        }
    }

    output
}

/// Write row-aligned binary output for grayscale data
/// This is a convenience wrapper with stride=1 (no additional alignment beyond byte boundary) and black fill
pub fn encode_gray_row_aligned(
    gray_data: &[u8],
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    encode_gray_row_aligned_stride(gray_data, width, height, bits, 1, StrideFill::Black)
}

/// Write row-aligned binary output for a single channel extracted from interleaved data
/// interleaved_data: Interleaved pixel data (e.g., RGBRGB... or RGBARGBA...)
/// num_channels: Number of channels in the interleaved data (3 for RGB, 4 for RGBA)
/// channel: Which channel to extract (0=R, 1=G, 2=B, 3=A)
/// Reads directly from interleaved data without creating intermediate buffer
pub fn encode_channel_from_interleaved_row_aligned_stride(
    interleaved_data: &[u8],
    width: usize,
    height: usize,
    num_channels: usize,
    channel: usize,
    bits: u8,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    let total_bits = bits as usize;

    if total_bits >= 8 {
        // Byte-aligned pixels
        let raw_bytes_per_row = width;
        let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
        let padding = aligned_bytes_per_row - raw_bytes_per_row;
        let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

        for y in 0..height {
            let last_val = interleaved_data[(y * width + width - 1) * num_channels + channel];
            for x in 0..width {
                output.push(interleaved_data[(y * width + x) * num_channels + channel]);
            }
            match fill {
                StrideFill::Black => {
                    for _ in 0..padding {
                        output.push(0);
                    }
                }
                StrideFill::Repeat => {
                    for _ in 0..padding {
                        output.push(last_val);
                    }
                }
            }
        }
        return output;
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = 8 / total_bits;
    let raw_bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    let aligned_bytes_per_row = align_to_stride(raw_bytes_per_row, stride);
    let padding = aligned_bytes_per_row - raw_bytes_per_row;
    let mut output = Vec::with_capacity(aligned_bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        // Get last pixel for repeat fill
        let last_val = encode_gray_pixel(interleaved_data[(y * width + width - 1) * num_channels + channel], bits) as u8;

        for x in 0..width {
            let val = encode_gray_pixel(interleaved_data[(y * width + x) * num_channels + channel], bits);

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

        // Add padding bytes to align row to stride
        match fill {
            StrideFill::Black => {
                for _ in 0..padding {
                    output.push(0);
                }
            }
            StrideFill::Repeat => {
                for _ in 0..padding {
                    output.push(last_val);
                }
            }
        }
    }

    output
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

        // Single digit shorthand: RGB8 = RGB888
        let rgb8 = ColorFormat::parse("RGB8").unwrap();
        assert_eq!(rgb8.bits_r, 8);
        assert_eq!(rgb8.bits_g, 8);
        assert_eq!(rgb8.bits_b, 8);
        assert_eq!(rgb8.total_bits, 24);
        assert!(!rgb8.is_grayscale);
        assert!(rgb8.supports_binary());

        // RGB5 = RGB555
        let rgb5 = ColorFormat::parse("RGB5").unwrap();
        assert_eq!(rgb5.bits_r, 5);
        assert_eq!(rgb5.bits_g, 5);
        assert_eq!(rgb5.bits_b, 5);
        assert_eq!(rgb5.total_bits, 15);

        // ARGB formats
        let argb8 = ColorFormat::parse("ARGB8").unwrap();
        assert_eq!(argb8.bits_a, 8);
        assert_eq!(argb8.bits_r, 8);
        assert_eq!(argb8.bits_g, 8);
        assert_eq!(argb8.bits_b, 8);
        assert_eq!(argb8.total_bits, 32);
        assert!(argb8.has_alpha);
        assert!(!argb8.is_grayscale);
        assert!(argb8.supports_binary());

        let argb1555 = ColorFormat::parse("ARGB1555").unwrap();
        assert_eq!(argb1555.bits_a, 1);
        assert_eq!(argb1555.bits_r, 5);
        assert_eq!(argb1555.bits_g, 5);
        assert_eq!(argb1555.bits_b, 5);
        assert_eq!(argb1555.total_bits, 16);
        assert!(argb1555.has_alpha);

        let argb4444 = ColorFormat::parse("ARGB4444").unwrap();
        assert_eq!(argb4444.bits_a, 4);
        assert_eq!(argb4444.bits_r, 4);
        assert_eq!(argb4444.bits_g, 4);
        assert_eq!(argb4444.bits_b, 4);
        assert_eq!(argb4444.total_bits, 16);
        assert!(argb4444.has_alpha);
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

    #[test]
    fn test_rgb666_format() {
        let rgb666 = ColorFormat::parse("RGB666").unwrap();
        assert_eq!(rgb666.bits_r, 6);
        assert_eq!(rgb666.bits_g, 6);
        assert_eq!(rgb666.bits_b, 6);
        assert_eq!(rgb666.total_bits, 18);
        assert!(!rgb666.is_grayscale);
        assert!(rgb666.supports_binary()); // 18-bit is special case
        assert!(rgb666.is_rgb666());
    }

    /// Helper to create interleaved RGB from separate channels
    fn interleave(r: &[u8], g: &[u8], b: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(r.len() * 3);
        for i in 0..r.len() {
            result.push(r[i]);
            result.push(g[i]);
            result.push(b[i]);
        }
        result
    }

    #[test]
    fn test_rgb666_packing() {
        // Test RGB666 packing with 4 pixels -> 9 bytes
        // Create simple test data: 4 pixels (minimum for one complete block)
        let r = vec![255, 0, 128, 64];
        let g = vec![0, 255, 128, 192];
        let b = vec![128, 64, 255, 0];
        let rgb = interleave(&r, &g, &b);

        let packed = encode_rgb666_packed(&rgb, 4, 1, StrideFill::Black);
        assert_eq!(packed.len(), 9); // 4 pixels -> 9 bytes

        // Test encode_rgb_packed automatically routes to RGB666
        let packed2 = encode_rgb_packed(&rgb, 4, 1, 6, 6, 6, StrideFill::Black);
        assert_eq!(packed2.len(), 9);
        assert_eq!(packed, packed2);
    }

    #[test]
    fn test_rgb666_size_calculation() {
        // Test various image sizes for correct byte counts
        // Formula: ceil(pixels / 4) * 9

        // 64x64 = 4096 pixels -> 1024 groups -> 9216 bytes
        let rgb: Vec<u8> = vec![128u8; 4096 * 3]; // 128 for all R, G, B
        let packed = encode_rgb666_packed(&rgb, 64, 64, StrideFill::Black);
        assert_eq!(packed.len(), 9216);

        // 5 pixels -> 2 groups (1 full, 1 partial) -> 18 bytes
        let rgb5: Vec<u8> = vec![128u8; 5 * 3];
        let packed5 = encode_rgb666_packed(&rgb5, 5, 1, StrideFill::Black);
        assert_eq!(packed5.len(), 18); // ceil(5/4) * 9 = 2 * 9 = 18
    }

    #[test]
    fn test_rgb666_partial_group_fill() {
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

    #[test]
    fn test_rgb666_packed_fill() {
        // Test that packed version also supports fill mode
        // 5 pixels total: last group has 1 real pixel + 3 padding pixels

        // First 4 pixels are black (0,0,0), last pixel is bright red (252,0,0)
        let mut rgb = vec![0u8; 5 * 3];
        rgb[4 * 3] = 252; // Last pixel R = 252

        let black_fill = encode_rgb666_packed(&rgb, 5, 1, StrideFill::Black);
        let repeat_fill = encode_rgb666_packed(&rgb, 5, 1, StrideFill::Repeat);

        assert_eq!(black_fill.len(), 18);
        assert_eq!(repeat_fill.len(), 18);
        assert_ne!(black_fill, repeat_fill);

        // Also test through encode_rgb_packed routing
        let black_via_rgb = encode_rgb_packed(&rgb, 5, 1, 6, 6, 6, StrideFill::Black);
        let repeat_via_rgb = encode_rgb_packed(&rgb, 5, 1, 6, 6, 6, StrideFill::Repeat);

        assert_eq!(black_fill, black_via_rgb);
        assert_eq!(repeat_fill, repeat_via_rgb);
    }

    #[test]
    fn test_argb_pixel_encoding() {
        // Test ARGB1555: A[15] R[14:10] G[9:5] B[4:0]
        // A=255 (1 bit = 1), R=255 (5 bits = 31), G=255 (5 bits = 31), B=255 (5 bits = 31)
        // Packed: 1 << 15 | 31 << 10 | 31 << 5 | 31 = 0xFFFF
        let val = encode_argb_pixel(255, 255, 255, 255, 1, 5, 5, 5);
        assert_eq!(val, 0xFFFF);

        // A=0, R=0, G=0, B=0
        let val = encode_argb_pixel(0, 0, 0, 0, 1, 5, 5, 5);
        assert_eq!(val, 0);

        // Test ARGB8888: A[31:24] R[23:16] G[15:8] B[7:0]
        // A=255, R=128, G=64, B=32
        let val = encode_argb_pixel(255, 128, 64, 32, 8, 8, 8, 8);
        // 255 << 24 | 128 << 16 | 64 << 8 | 32 = 0xFF804020
        assert_eq!(val, 0xFF804020);

        // Test ARGB4444: A[15:12] R[11:8] G[7:4] B[3:0]
        // A=255 (4 bits = 15), R=255 (4 bits = 15), G=255 (4 bits = 15), B=255 (4 bits = 15)
        let val = encode_argb_pixel(255, 255, 255, 255, 4, 4, 4, 4);
        // 15 << 12 | 15 << 8 | 15 << 4 | 15 = 0xFFFF
        assert_eq!(val, 0xFFFF);
    }

    #[test]
    fn test_argb_packed_encoding() {
        // Test ARGB8888 packed encoding
        // Create 2 pixels of RGBA data (memory order: R, G, B, A)
        let rgba = vec![
            255, 0, 0, 128,     // Pixel 0: R=255, G=0, B=0, A=128
            0, 255, 0, 255,     // Pixel 1: R=0, G=255, B=0, A=255
        ];

        let packed = encode_argb_packed(&rgba, 2, 1, 8, 8, 8, 8);
        assert_eq!(packed.len(), 8); // 2 pixels * 4 bytes

        // Pixel 0: A=128, R=255, G=0, B=0 -> 0x80FF0000, little-endian: 00 00 FF 80
        assert_eq!(packed[0], 0x00); // B LSB
        assert_eq!(packed[1], 0x00); // G
        assert_eq!(packed[2], 0xFF); // R
        assert_eq!(packed[3], 0x80); // A MSB

        // Pixel 1: A=255, R=0, G=255, B=0 -> 0xFF00FF00, little-endian: 00 FF 00 FF
        assert_eq!(packed[4], 0x00); // B LSB
        assert_eq!(packed[5], 0xFF); // G
        assert_eq!(packed[6], 0x00); // R
        assert_eq!(packed[7], 0xFF); // A MSB
    }
}
