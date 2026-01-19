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

        // Luminosity+Alpha formats: LA1, LA2, LA4, LA8 (shorthand for equal bits)
        // or LA11, LA22, LA44, LA88 (explicit bits)
        // Layout: Alpha in MSB, Luminosity in LSB
        if format_upper.starts_with("LA") {
            let bits_str = &format_upper[2..];

            let (bits_l, bits_a): (u8, u8) = match bits_str.len() {
                // Single digit: same bits for both channels (LA4 = LA44)
                1 => {
                    let bits: u8 = bits_str
                        .parse()
                        .map_err(|_| format!("Invalid bit count in '{}'", format))?;
                    (bits, bits)
                }
                // Two digits: individual channel bits (LA44, LA88, etc.)
                2 => {
                    let l: u8 = bits_str[0..1]
                        .parse()
                        .map_err(|_| format!("Invalid luminosity bit count in '{}'", format))?;
                    let a: u8 = bits_str[1..2]
                        .parse()
                        .map_err(|_| format!("Invalid alpha bit count in '{}'", format))?;
                    (l, a)
                }
                _ => {
                    return Err(format!(
                        "Invalid LA format '{}': expected LA followed by 1 digit (e.g., LA4) or 2 digits (e.g., LA44)",
                        format
                    ));
                }
            };

            if bits_l < 1 || bits_l > 8 {
                return Err(format!("Luminosity bits must be 1-8, got {}", bits_l));
            }
            if bits_a < 1 || bits_a > 8 {
                return Err(format!("Alpha bits must be 1-8, got {}", bits_a));
            }

            let total_bits = bits_l + bits_a;

            // Use shorthand name when bits are equal (LA4 instead of LA44)
            let name = if bits_l == bits_a {
                format!("LA{}", bits_l)
            } else {
                format!("LA{}{}", bits_l, bits_a)
            };

            return Ok(ColorFormat {
                name,
                is_grayscale: true, // LA is still grayscale-based
                has_alpha: true,
                bits_a,
                bits_r: bits_l, // Store luminosity bits in bits_r
                bits_g: 0,
                bits_b: 0,
                total_bits,
            });
        }

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

            // Use shorthand name when all bits are equal (ARGB8 instead of ARGB8888)
            let name = if bits_a == bits_r && bits_r == bits_g && bits_g == bits_b {
                format!("ARGB{}", bits_a)
            } else {
                format!("ARGB{}{}{}{}", bits_a, bits_r, bits_g, bits_b)
            };

            return Ok(ColorFormat {
                name,
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

            // Use shorthand name when all bits are equal (RGB8 instead of RGB888)
            let name = if bits_r == bits_g && bits_g == bits_b {
                format!("RGB{}", bits_r)
            } else {
                format!("RGB{}{}{}", bits_r, bits_g, bits_b)
            };

            return Ok(ColorFormat {
                name,
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
            "Unknown format '{}': expected ARGB#### (e.g., ARGB1555), RGB### (e.g., RGB565), LA## (e.g., LA44), or L# (e.g., L4)",
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

    /// Check if this is an ARGB format (has alpha channel, RGB color)
    pub fn is_argb(&self) -> bool {
        self.has_alpha && !self.is_grayscale
    }

    /// Check if this is an LA format (grayscale with alpha)
    pub fn is_la(&self) -> bool {
        self.is_grayscale && self.has_alpha
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

/// Encode LA (luminosity+alpha) pixel to packed binary format
/// Layout: Alpha in MSB, Luminosity in LSB (matching hardware ordering)
/// Example LA44: A[7:4] L[3:0]
/// Example LA88: A[15:8] L[7:0]
#[inline]
pub fn encode_la_pixel(l: u8, a: u8, bits_l: u8, bits_a: u8) -> u32 {
    let l_val = (l >> (8 - bits_l)) as u32;
    let a_val = (a >> (8 - bits_a)) as u32;
    (a_val << bits_l) | l_val
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

// ============================================================================
// LA (Luminosity+Alpha) Encoding Functions
// Layout: Alpha in MSB, Luminosity in LSB
// ============================================================================

/// Write packed binary output for LA data (continuous bit stream, no row padding)
///
/// Args:
///     la_data: Interleaved LA data (LALA..., 2 bytes per pixel, L first then A)
///     width, height: Image dimensions
///     bits_l: Output bit depth for luminosity channel
///     bits_a: Output bit depth for alpha channel
pub fn encode_la_packed(
    la_data: &[u8],
    width: usize,
    height: usize,
    bits_l: u8,
    bits_a: u8,
) -> Vec<u8> {
    let total_bits = (bits_l + bits_a) as usize;
    let total_pixels = width * height;
    let mut output = Vec::new();

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = (total_bits + 7) / 8;
        output.reserve(total_pixels * bytes_per_pixel);

        for i in 0..total_pixels {
            let l = la_data[i * 2];
            let a = la_data[i * 2 + 1];
            let val = encode_la_pixel(l, a, bits_l, bits_a);
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
            let l = la_data[i * 2];
            let a = la_data[i * 2 + 1];
            let val = encode_la_pixel(l, a, bits_l, bits_a);

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

/// Write row-aligned binary output for LA data with configurable stride
///
/// Args:
///     la_data: Interleaved LA data (LALA..., 2 bytes per pixel)
///     width, height: Image dimensions
///     bits_l, bits_a: Output bit depth per channel
///     stride: Row stride alignment in bytes (must be power of 2, 1-128)
///     fill: How to fill padding bytes (Black = zeros, Repeat = repeat last pixel)
pub fn encode_la_row_aligned_stride(
    la_data: &[u8],
    width: usize,
    height: usize,
    bits_l: u8,
    bits_a: u8,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    let total_bits = (bits_l + bits_a) as usize;

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
            let last_l = la_data[last_i * 2];
            let last_a = la_data[last_i * 2 + 1];
            let last_val = encode_la_pixel(last_l, last_a, bits_l, bits_a);

            for x in 0..width {
                let i = y * width + x;
                let l = la_data[i * 2];
                let a = la_data[i * 2 + 1];
                let val = encode_la_pixel(l, a, bits_l, bits_a);
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
        let last_l = la_data[last_i * 2];
        let last_a = la_data[last_i * 2 + 1];
        let last_val = encode_la_pixel(last_l, last_a, bits_l, bits_a) as u8;

        for x in 0..width {
            let i = y * width + x;
            let l = la_data[i * 2];
            let a = la_data[i * 2 + 1];
            let val = encode_la_pixel(l, a, bits_l, bits_a);

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

/// Write row-aligned binary output for LA data (each row padded to byte boundary)
/// This is a convenience wrapper with stride=1 and black fill
pub fn encode_la_row_aligned(
    la_data: &[u8],
    width: usize,
    height: usize,
    bits_l: u8,
    bits_a: u8,
) -> Vec<u8> {
    encode_la_row_aligned_stride(la_data, width, height, bits_l, bits_a, 1, StrideFill::Black)
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

// ============================================================================
// Raw File Decoding (binary to sRGB u8)
// ============================================================================

/// Metadata for loading a raw binary image file
#[derive(Debug, Clone)]
pub struct RawImageMetadata {
    /// Color format string (e.g., "RGB565", "RGB888", "L8", "ARGB8888")
    pub format: String,
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Row stride in bytes (0 = packed, no padding)
    pub stride: usize,
}

impl RawImageMetadata {
    /// Parse from a JSON string
    pub fn from_json(json: &str) -> Result<Self, String> {
        // Simple manual JSON parsing to avoid adding serde dependency for this
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err("Invalid JSON: expected object".to_string());
        }

        let inner = &json[1..json.len() - 1];
        let mut format: Option<String> = None;
        let mut width: Option<usize> = None;
        let mut height: Option<usize> = None;
        let mut stride: usize = 0;

        // Split by commas (being careful about nested structures)
        for part in inner.split(',') {
            let part = part.trim();
            if let Some(colon_pos) = part.find(':') {
                let key = part[..colon_pos].trim().trim_matches('"');
                let value = part[colon_pos + 1..].trim();

                match key {
                    "format" => {
                        format = Some(value.trim_matches('"').to_string());
                    }
                    "width" => {
                        width = Some(value.parse().map_err(|_| format!("Invalid width: {}", value))?);
                    }
                    "height" => {
                        height = Some(value.parse().map_err(|_| format!("Invalid height: {}", value))?);
                    }
                    "stride" => {
                        stride = value.parse().map_err(|_| format!("Invalid stride: {}", value))?;
                    }
                    _ => {} // Ignore unknown keys
                }
            }
        }

        let format = format.ok_or("Missing required field: format")?;
        let width = width.ok_or("Missing required field: width")?;
        let height = height.ok_or("Missing required field: height")?;

        Ok(RawImageMetadata {
            format,
            width,
            height,
            stride,
        })
    }
}

/// Bit-replicate a value from src_bits to 8 bits (per BITDEPTH.md)
/// This is the correct way to extend bit depths to u8.
#[inline]
pub fn bit_replicate_to_u8(value: u32, src_bits: u8) -> u8 {
    if src_bits >= 8 {
        return (value & 0xFF) as u8;
    }
    if src_bits == 0 {
        return 0;
    }

    // Bit replication: ABC -> ABCABCAB for 3-bit to 8-bit
    // General: shift value left, OR with shifted copies
    let mut result = 0u32;
    let mut shift = 8 - src_bits;
    result |= value << shift;

    // Fill remaining bits by replicating
    while shift >= src_bits {
        shift -= src_bits;
        result |= value << shift;
    }

    // Handle partial replication at the end
    if shift > 0 {
        result |= value >> (src_bits - shift);
    }

    result as u8
}

/// Decode a single RGB pixel from packed bits
/// Returns (R, G, B) as u8 values (bit-replicated to 8 bits)
#[inline]
pub fn decode_rgb_pixel(packed: u32, bits_r: u8, bits_g: u8, bits_b: u8) -> (u8, u8, u8) {
    // Layout: R in MSB, then G, then B in LSB
    let b_mask = (1u32 << bits_b) - 1;
    let g_mask = (1u32 << bits_g) - 1;
    let r_mask = (1u32 << bits_r) - 1;

    let b_val = packed & b_mask;
    let g_val = (packed >> bits_b) & g_mask;
    let r_val = (packed >> (bits_b + bits_g)) & r_mask;

    (
        bit_replicate_to_u8(r_val, bits_r),
        bit_replicate_to_u8(g_val, bits_g),
        bit_replicate_to_u8(b_val, bits_b),
    )
}

/// Decode a single ARGB pixel from packed bits
/// Returns (R, G, B, A) as u8 values (bit-replicated to 8 bits)
/// Layout: A in MSB, then R, G, B toward LSB
#[inline]
pub fn decode_argb_pixel(packed: u32, bits_a: u8, bits_r: u8, bits_g: u8, bits_b: u8) -> (u8, u8, u8, u8) {
    let b_mask = (1u32 << bits_b) - 1;
    let g_mask = (1u32 << bits_g) - 1;
    let r_mask = (1u32 << bits_r) - 1;
    let a_mask = (1u32 << bits_a) - 1;

    let b_val = packed & b_mask;
    let g_val = (packed >> bits_b) & g_mask;
    let r_val = (packed >> (bits_b + bits_g)) & r_mask;
    let a_val = (packed >> (bits_b + bits_g + bits_r)) & a_mask;

    (
        bit_replicate_to_u8(r_val, bits_r),
        bit_replicate_to_u8(g_val, bits_g),
        bit_replicate_to_u8(b_val, bits_b),
        bit_replicate_to_u8(a_val, bits_a),
    )
}

/// Decode a grayscale pixel from packed bits
/// Returns L as u8 (bit-replicated to 8 bits)
#[inline]
pub fn decode_gray_pixel(packed: u32, bits: u8) -> u8 {
    let mask = (1u32 << bits) - 1;
    let val = packed & mask;
    bit_replicate_to_u8(val, bits)
}

/// Decode an LA (luminosity+alpha) pixel from packed bits
/// Layout: Alpha in MSB, Luminosity in LSB
/// Returns (L, A) as u8 values (bit-replicated to 8 bits)
#[inline]
pub fn decode_la_pixel(packed: u32, bits_l: u8, bits_a: u8) -> (u8, u8) {
    let l_mask = (1u32 << bits_l) - 1;
    let a_mask = (1u32 << bits_a) - 1;

    let l_val = packed & l_mask;
    let a_val = (packed >> bits_l) & a_mask;

    (
        bit_replicate_to_u8(l_val, bits_l),
        bit_replicate_to_u8(a_val, bits_a),
    )
}

/// Result of decoding a raw image
pub struct DecodedRawImage {
    /// Interleaved RGBA u8 data (always 4 bytes per pixel)
    pub pixels: Vec<u8>,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Whether the original format had alpha
    pub has_alpha: bool,
    /// Whether the original format was grayscale
    pub is_grayscale: bool,
}

/// Decode a raw binary image file to interleaved RGBA u8
///
/// This decodes packed binary formats (RGB565, RGB888, L8, ARGB8888, etc.)
/// and converts them to standard RGBA u8 format using bit replication
/// as specified in BITDEPTH.md.
///
/// The output is always sRGB RGBA with 8 bits per channel.
/// For RGB formats, alpha is set to 255 (fully opaque).
/// For grayscale formats, R=G=B=L and A=255.
pub fn decode_raw_image(data: &[u8], metadata: &RawImageMetadata) -> Result<DecodedRawImage, String> {
    let format = ColorFormat::parse(&metadata.format)?;
    let width = metadata.width;
    let height = metadata.height;
    let pixel_count = width * height;

    // Calculate expected row stride
    let total_bits = format.total_bits as usize;
    let bytes_per_pixel = (total_bits + 7) / 8;
    let packed_bytes_per_row = if total_bits >= 8 {
        width * bytes_per_pixel
    } else {
        let pixels_per_byte = 8 / total_bits;
        (width + pixels_per_byte - 1) / pixels_per_byte
    };

    let row_stride = if metadata.stride > 0 {
        metadata.stride
    } else {
        packed_bytes_per_row
    };

    // Validate data size
    let expected_size = row_stride * height;
    if data.len() < expected_size {
        return Err(format!(
            "Raw data too small: got {} bytes, expected at least {} bytes ({}x{}, stride={})",
            data.len(), expected_size, width, height, row_stride
        ));
    }

    // Allocate output buffer (always RGBA)
    let mut pixels = vec![0u8; pixel_count * 4];

    // Special case: RGB666 (4 pixels per 9 bytes)
    if format.is_rgb666() {
        decode_rgb666_raw(data, width, height, row_stride, &mut pixels)?;
        return Ok(DecodedRawImage {
            pixels,
            width,
            height,
            has_alpha: false,
            is_grayscale: false,
        });
    }

    // Decode based on format type
    // Check LA before generic grayscale (LA is is_grayscale && has_alpha)
    if format.is_la() {
        decode_la_raw(data, width, height, row_stride, format.bits_r, format.bits_a, &mut pixels)?;
        Ok(DecodedRawImage {
            pixels,
            width,
            height,
            has_alpha: true,
            is_grayscale: true,
        })
    } else if format.is_grayscale {
        decode_gray_raw(data, width, height, row_stride, format.bits_r, &mut pixels)?;
        Ok(DecodedRawImage {
            pixels,
            width,
            height,
            has_alpha: false,
            is_grayscale: true,
        })
    } else if format.has_alpha {
        decode_argb_raw(data, width, height, row_stride, format.bits_a, format.bits_r, format.bits_g, format.bits_b, &mut pixels)?;
        Ok(DecodedRawImage {
            pixels,
            width,
            height,
            has_alpha: true,
            is_grayscale: false,
        })
    } else {
        decode_rgb_raw(data, width, height, row_stride, format.bits_r, format.bits_g, format.bits_b, &mut pixels)?;
        Ok(DecodedRawImage {
            pixels,
            width,
            height,
            has_alpha: false,
            is_grayscale: false,
        })
    }
}

/// Decode grayscale raw data
fn decode_gray_raw(
    data: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
    bits: u8,
    output: &mut [u8],
) -> Result<(), String> {
    let total_bits = bits as usize;

    if total_bits >= 8 {
        // Byte-aligned (L8)
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let l = data[row_start + x];
                let out_idx = (y * width + x) * 4;
                output[out_idx] = l;
                output[out_idx + 1] = l;
                output[out_idx + 2] = l;
                output[out_idx + 3] = 255;
            }
        }
    } else {
        // Sub-byte (L1, L2, L4)
        let pixels_per_byte = 8 / total_bits;
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let byte_idx = x / pixels_per_byte;
                let bit_offset = (pixels_per_byte - 1 - (x % pixels_per_byte)) * total_bits;
                let mask = (1u32 << bits) - 1;
                let packed = ((data[row_start + byte_idx] >> bit_offset) as u32) & mask;
                let l = decode_gray_pixel(packed, bits);

                let out_idx = (y * width + x) * 4;
                output[out_idx] = l;
                output[out_idx + 1] = l;
                output[out_idx + 2] = l;
                output[out_idx + 3] = 255;
            }
        }
    }

    Ok(())
}

/// Decode RGB raw data
fn decode_rgb_raw(
    data: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    output: &mut [u8],
) -> Result<(), String> {
    let total_bits = (bits_r + bits_g + bits_b) as usize;
    let bytes_per_pixel = (total_bits + 7) / 8;

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let pixel_start = row_start + x * bytes_per_pixel;

                // Read little-endian
                let mut packed = 0u32;
                for i in 0..bytes_per_pixel {
                    packed |= (data[pixel_start + i] as u32) << (i * 8);
                }

                let (r, g, b) = decode_rgb_pixel(packed, bits_r, bits_g, bits_b);
                let out_idx = (y * width + x) * 4;
                output[out_idx] = r;
                output[out_idx + 1] = g;
                output[out_idx + 2] = b;
                output[out_idx + 3] = 255;
            }
        }
    } else {
        // Sub-byte pixels (rare for RGB, but handle it)
        let pixels_per_byte = 8 / total_bits;
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let byte_idx = x / pixels_per_byte;
                let bit_offset = (pixels_per_byte - 1 - (x % pixels_per_byte)) * total_bits;
                let mask = (1u32 << total_bits) - 1;
                let packed = ((data[row_start + byte_idx] >> bit_offset) as u32) & mask;

                let (r, g, b) = decode_rgb_pixel(packed, bits_r, bits_g, bits_b);
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

/// Decode LA raw data (grayscale + alpha)
fn decode_la_raw(
    data: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
    bits_l: u8,
    bits_a: u8,
    output: &mut [u8],
) -> Result<(), String> {
    let total_bits = (bits_l + bits_a) as usize;
    let bytes_per_pixel = (total_bits + 7) / 8;

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels (LA44, LA88)
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let pixel_start = row_start + x * bytes_per_pixel;

                // Read little-endian
                let mut packed = 0u32;
                for i in 0..bytes_per_pixel {
                    packed |= (data[pixel_start + i] as u32) << (i * 8);
                }

                let (l, a) = decode_la_pixel(packed, bits_l, bits_a);
                let out_idx = (y * width + x) * 4;
                output[out_idx] = l;
                output[out_idx + 1] = l;
                output[out_idx + 2] = l;
                output[out_idx + 3] = a;
            }
        }
    } else {
        // Sub-byte pixels (LA11, LA22)
        let pixels_per_byte = 8 / total_bits;
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let byte_idx = x / pixels_per_byte;
                let bit_offset = (pixels_per_byte - 1 - (x % pixels_per_byte)) * total_bits;
                let mask = (1u32 << total_bits) - 1;
                let packed = ((data[row_start + byte_idx] >> bit_offset) as u32) & mask;

                let (l, a) = decode_la_pixel(packed, bits_l, bits_a);
                let out_idx = (y * width + x) * 4;
                output[out_idx] = l;
                output[out_idx + 1] = l;
                output[out_idx + 2] = l;
                output[out_idx + 3] = a;
            }
        }
    }

    Ok(())
}

/// Decode ARGB raw data
fn decode_argb_raw(
    data: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
    bits_a: u8,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    output: &mut [u8],
) -> Result<(), String> {
    let total_bits = (bits_a + bits_r + bits_g + bits_b) as usize;
    let bytes_per_pixel = (total_bits + 7) / 8;

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let pixel_start = row_start + x * bytes_per_pixel;

                // Read little-endian
                let mut packed = 0u32;
                for i in 0..bytes_per_pixel {
                    packed |= (data[pixel_start + i] as u32) << (i * 8);
                }

                let (r, g, b, a) = decode_argb_pixel(packed, bits_a, bits_r, bits_g, bits_b);
                let out_idx = (y * width + x) * 4;
                output[out_idx] = r;
                output[out_idx + 1] = g;
                output[out_idx + 2] = b;
                output[out_idx + 3] = a;
            }
        }
    } else {
        // Sub-byte pixels (rare for ARGB)
        let pixels_per_byte = 8 / total_bits;
        for y in 0..height {
            let row_start = y * row_stride;
            for x in 0..width {
                let byte_idx = x / pixels_per_byte;
                let bit_offset = (pixels_per_byte - 1 - (x % pixels_per_byte)) * total_bits;
                let mask = (1u32 << total_bits) - 1;
                let packed = ((data[row_start + byte_idx] >> bit_offset) as u32) & mask;

                let (r, g, b, a) = decode_argb_pixel(packed, bits_a, bits_r, bits_g, bits_b);
                let out_idx = (y * width + x) * 4;
                output[out_idx] = r;
                output[out_idx + 1] = g;
                output[out_idx + 2] = b;
                output[out_idx + 3] = a;
            }
        }
    }

    Ok(())
}

/// Decode RGB666 raw data (special 4-pixels-per-9-bytes packing)
fn decode_rgb666_raw(
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

        // LA formats (Luminosity + Alpha)
        // LA4 shorthand = LA44
        let la4 = ColorFormat::parse("LA4").unwrap();
        assert_eq!(la4.bits_r, 4); // luminosity stored in bits_r
        assert_eq!(la4.bits_a, 4);
        assert_eq!(la4.total_bits, 8);
        assert!(la4.is_grayscale);
        assert!(la4.has_alpha);
        assert!(la4.is_la());
        assert!(!la4.is_argb());
        assert!(la4.supports_binary());

        // LA8 shorthand = LA88
        let la8 = ColorFormat::parse("LA8").unwrap();
        assert_eq!(la8.bits_r, 8);
        assert_eq!(la8.bits_a, 8);
        assert_eq!(la8.total_bits, 16);
        assert!(la8.is_la());
        assert!(la8.supports_binary());

        // Explicit LA44
        let la44 = ColorFormat::parse("LA44").unwrap();
        assert_eq!(la44.bits_r, 4);
        assert_eq!(la44.bits_a, 4);
        assert_eq!(la44.total_bits, 8);
        assert!(la44.is_la());

        // LA11 = 2 bits per pixel
        let la1 = ColorFormat::parse("LA1").unwrap();
        assert_eq!(la1.bits_r, 1);
        assert_eq!(la1.bits_a, 1);
        assert_eq!(la1.total_bits, 2);
        assert!(la1.is_la());
        assert!(la1.supports_binary()); // 2 bits is power-of-2

        // LA22 = 4 bits per pixel
        let la2 = ColorFormat::parse("LA2").unwrap();
        assert_eq!(la2.bits_r, 2);
        assert_eq!(la2.bits_a, 2);
        assert_eq!(la2.total_bits, 4);
        assert!(la2.is_la());
        assert!(la2.supports_binary()); // 4 bits is power-of-2
    }

    #[test]
    fn test_la_pixel_encoding() {
        // Test LA encoding: Alpha in MSB, Luminosity in LSB
        // LA44: L=255 (4 bits = 15), A=255 (4 bits = 15)
        // Packed: 15 << 4 | 15 = 0xFF
        let val = encode_la_pixel(255, 255, 4, 4);
        assert_eq!(val, 0xFF);

        // LA44: L=0, A=255
        // Packed: 15 << 4 | 0 = 0xF0
        let val = encode_la_pixel(0, 255, 4, 4);
        assert_eq!(val, 0xF0);

        // LA44: L=255, A=0
        // Packed: 0 << 4 | 15 = 0x0F
        let val = encode_la_pixel(255, 0, 4, 4);
        assert_eq!(val, 0x0F);

        // LA88: L=128, A=64
        // L = 128 >> 0 = 128, A = 64 >> 0 = 64
        // Packed: 64 << 8 | 128 = 0x4080
        let val = encode_la_pixel(128, 64, 8, 8);
        assert_eq!(val, 0x4080);

        // LA11: L=255 (1 bit = 1), A=255 (1 bit = 1)
        // Packed: 1 << 1 | 1 = 0b11 = 3
        let val = encode_la_pixel(255, 255, 1, 1);
        assert_eq!(val, 3);

        // LA11: L=0, A=255
        // Packed: 1 << 1 | 0 = 0b10 = 2
        let val = encode_la_pixel(0, 255, 1, 1);
        assert_eq!(val, 2);
    }

    #[test]
    fn test_la_pixel_decoding() {
        // Test LA decoding: Alpha in MSB, Luminosity in LSB
        // LA44: 0xFF -> L=255, A=255
        let (l, a) = decode_la_pixel(0xFF, 4, 4);
        assert_eq!(l, 255); // 15 bit-replicated to 8 bits
        assert_eq!(a, 255);

        // LA44: 0xF0 -> L=0, A=255
        let (l, a) = decode_la_pixel(0xF0, 4, 4);
        assert_eq!(l, 0);
        assert_eq!(a, 255);

        // LA44: 0x0F -> L=255, A=0
        let (l, a) = decode_la_pixel(0x0F, 4, 4);
        assert_eq!(l, 255);
        assert_eq!(a, 0);

        // LA88: 0x4080 -> L=128, A=64
        let (l, a) = decode_la_pixel(0x4080, 8, 8);
        assert_eq!(l, 128);
        assert_eq!(a, 64);
    }

    #[test]
    fn test_la_encode_decode_roundtrip() {
        // Test encode/decode roundtrip for LA44
        let test_values = [(255, 255), (0, 0), (128, 128), (255, 0), (0, 255)];

        for (l_in, a_in) in test_values {
            let encoded = encode_la_pixel(l_in, a_in, 4, 4);
            let (l_out, a_out) = decode_la_pixel(encoded, 4, 4);
            // Values should match after bit-replication
            // 4-bit quantization: 255 -> 15 -> 255, 128 -> 8 -> 136, etc.
            let l_expected = bit_replicate_to_u8((l_in >> 4) as u32, 4);
            let a_expected = bit_replicate_to_u8((a_in >> 4) as u32, 4);
            assert_eq!(l_out, l_expected, "L mismatch for input {}", l_in);
            assert_eq!(a_out, a_expected, "A mismatch for input {}", a_in);
        }
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

    // ============================================================================
    // Raw Decoding Tests
    // ============================================================================

    #[test]
    fn test_bit_replicate_to_u8() {
        // 1-bit to 8-bit: 1 -> 11111111 = 255
        assert_eq!(bit_replicate_to_u8(1, 1), 255);
        assert_eq!(bit_replicate_to_u8(0, 1), 0);

        // 2-bit to 8-bit: 11 -> 11111111 = 255
        assert_eq!(bit_replicate_to_u8(3, 2), 255);
        assert_eq!(bit_replicate_to_u8(2, 2), 0xAA); // 10 -> 10101010

        // 4-bit to 8-bit: 1111 -> 11111111 = 255
        assert_eq!(bit_replicate_to_u8(15, 4), 255);
        assert_eq!(bit_replicate_to_u8(8, 4), 0x88); // 1000 -> 10001000

        // 5-bit to 8-bit: 11111 -> 11111111 (with truncation)
        assert_eq!(bit_replicate_to_u8(31, 5), 255);

        // 6-bit to 8-bit: 111111 -> 11111111
        assert_eq!(bit_replicate_to_u8(63, 6), 255);
        assert_eq!(bit_replicate_to_u8(32, 6), 0x82); // 100000 -> 10000010

        // 8-bit: passthrough
        assert_eq!(bit_replicate_to_u8(128, 8), 128);
    }

    #[test]
    fn test_decode_rgb565() {
        // White pixel: R=31, G=63, B=31 packed as little-endian 0xFFFF
        let packed: u32 = 0xFFFF;
        let (r, g, b) = decode_rgb_pixel(packed, 5, 6, 5);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);

        // Black pixel
        let (r, g, b) = decode_rgb_pixel(0, 5, 6, 5);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 0);

        // Red pixel: R=31, G=0, B=0
        // R at bits [15:11], so 31 << 11 = 0xF800
        let packed: u32 = 0xF800;
        let (r, g, b) = decode_rgb_pixel(packed, 5, 6, 5);
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_decode_raw_image_rgb888() {
        // Create a simple 2x2 RGB888 image
        // RGB888 is packed with R in MSB, stored little-endian (so BGR memory order)
        // Encode test data properly to match our format
        let rgb_test = vec![
            255, 0, 0,    // Red pixel (R=255, G=0, B=0)
            0, 255, 0,    // Green pixel
            0, 0, 255,    // Blue pixel
            255, 255, 0,  // Yellow pixel
        ];

        // Encode using our encoder to get proper format
        let data = encode_rgb_packed(&rgb_test, 2, 2, 8, 8, 8, StrideFill::Black);

        let metadata = RawImageMetadata {
            format: "RGB888".to_string(),
            width: 2,
            height: 2,
            stride: 0, // packed
        };

        let decoded = decode_raw_image(&data, &metadata).unwrap();
        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert!(!decoded.has_alpha);
        assert!(!decoded.is_grayscale);

        // Check pixels (RGBA format) - should match original RGB values
        assert_eq!(&decoded.pixels[0..4], &[255, 0, 0, 255]); // Red
        assert_eq!(&decoded.pixels[4..8], &[0, 255, 0, 255]); // Green
        assert_eq!(&decoded.pixels[8..12], &[0, 0, 255, 255]); // Blue
        assert_eq!(&decoded.pixels[12..16], &[255, 255, 0, 255]); // Yellow
    }

    #[test]
    fn test_decode_raw_image_l8() {
        // Create a simple 2x2 L8 image
        let data = vec![0, 128, 192, 255];

        let metadata = RawImageMetadata {
            format: "L8".to_string(),
            width: 2,
            height: 2,
            stride: 0,
        };

        let decoded = decode_raw_image(&data, &metadata).unwrap();
        assert!(decoded.is_grayscale);

        // Check pixels (grayscale expanded to RGBA)
        assert_eq!(&decoded.pixels[0..4], &[0, 0, 0, 255]);
        assert_eq!(&decoded.pixels[4..8], &[128, 128, 128, 255]);
        assert_eq!(&decoded.pixels[8..12], &[192, 192, 192, 255]);
        assert_eq!(&decoded.pixels[12..16], &[255, 255, 255, 255]);
    }

    #[test]
    fn test_raw_metadata_from_json() {
        let json = r#"{"format": "RGB565", "width": 128, "height": 64, "stride": 256}"#;
        let meta = RawImageMetadata::from_json(json).unwrap();
        assert_eq!(meta.format, "RGB565");
        assert_eq!(meta.width, 128);
        assert_eq!(meta.height, 64);
        assert_eq!(meta.stride, 256);

        // Without stride (defaults to 0)
        let json = r#"{"format": "L8", "width": 32, "height": 32}"#;
        let meta = RawImageMetadata::from_json(json).unwrap();
        assert_eq!(meta.format, "L8");
        assert_eq!(meta.stride, 0);
    }

    #[test]
    fn test_encode_decode_roundtrip_rgb565() {
        // Create test data: 4 pixels
        let r = vec![255, 0, 128, 64];
        let g = vec![0, 255, 128, 192];
        let b = vec![128, 64, 255, 0];
        let rgb: Vec<u8> = r.iter().zip(g.iter()).zip(b.iter())
            .flat_map(|((r, g), b)| vec![*r, *g, *b])
            .collect();

        // Encode to RGB565
        let encoded = encode_rgb_packed(&rgb, 4, 1, 5, 6, 5, StrideFill::Black);

        // Decode back
        let metadata = RawImageMetadata {
            format: "RGB565".to_string(),
            width: 4,
            height: 1,
            stride: 0,
        };
        let decoded = decode_raw_image(&encoded, &metadata).unwrap();

        // Values should match after bit replication (5/6 bits expanded to 8)
        // The decoding correctly bit-replicates, so 5-bit 31 -> 255, not 248
        assert_eq!(decoded.pixels[0], 255); // R: 255 -> 5bit 31 -> 8bit 255
        assert_eq!(decoded.pixels[4], 0);   // R: 0 -> 5bit 0 -> 8bit 0
    }
}
