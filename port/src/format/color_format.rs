//! Color format definition and parsing.
//!
//! Provides the `ColorFormat` struct for representing packed binary color formats
//! like RGB565, ARGB8888, L4, LA44, etc.

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
    ///
    /// Handles LA formats: when bits_g == 0 && bits_b == 0 && bits_a > 0,
    /// creates an LA format with bits_r as luminosity bits.
    pub fn from_bits_rgba(bits_a: u8, bits_r: u8, bits_g: u8, bits_b: u8) -> Self {
        let has_alpha = bits_a > 0;
        // LA format: grayscale with alpha (bits_g and bits_b are 0, but bits_a > 0)
        let is_la = bits_g == 0 && bits_b == 0 && has_alpha;
        let is_grayscale = bits_g == 0 && bits_b == 0; // Both L and LA are grayscale-based

        let total_bits = if is_grayscale {
            bits_r + bits_a // For L formats, bits_a is 0; for LA, it's the alpha bits
        } else {
            bits_a + bits_r + bits_g + bits_b
        };

        let name = if is_la {
            // LA format: use shorthand when bits are equal
            if bits_r == bits_a {
                format!("LA{}", bits_r)
            } else {
                format!("LA{}{}", bits_r, bits_a)
            }
        } else if is_grayscale {
            format!("L{}", bits_r)
        } else if has_alpha {
            // ARGB format: use shorthand when all bits are equal
            if bits_a == bits_r && bits_r == bits_g && bits_g == bits_b {
                format!("ARGB{}", bits_a)
            } else {
                format!("ARGB{}{}{}{}", bits_a, bits_r, bits_g, bits_b)
            }
        } else {
            // RGB format: use shorthand when all bits are equal
            if bits_r == bits_g && bits_g == bits_b {
                format!("RGB{}", bits_r)
            } else {
                format!("RGB{}{}{}", bits_r, bits_g, bits_b)
            }
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
    fn test_from_bits_rgba() {
        // Test that from_bits_rgba correctly handles different format types

        // RGB format
        let rgb565 = ColorFormat::from_bits_rgba(0, 5, 6, 5);
        assert_eq!(rgb565.name, "RGB565");
        assert!(!rgb565.is_grayscale);
        assert!(!rgb565.has_alpha);
        assert!(!rgb565.is_la());

        // RGB shorthand (equal bits)
        let rgb8 = ColorFormat::from_bits_rgba(0, 8, 8, 8);
        assert_eq!(rgb8.name, "RGB8");

        // ARGB format
        let argb1555 = ColorFormat::from_bits_rgba(1, 5, 5, 5);
        assert_eq!(argb1555.name, "ARGB1555");
        assert!(!argb1555.is_grayscale);
        assert!(argb1555.has_alpha);
        assert!(argb1555.is_argb());
        assert!(!argb1555.is_la());

        // ARGB shorthand (equal bits)
        let argb8 = ColorFormat::from_bits_rgba(8, 8, 8, 8);
        assert_eq!(argb8.name, "ARGB8");

        // L format (grayscale, no alpha)
        let l4 = ColorFormat::from_bits_rgba(0, 4, 0, 0);
        assert_eq!(l4.name, "L4");
        assert!(l4.is_grayscale);
        assert!(!l4.has_alpha);
        assert!(!l4.is_la());

        // LA format (grayscale with alpha) - this was the bug!
        let la44 = ColorFormat::from_bits_rgba(4, 4, 0, 0);
        assert_eq!(la44.name, "LA4"); // shorthand since bits are equal
        assert!(la44.is_grayscale);
        assert!(la44.has_alpha);
        assert!(la44.is_la());
        assert!(!la44.is_argb());
        assert_eq!(la44.bits_r, 4); // luminosity
        assert_eq!(la44.bits_a, 4); // alpha
        assert_eq!(la44.total_bits, 8);

        // LA format with different bits
        let la48 = ColorFormat::from_bits_rgba(8, 4, 0, 0);
        assert_eq!(la48.name, "LA48");
        assert!(la48.is_grayscale);
        assert!(la48.has_alpha);
        assert!(la48.is_la());
        assert_eq!(la48.total_bits, 12);
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
}
