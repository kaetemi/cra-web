/// Error diffusion dithering implementations.
/// Supports Floyd-Steinberg, Jarvis-Judice-Ninke, and Mixed algorithms.

/// Quantization parameters for reduced bit depth dithering.
/// Pre-computed to avoid repeated calculations in the hot loop.
#[derive(Debug, Clone, Copy)]
struct QuantParams {
    /// Number of bits (1-8)
    bits: u8,
    /// Maximum level value (2^bits - 1), e.g., 31 for 5 bits, 255 for 8 bits
    max_level: f32,
    /// Pre-computed lookup table for bit-replicated values (only for non-8-bit)
    /// Maps level index (0..=max_level) to the 8-bit extended value
    lut: [f32; 256],
}

impl QuantParams {
    /// Create quantization parameters for given bit depth.
    /// Bit depth of 8 gives standard 256-level quantization (round to integers).
    /// Lower values give fewer levels, e.g., 3 bits = 8 levels.
    #[inline]
    fn new(bits: u8) -> Self {
        debug_assert!(bits >= 1 && bits <= 8, "bits must be 1-8");
        let levels = 1u32 << bits; // 2^bits
        let max_level = (levels - 1) as f32;

        // Pre-compute bit-replicated values for each level
        let mut lut = [0.0f32; 256];
        for v in 0..levels {
            lut[v as usize] = Self::bit_replicate(v as u8, bits) as f32;
        }

        Self { bits, max_level, lut }
    }

    /// Extend n-bit value to 8 bits by repeating the bit pattern.
    /// e.g., 3-bit value ABC becomes ABCABCAB
    #[inline]
    fn bit_replicate(value: u8, bits: u8) -> u8 {
        if bits == 8 {
            return value;
        }
        let mut result: u16 = 0;
        let mut shift = 8i8;
        while shift > 0 {
            shift -= bits as i8;
            if shift >= 0 {
                result |= (value as u16) << shift;
            } else {
                // Partial bits at the end
                result |= (value as u16) >> (-shift);
            }
        }
        result as u8
    }

    /// Quantize a value to the nearest level and return the bit-replicated value.
    /// Input and output are in 0-255 range.
    ///
    /// First finds the nearest n-bit level, then returns the 8-bit value with
    /// bits extended by repetition (e.g., 3-bit ABC becomes ABCABCAB).
    #[inline]
    fn quantize(&self, value: f32) -> f32 {
        if self.bits == 8 {
            // 8-bit: just round to nearest integer
            value.round()
        } else {
            // Find nearest level: round(value * max_level / 255)
            let level = (value * self.max_level / 255.0).round() as usize;
            // Look up the bit-replicated value
            self.lut[level.min(self.max_level as usize)]
        }
    }

    /// Create default 8-bit quantization (standard rounding to integers)
    #[inline]
    fn default_8bit() -> Self {
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            lut[i] = i as f32;
        }
        Self {
            bits: 8,
            max_level: 255.0,
            lut,
        }
    }
}

/// Wang hash for deterministic randomization - excellent avalanche properties.
/// Each bit of input affects all bits of output.
#[inline]
fn wang_hash(mut x: u32) -> u32 {
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_mul(9);
    x = x ^ (x >> 4);
    x = x.wrapping_mul(0x27d4eb2d);
    x = x ^ (x >> 15);
    x
}

/// Dithering mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitherMode {
    /// Floyd-Steinberg: Standard left-to-right scanning on all rows (default)
    #[default]
    Standard,
    /// Floyd-Steinberg: Serpentine scanning (alternating direction each row)
    /// Reduces diagonal banding artifacts
    Serpentine,
    /// Jarvis-Judice-Ninke: Standard left-to-right scanning
    /// Larger kernel (3 rows) produces smoother results but slower
    JarvisStandard,
    /// Jarvis-Judice-Ninke: Serpentine scanning
    /// Combines larger kernel with alternating scan direction
    JarvisSerpentine,
    /// Mixed: Randomly selects between FS and JJN kernels per-pixel
    /// Standard left-to-right scanning
    MixedStandard,
    /// Mixed: Randomly selects between FS and JJN kernels per-pixel
    /// Serpentine scanning (alternating direction each row)
    MixedSerpentine,
    /// Mixed: Randomly selects between FS and JJN kernels per-pixel
    /// AND randomly selects scan direction per row
    MixedRandom,
}

/// Floyd-Steinberg dithering with linear buffer and overflow padding.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///
/// Returns:
///     flat array of u8 values, same length as input
pub fn floyd_steinberg_dither(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    dither_with_mode(img, width, height, DitherMode::Standard, 0)
}

/// Floyd-Steinberg dithering with configurable bit depth.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     bits: output bit depth (1-8), controls number of quantization levels
///
/// Returns:
///     flat array of u8 values, same length as input.
///     Lower bit depths are represented in uint8 by extending bits through repetition.
pub fn floyd_steinberg_dither_bits(img: &[f32], width: usize, height: usize, bits: u8) -> Vec<u8> {
    dither_with_mode_bits(img, width, height, DitherMode::Standard, 0, bits)
}

/// Dithering with selectable algorithm and scanning mode (standard 8-bit depth).
///
/// This is the standard function used internally by histogram processing.
/// For research with configurable bit depth, use `dither_with_mode_bits`.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     mode: DitherMode variant selecting algorithm and scan pattern
///     seed: random seed for mixed modes (ignored for non-mixed modes)
///
/// Returns:
///     flat array of u8 values, same length as input
pub fn dither_with_mode(
    img: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    dither_with_mode_bits(img, width, height, mode, seed, 8)
}

/// Dithering with selectable algorithm, scanning mode, and bit depth.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     mode: DitherMode variant selecting algorithm and scan pattern
///     seed: random seed for mixed modes (ignored for non-mixed modes)
///     bits: output bit depth (1-8), controls number of quantization levels
///
/// Returns:
///     flat array of u8 values, same length as input.
///     Lower bit depths are represented in uint8 by extending bits through repetition.
pub fn dither_with_mode_bits(
    img: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
    bits: u8,
) -> Vec<u8> {
    let quant = if bits == 8 {
        QuantParams::default_8bit()
    } else {
        QuantParams::new(bits.clamp(1, 8))
    };

    match mode {
        DitherMode::Standard => floyd_steinberg_standard(img, width, height, quant),
        DitherMode::Serpentine => floyd_steinberg_serpentine(img, width, height, quant),
        DitherMode::JarvisStandard => jarvis_judice_ninke_standard(img, width, height, quant),
        DitherMode::JarvisSerpentine => jarvis_judice_ninke_serpentine(img, width, height, quant),
        DitherMode::MixedStandard => mixed_dither_standard(img, width, height, seed, quant),
        DitherMode::MixedSerpentine => mixed_dither_serpentine(img, width, height, seed, quant),
        DitherMode::MixedRandom => mixed_dither_random(img, width, height, seed, quant),
    }
}

/// Standard Floyd-Steinberg dithering (left-to-right on all rows)
fn floyd_steinberg_standard(img: &[f32], width: usize, height: usize, quant: QuantParams) -> Vec<u8> {
    let len = width * height;

    // Allocate buffer with overflow padding
    let mut buf = vec![0.0f32; len + width + 2];
    buf[..len].copy_from_slice(img);

    for i in 0..len {
        let old = buf[i];
        let new = quant.quantize(old);
        buf[i] = new;
        let err = old - new;

        // Distribute error to neighbors
        // Overflow writes hit padding, which we discard
        buf[i + 1] += err * (7.0 / 16.0);
        buf[i + width - 1] += err * (3.0 / 16.0);
        buf[i + width] += err * (5.0 / 16.0);
        buf[i + width + 1] += err * (1.0 / 16.0);
    }

    // Clamp and convert to u8, discard padding
    buf[..len]
        .iter()
        .map(|&v| v.clamp(0.0, 255.0).round() as u8)
        .collect()
}

/// Serpentine Floyd-Steinberg dithering (alternating direction each row)
/// This reduces diagonal banding artifacts by reversing scan direction on odd rows
fn floyd_steinberg_serpentine(img: &[f32], width: usize, height: usize, quant: QuantParams) -> Vec<u8> {
    // Use 2D buffer for easier coordinate handling
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();

    // Add a padding row at the bottom
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        let is_reverse = y % 2 == 1;

        if is_reverse {
            // Right-to-left scanning
            for x in (0..width).rev() {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                // Distribute error to neighbors (mirrored for reverse direction)
                // Left neighbor (was right)
                if x > 0 {
                    buf[y][x - 1] += err * (7.0 / 16.0);
                }
                // Bottom-right (was bottom-left)
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (3.0 / 16.0);
                }
                // Bottom
                buf[y + 1][x] += err * (5.0 / 16.0);
                // Bottom-left (was bottom-right)
                if x > 0 {
                    buf[y + 1][x - 1] += err * (1.0 / 16.0);
                }
            }
        } else {
            // Left-to-right scanning (standard)
            for x in 0..width {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                // Distribute error to neighbors
                // Right neighbor
                if x + 1 < width {
                    buf[y][x + 1] += err * (7.0 / 16.0);
                }
                // Bottom-left
                if x > 0 {
                    buf[y + 1][x - 1] += err * (3.0 / 16.0);
                }
                // Bottom
                buf[y + 1][x] += err * (5.0 / 16.0);
                // Bottom-right
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (1.0 / 16.0);
                }
            }
        }
    }

    // Flatten, clamp, and convert to u8
    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8))
        .collect()
}

/// Jarvis-Judice-Ninke dithering (standard left-to-right scanning)
/// Uses a larger 3-row kernel for smoother gradients than Floyd-Steinberg
///
/// Kernel (divided by 48):
///         *   7   5
///     3   5   7   5   3
///     1   3   5   3   1
fn jarvis_judice_ninke_standard(img: &[f32], width: usize, height: usize, quant: QuantParams) -> Vec<u8> {
    // Use 2D buffer for easier coordinate handling with 3-row kernel
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();

    // Add 2 padding rows at the bottom for the 3-row kernel
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        for x in 0..width {
            let old = buf[y][x];
            let new = quant.quantize(old);
            buf[y][x] = new;
            let err = old - new;

            // Distribute error using JJN kernel (divided by 48)
            // Row 0 (current row): * 7 5
            if x + 1 < width {
                buf[y][x + 1] += err * (7.0 / 48.0);
            }
            if x + 2 < width {
                buf[y][x + 2] += err * (5.0 / 48.0);
            }

            // Row 1: 3 5 7 5 3
            if x >= 2 {
                buf[y + 1][x - 2] += err * (3.0 / 48.0);
            }
            if x >= 1 {
                buf[y + 1][x - 1] += err * (5.0 / 48.0);
            }
            buf[y + 1][x] += err * (7.0 / 48.0);
            if x + 1 < width {
                buf[y + 1][x + 1] += err * (5.0 / 48.0);
            }
            if x + 2 < width {
                buf[y + 1][x + 2] += err * (3.0 / 48.0);
            }

            // Row 2: 1 3 5 3 1
            if x >= 2 {
                buf[y + 2][x - 2] += err * (1.0 / 48.0);
            }
            if x >= 1 {
                buf[y + 2][x - 1] += err * (3.0 / 48.0);
            }
            buf[y + 2][x] += err * (5.0 / 48.0);
            if x + 1 < width {
                buf[y + 2][x + 1] += err * (3.0 / 48.0);
            }
            if x + 2 < width {
                buf[y + 2][x + 2] += err * (1.0 / 48.0);
            }
        }
    }

    // Flatten, clamp, and convert to u8
    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8))
        .collect()
}

/// Jarvis-Judice-Ninke dithering with serpentine scanning
/// Alternates scan direction each row to reduce diagonal artifacts
fn jarvis_judice_ninke_serpentine(img: &[f32], width: usize, height: usize, quant: QuantParams) -> Vec<u8> {
    // Use 2D buffer for easier coordinate handling
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();

    // Add 2 padding rows at the bottom for the 3-row kernel
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        let is_reverse = y % 2 == 1;

        if is_reverse {
            // Right-to-left scanning
            for x in (0..width).rev() {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                // Distribute error using mirrored JJN kernel
                // Row 0 (current row): 5 7 *
                if x >= 1 {
                    buf[y][x - 1] += err * (7.0 / 48.0);
                }
                if x >= 2 {
                    buf[y][x - 2] += err * (5.0 / 48.0);
                }

                // Row 1: 3 5 7 5 3 (mirrored)
                if x + 2 < width {
                    buf[y + 1][x + 2] += err * (3.0 / 48.0);
                }
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (5.0 / 48.0);
                }
                buf[y + 1][x] += err * (7.0 / 48.0);
                if x >= 1 {
                    buf[y + 1][x - 1] += err * (5.0 / 48.0);
                }
                if x >= 2 {
                    buf[y + 1][x - 2] += err * (3.0 / 48.0);
                }

                // Row 2: 1 3 5 3 1 (mirrored)
                if x + 2 < width {
                    buf[y + 2][x + 2] += err * (1.0 / 48.0);
                }
                if x + 1 < width {
                    buf[y + 2][x + 1] += err * (3.0 / 48.0);
                }
                buf[y + 2][x] += err * (5.0 / 48.0);
                if x >= 1 {
                    buf[y + 2][x - 1] += err * (3.0 / 48.0);
                }
                if x >= 2 {
                    buf[y + 2][x - 2] += err * (1.0 / 48.0);
                }
            }
        } else {
            // Left-to-right scanning (same as standard)
            for x in 0..width {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                // Row 0: * 7 5
                if x + 1 < width {
                    buf[y][x + 1] += err * (7.0 / 48.0);
                }
                if x + 2 < width {
                    buf[y][x + 2] += err * (5.0 / 48.0);
                }

                // Row 1: 3 5 7 5 3
                if x >= 2 {
                    buf[y + 1][x - 2] += err * (3.0 / 48.0);
                }
                if x >= 1 {
                    buf[y + 1][x - 1] += err * (5.0 / 48.0);
                }
                buf[y + 1][x] += err * (7.0 / 48.0);
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (5.0 / 48.0);
                }
                if x + 2 < width {
                    buf[y + 1][x + 2] += err * (3.0 / 48.0);
                }

                // Row 2: 1 3 5 3 1
                if x >= 2 {
                    buf[y + 2][x - 2] += err * (1.0 / 48.0);
                }
                if x >= 1 {
                    buf[y + 2][x - 1] += err * (3.0 / 48.0);
                }
                buf[y + 2][x] += err * (5.0 / 48.0);
                if x + 1 < width {
                    buf[y + 2][x + 1] += err * (3.0 / 48.0);
                }
                if x + 2 < width {
                    buf[y + 2][x + 2] += err * (1.0 / 48.0);
                }
            }
        }
    }

    // Flatten, clamp, and convert to u8
    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8))
        .collect()
}

/// Apply Floyd-Steinberg error diffusion kernel at position (x, y)
/// For left-to-right scanning
#[inline]
fn apply_fs_kernel_ltr(buf: &mut [Vec<f32>], x: usize, y: usize, err: f32, width: usize) {
    // FS kernel:   * 7
    //            3 5 1
    if x + 1 < width {
        buf[y][x + 1] += err * (7.0 / 16.0);
    }
    if x > 0 {
        buf[y + 1][x - 1] += err * (3.0 / 16.0);
    }
    buf[y + 1][x] += err * (5.0 / 16.0);
    if x + 1 < width {
        buf[y + 1][x + 1] += err * (1.0 / 16.0);
    }
}

/// Apply Floyd-Steinberg error diffusion kernel at position (x, y)
/// For right-to-left scanning (mirrored)
#[inline]
fn apply_fs_kernel_rtl(buf: &mut [Vec<f32>], x: usize, y: usize, err: f32, width: usize) {
    // FS kernel mirrored: 7 *
    //                     1 5 3
    if x > 0 {
        buf[y][x - 1] += err * (7.0 / 16.0);
    }
    if x + 1 < width {
        buf[y + 1][x + 1] += err * (3.0 / 16.0);
    }
    buf[y + 1][x] += err * (5.0 / 16.0);
    if x > 0 {
        buf[y + 1][x - 1] += err * (1.0 / 16.0);
    }
}

/// Apply Jarvis-Judice-Ninke error diffusion kernel at position (x, y)
/// For left-to-right scanning
#[inline]
fn apply_jjn_kernel_ltr(buf: &mut [Vec<f32>], x: usize, y: usize, err: f32, width: usize) {
    // JJN kernel:     * 7 5
    //             3 5 7 5 3
    //             1 3 5 3 1
    // Row 0
    if x + 1 < width {
        buf[y][x + 1] += err * (7.0 / 48.0);
    }
    if x + 2 < width {
        buf[y][x + 2] += err * (5.0 / 48.0);
    }
    // Row 1
    if x >= 2 {
        buf[y + 1][x - 2] += err * (3.0 / 48.0);
    }
    if x >= 1 {
        buf[y + 1][x - 1] += err * (5.0 / 48.0);
    }
    buf[y + 1][x] += err * (7.0 / 48.0);
    if x + 1 < width {
        buf[y + 1][x + 1] += err * (5.0 / 48.0);
    }
    if x + 2 < width {
        buf[y + 1][x + 2] += err * (3.0 / 48.0);
    }
    // Row 2
    if x >= 2 {
        buf[y + 2][x - 2] += err * (1.0 / 48.0);
    }
    if x >= 1 {
        buf[y + 2][x - 1] += err * (3.0 / 48.0);
    }
    buf[y + 2][x] += err * (5.0 / 48.0);
    if x + 1 < width {
        buf[y + 2][x + 1] += err * (3.0 / 48.0);
    }
    if x + 2 < width {
        buf[y + 2][x + 2] += err * (1.0 / 48.0);
    }
}

/// Apply Jarvis-Judice-Ninke error diffusion kernel at position (x, y)
/// For right-to-left scanning (mirrored)
#[inline]
fn apply_jjn_kernel_rtl(buf: &mut [Vec<f32>], x: usize, y: usize, err: f32, width: usize) {
    // JJN kernel mirrored: 5 7 *
    //                    3 5 7 5 3
    //                    1 3 5 3 1
    // Row 0
    if x >= 1 {
        buf[y][x - 1] += err * (7.0 / 48.0);
    }
    if x >= 2 {
        buf[y][x - 2] += err * (5.0 / 48.0);
    }
    // Row 1
    if x + 2 < width {
        buf[y + 1][x + 2] += err * (3.0 / 48.0);
    }
    if x + 1 < width {
        buf[y + 1][x + 1] += err * (5.0 / 48.0);
    }
    buf[y + 1][x] += err * (7.0 / 48.0);
    if x >= 1 {
        buf[y + 1][x - 1] += err * (5.0 / 48.0);
    }
    if x >= 2 {
        buf[y + 1][x - 2] += err * (3.0 / 48.0);
    }
    // Row 2
    if x + 2 < width {
        buf[y + 2][x + 2] += err * (1.0 / 48.0);
    }
    if x + 1 < width {
        buf[y + 2][x + 1] += err * (3.0 / 48.0);
    }
    buf[y + 2][x] += err * (5.0 / 48.0);
    if x >= 1 {
        buf[y + 2][x - 1] += err * (3.0 / 48.0);
    }
    if x >= 2 {
        buf[y + 2][x - 2] += err * (1.0 / 48.0);
    }
}

/// Mixed dithering with standard left-to-right scanning
/// Randomly selects between Floyd-Steinberg and Jarvis-Judice-Ninke kernels per pixel
fn mixed_dither_standard(img: &[f32], width: usize, height: usize, seed: u32, quant: QuantParams) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);

    // Use 2D buffer with 2 padding rows for JJN kernel
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        for x in 0..width {
            let old = buf[y][x];
            let new = quant.quantize(old);
            buf[y][x] = new;
            let err = old - new;

            // Deterministically choose kernel based on position and seed
            let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            let use_jjn = pixel_hash & 1 == 1;

            if use_jjn {
                apply_jjn_kernel_ltr(&mut buf, x, y, err, width);
            } else {
                apply_fs_kernel_ltr(&mut buf, x, y, err, width);
            }
        }
    }

    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8))
        .collect()
}

/// Mixed dithering with serpentine scanning
/// Randomly selects between Floyd-Steinberg and Jarvis-Judice-Ninke kernels per pixel
/// Alternates scan direction each row
fn mixed_dither_serpentine(img: &[f32], width: usize, height: usize, seed: u32, quant: QuantParams) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);

    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        let is_reverse = y % 2 == 1;

        if is_reverse {
            for x in (0..width).rev() {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;

                if use_jjn {
                    apply_jjn_kernel_rtl(&mut buf, x, y, err, width);
                } else {
                    apply_fs_kernel_rtl(&mut buf, x, y, err, width);
                }
            }
        } else {
            for x in 0..width {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;

                if use_jjn {
                    apply_jjn_kernel_ltr(&mut buf, x, y, err, width);
                } else {
                    apply_fs_kernel_ltr(&mut buf, x, y, err, width);
                }
            }
        }
    }

    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8))
        .collect()
}

/// Mixed dithering with random scan direction
/// Randomly selects between Floyd-Steinberg and Jarvis-Judice-Ninke kernels per pixel
/// AND randomly selects scan direction per row (instead of alternating)
fn mixed_dither_random(img: &[f32], width: usize, height: usize, seed: u32, quant: QuantParams) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);

    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        // Randomly determine scan direction for this row
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_reverse = row_hash & 1 == 1;

        if is_reverse {
            for x in (0..width).rev() {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;

                if use_jjn {
                    apply_jjn_kernel_rtl(&mut buf, x, y, err, width);
                } else {
                    apply_fs_kernel_rtl(&mut buf, x, y, err, width);
                }
            }
        } else {
            for x in 0..width {
                let old = buf[y][x];
                let new = quant.quantize(old);
                buf[y][x] = new;
                let err = old - new;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;

                if use_jjn {
                    apply_jjn_kernel_ltr(&mut buf, x, y, err, width);
                } else {
                    apply_fs_kernel_ltr(&mut buf, x, y, err, width);
                }
            }
        }
    }

    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8))
        .collect()
}

/// Dither multiple channels and interleave them
/// channels: Vec of channel data, each scaled to 0-255
/// Returns interleaved u8 data
#[allow(dead_code)]
pub fn dither_channel_stack(channels: &[Vec<f32>], width: usize, height: usize) -> Vec<u8> {
    let num_channels = channels.len();
    let pixels = width * height;
    let mut result = vec![0u8; pixels * num_channels];

    // Dither each channel
    let dithered: Vec<Vec<u8>> = channels
        .iter()
        .map(|ch| floyd_steinberg_dither(ch, width, height))
        .collect();

    // Interleave
    for i in 0..pixels {
        for (ch, dithered_ch) in dithered.iter().enumerate() {
            result[i * num_channels + ch] = dithered_ch[i];
        }
    }

    result
}

/// Dither RGB float image (0-1 range) to uint8
#[allow(dead_code)]
pub fn dither_rgb(rgb: &[f32], width: usize, height: usize) -> Vec<u8> {
    let pixels = width * height;

    // Extract and scale channels
    let channels: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            (0..pixels)
                .map(|i| (rgb[i * 3 + ch] * 255.0).clamp(0.0, 255.0))
                .collect()
        })
        .collect();

    dither_channel_stack(&channels, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dither_exact_values() {
        // Values that are already integers should stay the same
        let img = vec![0.0, 128.0, 255.0, 64.0];
        let result = floyd_steinberg_dither(&img, 2, 2);
        // Note: due to error diffusion, exact matches aren't guaranteed
        // but for a simple case like this, corners should be close
        assert!(result[0] == 0 || result[0] == 1);
        assert!(result[2] == 254 || result[2] == 255);
    }

    #[test]
    fn test_dither_clamping() {
        // Values outside 0-255 should be clamped
        let img = vec![-10.0, 300.0];
        let result = floyd_steinberg_dither(&img, 2, 1);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 255);
    }

    #[test]
    fn test_serpentine_dither() {
        // Test that serpentine mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = dither_with_mode(&img, 4, 4, DitherMode::Serpentine, 0);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v <= 255);
        }
    }

    #[test]
    fn test_serpentine_vs_standard() {
        // Serpentine and standard should produce different results on larger images
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let standard = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::Serpentine, 0);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        // They should produce different results (not identical)
        assert_ne!(standard, serpentine);
    }

    #[test]
    fn test_jarvis_standard() {
        // Test that JJN standard mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = dither_with_mode(&img, 4, 4, DitherMode::JarvisStandard, 0);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_jarvis_serpentine() {
        // Test that JJN serpentine mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = dither_with_mode(&img, 4, 4, DitherMode::JarvisSerpentine, 0);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_jarvis_vs_floyd_steinberg() {
        // JJN and Floyd-Steinberg should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let fs_standard = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let jjn_standard = dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard, 0);
        assert_eq!(fs_standard.len(), 100);
        assert_eq!(jjn_standard.len(), 100);
        // They should produce different results
        assert_ne!(fs_standard, jjn_standard);
    }

    #[test]
    fn test_jarvis_serpentine_vs_standard() {
        // JJN serpentine and standard should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let standard = dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard, 0);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::JarvisSerpentine, 0);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        // They should produce different results
        assert_ne!(standard, serpentine);
    }

    #[test]
    fn test_mixed_standard() {
        // Test that mixed standard mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        assert_eq!(result.len(), 100);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_mixed_serpentine() {
        // Test that mixed serpentine mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::MixedSerpentine, 42);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_mixed_random() {
        // Test that mixed random mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::MixedRandom, 42);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_mixed_vs_pure_algorithms() {
        // Mixed modes should produce different results from pure FS and JJN
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let fs = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let jjn = dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard, 0);
        let mixed = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        assert_eq!(fs.len(), 100);
        assert_eq!(jjn.len(), 100);
        assert_eq!(mixed.len(), 100);
        // Mixed should differ from both pure algorithms
        assert_ne!(fs, mixed);
        assert_ne!(jjn, mixed);
    }

    #[test]
    fn test_mixed_deterministic_with_same_seed() {
        // Same seed should produce identical results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result1 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        let result2 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        assert_eq!(result1.len(), 100);
        assert_eq!(result2.len(), 100);
        // Same seed should produce identical results
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_mixed_different_seeds_produce_different_results() {
        // Different seeds should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result1 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        let result2 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 99);
        assert_eq!(result1.len(), 100);
        assert_eq!(result2.len(), 100);
        // Different seeds should produce different results
        assert_ne!(result1, result2);
    }

    #[test]
    fn test_mixed_all_modes_produce_different_results() {
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let standard = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::MixedSerpentine, 42);
        let random = dither_with_mode(&img, 10, 10, DitherMode::MixedRandom, 42);
        // All three mixed modes should produce different results
        assert_ne!(standard, serpentine);
        assert_ne!(standard, random);
        assert_ne!(serpentine, random);
    }

    // Bit depth tests

    #[test]
    fn test_quant_params_8bit() {
        // 8-bit should behave like normal rounding
        let quant = QuantParams::default_8bit();
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(127.5), 128.0);
        assert_eq!(quant.quantize(255.0), 255.0);
        assert_eq!(quant.quantize(100.4), 100.0);
        assert_eq!(quant.quantize(100.6), 101.0);
    }

    #[test]
    fn test_quant_params_1bit() {
        // 1-bit = 2 levels: 0 and 255
        let quant = QuantParams::new(1);
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(127.0), 0.0);   // < 127.5 rounds to 0
        assert_eq!(quant.quantize(128.0), 255.0); // > 127.5 rounds to 255
        assert_eq!(quant.quantize(255.0), 255.0);
    }

    #[test]
    fn test_quant_params_2bit() {
        // 2-bit = 4 levels: 0, 85, 170, 255
        let quant = QuantParams::new(2);
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(42.0), 0.0);    // rounds to 0
        assert_eq!(quant.quantize(43.0), 85.0);   // rounds to 85
        assert_eq!(quant.quantize(85.0), 85.0);
        assert_eq!(quant.quantize(127.0), 85.0);  // rounds to 85
        assert_eq!(quant.quantize(128.0), 170.0); // rounds to 170
        assert_eq!(quant.quantize(170.0), 170.0);
        assert_eq!(quant.quantize(212.0), 170.0); // rounds to 170
        assert_eq!(quant.quantize(213.0), 255.0); // rounds to 255
        assert_eq!(quant.quantize(255.0), 255.0);
    }

    #[test]
    fn test_quant_params_3bit() {
        // 3-bit = 8 levels with bit-replicated values:
        // 000→00000000=0, 001→00100100=36, 010→01001001=73, 011→01101101=109
        // 100→10010010=146, 101→10110110=182, 110→11011011=219, 111→11111111=255
        let quant = QuantParams::new(3);
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(255.0), 255.0);
        // Check that 127.5 quantizes to nearest level (level 3 or 4)
        // 127.5 * 7 / 255 = 3.5, rounds to level 4
        // Level 4 (100) → 10010010 = 146
        assert_eq!(quant.quantize(127.5), 146.0);
    }

    #[test]
    fn test_quantization_uses_bit_replication() {
        // Verify that quantization produces correct bit-replicated values
        // For 5 bits, value 3: should be 00011 → 00011000 = 24 (not 25 from linear scaling)
        let quant_5bit = QuantParams::new(5);
        // Input value that rounds to level 3: level = round(v * 31 / 255)
        // For level 3: v ≈ 3 * 255 / 31 ≈ 24.68, so input around 24-25 should give level 3
        let quantized = quant_5bit.quantize(24.68);
        assert_eq!(quantized, 24.0, "5-bit level 3 should be 24 (bit replication), not 25");

        // Verify some known bit replication values
        // 3-bit: 101 (5) → 10110110 = 182
        let quant_3bit = QuantParams::new(3);
        let level_5_input = 5.0 * 255.0 / 7.0; // ≈ 182.14
        assert_eq!(quant_3bit.quantize(level_5_input), 182.0);

        // 2-bit: all values should match (2 divides 8 evenly)
        let quant_2bit = QuantParams::new(2);
        assert_eq!(quant_2bit.quantize(0.0), 0.0);     // 00 → 00000000
        assert_eq!(quant_2bit.quantize(85.0), 85.0);   // 01 → 01010101
        assert_eq!(quant_2bit.quantize(170.0), 170.0); // 10 → 10101010
        assert_eq!(quant_2bit.quantize(255.0), 255.0); // 11 → 11111111

        // 4-bit: all values should match (4 divides 8 evenly)
        let quant_4bit = QuantParams::new(4);
        assert_eq!(quant_4bit.quantize(0.0), 0.0);     // 0000 → 00000000
        assert_eq!(quant_4bit.quantize(17.0), 17.0);   // 0001 → 00010001
        assert_eq!(quant_4bit.quantize(255.0), 255.0); // 1111 → 11111111
    }

    #[test]
    fn test_bit_replicate_correctness() {
        // Verify the bit_replicate function produces correct values
        // 1-bit
        assert_eq!(QuantParams::bit_replicate(0, 1), 0b00000000);
        assert_eq!(QuantParams::bit_replicate(1, 1), 0b11111111);

        // 2-bit
        assert_eq!(QuantParams::bit_replicate(0, 2), 0b00000000);
        assert_eq!(QuantParams::bit_replicate(1, 2), 0b01010101); // 85
        assert_eq!(QuantParams::bit_replicate(2, 2), 0b10101010); // 170
        assert_eq!(QuantParams::bit_replicate(3, 2), 0b11111111); // 255

        // 3-bit: ABC → ABCABCAB
        assert_eq!(QuantParams::bit_replicate(0b000, 3), 0b00000000); // 0
        assert_eq!(QuantParams::bit_replicate(0b001, 3), 0b00100100); // 36
        assert_eq!(QuantParams::bit_replicate(0b101, 3), 0b10110110); // 182
        assert_eq!(QuantParams::bit_replicate(0b111, 3), 0b11111111); // 255

        // 4-bit: ABCD → ABCDABCD
        assert_eq!(QuantParams::bit_replicate(0b0001, 4), 0b00010001); // 17
        assert_eq!(QuantParams::bit_replicate(0b1010, 4), 0b10101010); // 170

        // 5-bit: ABCDE → ABCDEABC
        assert_eq!(QuantParams::bit_replicate(0b00011, 5), 0b00011000); // 24
        assert_eq!(QuantParams::bit_replicate(0b11111, 5), 0b11111111); // 255
    }

    #[test]
    fn test_dither_1bit_produces_only_0_and_255() {
        // With 1-bit depth, output should only contain 0 and 255
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = floyd_steinberg_dither_bits(&img, 10, 10, 1);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 0 || v == 255, "1-bit dither should only produce 0 or 255, got {}", v);
        }
    }

    #[test]
    fn test_dither_2bit_produces_valid_levels() {
        // With 2-bit depth, output should only contain 0, 85, 170, 255
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = floyd_steinberg_dither_bits(&img, 10, 10, 2);
        assert_eq!(result.len(), 100);
        let valid_levels = [0u8, 85, 170, 255];
        for &v in &result {
            assert!(valid_levels.contains(&v), "2-bit dither should only produce 0, 85, 170, or 255, got {}", v);
        }
    }

    #[test]
    fn test_dither_3bit_produces_valid_levels() {
        // With 3-bit depth, output should only contain 8 specific levels (bit-replicated)
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = floyd_steinberg_dither_bits(&img, 10, 10, 3);
        assert_eq!(result.len(), 100);
        // 3-bit = 8 levels with bit replication: 0, 36, 73, 109, 146, 182, 219, 255
        let valid_levels: Vec<u8> = (0..8).map(|i| QuantParams::bit_replicate(i, 3)).collect();
        for &v in &result {
            assert!(valid_levels.contains(&v), "3-bit dither should only produce valid 3-bit levels, got {}", v);
        }
    }

    #[test]
    fn test_dither_different_bit_depths_produce_different_results() {
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result_8bit = floyd_steinberg_dither_bits(&img, 10, 10, 8);
        let result_4bit = floyd_steinberg_dither_bits(&img, 10, 10, 4);
        let result_2bit = floyd_steinberg_dither_bits(&img, 10, 10, 2);
        let result_1bit = floyd_steinberg_dither_bits(&img, 10, 10, 1);
        // Different bit depths should produce different results
        assert_ne!(result_8bit, result_4bit);
        assert_ne!(result_4bit, result_2bit);
        assert_ne!(result_2bit, result_1bit);
    }

    #[test]
    fn test_dither_8bit_matches_default() {
        // 8-bit should produce the same result as the default function
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result_default = floyd_steinberg_dither(&img, 10, 10);
        let result_8bit = floyd_steinberg_dither_bits(&img, 10, 10, 8);
        assert_eq!(result_default, result_8bit);
    }

    #[test]
    fn test_dither_with_mode_bits_parameter() {
        // Test that dither_with_mode_bits correctly uses the bits parameter
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result_8bit = dither_with_mode_bits(&img, 10, 10, DitherMode::Standard, 0, 8);
        let result_2bit = dither_with_mode_bits(&img, 10, 10, DitherMode::Standard, 0, 2);
        assert_ne!(result_8bit, result_2bit);
        // 2-bit should only contain valid levels
        let valid_levels = [0u8, 85, 170, 255];
        for &v in &result_2bit {
            assert!(valid_levels.contains(&v));
        }
    }

    #[test]
    fn test_dither_bits_all_modes() {
        // Test that bit depth works with all dithering modes
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let modes = [
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
        ];
        let valid_levels = [0u8, 85, 170, 255]; // 2-bit levels
        for mode in modes {
            let result = dither_with_mode_bits(&img, 10, 10, mode, 42, 2);
            assert_eq!(result.len(), 100);
            for &v in &result {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid 2-bit value: {}", mode, v);
            }
        }
    }
}
