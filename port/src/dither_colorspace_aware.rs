/// Color space aware dithering implementation.
///
/// Uses perceptual color space (CIELAB or OKLab) for candidate selection
/// with error diffusion in linear RGB space for physically correct light mixing.
/// Processes all RGB channels jointly rather than independently.
///
/// Supports multiple dithering algorithms:
/// - Floyd-Steinberg: Standard 2-row kernel (7/16, 3/16, 5/16, 1/16)
/// - Jarvis-Judice-Ninke: Larger 3-row kernel for smoother gradients
/// - Mixed: Random kernel selection per-pixel for reduced pattern visibility
///
/// And multiple scanning modes:
/// - Standard: Left-to-right for all rows
/// - Serpentine: Alternating direction each row
/// - Random: Random direction per row (mixed modes only)

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_to_srgb_single, srgb_to_linear_single,
};

/// Perceptual color space for distance calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerceptualSpace {
    /// CIELAB color space (L*a*b*)
    #[default]
    Lab,
    /// OKLab color space
    OkLab,
}

/// Dithering mode selection for color-space aware dithering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitherMode {
    /// Floyd-Steinberg: Standard left-to-right scanning (default)
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

/// Wang hash for deterministic randomization - excellent avalanche properties.
/// Each bit of input affects all bits of output.
#[inline]
pub fn wang_hash(mut x: u32) -> u32 {
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_mul(9);
    x = x ^ (x >> 4);
    x = x.wrapping_mul(0x27d4eb2d);
    x = x ^ (x >> 15);
    x
}

/// Extend n-bit value to 8 bits by repeating the bit pattern.
/// e.g., 3-bit value ABC becomes ABCABCAB
#[inline]
pub fn bit_replicate(value: u8, bits: u8) -> u8 {
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

/// Perceptual quantization parameters for joint RGB dithering.
/// Uses perceptual distance (Lab/OkLab) for candidate selection,
/// linear RGB for error diffusion.
struct PerceptualQuantParams {
    /// Number of quantization levels (2^bits)
    num_levels: usize,
    /// Level index → extended sRGB value (0-255)
    level_values: Vec<u8>,
    /// sRGB value → floor level index
    lut_floor_level: [u8; 256],
    /// sRGB value → ceil level index
    lut_ceil_level: [u8; 256],
}

impl PerceptualQuantParams {
    /// Create perceptual quantization parameters for given bit depth.
    fn new(bits: u8) -> Self {
        debug_assert!(bits >= 1 && bits <= 8, "bits must be 1-8");
        let num_levels = 1usize << bits;
        let max_idx = num_levels - 1;
        let shift = 8 - bits;

        // Pre-compute bit-replicated values for each level
        let level_values: Vec<u8> = (0..num_levels)
            .map(|l| bit_replicate(l as u8, bits))
            .collect();

        let mut lut_floor_level = [0u8; 256];
        let mut lut_ceil_level = [0u8; 256];

        for v in 0..256u16 {
            // Use bit truncation to find a nearby level
            let trunc_idx = (v as u8 >> shift) as usize;
            let trunc_val = level_values[trunc_idx];

            let (floor_idx, ceil_idx) = if trunc_val == v as u8 {
                (trunc_idx, trunc_idx)
            } else if trunc_val < v as u8 {
                // trunc is floor, ceil is trunc+1
                let ceil = if trunc_idx < max_idx {
                    trunc_idx + 1
                } else {
                    trunc_idx
                };
                (trunc_idx, ceil)
            } else {
                // trunc is ceil, floor is trunc-1
                let floor = if trunc_idx > 0 { trunc_idx - 1 } else { trunc_idx };
                (floor, trunc_idx)
            };

            lut_floor_level[v as usize] = floor_idx as u8;
            lut_ceil_level[v as usize] = ceil_idx as u8;
        }

        Self {
            num_levels,
            level_values,
            lut_floor_level,
            lut_ceil_level,
        }
    }

    /// Get floor level index for a sRGB value (0-255)
    #[inline]
    fn floor_level(&self, srgb_value: u8) -> usize {
        self.lut_floor_level[srgb_value as usize] as usize
    }

    /// Get ceil level index for a sRGB value (0-255)
    #[inline]
    fn ceil_level(&self, srgb_value: u8) -> usize {
        self.lut_ceil_level[srgb_value as usize] as usize
    }

    /// Get extended sRGB value (0-255) for a level index
    #[inline]
    fn level_to_srgb(&self, level: usize) -> u8 {
        self.level_values[level]
    }
}

/// Pre-computed LUT for sRGB to linear conversion (256 entries)
fn build_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = srgb_to_linear_single(i as f32 / 255.0);
    }
    lut
}

/// Lab color value for LUT storage
#[derive(Clone, Copy, Default)]
struct LabValue {
    l: f32,
    a: f32,
    b: f32,
}

/// Build Lab/OkLab LUT indexed by (r_level, g_level, b_level)
/// Returns a flat Vec where index = r_level * num_levels^2 + g_level * num_levels + b_level
fn build_perceptual_lut(
    quant: &PerceptualQuantParams,
    linear_lut: &[f32; 256],
    space: PerceptualSpace,
) -> Vec<LabValue> {
    let n = quant.num_levels;
    let mut lut = vec![LabValue::default(); n * n * n];

    for r_level in 0..n {
        let r_ext = quant.level_values[r_level];
        let r_lin = linear_lut[r_ext as usize];

        for g_level in 0..n {
            let g_ext = quant.level_values[g_level];
            let g_lin = linear_lut[g_ext as usize];

            for b_level in 0..n {
                let b_ext = quant.level_values[b_level];
                let b_lin = linear_lut[b_ext as usize];

                let (l, a, b_ch) = match space {
                    PerceptualSpace::Lab => linear_rgb_to_lab(r_lin, g_lin, b_lin),
                    PerceptualSpace::OkLab => linear_rgb_to_oklab(r_lin, g_lin, b_lin),
                };

                let idx = r_level * n * n + g_level * n + b_level;
                lut[idx] = LabValue { l, a, b: b_ch };
            }
        }
    }

    lut
}

// ============================================================================
// Trait-based kernel abstraction for RGB error diffusion
// ============================================================================

/// Trait for RGB error diffusion dithering kernels.
/// Implementations define the kernel shape (padding) and error distribution pattern.
/// Unlike single-channel kernels, these operate on all three RGB error buffers jointly.
trait RgbDitherKernel {
    /// Padding required on the left side of the buffer
    const PAD_LEFT: usize;
    /// Padding required on the right side of the buffer
    const PAD_RIGHT: usize;
    /// Padding required on the bottom of the buffer
    const PAD_BOTTOM: usize;

    /// Apply the kernel for left-to-right scanning.
    /// Distributes quantization error to neighboring pixels in all three channels.
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    );

    /// Apply the kernel for right-to-left scanning (mirrored).
    /// Used for serpentine scanning on odd rows.
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    );

    /// Apply the kernel to a single channel (left-to-right).
    /// Used for mixed modes where each channel may use a different kernel.
    fn apply_single_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32);

    /// Apply the kernel to a single channel (right-to-left).
    /// Used for mixed modes where each channel may use a different kernel.
    fn apply_single_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32);
}

/// Floyd-Steinberg error diffusion kernel for RGB.
/// Compact 2-row kernel with good speed/quality trade-off.
///
/// Kernel (divided by 16):
///       * 7
///     3 5 1
struct FloydSteinberg;

impl RgbDitherKernel for FloydSteinberg {
    const PAD_LEFT: usize = 1;
    const PAD_RIGHT: usize = 1;
    const PAD_BOTTOM: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        // Right: 7/16
        err_r[y][bx + 1] += err_r_val * (7.0 / 16.0);
        err_g[y][bx + 1] += err_g_val * (7.0 / 16.0);
        err_b[y][bx + 1] += err_b_val * (7.0 / 16.0);

        // Bottom-left: 3/16
        err_r[y + 1][bx - 1] += err_r_val * (3.0 / 16.0);
        err_g[y + 1][bx - 1] += err_g_val * (3.0 / 16.0);
        err_b[y + 1][bx - 1] += err_b_val * (3.0 / 16.0);

        // Bottom: 5/16
        err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
        err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
        err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);

        // Bottom-right: 1/16
        err_r[y + 1][bx + 1] += err_r_val * (1.0 / 16.0);
        err_g[y + 1][bx + 1] += err_g_val * (1.0 / 16.0);
        err_b[y + 1][bx + 1] += err_b_val * (1.0 / 16.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        // Left: 7/16
        err_r[y][bx - 1] += err_r_val * (7.0 / 16.0);
        err_g[y][bx - 1] += err_g_val * (7.0 / 16.0);
        err_b[y][bx - 1] += err_b_val * (7.0 / 16.0);

        // Bottom-right: 3/16
        err_r[y + 1][bx + 1] += err_r_val * (3.0 / 16.0);
        err_g[y + 1][bx + 1] += err_g_val * (3.0 / 16.0);
        err_b[y + 1][bx + 1] += err_b_val * (3.0 / 16.0);

        // Bottom: 5/16
        err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
        err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
        err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);

        // Bottom-left: 1/16
        err_r[y + 1][bx - 1] += err_r_val * (1.0 / 16.0);
        err_g[y + 1][bx - 1] += err_g_val * (1.0 / 16.0);
        err_b[y + 1][bx - 1] += err_b_val * (1.0 / 16.0);
    }

    #[inline]
    fn apply_single_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        err[y][bx + 1] += err_val * (7.0 / 16.0);
        err[y + 1][bx - 1] += err_val * (3.0 / 16.0);
        err[y + 1][bx] += err_val * (5.0 / 16.0);
        err[y + 1][bx + 1] += err_val * (1.0 / 16.0);
    }

    #[inline]
    fn apply_single_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        err[y][bx - 1] += err_val * (7.0 / 16.0);
        err[y + 1][bx + 1] += err_val * (3.0 / 16.0);
        err[y + 1][bx] += err_val * (5.0 / 16.0);
        err[y + 1][bx - 1] += err_val * (1.0 / 16.0);
    }
}

/// Jarvis-Judice-Ninke error diffusion kernel for RGB.
/// Larger 3-row kernel produces smoother gradients than Floyd-Steinberg.
///
/// Kernel (divided by 48):
///         * 7 5
///     3 5 7 5 3
///     1 3 5 3 1
struct JarvisJudiceNinke;

impl RgbDitherKernel for JarvisJudiceNinke {
    const PAD_LEFT: usize = 2;
    const PAD_RIGHT: usize = 2;
    const PAD_BOTTOM: usize = 2;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        // Row 0
        err_r[y][bx + 1] += err_r_val * (7.0 / 48.0);
        err_g[y][bx + 1] += err_g_val * (7.0 / 48.0);
        err_b[y][bx + 1] += err_b_val * (7.0 / 48.0);

        err_r[y][bx + 2] += err_r_val * (5.0 / 48.0);
        err_g[y][bx + 2] += err_g_val * (5.0 / 48.0);
        err_b[y][bx + 2] += err_b_val * (5.0 / 48.0);

        // Row 1
        err_r[y + 1][bx - 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx - 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx - 2] += err_b_val * (3.0 / 48.0);

        err_r[y + 1][bx - 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx - 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx - 1] += err_b_val * (5.0 / 48.0);

        err_r[y + 1][bx] += err_r_val * (7.0 / 48.0);
        err_g[y + 1][bx] += err_g_val * (7.0 / 48.0);
        err_b[y + 1][bx] += err_b_val * (7.0 / 48.0);

        err_r[y + 1][bx + 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx + 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx + 1] += err_b_val * (5.0 / 48.0);

        err_r[y + 1][bx + 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx + 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx + 2] += err_b_val * (3.0 / 48.0);

        // Row 2
        err_r[y + 2][bx - 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx - 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx - 2] += err_b_val * (1.0 / 48.0);

        err_r[y + 2][bx - 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx - 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx - 1] += err_b_val * (3.0 / 48.0);

        err_r[y + 2][bx] += err_r_val * (5.0 / 48.0);
        err_g[y + 2][bx] += err_g_val * (5.0 / 48.0);
        err_b[y + 2][bx] += err_b_val * (5.0 / 48.0);

        err_r[y + 2][bx + 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx + 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx + 1] += err_b_val * (3.0 / 48.0);

        err_r[y + 2][bx + 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx + 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx + 2] += err_b_val * (1.0 / 48.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        // Row 0
        err_r[y][bx - 1] += err_r_val * (7.0 / 48.0);
        err_g[y][bx - 1] += err_g_val * (7.0 / 48.0);
        err_b[y][bx - 1] += err_b_val * (7.0 / 48.0);

        err_r[y][bx - 2] += err_r_val * (5.0 / 48.0);
        err_g[y][bx - 2] += err_g_val * (5.0 / 48.0);
        err_b[y][bx - 2] += err_b_val * (5.0 / 48.0);

        // Row 1
        err_r[y + 1][bx + 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx + 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx + 2] += err_b_val * (3.0 / 48.0);

        err_r[y + 1][bx + 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx + 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx + 1] += err_b_val * (5.0 / 48.0);

        err_r[y + 1][bx] += err_r_val * (7.0 / 48.0);
        err_g[y + 1][bx] += err_g_val * (7.0 / 48.0);
        err_b[y + 1][bx] += err_b_val * (7.0 / 48.0);

        err_r[y + 1][bx - 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx - 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx - 1] += err_b_val * (5.0 / 48.0);

        err_r[y + 1][bx - 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx - 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx - 2] += err_b_val * (3.0 / 48.0);

        // Row 2
        err_r[y + 2][bx + 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx + 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx + 2] += err_b_val * (1.0 / 48.0);

        err_r[y + 2][bx + 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx + 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx + 1] += err_b_val * (3.0 / 48.0);

        err_r[y + 2][bx] += err_r_val * (5.0 / 48.0);
        err_g[y + 2][bx] += err_g_val * (5.0 / 48.0);
        err_b[y + 2][bx] += err_b_val * (5.0 / 48.0);

        err_r[y + 2][bx - 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx - 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx - 1] += err_b_val * (3.0 / 48.0);

        err_r[y + 2][bx - 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx - 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx - 2] += err_b_val * (1.0 / 48.0);
    }

    #[inline]
    fn apply_single_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        // Row 0
        err[y][bx + 1] += err_val * (7.0 / 48.0);
        err[y][bx + 2] += err_val * (5.0 / 48.0);
        // Row 1
        err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
        err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx] += err_val * (7.0 / 48.0);
        err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
        // Row 2
        err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
        err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx] += err_val * (5.0 / 48.0);
        err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
    }

    #[inline]
    fn apply_single_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        // Row 0
        err[y][bx - 1] += err_val * (7.0 / 48.0);
        err[y][bx - 2] += err_val * (5.0 / 48.0);
        // Row 1
        err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
        err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx] += err_val * (7.0 / 48.0);
        err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
        // Row 2
        err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
        err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx] += err_val * (5.0 / 48.0);
        err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
    }
}

/// Apply kernel for a single channel based on runtime selection.
#[inline]
fn apply_single_channel_kernel(
    err: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_val: f32,
    use_jjn: bool,
    is_rtl: bool,
) {
    match (use_jjn, is_rtl) {
        (true, false) => JarvisJudiceNinke::apply_single_ltr(err, bx, y, err_val),
        (true, true) => JarvisJudiceNinke::apply_single_rtl(err, bx, y, err_val),
        (false, false) => FloydSteinberg::apply_single_ltr(err, bx, y, err_val),
        (false, true) => FloydSteinberg::apply_single_rtl(err, bx, y, err_val),
    }
}

/// Apply kernel based on runtime selection (for mixed modes).
/// Each channel can use a different kernel, selected by per-channel flags.
/// Hash bits: bit 0 = R kernel, bit 1 = G kernel, bit 2 = B kernel
#[inline]
fn apply_mixed_kernel_per_channel(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    use_jjn_r: bool,
    use_jjn_g: bool,
    use_jjn_b: bool,
    is_rtl: bool,
) {
    apply_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
}

// ============================================================================
// Generic scan pattern implementations
// ============================================================================

/// Generic standard (left-to-right) dithering with any RGB kernel.
/// Processes all rows left-to-right.
#[inline]
fn dither_standard_rgb<K: RgbDitherKernel>(
    ctx: &DitherContext,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    width: usize,
    height: usize,
    pad_left: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

            r_out[idx] = best_r;
            g_out[idx] = best_g;
            b_out[idx] = best_b;

            K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val);
        }
    }
}

/// Generic serpentine dithering with any RGB kernel.
/// Alternates scan direction each row to reduce diagonal banding.
#[inline]
fn dither_serpentine_rgb<K: RgbDitherKernel>(
    ctx: &DitherContext,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    width: usize,
    height: usize,
    pad_left: usize,
) {
    for y in 0..height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for x in (0..width).rev() {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                K::apply_rtl(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val);
            }
        } else {
            // Left-to-right on even rows
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val);
            }
        }
    }
}

/// Mixed kernel dithering with standard (left-to-right) scanning.
/// Randomly selects between Floyd-Steinberg and Jarvis-Judice-Ninke per channel per pixel.
/// Each channel uses a separate bit from the hash for kernel selection.
#[inline]
fn dither_mixed_standard_rgb(
    ctx: &DitherContext,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    width: usize,
    height: usize,
    pad_left: usize,
    hashed_seed: u32,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

            r_out[idx] = best_r;
            g_out[idx] = best_g;
            b_out[idx] = best_b;

            // Extract 3 bits for per-channel kernel selection
            let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            let use_jjn_r = pixel_hash & 1 != 0;
            let use_jjn_g = pixel_hash & 2 != 0;
            let use_jjn_b = pixel_hash & 4 != 0;
            apply_mixed_kernel_per_channel(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, use_jjn_r, use_jjn_g, use_jjn_b, false);
        }
    }
}

/// Mixed kernel dithering with serpentine scanning.
/// Randomly selects kernel per channel per pixel, alternates direction per row.
#[inline]
fn dither_mixed_serpentine_rgb(
    ctx: &DitherContext,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    width: usize,
    height: usize,
    pad_left: usize,
    hashed_seed: u32,
) {
    for y in 0..height {
        if y % 2 == 1 {
            for x in (0..width).rev() {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn_r = pixel_hash & 1 != 0;
                let use_jjn_g = pixel_hash & 2 != 0;
                let use_jjn_b = pixel_hash & 4 != 0;
                apply_mixed_kernel_per_channel(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, use_jjn_r, use_jjn_g, use_jjn_b, true);
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn_r = pixel_hash & 1 != 0;
                let use_jjn_g = pixel_hash & 2 != 0;
                let use_jjn_b = pixel_hash & 4 != 0;
                apply_mixed_kernel_per_channel(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, use_jjn_r, use_jjn_g, use_jjn_b, false);
            }
        }
    }
}

/// Mixed kernel dithering with random direction per row.
/// Randomly selects kernel per channel per pixel and direction per row.
#[inline]
fn dither_mixed_random_rgb(
    ctx: &DitherContext,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    width: usize,
    height: usize,
    pad_left: usize,
    hashed_seed: u32,
) {
    for y in 0..height {
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_rtl = row_hash & 1 == 1;

        if is_rtl {
            for x in (0..width).rev() {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn_r = pixel_hash & 1 != 0;
                let use_jjn_g = pixel_hash & 2 != 0;
                let use_jjn_b = pixel_hash & 4 != 0;
                apply_mixed_kernel_per_channel(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, use_jjn_r, use_jjn_g, use_jjn_b, true);
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn_r = pixel_hash & 1 != 0;
                let use_jjn_g = pixel_hash & 2 != 0;
                let use_jjn_b = pixel_hash & 4 != 0;
                apply_mixed_kernel_per_channel(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, use_jjn_r, use_jjn_g, use_jjn_b, false);
            }
        }
    }
}

// ============================================================================
// Main dithering implementation
// ============================================================================

/// Context for pixel processing, containing pre-computed values
struct DitherContext<'a> {
    quant_r: &'a PerceptualQuantParams,
    quant_g: &'a PerceptualQuantParams,
    quant_b: &'a PerceptualQuantParams,
    linear_lut: &'a [f32; 256],
    lab_lut: &'a Option<Vec<LabValue>>,
    space: PerceptualSpace,
}

/// Process a single pixel: find best quantization and compute error.
/// Returns (best_r, best_g, best_b, err_r, err_g, err_b)
#[inline]
fn process_pixel(
    ctx: &DitherContext,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &[Vec<f32>],
    err_g: &[Vec<f32>],
    err_b: &[Vec<f32>],
    idx: usize,
    bx: usize,
    y: usize,
) -> (u8, u8, u8, f32, f32, f32) {
    // 1. Read input, convert to Linear RGB
    let srgb_r = r_channel[idx] / 255.0;
    let srgb_g = g_channel[idx] / 255.0;
    let srgb_b = b_channel[idx] / 255.0;

    let lin_r_orig = srgb_to_linear_single(srgb_r);
    let lin_g_orig = srgb_to_linear_single(srgb_g);
    let lin_b_orig = srgb_to_linear_single(srgb_b);

    // 2. Add accumulated error
    let lin_r_adj = lin_r_orig + err_r[y][bx];
    let lin_g_adj = lin_g_orig + err_g[y][bx];
    let lin_b_adj = lin_b_orig + err_b[y][bx];

    // 3. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
    let lin_r_clamped = lin_r_adj.clamp(0.0, 1.0);
    let lin_g_clamped = lin_g_adj.clamp(0.0, 1.0);
    let lin_b_clamped = lin_b_adj.clamp(0.0, 1.0);

    let srgb_r_adj = (linear_to_srgb_single(lin_r_clamped) * 255.0).clamp(0.0, 255.0);
    let srgb_g_adj = (linear_to_srgb_single(lin_g_clamped) * 255.0).clamp(0.0, 255.0);
    let srgb_b_adj = (linear_to_srgb_single(lin_b_clamped) * 255.0).clamp(0.0, 255.0);

    // 4. Get level index bounds
    let r_min = ctx.quant_r.floor_level(srgb_r_adj.floor() as u8);
    let r_max = ctx.quant_r.ceil_level((srgb_r_adj.ceil() as u8).min(255));

    let g_min = ctx.quant_g.floor_level(srgb_g_adj.floor() as u8);
    let g_max = ctx.quant_g.ceil_level((srgb_g_adj.ceil() as u8).min(255));

    let b_min = ctx.quant_b.floor_level(srgb_b_adj.floor() as u8);
    let b_max = ctx.quant_b.ceil_level((srgb_b_adj.ceil() as u8).min(255));

    // 5. Convert target to Lab (use unclamped for true distance)
    let lab_target = match ctx.space {
        PerceptualSpace::Lab => linear_rgb_to_lab(lin_r_adj, lin_g_adj, lin_b_adj),
        PerceptualSpace::OkLab => linear_rgb_to_oklab(lin_r_adj, lin_g_adj, lin_b_adj),
    };

    // 6. Search candidates
    let mut best_r_level = r_min;
    let mut best_g_level = g_min;
    let mut best_b_level = b_min;
    let mut best_dist = f32::INFINITY;

    for r_level in r_min..=r_max {
        for g_level in g_min..=g_max {
            for b_level in b_min..=b_max {
                let lab_candidate = if let Some(ref lut) = ctx.lab_lut {
                    // Same bit depths: use LUT
                    let n = ctx.quant_r.num_levels;
                    let lut_idx = r_level * n * n + g_level * n + b_level;
                    lut[lut_idx]
                } else {
                    // Different bit depths: compute on-the-fly
                    let r_ext = ctx.quant_r.level_to_srgb(r_level);
                    let g_ext = ctx.quant_g.level_to_srgb(g_level);
                    let b_ext = ctx.quant_b.level_to_srgb(b_level);

                    let r_lin = ctx.linear_lut[r_ext as usize];
                    let g_lin = ctx.linear_lut[g_ext as usize];
                    let b_lin = ctx.linear_lut[b_ext as usize];

                    let (l, a, b_ch) = match ctx.space {
                        PerceptualSpace::Lab => linear_rgb_to_lab(r_lin, g_lin, b_lin),
                        PerceptualSpace::OkLab => linear_rgb_to_oklab(r_lin, g_lin, b_lin),
                    };
                    LabValue { l, a, b: b_ch }
                };

                let dl = lab_target.0 - lab_candidate.l;
                let da = lab_target.1 - lab_candidate.a;
                let db = lab_target.2 - lab_candidate.b;
                let dist = dl * dl + da * da + db * db;

                if dist < best_dist {
                    best_dist = dist;
                    best_r_level = r_level;
                    best_g_level = g_level;
                    best_b_level = b_level;
                }
            }
        }
    }

    // 7. Get extended values for output and error calculation
    let best_r = ctx.quant_r.level_to_srgb(best_r_level);
    let best_g = ctx.quant_g.level_to_srgb(best_g_level);
    let best_b = ctx.quant_b.level_to_srgb(best_b_level);

    // 8. Compute error in Linear RGB
    let best_lin_r = ctx.linear_lut[best_r as usize];
    let best_lin_g = ctx.linear_lut[best_g as usize];
    let best_lin_b = ctx.linear_lut[best_b as usize];

    let err_r_val = lin_r_adj - best_lin_r;
    let err_g_val = lin_g_adj - best_lin_g;
    let err_b_val = lin_b_adj - best_lin_b;

    (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val)
}

/// Color space aware dithering with Floyd-Steinberg algorithm (default).
///
/// This is the original API that uses Floyd-Steinberg with standard (left-to-right) scanning.
/// For other algorithms and scan patterns, use `colorspace_aware_dither_rgb_with_mode`.
///
/// Args:
///     r_channel, g_channel, b_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Bit depth for each channel (1-8)
///     space: Perceptual color space for distance calculation
///
/// Returns:
///     (r_out, g_out, b_out): Output channels as u8
pub fn colorspace_aware_dither_rgb(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    space: PerceptualSpace,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    colorspace_aware_dither_rgb_with_mode(
        r_channel, g_channel, b_channel,
        width, height,
        bits_r, bits_g, bits_b,
        space,
        DitherMode::Standard,
        0, // seed not used for Standard mode
    )
}

/// Color space aware dithering with selectable algorithm and scanning mode.
///
/// Uses perceptual color space (Lab or OkLab) for finding the best quantization
/// candidate, and accumulates/diffuses error in linear RGB space for physically
/// correct light mixing. Processes all three channels jointly.
///
/// Args:
///     r_channel, g_channel, b_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Bit depth for each channel (1-8)
///     space: Perceptual color space for distance calculation
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///
/// Returns:
///     (r_out, g_out, b_out): Output channels as u8
pub fn colorspace_aware_dither_rgb_with_mode(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let quant_r = PerceptualQuantParams::new(bits_r);
    let quant_g = PerceptualQuantParams::new(bits_g);
    let quant_b = PerceptualQuantParams::new(bits_b);

    let linear_lut = build_linear_lut();

    // Build combined perceptual LUT if all channels have same bit depth
    let same_bits = bits_r == bits_g && bits_g == bits_b;
    let lab_lut = if same_bits {
        Some(build_perceptual_lut(&quant_r, &linear_lut, space))
    } else {
        None
    };

    let ctx = DitherContext {
        quant_r: &quant_r,
        quant_g: &quant_g,
        quant_b: &quant_b,
        linear_lut: &linear_lut,
        lab_lut: &lab_lut,
        space,
    };

    let pixels = width * height;

    // Use JJN padding for all modes to accommodate both kernels (JJN is larger)
    let pad_left = JarvisJudiceNinke::PAD_LEFT;
    let pad_right = JarvisJudiceNinke::PAD_RIGHT;
    let pad_bottom = JarvisJudiceNinke::PAD_BOTTOM;
    let buf_width = width + pad_left + pad_right;

    // Error buffers in linear RGB space
    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];

    // Output buffers
    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    // Dispatch to appropriate generic scan function
    match mode {
        DitherMode::Standard => {
            dither_standard_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_rgb::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_rgb::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, hashed_seed,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, hashed_seed,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, hashed_seed,
            );
        }
    }

    (r_out, g_out, b_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_dither_basic() {
        // Test that perceptual dithering produces valid output
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::Lab);

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_perceptual_dither_produces_valid_levels() {
        // With 2-bit depth, output should only contain 0, 85, 170, 255
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 2, 2, 2, PerceptualSpace::Lab);

        let valid_levels = [0u8, 85, 170, 255];
        for &v in &r_out {
            assert!(valid_levels.contains(&v), "R channel produced invalid 2-bit value: {}", v);
        }
        for &v in &g_out {
            assert!(valid_levels.contains(&v), "G channel produced invalid 2-bit value: {}", v);
        }
        for &v in &b_out {
            assert!(valid_levels.contains(&v), "B channel produced invalid 2-bit value: {}", v);
        }
    }

    #[test]
    fn test_perceptual_dither_lab_vs_oklab() {
        // Lab and OkLab should produce different results
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_lab, g_lab, b_lab) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::Lab);
        let (r_oklab, g_oklab, b_oklab) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::OkLab);

        // Results should differ (different perceptual spaces have different gamut mappings)
        let lab_combined: Vec<u8> = r_lab.iter().chain(g_lab.iter()).chain(b_lab.iter()).copied().collect();
        let oklab_combined: Vec<u8> = r_oklab.iter().chain(g_oklab.iter()).chain(b_oklab.iter()).copied().collect();
        assert_ne!(lab_combined, oklab_combined);
    }

    #[test]
    fn test_perceptual_dither_neutral_gray() {
        // Neutral gray should dither to nearby levels without color shift
        let gray_val = 128.0f32;
        let r: Vec<f32> = vec![gray_val; 100];
        let g: Vec<f32> = vec![gray_val; 100];
        let b: Vec<f32> = vec![gray_val; 100];

        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab);

        // For neutral gray input, output should remain relatively neutral
        // (R, G, B should be similar for each pixel)
        for i in 0..100 {
            let r_v = r_out[i] as i32;
            let g_v = g_out[i] as i32;
            let b_v = b_out[i] as i32;
            // Allow some dithering variation but channels should be close
            assert!((r_v - g_v).abs() <= 36, "Neutral gray has color shift: R={} G={}", r_v, g_v);
            assert!((g_v - b_v).abs() <= 36, "Neutral gray has color shift: G={} B={}", g_v, b_v);
        }
    }

    #[test]
    fn test_perceptual_dither_different_bit_depths() {
        // Test with different bit depths per channel (like RGB565)
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        // 5-bit R and B, 6-bit G (RGB565 format)
        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::Lab);

        // Check that outputs are valid for their respective bit depths
        let valid_5bit: Vec<u8> = (0..32).map(|l| bit_replicate(l, 5)).collect();
        let valid_6bit: Vec<u8> = (0..64).map(|l| bit_replicate(l, 6)).collect();

        for &v in &r_out {
            assert!(valid_5bit.contains(&v), "R channel produced invalid 5-bit value: {}", v);
        }
        for &v in &g_out {
            assert!(valid_6bit.contains(&v), "G channel produced invalid 6-bit value: {}", v);
        }
        for &v in &b_out {
            assert!(valid_5bit.contains(&v), "B channel produced invalid 5-bit value: {}", v);
        }
    }

    // Tests for different dithering modes

    #[test]
    fn test_all_modes_produce_valid_output() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

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
            let (r_out, g_out, b_out) = colorspace_aware_dither_rgb_with_mode(
                &r, &g, &b, 10, 10, 2, 2, 2, PerceptualSpace::Lab, mode, 42
            );

            assert_eq!(r_out.len(), 100, "Mode {:?} produced wrong length", mode);
            assert_eq!(g_out.len(), 100, "Mode {:?} produced wrong length", mode);
            assert_eq!(b_out.len(), 100, "Mode {:?} produced wrong length", mode);

            for &v in &r_out {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid 2-bit value: {}", mode, v);
            }
        }
    }

    #[test]
    fn test_serpentine_vs_standard() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_std, g_std, b_std) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::Standard, 0
        );
        let (r_serp, g_serp, b_serp) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::Serpentine, 0
        );

        // Results should differ
        let std_combined: Vec<u8> = r_std.iter().chain(g_std.iter()).chain(b_std.iter()).copied().collect();
        let serp_combined: Vec<u8> = r_serp.iter().chain(g_serp.iter()).chain(b_serp.iter()).copied().collect();
        assert_ne!(std_combined, serp_combined);
    }

    #[test]
    fn test_jarvis_vs_floyd_steinberg() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_fs, g_fs, b_fs) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::Standard, 0
        );
        let (r_jjn, g_jjn, b_jjn) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::JarvisStandard, 0
        );

        // Results should differ
        let fs_combined: Vec<u8> = r_fs.iter().chain(g_fs.iter()).chain(b_fs.iter()).copied().collect();
        let jjn_combined: Vec<u8> = r_jjn.iter().chain(g_jjn.iter()).chain(b_jjn.iter()).copied().collect();
        assert_ne!(fs_combined, jjn_combined);
    }

    #[test]
    fn test_mixed_mode_produces_different_result() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_fs, g_fs, b_fs) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::Standard, 0
        );
        let (r_jjn, g_jjn, b_jjn) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::JarvisStandard, 0
        );
        let (r_mix, g_mix, b_mix) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedStandard, 42
        );

        // Mixed should differ from both pure algorithms
        let fs_combined: Vec<u8> = r_fs.iter().chain(g_fs.iter()).chain(b_fs.iter()).copied().collect();
        let jjn_combined: Vec<u8> = r_jjn.iter().chain(g_jjn.iter()).chain(b_jjn.iter()).copied().collect();
        let mix_combined: Vec<u8> = r_mix.iter().chain(g_mix.iter()).chain(b_mix.iter()).copied().collect();

        assert_ne!(fs_combined, mix_combined);
        assert_ne!(jjn_combined, mix_combined);
    }

    #[test]
    fn test_mixed_deterministic_with_same_seed() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r1, g1, b1) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedStandard, 42
        );
        let (r2, g2, b2) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedStandard, 42
        );

        // Same seed should produce identical results
        assert_eq!(r1, r2);
        assert_eq!(g1, g2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_mixed_different_seeds_produce_different_results() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r1, g1, b1) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedStandard, 42
        );
        let (r2, g2, b2) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedStandard, 99
        );

        // Different seeds should produce different results
        let combined1: Vec<u8> = r1.iter().chain(g1.iter()).chain(b1.iter()).copied().collect();
        let combined2: Vec<u8> = r2.iter().chain(g2.iter()).chain(b2.iter()).copied().collect();
        assert_ne!(combined1, combined2);
    }

    #[test]
    fn test_all_mixed_modes_produce_different_results() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_std, g_std, b_std) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedStandard, 42
        );
        let (r_serp, g_serp, b_serp) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedSerpentine, 42
        );
        let (r_rand, g_rand, b_rand) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::MixedRandom, 42
        );

        let std_combined: Vec<u8> = r_std.iter().chain(g_std.iter()).chain(b_std.iter()).copied().collect();
        let serp_combined: Vec<u8> = r_serp.iter().chain(g_serp.iter()).chain(b_serp.iter()).copied().collect();
        let rand_combined: Vec<u8> = r_rand.iter().chain(g_rand.iter()).chain(b_rand.iter()).copied().collect();

        // All three mixed modes should produce different results
        assert_ne!(std_combined, serp_combined);
        assert_ne!(std_combined, rand_combined);
        assert_ne!(serp_combined, rand_combined);
    }

    #[test]
    fn test_default_mode_matches_original_api() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        // Original API
        let (r1, g1, b1) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab);
        // New API with Standard mode
        let (r2, g2, b2) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::Standard, 0
        );

        // Should produce identical results
        assert_eq!(r1, r2);
        assert_eq!(g1, g2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_jarvis_serpentine_vs_standard() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_std, g_std, b_std) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::JarvisStandard, 0
        );
        let (r_serp, g_serp, b_serp) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab, DitherMode::JarvisSerpentine, 0
        );

        // JJN standard and serpentine should produce different results
        let std_combined: Vec<u8> = r_std.iter().chain(g_std.iter()).chain(b_std.iter()).copied().collect();
        let serp_combined: Vec<u8> = r_serp.iter().chain(g_serp.iter()).chain(b_serp.iter()).copied().collect();
        assert_ne!(std_combined, serp_combined);
    }
}
