/// Color space aware RGBA dithering implementation.
///
/// Extends RGB dithering to handle alpha channel with proper error propagation:
/// - Alpha channel is dithered first using standard single-channel dithering
/// - RGB channels are then dithered with alpha-aware error diffusion:
///   - Error that couldn't be applied due to transparency is fully propagated
///   - Quantization error is weighted by pixel visibility (alpha)
///
/// This ensures that:
/// - Fully transparent pixels pass their accumulated error to neighbors
/// - Partially transparent pixels proportionally absorb/propagate error
/// - The visual result of compositing is properly optimized

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_rgb_to_ycbcr, linear_rgb_to_ycbcr_clamped,
    linear_to_srgb_single, srgb_to_linear_single,
};
use crate::color_distance::{
    is_lab_space, is_linear_rgb_space, is_srgb_space, is_ycbcr_space, perceptual_distance_sq,
};
use super::basic::dither_with_mode_bits;
use super::common::{bit_replicate, wang_hash, DitherMode, PerceptualSpace};

// ============================================================================
// Quantization and LUT structures (same as dither_rgb.rs)
// ============================================================================

/// Perceptual quantization parameters for joint RGB dithering.
struct PerceptualQuantParams {
    num_levels: usize,
    level_values: Vec<u8>,
    lut_floor_level: [u8; 256],
    lut_ceil_level: [u8; 256],
}

impl PerceptualQuantParams {
    fn new(bits: u8) -> Self {
        debug_assert!(bits >= 1 && bits <= 8, "bits must be 1-8");
        let num_levels = 1usize << bits;
        let max_idx = num_levels - 1;
        let shift = 8 - bits;

        let level_values: Vec<u8> = (0..num_levels)
            .map(|l| bit_replicate(l as u8, bits))
            .collect();

        let mut lut_floor_level = [0u8; 256];
        let mut lut_ceil_level = [0u8; 256];

        for v in 0..256u16 {
            let trunc_idx = (v as u8 >> shift) as usize;
            let trunc_val = level_values[trunc_idx];

            let (floor_idx, ceil_idx) = if trunc_val == v as u8 {
                (trunc_idx, trunc_idx)
            } else if trunc_val < v as u8 {
                let ceil = if trunc_idx < max_idx { trunc_idx + 1 } else { trunc_idx };
                (trunc_idx, ceil)
            } else {
                let floor = if trunc_idx > 0 { trunc_idx - 1 } else { trunc_idx };
                (floor, trunc_idx)
            };

            lut_floor_level[v as usize] = floor_idx as u8;
            lut_ceil_level[v as usize] = ceil_idx as u8;
        }

        Self { num_levels, level_values, lut_floor_level, lut_ceil_level }
    }

    #[inline]
    fn floor_level(&self, srgb_value: u8) -> usize {
        self.lut_floor_level[srgb_value as usize] as usize
    }

    #[inline]
    fn ceil_level(&self, srgb_value: u8) -> usize {
        self.lut_ceil_level[srgb_value as usize] as usize
    }

    #[inline]
    fn level_to_srgb(&self, level: usize) -> u8 {
        self.level_values[level]
    }
}

fn build_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = srgb_to_linear_single(i as f32 / 255.0);
    }
    lut
}

#[derive(Clone, Copy, Default)]
struct LabValue {
    l: f32,
    a: f32,
    b: f32,
}

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

                let (l, a, b_ch) = if is_srgb_space(space) {
                    // sRGB mode: use gamma-encoded values (normalized 0-1)
                    (r_ext as f32 / 255.0, g_ext as f32 / 255.0, b_ext as f32 / 255.0)
                } else if is_linear_rgb_space(space) {
                    (r_lin, g_lin, b_lin)
                } else if is_ycbcr_space(space) {
                    linear_rgb_to_ycbcr_clamped(r_lin, g_lin, b_lin)
                } else if is_lab_space(space) {
                    linear_rgb_to_lab(r_lin, g_lin, b_lin)
                } else {
                    linear_rgb_to_oklab(r_lin, g_lin, b_lin)
                };

                let idx = r_level * n * n + g_level * n + b_level;
                lut[idx] = LabValue { l, a, b: b_ch };
            }
        }
    }

    lut
}

// ============================================================================
// RGB Dither Kernel trait and implementations
// ============================================================================

trait RgbDitherKernel {
    const PAD_LEFT: usize;
    const PAD_RIGHT: usize;
    const PAD_BOTTOM: usize;

    fn apply_ltr(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>], err_b: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32,
    );

    fn apply_rtl(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>], err_b: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32,
    );

    fn apply_single_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32);
    fn apply_single_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32);
}

struct FloydSteinberg;

impl RgbDitherKernel for FloydSteinberg {
    const PAD_LEFT: usize = 1;
    const PAD_RIGHT: usize = 1;
    const PAD_BOTTOM: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>], err_b: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32,
    ) {
        err_r[y][bx + 1] += err_r_val * (7.0 / 16.0);
        err_g[y][bx + 1] += err_g_val * (7.0 / 16.0);
        err_b[y][bx + 1] += err_b_val * (7.0 / 16.0);

        err_r[y + 1][bx - 1] += err_r_val * (3.0 / 16.0);
        err_g[y + 1][bx - 1] += err_g_val * (3.0 / 16.0);
        err_b[y + 1][bx - 1] += err_b_val * (3.0 / 16.0);

        err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
        err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
        err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);

        err_r[y + 1][bx + 1] += err_r_val * (1.0 / 16.0);
        err_g[y + 1][bx + 1] += err_g_val * (1.0 / 16.0);
        err_b[y + 1][bx + 1] += err_b_val * (1.0 / 16.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>], err_b: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32,
    ) {
        err_r[y][bx - 1] += err_r_val * (7.0 / 16.0);
        err_g[y][bx - 1] += err_g_val * (7.0 / 16.0);
        err_b[y][bx - 1] += err_b_val * (7.0 / 16.0);

        err_r[y + 1][bx + 1] += err_r_val * (3.0 / 16.0);
        err_g[y + 1][bx + 1] += err_g_val * (3.0 / 16.0);
        err_b[y + 1][bx + 1] += err_b_val * (3.0 / 16.0);

        err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
        err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
        err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);

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

struct JarvisJudiceNinke;

impl RgbDitherKernel for JarvisJudiceNinke {
    const PAD_LEFT: usize = 2;
    const PAD_RIGHT: usize = 2;
    const PAD_BOTTOM: usize = 2;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>], err_b: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32,
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
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>], err_b: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32,
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
        err[y][bx + 1] += err_val * (7.0 / 48.0);
        err[y][bx + 2] += err_val * (5.0 / 48.0);
        err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
        err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx] += err_val * (7.0 / 48.0);
        err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
        err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
        err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx] += err_val * (5.0 / 48.0);
        err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
    }

    #[inline]
    fn apply_single_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        err[y][bx - 1] += err_val * (7.0 / 48.0);
        err[y][bx - 2] += err_val * (5.0 / 48.0);
        err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
        err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx] += err_val * (7.0 / 48.0);
        err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
        err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
        err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
        err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx] += err_val * (5.0 / 48.0);
        err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
        err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
    }
}

struct NoneKernel;

impl RgbDitherKernel for NoneKernel {
    const PAD_LEFT: usize = 0;
    const PAD_RIGHT: usize = 0;
    const PAD_BOTTOM: usize = 0;

    #[inline]
    fn apply_ltr(
        _err_r: &mut [Vec<f32>], _err_g: &mut [Vec<f32>], _err_b: &mut [Vec<f32>],
        _bx: usize, _y: usize,
        _err_r_val: f32, _err_g_val: f32, _err_b_val: f32,
    ) {}

    #[inline]
    fn apply_rtl(
        _err_r: &mut [Vec<f32>], _err_g: &mut [Vec<f32>], _err_b: &mut [Vec<f32>],
        _bx: usize, _y: usize,
        _err_r_val: f32, _err_g_val: f32, _err_b_val: f32,
    ) {}

    #[inline]
    fn apply_single_ltr(_err: &mut [Vec<f32>], _bx: usize, _y: usize, _err_val: f32) {}

    #[inline]
    fn apply_single_rtl(_err: &mut [Vec<f32>], _bx: usize, _y: usize, _err_val: f32) {}
}

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
// Alpha-aware dithering context and pixel processing
// ============================================================================

/// Context for alpha-aware pixel processing
struct DitherContextRgba<'a> {
    quant_r: &'a PerceptualQuantParams,
    quant_g: &'a PerceptualQuantParams,
    quant_b: &'a PerceptualQuantParams,
    linear_lut: &'a [f32; 256],
    lab_lut: &'a Option<Vec<LabValue>>,
    space: PerceptualSpace,
    /// Pre-dithered alpha channel (u8 values, 0-255)
    alpha_dithered: &'a [u8],
}

/// Process a single pixel with alpha-aware error diffusion.
///
/// The key difference from RGB-only dithering is how error is calculated:
/// - `e_in`: accumulated incoming error from previous pixels
/// - `α`: normalized alpha (0-1) of this pixel
/// - `q_err`: quantization error = adjusted - quantized
///
/// Error to diffuse = (1 - α) × e_in + α × q_err
///
/// This ensures:
/// - Fully transparent pixels (α=0) pass all incoming error to neighbors
/// - Fully opaque pixels (α=1) behave like standard dithering
/// - Partially transparent pixels proportionally absorb/propagate error
///
/// Returns (best_r, best_g, best_b, err_r, err_g, err_b)
#[inline]
fn process_pixel_rgba(
    ctx: &DitherContextRgba,
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
    // Get dithered alpha for this pixel (normalized to 0-1)
    let alpha = ctx.alpha_dithered[idx] as f32 / 255.0;

    // 1. Read accumulated error (e_in)
    let err_r_in = err_r[y][bx];
    let err_g_in = err_g[y][bx];
    let err_b_in = err_b[y][bx];

    // 2. Read input, convert to Linear RGB
    let srgb_r = r_channel[idx] / 255.0;
    let srgb_g = g_channel[idx] / 255.0;
    let srgb_b = b_channel[idx] / 255.0;

    let lin_r_orig = srgb_to_linear_single(srgb_r);
    let lin_g_orig = srgb_to_linear_single(srgb_g);
    let lin_b_orig = srgb_to_linear_single(srgb_b);

    // 3. Add accumulated error (skip for fully transparent pixels)
    // For α=0, we quantize the original value to produce cleaner RGB output
    // (useful if alpha is later stripped). Error diffusion still works since
    // the quantization error term is multiplied by α anyway.
    let (lin_r_adj, lin_g_adj, lin_b_adj) = if alpha == 0.0 {
        (lin_r_orig, lin_g_orig, lin_b_orig)
    } else {
        (lin_r_orig + err_r_in, lin_g_orig + err_g_in, lin_b_orig + err_b_in)
    };

    // 4. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
    let lin_r_clamped = lin_r_adj.clamp(0.0, 1.0);
    let lin_g_clamped = lin_g_adj.clamp(0.0, 1.0);
    let lin_b_clamped = lin_b_adj.clamp(0.0, 1.0);

    let srgb_r_adj = (linear_to_srgb_single(lin_r_clamped) * 255.0).clamp(0.0, 255.0);
    let srgb_g_adj = (linear_to_srgb_single(lin_g_clamped) * 255.0).clamp(0.0, 255.0);
    let srgb_b_adj = (linear_to_srgb_single(lin_b_clamped) * 255.0).clamp(0.0, 255.0);

    // 5. Get level index bounds
    let r_min = ctx.quant_r.floor_level(srgb_r_adj.floor() as u8);
    let r_max = ctx.quant_r.ceil_level((srgb_r_adj.ceil() as u8).min(255));

    let g_min = ctx.quant_g.floor_level(srgb_g_adj.floor() as u8);
    let g_max = ctx.quant_g.ceil_level((srgb_g_adj.ceil() as u8).min(255));

    let b_min = ctx.quant_b.floor_level(srgb_b_adj.floor() as u8);
    let b_max = ctx.quant_b.ceil_level((srgb_b_adj.ceil() as u8).min(255));

    // 6. Convert target to perceptual space (use unclamped for true distance)
    let lab_target = if is_srgb_space(ctx.space) {
        // sRGB mode: use gamma-encoded values directly (normalized 0-1)
        (srgb_r_adj / 255.0, srgb_g_adj / 255.0, srgb_b_adj / 255.0)
    } else if is_linear_rgb_space(ctx.space) {
        (lin_r_adj, lin_g_adj, lin_b_adj)
    } else if is_ycbcr_space(ctx.space) {
        linear_rgb_to_ycbcr(lin_r_adj, lin_g_adj, lin_b_adj)
    } else if is_lab_space(ctx.space) {
        linear_rgb_to_lab(lin_r_adj, lin_g_adj, lin_b_adj)
    } else {
        linear_rgb_to_oklab(lin_r_adj, lin_g_adj, lin_b_adj)
    };

    // 7. Search candidates for best quantization
    let mut best_r_level = r_min;
    let mut best_g_level = g_min;
    let mut best_b_level = b_min;
    let mut best_dist = f32::INFINITY;

    for r_level in r_min..=r_max {
        for g_level in g_min..=g_max {
            for b_level in b_min..=b_max {
                let lab_candidate = if let Some(lut) = ctx.lab_lut {
                    let n = ctx.quant_r.num_levels;
                    let lut_idx = r_level * n * n + g_level * n + b_level;
                    lut[lut_idx]
                } else {
                    let r_ext = ctx.quant_r.level_to_srgb(r_level);
                    let g_ext = ctx.quant_g.level_to_srgb(g_level);
                    let b_ext = ctx.quant_b.level_to_srgb(b_level);

                    let r_lin = ctx.linear_lut[r_ext as usize];
                    let g_lin = ctx.linear_lut[g_ext as usize];
                    let b_lin = ctx.linear_lut[b_ext as usize];

                    let (l, a, b_ch) = if is_srgb_space(ctx.space) {
                        // sRGB mode: use gamma-encoded values (normalized 0-1)
                        (r_ext as f32 / 255.0, g_ext as f32 / 255.0, b_ext as f32 / 255.0)
                    } else if is_linear_rgb_space(ctx.space) {
                        (r_lin, g_lin, b_lin)
                    } else if is_ycbcr_space(ctx.space) {
                        linear_rgb_to_ycbcr_clamped(r_lin, g_lin, b_lin)
                    } else if is_lab_space(ctx.space) {
                        linear_rgb_to_lab(r_lin, g_lin, b_lin)
                    } else {
                        linear_rgb_to_oklab(r_lin, g_lin, b_lin)
                    };
                    LabValue { l, a, b: b_ch }
                };

                let dist = perceptual_distance_sq(
                    ctx.space,
                    lab_target.0, lab_target.1, lab_target.2,
                    lab_candidate.l, lab_candidate.a, lab_candidate.b,
                );

                if dist < best_dist {
                    best_dist = dist;
                    best_r_level = r_level;
                    best_g_level = g_level;
                    best_b_level = b_level;
                }
            }
        }
    }

    // 8. Get extended values for output
    let best_r = ctx.quant_r.level_to_srgb(best_r_level);
    let best_g = ctx.quant_g.level_to_srgb(best_g_level);
    let best_b = ctx.quant_b.level_to_srgb(best_b_level);

    // 9. Compute quantized linear values
    let best_lin_r = ctx.linear_lut[best_r as usize];
    let best_lin_g = ctx.linear_lut[best_g as usize];
    let best_lin_b = ctx.linear_lut[best_b as usize];

    // 10. Compute alpha-aware error to diffuse
    // Formula: error = (1 - α) × e_in + α × q_err
    // Where q_err = linear_adj - linear_quant
    let q_err_r = lin_r_adj - best_lin_r;
    let q_err_g = lin_g_adj - best_lin_g;
    let q_err_b = lin_b_adj - best_lin_b;

    let one_minus_alpha = 1.0 - alpha;
    let err_r_val = one_minus_alpha * err_r_in + alpha * q_err_r;
    let err_g_val = one_minus_alpha * err_g_in + alpha * q_err_g;
    let err_b_val = one_minus_alpha * err_b_in + alpha * q_err_b;

    (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val)
}

// ============================================================================
// Generic scan pattern implementations for RGBA
// ============================================================================

#[inline]
fn dither_standard_rgba<K: RgbDitherKernel>(
    ctx: &DitherContextRgba,
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
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

            r_out[idx] = best_r;
            g_out[idx] = best_g;
            b_out[idx] = best_b;

            K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val);
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_serpentine_rgba<K: RgbDitherKernel>(
    ctx: &DitherContextRgba,
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
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    for y in 0..height {
        if y % 2 == 1 {
            for x in (0..width).rev() {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                K::apply_rtl(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val);
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;

                K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val);
            }
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_mixed_standard_rgba(
    ctx: &DitherContextRgba,
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
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

            r_out[idx] = best_r;
            g_out[idx] = best_g;
            b_out[idx] = best_b;

            let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            let use_jjn_r = pixel_hash & 1 != 0;
            let use_jjn_g = pixel_hash & 2 != 0;
            let use_jjn_b = pixel_hash & 4 != 0;
            apply_mixed_kernel_per_channel(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, use_jjn_r, use_jjn_g, use_jjn_b, false);
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_mixed_serpentine_rgba(
    ctx: &DitherContextRgba,
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
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    for y in 0..height {
        if y % 2 == 1 {
            for x in (0..width).rev() {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

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
                    process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

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
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_mixed_random_rgba(
    ctx: &DitherContextRgba,
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
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    for y in 0..height {
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_rtl = row_hash & 1 == 1;

        if is_rtl {
            for x in (0..width).rev() {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

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
                    process_pixel_rgba(ctx, r_channel, g_channel, b_channel, err_r, err_g, err_b, idx, bx, y);

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
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Color space aware RGBA dithering.
///
/// This is the simplified API that uses Floyd-Steinberg with standard scanning.
/// For other algorithms and scan patterns, use `colorspace_aware_dither_rgba_with_mode`.
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b, bits_a: Bit depth for each channel (1-8)
///     space: Perceptual color space for RGB distance calculation
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as u8
pub fn colorspace_aware_dither_rgba(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    bits_a: u8,
    space: PerceptualSpace,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    colorspace_aware_dither_rgba_with_mode(
        r_channel, g_channel, b_channel, a_channel,
        width, height,
        bits_r, bits_g, bits_b, bits_a,
        space,
        DitherMode::Standard,
        DitherMode::Standard,
        0,
        None,
    )
}

/// Color space aware RGBA dithering with selectable algorithm and scanning mode.
///
/// Process:
/// 1. Alpha channel is dithered first using the specified alpha dithering mode
/// 2. RGB channels are then dithered with alpha-aware error propagation:
///    - Error that couldn't be applied due to transparency is fully propagated
///    - Quantization error is weighted by pixel visibility (alpha)
///
/// The alpha-aware error formula is:
///     error_to_diffuse = (1 - α) × e_in + α × q_err
///
/// Where:
/// - α: normalized alpha (0-1) after dithering
/// - e_in: accumulated incoming error from previous pixels
/// - q_err: quantization error (adjusted - quantized)
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b, bits_a: Bit depth for each channel (1-8)
///     space: Perceptual color space for RGB distance calculation
///     mode: Dithering algorithm and scanning mode for RGB channels
///     alpha_mode: Dithering algorithm and scanning mode for alpha channel
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as u8
pub fn colorspace_aware_dither_rgba_with_mode(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    bits_a: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    alpha_mode: DitherMode,
    seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = width * height;

    // Step 1: Dither alpha channel first using the specified alpha mode
    // Alpha is linear, so standard dithering is correct
    let alpha_dithered = dither_with_mode_bits(a_channel, width, height, alpha_mode, seed.wrapping_add(3), bits_a, None);

    // Report alpha dithering complete (10% of total progress)
    if let Some(ref mut cb) = progress {
        cb(0.1);
    }

    // Step 2: Set up RGB dithering with alpha awareness
    let quant_r = PerceptualQuantParams::new(bits_r);
    let quant_g = PerceptualQuantParams::new(bits_g);
    let quant_b = PerceptualQuantParams::new(bits_b);

    let linear_lut = build_linear_lut();

    let same_bits = bits_r == bits_g && bits_g == bits_b;
    let lab_lut = if same_bits {
        Some(build_perceptual_lut(&quant_r, &linear_lut, space))
    } else {
        None
    };

    let ctx = DitherContextRgba {
        quant_r: &quant_r,
        quant_g: &quant_g,
        quant_b: &quant_b,
        linear_lut: &linear_lut,
        lab_lut: &lab_lut,
        space,
        alpha_dithered: &alpha_dithered,
    };

    // Use JJN padding for all modes
    let pad_left = JarvisJudiceNinke::PAD_LEFT;
    let pad_right = JarvisJudiceNinke::PAD_RIGHT;
    let pad_bottom = JarvisJudiceNinke::PAD_BOTTOM;
    let buf_width = width + pad_left + pad_right;

    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];

    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    match mode {
        DitherMode::None => {
            dither_standard_rgba::<NoneKernel>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_rgba::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_rgba::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_rgba(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_rgba(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_rgba(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, pad_left, hashed_seed, progress,
            );
        }
    }

    (r_out, g_out, b_out, alpha_dithered)
}

// ============================================================================
// Pixel4 convenience wrappers
// ============================================================================

use crate::color::interleave_rgba_u8;
use crate::pixel::{pixels_to_channels_rgba, Pixel4};

/// Color-aware dither for Pixel4 array (sRGB 0-255 range) to separate RGBA channels.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range (including alpha)
///     width, height: image dimensions
///     bits_r, bits_g, bits_b, bits_a: output bit depth per channel (1-8)
///     space: perceptual color space for RGB distance calculation
///     mode: dither algorithm and scan pattern for RGB channels
///     alpha_mode: dither algorithm and scan pattern for alpha channel
///     seed: random seed for mixed modes
///     progress: optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     Tuple of (R, G, B, A) u8 vectors
pub fn colorspace_aware_dither_rgba_channels(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    bits_a: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    alpha_mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (r, g, b, a) = pixels_to_channels_rgba(pixels);
    colorspace_aware_dither_rgba_with_mode(
        &r, &g, &b, &a,
        width, height,
        bits_r, bits_g, bits_b, bits_a,
        space, mode, alpha_mode, seed, progress,
    )
}

/// Color-aware dither for Pixel4 array to interleaved RGBA u8.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range (including alpha)
///     width, height: image dimensions
///     space: perceptual color space for RGB distance calculation
///     mode: dither algorithm and scan pattern for RGB channels
///     alpha_mode: dither algorithm and scan pattern for alpha channel
///     seed: random seed for mixed modes
///     progress: optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     Interleaved RGBA u8 data (RGBARGBA...)
pub fn colorspace_aware_dither_rgba_interleaved(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    mode: DitherMode,
    alpha_mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let (r, g, b, a) = pixels_to_channels_rgba(pixels);
    let (r_u8, g_u8, b_u8, a_u8) = colorspace_aware_dither_rgba_with_mode(
        &r, &g, &b, &a,
        width, height,
        8, 8, 8, 8,
        space, mode, alpha_mode, seed, progress,
    );
    interleave_rgba_u8(&r_u8, &g_u8, &b_u8, &a_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_dither_basic() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let (r_out, g_out, b_out, a_out) = colorspace_aware_dither_rgba(
            &r, &g, &b, &a, 10, 10, 5, 6, 5, 8, PerceptualSpace::OkLab
        );

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
        assert_eq!(a_out.len(), 100);
    }

    #[test]
    fn test_rgba_fully_transparent_passes_error() {
        // With fully transparent pixels, RGB values don't matter visually
        // but error should be passed through
        let r: Vec<f32> = vec![128.0; 100];
        let g: Vec<f32> = vec![128.0; 100];
        let b: Vec<f32> = vec![128.0; 100];
        let a: Vec<f32> = vec![0.0; 100]; // Fully transparent

        let (r_out, g_out, b_out, a_out) = colorspace_aware_dither_rgba(
            &r, &g, &b, &a, 10, 10, 5, 5, 5, 8, PerceptualSpace::OkLab
        );

        // Alpha should be 0
        for &v in &a_out {
            assert_eq!(v, 0, "Transparent alpha should dither to 0");
        }

        // RGB values will still be quantized (implementation detail)
        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_rgba_fully_opaque_matches_rgb() {
        // With fully opaque pixels, RGBA dithering should behave similarly to RGB
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();
        let a: Vec<f32> = vec![255.0; 100]; // Fully opaque

        let (r_out, g_out, b_out, a_out) = colorspace_aware_dither_rgba(
            &r, &g, &b, &a, 10, 10, 5, 5, 5, 8, PerceptualSpace::OkLab
        );

        // Alpha should be 255
        for &v in &a_out {
            assert_eq!(v, 255, "Opaque alpha should dither to 255");
        }

        // Verify valid output
        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_rgba_produces_valid_levels() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let (r_out, g_out, b_out, a_out) = colorspace_aware_dither_rgba(
            &r, &g, &b, &a, 10, 10, 2, 2, 2, 2, PerceptualSpace::OkLab
        );

        let valid_levels = [0u8, 85, 170, 255];
        for &v in &r_out {
            assert!(valid_levels.contains(&v), "R produced invalid 2-bit value: {}", v);
        }
        for &v in &g_out {
            assert!(valid_levels.contains(&v), "G produced invalid 2-bit value: {}", v);
        }
        for &v in &b_out {
            assert!(valid_levels.contains(&v), "B produced invalid 2-bit value: {}", v);
        }
        for &v in &a_out {
            assert!(valid_levels.contains(&v), "A produced invalid 2-bit value: {}", v);
        }
    }

    #[test]
    fn test_rgba_all_modes_produce_valid_output() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();
        let a: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 * 2.55).collect();

        let modes = [
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
        ];

        let valid_levels = [0u8, 85, 170, 255];

        for mode in modes {
            let (r_out, g_out, b_out, a_out) = colorspace_aware_dither_rgba_with_mode(
                &r, &g, &b, &a, 10, 10, 2, 2, 2, 2, PerceptualSpace::OkLab, mode, mode, 42, None
            );

            assert_eq!(r_out.len(), 100, "Mode {:?} produced wrong R length", mode);
            assert_eq!(a_out.len(), 100, "Mode {:?} produced wrong A length", mode);

            for &v in &r_out {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid 2-bit R value: {}", mode, v);
            }
            for &v in &a_out {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid 2-bit A value: {}", mode, v);
            }
        }
    }

    #[test]
    fn test_rgba_mixed_deterministic() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();
        let a: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 * 2.55).collect();

        let result1 = colorspace_aware_dither_rgba_with_mode(
            &r, &g, &b, &a, 10, 10, 5, 5, 5, 8, PerceptualSpace::OkLab, DitherMode::MixedStandard, DitherMode::MixedStandard, 42, None
        );
        let result2 = colorspace_aware_dither_rgba_with_mode(
            &r, &g, &b, &a, 10, 10, 5, 5, 5, 8, PerceptualSpace::OkLab, DitherMode::MixedStandard, DitherMode::MixedStandard, 42, None
        );

        assert_eq!(result1.0, result2.0);
        assert_eq!(result1.1, result2.1);
        assert_eq!(result1.2, result2.2);
        assert_eq!(result1.3, result2.3);
    }

    #[test]
    fn test_rgba_semi_transparent_behavior() {
        // Test that semi-transparent pixels have intermediate error behavior
        let r: Vec<f32> = vec![127.5; 100];
        let g: Vec<f32> = vec![127.5; 100];
        let b: Vec<f32> = vec![127.5; 100];
        let a: Vec<f32> = vec![127.5; 100]; // 50% transparent

        let (r_out, g_out, b_out, a_out) = colorspace_aware_dither_rgba(
            &r, &g, &b, &a, 10, 10, 5, 5, 5, 8, PerceptualSpace::OkLab
        );

        // Just verify we get valid output
        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
        assert_eq!(a_out.len(), 100);

        // Alpha should have some variation around 127-128
        let alpha_sum: u32 = a_out.iter().map(|&v| v as u32).sum();
        let alpha_avg = alpha_sum as f32 / 100.0;
        assert!((alpha_avg - 127.5).abs() < 5.0, "Alpha average should be near 127.5, got {}", alpha_avg);
    }
}
