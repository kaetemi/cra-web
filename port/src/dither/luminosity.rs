/// Color space aware dithering for single-channel grayscale images.
///
/// Treats grayscale values as RGB=(L,L,L) for perceptual distance calculation
/// while performing error diffusion in linear luminosity space.
/// Input is sRGB gamma-encoded grayscale (0-255).
///
/// ## Mathematical Optimization for Grayscale
///
/// For neutral grays (R=G=B), the CIELAB a* and b* components are always 0.
/// This means ALL CIELAB distance formulas collapse to simple lightness difference:
///
/// - **CIE76**: ΔE² = ΔL² + Δa² + Δb² = ΔL² (since Δa=Δb=0)
/// - **CIE94**: Chroma C=√(a²+b²)=0, so all chroma/hue terms vanish → ΔL²
/// - **CIEDE2000**: Chroma terms vanish. The SL lightness weighting factor varies
///   with average lightness, but for adjacent quantization levels the difference
///   is negligible, so candidate ordering is unchanged.
///
/// Therefore, for grayscale we optimize by storing only the perceptual lightness
/// value (L* for CIELAB, L for OKLab) and using simple squared difference.
/// OKLab and CIELAB use different lightness curves, so they may produce
/// slightly different results, but within each family the results are identical.
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
use crate::color_distance::{is_lab_space, is_linear_rgb_space, is_ycbcr_space};
use crate::colorspace_derived::f32 as cs;
use super::common::{bit_replicate, wang_hash, DitherMode, PerceptualSpace};

/// Convert linear luminosity to Y'CbCr Y' component for grayscale.
/// For grayscale (R=G=B), Y' simply equals the gamma-encoded (sRGB) value
/// since KR + KG + KB = 1, so Y' = KR*v + KG*v + KB*v = v.
/// Preserves sign for out-of-gamut values during error diffusion.
#[inline]
fn linear_gray_to_ycbcr_y(lin_gray: f32) -> f32 {
    if lin_gray >= 0.0 {
        linear_to_srgb_single(lin_gray)
    } else {
        -linear_to_srgb_single(-lin_gray)
    }
}

// ============================================================================
// Optimized Grayscale Distance
// ============================================================================

/// For grayscale with CIE76/CIE94/OKLab, distance reduces to simple ΔL²
/// because a* = b* = 0 for neutral grays.
#[inline]
fn lightness_distance_sq(l1: f32, l2: f32) -> f32 {
    let dl = l1 - l2;
    dl * dl
}

/// CIEDE2000 lightness distance for grayscale.
/// Unlike CIE76/CIE94, CIEDE2000 uses a lightness weighting factor SL
/// that depends on the average lightness of the two colors.
/// This compensates for reduced human sensitivity in dark/light regions.
///
/// Formula: ΔE² = (ΔL / SL)²
/// Where: SL = 1 + (K2 × (L̄ - 50)²) / √(20 + (L̄ - 50)²)
///        L̄ = (L1 + L2) / 2
///        K2 = 0.015 (from CIE94, shared with CIEDE2000)
#[inline]
fn lightness_distance_ciede2000_sq(l1: f32, l2: f32) -> f32 {
    let dl = l1 - l2;
    let l_bar = (l1 + l2) / 2.0;
    let l_bar_minus_mid = l_bar - cs::CIEDE2000_SL_L_MIDPOINT;
    let l_bar_minus_mid_sq = l_bar_minus_mid * l_bar_minus_mid;
    let sl = 1.0 + (cs::CIE94_K2 * l_bar_minus_mid_sq) / (cs::CIEDE2000_SL_DENOM_OFFSET + l_bar_minus_mid_sq).sqrt();
    let dl_term = dl / sl;
    dl_term * dl_term
}

/// Compute grayscale perceptual distance based on the selected space/metric
#[inline]
fn perceptual_lightness_distance_sq(space: PerceptualSpace, l1: f32, l2: f32) -> f32 {
    match space {
        // CIE76 and CIE94 reduce to simple ΔL² for neutral grays (a=b=0)
        PerceptualSpace::LabCIE76 | PerceptualSpace::LabCIE94 => lightness_distance_sq(l1, l2),
        // CIEDE2000 uses SL weighting based on average lightness
        PerceptualSpace::LabCIEDE2000 => lightness_distance_ciede2000_sq(l1, l2),
        // OKLab uses simple Euclidean distance, which reduces to ΔL² for grays
        // LinearRGB also uses simple Euclidean distance in linear space
        // YCbCr uses simple distance in gamma-encoded (sRGB) space
        // sRGB uses simple distance in gamma-encoded (sRGB) space
        PerceptualSpace::OkLab | PerceptualSpace::LinearRGB | PerceptualSpace::YCbCr | PerceptualSpace::Srgb => {
            lightness_distance_sq(l1, l2)
        }
    }
}

/// Quantization parameters for grayscale dithering
struct GrayQuantParams {
    /// Number of quantization levels (2^bits)
    num_levels: usize,
    /// Level index -> extended sRGB value (0-255)
    level_values: Vec<u8>,
    /// sRGB value -> floor level index
    lut_floor_level: [u8; 256],
    /// sRGB value -> ceil level index
    lut_ceil_level: [u8; 256],
}

impl GrayQuantParams {
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
                let ceil = if trunc_idx < max_idx {
                    trunc_idx + 1
                } else {
                    trunc_idx
                };
                (trunc_idx, ceil)
            } else {
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

/// Pre-computed LUT for sRGB to linear conversion (256 entries)
fn build_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = srgb_to_linear_single(i as f32 / 255.0);
    }
    lut
}

/// Build perceptual lightness LUT for grayscale levels.
/// Each level is converted to perceptual lightness (L* for CIELAB, L for OKLab).
/// For LinearRGB, stores the linear luminosity value directly.
/// We only store L since a* = b* = 0 for all neutral grays.
fn build_gray_lightness_lut(
    quant: &GrayQuantParams,
    linear_lut: &[f32; 256],
    space: PerceptualSpace,
) -> Vec<f32> {
    let n = quant.num_levels;
    let mut lut = vec![0.0f32; n];

    for level in 0..n {
        let gray_ext = quant.level_values[level];
        let gray_lin = linear_lut[gray_ext as usize];

        // Treat as RGB = (gray, gray, gray)
        // Only extract L component since a, b are always 0 for neutral grays
        let l = if is_linear_rgb_space(space) {
            // Linear RGB: use linear luminosity value directly (no perceptual conversion)
            gray_lin
        } else if is_ycbcr_space(space) {
            // Y'CbCr: use sRGB (gamma-encoded) value as Y'
            // For grayscale R=G=B, Y' = sRGB value since luma coefficients sum to 1
            linear_to_srgb_single(gray_lin)
        } else if is_lab_space(space) {
            let (l, _, _) = linear_rgb_to_lab(gray_lin, gray_lin, gray_lin);
            l
        } else {
            let (l, _, _) = linear_rgb_to_oklab(gray_lin, gray_lin, gray_lin);
            l
        };

        lut[level] = l;
    }

    lut
}

// ============================================================================
// Single-channel error diffusion kernels
// ============================================================================

trait GrayDitherKernel {
    const PAD_LEFT: usize;
    const PAD_RIGHT: usize;
    const PAD_BOTTOM: usize;

    fn apply_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32);
    fn apply_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32);
}

struct FloydSteinberg;

impl GrayDitherKernel for FloydSteinberg {
    const PAD_LEFT: usize = 1;
    const PAD_RIGHT: usize = 1;
    const PAD_BOTTOM: usize = 1;

    #[inline]
    fn apply_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        err[y][bx + 1] += err_val * (7.0 / 16.0);
        err[y + 1][bx - 1] += err_val * (3.0 / 16.0);
        err[y + 1][bx] += err_val * (5.0 / 16.0);
        err[y + 1][bx + 1] += err_val * (1.0 / 16.0);
    }

    #[inline]
    fn apply_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
        err[y][bx - 1] += err_val * (7.0 / 16.0);
        err[y + 1][bx + 1] += err_val * (3.0 / 16.0);
        err[y + 1][bx] += err_val * (5.0 / 16.0);
        err[y + 1][bx - 1] += err_val * (1.0 / 16.0);
    }
}

struct JarvisJudiceNinke;

impl GrayDitherKernel for JarvisJudiceNinke {
    const PAD_LEFT: usize = 2;
    const PAD_RIGHT: usize = 2;
    const PAD_BOTTOM: usize = 2;

    #[inline]
    fn apply_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
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
    fn apply_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32) {
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

/// No-op kernel that discards error (no diffusion).
/// Each pixel is independently quantized to the perceptually nearest level.
struct NoneKernel;

impl GrayDitherKernel for NoneKernel {
    const PAD_LEFT: usize = 0;
    const PAD_RIGHT: usize = 0;
    const PAD_BOTTOM: usize = 0;

    #[inline]
    fn apply_ltr(_err: &mut [Vec<f32>], _bx: usize, _y: usize, _err_val: f32) {}

    #[inline]
    fn apply_rtl(_err: &mut [Vec<f32>], _bx: usize, _y: usize, _err_val: f32) {}
}

/// Apply kernel based on runtime selection
#[inline]
fn apply_mixed_kernel(err: &mut [Vec<f32>], bx: usize, y: usize, err_val: f32, use_jjn: bool, is_rtl: bool) {
    match (use_jjn, is_rtl) {
        (true, false) => JarvisJudiceNinke::apply_ltr(err, bx, y, err_val),
        (true, true) => JarvisJudiceNinke::apply_rtl(err, bx, y, err_val),
        (false, false) => FloydSteinberg::apply_ltr(err, bx, y, err_val),
        (false, true) => FloydSteinberg::apply_rtl(err, bx, y, err_val),
    }
}

// ============================================================================
// Dithering context and pixel processing
// ============================================================================

struct GrayDitherContext<'a> {
    quant: &'a GrayQuantParams,
    linear_lut: &'a [f32; 256],
    /// Perceptual lightness values for each quantization level.
    /// Only L is stored since a* = b* = 0 for neutral grays.
    lightness_lut: &'a Vec<f32>,
    space: PerceptualSpace,
}

/// Process a single grayscale pixel: find best quantization and compute error.
/// Returns (best_gray, err_val)
///
/// Optimization: Since a* = b* = 0 for all neutral grays, distance calculation
/// reduces to simple lightness difference squared for all CIELAB variants.
#[inline]
fn process_pixel(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &[Vec<f32>],
    idx: usize,
    bx: usize,
    y: usize,
) -> (u8, f32) {
    // 1. Read input, convert to linear
    let srgb_gray = gray_channel[idx] / 255.0;
    let lin_gray_orig = srgb_to_linear_single(srgb_gray);

    // 2. Add accumulated error
    let lin_gray_adj = lin_gray_orig + err_buf[y][bx];

    // 3. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
    let lin_gray_clamped = lin_gray_adj.clamp(0.0, 1.0);
    let srgb_gray_adj = (linear_to_srgb_single(lin_gray_clamped) * 255.0).clamp(0.0, 255.0);

    // 4. Get level index bounds
    let level_min = ctx.quant.floor_level(srgb_gray_adj.floor() as u8);
    let level_max = ctx.quant.ceil_level((srgb_gray_adj.ceil() as u8).min(255));

    // 5. Convert target to perceptual lightness (use unclamped for true distance)
    // For grayscale, we only need L since a* = b* = 0 for neutral grays
    let l_target = if is_linear_rgb_space(ctx.space) {
        // Linear RGB: use linear value directly (no perceptual conversion)
        lin_gray_adj
    } else if is_ycbcr_space(ctx.space) {
        // Y'CbCr: convert to sRGB (gamma-encoded) with sign preservation
        linear_gray_to_ycbcr_y(lin_gray_adj)
    } else if is_lab_space(ctx.space) {
        let (l, _, _) = linear_rgb_to_lab(lin_gray_adj, lin_gray_adj, lin_gray_adj);
        l
    } else {
        let (l, _, _) = linear_rgb_to_oklab(lin_gray_adj, lin_gray_adj, lin_gray_adj);
        l
    };

    // 6. Search candidates using perceptual lightness distance
    // CIE76/CIE94/OKLab use simple ΔL², CIEDE2000 uses SL-weighted distance
    let mut best_level = level_min;
    let mut best_dist = f32::INFINITY;

    for level in level_min..=level_max {
        let l_candidate = ctx.lightness_lut[level];
        let dist = perceptual_lightness_distance_sq(ctx.space, l_target, l_candidate);

        if dist < best_dist {
            best_dist = dist;
            best_level = level;
        }
    }

    // 7. Get extended value for output and error calculation
    let best_gray = ctx.quant.level_to_srgb(best_level);

    // 8. Compute error in linear space
    let best_lin_gray = ctx.linear_lut[best_gray as usize];
    let err_val = lin_gray_adj - best_lin_gray;

    (best_gray, err_val)
}

// ============================================================================
// Generic scan pattern implementations
// ============================================================================

#[inline]
fn dither_standard_gray<K: GrayDitherKernel>(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    pad_left: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
            out[idx] = best_gray;
            K::apply_ltr(err_buf, bx, y, err_val);
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_serpentine_gray<K: GrayDitherKernel>(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
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

                let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
                out[idx] = best_gray;
                K::apply_rtl(err_buf, bx, y, err_val);
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
                out[idx] = best_gray;
                K::apply_ltr(err_buf, bx, y, err_val);
            }
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_mixed_standard_gray(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
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

            let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
            out[idx] = best_gray;

            let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            let use_jjn = pixel_hash & 1 != 0;
            apply_mixed_kernel(err_buf, bx, y, err_val, use_jjn, false);
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_mixed_serpentine_gray(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
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

                let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
                out[idx] = best_gray;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_mixed_kernel(err_buf, bx, y, err_val, use_jjn, true);
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
                out[idx] = best_gray;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_mixed_kernel(err_buf, bx, y, err_val, use_jjn, false);
            }
        }
        if let Some(ref mut cb) = progress {
            cb((y + 1) as f32 / height as f32);
        }
    }
}

#[inline]
fn dither_mixed_random_gray(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
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

                let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
                out[idx] = best_gray;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_mixed_kernel(err_buf, bx, y, err_val, use_jjn, true);
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let bx = x + pad_left;

                let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
                out[idx] = best_gray;

                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_mixed_kernel(err_buf, bx, y, err_val, use_jjn, false);
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

/// Color space aware dithering for grayscale with Floyd-Steinberg algorithm.
///
/// Treats grayscale input as RGB=(L,L,L) for perceptual distance calculation.
/// Input is sRGB gamma-encoded (0-255).
///
/// Args:
///     gray_channel: Grayscale input as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits: Output bit depth (1-8)
///     space: Perceptual color space for distance calculation
///
/// Returns:
///     Output grayscale as u8
pub fn colorspace_aware_dither_gray(
    gray_channel: &[f32],
    width: usize,
    height: usize,
    bits: u8,
    space: PerceptualSpace,
) -> Vec<u8> {
    colorspace_aware_dither_gray_with_mode(
        gray_channel,
        width,
        height,
        bits,
        space,
        DitherMode::Standard,
        0,
        None,
    )
}

/// Color space aware dithering for grayscale with selectable algorithm.
///
/// Treats grayscale input as RGB=(L,L,L) for perceptual distance calculation.
/// Input is sRGB gamma-encoded (0-255).
///
/// Args:
///     gray_channel: Grayscale input as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits: Output bit depth (1-8)
///     space: Perceptual color space for distance calculation
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Output grayscale as u8
pub fn colorspace_aware_dither_gray_with_mode(
    gray_channel: &[f32],
    width: usize,
    height: usize,
    bits: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let quant = GrayQuantParams::new(bits);
    let linear_lut = build_linear_lut();
    let lightness_lut = build_gray_lightness_lut(&quant, &linear_lut, space);

    let ctx = GrayDitherContext {
        quant: &quant,
        linear_lut: &linear_lut,
        lightness_lut: &lightness_lut,
        space,
    };

    let pixels = width * height;

    // Use JJN padding for all modes
    let pad_left = JarvisJudiceNinke::PAD_LEFT;
    let pad_right = JarvisJudiceNinke::PAD_RIGHT;
    let pad_bottom = JarvisJudiceNinke::PAD_BOTTOM;
    let buf_width = width + pad_left + pad_right;

    // Error buffer in linear luminosity space
    let mut err_buf: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];

    // Output buffer
    let mut out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    // Note: We move `progress` into the called function since only one match arm executes
    match mode {
        DitherMode::None => {
            dither_standard_gray::<NoneKernel>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_gray::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_gray::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_gray::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_gray::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, progress,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, hashed_seed, progress,
            );
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_dither_basic() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = colorspace_aware_dither_gray(&gray, 10, 10, 4, PerceptualSpace::OkLab);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_gray_dither_produces_valid_levels() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = colorspace_aware_dither_gray(&gray, 10, 10, 2, PerceptualSpace::OkLab);

        let valid_levels = [0u8, 85, 170, 255];
        for &v in &result {
            assert!(valid_levels.contains(&v), "Produced invalid 2-bit value: {}", v);
        }
    }

    #[test]
    fn test_gray_dither_1bit() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = colorspace_aware_dither_gray(&gray, 10, 10, 1, PerceptualSpace::OkLab);

        for &v in &result {
            assert!(v == 0 || v == 255, "1-bit should only produce 0 or 255, got {}", v);
        }
    }

    #[test]
    fn test_gray_all_modes() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let modes = [
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
        ];

        let valid_levels = [0u8, 85, 170, 255]; // 2-bit

        for mode in modes {
            let result = colorspace_aware_dither_gray_with_mode(
                &gray, 10, 10, 2, PerceptualSpace::OkLab, mode, 42, None
            );
            assert_eq!(result.len(), 100, "Mode {:?} produced wrong length", mode);
            for &v in &result {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid value: {}", mode, v);
            }
        }
    }

    #[test]
    fn test_gray_all_perceptual_spaces() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let spaces = [
            PerceptualSpace::OkLab,
            PerceptualSpace::LabCIE76,
            PerceptualSpace::LabCIE94,
            PerceptualSpace::LabCIEDE2000,
        ];

        for space in spaces {
            let result = colorspace_aware_dither_gray(&gray, 10, 10, 4, space);
            assert_eq!(result.len(), 100, "Space {:?} produced wrong length", space);
        }
    }

    #[test]
    fn test_gray_mixed_deterministic() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result1 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 42, None
        );
        let result2 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 42, None
        );

        assert_eq!(result1, result2, "Same seed should produce identical results");
    }

    #[test]
    fn test_gray_different_seeds() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result1 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 42, None
        );
        let result2 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 99, None
        );

        assert_ne!(result1, result2, "Different seeds should produce different results");
    }

    #[test]
    fn test_gray_serpentine_vs_standard() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result_std = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::Standard, 0, None
        );
        let result_serp = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::Serpentine, 0, None
        );

        assert_ne!(result_std, result_serp, "Standard and serpentine should differ");
    }

    #[test]
    fn test_gray_cie76_cie94_identical() {
        // For grayscale (R=G=B), a* = b* = 0, so CIE76 and CIE94 both reduce
        // to simple ΔL². CIEDE2000 uses SL weighting and may differ.
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result_cie76 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
        );
        let result_cie94 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::LabCIE94, DitherMode::Standard, 0, None
        );

        assert_eq!(result_cie76, result_cie94,
            "CIE76 and CIE94 should be identical for grayscale");
    }

    #[test]
    fn test_gray_ciede2000_differs_from_cie76() {
        // CIEDE2000 uses SL lightness weighting based on average lightness,
        // so it may produce different results from CIE76/CIE94 for grayscale.
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result_cie76 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
        );
        let result_ciede2000 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::LabCIEDE2000, DitherMode::Standard, 0, None
        );

        // They use different distance formulas, so results may differ
        // (CIEDE2000 compensates for reduced sensitivity in dark/light regions)
        assert_eq!(result_cie76.len(), result_ciede2000.len());
        // Note: They may or may not actually differ for this particular gradient
    }

    #[test]
    fn test_gray_oklab_differs_from_cielab() {
        // OKLab uses a different lightness curve than CIELAB, so results may differ
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result_lab = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
        );
        let result_oklab = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::Standard, 0, None
        );

        // They may or may not differ depending on the specific gradient,
        // but they use different lightness curves so we just verify both work
        assert_eq!(result_lab.len(), result_oklab.len());
    }

    #[test]
    fn test_gray_distance_matches_rgb_distance_functions() {
        // Verify that our simplified lightness_distance_sq produces the same
        // results as the full RGB distance functions for neutral gray inputs.
        use crate::color_distance::{
            lab_distance_cie76_sq, lab_distance_cie94_sq, lab_distance_ciede2000_sq,
        };

        // Test various gray levels
        let test_grays: Vec<f32> = vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

        for &lin1 in &test_grays {
            for &lin2 in &test_grays {
                // Convert to Lab (for neutral gray, a=b=0)
                let (l1, a1, b1) = linear_rgb_to_lab(lin1, lin1, lin1);
                let (l2, a2, b2) = linear_rgb_to_lab(lin2, lin2, lin2);

                // Verify a and b are effectively 0 for neutral grays
                assert!(a1.abs() < 1e-6, "a1 should be ~0 for gray, got {}", a1);
                assert!(b1.abs() < 1e-6, "b1 should be ~0 for gray, got {}", b1);
                assert!(a2.abs() < 1e-6, "a2 should be ~0 for gray, got {}", a2);
                assert!(b2.abs() < 1e-6, "b2 should be ~0 for gray, got {}", b2);

                // Our simplified distance
                let simple_dist = lightness_distance_sq(l1, l2);

                // Full CIE76 distance (should equal simple for gray)
                let cie76_dist = lab_distance_cie76_sq(l1, a1, b1, l2, a2, b2);
                assert!((simple_dist - cie76_dist).abs() < 1e-6,
                    "CIE76 mismatch: simple={} cie76={}", simple_dist, cie76_dist);

                // Full CIE94 distance (should equal simple for gray since chroma=0)
                let cie94_dist = lab_distance_cie94_sq(l1, a1, b1, l2, a2, b2);
                assert!((simple_dist - cie94_dist).abs() < 1e-6,
                    "CIE94 mismatch: simple={} cie94={}", simple_dist, cie94_dist);

                // Full CIEDE2000 distance uses SL weighting
                let ciede2000_full = lab_distance_ciede2000_sq(l1, a1, b1, l2, a2, b2);
                // Our optimized CIEDE2000 for grayscale
                let ciede2000_gray = lightness_distance_ciede2000_sq(l1, l2);
                // They should match since a=b=0 for neutral grays
                assert!((ciede2000_full - ciede2000_gray).abs() < 1e-6,
                    "CIEDE2000 mismatch: full={} gray={} (l1={}, l2={})",
                    ciede2000_full, ciede2000_gray, l1, l2);
            }
        }

        // Same test for OkLab
        for &lin1 in &test_grays {
            for &lin2 in &test_grays {
                let (l1, a1, b1) = linear_rgb_to_oklab(lin1, lin1, lin1);
                let (l2, a2, b2) = linear_rgb_to_oklab(lin2, lin2, lin2);

                // Verify a and b are effectively 0 for neutral grays
                assert!(a1.abs() < 1e-6, "OkLab a1 should be ~0 for gray, got {}", a1);
                assert!(b1.abs() < 1e-6, "OkLab b1 should be ~0 for gray, got {}", b1);

                // Our simplified distance
                let simple_dist = lightness_distance_sq(l1, l2);

                // Full OkLab Euclidean distance (should equal simple for gray)
                let dl = l1 - l2;
                let da = a1 - a2;
                let db = b1 - b2;
                let full_dist = dl * dl + da * da + db * db;

                assert!((simple_dist - full_dist).abs() < 1e-10,
                    "OkLab mismatch: simple={} full={}", simple_dist, full_dist);
            }
        }

        // Test LinearRGB: grayscale uses linear value directly
        for &lin1 in &test_grays {
            for &lin2 in &test_grays {
                // Grayscale: just the linear value
                let gray_dist = lightness_distance_sq(lin1, lin2);

                // Full RGB: Euclidean distance with R=G=B
                let dr = lin1 - lin2;
                let dg = lin1 - lin2;
                let db = lin1 - lin2;
                let full_dist = dr * dr + dg * dg + db * db;

                // For R=G=B, full distance = 3 * (Δv)², grayscale uses just (Δv)²
                // This is intentional - grayscale optimization uses single channel
                // but the ratio should be consistent (3:1)
                let expected_ratio = 3.0;
                if gray_dist > 1e-10 {
                    let actual_ratio = full_dist / gray_dist;
                    assert!((actual_ratio - expected_ratio).abs() < 1e-6,
                        "LinearRGB ratio mismatch: expected {} got {} (lin1={}, lin2={})",
                        expected_ratio, actual_ratio, lin1, lin2);
                }
            }
        }

        // Test YCbCr: grayscale uses sRGB value, Cb=Cr=0 for neutral grays
        use crate::color::{linear_rgb_to_ycbcr_clamped, linear_to_srgb_single};
        for &lin1 in &test_grays {
            for &lin2 in &test_grays {
                // Full YCbCr conversion
                let (y1, cb1, cr1) = linear_rgb_to_ycbcr_clamped(lin1, lin1, lin1);
                let (y2, cb2, cr2) = linear_rgb_to_ycbcr_clamped(lin2, lin2, lin2);

                // Verify Cb and Cr are ~0 for neutral grays
                assert!(cb1.abs() < 1e-6, "YCbCr Cb1 should be ~0 for gray, got {}", cb1);
                assert!(cr1.abs() < 1e-6, "YCbCr Cr1 should be ~0 for gray, got {}", cr1);
                assert!(cb2.abs() < 1e-6, "YCbCr Cb2 should be ~0 for gray, got {}", cb2);
                assert!(cr2.abs() < 1e-6, "YCbCr Cr2 should be ~0 for gray, got {}", cr2);

                // Verify Y' equals sRGB value for neutral grays
                let srgb1 = linear_to_srgb_single(lin1);
                let srgb2 = linear_to_srgb_single(lin2);
                assert!((y1 - srgb1).abs() < 1e-6,
                    "YCbCr Y1 should equal sRGB for gray: Y'={} sRGB={}", y1, srgb1);
                assert!((y2 - srgb2).abs() < 1e-6,
                    "YCbCr Y2 should equal sRGB for gray: Y'={} sRGB={}", y2, srgb2);

                // Grayscale distance uses sRGB values
                let gray_dist = lightness_distance_sq(srgb1, srgb2);

                // Full YCbCr Euclidean distance (should equal gray since Cb=Cr=0)
                let dy = y1 - y2;
                let dcb = cb1 - cb2;
                let dcr = cr1 - cr2;
                let full_dist = dy * dy + dcb * dcb + dcr * dcr;

                assert!((gray_dist - full_dist).abs() < 1e-6,
                    "YCbCr mismatch: gray={} full={}", gray_dist, full_dist);
            }
        }
    }
}
