/// Color space aware dithering for single-channel grayscale images.
///
/// Treats grayscale values as RGB=(L,L,L) for perceptual distance calculation
/// while performing error diffusion in linear luminosity space.
/// Input is sRGB gamma-encoded grayscale (0-255).
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
use crate::dither_colorspace_aware::{
    bit_replicate, wang_hash, DitherMode, PerceptualSpace,
};
use crate::colorspace_derived::f32 as cs;

// ============================================================================
// CIELAB Distance Formulas (adapted for grayscale)
// ============================================================================

/// CIE76 (ΔE*ab): Simple Euclidean distance squared in L*a*b* space
#[inline]
fn lab_distance_cie76_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let dl = l1 - l2;
    let da = a1 - a2;
    let db = b1 - b2;
    dl * dl + da * da + db * db
}

/// CIE94 (ΔE*94): Weighted distance squared
#[inline]
fn lab_distance_cie94_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let dl = l1 - l2;
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let dc = c1 - c2;

    let da = a1 - a2;
    let db = b1 - b2;
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);

    let sc = 1.0 + cs::CIE94_K1 * c1;
    let sh = 1.0 + cs::CIE94_K2 * c1;

    let dl_term = dl;
    let dc_term = dc / sc;
    let dh_term = dh_sq.sqrt() / sh;

    dl_term * dl_term + dc_term * dc_term + dh_term * dh_term
}

/// CIEDE2000 (ΔE00): Most accurate perceptual distance squared
#[inline]
fn lab_distance_ciede2000_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    use std::f32::consts::PI;
    const TWO_PI: f32 = 2.0 * PI;

    let c1_star = (a1 * a1 + b1 * b1).sqrt();
    let c2_star = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1_star + c2_star) / 2.0;

    let c_bar_7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar_7 / (c_bar_7 + cs::CIEDE2000_POW25_7)).sqrt());

    let a1_prime = a1 * (1.0 + g);
    let a2_prime = a2 * (1.0 + g);

    let c1_prime = (a1_prime * a1_prime + b1 * b1).sqrt();
    let c2_prime = (a2_prime * a2_prime + b2 * b2).sqrt();

    let h1_prime = if a1_prime == 0.0 && b1 == 0.0 {
        0.0
    } else {
        let h = b1.atan2(a1_prime);
        if h < 0.0 { h + TWO_PI } else { h }
    };

    let h2_prime = if a2_prime == 0.0 && b2 == 0.0 {
        0.0
    } else {
        let h = b2.atan2(a2_prime);
        if h < 0.0 { h + TWO_PI } else { h }
    };

    let dl_prime = l2 - l1;
    let dc_prime = c2_prime - c1_prime;

    let dh_prime = if c1_prime * c2_prime == 0.0 {
        0.0
    } else {
        let diff = h2_prime - h1_prime;
        if diff.abs() <= PI {
            diff
        } else if diff > PI {
            diff - TWO_PI
        } else {
            diff + TWO_PI
        }
    };

    let dh_prime_big = 2.0 * (c1_prime * c2_prime).sqrt() * (dh_prime / 2.0).sin();

    let l_bar_prime = (l1 + l2) / 2.0;
    let c_bar_prime = (c1_prime + c2_prime) / 2.0;

    let h_bar_prime = if c1_prime * c2_prime == 0.0 {
        h1_prime + h2_prime
    } else if (h1_prime - h2_prime).abs() <= PI {
        (h1_prime + h2_prime) / 2.0
    } else if h1_prime + h2_prime < TWO_PI {
        (h1_prime + h2_prime + TWO_PI) / 2.0
    } else {
        (h1_prime + h2_prime - TWO_PI) / 2.0
    };

    let t = 1.0
        - 0.17 * (h_bar_prime - cs::CIEDE2000_T_30_RAD).cos()
        + 0.24 * (2.0 * h_bar_prime).cos()
        + 0.32 * (3.0 * h_bar_prime + cs::CIEDE2000_T_6_RAD).cos()
        - 0.20 * (4.0 * h_bar_prime - cs::CIEDE2000_T_63_RAD).cos();

    let l_bar_minus_50_sq = (l_bar_prime - 50.0) * (l_bar_prime - 50.0);
    let sl = 1.0 + (cs::CIE94_K2 * l_bar_minus_50_sq) / (20.0 + l_bar_minus_50_sq).sqrt();
    let sc = 1.0 + cs::CIE94_K1 * c_bar_prime;
    let sh = 1.0 + cs::CIE94_K2 * c_bar_prime * t;

    let h_bar_minus_275 = h_bar_prime - cs::CIEDE2000_RT_275_RAD;
    let delta_theta_rad: f32 = cs::CIEDE2000_RT_30_RAD
        * (-((h_bar_minus_275 / cs::CIEDE2000_RT_25_RAD).powi(2))).exp();
    let c_bar_prime_7 = c_bar_prime.powi(7);
    let rc = 2.0_f32 * (c_bar_prime_7 / (c_bar_prime_7 + cs::CIEDE2000_POW25_7)).sqrt();
    let rt = -rc * (2.0_f32 * delta_theta_rad).sin();

    let dl_term = dl_prime / sl;
    let dc_term = dc_prime / sc;
    let dh_term = dh_prime_big / sh;

    dl_term * dl_term + dc_term * dc_term + dh_term * dh_term + rt * dc_term * dh_term
}

/// Compute perceptual distance squared based on the selected space/metric
#[inline]
fn perceptual_distance_sq(
    space: PerceptualSpace,
    l1: f32, a1: f32, b1: f32,
    l2: f32, a2: f32, b2: f32,
) -> f32 {
    match space {
        PerceptualSpace::LabCIE76 => lab_distance_cie76_sq(l1, a1, b1, l2, a2, b2),
        PerceptualSpace::LabCIE94 => lab_distance_cie94_sq(l1, a1, b1, l2, a2, b2),
        PerceptualSpace::LabCIEDE2000 => lab_distance_ciede2000_sq(l1, a1, b1, l2, a2, b2),
        PerceptualSpace::OkLab => {
            let dl = l1 - l2;
            let da = a1 - a2;
            let db = b1 - b2;
            dl * dl + da * da + db * db
        }
    }
}

/// Check if a PerceptualSpace variant uses CIELAB
#[inline]
fn is_lab_space(space: PerceptualSpace) -> bool {
    matches!(space, PerceptualSpace::LabCIE76 | PerceptualSpace::LabCIE94 | PerceptualSpace::LabCIEDE2000)
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

/// Lab color value for LUT storage
#[derive(Clone, Copy, Default)]
struct LabValue {
    l: f32,
    a: f32,
    b: f32,
}

/// Build perceptual LUT for grayscale levels
/// Each level is converted to Lab/OkLab assuming R=G=B
fn build_gray_perceptual_lut(
    quant: &GrayQuantParams,
    linear_lut: &[f32; 256],
    space: PerceptualSpace,
) -> Vec<LabValue> {
    let n = quant.num_levels;
    let mut lut = vec![LabValue::default(); n];

    for level in 0..n {
        let gray_ext = quant.level_values[level];
        let gray_lin = linear_lut[gray_ext as usize];

        // Treat as RGB = (gray, gray, gray)
        let (l, a, b) = if is_lab_space(space) {
            linear_rgb_to_lab(gray_lin, gray_lin, gray_lin)
        } else {
            linear_rgb_to_oklab(gray_lin, gray_lin, gray_lin)
        };

        lut[level] = LabValue { l, a, b };
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
    lab_lut: &'a Vec<LabValue>,
    space: PerceptualSpace,
}

/// Process a single grayscale pixel: find best quantization and compute error.
/// Returns (best_gray, err_val)
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

    // 5. Convert target to Lab/OkLab (use unclamped for true distance)
    // Treat gray as RGB = (gray, gray, gray)
    let lab_target = if is_lab_space(ctx.space) {
        linear_rgb_to_lab(lin_gray_adj, lin_gray_adj, lin_gray_adj)
    } else {
        linear_rgb_to_oklab(lin_gray_adj, lin_gray_adj, lin_gray_adj)
    };

    // 6. Search candidates (1D search since all channels are equal)
    let mut best_level = level_min;
    let mut best_dist = f32::INFINITY;

    for level in level_min..=level_max {
        let lab_candidate = ctx.lab_lut[level];
        let dist = perceptual_distance_sq(
            ctx.space,
            lab_target.0, lab_target.1, lab_target.2,
            lab_candidate.l, lab_candidate.a, lab_candidate.b,
        );

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
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            let (best_gray, err_val) = process_pixel(ctx, gray_channel, err_buf, idx, bx, y);
            out[idx] = best_gray;
            K::apply_ltr(err_buf, bx, y, err_val);
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
) -> Vec<u8> {
    let quant = GrayQuantParams::new(bits);
    let linear_lut = build_linear_lut();
    let lab_lut = build_gray_perceptual_lut(&quant, &linear_lut, space);

    let ctx = GrayDitherContext {
        quant: &quant,
        linear_lut: &linear_lut,
        lab_lut: &lab_lut,
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

    match mode {
        DitherMode::Standard => {
            dither_standard_gray::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_gray::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_gray::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_gray::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, hashed_seed,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, hashed_seed,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, pad_left, hashed_seed,
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
                &gray, 10, 10, 2, PerceptualSpace::OkLab, mode, 42
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
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 42
        );
        let result2 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 42
        );

        assert_eq!(result1, result2, "Same seed should produce identical results");
    }

    #[test]
    fn test_gray_different_seeds() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result1 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 42
        );
        let result2 = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::MixedStandard, 99
        );

        assert_ne!(result1, result2, "Different seeds should produce different results");
    }

    #[test]
    fn test_gray_serpentine_vs_standard() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let result_std = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::Standard, 0
        );
        let result_serp = colorspace_aware_dither_gray_with_mode(
            &gray, 10, 10, 4, PerceptualSpace::OkLab, DitherMode::Serpentine, 0
        );

        assert_ne!(result_std, result_serp, "Standard and serpentine should differ");
    }
}
