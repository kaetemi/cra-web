/// Color space aware dithering for grayscale images with alpha channel.
///
/// Extends grayscale luminosity dithering to handle alpha channel with proper error propagation:
/// - Alpha channel is dithered first using standard single-channel dithering
/// - Luminosity channel is then dithered with alpha-aware error diffusion:
///   - Error that couldn't be applied due to transparency is fully propagated
///   - Quantization error is weighted by pixel visibility (alpha)
///
/// This ensures that:
/// - Fully transparent pixels pass their accumulated error to neighbors
/// - Partially transparent pixels proportionally absorb/propagate error
/// - The visual result of compositing is properly optimized
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

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_to_srgb_single, srgb_to_linear_single,
};
use crate::color_distance::{is_lab_space, is_linear_rgb_space, is_ycbcr_space};
use super::basic::dither_with_mode_bits;
use super::bitdepth::{build_linear_lut, QuantLevelParams};
use super::common::{
    apply_single_channel_kernel, perceptual_lightness_distance_sq, wang_hash, DitherMode,
    FloydSteinberg, JarvisJudiceNinke, NoneKernel, PerceptualSpace, SingleChannelKernel,
};

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

/// Build perceptual lightness LUT for grayscale levels.
fn build_gray_lightness_lut(
    quant: &QuantLevelParams,
    linear_lut: &[f32; 256],
    space: PerceptualSpace,
) -> Vec<f32> {
    let n = quant.num_levels;
    let mut lut = vec![0.0f32; n];

    for level in 0..n {
        let gray_ext = quant.level_values[level];
        let gray_lin = linear_lut[gray_ext as usize];

        let l = if is_linear_rgb_space(space) {
            gray_lin
        } else if is_ycbcr_space(space) {
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
// Buffer creation with edge seeding
// ============================================================================

/// Create error buffer with edge seeding for normalized dithering at boundaries.
fn create_seeded_error_buffer(reach: usize, width: usize, height: usize) -> Vec<Vec<f32>> {
    let total_left = reach * 2;  // overshoot + seeding
    let total_right = reach * 2; // seeding + overshoot
    let total_top = reach;       // seeding only
    let total_bottom = reach;    // overshoot only
    let buf_width = total_left + width + total_right;
    let buf_height = total_top + height + total_bottom;
    vec![vec![0.0f32; buf_width]; buf_height]
}

/// Get grayscale value for seeding coordinates, mapping to edge pixels.
#[inline]
fn get_seeding_gray(gray_channel: &[f32], width: usize, px: usize, py: usize, reach: usize) -> f32 {
    let img_y = if py < reach { 0 } else { py - reach };
    let img_x = if px < reach {
        0
    } else if px >= reach + width {
        width - 1
    } else {
        px - reach
    };
    let idx = img_y * width + img_x;
    gray_channel[idx]
}

/// Get alpha value for seeding coordinates, mapping to edge pixels.
#[inline]
fn get_seeding_alpha(alpha_dithered: &[u8], width: usize, px: usize, py: usize, reach: usize) -> u8 {
    let img_y = if py < reach { 0 } else { py - reach };
    let img_x = if px < reach {
        0
    } else if px >= reach + width {
        width - 1
    } else {
        px - reach
    };
    let idx = img_y * width + img_x;
    alpha_dithered[idx]
}

// ============================================================================
// Alpha-aware dithering context and pixel processing
// ============================================================================

struct GrayAlphaDitherContext<'a> {
    quant: &'a QuantLevelParams,
    linear_lut: &'a [f32; 256],
    lightness_lut: &'a Vec<f32>,
    space: PerceptualSpace,
    /// Pre-dithered alpha channel (u8 values, 0-255)
    alpha_dithered: &'a [u8],
}

/// Process a single pixel with alpha-aware error diffusion.
///
/// The key difference from grayscale-only dithering is how error is calculated:
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
/// Returns (best_gray, err_val)
#[inline]
fn process_pixel_gray_alpha(
    ctx: &GrayAlphaDitherContext,
    gray_channel: &[f32],
    err_buf: &[Vec<f32>],
    idx: usize,
    bx: usize,
    y: usize,
) -> (u8, f32) {
    // Get dithered alpha for this pixel (normalized to 0-1)
    let alpha = ctx.alpha_dithered[idx] as f32 / 255.0;

    // 1. Read accumulated error (e_in)
    let err_in = err_buf[y][bx];

    // 2. Read input, convert to linear
    let srgb_gray = gray_channel[idx] / 255.0;
    let lin_gray_orig = srgb_to_linear_single(srgb_gray);

    // 3. Add accumulated error (skip for fully transparent pixels)
    let lin_gray_adj = if alpha == 0.0 {
        lin_gray_orig
    } else {
        lin_gray_orig + err_in
    };

    // 4. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
    let lin_gray_clamped = lin_gray_adj.clamp(0.0, 1.0);
    let srgb_gray_adj = (linear_to_srgb_single(lin_gray_clamped) * 255.0).clamp(0.0, 255.0);

    // 5. Get level index bounds
    let level_min = ctx.quant.floor_level(srgb_gray_adj.floor() as u8);
    let level_max = ctx.quant.ceil_level((srgb_gray_adj.ceil() as u8).min(255));

    // 6. Convert target to perceptual lightness (use unclamped for true distance)
    let l_target = if is_linear_rgb_space(ctx.space) {
        lin_gray_adj
    } else if is_ycbcr_space(ctx.space) {
        linear_gray_to_ycbcr_y(lin_gray_adj)
    } else if is_lab_space(ctx.space) {
        let (l, _, _) = linear_rgb_to_lab(lin_gray_adj, lin_gray_adj, lin_gray_adj);
        l
    } else {
        let (l, _, _) = linear_rgb_to_oklab(lin_gray_adj, lin_gray_adj, lin_gray_adj);
        l
    };

    // 7. Search candidates using perceptual lightness distance
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

    // 8. Get extended value for output
    let best_gray = ctx.quant.level_to_srgb(best_level);

    // 9. Compute quantized linear value
    let best_lin_gray = ctx.linear_lut[best_gray as usize];

    // 10. Compute alpha-aware error to diffuse
    // Formula: error = (1 - α) × e_in + α × q_err
    // Where q_err = linear_adj - linear_quant
    let q_err = lin_gray_adj - best_lin_gray;
    let one_minus_alpha = 1.0 - alpha;
    let err_val = one_minus_alpha * err_in + alpha * q_err;

    (best_gray, err_val)
}

/// Process a single pixel with alpha-aware error diffusion, taking values directly (for seeding support).
/// Returns (best_gray, err_val)
#[inline]
fn process_pixel_gray_alpha_with_values(
    ctx: &GrayAlphaDitherContext,
    gray_value: f32,
    alpha_value: u8,
    err_buf: &[Vec<f32>],
    bx: usize,
    y: usize,
) -> (u8, f32) {
    // Get alpha (normalized to 0-1)
    let alpha = alpha_value as f32 / 255.0;

    // 1. Read accumulated error (e_in)
    let err_in = err_buf[y][bx];

    // 2. Convert input to linear
    let srgb_gray = gray_value / 255.0;
    let lin_gray_orig = srgb_to_linear_single(srgb_gray);

    // 3. Add accumulated error (skip for fully transparent pixels)
    let lin_gray_adj = if alpha == 0.0 {
        lin_gray_orig
    } else {
        lin_gray_orig + err_in
    };

    // 4. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
    let lin_gray_clamped = lin_gray_adj.clamp(0.0, 1.0);
    let srgb_gray_adj = (linear_to_srgb_single(lin_gray_clamped) * 255.0).clamp(0.0, 255.0);

    // 5. Get level index bounds
    let level_min = ctx.quant.floor_level(srgb_gray_adj.floor() as u8);
    let level_max = ctx.quant.ceil_level((srgb_gray_adj.ceil() as u8).min(255));

    // 6. Convert target to perceptual lightness
    let l_target = if is_linear_rgb_space(ctx.space) {
        lin_gray_adj
    } else if is_ycbcr_space(ctx.space) {
        linear_gray_to_ycbcr_y(lin_gray_adj)
    } else if is_lab_space(ctx.space) {
        let (l, _, _) = linear_rgb_to_lab(lin_gray_adj, lin_gray_adj, lin_gray_adj);
        l
    } else {
        let (l, _, _) = linear_rgb_to_oklab(lin_gray_adj, lin_gray_adj, lin_gray_adj);
        l
    };

    // 7. Search candidates using perceptual lightness distance
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

    // 8. Get extended value for output
    let best_gray = ctx.quant.level_to_srgb(best_level);

    // 9. Compute quantized linear value
    let best_lin_gray = ctx.linear_lut[best_gray as usize];

    // 10. Compute alpha-aware error to diffuse
    let q_err = lin_gray_adj - best_lin_gray;
    let one_minus_alpha = 1.0 - alpha;
    let err_val = one_minus_alpha * err_in + alpha * q_err;

    (best_gray, err_val)
}

// ============================================================================
// Generic scan pattern implementations
// ============================================================================

#[inline]
fn dither_standard_gray_alpha<K: SingleChannelKernel>(
    ctx: &GrayAlphaDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
            let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
            let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

            let in_real_y = y >= reach;
            let in_real_x = px >= reach && px < reach + width;
            if in_real_y && in_real_x {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                out[idx] = best_gray;
            }

            K::apply_ltr(err_buf, bx, y, err_val);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_serpentine_gray_alpha<K: SingleChannelKernel>(
    ctx: &GrayAlphaDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // RTL scan
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
                let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
                let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    out[idx] = best_gray;
                }

                K::apply_rtl(err_buf, bx, y, err_val);
            }
        } else {
            // LTR scan
            for px in 0..process_width {
                let bx = bx_start + px;
                let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
                let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
                let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    out[idx] = best_gray;
                }

                K::apply_ltr(err_buf, bx, y, err_val);
            }
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_standard_gray_alpha(
    ctx: &GrayAlphaDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
            let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
            let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

            let in_real_y = y >= reach;
            let in_real_x = px >= reach && px < reach + width;
            if in_real_y && in_real_x {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                out[idx] = best_gray;
            }

            let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            let use_jjn = pixel_hash & 1 != 0;
            apply_single_channel_kernel(err_buf, bx, y, err_val, use_jjn, false);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_serpentine_gray_alpha(
    ctx: &GrayAlphaDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // RTL scan
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
                let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
                let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    out[idx] = best_gray;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_single_channel_kernel(err_buf, bx, y, err_val, use_jjn, true);
            }
        } else {
            // LTR scan
            for px in 0..process_width {
                let bx = bx_start + px;
                let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
                let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
                let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    out[idx] = best_gray;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_single_channel_kernel(err_buf, bx, y, err_val, use_jjn, false);
            }
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_random_gray_alpha(
    ctx: &GrayAlphaDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_rtl = row_hash & 1 == 1;

        if is_rtl {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
                let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
                let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    out[idx] = best_gray;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_single_channel_kernel(err_buf, bx, y, err_val, use_jjn, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
                let alpha_value = get_seeding_alpha(ctx.alpha_dithered, width, px, y, reach);
                let (best_gray, err_val) = process_pixel_gray_alpha_with_values(ctx, gray_value, alpha_value, err_buf, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    out[idx] = best_gray;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_single_channel_kernel(err_buf, bx, y, err_val, use_jjn, false);
            }
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Color space aware dithering for grayscale with alpha channel.
///
/// This is the simplified API that uses Floyd-Steinberg with standard scanning.
/// For other algorithms and scan patterns, use `colorspace_aware_dither_gray_alpha_with_mode`.
///
/// Args:
///     gray_channel: Grayscale input as f32 in range [0, 255]
///     alpha_channel: Alpha input as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_gray: Bit depth for grayscale channel (1-8)
///     bits_alpha: Bit depth for alpha channel (1-8)
///     space: Perceptual color space for distance calculation
///
/// Returns:
///     (gray_out, alpha_out): Output channels as u8
pub fn colorspace_aware_dither_gray_alpha(
    gray_channel: &[f32],
    alpha_channel: &[f32],
    width: usize,
    height: usize,
    bits_gray: u8,
    bits_alpha: u8,
    space: PerceptualSpace,
) -> (Vec<u8>, Vec<u8>) {
    colorspace_aware_dither_gray_alpha_with_mode(
        gray_channel,
        alpha_channel,
        width,
        height,
        bits_gray,
        bits_alpha,
        space,
        DitherMode::Standard,
        DitherMode::Standard,
        0,
        None,
    )
}

/// Color space aware dithering for grayscale with alpha channel and selectable algorithm.
///
/// Process:
/// 1. Alpha channel is dithered first using the specified alpha dithering mode
/// 2. Grayscale channel is then dithered with alpha-aware error propagation:
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
///     gray_channel: Grayscale input as f32 in range [0, 255]
///     alpha_channel: Alpha input as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_gray: Bit depth for grayscale channel (1-8)
///     bits_alpha: Bit depth for alpha channel (1-8)
///     space: Perceptual color space for distance calculation
///     mode: Dithering algorithm and scanning mode for grayscale channel
///     alpha_mode: Dithering algorithm and scanning mode for alpha channel
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (gray_out, alpha_out): Output channels as u8
pub fn colorspace_aware_dither_gray_alpha_with_mode(
    gray_channel: &[f32],
    alpha_channel: &[f32],
    width: usize,
    height: usize,
    bits_gray: u8,
    bits_alpha: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    alpha_mode: DitherMode,
    seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>) {
    let pixels = width * height;

    // Step 1: Dither alpha channel first using the specified alpha mode
    // Alpha is linear, so standard dithering is correct
    let alpha_dithered = dither_with_mode_bits(alpha_channel, width, height, alpha_mode, seed.wrapping_add(1), bits_alpha, None);

    // Report alpha dithering complete (10% of total progress)
    if let Some(ref mut cb) = progress {
        cb(0.1);
    }

    // Step 2: Set up grayscale dithering with alpha awareness
    let quant = QuantLevelParams::new(bits_gray);
    let linear_lut = build_linear_lut();
    let lightness_lut = build_gray_lightness_lut(&quant, &linear_lut, space);

    let ctx = GrayAlphaDitherContext {
        quant: &quant,
        linear_lut: &linear_lut,
        lightness_lut: &lightness_lut,
        space,
        alpha_dithered: &alpha_dithered,
    };

    // Use JJN reach for all modes (largest kernel)
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;

    // Error buffer with edge seeding
    let mut err_buf = create_seeded_error_buffer(reach, width, height);
    let mut out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    match mode {
        DitherMode::None => {
            dither_standard_gray_alpha::<NoneKernel>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_gray_alpha::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_gray_alpha::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_gray_alpha::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_gray_alpha::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_gray_alpha(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_gray_alpha(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_gray_alpha(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, hashed_seed, progress,
            );
        }
    }

    (out, alpha_dithered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_alpha_dither_basic() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab
        );

        assert_eq!(gray_out.len(), 100);
        assert_eq!(alpha_out.len(), 100);
    }

    #[test]
    fn test_gray_alpha_produces_valid_levels() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
            &gray, &alpha, 10, 10, 2, 2, PerceptualSpace::OkLab
        );

        let valid_levels = [0u8, 85, 170, 255];
        for &v in &gray_out {
            assert!(valid_levels.contains(&v), "Gray produced invalid 2-bit value: {}", v);
        }
        for &v in &alpha_out {
            assert!(valid_levels.contains(&v), "Alpha produced invalid 2-bit value: {}", v);
        }
    }

    #[test]
    fn test_gray_alpha_fully_transparent() {
        let gray: Vec<f32> = vec![128.0; 100];
        let alpha: Vec<f32> = vec![0.0; 100]; // Fully transparent

        let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab
        );

        // Alpha should be 0
        for &v in &alpha_out {
            assert_eq!(v, 0, "Transparent alpha should dither to 0");
        }

        // Gray values will still be quantized
        assert_eq!(gray_out.len(), 100);
    }

    #[test]
    fn test_gray_alpha_fully_opaque() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = vec![255.0; 100]; // Fully opaque

        let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab
        );

        // Alpha should be 255
        for &v in &alpha_out {
            assert_eq!(v, 255, "Opaque alpha should dither to 255");
        }

        assert_eq!(gray_out.len(), 100);
    }

    #[test]
    fn test_gray_alpha_all_modes() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 * 2.55).collect();

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
            let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha_with_mode(
                &gray, &alpha, 10, 10, 2, 2, PerceptualSpace::OkLab, mode, mode, 42, None
            );

            assert_eq!(gray_out.len(), 100, "Mode {:?} produced wrong gray length", mode);
            assert_eq!(alpha_out.len(), 100, "Mode {:?} produced wrong alpha length", mode);

            for &v in &gray_out {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid gray value: {}", mode, v);
            }
            for &v in &alpha_out {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid alpha value: {}", mode, v);
            }
        }
    }

    #[test]
    fn test_gray_alpha_mixed_deterministic() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 * 2.55).collect();

        let result1 = colorspace_aware_dither_gray_alpha_with_mode(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab, DitherMode::MixedStandard, DitherMode::MixedStandard, 42, None
        );
        let result2 = colorspace_aware_dither_gray_alpha_with_mode(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab, DitherMode::MixedStandard, DitherMode::MixedStandard, 42, None
        );

        assert_eq!(result1.0, result2.0, "Gray should be deterministic");
        assert_eq!(result1.1, result2.1, "Alpha should be deterministic");
    }

    #[test]
    fn test_gray_alpha_different_seeds() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 * 2.55).collect();

        let result1 = colorspace_aware_dither_gray_alpha_with_mode(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab, DitherMode::MixedStandard, DitherMode::MixedStandard, 42, None
        );
        let result2 = colorspace_aware_dither_gray_alpha_with_mode(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab, DitherMode::MixedStandard, DitherMode::MixedStandard, 99, None
        );

        // At least one should differ
        assert!(result1.0 != result2.0 || result1.1 != result2.1,
            "Different seeds should produce different results");
    }

    #[test]
    fn test_gray_alpha_semi_transparent() {
        let gray: Vec<f32> = vec![127.5; 100];
        let alpha: Vec<f32> = vec![127.5; 100]; // 50% transparent

        let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
            &gray, &alpha, 10, 10, 4, 8, PerceptualSpace::OkLab
        );

        assert_eq!(gray_out.len(), 100);
        assert_eq!(alpha_out.len(), 100);

        // Alpha should have some variation around 127-128
        let alpha_sum: u32 = alpha_out.iter().map(|&v| v as u32).sum();
        let alpha_avg = alpha_sum as f32 / 100.0;
        assert!((alpha_avg - 127.5).abs() < 5.0, "Alpha average should be near 127.5, got {}", alpha_avg);
    }

    #[test]
    fn test_gray_alpha_all_perceptual_spaces() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let spaces = [
            PerceptualSpace::OkLab,
            PerceptualSpace::LabCIE76,
            PerceptualSpace::LabCIE94,
            PerceptualSpace::LabCIEDE2000,
            PerceptualSpace::LinearRGB,
            PerceptualSpace::YCbCr,
        ];

        for space in spaces {
            let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
                &gray, &alpha, 10, 10, 4, 8, space
            );
            assert_eq!(gray_out.len(), 100, "Space {:?} produced wrong gray length", space);
            assert_eq!(alpha_out.len(), 100, "Space {:?} produced wrong alpha length", space);
        }
    }

    #[test]
    fn test_gray_alpha_1bit() {
        let gray: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let alpha: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        let (gray_out, alpha_out) = colorspace_aware_dither_gray_alpha(
            &gray, &alpha, 10, 10, 1, 1, PerceptualSpace::OkLab
        );

        for &v in &gray_out {
            assert!(v == 0 || v == 255, "1-bit gray should only produce 0 or 255, got {}", v);
        }
        for &v in &alpha_out {
            assert!(v == 0 || v == 255, "1-bit alpha should only produce 0 or 255, got {}", v);
        }
    }
}
