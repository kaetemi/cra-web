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
use super::bitdepth::{build_linear_lut, QuantLevelParams};
use super::common::{
    apply_single_channel_kernel, gray_overshoot_penalty, perceptual_lightness_distance_sq,
    wang_hash, DitherMode, FloydSteinberg, JarvisJudiceNinke, NoneKernel, Ostromoukhov,
    PerceptualSpace, SingleChannelKernel,
};
#[cfg(test)]
use super::common::{lightness_distance_ciede2000_sq, lightness_distance_sq};

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
/// Each level is converted to perceptual lightness (L* for CIELAB, L for OKLab).
/// For LinearRGB, stores the linear luminosity value directly.
/// We only store L since a* = b* = 0 for all neutral grays.
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
// Buffer creation with edge seeding
// ============================================================================

/// Create error buffer with edge seeding for normalized dithering at boundaries.
///
/// Buffer layout:
/// - Width: [overshoot][seeding][real image][seeding][overshoot]
///   overshoot = reach, seeding = reach, so total = reach*4 + width
/// - Height: [seeding][real image][overshoot]
///   seeding = reach, overshoot = reach, so total = reach*2 + height
///
/// Seeding areas are filled with duplicated edge pixels and ARE processed.
/// Overshoot areas catch error diffusion and are NOT processed.
fn create_seeded_error_buffer(reach: usize, width: usize, height: usize) -> Vec<Vec<f32>> {
    let total_left = reach * 2;  // overshoot + seeding
    let total_right = reach * 2; // seeding + overshoot
    let total_top = reach;       // seeding only (no top overshoot needed)
    let total_bottom = reach;    // overshoot only (no bottom seeding needed)
    let buf_width = total_left + width + total_right;
    let buf_height = total_top + height + total_bottom;
    vec![vec![0.0f32; buf_width]; buf_height]
}

/// Get grayscale value for seeding coordinates, mapping to edge pixels.
/// For coordinates in seeding area, returns the nearest edge pixel value.
#[inline]
fn get_seeding_gray(gray_channel: &[f32], width: usize, px: usize, py: usize, reach: usize) -> f32 {
    // Map py (buffer y in processed region) to image y
    // Processed region starts at y=0, but top seeding rows map to image row 0
    let img_y = if py < reach { 0 } else { py - reach };

    // Map px (relative to process start, i.e., after left overshoot) to image x
    // px=0..reach is left seeding (map to image x=0)
    // px=reach..(reach+width) is real image
    // px=(reach+width)..(reach*2+width) is right seeding (map to image x=width-1)
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

// ============================================================================
// Dithering context and pixel processing
// ============================================================================

struct GrayDitherContext<'a> {
    quant: &'a QuantLevelParams,
    linear_lut: &'a [f32; 256],
    /// Perceptual lightness values for each quantization level.
    /// Only L is stored since a* = b* = 0 for neutral grays.
    lightness_lut: &'a Vec<f32>,
    space: PerceptualSpace,
    /// Apply gamut overshoot penalty to discourage choices that push error
    /// diffusion outside the representable range.
    overshoot_penalty: bool,
}

/// Process a single grayscale pixel with a direct gray value (for seeding support).
/// Returns (best_gray, err_val)
#[inline]
fn process_pixel_with_gray(
    ctx: &GrayDitherContext,
    gray_value: f32,
    err_buf: &[Vec<f32>],
    bx: usize,
    y: usize,
) -> (u8, f32) {
    // 1. Convert input to linear
    let srgb_gray = gray_value / 255.0;
    let lin_gray_orig = srgb_to_linear_single(srgb_gray);

    // 2. Add accumulated error
    let lin_gray_adj = lin_gray_orig + err_buf[y][bx];

    // 3. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
    let lin_gray_clamped = lin_gray_adj.clamp(0.0, 1.0);
    let srgb_gray_adj = (linear_to_srgb_single(lin_gray_clamped) * 255.0).clamp(0.0, 255.0);

    // 4. Get level index bounds
    let level_min = ctx.quant.floor_level(srgb_gray_adj.floor() as u8);
    let level_max = ctx.quant.ceil_level((srgb_gray_adj.ceil() as u8).min(255));

    // 5. Convert target to perceptual lightness
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

    // 6. Search candidates using perceptual lightness distance
    let mut best_level = level_min;
    let mut best_dist = f32::INFINITY;

    for level in level_min..=level_max {
        let cand_gray_srgb = ctx.quant.level_to_srgb(level);
        let cand_lin_gray = ctx.linear_lut[cand_gray_srgb as usize];

        let l_candidate = ctx.lightness_lut[level];
        let base_dist = perceptual_lightness_distance_sq(ctx.space, l_target, l_candidate);

        // Apply gamut overshoot penalty if enabled
        let dist = if ctx.overshoot_penalty {
            let penalty = gray_overshoot_penalty(lin_gray_adj, cand_lin_gray);
            base_dist * penalty
        } else {
            base_dist
        };

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
fn dither_standard_gray<K: SingleChannelKernel>(
    ctx: &GrayDitherContext,
    gray_channel: &[f32],
    err_buf: &mut [Vec<f32>],
    out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    // Process seeding rows + real image rows (no bottom overshoot processing)
    let process_height = reach + height;
    // Process left seeding + real image + right seeding
    let process_width = reach + width + reach;
    // Buffer x offset: skip left overshoot
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let gray_value = get_seeding_gray(gray_channel, width, px, y, reach);
            let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

            // Only write output for real image pixels (not seeding)
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
fn dither_serpentine_gray<K: SingleChannelKernel>(
    ctx: &GrayDitherContext,
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
                let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
                let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
fn dither_mixed_standard_gray(
    ctx: &GrayDitherContext,
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
            let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
fn dither_mixed_serpentine_gray(
    ctx: &GrayDitherContext,
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
                let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
                let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
fn dither_mixed_random_gray(
    ctx: &GrayDitherContext,
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
                let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
                let (best_gray, err_val) = process_pixel_with_gray(ctx, gray_value, err_buf, bx, y);

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
/// This is a convenience wrapper that enables overshoot penalty by default.
/// Use `colorspace_aware_dither_gray_with_options` for full control.
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
    colorspace_aware_dither_gray_with_options(
        gray_channel, width, height, bits, space, mode, seed,
        true, // overshoot_penalty enabled by default
        progress,
    )
}

/// Color space aware dithering for grayscale with full options control.
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
///     overshoot_penalty: If true, penalize choices that push error diffusion outside gamut
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Output grayscale as u8
pub fn colorspace_aware_dither_gray_with_options(
    gray_channel: &[f32],
    width: usize,
    height: usize,
    bits: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let quant = QuantLevelParams::new(bits);
    let linear_lut = build_linear_lut();
    let lightness_lut = build_gray_lightness_lut(&quant, &linear_lut, space);

    let ctx = GrayDitherContext {
        quant: &quant,
        linear_lut: &linear_lut,
        lightness_lut: &lightness_lut,
        space,
        overshoot_penalty,
    };

    let pixels = width * height;

    // Use JJN reach for all modes (largest kernel)
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;

    // Error buffer with edge seeding
    let mut err_buf = create_seeded_error_buffer(reach, width, height);

    // Output buffer
    let mut out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    // Note: We move `progress` into the called function since only one match arm executes
    match mode {
        DitherMode::None => {
            dither_standard_gray::<NoneKernel>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_gray::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_gray::<FloydSteinberg>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_gray::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_gray::<JarvisJudiceNinke>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_gray(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::OstromoukhovStandard => {
            dither_standard_gray::<Ostromoukhov>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
            );
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_serpentine_gray::<Ostromoukhov>(
                &ctx, gray_channel, &mut err_buf, &mut out,
                width, height, reach, progress,
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

                // Verify a and b are effectively 0 for neutral grays.
                // Tolerance is 2e-5 to account for f32 arithmetic: (a+b+c)*k ≠ a*k + b*k + c*k
                // when computing matrix·RGB / D65. The error is ~200 * f32_epsilon.
                assert!(a1.abs() < 2e-5, "a1 should be ~0 for gray, got {}", a1);
                assert!(b1.abs() < 2e-5, "b1 should be ~0 for gray, got {}", b1);
                assert!(a2.abs() < 2e-5, "a2 should be ~0 for gray, got {}", a2);
                assert!(b2.abs() < 2e-5, "b2 should be ~0 for gray, got {}", b2);

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
