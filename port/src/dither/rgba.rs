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

use crate::color::{linear_to_srgb_single, srgb_to_linear_single};
use crate::color_distance::perceptual_distance_sq;
use super::basic::dither_with_mode_bits;
use super::bitdepth::{build_linear_lut, QuantLevelParams};
use super::common::{
    apply_mixed_kernel_rgb, gamut_overshoot_penalty, linear_rgb_to_perceptual,
    linear_rgb_to_perceptual_clamped, triple32, wang_hash, DitherMode, FloydSteinberg,
    JarvisJudiceNinke, NoneKernel, Ostromoukhov, PerceptualSpace, RgbKernel,
};
use super::kernels::{apply_h2_kernel_rgb, H2_REACH, H2_SEED};

#[derive(Clone, Copy, Default)]
struct LabValue {
    l: f32,
    a: f32,
    b: f32,
}

fn build_perceptual_lut(
    quant: &QuantLevelParams,
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

                let (l, a, b_ch) = linear_rgb_to_perceptual_clamped(space, r_lin, g_lin, b_lin);

                let idx = r_level * n * n + g_level * n + b_level;
                lut[idx] = LabValue { l, a, b: b_ch };
            }
        }
    }

    lut
}

// ============================================================================
// Edge seeding helpers
// ============================================================================

/// Get RGB values for a processing coordinate, handling seeding area mapping.
/// For seeding coordinates, returns the corresponding edge pixel values.
/// For real coordinates, returns the actual pixel values.
#[inline]
fn get_seeding_rgba(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    px: usize,  // processing x (0..seeding_width + real_width + seeding_width)
    py: usize,  // processing y (0..seeding_height + real_height)
    reach: usize,
) -> (f32, f32, f32) {
    // Map processing coordinates to real image coordinates, clamping seeding to edges
    let img_y = if py < reach { 0 } else { py - reach };
    let img_x = if px < reach {
        0  // Left seeding: use first column
    } else if px >= reach + width {
        width - 1  // Right seeding: use last column
    } else {
        px - reach  // Real column
    };
    let idx = img_y * width + img_x;
    (r_channel[idx], g_channel[idx], b_channel[idx])
}

/// Get dithered alpha value for edge seeding.
/// Maps seeding coordinates to edge pixels of the real dithered alpha.
#[inline]
fn get_seeding_alpha_dithered(alpha_dithered: &[u8], width: usize, px: usize, py: usize, reach: usize) -> u8 {
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

/// Context for alpha-aware pixel processing
struct DitherContextRgba<'a> {
    quant_r: &'a QuantLevelParams,
    quant_g: &'a QuantLevelParams,
    quant_b: &'a QuantLevelParams,
    linear_lut: &'a [f32; 256],
    lab_lut: &'a Option<Vec<LabValue>>,
    space: PerceptualSpace,
    /// Pre-dithered alpha channel (u8 values, 0-255)
    alpha_dithered: &'a [u8],
    /// Apply gamut overshoot penalty to discourage choices that push error
    /// diffusion outside the representable color gamut.
    overshoot_penalty: bool,
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
/// Process a single pixel with pre-fetched RGB and alpha values (for seeding support).
/// Same as process_pixel_rgba but takes values directly instead of reading from arrays.
#[inline]
fn process_pixel_rgba_with_values(
    ctx: &DitherContextRgba,
    srgb_r_in: f32,
    srgb_g_in: f32,
    srgb_b_in: f32,
    alpha_dithered_u8: u8,
    err_r: &[Vec<f32>],
    err_g: &[Vec<f32>],
    err_b: &[Vec<f32>],
    bx: usize,
    y: usize,
) -> (u8, u8, u8, f32, f32, f32) {
    // Get dithered alpha for this pixel (normalized to 0-1)
    let alpha = alpha_dithered_u8 as f32 / 255.0;

    // 1. Read accumulated error (e_in)
    let err_r_in = err_r[y][bx];
    let err_g_in = err_g[y][bx];
    let err_b_in = err_b[y][bx];

    // 2. Convert input to Linear RGB
    let srgb_r = srgb_r_in / 255.0;
    let srgb_g = srgb_g_in / 255.0;
    let srgb_b = srgb_b_in / 255.0;

    let lin_r_orig = srgb_to_linear_single(srgb_r);
    let lin_g_orig = srgb_to_linear_single(srgb_g);
    let lin_b_orig = srgb_to_linear_single(srgb_b);

    // 3. Add accumulated error (skip for fully transparent pixels)
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

    // 6. Convert target to perceptual space (unclamped for true distance)
    let lab_target = linear_rgb_to_perceptual(ctx.space, lin_r_adj, lin_g_adj, lin_b_adj);

    // 7. Search candidates for best quantization
    let mut best_r_level = r_min;
    let mut best_g_level = g_min;
    let mut best_b_level = b_min;
    let mut best_dist = f32::INFINITY;

    for r_level in r_min..=r_max {
        let r_ext = ctx.quant_r.level_to_srgb(r_level);
        let cand_r_lin = ctx.linear_lut[r_ext as usize];

        for g_level in g_min..=g_max {
            let g_ext = ctx.quant_g.level_to_srgb(g_level);
            let cand_g_lin = ctx.linear_lut[g_ext as usize];

            for b_level in b_min..=b_max {
                let b_ext = ctx.quant_b.level_to_srgb(b_level);
                let cand_b_lin = ctx.linear_lut[b_ext as usize];

                let lab_candidate = if let Some(lut) = ctx.lab_lut {
                    let n = ctx.quant_r.num_levels;
                    let lut_idx = r_level * n * n + g_level * n + b_level;
                    lut[lut_idx]
                } else {
                    let (l, a, b_ch) =
                        linear_rgb_to_perceptual_clamped(ctx.space, cand_r_lin, cand_g_lin, cand_b_lin);
                    LabValue { l, a, b: b_ch }
                };

                let base_dist = perceptual_distance_sq(
                    ctx.space,
                    lab_target.0, lab_target.1, lab_target.2,
                    lab_candidate.l, lab_candidate.a, lab_candidate.b,
                );

                // Apply gamut overshoot penalty if enabled
                let dist = if ctx.overshoot_penalty {
                    let penalty = gamut_overshoot_penalty(
                        lin_r_adj, lin_g_adj, lin_b_adj,
                        cand_r_lin, cand_g_lin, cand_b_lin,
                    );
                    base_dist * penalty
                } else {
                    base_dist
                };

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
fn dither_standard_rgba<K: RgbKernel>(
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
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for bx in bx_start..bx_start + process_width {
            let px = bx - bx_start;
            let in_real_image = y >= reach && px >= reach && px < reach + width;

            let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
            } else {
                let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                (r, g, b, a)
            };

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

            if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
            }

            K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_serpentine_rgba<K: RgbKernel>(
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
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            for bx in (bx_start..bx_start + process_width).rev() {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
                } else {
                    let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                    let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                    (r, g, b, a)
                };

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                K::apply_rtl(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
            }
        } else {
            for bx in bx_start..bx_start + process_width {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
                } else {
                    let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                    let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                    (r, g, b, a)
                };

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
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
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for bx in bx_start..bx_start + process_width {
            let px = bx - bx_start;
            let in_real_image = y >= reach && px >= reach && px < reach + width;

            let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
            } else {
                let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                (r, g, b, a)
            };

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

            if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
            }

            let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
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
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            for bx in (bx_start..bx_start + process_width).rev() {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
                } else {
                    let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                    let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                    (r, g, b, a)
                };

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            for bx in bx_start..bx_start + process_width {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
                } else {
                    let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                    let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                    (r, g, b, a)
                };

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
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
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_rtl = row_hash >> 31 != 0;

        if is_rtl {
            for bx in (bx_start..bx_start + process_width).rev() {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
                } else {
                    let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                    let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                    (r, g, b, a)
                };

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            for bx in bx_start..bx_start + process_width {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
                } else {
                    let (r, g, b) = get_seeding_rgba(r_channel, g_channel, b_channel, width, px, y, reach);
                    let a = get_seeding_alpha_dithered(ctx.alpha_dithered, width, px, y, reach);
                    (r, g, b, a)
                };

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
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
// Mixed H2 (second-order kernel) RGBA dithering
// ============================================================================

/// Get RGBA values for H2 processing coordinate, handling seeding area mapping.
#[inline]
fn get_seeding_rgba_h2(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    px: usize,
    py: usize,
    seed: usize,
) -> (f32, f32, f32) {
    let img_y = if py < seed { 0 } else { py - seed };
    let img_x = if px < seed {
        0
    } else if px >= seed + width {
        width - 1
    } else {
        px - seed
    };
    let idx = img_y * width + img_x;
    (r_channel[idx], g_channel[idx], b_channel[idx])
}

/// Get dithered alpha value for H2 edge seeding.
#[inline]
fn get_seeding_alpha_dithered_h2(alpha_dithered: &[u8], width: usize, px: usize, py: usize, seed: usize) -> u8 {
    let img_y = if py < seed { 0 } else { py - seed };
    let img_x = if px < seed {
        0
    } else if px >= seed + width {
        width - 1
    } else {
        px - seed
    };
    let idx = img_y * width + img_x;
    alpha_dithered[idx]
}

/// Mixed H2 kernel dithering for colorspace-aware RGBA, LTR only.
#[inline]
fn dither_mixed_h2_standard_rgba(
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;

            let in_real_image = y >= seed && px >= seed && px < seed + width;

            let (r_val, g_val, b_val, alpha_u8) = if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], ctx.alpha_dithered[idx])
            } else {
                let (r, g, b) = get_seeding_rgba_h2(r_channel, g_channel, b_channel, width, px, y, seed);
                let a = get_seeding_alpha_dithered_h2(ctx.alpha_dithered, width, px, y, seed);
                (r, g, b, a)
            };

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_rgba_with_values(ctx, r_val, g_val, b_val, alpha_u8, err_r, err_g, err_b, bx, y);

            if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
            }

            let img_x = px.wrapping_sub(seed);
            let img_y = y.wrapping_sub(seed);
            let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            apply_h2_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
        }
        if let Some(ref mut cb) = progress {
            if y >= seed {
                cb((y - seed + 1) as f32 / height as f32);
            }
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
/// This is a convenience wrapper that enables overshoot penalty by default.
/// Use `colorspace_aware_dither_rgba_with_options` for full control.
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
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    colorspace_aware_dither_rgba_with_options(
        r_channel, g_channel, b_channel, a_channel,
        width, height,
        bits_r, bits_g, bits_b, bits_a,
        space, mode, alpha_mode, seed,
        true, // overshoot_penalty enabled by default
        progress,
    )
}

/// Color space aware RGBA dithering with full options control.
///
/// Uses perceptual color space (Lab or OkLab) for finding the best RGB quantization
/// candidate, with alpha-aware error diffusion.
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b, bits_a: Bit depth for each channel (1-8)
///     space: Perceptual color space for RGB distance calculation
///     mode: Dithering algorithm and scanning mode for RGB channels
///     alpha_mode: Dithering algorithm and scanning mode for alpha channel
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     overshoot_penalty: If true, penalize quantization choices that would push
///         error diffusion outside the representable color gamut (recommended)
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as u8
pub fn colorspace_aware_dither_rgba_with_options(
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
    overshoot_penalty: bool,
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
    let quant_r = QuantLevelParams::new(bits_r);
    let quant_g = QuantLevelParams::new(bits_g);
    let quant_b = QuantLevelParams::new(bits_b);

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
        overshoot_penalty,
    };

    // H2 needs different buffer dimensions (REACH=4, SEED=16), handle as early return
    if mode == DitherMode::MixedH2Standard {
        let h2_reach = H2_REACH;
        let h2_seed = H2_SEED;
        let buf_width = h2_reach + h2_seed + width + h2_seed + h2_reach;
        let buf_height = h2_seed + height + h2_reach;

        let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

        let mut r_out = vec![0u8; pixels];
        let mut g_out = vec![0u8; pixels];
        let mut b_out = vec![0u8; pixels];

        let hashed_seed = triple32(seed);

        dither_mixed_h2_standard_rgba(
            &ctx, r_channel, g_channel, b_channel,
            &mut err_r, &mut err_g, &mut err_b,
            &mut r_out, &mut g_out, &mut b_out,
            width, height, hashed_seed, progress,
        );

        return (r_out, g_out, b_out, alpha_dithered);
    }


    // Use JJN reach for all modes (largest kernel)
    // Buffer layout: [overshoot][seeding][real][seeding][overshoot]
    let reach = <JarvisJudiceNinke as RgbKernel>::REACH;
    let buf_width = reach * 4 + width;  // overshoot + seeding + real + seeding + overshoot
    let buf_height = reach * 2 + height;  // seeding + real + overshoot

    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

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
                width, height, reach, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_rgba::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_rgba::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedStandard | DitherMode::MixedWangStandard | DitherMode::MixedLowbiasOldStandard => {
            dither_mixed_standard_rgba(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine | DitherMode::MixedWangSerpentine | DitherMode::MixedLowbiasOldSerpentine => {
            dither_mixed_serpentine_rgba(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_rgba(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::OstromoukhovStandard => {
            dither_standard_rgba::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_serpentine_rgba::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        // Zhou-Fang: fall back to Ostromoukhov for colorspace-aware dithering
        DitherMode::ZhouFangStandard => {
            dither_standard_rgba::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::ZhouFangSerpentine => {
            dither_serpentine_rgba::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        // Ulichney: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::UlichneyStandard | DitherMode::UlichneyWeightStandard => {
            dither_standard_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::UlichneySerpentine | DitherMode::UlichneyWeightSerpentine => {
            dither_serpentine_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        // FS+TPDF: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::FsTpdfStandard => {
            dither_standard_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::FsTpdfSerpentine => {
            dither_serpentine_rgba::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedH2Standard => {
            unreachable!("H2 modes handled in early return above");
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
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (r, g, b, a) = pixels_to_channels_rgba(pixels);
    colorspace_aware_dither_rgba_with_options(
        &r, &g, &b, &a,
        width, height,
        bits_r, bits_g, bits_b, bits_a,
        space, mode, alpha_mode, seed, overshoot_penalty, progress,
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
