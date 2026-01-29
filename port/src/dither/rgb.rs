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

use crate::color::{linear_to_srgb_single, srgb_to_linear_single};
use crate::color_distance::perceptual_distance_sq;
use super::bitdepth::{build_linear_lut, QuantLevelParams};
use super::common::{
    apply_mixed_kernel_rgb, gamut_overshoot_penalty, linear_rgb_to_perceptual,
    linear_rgb_to_perceptual_clamped, lowbias32, FloydSteinberg, JarvisJudiceNinke, NoneKernel, Ostromoukhov, RgbKernel,
};
use super::kernels::{apply_h2_kernel_rgb, H2_REACH, H2_SEED};

// Re-export common types for backwards compatibility
#[allow(unused_imports)]
pub use crate::color_distance::{
    lab_distance_cie76_sq, lab_distance_cie94_sq, lab_distance_ciede2000_sq,
};
#[allow(unused_imports)]
pub use super::common::{bit_replicate, wang_hash, DitherMode, PerceptualSpace};

/// Lab color value for LUT storage
#[derive(Clone, Copy, Default)]
struct LabValue {
    l: f32,
    a: f32,
    b: f32,
}

/// Build Lab/OkLab/LinearRGB/sRGB LUT indexed by (r_level, g_level, b_level)
/// Returns a flat Vec where index = r_level * num_levels^2 + g_level * num_levels + b_level
/// For LinearRGB mode, stores linear R/G/B values in the l/a/b fields
/// For sRGB mode, stores gamma-encoded R/G/B values in the l/a/b fields
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
// Generic scan pattern implementations
// ============================================================================

/// Get RGB values for a processing coordinate, handling seeding area mapping.
/// For seeding coordinates, returns the corresponding edge pixel values.
/// For real coordinates, returns the actual pixel values.
#[inline]
fn get_seeding_rgb(
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

/// Process a single pixel given RGB values directly (for seeding support).
/// Returns (best_r, best_g, best_b, err_r, err_g, err_b)
#[inline]
fn process_pixel_with_rgb(
    ctx: &DitherContext,
    srgb_r_in: f32,
    srgb_g_in: f32,
    srgb_b_in: f32,
    err_r: &[Vec<f32>],
    err_g: &[Vec<f32>],
    err_b: &[Vec<f32>],
    bx: usize,
    y: usize,
) -> (u8, u8, u8, f32, f32, f32) {
    // 1. Convert input to Linear RGB
    let srgb_r = srgb_r_in / 255.0;
    let srgb_g = srgb_g_in / 255.0;
    let srgb_b = srgb_b_in / 255.0;

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

    // 5. Convert target to perceptual space (unclamped for true distance)
    let lab_target = linear_rgb_to_perceptual(ctx.space, lin_r_adj, lin_g_adj, lin_b_adj);

    // 6. Search candidates
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

    // 7. Get final quantized sRGB values
    let best_r = ctx.quant_r.level_to_srgb(best_r_level);
    let best_g = ctx.quant_g.level_to_srgb(best_g_level);
    let best_b = ctx.quant_b.level_to_srgb(best_b_level);

    // 8. Compute error in linear space
    let best_lin_r = ctx.linear_lut[best_r as usize];
    let best_lin_g = ctx.linear_lut[best_g as usize];
    let best_lin_b = ctx.linear_lut[best_b as usize];

    let err_r_val = lin_r_adj - best_lin_r;
    let err_g_val = lin_g_adj - best_lin_g;
    let err_b_val = lin_b_adj - best_lin_b;

    (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val)
}

/// Generic standard (left-to-right) dithering with any RGB kernel.
/// Processes seeding rows/columns plus real image, all left-to-right.
#[inline]
fn dither_standard_rgb<K: RgbKernel>(
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
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    // Processing area: seeding rows + real rows, seeding cols + real cols + seeding cols
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;  // skip left overshoot

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;

            // Get RGB values (handles seeding coordinate mapping)
            let (r_val, g_val, b_val) = get_seeding_rgb(
                r_channel, g_channel, b_channel, width, px, y, reach
            );

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

            // Only write output for real image pixels (not seeding)
            let in_real_y = y >= reach;
            let in_real_x = px >= reach && px < reach + width;
            if in_real_y && in_real_x {
                let real_y = y - reach;
                let real_x = px - reach;
                let idx = real_y * width + real_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
            }

            K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

/// Generic serpentine dithering with any RGB kernel.
/// Alternates scan direction each row to reduce diagonal banding.
/// Processes seeding rows/columns plus real image.
#[inline]
fn dither_serpentine_rgb<K: RgbKernel>(
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
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for px in (0..process_width).rev() {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb(
                    r_channel, g_channel, b_channel, width, px, y, reach
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let real_y = y - reach;
                    let real_x = px - reach;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                K::apply_rtl(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
            }
        } else {
            // Left-to-right on even rows
            for px in 0..process_width {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb(
                    r_channel, g_channel, b_channel, width, px, y, reach
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let real_y = y - reach;
                    let real_x = px - reach;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                K::apply_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
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

            let (r_val, g_val, b_val) = get_seeding_rgb(
                r_channel, g_channel, b_channel, width, px, y, reach
            );

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

            let in_real_y = y >= reach;
            let in_real_x = px >= reach && px < reach + width;
            if in_real_y && in_real_x {
                let real_y = y - reach;
                let real_x = px - reach;
                let idx = real_y * width + real_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
            }

            // Extract 3 bits for per-channel kernel selection (use image coords for hash)
            let img_x = px.saturating_sub(reach);
            let img_y = y.saturating_sub(reach);
            let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
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
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb(
                    r_channel, g_channel, b_channel, width, px, y, reach
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let real_y = y - reach;
                    let real_x = px - reach;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb(
                    r_channel, g_channel, b_channel, width, px, y, reach
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let real_y = y - reach;
                    let real_x = px - reach;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
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
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        let img_y = y.saturating_sub(reach);
        let row_hash = wang_hash((img_y as u32) ^ hashed_seed);
        let is_rtl = row_hash & 1 == 1;

        if is_rtl {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb(
                    r_channel, g_channel, b_channel, width, px, y, reach
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let real_y = y - reach;
                    let real_x = px - reach;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let img_x = px.saturating_sub(reach);
                let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb(
                    r_channel, g_channel, b_channel, width, px, y, reach
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let real_y = y - reach;
                    let real_x = px - reach;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let img_x = px.saturating_sub(reach);
                let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

// ============================================================================
// Mixed H2 (second-order kernel) RGB dithering
// ============================================================================

/// Get RGB values for H2 processing coordinate, handling seeding area mapping.
/// H2 uses SEED=16 instead of REACH for seeding width.
#[inline]
fn get_seeding_rgb_h2(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    px: usize,  // processing x (0..seed + real_width + seed)
    py: usize,  // processing y (0..seed + real_height)
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

/// Mixed H2 kernel dithering for colorspace-aware RGB, LTR only.
/// Uses FS² and JJN² kernels with per-channel hash-based selection.
/// Requires wider buffer (REACH=4, SEED=16) due to negative weights and larger footprint.
#[inline]
fn dither_mixed_h2_standard_rgb(
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    // bx_start = overshoot (reach) — skip it, start at seeding
    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;

            // Get RGB values (handles seeding coordinate mapping)
            let (r_val, g_val, b_val) = get_seeding_rgb_h2(
                r_channel, g_channel, b_channel, width, px, y, seed
            );

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

            // Only write output for real image pixels (not seeding)
            let in_real_y = y >= seed;
            let in_real_x = px >= seed && px < seed + width;
            if in_real_y && in_real_x {
                let real_y = y - seed;
                let real_x = px - seed;
                let idx = real_y * width + real_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
            }

            // Per-channel kernel selection using hash bits
            let img_x = px.wrapping_sub(seed);
            let img_y = y.wrapping_sub(seed);
            let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            apply_h2_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
        }
        if let Some(ref mut cb) = progress {
            if y >= seed {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

/// Mixed H2 dithering with serpentine scanning for RGB.
/// Uses precomputed second-order kernels (FS² and JJN²) selected per-pixel via hash.
/// Alternates scan direction per row (even=LTR, odd=RTL).
fn dither_mixed_h2_serpentine_rgb(
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for px in (0..process_width).rev() {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb_h2(
                    r_channel, g_channel, b_channel, width, px, y, seed
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= seed;
                let in_real_x = px >= seed && px < seed + width;
                if in_real_y && in_real_x {
                    let real_y = y - seed;
                    let real_x = px - seed;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let img_x = px.wrapping_sub(seed);
                let img_y = y.wrapping_sub(seed);
                let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_h2_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            // Left-to-right on even rows
            for px in 0..process_width {
                let bx = bx_start + px;

                let (r_val, g_val, b_val) = get_seeding_rgb_h2(
                    r_channel, g_channel, b_channel, width, px, y, seed
                );

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_with_rgb(ctx, r_val, g_val, b_val, err_r, err_g, err_b, bx, y);

                let in_real_y = y >= seed;
                let in_real_x = px >= seed && px < seed + width;
                if in_real_y && in_real_x {
                    let real_y = y - seed;
                    let real_x = px - seed;
                    let idx = real_y * width + real_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let img_x = px.wrapping_sub(seed);
                let img_y = y.wrapping_sub(seed);
                let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_h2_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= seed {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

// ============================================================================
// Main dithering implementation
// ============================================================================

/// Context for pixel processing, containing pre-computed values
struct DitherContext<'a> {
    quant_r: &'a QuantLevelParams,
    quant_g: &'a QuantLevelParams,
    quant_b: &'a QuantLevelParams,
    linear_lut: &'a [f32; 256],
    lab_lut: &'a Option<Vec<LabValue>>,
    space: PerceptualSpace,
    /// Apply gamut overshoot penalty to discourage choices that push error
    /// diffusion outside the representable color gamut.
    overshoot_penalty: bool,
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
        None,
    )
}

/// Color space aware dithering with selectable algorithm and scanning mode.
///
/// Uses perceptual color space (Lab or OkLab) for finding the best quantization
/// candidate, and accumulates/diffuses error in linear RGB space for physically
/// correct light mixing. Processes all three channels jointly.
///
/// This is a convenience wrapper that enables overshoot penalty by default.
/// Use `colorspace_aware_dither_rgb_with_options` for full control.
///
/// Args:
///     r_channel, g_channel, b_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Bit depth for each channel (1-8)
///     space: Perceptual color space for distance calculation
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
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
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    colorspace_aware_dither_rgb_with_options(
        r_channel, g_channel, b_channel,
        width, height,
        bits_r, bits_g, bits_b,
        space, mode, seed,
        true, // overshoot_penalty enabled by default
        progress,
    )
}

/// Color space aware dithering with full options control.
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
///     overshoot_penalty: If true, penalize quantization choices that would push
///         error diffusion outside the representable color gamut (recommended)
///     progress: Optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out): Output channels as u8
pub fn colorspace_aware_dither_rgb_with_options(
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
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let quant_r = QuantLevelParams::new(bits_r);
    let quant_g = QuantLevelParams::new(bits_g);
    let quant_b = QuantLevelParams::new(bits_b);

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
        overshoot_penalty,
    };

    let pixels = width * height;

    // H2 needs different buffer dimensions (REACH=4, SEED=16), handle as early return
    if mode == DitherMode::MixedH2Standard || mode == DitherMode::MixedH2Serpentine {
        let h2_reach = H2_REACH;
        let h2_seed = H2_SEED;
        let buf_width = h2_reach + h2_seed + width + h2_seed + h2_reach; // 4+16+W+16+4 = W+40
        let buf_height = h2_seed + height + h2_reach;                     // 16+H+4 = H+20

        let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

        let mut r_out = vec![0u8; pixels];
        let mut g_out = vec![0u8; pixels];
        let mut b_out = vec![0u8; pixels];

        let hashed_seed = lowbias32(seed);

        if mode == DitherMode::MixedH2Serpentine {
            dither_mixed_h2_serpentine_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, hashed_seed, progress,
            );
        } else {
            dither_mixed_h2_standard_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, hashed_seed, progress,
            );
        }

        return (r_out, g_out, b_out);
    }

    // Use JJN reach (larger) for all modes to accommodate both kernels
    // Buffer layout: [overshoot][seeding][real][seeding][overshoot]
    let reach = <JarvisJudiceNinke as RgbKernel>::REACH;
    let buf_width = reach * 4 + width;  // overshoot + seeding + real + seeding + overshoot
    let buf_height = reach * 2 + height;  // seeding + real + overshoot

    // Error buffers in linear RGB space
    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

    // Output buffers
    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    // Dispatch to appropriate generic scan function
    // Note: We move `progress` into the called function since only one match arm executes
    match mode {
        DitherMode::None => {
            dither_standard_rgb::<NoneKernel>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_rgb::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_rgb::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedStandard | DitherMode::MixedWangStandard | DitherMode::MixedLowbiasOldStandard | DitherMode::MixedH2Standard => {
            dither_mixed_standard_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine | DitherMode::MixedWangSerpentine | DitherMode::MixedLowbiasOldSerpentine | DitherMode::MixedH2Serpentine => {
            dither_mixed_serpentine_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_rgb(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::OstromoukhovStandard => {
            dither_standard_rgb::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_serpentine_rgb::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        // Zhou-Fang: fall back to Ostromoukhov for colorspace-aware dithering
        DitherMode::ZhouFangStandard => {
            dither_standard_rgb::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::ZhouFangSerpentine => {
            dither_serpentine_rgb::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        // Ulichney: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::UlichneyStandard | DitherMode::UlichneyWeightStandard => {
            dither_standard_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::UlichneySerpentine | DitherMode::UlichneyWeightSerpentine => {
            dither_serpentine_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        // FS+TPDF: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::FsTpdfStandard => {
            dither_standard_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
        DitherMode::FsTpdfSerpentine => {
            dither_serpentine_rgb::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, progress,
            );
        }
    }

    (r_out, g_out, b_out)
}

// ============================================================================
// Pixel4 convenience wrappers
// ============================================================================

use crate::color::interleave_rgb_u8;
use crate::pixel::{pixels_to_channels, Pixel4};

/// Color-aware dither for Pixel4 array (sRGB 0-255 range) to separate RGB channels.
///
/// This is a convenience wrapper that extracts channels, performs color-aware
/// dithering with joint RGB processing, and returns separate channel outputs.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range
///     width, height: image dimensions
///     bits_r, bits_g, bits_b: output bit depth per channel (1-8)
///     space: perceptual color space for distance calculation
///     mode: dither algorithm and scan pattern
///     seed: random seed for mixed modes
///     progress: optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Tuple of (R, G, B) u8 vectors
pub fn colorspace_aware_dither_rgb_channels(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (r, g, b) = pixels_to_channels(pixels);
    colorspace_aware_dither_rgb_with_options(&r, &g, &b, width, height, bits_r, bits_g, bits_b, space, mode, seed, overshoot_penalty, progress)
}

/// Color-aware dither for Pixel4 array (sRGB 0-255 range) to interleaved u8.
///
/// This is a convenience wrapper for 8-bit RGB output that extracts channels,
/// performs color-aware dithering with joint RGB processing, and interleaves
/// the result.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range
///     width, height: image dimensions
///     space: perceptual color space for distance calculation
///     mode: dither algorithm and scan pattern
///     seed: random seed for mixed modes
///     progress: optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     Interleaved RGB u8 data (RGBRGB...)
pub fn colorspace_aware_dither_rgb_interleaved(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let (r, g, b) = pixels_to_channels(pixels);
    let (r_u8, g_u8, b_u8) = colorspace_aware_dither_rgb_with_mode(&r, &g, &b, width, height, 8, 8, 8, space, mode, seed, progress);
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
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

        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::LabCIE76);

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

        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 2, 2, 2, PerceptualSpace::LabCIE76);

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

        let (r_lab, g_lab, b_lab) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::LabCIE76);
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

        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76);

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
        let (r_out, g_out, b_out) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::LabCIE76);

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
                &r, &g, &b, 10, 10, 2, 2, 2, PerceptualSpace::LabCIE76, mode, 42, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
        );
        let (r_serp, g_serp, b_serp) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::Serpentine, 0, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
        );
        let (r_jjn, g_jjn, b_jjn) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::JarvisStandard, 0, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
        );
        let (r_jjn, g_jjn, b_jjn) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::JarvisStandard, 0, None
        );
        let (r_mix, g_mix, b_mix) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedStandard, 42, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedStandard, 42, None
        );
        let (r2, g2, b2) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedStandard, 42, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedStandard, 42, None
        );
        let (r2, g2, b2) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedStandard, 99, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedStandard, 42, None
        );
        let (r_serp, g_serp, b_serp) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedSerpentine, 42, None
        );
        let (r_rand, g_rand, b_rand) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::MixedRandom, 42, None
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
        let (r1, g1, b1) = colorspace_aware_dither_rgb(&r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76);
        // New API with Standard mode
        let (r2, g2, b2) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::Standard, 0, None
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
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::JarvisStandard, 0, None
        );
        let (r_serp, g_serp, b_serp) = colorspace_aware_dither_rgb_with_mode(
            &r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::LabCIE76, DitherMode::JarvisSerpentine, 0, None
        );

        // JJN standard and serpentine should produce different results
        let std_combined: Vec<u8> = r_std.iter().chain(g_std.iter()).chain(b_std.iter()).copied().collect();
        let serp_combined: Vec<u8> = r_serp.iter().chain(g_serp.iter()).chain(b_serp.iter()).copied().collect();
        assert_ne!(std_combined, serp_combined);
    }
}
