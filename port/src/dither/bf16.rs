/// Color space aware RGBA dithering for f32 to bf16 conversion.
///
/// Extends RGB dithering to handle alpha channel with proper error propagation:
/// - Alpha channel is dithered first using standard single-channel dithering
/// - RGB channels are then dithered with alpha-aware error diffusion
///
/// Supports both linear and sRGB working spaces:
/// - Linear: error diffusion and quantization happen in linear space
/// - sRGB: error diffusion in linear space, quantization in sRGB space

use half::bf16;

use crate::color::{linear_to_srgb_single, srgb_to_linear_single};
use crate::color_distance::perceptual_distance_sq;
use super::common::{
    apply_mixed_kernel_rgb, apply_single_channel_kernel, gamut_overshoot_penalty,
    linear_rgb_to_perceptual, linear_rgb_to_perceptual_clamped, wang_hash, DitherMode,
    FloydSteinberg, JarvisJudiceNinke, NoneKernel, Ostromoukhov, PerceptualSpace, RgbKernel,
    SingleChannelKernel,
};

// ============================================================================
// Working space enum
// ============================================================================

/// Specifies the color space for input/output values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16WorkingSpace {
    /// Input and output are linear RGB (error diffusion native)
    Linear,
    /// Input and output are sRGB (error diffusion happens in linear space internally)
    Srgb,
}

// ============================================================================
// bf16 quantization utilities
// ============================================================================

/// Get the next representable bf16 toward +∞
#[inline]
fn next_up_bf16(x: bf16) -> bf16 {
    let bits = x.to_bits();
    // NaN or +inf: return as-is
    if x.is_nan() || bits == 0x7F80 {
        return x;
    }
    // -0.0 → smallest positive subnormal
    if bits == 0x8000 {
        return bf16::from_bits(0x0001);
    }
    if bits & 0x8000 == 0 {
        // Positive: increment moves away from zero (larger)
        bf16::from_bits(bits + 1)
    } else {
        // Negative: decrement moves toward zero (larger)
        bf16::from_bits(bits - 1)
    }
}

/// Get the next representable bf16 toward -∞
#[inline]
fn next_down_bf16(x: bf16) -> bf16 {
    let bits = x.to_bits();
    // NaN or -inf: return as-is
    if x.is_nan() || bits == 0xFF80 {
        return x;
    }
    // +0.0 → smallest negative subnormal
    if bits == 0x0000 {
        return bf16::from_bits(0x8001);
    }
    if bits & 0x8000 == 0 {
        // Positive: decrement moves toward zero (smaller)
        bf16::from_bits(bits - 1)
    } else {
        // Negative: increment moves away from zero (smaller / more negative)
        bf16::from_bits(bits + 1)
    }
}

/// Get floor and ceiling bf16 candidates for a given f32 value.
/// Since half 2.3 doesn't have rounding modes, we compute manually.
#[inline]
fn get_bf16_bounds(value: f32) -> (bf16, bf16) {
    // Handle NaN - preserve it
    if value.is_nan() {
        let nan = bf16::from_f32(value);
        return (nan, nan);
    }

    // Round to nearest first
    let rounded = bf16::from_f32(value);
    let rounded_f32 = rounded.to_f32();

    if rounded_f32 == value {
        // Exactly representable
        (rounded, rounded)
    } else if rounded_f32 > value {
        // rounded is ceil, need floor
        (next_down_bf16(rounded), rounded)
    } else {
        // rounded is floor, need ceil
        (rounded, next_up_bf16(rounded))
    }
}

/// Convert sRGB f32 (0-1) to linear f32
#[inline]
fn srgb_to_linear(srgb: f32) -> f32 {
    srgb_to_linear_single(srgb)
}

/// Convert linear f32 to sRGB f32 (0-1)
#[inline]
#[allow(dead_code)]
fn linear_to_srgb(linear: f32) -> f32 {
    linear_to_srgb_single(linear)
}

/// Convert linear f32 to sRGB f32, handling negative values (for error diffusion)
#[inline]
fn linear_to_srgb_signed(linear: f32) -> f32 {
    if linear >= 0.0 {
        linear_to_srgb_single(linear)
    } else {
        -linear_to_srgb_single(-linear)
    }
}

/// Convert sRGB f32 to linear f32, handling negative values (for error diffusion)
#[inline]
fn srgb_to_linear_signed(srgb: f32) -> f32 {
    if srgb >= 0.0 {
        srgb_to_linear_single(srgb)
    } else {
        -srgb_to_linear_single(-srgb)
    }
}

// ============================================================================
// Edge seeding helper functions
// ============================================================================

/// Get RGB channel values for edge seeding.
/// Maps seeding coordinates to edge pixels of the real image.
#[inline]
fn get_seeding_rgb(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    px: usize,
    py: usize,
    reach: usize,
) -> (f32, f32, f32) {
    let img_y = if py < reach { 0 } else { py - reach };
    let img_x = if px < reach {
        0
    } else if px >= reach + width {
        width - 1
    } else {
        px - reach
    };
    let idx = img_y * width + img_x;
    (r_channel[idx], g_channel[idx], b_channel[idx])
}

/// Get alpha channel value for edge seeding.
/// Maps seeding coordinates to edge pixels of the real image.
#[inline]
fn get_seeding_alpha(alpha: &[f32], width: usize, px: usize, py: usize, reach: usize) -> f32 {
    let img_y = if py < reach { 0 } else { py - reach };
    let img_x = if px < reach {
        0
    } else if px >= reach + width {
        width - 1
    } else {
        px - reach
    };
    let idx = img_y * width + img_x;
    alpha[idx]
}

/// Get dithered alpha value for edge seeding.
/// Maps seeding coordinates to edge pixels of the real dithered alpha.
#[inline]
fn get_seeding_alpha_dithered(alpha_dithered: &[bf16], width: usize, px: usize, py: usize, reach: usize) -> bf16 {
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
// Alpha channel dithering for bf16
// ============================================================================

/// Dither a single alpha channel from f32 to bf16.
/// Alpha is always treated as linear.
fn dither_alpha_bf16<K: SingleChannelKernel>(
    alpha: &[f32],
    width: usize,
    height: usize,
    serpentine: bool,
) -> Vec<bf16> {
    let pixels = width * height;
    let reach = K::REACH;

    // Buffer layout for width: [overshoot][seeding][real image][seeding][overshoot]
    // Buffer layout for height: [seeding][real image][overshoot]
    let buf_width = reach * 4 + width;
    let buf_height = reach * 2 + height;

    let mut err: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut out = vec![bf16::ZERO; pixels];

    // Process area: [seeding][real image] for height, [seeding][real][seeding] for width
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        let is_rtl = serpentine && y % 2 == 1;

        if is_rtl {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let alpha_value = get_seeding_alpha(alpha, width, px, y, reach);

                // Read input and add accumulated error
                let adjusted = alpha_value + err[y][bx];

                // Get bf16 floor/ceil bounds
                let (floor_bf16, ceil_bf16) = get_bf16_bounds(adjusted);
                let floor_val = floor_bf16.to_f32();
                let ceil_val = ceil_bf16.to_f32();

                // Pick closer candidate
                let (best_bf16, best_val) = if (adjusted - floor_val).abs() <= (adjusted - ceil_val).abs() {
                    (floor_bf16, floor_val)
                } else {
                    (ceil_bf16, ceil_val)
                };

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    out[idx] = best_bf16;
                }

                // Compute and diffuse error
                let err_val = adjusted - best_val;
                apply_single_channel_kernel(&mut err, bx, y, err_val, K::REACH >= 2, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let alpha_value = get_seeding_alpha(alpha, width, px, y, reach);

                // Read input and add accumulated error
                let adjusted = alpha_value + err[y][bx];

                // Get bf16 floor/ceil bounds
                let (floor_bf16, ceil_bf16) = get_bf16_bounds(adjusted);
                let floor_val = floor_bf16.to_f32();
                let ceil_val = ceil_bf16.to_f32();

                // Pick closer candidate
                let (best_bf16, best_val) = if (adjusted - floor_val).abs() <= (adjusted - ceil_val).abs() {
                    (floor_bf16, floor_val)
                } else {
                    (ceil_bf16, ceil_val)
                };

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    out[idx] = best_bf16;
                }

                // Compute and diffuse error
                let err_val = adjusted - best_val;
                apply_single_channel_kernel(&mut err, bx, y, err_val, K::REACH >= 2, false);
            }
        }
    }

    out
}

/// Dither alpha channel with mode selection
fn dither_alpha_bf16_with_mode(
    alpha: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
) -> Vec<bf16> {
    match mode {
        DitherMode::None => {
            // Direct conversion without dithering
            alpha.iter().map(|&v| bf16::from_f32(v)).collect()
        }
        DitherMode::Standard => {
            dither_alpha_bf16::<FloydSteinberg>(alpha, width, height, false)
        }
        DitherMode::Serpentine => {
            dither_alpha_bf16::<FloydSteinberg>(alpha, width, height, true)
        }
        DitherMode::JarvisStandard => {
            dither_alpha_bf16::<JarvisJudiceNinke>(alpha, width, height, false)
        }
        DitherMode::JarvisSerpentine => {
            dither_alpha_bf16::<JarvisJudiceNinke>(alpha, width, height, true)
        }
        DitherMode::MixedStandard | DitherMode::MixedSerpentine | DitherMode::MixedRandom => {
            // For mixed modes, use JJN padding but random kernel selection
            dither_alpha_bf16_mixed(alpha, width, height, mode, seed)
        }
        DitherMode::OstromoukhovStandard => {
            dither_alpha_bf16::<Ostromoukhov>(alpha, width, height, false)
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_alpha_bf16::<Ostromoukhov>(alpha, width, height, true)
        }
    }
}

fn dither_alpha_bf16_mixed(
    alpha: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
) -> Vec<bf16> {
    let pixels = width * height;
    let reach = <JarvisJudiceNinke as RgbKernel>::REACH;

    // Buffer layout for width: [overshoot][seeding][real image][seeding][overshoot]
    // Buffer layout for height: [seeding][real image][overshoot]
    let buf_width = reach * 4 + width;
    let buf_height = reach * 2 + height;

    let mut err: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut out = vec![bf16::ZERO; pixels];
    let hashed_seed = wang_hash(seed);

    // Process area: [seeding][real image] for height, [seeding][real][seeding] for width
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        let is_rtl = match mode {
            DitherMode::MixedSerpentine => y % 2 == 1,
            DitherMode::MixedRandom => wang_hash((y as u32) ^ hashed_seed) & 1 == 1,
            _ => false,
        };

        if is_rtl {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let alpha_value = get_seeding_alpha(alpha, width, px, y, reach);

                let adjusted = alpha_value + err[y][bx];
                let (floor_bf16, ceil_bf16) = get_bf16_bounds(adjusted);
                let floor_val = floor_bf16.to_f32();
                let ceil_val = ceil_bf16.to_f32();

                let (best_bf16, best_val) = if (adjusted - floor_val).abs() <= (adjusted - ceil_val).abs() {
                    (floor_bf16, floor_val)
                } else {
                    (ceil_bf16, ceil_val)
                };

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    out[idx] = best_bf16;
                }

                let err_val = adjusted - best_val;
                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_single_channel_kernel(&mut err, bx, y, err_val, use_jjn, is_rtl);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let alpha_value = get_seeding_alpha(alpha, width, px, y, reach);

                let adjusted = alpha_value + err[y][bx];
                let (floor_bf16, ceil_bf16) = get_bf16_bounds(adjusted);
                let floor_val = floor_bf16.to_f32();
                let ceil_val = ceil_bf16.to_f32();

                let (best_bf16, best_val) = if (adjusted - floor_val).abs() <= (adjusted - ceil_val).abs() {
                    (floor_bf16, floor_val)
                } else {
                    (ceil_bf16, ceil_val)
                };

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    out[idx] = best_bf16;
                }

                let err_val = adjusted - best_val;
                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 != 0;
                apply_single_channel_kernel(&mut err, bx, y, err_val, use_jjn, is_rtl);
            }
        }
    }

    out
}

// ============================================================================
// Alpha-aware RGB dithering pixel processing for bf16
// ============================================================================

/// Convert a working-space bf16 value to linear f32 for perceptual calculations
#[inline]
fn bf16_to_linear(value: bf16, working_space: Bf16WorkingSpace) -> f32 {
    let v = value.to_f32();
    match working_space {
        Bf16WorkingSpace::Linear => v,
        Bf16WorkingSpace::Srgb => srgb_to_linear_signed(v),
    }
}

/// Process a single pixel with alpha-aware error diffusion for bf16 output.
/// Takes pre-fetched RGB and alpha values for seeding support.
///
/// Returns (best_r, best_g, best_b, err_r, err_g, err_b) where:
/// - best_* are bf16 values in the working space
/// - err_* are f32 values in linear space
#[inline]
fn process_pixel_bf16_with_values(
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    r_val: f32,
    g_val: f32,
    b_val: f32,
    alpha_bf16: bf16,
    err_r: &[Vec<f32>],
    err_g: &[Vec<f32>],
    err_b: &[Vec<f32>],
    bx: usize,
    y: usize,
    overshoot_penalty: bool,
) -> (bf16, bf16, bf16, f32, f32, f32) {
    // Get alpha (normalized)
    let alpha = alpha_bf16.to_f32().clamp(0.0, 1.0);

    // 1. Read accumulated error (in linear space)
    let err_r_in = err_r[y][bx];
    let err_g_in = err_g[y][bx];
    let err_b_in = err_b[y][bx];

    // 2. Convert input to linear space
    let (lin_r_orig, lin_g_orig, lin_b_orig) = match working_space {
        Bf16WorkingSpace::Linear => (r_val, g_val, b_val),
        Bf16WorkingSpace::Srgb => (
            srgb_to_linear(r_val),
            srgb_to_linear(g_val),
            srgb_to_linear(b_val),
        ),
    };

    // 3. Add accumulated error in linear space (skip for fully transparent pixels)
    let (lin_r_adj, lin_g_adj, lin_b_adj) = if alpha == 0.0 {
        (lin_r_orig, lin_g_orig, lin_b_orig)
    } else {
        (lin_r_orig + err_r_in, lin_g_orig + err_g_in, lin_b_orig + err_b_in)
    };

    // 4. Convert adjusted values to working space for quantization
    let (ws_r_adj, ws_g_adj, ws_b_adj) = match working_space {
        Bf16WorkingSpace::Linear => (lin_r_adj, lin_g_adj, lin_b_adj),
        Bf16WorkingSpace::Srgb => (
            linear_to_srgb_signed(lin_r_adj),
            linear_to_srgb_signed(lin_g_adj),
            linear_to_srgb_signed(lin_b_adj),
        ),
    };

    // 5. Get bf16 floor/ceil bounds for each channel
    let (r_floor, r_ceil) = get_bf16_bounds(ws_r_adj);
    let (g_floor, g_ceil) = get_bf16_bounds(ws_g_adj);
    let (b_floor, b_ceil) = get_bf16_bounds(ws_b_adj);

    // 6. Convert target to perceptual space (unclamped for true distance)
    let lab_target = linear_rgb_to_perceptual(space, lin_r_adj, lin_g_adj, lin_b_adj);

    // 7. Search all 8 candidate combinations for best quantization
    let r_candidates = [r_floor, r_ceil];
    let g_candidates = [g_floor, g_ceil];
    let b_candidates = [b_floor, b_ceil];

    let mut best_r = r_floor;
    let mut best_g = g_floor;
    let mut best_b = b_floor;
    let mut best_dist = f32::INFINITY;

    for &r_cand in &r_candidates {
        for &g_cand in &g_candidates {
            for &b_cand in &b_candidates {
                // Convert candidate to linear space
                let r_lin = bf16_to_linear(r_cand, working_space);
                let g_lin = bf16_to_linear(g_cand, working_space);
                let b_lin = bf16_to_linear(b_cand, working_space);

                // Convert to perceptual space (clamped for candidate)
                let lab_cand = linear_rgb_to_perceptual_clamped(space, r_lin, g_lin, b_lin);

                let base_dist = perceptual_distance_sq(
                    space,
                    lab_target.0, lab_target.1, lab_target.2,
                    lab_cand.0, lab_cand.1, lab_cand.2,
                );

                // Apply gamut overshoot penalty if enabled
                let dist = if overshoot_penalty {
                    let penalty = gamut_overshoot_penalty(
                        lin_r_adj, lin_g_adj, lin_b_adj,
                        r_lin, g_lin, b_lin,
                    );
                    base_dist * penalty
                } else {
                    base_dist
                };

                if dist < best_dist {
                    best_dist = dist;
                    best_r = r_cand;
                    best_g = g_cand;
                    best_b = b_cand;
                }
            }
        }
    }

    // 8. Compute quantized linear values for error calculation
    let best_lin_r = bf16_to_linear(best_r, working_space);
    let best_lin_g = bf16_to_linear(best_g, working_space);
    let best_lin_b = bf16_to_linear(best_b, working_space);

    // 9. Compute alpha-aware error to diffuse (in linear space)
    // Formula: error = (1 - α) × e_in + α × q_err
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
// Generic scan pattern implementations for bf16
// ============================================================================

#[inline]
fn dither_standard_bf16<K: RgbKernel>(
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    alpha_dithered: &[bf16],
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [bf16],
    g_out: &mut [bf16],
    b_out: &mut [bf16],
    width: usize,
    height: usize,
    reach: usize,
    overshoot_penalty: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
            let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

            // Only write output for real image pixels
            let in_real_y = y >= reach;
            let in_real_x = px >= reach && px < reach + width;
            if in_real_y && in_real_x {
                let idx = (y - reach) * width + (px - reach);
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
fn dither_serpentine_bf16<K: RgbKernel>(
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    alpha_dithered: &[bf16],
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [bf16],
    g_out: &mut [bf16],
    b_out: &mut [bf16],
    width: usize,
    height: usize,
    reach: usize,
    overshoot_penalty: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
                let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                K::apply_rtl(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, r_val, g_val, b_val);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
                let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
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
fn dither_mixed_standard_bf16(
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    alpha_dithered: &[bf16],
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [bf16],
    g_out: &mut [bf16],
    b_out: &mut [bf16],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    overshoot_penalty: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
            let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

            let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

            // Only write output for real image pixels
            let in_real_y = y >= reach;
            let in_real_x = px >= reach && px < reach + width;
            if in_real_y && in_real_x {
                let idx = (y - reach) * width + (px - reach);
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
fn dither_mixed_serpentine_bf16(
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    alpha_dithered: &[bf16],
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [bf16],
    g_out: &mut [bf16],
    b_out: &mut [bf16],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    overshoot_penalty: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
                let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
                let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
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
fn dither_mixed_random_bf16(
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    alpha_dithered: &[bf16],
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    r_out: &mut [bf16],
    g_out: &mut [bf16],
    b_out: &mut [bf16],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    overshoot_penalty: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach; // Skip left overshoot

    for y in 0..process_height {
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_rtl = row_hash & 1 == 1;

        if is_rtl {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
                let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel_rgb(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, pixel_hash, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let (r_val, g_val, b_val) = get_seeding_rgb(r_channel, g_channel, b_channel, width, px, y, reach);
                let alpha_bf16 = get_seeding_alpha_dithered(alpha_dithered, width, px, y, reach);

                let (best_r, best_g, best_b, err_r_val, err_g_val, err_b_val) =
                    process_pixel_bf16_with_values(space, working_space, r_val, g_val, b_val, alpha_bf16, err_r, err_g, err_b, bx, y, overshoot_penalty);

                // Only write output for real image pixels
                let in_real_y = y >= reach;
                let in_real_x = px >= reach && px < reach + width;
                if in_real_y && in_real_x {
                    let idx = (y - reach) * width + (px - reach);
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
// Public API
// ============================================================================

/// Color space aware RGBA dithering from f32 to bf16.
///
/// This is the simplified API that uses Floyd-Steinberg with standard scanning.
/// For other algorithms and scan patterns, use `dither_rgba_bf16_with_mode`.
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32
///     width, height: Image dimensions
///     space: Perceptual color space for RGB distance calculation
///     working_space: Whether input/output are in linear or sRGB space
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as bf16
pub fn dither_rgba_bf16(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    dither_rgba_bf16_with_mode(
        r_channel, g_channel, b_channel, a_channel,
        width, height,
        space,
        working_space,
        DitherMode::Standard,
        0,
        None,
    )
}

/// Color space aware RGBA dithering from f32 to bf16 with selectable algorithm and scanning mode.
///
/// Process:
/// 1. Alpha channel is dithered first using standard single-channel error diffusion
/// 2. RGB channels are then dithered with alpha-aware error propagation
///
/// Error diffusion always happens in linear space for perceptual correctness,
/// regardless of the working space setting.
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32
///     width, height: Image dimensions
///     space: Perceptual color space for RGB distance calculation
///     working_space: Whether input/output are in linear or sRGB space
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as bf16
pub fn dither_rgba_bf16_with_mode(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    dither_rgba_bf16_with_options(
        r_channel, g_channel, b_channel, a_channel,
        width, height, space, working_space, mode, seed, true, progress,
    )
}

/// Color space aware RGBA dithering from f32 to bf16 with full options.
///
/// Same as `dither_rgba_bf16_with_mode` but with additional control over overshoot penalty.
///
/// Args:
///     overshoot_penalty: If true, penalize candidates that would cause the opposing
///         point to fall outside the [0,1]³ RGB cube (reduces color fringing)
pub fn dither_rgba_bf16_with_options(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    mode: DitherMode,
    seed: u32,
    overshoot_penalty: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let pixels = width * height;

    // Step 1: Dither alpha channel first (always linear)
    let alpha_dithered = dither_alpha_bf16_with_mode(a_channel, width, height, mode, seed.wrapping_add(3));

    // Report alpha dithering complete (10% of total progress)
    if let Some(ref mut cb) = progress {
        cb(0.1);
    }

    // Step 2: Set up RGB dithering with alpha awareness
    // Use JJN reach for all modes (largest kernel)
    let reach = <JarvisJudiceNinke as RgbKernel>::REACH;

    // Buffer layout for width: [overshoot][seeding][real image][seeding][overshoot]
    // Buffer layout for height: [seeding][real image][overshoot]
    let buf_width = reach * 4 + width;
    let buf_height = reach * 2 + height;

    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

    let mut r_out = vec![bf16::ZERO; pixels];
    let mut g_out = vec![bf16::ZERO; pixels];
    let mut b_out = vec![bf16::ZERO; pixels];

    let hashed_seed = wang_hash(seed);

    match mode {
        DitherMode::None => {
            dither_standard_bf16::<NoneKernel>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_bf16::<FloydSteinberg>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_bf16::<FloydSteinberg>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_bf16::<JarvisJudiceNinke>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_bf16::<JarvisJudiceNinke>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_bf16(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, overshoot_penalty, progress,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_bf16(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, overshoot_penalty, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_bf16(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, hashed_seed, overshoot_penalty, progress,
            );
        }
        DitherMode::OstromoukhovStandard => {
            dither_standard_bf16::<Ostromoukhov>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_serpentine_bf16::<Ostromoukhov>(
                space, working_space, &alpha_dithered,
                r_channel, g_channel, b_channel,
                &mut err_r, &mut err_g, &mut err_b,
                &mut r_out, &mut g_out, &mut b_out,
                width, height, reach, overshoot_penalty, progress,
            );
        }
    }

    (r_out, g_out, b_out, alpha_dithered)
}

/// Color space aware RGB dithering from f32 to bf16 (no alpha channel).
///
/// Args:
///     r_channel, g_channel, b_channel: Input channels as f32
///     width, height: Image dimensions
///     space: Perceptual color space for RGB distance calculation
///     working_space: Whether input/output are in linear or sRGB space
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out): Output channels as bf16
pub fn dither_rgb_bf16_with_mode(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    dither_rgb_bf16_with_options(
        r_channel, g_channel, b_channel,
        width, height, space, working_space, mode, seed, true, progress,
    )
}

/// Color space aware RGB dithering from f32 to bf16 with full options.
///
/// Same as `dither_rgb_bf16_with_mode` but with additional control over overshoot penalty.
pub fn dither_rgb_bf16_with_options(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    space: PerceptualSpace,
    working_space: Bf16WorkingSpace,
    mode: DitherMode,
    seed: u32,
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    // Create fully opaque alpha channel
    let pixels = width * height;
    let a_channel = vec![1.0f32; pixels];

    let (r_out, g_out, b_out, _) = dither_rgba_bf16_with_options(
        r_channel, g_channel, b_channel, &a_channel,
        width, height,
        space, working_space,
        mode, seed, overshoot_penalty, progress,
    );

    (r_out, g_out, b_out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_bounds() {
        // Test that floor <= value <= ceil
        let test_values = [0.0f32, 0.5, 1.0, -0.5, 0.123456789, 1e-6];
        for v in test_values {
            let (floor, ceil) = get_bf16_bounds(v);
            assert!(floor.to_f32() <= v, "floor({}) = {} > {}", v, floor, v);
            assert!(ceil.to_f32() >= v, "ceil({}) = {} < {}", v, ceil, v);
        }
    }

    #[test]
    fn test_bf16_bounds_exact() {
        // When value is exactly representable, floor == ceil
        let exact = bf16::from_f32(0.5).to_f32();
        let (floor, ceil) = get_bf16_bounds(exact);
        assert_eq!(floor, ceil);
    }

    #[test]
    fn test_bf16_larger_values() {
        // bf16 has same exponent range as f32, so test larger values
        let test_values = [100.0f32, 1000.0, 10000.0, 1e10, 1e30];
        for v in test_values {
            let (floor, ceil) = get_bf16_bounds(v);
            assert!(floor.to_f32() <= v, "floor({}) = {} > {}", v, floor, v);
            assert!(ceil.to_f32() >= v, "ceil({}) = {} < {}", v, ceil, v);
        }
    }

    #[test]
    fn test_bf16_bounds_negative_numbers() {
        // Test negative numbers - the key bug case
        let test_values = [-0.5f32, -1.0, -0.123456789, -0.001, -100.0, -1e10, -1e30];
        for v in test_values {
            let (floor, ceil) = get_bf16_bounds(v);
            assert!(
                floor.to_f32() <= v,
                "floor({}) = {} > {} (bits: floor=0x{:04X})",
                v, floor.to_f32(), v, floor.to_bits()
            );
            assert!(
                ceil.to_f32() >= v,
                "ceil({}) = {} < {} (bits: ceil=0x{:04X})",
                v, ceil.to_f32(), v, ceil.to_bits()
            );
        }
    }

    #[test]
    fn test_bf16_bounds_tiny_negative_rounding_to_minus_zero() {
        // This is the critical bug case: tiny negative value that rounds to -0.0
        let value = -1e-45_f32; // Tiny negative, rounds to -0.0
        let rounded = bf16::from_f32(value);
        assert_eq!(rounded.to_bits(), 0x8000, "Expected -0.0"); // Verify it rounds to -0.0

        let (floor, ceil) = get_bf16_bounds(value);

        // floor should be negative (smallest negative subnormal or less)
        assert!(floor.to_f32() <= value, "floor({}) = {} should be <= {}", value, floor.to_f32(), value);
        assert!(floor.to_bits() >= 0x8000, "floor should be negative, got bits 0x{:04X}", floor.to_bits());

        // ceil should be -0.0 (which equals 0.0)
        assert!(ceil.to_f32() >= value, "ceil({}) = {} should be >= {}", value, ceil.to_f32(), value);
    }

    #[test]
    fn test_bf16_bounds_tiny_positive_rounding_to_zero() {
        // Tiny positive value that rounds to +0.0
        let value = 1e-45_f32; // Tiny positive, rounds to +0.0
        let rounded = bf16::from_f32(value);
        assert_eq!(rounded.to_bits(), 0x0000, "Expected +0.0"); // Verify it rounds to +0.0

        let (floor, ceil) = get_bf16_bounds(value);

        // floor should be +0.0
        assert!(floor.to_f32() <= value, "floor({}) = {} should be <= {}", value, floor.to_f32(), value);

        // ceil should be smallest positive subnormal
        assert!(ceil.to_f32() >= value, "ceil({}) = {} should be >= {}", value, ceil.to_f32(), value);
    }

    #[test]
    fn test_bf16_bounds_zero_exact() {
        // +0.0 is exactly representable
        let (floor, ceil) = get_bf16_bounds(0.0f32);
        assert_eq!(floor, ceil);
        assert_eq!(floor.to_f32(), 0.0);
    }

    #[test]
    fn test_next_up_bf16() {
        // +0.0 -> smallest positive subnormal
        let x = bf16::from_bits(0x0000);
        let up = next_up_bf16(x);
        assert_eq!(up.to_bits(), 0x0001);

        // -0.0 -> smallest positive subnormal (crosses zero boundary)
        let x = bf16::from_bits(0x8000);
        let up = next_up_bf16(x);
        assert_eq!(up.to_bits(), 0x0001);

        // Positive number: increment
        let x = bf16::from_f32(1.0);
        let up = next_up_bf16(x);
        assert!(up.to_f32() > x.to_f32());

        // Negative number: decrement bits (toward zero)
        let x = bf16::from_f32(-1.0);
        let up = next_up_bf16(x);
        assert!(up.to_f32() > x.to_f32());
        assert!(up.to_f32() < 0.0); // Still negative, just closer to zero

        // +inf stays +inf
        let x = bf16::from_bits(0x7F80);
        let up = next_up_bf16(x);
        assert_eq!(up.to_bits(), 0x7F80);
    }

    #[test]
    fn test_next_down_bf16() {
        // +0.0 -> smallest negative subnormal (crosses zero boundary)
        let x = bf16::from_bits(0x0000);
        let down = next_down_bf16(x);
        assert_eq!(down.to_bits(), 0x8001);

        // -0.0 -> smallest negative subnormal
        let x = bf16::from_bits(0x8000);
        let down = next_down_bf16(x);
        assert_eq!(down.to_bits(), 0x8001);

        // Positive number: decrement
        let x = bf16::from_f32(1.0);
        let down = next_down_bf16(x);
        assert!(down.to_f32() < x.to_f32());

        // Negative number: increment bits (away from zero)
        let x = bf16::from_f32(-1.0);
        let down = next_down_bf16(x);
        assert!(down.to_f32() < x.to_f32()); // More negative

        // -inf stays -inf
        let x = bf16::from_bits(0xFF80);
        let down = next_down_bf16(x);
        assert_eq!(down.to_bits(), 0xFF80);
    }

    #[test]
    fn test_bf16_bounds_infinity() {
        // +inf
        let (floor, ceil) = get_bf16_bounds(f32::INFINITY);
        assert!(floor.is_infinite() && floor.to_f32() > 0.0);
        assert!(ceil.is_infinite() && ceil.to_f32() > 0.0);

        // -inf
        let (floor, ceil) = get_bf16_bounds(f32::NEG_INFINITY);
        assert!(floor.is_infinite() && floor.to_f32() < 0.0);
        assert!(ceil.is_infinite() && ceil.to_f32() < 0.0);
    }

    #[test]
    fn test_bf16_bounds_nan() {
        let (floor, ceil) = get_bf16_bounds(f32::NAN);
        assert!(floor.is_nan());
        assert!(ceil.is_nan());
    }

    #[test]
    fn test_bf16_bounds_subnormal() {
        // Test values in subnormal range
        let smallest_positive_subnormal = bf16::from_bits(0x0001).to_f32();
        let (floor, ceil) = get_bf16_bounds(smallest_positive_subnormal);
        assert_eq!(floor, ceil); // Exactly representable

        // Value between 0 and smallest subnormal
        let tiny = smallest_positive_subnormal / 2.0;
        let (floor, ceil) = get_bf16_bounds(tiny);
        assert!(floor.to_f32() <= tiny);
        assert!(ceil.to_f32() >= tiny);
    }

    #[test]
    fn test_bf16_bounds_adjacency() {
        // Verify floor and ceil are adjacent (or equal)
        let test_values = [0.123456789f32, -0.987654321, 1.5, -1.5, 0.001, -0.001, 1000.0, -1000.0];
        for v in test_values {
            let (floor, ceil) = get_bf16_bounds(v);
            if floor != ceil {
                // They should be adjacent bf16 values
                let floor_bits = floor.to_bits();
                let ceil_bits = ceil.to_bits();

                // For positive floor: ceil = floor + 1 bit
                // For negative ceil near zero: trickier
                if floor.to_f32() >= 0.0 {
                    assert_eq!(ceil_bits, floor_bits + 1,
                        "For v={}, floor=0x{:04X}, ceil=0x{:04X} are not adjacent",
                        v, floor_bits, ceil_bits);
                } else if ceil.to_f32() <= 0.0 {
                    // Both negative: floor has higher bit value
                    assert_eq!(floor_bits, ceil_bits + 1,
                        "For v={}, floor=0x{:04X}, ceil=0x{:04X} are not adjacent",
                        v, floor_bits, ceil_bits);
                }
                // else: floor negative, ceil positive/zero - crossing zero boundary
            }
        }
    }

    #[test]
    fn test_dither_rgba_bf16_basic_linear() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();
        let a: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();

        let (r_out, g_out, b_out, a_out) = dither_rgba_bf16(
            &r, &g, &b, &a, 10, 10, PerceptualSpace::OkLab, Bf16WorkingSpace::Linear
        );

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
        assert_eq!(a_out.len(), 100);
    }

    #[test]
    fn test_dither_rgba_bf16_basic_srgb() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();
        let a: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();

        let (r_out, g_out, b_out, a_out) = dither_rgba_bf16(
            &r, &g, &b, &a, 10, 10, PerceptualSpace::OkLab, Bf16WorkingSpace::Srgb
        );

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
        assert_eq!(a_out.len(), 100);
    }

    #[test]
    fn test_bf16_fully_transparent_passes_error() {
        let r: Vec<f32> = vec![0.5; 100];
        let g: Vec<f32> = vec![0.5; 100];
        let b: Vec<f32> = vec![0.5; 100];
        let a: Vec<f32> = vec![0.0; 100]; // Fully transparent

        let (r_out, g_out, b_out, a_out) = dither_rgba_bf16(
            &r, &g, &b, &a, 10, 10, PerceptualSpace::OkLab, Bf16WorkingSpace::Linear
        );

        // Alpha should be 0
        for &v in &a_out {
            assert_eq!(v.to_f32(), 0.0, "Transparent alpha should dither to 0");
        }

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_bf16_fully_opaque() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();
        let a: Vec<f32> = vec![1.0; 100]; // Fully opaque

        let (r_out, g_out, b_out, a_out) = dither_rgba_bf16(
            &r, &g, &b, &a, 10, 10, PerceptualSpace::OkLab, Bf16WorkingSpace::Linear
        );

        // Alpha should be 1.0
        for &v in &a_out {
            assert_eq!(v.to_f32(), 1.0, "Opaque alpha should dither to 1.0");
        }

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_bf16_all_modes() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();
        let a: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 / 100.0).collect();

        let modes = [
            DitherMode::None,
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
        ];

        for mode in modes {
            let (r_out, g_out, b_out, a_out) = dither_rgba_bf16_with_mode(
                &r, &g, &b, &a, 10, 10,
                PerceptualSpace::OkLab, Bf16WorkingSpace::Linear,
                mode, 42, None
            );

            assert_eq!(r_out.len(), 100, "Mode {:?} produced wrong R length", mode);
            assert_eq!(g_out.len(), 100, "Mode {:?} produced wrong G length", mode);
            assert_eq!(b_out.len(), 100, "Mode {:?} produced wrong B length", mode);
            assert_eq!(a_out.len(), 100, "Mode {:?} produced wrong A length", mode);
        }
    }

    #[test]
    fn test_bf16_mixed_deterministic() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();
        let a: Vec<f32> = (0..100).map(|i| ((i + 50) % 100) as f32 / 100.0).collect();

        let result1 = dither_rgba_bf16_with_mode(
            &r, &g, &b, &a, 10, 10,
            PerceptualSpace::OkLab, Bf16WorkingSpace::Linear,
            DitherMode::MixedStandard, 42, None
        );
        let result2 = dither_rgba_bf16_with_mode(
            &r, &g, &b, &a, 10, 10,
            PerceptualSpace::OkLab, Bf16WorkingSpace::Linear,
            DitherMode::MixedStandard, 42, None
        );

        assert_eq!(result1.0, result2.0);
        assert_eq!(result1.1, result2.1);
        assert_eq!(result1.2, result2.2);
        assert_eq!(result1.3, result2.3);
    }

    #[test]
    fn test_bf16_output_within_bounds() {
        // Test that dithered output is reasonably close to input
        // bf16 has less precision than f16, so allow more tolerance
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let a: Vec<f32> = vec![1.0; 100];

        let (r_out, g_out, b_out, _) = dither_rgba_bf16(
            &r, &g, &b, &a, 10, 10,
            PerceptualSpace::OkLab, Bf16WorkingSpace::Linear
        );

        // Each output should be within bf16 precision of input
        // bf16 has ~2-3 decimal digits of precision (7-bit mantissa)
        for i in 0..100 {
            let r_diff = (r[i] - r_out[i].to_f32()).abs();
            let g_diff = (g[i] - g_out[i].to_f32()).abs();
            let b_diff = (b[i] - b_out[i].to_f32()).abs();

            // bf16 has coarser precision than f16, allow larger tolerance
            assert!(r_diff < 0.02, "R[{}] diff too large: {} vs {}", i, r[i], r_out[i]);
            assert!(g_diff < 0.02, "G[{}] diff too large: {} vs {}", i, g[i], g_out[i]);
            assert!(b_diff < 0.02, "B[{}] diff too large: {} vs {}", i, b[i], b_out[i]);
        }
    }

    #[test]
    fn test_rgb_bf16_no_alpha() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();

        let (r_out, g_out, b_out) = dither_rgb_bf16_with_mode(
            &r, &g, &b, 10, 10,
            PerceptualSpace::OkLab, Bf16WorkingSpace::Linear,
            DitherMode::Standard, 0, None
        );

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_perceptual_spaces() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 / 100.0).collect();
        let a: Vec<f32> = vec![1.0; 100];

        let spaces = [
            PerceptualSpace::LinearRGB,
            PerceptualSpace::OkLab,
            PerceptualSpace::LabCIE76,
            PerceptualSpace::YCbCr,
        ];

        for space in spaces {
            let (r_out, g_out, b_out, a_out) = dither_rgba_bf16(
                &r, &g, &b, &a, 10, 10, space, Bf16WorkingSpace::Linear
            );

            assert_eq!(r_out.len(), 100, "Space {:?} produced wrong R length", space);
            assert_eq!(g_out.len(), 100, "Space {:?} produced wrong G length", space);
            assert_eq!(b_out.len(), 100, "Space {:?} produced wrong B length", space);
            assert_eq!(a_out.len(), 100, "Space {:?} produced wrong A length", space);
        }
    }

    #[test]
    fn test_bf16_hdr_values() {
        // Test with HDR values > 1.0 (bf16 supports same range as f32)
        let r: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect(); // 0 to 9.9
        let g: Vec<f32> = (0..100).map(|i| i as f32 / 20.0).collect(); // 0 to 4.95
        let b: Vec<f32> = (0..100).map(|i| i as f32 / 50.0).collect(); // 0 to 1.98
        let a: Vec<f32> = vec![1.0; 100];

        let (r_out, g_out, b_out, _) = dither_rgba_bf16(
            &r, &g, &b, &a, 10, 10,
            PerceptualSpace::LinearRGB, Bf16WorkingSpace::Linear
        );

        // Verify HDR values are preserved (approximately)
        for i in 0..100 {
            let r_diff = (r[i] - r_out[i].to_f32()).abs();
            // For larger values, relative error matters more
            let r_rel = if r[i] > 0.1 { r_diff / r[i] } else { r_diff };
            assert!(r_rel < 0.02, "R[{}] relative error too large: {} vs {}", i, r[i], r_out[i]);
        }
    }
}
