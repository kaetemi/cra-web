/// Lab color space dithering with rotation-aware quantization.
///
/// Unlike the RGB quantizer (dither_colorspace_aware.rs), this module:
/// - Quantizes directly in Lab space (CIELAB or OKLab)
/// - Supports rotation of the a/b plane before quantization
/// - Provides offset and scaling for quantization ranges
/// - Can optionally preserve the L channel (no quantization)
/// - Maintains error diffusion in linear RGB for physically correct light mixing
///
/// The color rotation is applied right before quantization (rounding candidates),
/// while error accumulation remains in linear RGB (unrotated space).

use crate::color::{
    lab_to_linear_rgb, linear_rgb_to_lab, linear_rgb_to_oklab, linear_to_srgb_single,
    oklab_to_linear_rgb, srgb_to_linear_single,
};
use crate::color_distance::perceptual_distance_sq;
use crate::dither_common::{wang_hash, DitherMode, PerceptualSpace};
use crate::rotation::{compute_ab_ranges, compute_oklab_ab_ranges, deg_to_rad};

/// Color space for rotation and quantization operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabQuantSpace {
    /// CIELAB: L* 0-100, a*/b* roughly -127 to +127
    #[default]
    CIELab,
    /// OKLab: L 0-1, a/b roughly -0.5 to +0.5
    OkLab,
}

/// Quantization parameters for Lab-space dithering
#[derive(Debug, Clone, Copy)]
pub struct LabQuantParams {
    /// Number of quantization levels for a/b channels (power of 2, e.g., 16 = 4 bits)
    pub levels_ab: usize,
    /// Number of quantization levels for L channel (power of 2, e.g., 256 = 8 bits)
    /// Ignored if quantize_l is false
    pub levels_l: usize,
    /// Whether to quantize the L channel
    /// If false, L input is used for distance but output L is not quantized
    pub quantize_l: bool,
    /// Rotation angle in degrees for a/b plane
    pub rotation_deg: f32,
    /// Offset for quantization range (shifts all levels by this fraction of the range)
    /// 0.0 = no offset, 0.5 = shift by half a level
    pub offset: f32,
    /// Scaling factor for quantization range (1.0 = full range, <1.0 = narrower)
    pub scale: f32,
}

impl Default for LabQuantParams {
    fn default() -> Self {
        Self {
            levels_ab: 16,  // 4 bits
            levels_l: 256,  // 8 bits
            quantize_l: true,
            rotation_deg: 0.0,
            offset: 0.0,
            scale: 1.0,
        }
    }
}

/// Pre-computed quantization levels for a Lab channel
struct ChannelQuantLevels {
    /// Quantized values for each level
    values: Vec<f32>,
    /// Minimum value of the quantization range
    min_val: f32,
    /// Maximum value of the quantization range
    max_val: f32,
    /// Step size between levels
    step: f32,
}

impl ChannelQuantLevels {
    /// Create quantization levels for a channel
    fn new(num_levels: usize, min_val: f32, max_val: f32, offset: f32, scale: f32) -> Self {
        let range = max_val - min_val;
        let center = (min_val + max_val) / 2.0;

        // Apply scaling around center
        let scaled_range = range * scale;
        let scaled_min = center - scaled_range / 2.0;
        let scaled_max = center + scaled_range / 2.0;

        // Calculate step size
        let step = scaled_range / (num_levels as f32);

        // Apply offset (fraction of step)
        let offset_amount = offset * step;
        let final_min = scaled_min + offset_amount;
        let final_max = scaled_max + offset_amount;

        // Generate quantization levels (midpoints of each bin)
        let values: Vec<f32> = (0..num_levels)
            .map(|i| {
                final_min + step * (i as f32 + 0.5)
            })
            .collect();

        Self {
            values,
            min_val: final_min,
            max_val: final_max,
            step,
        }
    }

    /// Find the two nearest quantization levels for a value
    #[inline]
    fn find_candidates(&self, val: f32) -> (usize, usize) {
        // Clamp to valid range
        let clamped = val.clamp(self.min_val, self.max_val);

        // Find position in terms of levels
        let pos = (clamped - self.min_val) / self.step;
        let floor_idx = (pos.floor() as usize).min(self.values.len() - 1);
        let ceil_idx = (floor_idx + 1).min(self.values.len() - 1);

        (floor_idx, ceil_idx)
    }

    /// Get the quantized value for a level index
    #[inline]
    fn level_value(&self, idx: usize) -> f32 {
        self.values[idx]
    }
}

/// Context for Lab-space dithering
struct LabDitherContext {
    /// Quantization levels for L channel
    l_levels: ChannelQuantLevels,
    /// Quantization levels for a channel (rotated)
    a_levels: ChannelQuantLevels,
    /// Quantization levels for b channel (rotated)
    b_levels: ChannelQuantLevels,
    /// Precomputed cos(theta)
    cos_theta: f32,
    /// Precomputed sin(theta)
    sin_theta: f32,
    /// Lab space for quantization
    quant_space: LabQuantSpace,
    /// Perceptual space for distance calculation
    distance_space: PerceptualSpace,
    /// Whether to quantize L channel
    quantize_l: bool,
}

impl LabDitherContext {
    fn new(params: &LabQuantParams, quant_space: LabQuantSpace, distance_space: PerceptualSpace) -> Self {
        let theta_rad = deg_to_rad(params.rotation_deg);
        let cos_theta = theta_rad.cos();
        let sin_theta = theta_rad.sin();

        // Determine ranges based on color space and rotation
        let (l_min, l_max, ab_ranges) = match quant_space {
            LabQuantSpace::CIELab => {
                let ranges = compute_ab_ranges(params.rotation_deg);
                (0.0, 100.0, ranges)
            }
            LabQuantSpace::OkLab => {
                let ranges = compute_oklab_ab_ranges(params.rotation_deg);
                (0.0, 1.0, ranges)
            }
        };

        let l_levels = ChannelQuantLevels::new(
            params.levels_l, l_min, l_max, params.offset, params.scale
        );
        let a_levels = ChannelQuantLevels::new(
            params.levels_ab, ab_ranges[0][0], ab_ranges[0][1], params.offset, params.scale
        );
        let b_levels = ChannelQuantLevels::new(
            params.levels_ab, ab_ranges[1][0], ab_ranges[1][1], params.offset, params.scale
        );

        Self {
            l_levels,
            a_levels,
            b_levels,
            cos_theta,
            sin_theta,
            quant_space,
            distance_space,
            quantize_l: params.quantize_l,
        }
    }

    /// Convert linear RGB to Lab (based on quant_space)
    #[inline]
    fn rgb_to_lab(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match self.quant_space {
            LabQuantSpace::CIELab => linear_rgb_to_lab(r, g, b),
            LabQuantSpace::OkLab => linear_rgb_to_oklab(r, g, b),
        }
    }

    /// Convert Lab to linear RGB (based on quant_space)
    #[inline]
    fn lab_to_rgb(&self, l: f32, a: f32, b: f32) -> (f32, f32, f32) {
        match self.quant_space {
            LabQuantSpace::CIELab => lab_to_linear_rgb(l, a, b),
            LabQuantSpace::OkLab => oklab_to_linear_rgb(l, a, b),
        }
    }

    /// Rotate a/b values forward (before quantization)
    #[inline]
    fn rotate_ab(&self, a: f32, b: f32) -> (f32, f32) {
        let a_rot = a * self.cos_theta - b * self.sin_theta;
        let b_rot = a * self.sin_theta + b * self.cos_theta;
        (a_rot, b_rot)
    }

    /// Rotate a/b values backward (after quantization)
    #[inline]
    fn unrotate_ab(&self, a_rot: f32, b_rot: f32) -> (f32, f32) {
        let a = a_rot * self.cos_theta + b_rot * self.sin_theta;
        let b = -a_rot * self.sin_theta + b_rot * self.cos_theta;
        (a, b)
    }

    /// Convert to distance space for comparison (uses unrotated Lab)
    #[inline]
    fn to_distance_space(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        use crate::color::{linear_rgb_to_ycbcr_unclamped};
        use crate::color_distance::{is_lab_space, is_linear_rgb_space, is_ycbcr_space};

        if is_linear_rgb_space(self.distance_space) {
            (r, g, b)
        } else if is_ycbcr_space(self.distance_space) {
            linear_rgb_to_ycbcr_unclamped(r, g, b)
        } else if is_lab_space(self.distance_space) {
            linear_rgb_to_lab(r, g, b)
        } else {
            linear_rgb_to_oklab(r, g, b)
        }
    }
}

// ============================================================================
// Error diffusion kernel implementations
// ============================================================================

/// Apply JJN error diffusion (left-to-right)
#[inline]
fn apply_jjn_ltr(err: &mut [Vec<f32>], bx: usize, y: usize, e_r: f32, e_g: f32, e_b: f32, buf_width: usize) {
    let row0 = y * buf_width;
    let row1 = (y + 1) * buf_width;
    let row2 = (y + 2) * buf_width;

    // Row 0
    err[0][row0 + bx + 1] += e_r * (7.0 / 48.0);
    err[1][row0 + bx + 1] += e_g * (7.0 / 48.0);
    err[2][row0 + bx + 1] += e_b * (7.0 / 48.0);
    err[0][row0 + bx + 2] += e_r * (5.0 / 48.0);
    err[1][row0 + bx + 2] += e_g * (5.0 / 48.0);
    err[2][row0 + bx + 2] += e_b * (5.0 / 48.0);

    // Row 1
    err[0][row1 + bx - 2] += e_r * (3.0 / 48.0);
    err[1][row1 + bx - 2] += e_g * (3.0 / 48.0);
    err[2][row1 + bx - 2] += e_b * (3.0 / 48.0);
    err[0][row1 + bx - 1] += e_r * (5.0 / 48.0);
    err[1][row1 + bx - 1] += e_g * (5.0 / 48.0);
    err[2][row1 + bx - 1] += e_b * (5.0 / 48.0);
    err[0][row1 + bx] += e_r * (7.0 / 48.0);
    err[1][row1 + bx] += e_g * (7.0 / 48.0);
    err[2][row1 + bx] += e_b * (7.0 / 48.0);
    err[0][row1 + bx + 1] += e_r * (5.0 / 48.0);
    err[1][row1 + bx + 1] += e_g * (5.0 / 48.0);
    err[2][row1 + bx + 1] += e_b * (5.0 / 48.0);
    err[0][row1 + bx + 2] += e_r * (3.0 / 48.0);
    err[1][row1 + bx + 2] += e_g * (3.0 / 48.0);
    err[2][row1 + bx + 2] += e_b * (3.0 / 48.0);

    // Row 2
    err[0][row2 + bx - 2] += e_r * (1.0 / 48.0);
    err[1][row2 + bx - 2] += e_g * (1.0 / 48.0);
    err[2][row2 + bx - 2] += e_b * (1.0 / 48.0);
    err[0][row2 + bx - 1] += e_r * (3.0 / 48.0);
    err[1][row2 + bx - 1] += e_g * (3.0 / 48.0);
    err[2][row2 + bx - 1] += e_b * (3.0 / 48.0);
    err[0][row2 + bx] += e_r * (5.0 / 48.0);
    err[1][row2 + bx] += e_g * (5.0 / 48.0);
    err[2][row2 + bx] += e_b * (5.0 / 48.0);
    err[0][row2 + bx + 1] += e_r * (3.0 / 48.0);
    err[1][row2 + bx + 1] += e_g * (3.0 / 48.0);
    err[2][row2 + bx + 1] += e_b * (3.0 / 48.0);
    err[0][row2 + bx + 2] += e_r * (1.0 / 48.0);
    err[1][row2 + bx + 2] += e_g * (1.0 / 48.0);
    err[2][row2 + bx + 2] += e_b * (1.0 / 48.0);
}

/// Apply JJN error diffusion (right-to-left)
#[inline]
fn apply_jjn_rtl(err: &mut [Vec<f32>], bx: usize, y: usize, e_r: f32, e_g: f32, e_b: f32, buf_width: usize) {
    let row0 = y * buf_width;
    let row1 = (y + 1) * buf_width;
    let row2 = (y + 2) * buf_width;

    // Row 0
    err[0][row0 + bx - 1] += e_r * (7.0 / 48.0);
    err[1][row0 + bx - 1] += e_g * (7.0 / 48.0);
    err[2][row0 + bx - 1] += e_b * (7.0 / 48.0);
    err[0][row0 + bx - 2] += e_r * (5.0 / 48.0);
    err[1][row0 + bx - 2] += e_g * (5.0 / 48.0);
    err[2][row0 + bx - 2] += e_b * (5.0 / 48.0);

    // Row 1
    err[0][row1 + bx + 2] += e_r * (3.0 / 48.0);
    err[1][row1 + bx + 2] += e_g * (3.0 / 48.0);
    err[2][row1 + bx + 2] += e_b * (3.0 / 48.0);
    err[0][row1 + bx + 1] += e_r * (5.0 / 48.0);
    err[1][row1 + bx + 1] += e_g * (5.0 / 48.0);
    err[2][row1 + bx + 1] += e_b * (5.0 / 48.0);
    err[0][row1 + bx] += e_r * (7.0 / 48.0);
    err[1][row1 + bx] += e_g * (7.0 / 48.0);
    err[2][row1 + bx] += e_b * (7.0 / 48.0);
    err[0][row1 + bx - 1] += e_r * (5.0 / 48.0);
    err[1][row1 + bx - 1] += e_g * (5.0 / 48.0);
    err[2][row1 + bx - 1] += e_b * (5.0 / 48.0);
    err[0][row1 + bx - 2] += e_r * (3.0 / 48.0);
    err[1][row1 + bx - 2] += e_g * (3.0 / 48.0);
    err[2][row1 + bx - 2] += e_b * (3.0 / 48.0);

    // Row 2
    err[0][row2 + bx + 2] += e_r * (1.0 / 48.0);
    err[1][row2 + bx + 2] += e_g * (1.0 / 48.0);
    err[2][row2 + bx + 2] += e_b * (1.0 / 48.0);
    err[0][row2 + bx + 1] += e_r * (3.0 / 48.0);
    err[1][row2 + bx + 1] += e_g * (3.0 / 48.0);
    err[2][row2 + bx + 1] += e_b * (3.0 / 48.0);
    err[0][row2 + bx] += e_r * (5.0 / 48.0);
    err[1][row2 + bx] += e_g * (5.0 / 48.0);
    err[2][row2 + bx] += e_b * (5.0 / 48.0);
    err[0][row2 + bx - 1] += e_r * (3.0 / 48.0);
    err[1][row2 + bx - 1] += e_g * (3.0 / 48.0);
    err[2][row2 + bx - 1] += e_b * (3.0 / 48.0);
    err[0][row2 + bx - 2] += e_r * (1.0 / 48.0);
    err[1][row2 + bx - 2] += e_g * (1.0 / 48.0);
    err[2][row2 + bx - 2] += e_b * (1.0 / 48.0);
}

// ============================================================================
// Core pixel processing
// ============================================================================

/// Process a single pixel: find best quantization in Lab space
/// Returns (output_l, output_a, output_b, error_r, error_g, error_b)
#[inline]
fn process_pixel_lab(
    ctx: &LabDitherContext,
    r_orig: f32,
    g_orig: f32,
    b_orig: f32,
    err_r: f32,
    err_g: f32,
    err_b: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    // Add accumulated error in linear RGB
    let r_adj = r_orig + err_r;
    let g_adj = g_orig + err_g;
    let b_adj = b_orig + err_b;

    // Convert to Lab for quantization
    let (l, a, b_ch) = ctx.rgb_to_lab(r_adj, g_adj, b_adj);

    // Rotate a/b for quantization
    let (a_rot, b_rot) = ctx.rotate_ab(a, b_ch);

    // Find candidate levels for each channel
    let (l_min, l_max) = if ctx.quantize_l {
        ctx.l_levels.find_candidates(l)
    } else {
        (0, 0) // Will be ignored
    };
    let (a_min, a_max) = ctx.a_levels.find_candidates(a_rot);
    let (b_min, b_max) = ctx.b_levels.find_candidates(b_rot);

    // Convert target to distance space (unrotated)
    let target_dist = ctx.to_distance_space(r_adj, g_adj, b_adj);

    // Search for best candidate
    let mut best_l = l;
    let mut best_a = a;
    let mut best_b = b_ch;
    let mut best_dist = f32::INFINITY;

    // L candidates
    let l_candidates = if ctx.quantize_l {
        vec![ctx.l_levels.level_value(l_min), ctx.l_levels.level_value(l_max)]
    } else {
        vec![l] // Keep original L
    };

    for &l_cand in &l_candidates {
        for a_idx in a_min..=a_max {
            let a_cand_rot = ctx.a_levels.level_value(a_idx);
            for b_idx in b_min..=b_max {
                let b_cand_rot = ctx.b_levels.level_value(b_idx);

                // Unrotate the candidate
                let (a_cand, b_cand) = ctx.unrotate_ab(a_cand_rot, b_cand_rot);

                // Convert candidate back to linear RGB
                let (r_cand, g_cand, b_cand_rgb) = ctx.lab_to_rgb(l_cand, a_cand, b_cand);

                // Calculate distance in the distance space (unrotated)
                let cand_dist = ctx.to_distance_space(r_cand, g_cand, b_cand_rgb);
                let dist = perceptual_distance_sq(
                    ctx.distance_space,
                    target_dist.0, target_dist.1, target_dist.2,
                    cand_dist.0, cand_dist.1, cand_dist.2,
                );

                if dist < best_dist {
                    best_dist = dist;
                    best_l = l_cand;
                    best_a = a_cand;
                    best_b = b_cand;
                }
            }
        }
    }

    // Convert best candidate to linear RGB for error calculation
    let (best_r, best_g, best_b_rgb) = ctx.lab_to_rgb(best_l, best_a, best_b);

    // Calculate error in linear RGB (unrotated space)
    let new_err_r = r_adj - best_r;
    let new_err_g = g_adj - best_g;
    let new_err_b = b_adj - best_b_rgb;

    (best_l, best_a, best_b, new_err_r, new_err_g, new_err_b)
}

// ============================================================================
// Main dithering function
// ============================================================================

/// Lab-space dithering with rotation-aware quantization.
///
/// Processes RGB input through Lab color space with optional rotation,
/// offset, and scaling for quantization. Error diffusion is performed
/// in linear RGB for physically correct light mixing.
///
/// Args:
///     r_channel, g_channel, b_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     params: Quantization parameters (levels, rotation, offset, scale)
///     quant_space: Color space for rotation and quantization (CIELAB or OKLab)
///     distance_space: Perceptual space for distance calculation
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes
///
/// Returns:
///     (l_out, a_out, b_out): Output Lab channels as f32
pub fn lab_space_dither_rgb_with_mode(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    params: &LabQuantParams,
    quant_space: LabQuantSpace,
    distance_space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = width * height;
    let ctx = LabDitherContext::new(params, quant_space, distance_space);

    // Padding for JJN kernel (largest)
    let pad_left = 2;
    let pad_right = 2;
    let pad_bottom = 2;
    let buf_width = width + pad_left + pad_right;
    let buf_height = height + pad_bottom;

    // Error buffers in linear RGB (3 channels, flattened)
    let mut err_bufs: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width * buf_height]; 3];

    // Output buffers in Lab space
    let mut l_out = vec![0.0f32; pixels];
    let mut a_out = vec![0.0f32; pixels];
    let mut b_out = vec![0.0f32; pixels];

    let hashed_seed = wang_hash(seed);

    // Determine if using JJN kernel
    let use_jjn = matches!(mode, DitherMode::JarvisStandard | DitherMode::JarvisSerpentine);
    let use_mixed = matches!(mode, DitherMode::MixedStandard | DitherMode::MixedSerpentine | DitherMode::MixedRandom);
    let use_serpentine = matches!(mode, DitherMode::Serpentine | DitherMode::JarvisSerpentine | DitherMode::MixedSerpentine);
    let use_random_dir = matches!(mode, DitherMode::MixedRandom);

    for y in 0..height {
        // Determine scan direction for this row
        let is_rtl = if use_random_dir {
            wang_hash((y as u32) ^ hashed_seed) & 1 == 1
        } else if use_serpentine {
            y % 2 == 1
        } else {
            false
        };

        let x_iter: Box<dyn Iterator<Item = usize>> = if is_rtl {
            Box::new((0..width).rev())
        } else {
            Box::new(0..width)
        };

        for x in x_iter {
            let idx = y * width + x;
            let bx = x + pad_left;
            let err_idx = y * buf_width + bx;

            // Get input in linear RGB
            let r_srgb = r_channel[idx] / 255.0;
            let g_srgb = g_channel[idx] / 255.0;
            let b_srgb = b_channel[idx] / 255.0;
            let r_lin = srgb_to_linear_single(r_srgb);
            let g_lin = srgb_to_linear_single(g_srgb);
            let b_lin = srgb_to_linear_single(b_srgb);

            // Get accumulated error
            let e_r = err_bufs[0][err_idx];
            let e_g = err_bufs[1][err_idx];
            let e_b = err_bufs[2][err_idx];

            // Process pixel
            let (out_l, out_a, out_b, new_err_r, new_err_g, new_err_b) =
                process_pixel_lab(&ctx, r_lin, g_lin, b_lin, e_r, e_g, e_b);

            // Store output
            l_out[idx] = out_l;
            a_out[idx] = out_a;
            b_out[idx] = out_b;

            // Apply error diffusion kernel
            if use_mixed {
                // Random kernel selection per pixel
                let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                let use_jjn_pixel = pixel_hash & 1 != 0;

                if use_jjn_pixel {
                    if is_rtl {
                        apply_jjn_rtl(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                    } else {
                        apply_jjn_ltr(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                    }
                } else {
                    // Use row-based indexing for FS
                    if is_rtl {
                        apply_fs_rtl_row(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                    } else {
                        apply_fs_ltr_row(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                    }
                }
            } else if use_jjn {
                if is_rtl {
                    apply_jjn_rtl(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                } else {
                    apply_jjn_ltr(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                }
            } else {
                // Floyd-Steinberg
                if is_rtl {
                    apply_fs_rtl_row(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                } else {
                    apply_fs_ltr_row(&mut err_bufs, bx, y, new_err_r, new_err_g, new_err_b, buf_width);
                }
            }
        }
    }

    (l_out, a_out, b_out)
}

/// Floyd-Steinberg with row-based indexing (for consistency with JJN)
#[inline]
fn apply_fs_ltr_row(err: &mut [Vec<f32>], bx: usize, y: usize, e_r: f32, e_g: f32, e_b: f32, buf_width: usize) {
    let row0 = y * buf_width;
    let row1 = (y + 1) * buf_width;

    // Right: 7/16
    err[0][row0 + bx + 1] += e_r * (7.0 / 16.0);
    err[1][row0 + bx + 1] += e_g * (7.0 / 16.0);
    err[2][row0 + bx + 1] += e_b * (7.0 / 16.0);
    // Bottom-left: 3/16
    err[0][row1 + bx - 1] += e_r * (3.0 / 16.0);
    err[1][row1 + bx - 1] += e_g * (3.0 / 16.0);
    err[2][row1 + bx - 1] += e_b * (3.0 / 16.0);
    // Bottom: 5/16
    err[0][row1 + bx] += e_r * (5.0 / 16.0);
    err[1][row1 + bx] += e_g * (5.0 / 16.0);
    err[2][row1 + bx] += e_b * (5.0 / 16.0);
    // Bottom-right: 1/16
    err[0][row1 + bx + 1] += e_r * (1.0 / 16.0);
    err[1][row1 + bx + 1] += e_g * (1.0 / 16.0);
    err[2][row1 + bx + 1] += e_b * (1.0 / 16.0);
}

#[inline]
fn apply_fs_rtl_row(err: &mut [Vec<f32>], bx: usize, y: usize, e_r: f32, e_g: f32, e_b: f32, buf_width: usize) {
    let row0 = y * buf_width;
    let row1 = (y + 1) * buf_width;

    // Left: 7/16
    err[0][row0 + bx - 1] += e_r * (7.0 / 16.0);
    err[1][row0 + bx - 1] += e_g * (7.0 / 16.0);
    err[2][row0 + bx - 1] += e_b * (7.0 / 16.0);
    // Bottom-right: 3/16
    err[0][row1 + bx + 1] += e_r * (3.0 / 16.0);
    err[1][row1 + bx + 1] += e_g * (3.0 / 16.0);
    err[2][row1 + bx + 1] += e_b * (3.0 / 16.0);
    // Bottom: 5/16
    err[0][row1 + bx] += e_r * (5.0 / 16.0);
    err[1][row1 + bx] += e_g * (5.0 / 16.0);
    err[2][row1 + bx] += e_b * (5.0 / 16.0);
    // Bottom-left: 1/16
    err[0][row1 + bx - 1] += e_r * (1.0 / 16.0);
    err[1][row1 + bx - 1] += e_g * (1.0 / 16.0);
    err[2][row1 + bx - 1] += e_b * (1.0 / 16.0);
}

/// Convenience function with default Floyd-Steinberg mode
pub fn lab_space_dither_rgb(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    params: &LabQuantParams,
    quant_space: LabQuantSpace,
    distance_space: PerceptualSpace,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    lab_space_dither_rgb_with_mode(
        r_channel, g_channel, b_channel,
        width, height,
        params,
        quant_space,
        distance_space,
        DitherMode::Standard,
        0,
    )
}

/// Convert Lab output back to sRGB u8
pub fn lab_to_srgb_u8(
    l_channel: &[f32],
    a_channel: &[f32],
    b_channel: &[f32],
    quant_space: LabQuantSpace,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = l_channel.len();
    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

    for i in 0..pixels {
        let (r_lin, g_lin, b_lin) = match quant_space {
            LabQuantSpace::CIELab => lab_to_linear_rgb(l_channel[i], a_channel[i], b_channel[i]),
            LabQuantSpace::OkLab => oklab_to_linear_rgb(l_channel[i], a_channel[i], b_channel[i]),
        };

        let r_srgb = (linear_to_srgb_single(r_lin.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let g_srgb = (linear_to_srgb_single(g_lin.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let b_srgb = (linear_to_srgb_single(b_lin.clamp(0.0, 1.0)) * 255.0).round() as u8;

        r_out[i] = r_srgb;
        g_out[i] = g_srgb;
        b_out[i] = b_srgb;
    }

    (r_out, g_out, b_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lab_dither_basic() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let params = LabQuantParams::default();
        let (l_out, a_out, b_out) = lab_space_dither_rgb(
            &r, &g, &b, 10, 10,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        assert_eq!(l_out.len(), 100);
        assert_eq!(a_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_lab_dither_with_rotation() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let mut params = LabQuantParams::default();
        params.rotation_deg = 45.0;

        let (l_out, a_out, b_out) = lab_space_dither_rgb(
            &r, &g, &b, 10, 10,
            &params,
            LabQuantSpace::CIELab,
            PerceptualSpace::LabCIE76,
        );

        assert_eq!(l_out.len(), 100);
        assert_eq!(a_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_lab_dither_no_l_quantization() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let mut params = LabQuantParams::default();
        params.quantize_l = false;

        let (l_out, _a_out, _b_out) = lab_space_dither_rgb(
            &r, &g, &b, 10, 10,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        // L values should be non-quantized (continuous)
        assert_eq!(l_out.len(), 100);
    }

    #[test]
    fn test_lab_dither_with_offset_and_scale() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let mut params = LabQuantParams::default();
        params.offset = 0.25;
        params.scale = 0.8;

        let (l_out, a_out, b_out) = lab_space_dither_rgb(
            &r, &g, &b, 10, 10,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        assert_eq!(l_out.len(), 100);
        assert_eq!(a_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_lab_to_srgb_roundtrip() {
        let r: Vec<f32> = vec![127.5; 4];
        let g: Vec<f32> = vec![127.5; 4];
        let b: Vec<f32> = vec![127.5; 4];

        let params = LabQuantParams {
            levels_ab: 256,
            levels_l: 256,
            quantize_l: true,
            rotation_deg: 0.0,
            offset: 0.0,
            scale: 1.0,
        };

        let (l_out, a_out, b_out) = lab_space_dither_rgb(
            &r, &g, &b, 2, 2,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        let (r_srgb, g_srgb, b_srgb) = lab_to_srgb_u8(&l_out, &a_out, &b_out, LabQuantSpace::OkLab);

        // With 256 levels and no rotation, should be close to original
        for i in 0..4 {
            assert!((r_srgb[i] as i32 - 128).abs() < 5, "R mismatch: {}", r_srgb[i]);
            assert!((g_srgb[i] as i32 - 128).abs() < 5, "G mismatch: {}", g_srgb[i]);
            assert!((b_srgb[i] as i32 - 128).abs() < 5, "B mismatch: {}", b_srgb[i]);
        }
    }

    #[test]
    fn test_all_modes_produce_output() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let params = LabQuantParams::default();
        let modes = [
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
        ];

        for mode in modes {
            let (l_out, a_out, b_out) = lab_space_dither_rgb_with_mode(
                &r, &g, &b, 10, 10,
                &params,
                LabQuantSpace::OkLab,
                PerceptualSpace::OkLab,
                mode,
                42,
            );

            assert_eq!(l_out.len(), 100, "Mode {:?} produced wrong length", mode);
            assert_eq!(a_out.len(), 100, "Mode {:?} produced wrong length", mode);
            assert_eq!(b_out.len(), 100, "Mode {:?} produced wrong length", mode);
        }
    }

    #[test]
    fn test_cielab_vs_oklab() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let params = LabQuantParams::default();

        let (l_cielab, a_cielab, b_cielab) = lab_space_dither_rgb(
            &r, &g, &b, 10, 10,
            &params,
            LabQuantSpace::CIELab,
            PerceptualSpace::LabCIE76,
        );

        let (l_oklab, a_oklab, b_oklab) = lab_space_dither_rgb(
            &r, &g, &b, 10, 10,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        // Results should differ between spaces
        let cielab_sum: f32 = l_cielab.iter().chain(a_cielab.iter()).chain(b_cielab.iter()).sum();
        let oklab_sum: f32 = l_oklab.iter().chain(a_oklab.iter()).chain(b_oklab.iter()).sum();
        assert!((cielab_sum - oklab_sum).abs() > 0.01, "CIELAB and OKLab should produce different results");
    }
}
