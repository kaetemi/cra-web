/// Lab color space dithering with rotation-aware quantization.
///
/// This module quantizes directly in Lab space (CIELAB or OKLab) with:
/// - Rotation of the a/b plane before quantization
/// - User-provided scale/offset to map Lab values to 0-255 output range
/// - Optional preservation of the L channel (no quantization)
/// - Error diffusion in linear RGB for physically correct light mixing
///
/// The color rotation is applied right before quantization (rounding candidates),
/// while error accumulation remains in linear RGB (unrotated space).

use crate::color::{
    lab_to_linear_rgb, linear_rgb_to_lab, linear_rgb_to_oklab, linear_to_srgb_single,
    oklab_to_linear_rgb,
};
use crate::color_distance::perceptual_distance_sq;
use crate::dither_common::{wang_hash, DitherMode, PerceptualSpace};
use crate::rotation::deg_to_rad;

/// Color space for rotation and quantization operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabQuantSpace {
    /// CIELAB: L* 0-100, a*/b* roughly -127 to +127
    #[default]
    CIELab,
    /// OKLab: L 0-1, a/b roughly -0.5 to +0.5
    OkLab,
}

/// Quantization parameters for Lab-space dithering.
///
/// The scale and offset values define how Lab f32 values map to u8 (0-255):
///   u8_value = lab_value * scale + offset
///
/// The user is responsible for choosing scale/offset values that map their
/// Lab range (including any rotation expansion for a/b) to 0-255.
#[derive(Debug, Clone, Copy)]
pub struct LabQuantParams {
    /// Whether to quantize the L channel.
    /// If false, L input is used for distance but output L is not quantized.
    pub quantize_l: bool,
    /// Rotation angle in degrees for a/b plane
    pub rotation_deg: f32,
    /// Scale for L channel: u8 = L * scale_l + offset_l
    pub scale_l: f32,
    /// Offset for L channel
    pub offset_l: f32,
    /// Scale for a/b channels (after rotation): u8 = a * scale_ab + offset_ab
    pub scale_ab: f32,
    /// Offset for a/b channels
    pub offset_ab: f32,
}

impl LabQuantParams {
    /// Default parameters for OKLab (L: 0-1, a/b: -0.5 to 0.5)
    pub fn default_oklab() -> Self {
        Self {
            quantize_l: true,
            rotation_deg: 0.0,
            scale_l: 255.0,      // L: 0-1 → 0-255
            offset_l: 0.0,
            scale_ab: 255.0,     // a/b: -0.5 to 0.5 → 0-255 (with offset 127.5)
            offset_ab: 127.5,
        }
    }

    /// Default parameters for CIELAB (L: 0-100, a/b: -127 to 127)
    pub fn default_cielab() -> Self {
        Self {
            quantize_l: true,
            rotation_deg: 0.0,
            scale_l: 2.55,       // L: 0-100 → 0-255
            offset_l: 0.0,
            scale_ab: 1.0,       // a/b: -127 to 127 → 0-254 (with offset 127)
            offset_ab: 127.0,
        }
    }
}

impl Default for LabQuantParams {
    fn default() -> Self {
        Self::default_oklab()
    }
}

/// Context for Lab-space dithering
struct LabDitherContext {
    /// Precomputed cos(theta) for rotation
    cos_theta: f32,
    /// Precomputed sin(theta) for rotation
    sin_theta: f32,
    /// Lab space for quantization
    quant_space: LabQuantSpace,
    /// Perceptual space for distance calculation
    distance_space: PerceptualSpace,
    /// Whether to quantize L channel
    quantize_l: bool,
    /// Scale for L channel
    scale_l: f32,
    /// Offset for L channel
    offset_l: f32,
    /// Scale for a/b channels
    scale_ab: f32,
    /// Offset for a/b channels
    offset_ab: f32,
}

impl LabDitherContext {
    fn new(params: &LabQuantParams, quant_space: LabQuantSpace, distance_space: PerceptualSpace) -> Self {
        let theta_rad = deg_to_rad(params.rotation_deg);
        let cos_theta = theta_rad.cos();
        let sin_theta = theta_rad.sin();

        Self {
            cos_theta,
            sin_theta,
            quant_space,
            distance_space,
            quantize_l: params.quantize_l,
            scale_l: params.scale_l,
            offset_l: params.offset_l,
            scale_ab: params.scale_ab,
            offset_ab: params.offset_ab,
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

    /// Convert Lab value to u8 position using scale/offset
    #[inline]
    fn lab_l_to_u8_pos(&self, l: f32) -> f32 {
        l * self.scale_l + self.offset_l
    }

    /// Convert Lab a or b value to u8 position using scale/offset
    #[inline]
    fn lab_ab_to_u8_pos(&self, ab: f32) -> f32 {
        ab * self.scale_ab + self.offset_ab
    }

    /// Convert u8 position back to Lab L value
    #[inline]
    fn u8_pos_to_lab_l(&self, pos: f32) -> f32 {
        (pos - self.offset_l) / self.scale_l
    }

    /// Convert u8 position back to Lab a or b value
    #[inline]
    fn u8_pos_to_lab_ab(&self, pos: f32) -> f32 {
        (pos - self.offset_ab) / self.scale_ab
    }

    /// Convert to distance space for comparison (uses unrotated Lab)
    #[inline]
    fn to_distance_space(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        use crate::color::linear_rgb_to_ycbcr_unclamped;
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

/// Floyd-Steinberg with row-based indexing (left-to-right)
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

/// Floyd-Steinberg with row-based indexing (right-to-left)
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

// ============================================================================
// Core pixel processing
// ============================================================================

/// Process a single pixel: find best quantization in Lab space
/// Returns (output_l_u8, output_a_u8, output_b_u8, error_r, error_g, error_b)
#[inline]
fn process_pixel_lab(
    ctx: &LabDitherContext,
    r_orig: f32,
    g_orig: f32,
    b_orig: f32,
    err_r: f32,
    err_g: f32,
    err_b: f32,
) -> (u8, u8, u8, f32, f32, f32) {
    // Add accumulated error in linear RGB
    let r_adj = r_orig + err_r;
    let g_adj = g_orig + err_g;
    let b_adj = b_orig + err_b;

    // Convert to Lab for quantization
    let (l, a, b_ch) = ctx.rgb_to_lab(r_adj, g_adj, b_adj);

    // Rotate a/b for quantization
    let (a_rot, b_rot) = ctx.rotate_ab(a, b_ch);

    // Convert to u8 positions using scale/offset
    let l_pos = ctx.lab_l_to_u8_pos(l);
    let a_pos = ctx.lab_ab_to_u8_pos(a_rot);
    let b_pos = ctx.lab_ab_to_u8_pos(b_rot);

    // Find floor/ceil candidates (clamped to 0-255)
    let l_floor = l_pos.floor().clamp(0.0, 255.0) as u8;
    let l_ceil = l_pos.ceil().clamp(0.0, 255.0) as u8;
    let a_floor = a_pos.floor().clamp(0.0, 255.0) as u8;
    let a_ceil = a_pos.ceil().clamp(0.0, 255.0) as u8;
    let b_floor = b_pos.floor().clamp(0.0, 255.0) as u8;
    let b_ceil = b_pos.ceil().clamp(0.0, 255.0) as u8;

    // Convert target to distance space (unrotated)
    let target_dist = ctx.to_distance_space(r_adj, g_adj, b_adj);

    // Search for best candidate
    let mut best_l_u8 = l_floor;
    let mut best_a_u8 = a_floor;
    let mut best_b_u8 = b_floor;
    let mut best_dist = f32::INFINITY;

    // L candidates
    let l_candidates: &[u8] = if ctx.quantize_l {
        if l_floor == l_ceil { &[l_floor] } else { &[l_floor, l_ceil] }
    } else {
        &[l_floor] // Placeholder, will use original L for output
    };

    // a/b candidates
    let a_candidates: &[u8] = if a_floor == a_ceil { &[a_floor] } else { &[a_floor, a_ceil] };
    let b_candidates: &[u8] = if b_floor == b_ceil { &[b_floor] } else { &[b_floor, b_ceil] };

    for &l_cand_u8 in l_candidates {
        // Convert L candidate back to Lab f32
        let l_cand = if ctx.quantize_l {
            ctx.u8_pos_to_lab_l(l_cand_u8 as f32)
        } else {
            l // Keep original L for distance calculation
        };

        for &a_cand_u8 in a_candidates {
            for &b_cand_u8 in b_candidates {
                // Convert a/b candidates back to Lab f32 (still rotated)
                let a_cand_rot = ctx.u8_pos_to_lab_ab(a_cand_u8 as f32);
                let b_cand_rot = ctx.u8_pos_to_lab_ab(b_cand_u8 as f32);

                // Unrotate the candidate
                let (a_cand, b_cand) = ctx.unrotate_ab(a_cand_rot, b_cand_rot);

                // Convert candidate to linear RGB
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
                    best_l_u8 = l_cand_u8;
                    best_a_u8 = a_cand_u8;
                    best_b_u8 = b_cand_u8;
                }
            }
        }
    }

    // Convert best candidate back to Lab f32 for error calculation
    let best_l = if ctx.quantize_l {
        ctx.u8_pos_to_lab_l(best_l_u8 as f32)
    } else {
        l // Use original L
    };
    let best_a_rot = ctx.u8_pos_to_lab_ab(best_a_u8 as f32);
    let best_b_rot = ctx.u8_pos_to_lab_ab(best_b_u8 as f32);
    let (best_a, best_b) = ctx.unrotate_ab(best_a_rot, best_b_rot);

    // Convert to linear RGB for error calculation
    let (best_r, best_g, best_b_rgb) = ctx.lab_to_rgb(best_l, best_a, best_b);

    // Calculate error in linear RGB (unrotated space)
    let new_err_r = r_adj - best_r;
    let new_err_g = g_adj - best_g;
    let new_err_b = b_adj - best_b_rgb;

    (best_l_u8, best_a_u8, best_b_u8, new_err_r, new_err_g, new_err_b)
}

// ============================================================================
// Main dithering function
// ============================================================================

/// Lab-space dithering with rotation-aware quantization.
///
/// Takes Lab input directly and performs quantization with rotation.
/// User-provided scale/offset map Lab values to 0-255 output range.
/// Error diffusion is performed in linear RGB for physically correct light mixing.
///
/// Args:
///     l_channel, a_channel, b_channel: Input Lab channels as f32
///         - For CIELAB: L is 0-100, a/b are roughly -127 to +127
///         - For OKLab: L is 0-1, a/b are roughly -0.5 to +0.5
///     width, height: Image dimensions
///     params: Quantization parameters (scale/offset for L and a/b, rotation)
///     quant_space: Color space for rotation and quantization (must match input format)
///     distance_space: Perceptual space for distance calculation
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes
///
/// Returns:
///     (l_out, a_out, b_out): Output Lab channels as u8 (0-255)
///     - L channel may be ignored if quantize_l is false
pub fn lab_space_dither_with_mode(
    l_channel: &[f32],
    a_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    params: &LabQuantParams,
    quant_space: LabQuantSpace,
    distance_space: PerceptualSpace,
    mode: DitherMode,
    seed: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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

    // Output buffers in Lab space (u8)
    let mut l_out = vec![0u8; pixels];
    let mut a_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

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

            // Get input Lab and convert to linear RGB (for error accumulation base)
            let (r_lin, g_lin, b_lin) = ctx.lab_to_rgb(
                l_channel[idx],
                a_channel[idx],
                b_channel[idx],
            );

            // Get accumulated error
            let e_r = err_bufs[0][err_idx];
            let e_g = err_bufs[1][err_idx];
            let e_b = err_bufs[2][err_idx];

            // Process pixel
            let (out_l, out_a, out_b, new_err_r, new_err_g, new_err_b) =
                process_pixel_lab(&ctx, r_lin, g_lin, b_lin, e_r, e_g, e_b);

            // Store output (already u8)
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

/// Convenience function with default Floyd-Steinberg mode
pub fn lab_space_dither(
    l_channel: &[f32],
    a_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    params: &LabQuantParams,
    quant_space: LabQuantSpace,
    distance_space: PerceptualSpace,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    lab_space_dither_with_mode(
        l_channel, a_channel, b_channel,
        width, height,
        params,
        quant_space,
        distance_space,
        DitherMode::Standard,
        0,
    )
}

/// Convert Lab u8 values back to f32 based on scale/offset
#[inline]
pub fn lab_u8_to_f32_with_params(l: u8, a: u8, b: u8, params: &LabQuantParams) -> (f32, f32, f32) {
    let l_f = (l as f32 - params.offset_l) / params.scale_l;
    let a_f = (a as f32 - params.offset_ab) / params.scale_ab;
    let b_f = (b as f32 - params.offset_ab) / params.scale_ab;
    (l_f, a_f, b_f)
}

/// Convert Lab u8 output back to sRGB u8
pub fn lab_to_srgb_u8(
    l_channel: &[u8],
    a_channel: &[u8],
    b_channel: &[u8],
    params: &LabQuantParams,
    quant_space: LabQuantSpace,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = l_channel.len();
    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

    for i in 0..pixels {
        // Convert u8 Lab back to f32 using scale/offset
        let (l_f, a_f, b_f) = lab_u8_to_f32_with_params(
            l_channel[i], a_channel[i], b_channel[i], params
        );

        let (r_lin, g_lin, b_lin) = match quant_space {
            LabQuantSpace::CIELab => lab_to_linear_rgb(l_f, a_f, b_f),
            LabQuantSpace::OkLab => oklab_to_linear_rgb(l_f, a_f, b_f),
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

    /// Generate test data in OKLab space (L: 0-1, a/b: -0.5 to 0.5)
    fn generate_oklab_test_data(pixels: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let l: Vec<f32> = (0..pixels).map(|i| (i as f32) / (pixels as f32)).collect();
        let a: Vec<f32> = (0..pixels).map(|i| ((i + pixels / 3) % pixels) as f32 / (pixels as f32) - 0.5).collect();
        let b: Vec<f32> = (0..pixels).map(|i| ((i + 2 * pixels / 3) % pixels) as f32 / (pixels as f32) - 0.5).collect();
        (l, a, b)
    }

    /// Generate test data in CIELAB space (L: 0-100, a/b: -127 to 127)
    fn generate_cielab_test_data(pixels: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let l: Vec<f32> = (0..pixels).map(|i| (i as f32) / (pixels as f32) * 100.0).collect();
        let a: Vec<f32> = (0..pixels).map(|i| ((i + pixels / 3) % pixels) as f32 / (pixels as f32) * 254.0 - 127.0).collect();
        let b: Vec<f32> = (0..pixels).map(|i| ((i + 2 * pixels / 3) % pixels) as f32 / (pixels as f32) * 254.0 - 127.0).collect();
        (l, a, b)
    }

    #[test]
    fn test_lab_dither_basic_oklab() {
        let (l, a, b) = generate_oklab_test_data(100);

        let params = LabQuantParams::default_oklab();
        let (l_out, a_out, b_out) = lab_space_dither(
            &l, &a, &b, 10, 10,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        assert_eq!(l_out.len(), 100);
        assert_eq!(a_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_lab_dither_basic_cielab() {
        let (l, a, b) = generate_cielab_test_data(100);

        let params = LabQuantParams::default_cielab();
        let (l_out, a_out, b_out) = lab_space_dither(
            &l, &a, &b, 10, 10,
            &params,
            LabQuantSpace::CIELab,
            PerceptualSpace::LabCIE76,
        );

        assert_eq!(l_out.len(), 100);
        assert_eq!(a_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_lab_dither_with_rotation() {
        let (l, a, b) = generate_cielab_test_data(100);

        let mut params = LabQuantParams::default_cielab();
        params.rotation_deg = 45.0;

        let (l_out, a_out, b_out) = lab_space_dither(
            &l, &a, &b, 10, 10,
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
        let (l, a, b) = generate_oklab_test_data(100);

        let mut params = LabQuantParams::default_oklab();
        params.quantize_l = false;

        let (l_out, _a_out, _b_out) = lab_space_dither(
            &l, &a, &b, 10, 10,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        // L values are output but may be ignored
        assert_eq!(l_out.len(), 100);
    }

    #[test]
    fn test_lab_dither_custom_scale_offset() {
        let (l, a, b) = generate_oklab_test_data(100);

        // Custom scale/offset
        let params = LabQuantParams {
            quantize_l: true,
            rotation_deg: 0.0,
            scale_l: 200.0,   // Narrower L range
            offset_l: 27.5,
            scale_ab: 200.0,  // Narrower a/b range
            offset_ab: 127.5,
        };

        let (l_out, a_out, b_out) = lab_space_dither(
            &l, &a, &b, 10, 10,
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
        // Gray in OKLab: L=0.5, a=0, b=0
        let l: Vec<f32> = vec![0.5; 4];
        let a: Vec<f32> = vec![0.0; 4];
        let b: Vec<f32> = vec![0.0; 4];

        let params = LabQuantParams::default_oklab();

        let (l_out, a_out, b_out) = lab_space_dither(
            &l, &a, &b, 2, 2,
            &params,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        let (r_srgb, g_srgb, b_srgb) = lab_to_srgb_u8(&l_out, &a_out, &b_out, &params, LabQuantSpace::OkLab);

        // Gray should remain gray (R=G=B)
        for i in 0..4 {
            assert!((r_srgb[i] as i32 - g_srgb[i] as i32).abs() < 5, "Not gray at {}: R={}, G={}", i, r_srgb[i], g_srgb[i]);
            assert!((g_srgb[i] as i32 - b_srgb[i] as i32).abs() < 5, "Not gray at {}: G={}, B={}", i, g_srgb[i], b_srgb[i]);
        }
    }

    #[test]
    fn test_all_modes_produce_output() {
        let (l, a, b) = generate_oklab_test_data(100);

        let params = LabQuantParams::default_oklab();
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
            let (l_out, a_out, b_out) = lab_space_dither_with_mode(
                &l, &a, &b, 10, 10,
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
    fn test_rotation_changes_output() {
        let (l, a, b) = generate_oklab_test_data(100);

        let params_no_rot = LabQuantParams::default_oklab();
        let mut params_rot = LabQuantParams::default_oklab();
        params_rot.rotation_deg = 45.0;

        let (_l_out1, a_out1, b_out1) = lab_space_dither(
            &l, &a, &b, 10, 10,
            &params_no_rot,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        let (_l_out2, a_out2, b_out2) = lab_space_dither(
            &l, &a, &b, 10, 10,
            &params_rot,
            LabQuantSpace::OkLab,
            PerceptualSpace::OkLab,
        );

        // Results should differ with rotation (compare sum of u8 values)
        let sum1: u32 = a_out1.iter().chain(b_out1.iter()).map(|&x| x as u32).sum();
        let sum2: u32 = a_out2.iter().chain(b_out2.iter()).map(|&x| x as u32).sum();
        assert!(sum1 != sum2, "Rotation should change output: {} vs {}", sum1, sum2);
    }
}
