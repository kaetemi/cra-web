//! Bilinear interpolation implementation
//!
//! Contains both standard and alpha-aware bilinear rescaling.

use crate::pixel::{Pixel4, lerp};
use super::{ScaleMode, TentMode, calculate_scales};

/// Calculate source coordinate for a destination position
///
/// Supports multiple coordinate mapping modes via `tent_mode`.
#[inline]
fn calculate_src_coord(
    dst_i: usize,
    src_len: usize,
    dst_len: usize,
    scale: f32,
    tent_mode: TentMode,
) -> f32 {
    let max = (src_len - 1) as f32;
    match tent_mode {
        TentMode::Off => {
            // Standard pixel-center mapping
            ((dst_i as f32 + 0.5) * scale - 0.5).clamp(0.0, max)
        }
        TentMode::SampleToSample => {
            // Sample-to-sample mapping: position 0→0, position N-1→M-1
            let tent_scale = if dst_len > 1 {
                (src_len - 1) as f32 / (dst_len - 1) as f32
            } else {
                1.0
            };
            let offset = 0.5 * (1.0 - tent_scale);
            ((dst_i as f32 + 0.5) * tent_scale - 0.5 + offset).clamp(0.0, max)
        }
        TentMode::Prescale => {
            // Tent-to-box prescale mapping:
            // src is tent-space (size 2W+1), dst is box-space (size dstW)
            // Maps dst pixel centers to tent sample positions such that:
            // - dst pixel 0 (center 0.5) → tent position where original pixel 0.5 would be
            // Formula: src_pos = (dst_x + 0.5) * (src_len - 1) / dst_len
            let prescale_factor = (src_len - 1) as f32 / dst_len as f32;
            ((dst_i as f32 + 0.5) * prescale_factor).clamp(0.0, max)
        }
    }
}

/// Rescale Pixel4 array using bilinear interpolation
/// Progress callback is optional - receives 0.0-1.0 after each row
///
/// `tent_mode` controls coordinate mapping:
/// - `TentMode::Off`: Standard pixel-center mapping
/// - `TentMode::SampleToSample`: Sample-to-sample mapping for tent-space
/// - `TentMode::Prescale`: Tent-to-box prescale (integrates tent_contract)
pub fn rescale_bilinear_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    tent_mode: TentMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = calculate_src_coord(dst_x, src_width, dst_width, scale_x, tent_mode);
            let src_y = calculate_src_coord(dst_y, src_height, dst_height, scale_y, tent_mode);

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            // Sample four corners
            let p00 = src[y0 * src_width + x0];
            let p10 = src[y0 * src_width + x1];
            let p01 = src[y1 * src_width + x0];
            let p11 = src[y1 * src_width + x1];

            // Bilinear interpolation using SIMD-friendly lerp
            let top = lerp(p00, p10, fx);
            let bottom = lerp(p01, p11, fx);
            dst[dst_y * dst_width + dst_x] = lerp(top, bottom, fy);
        }
        // Report progress after each row
        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

/// Alpha-aware bilinear interpolation helper
/// Weights RGB by alpha, falls back to unweighted if total alpha ≈ 0
#[inline]
fn bilinear_alpha_aware(
    p00: Pixel4, p10: Pixel4, p01: Pixel4, p11: Pixel4,
    fx: f32, fy: f32,
) -> Pixel4 {
    // Bilinear weights for each corner
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    // Alpha values
    let a00 = p00.a();
    let a10 = p10.a();
    let a01 = p01.a();
    let a11 = p11.a();

    // Alpha-weighted sum for RGB
    let aw00 = w00 * a00;
    let aw10 = w10 * a10;
    let aw01 = w01 * a01;
    let aw11 = w11 * a11;
    let total_alpha_weight = aw00 + aw10 + aw01 + aw11;

    // Interpolate alpha normally
    let out_alpha = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11;

    // RGB: use alpha-weighted interpolation, fall back to normal if all transparent
    let (out_r, out_g, out_b) = if total_alpha_weight > 1e-8 {
        let inv_aw = 1.0 / total_alpha_weight;
        (
            (aw00 * p00.r() + aw10 * p10.r() + aw01 * p01.r() + aw11 * p11.r()) * inv_aw,
            (aw00 * p00.g() + aw10 * p10.g() + aw01 * p01.g() + aw11 * p11.g()) * inv_aw,
            (aw00 * p00.b() + aw10 * p10.b() + aw01 * p01.b() + aw11 * p11.b()) * inv_aw,
        )
    } else {
        // All pixels transparent: use unweighted interpolation to preserve RGB
        (
            w00 * p00.r() + w10 * p10.r() + w01 * p01.r() + w11 * p11.r(),
            w00 * p00.g() + w10 * p10.g() + w01 * p01.g() + w11 * p11.g(),
            w00 * p00.b() + w10 * p10.b() + w01 * p01.b() + w11 * p11.b(),
        )
    };

    Pixel4::new(out_r, out_g, out_b, out_alpha)
}

/// Rescale Pixel4 array using alpha-aware bilinear interpolation
/// RGB channels are weighted by alpha during interpolation to prevent
/// transparent pixels from bleeding color into opaque regions.
/// Fully transparent regions preserve their underlying RGB values.
///
/// `tent_mode` controls coordinate mapping:
/// - `TentMode::Off`: Standard pixel-center mapping
/// - `TentMode::SampleToSample`: Sample-to-sample mapping for tent-space
/// - `TentMode::Prescale`: Tent-to-box prescale (integrates tent_contract)
pub fn rescale_bilinear_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    tent_mode: TentMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            let src_x = calculate_src_coord(dst_x, src_width, dst_width, scale_x, tent_mode);
            let src_y = calculate_src_coord(dst_y, src_height, dst_height, scale_y, tent_mode);

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let p00 = src[y0 * src_width + x0];
            let p10 = src[y0 * src_width + x1];
            let p01 = src[y1 * src_width + x0];
            let p11 = src[y1 * src_width + x1];

            dst[dst_y * dst_width + dst_x] = bilinear_alpha_aware(p00, p10, p01, p11, fx, fy);
        }
        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}
