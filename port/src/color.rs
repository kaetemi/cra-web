/// Color conversion utilities for sRGB, Linear RGB, Lab, and Oklab color spaces.

use crate::colorspace_derived::f32 as cs;
use crate::pixel::Pixel4;

/// Convert sRGB value (0-1) to linear RGB
#[inline]
pub fn srgb_to_linear_single(srgb: f32) -> f32 {
    if srgb <= cs::SRGB_DECODE_THRESHOLD {
        srgb / cs::SRGB_LINEAR_SLOPE
    } else {
        ((srgb + cs::SRGB_OFFSET) / cs::SRGB_SCALE).powf(cs::SRGB_GAMMA)
    }
}

/// Convert linear RGB value (0-1) to sRGB
#[inline]
pub fn linear_to_srgb_single(linear: f32) -> f32 {
    if linear <= cs::SRGB_THRESHOLD {
        linear * cs::SRGB_LINEAR_SLOPE
    } else {
        cs::SRGB_SCALE * linear.max(0.0).powf(1.0 / cs::SRGB_GAMMA) - cs::SRGB_OFFSET
    }
}

// ============================================================================
// Pixel color space conversions
// ============================================================================

#[cfg(test)]
#[inline]
fn srgb_to_linear_pixel(p: Pixel4) -> Pixel4 {
    Pixel4::new(
        srgb_to_linear_single(p[0]),
        srgb_to_linear_single(p[1]),
        srgb_to_linear_single(p[2]),
        p[3],
    )
}

#[cfg(test)]
#[inline]
fn linear_to_srgb_pixel(p: Pixel4) -> Pixel4 {
    Pixel4::new(
        linear_to_srgb_single(p[0]),
        linear_to_srgb_single(p[1]),
        linear_to_srgb_single(p[2]),
        p[3],
    )
}

/// Convert sRGB pixels to linear RGB in-place
pub fn srgb_to_linear_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] = srgb_to_linear_single(p[0]);
        p[1] = srgb_to_linear_single(p[1]);
        p[2] = srgb_to_linear_single(p[2]);
    }
}

/// Convert linear RGB pixels to sRGB in-place
pub fn linear_to_srgb_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] = linear_to_srgb_single(p[0]);
        p[1] = linear_to_srgb_single(p[1]);
        p[2] = linear_to_srgb_single(p[2]);
    }
}

// ============================================================================
// Grayscale conversion (Rec.709 luminance)
// ============================================================================

/// Compute luminance from linear RGB using Rec.709/BT.709 coefficients
#[inline]
pub fn linear_rgb_to_luminance(r: f32, g: f32, b: f32) -> f32 {
    r * cs::YCBCR_KR + g * cs::YCBCR_KG + b * cs::YCBCR_KB
}

/// Convert linear RGB Pixel4 array (0-1 range) to linear grayscale (0-1)
///
/// Only computes luminance using Rec.709 coefficients.
/// Does NOT apply gamma correction or denormalization.
/// Use linear_to_srgb_single + denormalize separately for sRGB output.
pub fn linear_pixels_to_grayscale(pixels: &[Pixel4]) -> Vec<f32> {
    let mut gray = Vec::with_capacity(pixels.len());

    for p in pixels {
        gray.push(linear_rgb_to_luminance(p[0], p[1], p[2]));
    }

    gray
}


/// Lab f(t) function - attempt to linearize cube root near zero.
///
/// The standard CIELAB f(t) function uses a linear segment near zero to avoid
/// the infinite slope of the cube root at t=0. This is defined only for t >= 0
/// in the official specification, since XYZ values are non-negative for in-gamut colors.
///
/// However, during error-diffusion dithering, accumulated quantization error can
/// push adjusted linear RGB values negative, resulting in negative XYZ components.
/// We extend the linear segment symmetrically into negative territory, ensuring
/// continuity at both boundaries:
///   - At t = EPSILON (~0.0089): linear meets cbrt from above
///   - At t = -NEG_EPSILON (~-0.0709): linear meets cbrt from below
///
/// This continuous extension is important for aggressive quantization (e.g., 1-bit)
/// where large errors can accumulate. Without continuity, incorrect Lab distances
/// could cause wrong quantization candidates and visible artifacts.
#[inline]
fn lab_f(t: f32) -> f32 {
    if t > cs::CIELAB_EPSILON {
        // Standard cube root region for normal positive values
        t.cbrt()
    } else if t >= -cs::CIELAB_NEG_EPSILON {
        // Linear segment for values near zero (both positive and negative).
        // This avoids the infinite derivative of cbrt at t=0 and ensures
        // continuity at both EPSILON and -NEG_EPSILON boundaries.
        cs::CIELAB_KAPPA * t + cs::CIELAB_OFFSET
    } else {
        // Very negative out-of-gamut values: use cube root for correct behavior.
        // Continuous with linear segment at t = -NEG_EPSILON.
        t.cbrt()
    }
}

/// Inverse of lab_f - converts from f(t) space back to linear XYZ-normalized space.
///
/// This mirrors the three regions defined in lab_f:
/// - Very negative f-values (< NEG_F_THRESHOLD): came from cbrt of t < -NEG_EPSILON
/// - Middle range [NEG_F_THRESHOLD, F_THRESHOLD]: came from linear segment
/// - Positive f-values (> F_THRESHOLD): came from cbrt of t > EPSILON
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    if t < cs::CIELAB_NEG_F_THRESHOLD {
        // Very negative: originated from cbrt of value < -NEG_EPSILON
        t * t * t
    } else if t > cs::CIELAB_F_THRESHOLD {
        // Above positive threshold: originated from cbrt region
        t * t * t
    } else {
        // Linear region: invert the linear approximation
        (t - cs::CIELAB_OFFSET) / cs::CIELAB_KAPPA
    }
}

/// Convert linear RGB (0-1) to Lab
/// Returns (L, a, b) where L is 0-100 and a,b are roughly -127 to +127
#[inline]
pub fn linear_rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // RGB -> XYZ, normalized by white point
    // Use D65_SRGB (from sRGB matrix) to ensure sRGB white maps exactly to L*=100, a*=0, b*=0
    let x = (cs::SRGB_TO_XYZ[0][0] * r + cs::SRGB_TO_XYZ[0][1] * g + cs::SRGB_TO_XYZ[0][2] * b) / cs::D65_SRGB_X;
    let y = cs::SRGB_TO_XYZ[1][0] * r + cs::SRGB_TO_XYZ[1][1] * g + cs::SRGB_TO_XYZ[1][2] * b; // D65_SRGB_Y = 1.0
    let z = (cs::SRGB_TO_XYZ[2][0] * r + cs::SRGB_TO_XYZ[2][1] * g + cs::SRGB_TO_XYZ[2][2] * b) / cs::D65_SRGB_Z;

    // Apply f(t)
    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);

    // XYZ -> Lab
    let l = cs::CIELAB_L_SCALE * fy - cs::CIELAB_L_OFFSET;
    let a = cs::CIELAB_A_SCALE * (fx - fy);
    let b_ch = cs::CIELAB_B_SCALE * (fy - fz);

    (l, a, b_ch)
}

/// Convert Lab to linear RGB (0-1)
#[inline]
pub fn lab_to_linear_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Lab -> f values
    let fy = (l + cs::CIELAB_L_OFFSET) / cs::CIELAB_L_SCALE;
    let fx = a / cs::CIELAB_A_SCALE + fy;
    let fz = fy - b / cs::CIELAB_B_SCALE;

    // Invert f(t) to get XYZ (normalized)
    let x = lab_f_inv(fx);
    let y = lab_f_inv(fy);
    let z = lab_f_inv(fz);

    // Denormalize by white point (D65_SRGB to match forward conversion)
    let x = x * cs::D65_SRGB_X;
    // y = y * D65_SRGB_Y where D65_SRGB_Y = 1.0
    let z = z * cs::D65_SRGB_Z;

    // XYZ -> linear RGB
    let r = cs::XYZ_TO_SRGB[0][0] * x + cs::XYZ_TO_SRGB[0][1] * y + cs::XYZ_TO_SRGB[0][2] * z;
    let g = cs::XYZ_TO_SRGB[1][0] * x + cs::XYZ_TO_SRGB[1][1] * y + cs::XYZ_TO_SRGB[1][2] * z;
    let b_out = cs::XYZ_TO_SRGB[2][0] * x + cs::XYZ_TO_SRGB[2][1] * y + cs::XYZ_TO_SRGB[2][2] * z;

    (r, g, b_out)
}

/// Convert linear RGB pixel to Lab pixel
/// Returns [L, a, b, _] where L is 0-100 and a,b are roughly -127 to +127
#[inline]
pub fn linear_rgb_to_lab_pixel(p: Pixel4) -> Pixel4 {
    let (l, a, b) = linear_rgb_to_lab(p[0], p[1], p[2]);
    Pixel4::new(l, a, b, p[3])
}

/// Convert Lab pixel to linear RGB pixel
#[inline]
pub fn lab_to_linear_rgb_pixel(p: Pixel4) -> Pixel4 {
    let (r, g, b) = lab_to_linear_rgb(p[0], p[1], p[2]);
    Pixel4::new(r, g, b, p[3])
}

// ============== Oklab color space ==============
// Oklab is a perceptually uniform color space developed by Björn Ottosson (2020)
// L: 0-1, a/b: roughly -0.4 to +0.4

/// Signed cube root - preserves sign for negative values.
/// Required for out-of-gamut colors in OkLab.
#[inline]
fn signed_cbrt(x: f32) -> f32 {
    x.signum() * x.abs().cbrt()
}

/// Convert linear RGB (0-1) to Oklab
/// Returns (L, a, b) where L is 0-1 and a,b are roughly -0.4 to +0.4
#[inline]
pub fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Linear sRGB to LMS
    let l = cs::OKLAB_M1[0][0] * r + cs::OKLAB_M1[0][1] * g + cs::OKLAB_M1[0][2] * b;
    let m = cs::OKLAB_M1[1][0] * r + cs::OKLAB_M1[1][1] * g + cs::OKLAB_M1[1][2] * b;
    let s = cs::OKLAB_M1[2][0] * r + cs::OKLAB_M1[2][1] * g + cs::OKLAB_M1[2][2] * b;

    // Signed cube root (preserves sign for out-of-gamut values)
    let l_ = signed_cbrt(l);
    let m_ = signed_cbrt(m);
    let s_ = signed_cbrt(s);

    // LMS' to Oklab
    let ok_l = cs::OKLAB_M2[0][0] * l_ + cs::OKLAB_M2[0][1] * m_ + cs::OKLAB_M2[0][2] * s_;
    let ok_a = cs::OKLAB_M2[1][0] * l_ + cs::OKLAB_M2[1][1] * m_ + cs::OKLAB_M2[1][2] * s_;
    let ok_b = cs::OKLAB_M2[2][0] * l_ + cs::OKLAB_M2[2][1] * m_ + cs::OKLAB_M2[2][2] * s_;

    (ok_l, ok_a, ok_b)
}

/// Convert Oklab to linear RGB (0-1)
#[inline]
pub fn oklab_to_linear_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Oklab to LMS'
    let l_ = cs::OKLAB_M2_INV[0][0] * l + cs::OKLAB_M2_INV[0][1] * a + cs::OKLAB_M2_INV[0][2] * b;
    let m_ = cs::OKLAB_M2_INV[1][0] * l + cs::OKLAB_M2_INV[1][1] * a + cs::OKLAB_M2_INV[1][2] * b;
    let s_ = cs::OKLAB_M2_INV[2][0] * l + cs::OKLAB_M2_INV[2][1] * a + cs::OKLAB_M2_INV[2][2] * b;

    // Cube LMS'
    let lms_l = l_ * l_ * l_;
    let lms_m = m_ * m_ * m_;
    let lms_s = s_ * s_ * s_;

    // LMS to linear sRGB
    let r = cs::OKLAB_M1_INV[0][0] * lms_l + cs::OKLAB_M1_INV[0][1] * lms_m + cs::OKLAB_M1_INV[0][2] * lms_s;
    let g = cs::OKLAB_M1_INV[1][0] * lms_l + cs::OKLAB_M1_INV[1][1] * lms_m + cs::OKLAB_M1_INV[1][2] * lms_s;
    let b_out = cs::OKLAB_M1_INV[2][0] * lms_l + cs::OKLAB_M1_INV[2][1] * lms_m + cs::OKLAB_M1_INV[2][2] * lms_s;

    (r, g, b_out)
}

/// Convert linear RGB pixel to Oklab pixel
/// Returns [L, a, b, _] where L is 0-1 and a,b are roughly -0.4 to +0.4
#[inline]
pub fn linear_rgb_to_oklab_pixel(p: Pixel4) -> Pixel4 {
    let (l, a, b) = linear_rgb_to_oklab(p[0], p[1], p[2]);
    Pixel4::new(l, a, b, p[3])
}

/// Convert Oklab pixel to linear RGB pixel
#[inline]
pub fn oklab_to_linear_rgb_pixel(p: Pixel4) -> Pixel4 {
    let (r, g, b) = oklab_to_linear_rgb(p[0], p[1], p[2]);
    Pixel4::new(r, g, b, p[3])
}

// ============== Y'CbCr color space ==============
// Y'CbCr is a luma-chroma separation, traditionally applied to gamma-encoded (sRGB) values.
// BT.709 coefficients are used (matching sRGB primaries).
// Y': 0-1, Cb/Cr: roughly -0.5 to +0.5

/// Convert linear RGB (0-1) to Y'CbCr
/// Internally converts to sRGB first, then applies the Y'CbCr matrix.
/// Returns (Y', Cb, Cr) where Y' is 0-1 and Cb,Cr are roughly -0.5 to +0.5
/// Clamps input to 0-1 range before conversion.
#[inline]
pub fn linear_rgb_to_ycbcr_clamped(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Convert linear to sRGB (gamma-encoded)
    let r_srgb = linear_to_srgb_single(r.clamp(0.0, 1.0));
    let g_srgb = linear_to_srgb_single(g.clamp(0.0, 1.0));
    let b_srgb = linear_to_srgb_single(b.clamp(0.0, 1.0));

    // Apply RGB to Y'CbCr matrix (BT.709)
    let y = cs::RGB_TO_YCBCR[0][0] * r_srgb + cs::RGB_TO_YCBCR[0][1] * g_srgb + cs::RGB_TO_YCBCR[0][2] * b_srgb;
    let cb = cs::RGB_TO_YCBCR[1][0] * r_srgb + cs::RGB_TO_YCBCR[1][1] * g_srgb + cs::RGB_TO_YCBCR[1][2] * b_srgb;
    let cr = cs::RGB_TO_YCBCR[2][0] * r_srgb + cs::RGB_TO_YCBCR[2][1] * g_srgb + cs::RGB_TO_YCBCR[2][2] * b_srgb;

    (y, cb, cr)
}

/// Convert linear RGB to Y'CbCr (for out-of-gamut values during dithering)
/// Uses signed sRGB conversion to handle negative values from error accumulation.
#[inline]
pub fn linear_rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Convert linear to sRGB, preserving sign for out-of-gamut values
    let r_srgb = if r >= 0.0 { linear_to_srgb_single(r) } else { -linear_to_srgb_single(-r) };
    let g_srgb = if g >= 0.0 { linear_to_srgb_single(g) } else { -linear_to_srgb_single(-g) };
    let b_srgb = if b >= 0.0 { linear_to_srgb_single(b) } else { -linear_to_srgb_single(-b) };

    // Apply RGB to Y'CbCr matrix (BT.709)
    let y = cs::RGB_TO_YCBCR[0][0] * r_srgb + cs::RGB_TO_YCBCR[0][1] * g_srgb + cs::RGB_TO_YCBCR[0][2] * b_srgb;
    let cb = cs::RGB_TO_YCBCR[1][0] * r_srgb + cs::RGB_TO_YCBCR[1][1] * g_srgb + cs::RGB_TO_YCBCR[1][2] * b_srgb;
    let cr = cs::RGB_TO_YCBCR[2][0] * r_srgb + cs::RGB_TO_YCBCR[2][1] * g_srgb + cs::RGB_TO_YCBCR[2][2] * b_srgb;

    (y, cb, cr)
}

// ============== Output utilities ==============

/// Interleave three u8 channels into RGB output
pub fn interleave_rgb_u8(r: &[u8], g: &[u8], b: &[u8]) -> Vec<u8> {
    let pixels = r.len();
    let mut out = vec![0u8; pixels * 3];

    for i in 0..pixels {
        out[i * 3] = r[i];
        out[i * 3 + 1] = g[i];
        out[i * 3 + 2] = b[i];
    }

    out
}

/// Interleave four u8 channels into RGBA output
pub fn interleave_rgba_u8(r: &[u8], g: &[u8], b: &[u8], a: &[u8]) -> Vec<u8> {
    let pixels = r.len();
    let mut out = vec![0u8; pixels * 4];

    for i in 0..pixels {
        out[i * 4] = r[i];
        out[i * 4 + 1] = g[i];
        out[i * 4 + 2] = b[i];
        out[i * 4 + 3] = a[i];
    }

    out
}

// ============================================================================
// Pixel scale operations
// ============================================================================

/// Scale pixel values from 0-1 to 0-255 in-place (all 4 channels including alpha)
pub fn scale_to_255_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] *= 255.0;
        p[1] *= 255.0;
        p[2] *= 255.0;
        p[3] *= 255.0;
    }
}

/// Scale pixel values from 0-255 to 0-1 in-place (all 4 channels including alpha)
pub fn scale_from_255_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] /= 255.0;
        p[1] /= 255.0;
        p[2] /= 255.0;
        p[3] /= 255.0;
    }
}

#[cfg(test)]
#[inline]
fn scale_to_255_pixel(p: Pixel4) -> Pixel4 {
    Pixel4::new(p[0] * 255.0, p[1] * 255.0, p[2] * 255.0, p[3] * 255.0)
}

#[cfg(test)]
#[inline]
fn scale_from_255_pixel(p: Pixel4) -> Pixel4 {
    Pixel4::new(p[0] / 255.0, p[1] / 255.0, p[2] / 255.0, p[3] / 255.0)
}

#[cfg(test)]
#[allow(dead_code)]
#[inline]
fn linear_to_srgb_255_pixel(p: Pixel4) -> Pixel4 {
    Pixel4::new(
        (linear_to_srgb_single(p[0]) * 255.0).clamp(0.0, 255.0),
        (linear_to_srgb_single(p[1]) * 255.0).clamp(0.0, 255.0),
        (linear_to_srgb_single(p[2]) * 255.0).clamp(0.0, 255.0),
        p[3],
    )
}

#[cfg(test)]
#[allow(dead_code)]
#[inline]
fn srgb_255_to_linear_pixel(p: Pixel4) -> Pixel4 {
    Pixel4::new(
        srgb_to_linear_single(p[0] / 255.0),
        srgb_to_linear_single(p[1] / 255.0),
        srgb_to_linear_single(p[2] / 255.0),
        p[3],
    )
}

/// Normalize pixels from 0-255 to 0-1 range in-place (all 4 channels including alpha)
pub fn normalize_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] /= 255.0;
        p[1] /= 255.0;
        p[2] /= 255.0;
        p[3] /= 255.0;
    }
}

/// Denormalize pixels from 0-1 to 0-255 range in-place (all 4 channels including alpha)
/// Clamps output to 0-255 range.
pub fn denormalize_inplace_clamped(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] = (p[0] * 255.0).clamp(0.0, 255.0);
        p[1] = (p[1] * 255.0).clamp(0.0, 255.0);
        p[2] = (p[2] * 255.0).clamp(0.0, 255.0);
        p[3] = (p[3] * 255.0).clamp(0.0, 255.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_linear_roundtrip() {
        let test_values = [0.0, 0.04045, 0.1, 0.5, 1.0];
        for &v in &test_values {
            let linear = srgb_to_linear_single(v);
            let back = linear_to_srgb_single(linear);
            assert!((v - back).abs() < 1e-5, "Failed at {}: got {}", v, back);
        }
    }

    #[test]
    fn test_lab_roundtrip() {
        let test_rgb = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.7)];
        for (r, g, b) in test_rgb {
            let (l, a, b_ch) = linear_rgb_to_lab(r, g, b);
            let (r2, g2, b2) = lab_to_linear_rgb(l, a, b_ch);
            assert!((r - r2).abs() < 1e-4, "R failed: {} vs {}", r, r2);
            assert!((g - g2).abs() < 1e-4, "G failed: {} vs {}", g, g2);
            assert!((b - b2).abs() < 1e-4, "B failed: {} vs {}", b, b2);
        }
    }

    #[test]
    fn test_lab_white_is_neutral() {
        // sRGB white (1,1,1) should map to exactly L*=100, a*=0, b*=0
        // This requires using D65_SRGB (from sRGB matrix) as reference white
        let (l, a, b) = linear_rgb_to_lab(1.0, 1.0, 1.0);
        assert!((l - 100.0).abs() < 1e-4, "L* should be 100, got {}", l);
        assert!(a.abs() < 1e-4, "a* should be 0, got {}", a);
        assert!(b.abs() < 1e-4, "b* should be 0, got {}", b);
    }

    #[test]
    fn test_oklab_roundtrip() {
        let test_rgb = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.7), (0.2, 0.8, 0.4)];
        for (r, g, b) in test_rgb {
            let (l, a, b_ch) = linear_rgb_to_oklab(r, g, b);
            let (r2, g2, b2) = oklab_to_linear_rgb(l, a, b_ch);
            assert!((r - r2).abs() < 1e-4, "R failed: {} vs {}", r, r2);
            assert!((g - g2).abs() < 1e-4, "G failed: {} vs {}", g, g2);
            assert!((b - b2).abs() < 1e-4, "B failed: {} vs {}", b, b2);
        }
    }

    #[test]
    fn test_lab_f_roundtrip_negative() {
        // Verify that lab_f and lab_f_inv are proper inverses for negative values
        // (out-of-gamut colors that can occur during error diffusion)
        let test_values = [-1.0, -0.5, -0.1, -0.01, -0.001];
        for &t in &test_values {
            let f_t = lab_f(t);
            let back = lab_f_inv(f_t);
            assert!(
                (t - back).abs() < 1e-6,
                "lab_f roundtrip failed for {}: f({}) = {}, inv = {}",
                t, t, f_t, back
            );
        }
    }

    #[test]
    fn test_lab_f_continuity() {
        // Verify continuity at the positive boundary (EPSILON ≈ 0.0089)
        let eps = cs::CIELAB_EPSILON;
        let f_below = lab_f(eps - 1e-6);
        let f_at = lab_f(eps);
        let f_above = lab_f(eps + 1e-6);
        assert!(
            (f_below - f_at).abs() < 1e-4,
            "Discontinuity at EPSILON: below={}, at={}",
            f_below, f_at
        );
        assert!(
            (f_above - f_at).abs() < 1e-4,
            "Discontinuity at EPSILON: at={}, above={}",
            f_at, f_above
        );

        // Verify continuity at the negative boundary (-NEG_EPSILON ≈ -0.0709)
        let neg_eps = -cs::CIELAB_NEG_EPSILON;
        let f_below_neg = lab_f(neg_eps - 1e-6);
        let f_at_neg = lab_f(neg_eps);
        let f_above_neg = lab_f(neg_eps + 1e-6);
        assert!(
            (f_below_neg - f_at_neg).abs() < 1e-4,
            "Discontinuity at -NEG_EPSILON: below={}, at={}",
            f_below_neg, f_at_neg
        );
        assert!(
            (f_above_neg - f_at_neg).abs() < 1e-4,
            "Discontinuity at -NEG_EPSILON: at={}, above={}",
            f_at_neg, f_above_neg
        );

        // Verify very negative values use cbrt
        let f_very_neg = lab_f(-0.5);
        assert!(
            (f_very_neg + (0.5_f32).cbrt()).abs() < 1e-6,
            "lab_f(-0.5) should equal -cbrt(0.5)"
        );
    }

    // ============== Pixel tests ==============

    #[test]
    fn test_srgb_linear_pixel_roundtrip() {
        let test_pixels: [Pixel4; 3] = [
            Pixel4::new(0.0, 0.5, 1.0, 0.0),
            Pixel4::new(0.1, 0.3, 0.7, 0.0),
            Pixel4::new(0.04045, 0.2, 0.8, 0.0),
        ];
        for p in test_pixels {
            let linear = srgb_to_linear_pixel(p);
            let back = linear_to_srgb_pixel(linear);
            assert!((p[0] - back[0]).abs() < 1e-5, "R failed");
            assert!((p[1] - back[1]).abs() < 1e-5, "G failed");
            assert!((p[2] - back[2]).abs() < 1e-5, "B failed");
        }
    }

    #[test]
    fn test_lab_pixel_roundtrip() {
        let test_pixels: [Pixel4; 3] = [
            Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.5, 0.3, 0.7, 0.0),
        ];
        for p in test_pixels {
            let lab = linear_rgb_to_lab_pixel(p);
            let back = lab_to_linear_rgb_pixel(lab);
            assert!((p[0] - back[0]).abs() < 1e-4, "R failed: {} vs {}", p[0], back[0]);
            assert!((p[1] - back[1]).abs() < 1e-4, "G failed: {} vs {}", p[1], back[1]);
            assert!((p[2] - back[2]).abs() < 1e-4, "B failed: {} vs {}", p[2], back[2]);
        }
    }

    #[test]
    fn test_oklab_pixel_roundtrip() {
        let test_pixels: [Pixel4; 3] = [
            Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.5, 0.3, 0.7, 0.0),
        ];
        for p in test_pixels {
            let oklab = linear_rgb_to_oklab_pixel(p);
            let back = oklab_to_linear_rgb_pixel(oklab);
            assert!((p[0] - back[0]).abs() < 1e-4, "R failed: {} vs {}", p[0], back[0]);
            assert!((p[1] - back[1]).abs() < 1e-4, "G failed: {} vs {}", p[1], back[1]);
            assert!((p[2] - back[2]).abs() < 1e-4, "B failed: {} vs {}", p[2], back[2]);
        }
    }

    #[test]
    fn test_scale_roundtrip() {
        let p = Pixel4::new(0.5, 0.25, 1.0, 0.0);
        let scaled = scale_to_255_pixel(p);
        assert!((scaled[0] - 127.5).abs() < 1e-5);
        assert!((scaled[1] - 63.75).abs() < 1e-5);
        assert!((scaled[2] - 255.0).abs() < 1e-5);

        let back = scale_from_255_pixel(scaled);
        assert!((p[0] - back[0]).abs() < 1e-5);
        assert!((p[1] - back[1]).abs() < 1e-5);
        assert!((p[2] - back[2]).abs() < 1e-5);
    }

    #[test]
    fn test_inplace_conversions() {
        let mut pixels: Vec<Pixel4> = vec![
            Pixel4::new(0.0, 0.5, 1.0, 0.0),
            Pixel4::new(0.2, 0.4, 0.8, 0.0),
        ];
        let original = pixels.clone();

        // Test sRGB -> linear -> sRGB roundtrip in-place
        srgb_to_linear_inplace(&mut pixels);
        linear_to_srgb_inplace(&mut pixels);

        for (orig, result) in original.iter().zip(pixels.iter()) {
            assert!((orig[0] - result[0]).abs() < 1e-5);
            assert!((orig[1] - result[1]).abs() < 1e-5);
            assert!((orig[2] - result[2]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_pixel_preserves_alpha() {
        let p = Pixel4::new(0.5, 0.3, 0.7, 0.99);

        // Color space conversions should preserve the alpha channel (alpha is linear)
        assert_eq!(srgb_to_linear_pixel(p)[3], 0.99);
        assert_eq!(linear_to_srgb_pixel(p)[3], 0.99);
        assert_eq!(linear_rgb_to_lab_pixel(p)[3], 0.99);
        assert_eq!(lab_to_linear_rgb_pixel(p)[3], 0.99);
        assert_eq!(linear_rgb_to_oklab_pixel(p)[3], 0.99);
        assert_eq!(oklab_to_linear_rgb_pixel(p)[3], 0.99);
    }

    #[test]
    fn test_scale_scales_alpha() {
        // Scale operations should scale all 4 channels including alpha
        let p = Pixel4::new(0.5, 0.3, 0.7, 0.8);

        let scaled = scale_to_255_pixel(p);
        assert!((scaled[0] - 127.5).abs() < 1e-5);
        assert!((scaled[1] - 76.5).abs() < 1e-5);
        assert!((scaled[2] - 178.5).abs() < 1e-5);
        assert!((scaled[3] - 204.0).abs() < 1e-5); // 0.8 * 255 = 204

        let back = scale_from_255_pixel(scaled);
        assert!((back[0] - 0.5).abs() < 1e-5);
        assert!((back[1] - 0.3).abs() < 1e-5);
        assert!((back[2] - 0.7).abs() < 1e-5);
        assert!((back[3] - 0.8).abs() < 1e-5);
    }
}
