/// Color conversion utilities for sRGB, Linear RGB, Lab, and Oklab color spaces.

use crate::colorspace_derived::f32 as cs;
use crate::pixel::Pixel4;

/// Convert sRGB value to linear RGB.
///
/// Supports the full sYCC extended gamut (IEC 61966-2-1 Amendment 1) with
/// point-symmetric handling of negative values for out-of-gamut colors.
#[inline]
pub fn srgb_to_linear_single(srgb: f32) -> f32 {
    if srgb < -cs::SRGB_DECODE_THRESHOLD {
        // Negative power curve region (sYCC extended gamut)
        -((-srgb + cs::SRGB_OFFSET) / cs::SRGB_SCALE).powf(cs::SRGB_GAMMA)
    } else if srgb <= cs::SRGB_DECODE_THRESHOLD {
        // Linear region (handles both positive and negative near zero)
        srgb / cs::SRGB_LINEAR_SLOPE
    } else {
        // Positive power curve region
        ((srgb + cs::SRGB_OFFSET) / cs::SRGB_SCALE).powf(cs::SRGB_GAMMA)
    }
}

/// Convert linear RGB value to sRGB.
///
/// Supports the full sYCC extended gamut (IEC 61966-2-1 Amendment 1) with
/// point-symmetric handling of negative values for out-of-gamut colors.
#[inline]
pub fn linear_to_srgb_single(linear: f32) -> f32 {
    if linear < -cs::SRGB_THRESHOLD {
        // Negative power curve region (sYCC extended gamut)
        -cs::SRGB_SCALE * (-linear).powf(1.0 / cs::SRGB_GAMMA) + cs::SRGB_OFFSET
    } else if linear <= cs::SRGB_THRESHOLD {
        // Linear region (handles both positive and negative near zero)
        linear * cs::SRGB_LINEAR_SLOPE
    } else {
        // Positive power curve region
        cs::SRGB_SCALE * linear.powf(1.0 / cs::SRGB_GAMMA) - cs::SRGB_OFFSET
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

// ============================================================================
// ACES Tonemapping
// ============================================================================

// ACES filmic tonemapping coefficients
const ACES_A: f32 = 2.51;
const ACES_B: f32 = 0.03;
const ACES_C: f32 = 2.43;
const ACES_D: f32 = 0.59;
const ACES_E: f32 = 0.14;

/// Apply ACES filmic tonemapping to a single linear value.
/// Maps HDR values to SDR range [0, 1].
#[inline]
pub fn tonemap_aces_single(value: f32) -> f32 {
    let v = value.max(0.0);
    let result = (v * (ACES_A * v + ACES_B)) / (v * (ACES_C * v + ACES_D) + ACES_E);
    result.min(1.0)
}

/// Apply inverse ACES tonemapping to a single value.
/// Maps SDR values back to HDR range (approximate inverse).
/// Uses the quadratic formula solution for the ACES curve.
#[inline]
pub fn tonemap_aces_inverse_single(y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    // Avoid division by zero when y approaches a/c
    let denom = 2.0 * (ACES_A - ACES_C * y);
    if denom.abs() < 1e-6 {
        return y;
    }
    let discriminant = 4.0 * ACES_E * y * (ACES_A - ACES_C * y) + (ACES_B - ACES_D * y).powi(2);
    if discriminant < 0.0 {
        return y;
    }
    let numerator = -ACES_B + ACES_D * y + discriminant.sqrt();
    (numerator / denom).max(0.0)
}

/// Apply ACES tonemapping to linear RGB pixels in-place.
/// Operates on each RGB channel independently.
pub fn tonemap_aces_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] = tonemap_aces_single(p[0]);
        p[1] = tonemap_aces_single(p[1]);
        p[2] = tonemap_aces_single(p[2]);
    }
}

/// Apply inverse ACES tonemapping to linear RGB pixels in-place.
/// Operates on each RGB channel independently.
pub fn tonemap_aces_inverse_inplace(pixels: &mut [Pixel4]) {
    for p in pixels.iter_mut() {
        p[0] = tonemap_aces_inverse_single(p[0]);
        p[1] = tonemap_aces_inverse_single(p[1]);
        p[2] = tonemap_aces_inverse_single(p[2]);
    }
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
    // RGB -> XYZ, normalized by D65 white point
    // sRGB matrices are derived from D65 (0.3127, 0.3290), so sRGB white maps to D65 XYZ.
    let x = (cs::SRGB_TO_XYZ[0][0] * r + cs::SRGB_TO_XYZ[0][1] * g + cs::SRGB_TO_XYZ[0][2] * b) / cs::D65_X;
    let y = cs::SRGB_TO_XYZ[1][0] * r + cs::SRGB_TO_XYZ[1][1] * g + cs::SRGB_TO_XYZ[1][2] * b; // D65_Y = 1.0
    let z = (cs::SRGB_TO_XYZ[2][0] * r + cs::SRGB_TO_XYZ[2][1] * g + cs::SRGB_TO_XYZ[2][2] * b) / cs::D65_Z;

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

    // Denormalize by D65 white point (to match forward conversion)
    let x = x * cs::D65_X;
    // y = y * D65_Y where D65_Y = 1.0
    let z = z * cs::D65_Z;

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

/// Convert OkLab L to revised lightness Lr (Ottosson's formula)
/// Lr better matches Munsell Value and expands dark values compared to L.
/// This helps dark colors stay distinct from grays in distance calculations.
///
/// Formula: Lr = 0.5 * (k3*L - k1 + sqrt((k3*L - k1)² + 4*k2*k3*L))
/// where k1 = 0.206, k2 = 0.03, k3 = (1+k1)/(1+k2)
#[inline]
pub fn oklab_L_to_Lr(l: f32) -> f32 {
    const K1: f32 = 0.206;
    const K2: f32 = 0.03;
    const K3: f32 = (1.0 + K1) / (1.0 + K2); // ≈ 1.170873786

    let k3l = K3 * l;
    let k3l_minus_k1 = k3l - K1;

    0.5 * (k3l_minus_k1 + (k3l_minus_k1 * k3l_minus_k1 + 4.0 * K2 * k3l).sqrt())
}

/// Convert linear RGB to OkLab with revised lightness (Lr, a, b)
/// Uses Ottosson's Lr formula for lightness instead of standard L.
#[inline]
pub fn linear_rgb_to_oklab_lr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (l, a, b_ch) = linear_rgb_to_oklab(r, g, b);
    (oklab_L_to_Lr(l), a, b_ch)
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

/// Convert linear RGB to Y'CbCr (supports out-of-gamut values during dithering).
///
/// Uses the sYCC extended gamut transfer function which natively handles
/// negative values from error accumulation via point-symmetric extension.
#[inline]
pub fn linear_rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Convert linear to sRGB (sYCC transfer handles negatives natively)
    let r_srgb = linear_to_srgb_single(r);
    let g_srgb = linear_to_srgb_single(g);
    let b_srgb = linear_to_srgb_single(b);

    // Apply RGB to Y'CbCr matrix (BT.709)
    let y = cs::RGB_TO_YCBCR[0][0] * r_srgb + cs::RGB_TO_YCBCR[0][1] * g_srgb + cs::RGB_TO_YCBCR[0][2] * b_srgb;
    let cb = cs::RGB_TO_YCBCR[1][0] * r_srgb + cs::RGB_TO_YCBCR[1][1] * g_srgb + cs::RGB_TO_YCBCR[1][2] * b_srgb;
    let cr = cs::RGB_TO_YCBCR[2][0] * r_srgb + cs::RGB_TO_YCBCR[2][1] * g_srgb + cs::RGB_TO_YCBCR[2][2] * b_srgb;

    (y, cb, cr)
}

// ============== Y'CbCr BT.601 (legacy) color space ==============
// Y'CbCr using legacy BT.601 coefficients (0.299/0.587/0.114).
// This is the JPEG/ITU-T T.871 encoding, historically from NTSC 1953.
// Applied to gamma-encoded (sRGB) values despite coefficient mismatch.
// Y': 0-1, Cb/Cr: roughly -0.5 to +0.5

/// Convert linear RGB (0-1) to Y'CbCr using BT.601 (legacy) coefficients.
/// Internally converts to sRGB first, then applies the BT.601 Y'CbCr matrix.
/// Returns (Y', Cb, Cr) where Y' is 0-1 and Cb,Cr are roughly -0.5 to +0.5
/// Clamps input to 0-1 range before conversion.
#[inline]
pub fn linear_rgb_to_ycbcr_bt601_clamped(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Convert linear to sRGB (gamma-encoded)
    let r_srgb = linear_to_srgb_single(r.clamp(0.0, 1.0));
    let g_srgb = linear_to_srgb_single(g.clamp(0.0, 1.0));
    let b_srgb = linear_to_srgb_single(b.clamp(0.0, 1.0));

    // Apply RGB to Y'CbCr matrix (BT.601 legacy coefficients)
    // Y'  = Kr*R' + Kg*G' + Kb*B'
    // Cb = 0.5*(B'-Y')/(1-Kb)
    // Cr = 0.5*(R'-Y')/(1-Kr)
    let y = cs::BT601_KR * r_srgb + cs::BT601_KG * g_srgb + cs::BT601_KB * b_srgb;
    let cb = cs::BT601_CB_R * r_srgb + cs::BT601_CB_G * g_srgb + cs::BT601_CB_B * b_srgb;
    let cr = cs::BT601_CR_R * r_srgb + cs::BT601_CR_G * g_srgb + cs::BT601_CR_B * b_srgb;

    (y, cb, cr)
}

/// Convert linear RGB to Y'CbCr BT.601 (supports out-of-gamut values during dithering).
///
/// Uses the sYCC extended gamut transfer function which natively handles
/// negative values from error accumulation via point-symmetric extension.
#[inline]
pub fn linear_rgb_to_ycbcr_bt601(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Convert linear to sRGB (sYCC transfer handles negatives natively)
    let r_srgb = linear_to_srgb_single(r);
    let g_srgb = linear_to_srgb_single(g);
    let b_srgb = linear_to_srgb_single(b);

    // Apply RGB to Y'CbCr matrix (BT.601 legacy coefficients)
    let y = cs::BT601_KR * r_srgb + cs::BT601_KG * g_srgb + cs::BT601_KB * b_srgb;
    let cb = cs::BT601_CB_R * r_srgb + cs::BT601_CB_G * g_srgb + cs::BT601_CB_B * b_srgb;
    let cr = cs::BT601_CR_R * r_srgb + cs::BT601_CR_G * g_srgb + cs::BT601_CR_B * b_srgb;

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

/// Interleave two u8 channels into LA (Luminosity+Alpha) output
pub fn interleave_la_u8(l: &[u8], a: &[u8]) -> Vec<u8> {
    let pixels = l.len();
    let mut out = vec![0u8; pixels * 2];

    for i in 0..pixels {
        out[i * 2] = l[i];
        out[i * 2 + 1] = a[i];
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

    // ============== sRGB Transfer Function Tests ==============
    //
    // The sRGB transfer function constants (THRESHOLD, LINEAR_SLOPE) are derived
    // from the authoritative curve parameters (γ=2.4, offset=0.055) to ensure
    // both value AND slope continuity at the junction. These tests verify that
    // the derived constants produce a mathematically consistent, fully reversible
    // transfer function.

    #[test]
    fn test_srgb_black_white_exact() {
        // Black and white must round-trip exactly (no floating point error)
        assert_eq!(srgb_to_linear_single(0.0), 0.0, "sRGB 0 should decode to linear 0");
        assert_eq!(linear_to_srgb_single(0.0), 0.0, "linear 0 should encode to sRGB 0");

        // White: 1.0 should round-trip within f32 epsilon
        let white_linear = srgb_to_linear_single(1.0);
        let white_back = linear_to_srgb_single(white_linear);
        assert!(
            (1.0 - white_back).abs() < 1e-6,
            "white round-trip failed: 1.0 -> {} -> {}",
            white_linear, white_back
        );

        let white_srgb = linear_to_srgb_single(1.0);
        let white_linear_back = srgb_to_linear_single(white_srgb);
        assert!(
            (1.0 - white_linear_back).abs() < 1e-6,
            "white round-trip failed: 1.0 -> {} -> {}",
            white_srgb, white_linear_back
        );
    }

    #[test]
    fn test_srgb_threshold_continuity() {
        // The transfer function should be continuous at the threshold.
        // With derived constants, both value AND slope should match.
        let t = cs::SRGB_THRESHOLD;
        let epsilon = 1e-6_f32;

        // Value just below threshold (linear segment)
        let below = t - epsilon;
        let f_below = linear_to_srgb_single(below);

        // Value at threshold
        let f_at = linear_to_srgb_single(t);

        // Value just above threshold (power curve)
        let above = t + epsilon;
        let f_above = linear_to_srgb_single(above);

        // Check continuity: f(t-ε) ≈ f(t) ≈ f(t+ε)
        assert!(
            (f_below - f_at).abs() < 1e-4,
            "sRGB transfer discontinuity below threshold: f({}) = {}, f({}) = {}",
            below, f_below, t, f_at
        );
        assert!(
            (f_above - f_at).abs() < 1e-4,
            "sRGB transfer discontinuity above threshold: f({}) = {}, f({}) = {}",
            t, f_at, above, f_above
        );

        // Check that the function is monotonically increasing
        assert!(f_below < f_at, "sRGB transfer not monotonic at threshold");
        assert!(f_at < f_above, "sRGB transfer not monotonic at threshold");
    }

    #[test]
    fn test_srgb_decode_threshold_continuity() {
        // Same test but for the decode direction
        let t = cs::SRGB_DECODE_THRESHOLD;
        let epsilon = 1e-6_f32;

        let below = t - epsilon;
        let f_below = srgb_to_linear_single(below);

        let f_at = srgb_to_linear_single(t);

        let above = t + epsilon;
        let f_above = srgb_to_linear_single(above);

        assert!(
            (f_below - f_at).abs() < 1e-4,
            "sRGB decode discontinuity below threshold: f({}) = {}, f({}) = {}",
            below, f_below, t, f_at
        );
        assert!(
            (f_above - f_at).abs() < 1e-4,
            "sRGB decode discontinuity above threshold: f({}) = {}, f({}) = {}",
            t, f_at, above, f_above
        );

        assert!(f_below < f_at, "sRGB decode not monotonic at threshold");
        assert!(f_at < f_above, "sRGB decode not monotonic at threshold");
    }

    #[test]
    fn test_srgb_full_range_roundtrip() {
        // Test round-trip across the full [0, 1] range with fine granularity
        // Direction 1: linear -> sRGB -> linear
        let mut max_error_linear = 0.0_f32;
        for i in 0..=1000 {
            let linear = i as f32 / 1000.0;
            let srgb = linear_to_srgb_single(linear);
            let back = srgb_to_linear_single(srgb);
            let error = (linear - back).abs();
            max_error_linear = max_error_linear.max(error);
        }
        assert!(
            max_error_linear < 1e-5,
            "linear->sRGB->linear max error too large: {}",
            max_error_linear
        );

        // Direction 2: sRGB -> linear -> sRGB
        let mut max_error_srgb = 0.0_f32;
        for i in 0..=1000 {
            let srgb = i as f32 / 1000.0;
            let linear = srgb_to_linear_single(srgb);
            let back = linear_to_srgb_single(linear);
            let error = (srgb - back).abs();
            max_error_srgb = max_error_srgb.max(error);
        }
        assert!(
            max_error_srgb < 1e-5,
            "sRGB->linear->sRGB max error too large: {}",
            max_error_srgb
        );
    }

    #[test]
    fn test_srgb_threshold_roundtrip() {
        // The threshold itself should round-trip exactly (this is the critical point)
        let t = cs::SRGB_THRESHOLD;
        let srgb = linear_to_srgb_single(t);
        let back = srgb_to_linear_single(srgb);
        assert!(
            (t - back).abs() < 1e-6,
            "threshold round-trip failed: {} -> {} -> {} (error {})",
            t, srgb, back, (t - back).abs()
        );

        // The decode threshold should also round-trip
        let dt = cs::SRGB_DECODE_THRESHOLD;
        let linear = srgb_to_linear_single(dt);
        let back = linear_to_srgb_single(linear);
        assert!(
            (dt - back).abs() < 1e-6,
            "decode threshold round-trip failed: {} -> {} -> {} (error {})",
            dt, linear, back, (dt - back).abs()
        );

        // Verify DECODE_THRESHOLD ≈ THRESHOLD * LINEAR_SLOPE
        let expected_dt = cs::SRGB_THRESHOLD * cs::SRGB_LINEAR_SLOPE;
        assert!(
            (dt - expected_dt).abs() < 1e-6,
            "DECODE_THRESHOLD ({}) != THRESHOLD * LINEAR_SLOPE ({})",
            dt, expected_dt
        );
    }

    #[test]
    fn test_srgb_constants_approximate_spec() {
        // Verify our derived constants round approximately to the IEC 61966-2-1 spec values
        // Spec: threshold ≈ 0.0031308, slope ≈ 12.92

        // Threshold should be within ~10% of spec (we derive ~0.00304 vs spec 0.0031308)
        assert!(
            (cs::SRGB_THRESHOLD - 0.0031308).abs() < 0.001,
            "SRGB_THRESHOLD {} differs too much from spec 0.0031308",
            cs::SRGB_THRESHOLD
        );

        // Slope should be very close to spec (we derive ~12.923 vs spec 12.92)
        assert!(
            (cs::SRGB_LINEAR_SLOPE - 12.92).abs() < 0.01,
            "SRGB_LINEAR_SLOPE {} differs too much from spec 12.92",
            cs::SRGB_LINEAR_SLOPE
        );

        // Gamma, scale, offset should be exact spec values
        assert_eq!(cs::SRGB_GAMMA, 2.4, "SRGB_GAMMA should be exactly 2.4");
        assert_eq!(cs::SRGB_SCALE, 1.055, "SRGB_SCALE should be exactly 1.055");
        assert_eq!(cs::SRGB_OFFSET, 0.055, "SRGB_OFFSET should be exactly 0.055");
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

    // ============== sRGB Smooth-Join vs Standard Spec Cross-Conversion Test ==============
    //
    // The smooth-join sRGB constants (SRGB_THRESHOLD, SRGB_LINEAR_SLOPE) are derived
    // from γ=2.4 and offset=0.055 to ensure continuous value AND slope at the junction.
    // This differs slightly from the IEC 61966-2-1 spec values (0.0031308, 12.92).
    //
    // This test verifies that cross-converting between implementations produces
    // identical u8 values, proving the smooth-join optimization has no practical
    // impact at 8-bit precision.

    /// Standard sRGB EOTF (decode: encoded -> linear) using IEC 61966-2-1 spec values.
    /// Threshold: 0.04045, slope: 12.92
    fn srgb_decode_standard(u: f32) -> f32 {
        if u <= 0.04045 {
            u / 12.92
        } else {
            ((u + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Standard sRGB OETF (encode: linear -> encoded) using IEC 61966-2-1 spec values.
    /// Threshold: 0.0031308, slope: 12.92
    fn srgb_encode_standard(x: f32) -> f32 {
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }

    /// Quantize to 8-bit with round-half-up
    fn quantize_u8(v: f32) -> u8 {
        let clamped = v.clamp(0.0, 1.0);
        (clamped * 255.0 + 0.5).floor() as u8
    }

    #[test]
    fn test_srgb_smooth_vs_standard_u8_encode() {
        // Test: encode linear samples (i/255) with both implementations.
        // Verifies that the smooth-join OETF produces the same u8 codes as the
        // standard spec OETF for all 256 possible 8-bit linear input values.
        let mut max_delta = 0i32;
        let mut diff_count = 0;

        for i in 0..=255 {
            let linear = i as f32 / 255.0;
            let std_code = quantize_u8(srgb_encode_standard(linear));
            let smooth_code = quantize_u8(linear_to_srgb_single(linear));
            let delta = (smooth_code as i32 - std_code as i32).abs();
            if delta != 0 {
                diff_count += 1;
            }
            max_delta = max_delta.max(delta);
        }

        assert_eq!(
            max_delta, 0,
            "Smooth-join encoding differs from standard: {} codes differ, max delta = {}",
            diff_count, max_delta
        );
    }

    #[test]
    fn test_srgb_smooth_vs_standard_u8_roundtrip() {
        // Test: decode sRGB codes with STANDARD EOTF, re-encode with SMOOTH OETF.
        // This is the critical test for cross-implementation compatibility.
        // If existing 8-bit sRGB content is decoded by standard software and
        // re-encoded by our smooth implementation, codes must be preserved.
        let mut max_delta = 0i32;
        let mut diff_count = 0;
        let mut first_diff: Option<(u8, u8)> = None;

        for c in 0..=255u8 {
            let u = c as f32 / 255.0;
            // Decode with STANDARD spec
            let linear = srgb_decode_standard(u);
            // Re-encode with SMOOTH (our implementation)
            let back_code = quantize_u8(linear_to_srgb_single(linear));
            let delta = (back_code as i32 - c as i32).abs();
            if delta != 0 {
                diff_count += 1;
                if first_diff.is_none() {
                    first_diff = Some((c, back_code));
                }
            }
            max_delta = max_delta.max(delta);
        }

        assert_eq!(
            max_delta, 0,
            "Standard decode -> smooth encode roundtrip fails: {} codes differ, max delta = {}, first diff: {:?}",
            diff_count, max_delta, first_diff
        );
    }

    #[test]
    fn test_srgb_standard_vs_smooth_u8_roundtrip() {
        // Inverse test: decode sRGB codes with SMOOTH EOTF (our implementation),
        // re-encode with STANDARD OETF. Verifies symmetry.
        let mut max_delta = 0i32;
        let mut diff_count = 0;
        let mut first_diff: Option<(u8, u8)> = None;

        for c in 0..=255u8 {
            let u = c as f32 / 255.0;
            // Decode with SMOOTH (our implementation)
            let linear = srgb_to_linear_single(u);
            // Re-encode with STANDARD spec
            let back_code = quantize_u8(srgb_encode_standard(linear));
            let delta = (back_code as i32 - c as i32).abs();
            if delta != 0 {
                diff_count += 1;
                if first_diff.is_none() {
                    first_diff = Some((c, back_code));
                }
            }
            max_delta = max_delta.max(delta);
        }

        assert_eq!(
            max_delta, 0,
            "Smooth decode -> standard encode roundtrip fails: {} codes differ, max delta = {}, first diff: {:?}",
            diff_count, max_delta, first_diff
        );
    }

    // ============== sYCC Extended Gamut (Negative Value) Tests ==============
    //
    // IEC 61966-2-1 Amendment 1 defines point-symmetric extension of the sRGB
    // transfer function for negative values, enabling representation of colors
    // outside the sRGB triangle but still within XYZ gamut.

    #[test]
    fn test_srgb_negative_symmetry() {
        // The transfer function should be point-symmetric around (0, 0).
        // f(-x) = -f(x) for all x
        let test_values = [0.001, 0.003, 0.01, 0.1, 0.5, 1.0, 2.0];

        for &x in &test_values {
            let pos = linear_to_srgb_single(x);
            let neg = linear_to_srgb_single(-x);
            assert!(
                (neg + pos).abs() < 1e-6,
                "Encode not point-symmetric: f({}) = {}, f({}) = {}, sum = {}",
                x, pos, -x, neg, neg + pos
            );
        }

        // Same for decode
        let encoded_values = [0.01, 0.04, 0.1, 0.5, 1.0];
        for &u in &encoded_values {
            let pos = srgb_to_linear_single(u);
            let neg = srgb_to_linear_single(-u);
            assert!(
                (neg + pos).abs() < 1e-6,
                "Decode not point-symmetric: f({}) = {}, f({}) = {}, sum = {}",
                u, pos, -u, neg, neg + pos
            );
        }
    }

    #[test]
    fn test_srgb_negative_roundtrip() {
        // Negative values should round-trip just like positive values
        let test_values = [-2.0, -1.0, -0.5, -0.1, -0.01, -0.001, -0.0001];

        for &linear in &test_values {
            let encoded = linear_to_srgb_single(linear);
            let back = srgb_to_linear_single(encoded);
            assert!(
                (linear - back).abs() < 1e-5,
                "Negative roundtrip failed: {} -> {} -> {} (error {})",
                linear, encoded, back, (linear - back).abs()
            );
        }

        // And encoded negative values
        let encoded_values = [-1.5, -1.0, -0.5, -0.1, -0.04, -0.01];
        for &encoded in &encoded_values {
            let linear = srgb_to_linear_single(encoded);
            let back = linear_to_srgb_single(linear);
            assert!(
                (encoded - back).abs() < 1e-5,
                "Negative encoded roundtrip failed: {} -> {} -> {} (error {})",
                encoded, linear, back, (encoded - back).abs()
            );
        }
    }

    #[test]
    fn test_srgb_negative_threshold_continuity() {
        // The negative threshold should also have continuous value and slope.
        // Test continuity at -SRGB_THRESHOLD (linear space)
        let t = -cs::SRGB_THRESHOLD;
        let epsilon = 1e-6_f32;

        let f_below = linear_to_srgb_single(t - epsilon);
        let f_at = linear_to_srgb_single(t);
        let f_above = linear_to_srgb_single(t + epsilon);

        assert!(
            (f_below - f_at).abs() < 1e-4,
            "Negative threshold discontinuity below: f({}) = {}, f({}) = {}",
            t - epsilon, f_below, t, f_at
        );
        assert!(
            (f_above - f_at).abs() < 1e-4,
            "Negative threshold discontinuity above: f({}) = {}, f({}) = {}",
            t, f_at, t + epsilon, f_above
        );

        // Monotonicity (should be increasing even for negatives)
        assert!(f_below < f_at, "Not monotonic below negative threshold");
        assert!(f_at < f_above, "Not monotonic above negative threshold");
    }

    #[test]
    fn test_srgb_negative_decode_threshold_continuity() {
        // Test continuity at -SRGB_DECODE_THRESHOLD (encoded space)
        let t = -cs::SRGB_DECODE_THRESHOLD;
        let epsilon = 1e-6_f32;

        let f_below = srgb_to_linear_single(t - epsilon);
        let f_at = srgb_to_linear_single(t);
        let f_above = srgb_to_linear_single(t + epsilon);

        assert!(
            (f_below - f_at).abs() < 1e-4,
            "Negative decode threshold discontinuity below: f({}) = {}, f({}) = {}",
            t - epsilon, f_below, t, f_at
        );
        assert!(
            (f_above - f_at).abs() < 1e-4,
            "Negative decode threshold discontinuity above: f({}) = {}, f({}) = {}",
            t, f_at, t + epsilon, f_above
        );

        // Monotonicity
        assert!(f_below < f_at, "Not monotonic below negative decode threshold");
        assert!(f_at < f_above, "Not monotonic above negative decode threshold");
    }

    #[test]
    fn test_srgb_extended_range_full_roundtrip() {
        // Test round-trip across the full extended range [-2, 2] with fine granularity
        // This covers well beyond the standard sRGB gamut
        let mut max_error = 0.0_f32;
        let mut worst_value = 0.0_f32;

        for i in -2000..=2000 {
            let linear = i as f32 / 1000.0;
            let encoded = linear_to_srgb_single(linear);
            let back = srgb_to_linear_single(encoded);
            let error = (linear - back).abs();
            if error > max_error {
                max_error = error;
                worst_value = linear;
            }
        }

        assert!(
            max_error < 1e-5,
            "Extended range roundtrip max error too large: {} at value {}",
            max_error, worst_value
        );
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
