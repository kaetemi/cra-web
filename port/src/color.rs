/// Color conversion utilities for sRGB, Linear RGB, Lab, and Oklab color spaces.

use crate::colorspace_derived::f32 as cs;

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

/// Convert sRGB image to linear RGB (in-place modification of flat array)
pub fn srgb_to_linear(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = srgb_to_linear_single(*v);
    }
}

/// Convert linear RGB image to sRGB (in-place modification of flat array)
#[allow(dead_code)]
pub fn linear_to_srgb(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = linear_to_srgb_single(*v);
    }
}

/// Lab f(t) function - attempt to linearize cube root near zero
#[inline]
fn lab_f(t: f32) -> f32 {
    if t > cs::CIELAB_EPSILON {
        t.cbrt()
    } else {
        cs::CIELAB_KAPPA * t + cs::CIELAB_OFFSET
    }
}

/// Inverse of lab_f
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    if t > cs::CIELAB_F_THRESHOLD {
        t * t * t
    } else {
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

/// Convert linear RGB image (HxWx3 flat array) to Lab
/// Output: L is 0-100, a,b are roughly -127 to +127
#[allow(dead_code)]
pub fn image_rgb_to_lab(rgb: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut lab = vec![0.0f32; width * height * 3];
    for i in 0..(width * height) {
        let idx = i * 3;
        let (l, a, b) = linear_rgb_to_lab(rgb[idx], rgb[idx + 1], rgb[idx + 2]);
        lab[idx] = l;
        lab[idx + 1] = a;
        lab[idx + 2] = b;
    }
    lab
}

/// Convert Lab image (HxWx3 flat array) to linear RGB
#[allow(dead_code)]
pub fn image_lab_to_rgb(lab: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut rgb = vec![0.0f32; width * height * 3];
    for i in 0..(width * height) {
        let idx = i * 3;
        let (r, g, b) = lab_to_linear_rgb(lab[idx], lab[idx + 1], lab[idx + 2]);
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }
    rgb
}

/// Extract a single channel from an interleaved image
#[allow(dead_code)]
pub fn extract_channel(img: &[f32], width: usize, height: usize, channel: usize) -> Vec<f32> {
    let mut ch = vec![0.0f32; width * height];
    for i in 0..(width * height) {
        ch[i] = img[i * 3 + channel];
    }
    ch
}

/// Set a single channel in an interleaved image
#[allow(dead_code)]
pub fn set_channel(img: &mut [f32], width: usize, height: usize, channel: usize, data: &[f32]) {
    for i in 0..(width * height) {
        img[i * 3 + channel] = data[i];
    }
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

// ============== Channel-separated processing ==============

/// Convert interleaved sRGB (0-1) to separate linear RGB channels
pub fn srgb_to_linear_channels(rgb: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = width * height;
    let mut r = vec![0.0f32; pixels];
    let mut g = vec![0.0f32; pixels];
    let mut b = vec![0.0f32; pixels];

    for i in 0..pixels {
        r[i] = srgb_to_linear_single(rgb[i * 3]);
        g[i] = srgb_to_linear_single(rgb[i * 3 + 1]);
        b[i] = srgb_to_linear_single(rgb[i * 3 + 2]);
    }

    (r, g, b)
}

/// Convert separate linear RGB channels to separate LAB channels
pub fn linear_rgb_to_lab_channels(r: &[f32], g: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = r.len();
    let mut l_ch = vec![0.0f32; pixels];
    let mut a_ch = vec![0.0f32; pixels];
    let mut b_ch = vec![0.0f32; pixels];

    for i in 0..pixels {
        let (l, a, b_val) = linear_rgb_to_lab(r[i], g[i], b[i]);
        l_ch[i] = l;
        a_ch[i] = a;
        b_ch[i] = b_val;
    }

    (l_ch, a_ch, b_ch)
}

/// Convert separate LAB channels to separate linear RGB channels
pub fn lab_to_linear_rgb_channels(l: &[f32], a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = l.len();
    let mut r_ch = vec![0.0f32; pixels];
    let mut g_ch = vec![0.0f32; pixels];
    let mut b_ch = vec![0.0f32; pixels];

    for i in 0..pixels {
        let (r, g, b_val) = lab_to_linear_rgb(l[i], a[i], b[i]);
        r_ch[i] = r;
        g_ch[i] = g;
        b_ch[i] = b_val;
    }

    (r_ch, g_ch, b_ch)
}

/// Convert separate linear RGB channels to separate Oklab channels
pub fn linear_rgb_to_oklab_channels(r: &[f32], g: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = r.len();
    let mut l_ch = vec![0.0f32; pixels];
    let mut a_ch = vec![0.0f32; pixels];
    let mut b_ch = vec![0.0f32; pixels];

    for i in 0..pixels {
        let (l, a, b_val) = linear_rgb_to_oklab(r[i], g[i], b[i]);
        l_ch[i] = l;
        a_ch[i] = a;
        b_ch[i] = b_val;
    }

    (l_ch, a_ch, b_ch)
}

/// Convert separate Oklab channels to separate linear RGB channels
pub fn oklab_to_linear_rgb_channels(l: &[f32], a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = l.len();
    let mut r_ch = vec![0.0f32; pixels];
    let mut g_ch = vec![0.0f32; pixels];
    let mut b_ch = vec![0.0f32; pixels];

    for i in 0..pixels {
        let (r, g, b_val) = oklab_to_linear_rgb(l[i], a[i], b[i]);
        r_ch[i] = r;
        g_ch[i] = g;
        b_ch[i] = b_val;
    }

    (r_ch, g_ch, b_ch)
}

/// Convert separate linear RGB channels to sRGB and scale to 0-255
pub fn linear_to_srgb_scaled_channels(r: &[f32], g: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = r.len();
    let mut r_out = vec![0.0f32; pixels];
    let mut g_out = vec![0.0f32; pixels];
    let mut b_out = vec![0.0f32; pixels];

    for i in 0..pixels {
        r_out[i] = (linear_to_srgb_single(r[i]) * 255.0).clamp(0.0, 255.0);
        g_out[i] = (linear_to_srgb_single(g[i]) * 255.0).clamp(0.0, 255.0);
        b_out[i] = (linear_to_srgb_single(b[i]) * 255.0).clamp(0.0, 255.0);
    }

    (r_out, g_out, b_out)
}

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
}
