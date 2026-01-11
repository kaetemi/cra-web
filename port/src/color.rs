/// Color conversion utilities for sRGB, Linear RGB, Lab, and Oklab color spaces.

// RGB to XYZ matrix (sRGB/Rec.709 primaries, D65 illuminant)
const RGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
];

// XYZ to RGB matrix (inverse of above)
const XYZ_TO_RGB: [[f32; 3]; 3] = [
    [3.240479, -1.537150, -0.498535],
    [-0.969256, 1.875991, 0.041556],
    [0.055648, -0.204043, 1.057311],
];

// D65 white point
const X_N: f32 = 0.950456;
#[allow(dead_code)]
const Y_N: f32 = 1.0;
const Z_N: f32 = 1.088754;

// Lab threshold: (6/29)^3
const EPSILON: f32 = 0.008856;

// Lab linear segment slope: used for f(t) linear segment
const KAPPA_INV: f32 = 7.787;

/// Convert sRGB value (0-1) to linear RGB
#[inline]
pub fn srgb_to_linear_single(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear RGB value (0-1) to sRGB
#[inline]
pub fn linear_to_srgb_single(linear: f32) -> f32 {
    if linear <= 0.04045 / 12.92 {
        linear * 12.92
    } else {
        1.055 * linear.max(0.0).powf(1.0 / 2.4) - 0.055
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
    if t > EPSILON {
        t.cbrt()
    } else {
        KAPPA_INV * t + 16.0 / 116.0
    }
}

/// Inverse of lab_f
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    // Threshold in f-space: f(EPSILON) ≈ 0.206893
    if t > 0.206893 {
        t * t * t
    } else {
        (t - 16.0 / 116.0) / KAPPA_INV
    }
}

/// Convert linear RGB (0-1) to Lab
/// Returns (L, a, b) where L is 0-100 and a,b are roughly -127 to +127
#[inline]
pub fn linear_rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // RGB -> XYZ, normalized by white point
    let x = (RGB_TO_XYZ[0][0] * r + RGB_TO_XYZ[0][1] * g + RGB_TO_XYZ[0][2] * b) / X_N;
    let y = RGB_TO_XYZ[1][0] * r + RGB_TO_XYZ[1][1] * g + RGB_TO_XYZ[1][2] * b; // Y_N = 1.0
    let z = (RGB_TO_XYZ[2][0] * r + RGB_TO_XYZ[2][1] * g + RGB_TO_XYZ[2][2] * b) / Z_N;

    // Apply f(t)
    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);

    // XYZ -> Lab
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_ch = 200.0 * (fy - fz);

    (l, a, b_ch)
}

/// Convert Lab to linear RGB (0-1)
#[inline]
pub fn lab_to_linear_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Lab -> f values
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    // Invert f(t) to get XYZ (normalized)
    let x = lab_f_inv(fx);
    let y = lab_f_inv(fy);
    let z = lab_f_inv(fz);

    // Denormalize by white point
    let x = x * X_N;
    // y = y * Y_N where Y_N = 1.0
    let z = z * Z_N;

    // XYZ -> linear RGB
    let r = XYZ_TO_RGB[0][0] * x + XYZ_TO_RGB[0][1] * y + XYZ_TO_RGB[0][2] * z;
    let g = XYZ_TO_RGB[1][0] * x + XYZ_TO_RGB[1][1] * y + XYZ_TO_RGB[1][2] * z;
    let b_out = XYZ_TO_RGB[2][0] * x + XYZ_TO_RGB[2][1] * y + XYZ_TO_RGB[2][2] * z;

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

/// Convert linear RGB (0-1) to Oklab
/// Returns (L, a, b) where L is 0-1 and a,b are roughly -0.4 to +0.4
#[inline]
pub fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Linear sRGB to LMS
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    // Cube root (with sign handling for out-of-gamut values)
    let l_ = l.max(0.0).cbrt();
    let m_ = m.max(0.0).cbrt();
    let s_ = s.max(0.0).cbrt();

    // LMS' to Oklab
    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    (ok_l, ok_a, ok_b)
}

/// Convert Oklab to linear RGB (0-1)
#[inline]
pub fn oklab_to_linear_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Oklab to LMS'
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    // Cube LMS'
    let lms_l = l_ * l_ * l_;
    let lms_m = m_ * m_ * m_;
    let lms_s = s_ * s_ * s_;

    // LMS to linear sRGB
    let r = 4.0767416621 * lms_l - 3.3077115913 * lms_m + 0.2309699292 * lms_s;
    let g = -1.2684380046 * lms_l + 2.6097574011 * lms_m - 0.3413193965 * lms_s;
    let b_out = -0.0041960863 * lms_l - 0.7034186147 * lms_m + 1.7076147010 * lms_s;

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
