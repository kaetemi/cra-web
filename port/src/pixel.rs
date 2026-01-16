/// SIMD-friendly pixel types and utilities
///
/// This module provides 16-byte aligned pixel types for efficient SIMD processing.
/// The 4th channel is typically unused (padding) for RGB data but enables 128-bit
/// aligned loads/stores on SSE/NEON architectures.

use std::ops::{Index, IndexMut, Add, Sub, Mul, Div};

/// 4-channel pixel type for SIMD-friendly RGB processing
/// Layout: [R, G, B, _] where _ is padding (typically 0.0 or 1.0)
/// 16-byte aligned for SIMD (SSE/NEON 128-bit registers)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pixel4(pub [f32; 4]);

impl Pixel4 {
    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }

    #[inline(always)]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self([r, g, b, 0.0])
    }

    #[inline(always)]
    pub fn r(&self) -> f32 { self.0[0] }
    #[inline(always)]
    pub fn g(&self) -> f32 { self.0[1] }
    #[inline(always)]
    pub fn b(&self) -> f32 { self.0[2] }
    #[inline(always)]
    pub fn a(&self) -> f32 { self.0[3] }

    #[inline(always)]
    pub fn set_r(&mut self, v: f32) { self.0[0] = v; }
    #[inline(always)]
    pub fn set_g(&mut self, v: f32) { self.0[1] = v; }
    #[inline(always)]
    pub fn set_b(&mut self, v: f32) { self.0[2] = v; }
    #[inline(always)]
    pub fn set_a(&mut self, v: f32) { self.0[3] = v; }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] { self.0 }

    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] { &self.0 }

    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] { &mut self.0 }
}

impl Index<usize> for Pixel4 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &f32 { &self.0[i] }
}

impl IndexMut<usize> for Pixel4 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 { &mut self.0[i] }
}

impl From<[f32; 4]> for Pixel4 {
    #[inline(always)]
    fn from(arr: [f32; 4]) -> Self { Self(arr) }
}

impl From<Pixel4> for [f32; 4] {
    #[inline(always)]
    fn from(p: Pixel4) -> [f32; 4] { p.0 }
}

impl Add for Pixel4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self([self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2], self[3] + rhs[3]])
    }
}

impl Sub for Pixel4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([self[0] - rhs[0], self[1] - rhs[1], self[2] - rhs[2], self[3] - rhs[3]])
    }
}

impl Mul for Pixel4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self([self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2], self[3] * rhs[3]])
    }
}

impl Mul<f32> for Pixel4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self([self[0] * s, self[1] * s, self[2] * s, self[3] * s])
    }
}

impl Div<f32> for Pixel4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        Self([self[0] / s, self[1] / s, self[2] / s, self[3] / s])
    }
}

/// Create a Pixel4 from RGB values (padding set to 0.0)
#[inline(always)]
pub fn rgb(r: f32, g: f32, b: f32) -> Pixel4 {
    Pixel4::rgb(r, g, b)
}

/// Create a Pixel4 from RGB values with alpha/padding
#[inline(always)]
pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Pixel4 {
    Pixel4::new(r, g, b, a)
}

/// Extract RGB components from a Pixel4
#[inline(always)]
pub fn to_rgb(p: Pixel4) -> (f32, f32, f32) {
    (p[0], p[1], p[2])
}

// ============================================================================
// Conversion between formats
// ============================================================================

/// Convert separate R, G, B channel arrays to packed Pixel4 array
pub fn channels_to_pixels(r: &[f32], g: &[f32], b: &[f32]) -> Vec<Pixel4> {
    debug_assert_eq!(r.len(), g.len());
    debug_assert_eq!(g.len(), b.len());

    (0..r.len())
        .map(|i| Pixel4::rgb(r[i], g[i], b[i]))
        .collect()
}

/// Convert packed Pixel4 array to separate R, G, B channel arrays
pub fn pixels_to_channels(pixels: &[Pixel4]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = pixels.len();
    let mut r = Vec::with_capacity(len);
    let mut g = Vec::with_capacity(len);
    let mut b = Vec::with_capacity(len);

    for p in pixels {
        r.push(p[0]);
        g.push(p[1]);
        b.push(p[2]);
    }

    (r, g, b)
}

/// Convert packed Pixel4 array to separate R, G, B, A channel arrays
/// Note: Alpha is stored in 0-1 range, not scaled to 0-255
pub fn pixels_to_channels_rgba(pixels: &[Pixel4]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = pixels.len();
    let mut r = Vec::with_capacity(len);
    let mut g = Vec::with_capacity(len);
    let mut b = Vec::with_capacity(len);
    let mut a = Vec::with_capacity(len);

    for p in pixels {
        r.push(p[0]);
        g.push(p[1]);
        b.push(p[2]);
        a.push(p[3]);
    }

    (r, g, b, a)
}

/// Convert interleaved RGB array (r0,g0,b0,r1,g1,b1,...) to packed Pixel4 array
pub fn interleaved_to_pixels(data: &[f32]) -> Vec<Pixel4> {
    debug_assert_eq!(data.len() % 3, 0);

    (0..data.len() / 3)
        .map(|i| Pixel4::rgb(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]))
        .collect()
}

/// Convert packed Pixel4 array to interleaved RGB array
pub fn pixels_to_interleaved(pixels: &[Pixel4]) -> Vec<f32> {
    let mut data = Vec::with_capacity(pixels.len() * 3);

    for p in pixels {
        data.push(p[0]);
        data.push(p[1]);
        data.push(p[2]);
    }

    data
}

/// Convert interleaved RGBA array to packed Pixel4 array
pub fn interleaved_rgba_to_pixels(data: &[f32]) -> Vec<Pixel4> {
    debug_assert_eq!(data.len() % 4, 0);

    (0..data.len() / 4)
        .map(|i| Pixel4::new(data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]))
        .collect()
}

/// Convert packed Pixel4 array to interleaved RGBA array
pub fn pixels_to_interleaved_rgba(pixels: &[Pixel4]) -> Vec<f32> {
    let mut data = Vec::with_capacity(pixels.len() * 4);

    for p in pixels {
        data.push(p[0]);
        data.push(p[1]);
        data.push(p[2]);
        data.push(p[3]);
    }

    data
}

/// Convert sRGB u8 interleaved (r,g,b,r,g,b,...) to Pixel4 array (0-255 scale)
pub fn srgb_u8_to_pixels(data: &[u8]) -> Vec<Pixel4> {
    debug_assert_eq!(data.len() % 3, 0);

    (0..data.len() / 3)
        .map(|i| Pixel4::rgb(data[i * 3] as f32, data[i * 3 + 1] as f32, data[i * 3 + 2] as f32))
        .collect()
}

/// Convert sRGB u8 interleaved RGBA to Pixel4 array (0-255 scale)
pub fn srgb_u8_rgba_to_pixels(data: &[u8]) -> Vec<Pixel4> {
    debug_assert_eq!(data.len() % 4, 0);

    (0..data.len() / 4)
        .map(|i| Pixel4::new(
            data[i * 4] as f32,
            data[i * 4 + 1] as f32,
            data[i * 4 + 2] as f32,
            data[i * 4 + 3] as f32,
        ))
        .collect()
}

/// Convert Pixel4 array (0-255 scale) to sRGB u8 interleaved
/// Clamps values to 0-255 range.
pub fn pixels_to_srgb_u8_clamped(pixels: &[Pixel4]) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixels.len() * 3);

    for p in pixels {
        data.push(p[0].round().clamp(0.0, 255.0) as u8);
        data.push(p[1].round().clamp(0.0, 255.0) as u8);
        data.push(p[2].round().clamp(0.0, 255.0) as u8);
    }

    data
}

/// Convert Pixel4 array (0-255 scale) to sRGB u8 interleaved RGBA
/// Clamps values to 0-255 range.
pub fn pixels_to_srgb_u8_rgba_clamped(pixels: &[Pixel4]) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixels.len() * 4);

    for p in pixels {
        data.push(p[0].round().clamp(0.0, 255.0) as u8);
        data.push(p[1].round().clamp(0.0, 255.0) as u8);
        data.push(p[2].round().clamp(0.0, 255.0) as u8);
        data.push(p[3].round().clamp(0.0, 255.0) as u8);
    }

    data
}

/// Convert Pixel4 array (RGB 0-255, alpha 0-1) to u8 interleaved with variable channel count.
/// - channels=3: outputs RGB only
/// - channels=4: outputs RGBA with alpha scaled from 0-1 to 0-255
/// Clamps all values to 0-255 range.
pub fn pixels_to_u8_clamped(pixels: &[Pixel4], channels: usize) -> Vec<u8> {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    let mut data = Vec::with_capacity(pixels.len() * channels);

    for p in pixels {
        data.push(p[0].round().clamp(0.0, 255.0) as u8);
        data.push(p[1].round().clamp(0.0, 255.0) as u8);
        data.push(p[2].round().clamp(0.0, 255.0) as u8);
        if channels == 4 {
            // Alpha is stored in 0-1 range, scale to 0-255 for output
            data.push((p[3] * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }

    data
}

/// Extract a single channel from Pixel4 array to u8.
/// channel: 0=R, 1=G, 2=B, 3=A (alpha scaled from 0-1 to 0-255)
pub fn pixels_to_u8_single_channel(pixels: &[Pixel4], channel: usize) -> Vec<u8> {
    assert!(channel < 4, "channel must be 0-3");
    let mut data = Vec::with_capacity(pixels.len());

    for p in pixels {
        let value = if channel == 3 {
            // Alpha is stored in 0-1 range, scale to 0-255
            (p[3] * 255.0).round().clamp(0.0, 255.0)
        } else {
            // RGB channels are already 0-255
            p[channel].round().clamp(0.0, 255.0)
        };
        data.push(value as u8);
    }

    data
}

// ============================================================================
// In-place operations
// ============================================================================

/// Apply a function to each pixel in place
#[inline]
pub fn map_pixels_inplace<F>(pixels: &mut [Pixel4], f: F)
where
    F: Fn(Pixel4) -> Pixel4,
{
    for p in pixels.iter_mut() {
        *p = f(*p);
    }
}

/// Apply a function to each pixel, producing new array
#[inline]
pub fn map_pixels<F>(pixels: &[Pixel4], f: F) -> Vec<Pixel4>
where
    F: Fn(Pixel4) -> Pixel4,
{
    pixels.iter().map(|&p| f(p)).collect()
}

/// Apply a function to each channel of a pixel
#[inline(always)]
pub fn map_rgb<F>(p: Pixel4, f: F) -> Pixel4
where
    F: Fn(f32) -> f32,
{
    Pixel4::new(f(p[0]), f(p[1]), f(p[2]), p[3])
}

// ============================================================================
// Arithmetic operations (free functions for backwards compatibility)
// ============================================================================

/// Add two pixels component-wise
#[inline(always)]
pub fn add(a: Pixel4, b: Pixel4) -> Pixel4 {
    a + b
}

/// Subtract two pixels component-wise
#[inline(always)]
pub fn sub(a: Pixel4, b: Pixel4) -> Pixel4 {
    a - b
}

/// Multiply two pixels component-wise
#[inline(always)]
pub fn mul(a: Pixel4, b: Pixel4) -> Pixel4 {
    a * b
}

/// Multiply pixel by scalar
#[inline(always)]
pub fn scale(p: Pixel4, s: f32) -> Pixel4 {
    p * s
}

/// Linear interpolation between two pixels
#[inline(always)]
pub fn lerp(a: Pixel4, b: Pixel4, t: f32) -> Pixel4 {
    a * (1.0 - t) + b * t
}

/// Clamp pixel values to a range
#[inline(always)]
pub fn clamp(p: Pixel4, min: f32, max: f32) -> Pixel4 {
    Pixel4::new(
        p[0].clamp(min, max),
        p[1].clamp(min, max),
        p[2].clamp(min, max),
        p[3].clamp(min, max),
    )
}

// ============================================================================
// Constants
// ============================================================================

pub const BLACK: Pixel4 = Pixel4([0.0, 0.0, 0.0, 0.0]);
pub const WHITE: Pixel4 = Pixel4([1.0, 1.0, 1.0, 1.0]);
pub const WHITE_255: Pixel4 = Pixel4([255.0, 255.0, 255.0, 255.0]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channels_to_pixels() {
        let r = vec![1.0, 2.0];
        let g = vec![3.0, 4.0];
        let b = vec![5.0, 6.0];

        let pixels = channels_to_pixels(&r, &g, &b);

        assert_eq!(pixels.len(), 2);
        assert_eq!(pixels[0], Pixel4::rgb(1.0, 3.0, 5.0));
        assert_eq!(pixels[1], Pixel4::rgb(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_pixels_to_channels() {
        let pixels = vec![Pixel4::rgb(1.0, 3.0, 5.0), Pixel4::rgb(2.0, 4.0, 6.0)];

        let (r, g, b) = pixels_to_channels(&pixels);

        assert_eq!(r, vec![1.0, 2.0]);
        assert_eq!(g, vec![3.0, 4.0]);
        assert_eq!(b, vec![5.0, 6.0]);
    }

    #[test]
    fn test_interleaved_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pixels = interleaved_to_pixels(&original);
        let result = pixels_to_interleaved(&pixels);

        assert_eq!(original, result);
    }

    #[test]
    fn test_lerp() {
        let a = Pixel4::new(0.0, 0.0, 0.0, 0.0);
        let b = Pixel4::new(1.0, 1.0, 1.0, 1.0);

        let mid = lerp(a, b, 0.5);
        assert_eq!(mid, Pixel4::new(0.5, 0.5, 0.5, 0.5));
    }

    #[test]
    fn test_map_rgb() {
        let p = Pixel4::new(1.0, 2.0, 3.0, 4.0);
        let doubled = map_rgb(p, |x| x * 2.0);
        assert_eq!(doubled, Pixel4::new(2.0, 4.0, 6.0, 4.0)); // alpha unchanged
    }

    #[test]
    fn test_alignment() {
        assert_eq!(std::mem::align_of::<Pixel4>(), 16);
        assert_eq!(std::mem::size_of::<Pixel4>(), 16);
    }
}
