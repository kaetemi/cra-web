/// SIMD-friendly pixel types and utilities
///
/// This module provides [f32; 4] aligned pixel types for efficient SIMD processing.
/// The 4th channel is typically unused (padding) for RGB data but enables 128-bit
/// aligned loads/stores on SSE/NEON architectures.

/// 4-channel pixel type for SIMD-friendly RGB processing
/// Layout: [R, G, B, _] where _ is padding (typically 0.0 or 1.0)
pub type Pixel4 = [f32; 4];

/// Create a Pixel4 from RGB values (padding set to 0.0)
#[inline(always)]
pub fn rgb(r: f32, g: f32, b: f32) -> Pixel4 {
    [r, g, b, 0.0]
}

/// Create a Pixel4 from RGB values with alpha/padding
#[inline(always)]
pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Pixel4 {
    [r, g, b, a]
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

    let len = r.len();
    let mut pixels = Vec::with_capacity(len);

    for i in 0..len {
        pixels.push([r[i], g[i], b[i], 0.0]);
    }

    pixels
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

/// Convert interleaved RGB array (r0,g0,b0,r1,g1,b1,...) to packed Pixel4 array
pub fn interleaved_to_pixels(data: &[f32]) -> Vec<Pixel4> {
    debug_assert_eq!(data.len() % 3, 0);

    let pixel_count = data.len() / 3;
    let mut pixels = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        pixels.push([
            data[i * 3],
            data[i * 3 + 1],
            data[i * 3 + 2],
            0.0,
        ]);
    }

    pixels
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

    let pixel_count = data.len() / 4;
    let mut pixels = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        pixels.push([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
    }

    pixels
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

    let pixel_count = data.len() / 3;
    let mut pixels = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        pixels.push([
            data[i * 3] as f32,
            data[i * 3 + 1] as f32,
            data[i * 3 + 2] as f32,
            0.0,
        ]);
    }

    pixels
}

/// Convert sRGB u8 interleaved RGBA to Pixel4 array (0-255 scale)
pub fn srgb_u8_rgba_to_pixels(data: &[u8]) -> Vec<Pixel4> {
    debug_assert_eq!(data.len() % 4, 0);

    let pixel_count = data.len() / 4;
    let mut pixels = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        pixels.push([
            data[i * 4] as f32,
            data[i * 4 + 1] as f32,
            data[i * 4 + 2] as f32,
            data[i * 4 + 3] as f32,
        ]);
    }

    pixels
}

/// Convert Pixel4 array (0-255 scale) to sRGB u8 interleaved
pub fn pixels_to_srgb_u8(pixels: &[Pixel4]) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixels.len() * 3);

    for p in pixels {
        data.push(p[0].round().clamp(0.0, 255.0) as u8);
        data.push(p[1].round().clamp(0.0, 255.0) as u8);
        data.push(p[2].round().clamp(0.0, 255.0) as u8);
    }

    data
}

/// Convert Pixel4 array (0-255 scale) to sRGB u8 interleaved RGBA
pub fn pixels_to_srgb_u8_rgba(pixels: &[Pixel4]) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixels.len() * 4);

    for p in pixels {
        data.push(p[0].round().clamp(0.0, 255.0) as u8);
        data.push(p[1].round().clamp(0.0, 255.0) as u8);
        data.push(p[2].round().clamp(0.0, 255.0) as u8);
        data.push(p[3].round().clamp(0.0, 255.0) as u8);
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
    [f(p[0]), f(p[1]), f(p[2]), p[3]]
}

// ============================================================================
// Arithmetic operations
// ============================================================================

/// Add two pixels component-wise
#[inline(always)]
pub fn add(a: Pixel4, b: Pixel4) -> Pixel4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Subtract two pixels component-wise
#[inline(always)]
pub fn sub(a: Pixel4, b: Pixel4) -> Pixel4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Multiply two pixels component-wise
#[inline(always)]
pub fn mul(a: Pixel4, b: Pixel4) -> Pixel4 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

/// Multiply pixel by scalar
#[inline(always)]
pub fn scale(p: Pixel4, s: f32) -> Pixel4 {
    [p[0] * s, p[1] * s, p[2] * s, p[3] * s]
}

/// Linear interpolation between two pixels
#[inline(always)]
pub fn lerp(a: Pixel4, b: Pixel4, t: f32) -> Pixel4 {
    let inv_t = 1.0 - t;
    [
        a[0] * inv_t + b[0] * t,
        a[1] * inv_t + b[1] * t,
        a[2] * inv_t + b[2] * t,
        a[3] * inv_t + b[3] * t,
    ]
}

/// Clamp pixel values to a range
#[inline(always)]
pub fn clamp(p: Pixel4, min: f32, max: f32) -> Pixel4 {
    [
        p[0].clamp(min, max),
        p[1].clamp(min, max),
        p[2].clamp(min, max),
        p[3].clamp(min, max),
    ]
}

// ============================================================================
// Constants
// ============================================================================

pub const BLACK: Pixel4 = [0.0, 0.0, 0.0, 0.0];
pub const WHITE: Pixel4 = [1.0, 1.0, 1.0, 1.0];
pub const WHITE_255: Pixel4 = [255.0, 255.0, 255.0, 255.0];

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
        assert_eq!(pixels[0], [1.0, 3.0, 5.0, 0.0]);
        assert_eq!(pixels[1], [2.0, 4.0, 6.0, 0.0]);
    }

    #[test]
    fn test_pixels_to_channels() {
        let pixels = vec![[1.0, 3.0, 5.0, 0.0], [2.0, 4.0, 6.0, 0.0]];

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
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0, 1.0];

        let mid = lerp(a, b, 0.5);
        assert_eq!(mid, [0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_map_rgb() {
        let p = [1.0, 2.0, 3.0, 4.0];
        let doubled = map_rgb(p, |x| x * 2.0);
        assert_eq!(doubled, [2.0, 4.0, 6.0, 4.0]); // alpha unchanged
    }
}
