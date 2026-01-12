/// Perceptual dithering implementation.
///
/// Uses perceptual color space (CIELAB or OKLab) for distance calculations
/// with error diffusion in linear RGB space.

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_to_srgb_single, srgb_to_linear_single,
};

/// Perceptual color space for distance calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerceptualSpace {
    /// CIELAB color space (L*a*b*)
    #[default]
    Lab,
    /// OKLab color space
    OkLab,
}

/// Extend n-bit value to 8 bits by repeating the bit pattern.
/// e.g., 3-bit value ABC becomes ABCABCAB
#[inline]
pub fn bit_replicate(value: u8, bits: u8) -> u8 {
    if bits == 8 {
        return value;
    }
    let mut result: u16 = 0;
    let mut shift = 8i8;
    while shift > 0 {
        shift -= bits as i8;
        if shift >= 0 {
            result |= (value as u16) << shift;
        } else {
            // Partial bits at the end
            result |= (value as u16) >> (-shift);
        }
    }
    result as u8
}

/// Perceptual quantization parameters for joint RGB dithering.
/// Uses perceptual distance (Lab/OkLab) for candidate selection,
/// linear RGB for error diffusion.
struct PerceptualQuantParams {
    /// Number of quantization levels (2^bits)
    num_levels: usize,
    /// Level index → extended sRGB value (0-255)
    level_values: Vec<u8>,
    /// sRGB value → floor level index
    lut_floor_level: [u8; 256],
    /// sRGB value → ceil level index
    lut_ceil_level: [u8; 256],
}

impl PerceptualQuantParams {
    /// Create perceptual quantization parameters for given bit depth.
    fn new(bits: u8) -> Self {
        debug_assert!(bits >= 1 && bits <= 8, "bits must be 1-8");
        let num_levels = 1usize << bits;
        let max_idx = num_levels - 1;
        let shift = 8 - bits;

        // Pre-compute bit-replicated values for each level
        let level_values: Vec<u8> = (0..num_levels)
            .map(|l| bit_replicate(l as u8, bits))
            .collect();

        let mut lut_floor_level = [0u8; 256];
        let mut lut_ceil_level = [0u8; 256];

        for v in 0..256u16 {
            // Use bit truncation to find a nearby level
            let trunc_idx = (v as u8 >> shift) as usize;
            let trunc_val = level_values[trunc_idx];

            let (floor_idx, ceil_idx) = if trunc_val == v as u8 {
                (trunc_idx, trunc_idx)
            } else if trunc_val < v as u8 {
                // trunc is floor, ceil is trunc+1
                let ceil = if trunc_idx < max_idx {
                    trunc_idx + 1
                } else {
                    trunc_idx
                };
                (trunc_idx, ceil)
            } else {
                // trunc is ceil, floor is trunc-1
                let floor = if trunc_idx > 0 { trunc_idx - 1 } else { trunc_idx };
                (floor, trunc_idx)
            };

            lut_floor_level[v as usize] = floor_idx as u8;
            lut_ceil_level[v as usize] = ceil_idx as u8;
        }

        Self {
            num_levels,
            level_values,
            lut_floor_level,
            lut_ceil_level,
        }
    }

    /// Get floor level index for a sRGB value (0-255)
    #[inline]
    fn floor_level(&self, srgb_value: u8) -> usize {
        self.lut_floor_level[srgb_value as usize] as usize
    }

    /// Get ceil level index for a sRGB value (0-255)
    #[inline]
    fn ceil_level(&self, srgb_value: u8) -> usize {
        self.lut_ceil_level[srgb_value as usize] as usize
    }

    /// Get extended sRGB value (0-255) for a level index
    #[inline]
    fn level_to_srgb(&self, level: usize) -> u8 {
        self.level_values[level]
    }
}

/// Pre-computed LUT for sRGB to linear conversion (256 entries)
fn build_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = srgb_to_linear_single(i as f32 / 255.0);
    }
    lut
}

/// Lab color value for LUT storage
#[derive(Clone, Copy, Default)]
struct LabValue {
    l: f32,
    a: f32,
    b: f32,
}

/// Build Lab/OkLab LUT indexed by (r_level, g_level, b_level)
/// Returns a flat Vec where index = r_level * num_levels^2 + g_level * num_levels + b_level
fn build_perceptual_lut(
    quant: &PerceptualQuantParams,
    linear_lut: &[f32; 256],
    space: PerceptualSpace,
) -> Vec<LabValue> {
    let n = quant.num_levels;
    let mut lut = vec![LabValue::default(); n * n * n];

    for r_level in 0..n {
        let r_ext = quant.level_values[r_level];
        let r_lin = linear_lut[r_ext as usize];

        for g_level in 0..n {
            let g_ext = quant.level_values[g_level];
            let g_lin = linear_lut[g_ext as usize];

            for b_level in 0..n {
                let b_ext = quant.level_values[b_level];
                let b_lin = linear_lut[b_ext as usize];

                let (l, a, b_ch) = match space {
                    PerceptualSpace::Lab => linear_rgb_to_lab(r_lin, g_lin, b_lin),
                    PerceptualSpace::OkLab => linear_rgb_to_oklab(r_lin, g_lin, b_lin),
                };

                let idx = r_level * n * n + g_level * n + b_level;
                lut[idx] = LabValue { l, a, b: b_ch };
            }
        }
    }

    lut
}

/// Perceptual Floyd-Steinberg dithering.
///
/// Uses perceptual color space (Lab or OkLab) for finding the best quantization
/// candidate, but accumulates and diffuses error in linear RGB space.
///
/// Args:
///     r_channel, g_channel, b_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Bit depth for each channel (1-8)
///     space: Perceptual color space for distance calculation
///
/// Returns:
///     (r_out, g_out, b_out): Output channels as u8
pub fn perceptual_dither_rgb(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    space: PerceptualSpace,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let quant_r = PerceptualQuantParams::new(bits_r);
    let quant_g = PerceptualQuantParams::new(bits_g);
    let quant_b = PerceptualQuantParams::new(bits_b);

    let linear_lut = build_linear_lut();

    // Build combined perceptual LUT
    // Since we may have different bit depths, we need to handle this carefully
    // For simplicity, build a LUT for each possible combination within the search bounds
    // We'll compute perceptual values on-the-fly for mixed bit depths
    let same_bits = bits_r == bits_g && bits_g == bits_b;

    // If all channels have same bit depth, use a pre-built LUT
    let lab_lut = if same_bits {
        Some(build_perceptual_lut(&quant_r, &linear_lut, space))
    } else {
        None
    };

    let pixels = width * height;
    let pad_left = 1usize;
    let pad_right = 1usize;
    let pad_bottom = 1usize;
    let buf_width = width + pad_left + pad_right;

    // Error buffers in linear RGB space
    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; height + pad_bottom];

    // Output buffers
    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let bx = x + pad_left;

            // 1. Read input, convert to Linear RGB
            let srgb_r = r_channel[idx] / 255.0;
            let srgb_g = g_channel[idx] / 255.0;
            let srgb_b = b_channel[idx] / 255.0;

            let lin_r_orig = srgb_to_linear_single(srgb_r);
            let lin_g_orig = srgb_to_linear_single(srgb_g);
            let lin_b_orig = srgb_to_linear_single(srgb_b);

            // 2. Add accumulated error
            let lin_r_adj = lin_r_orig + err_r[y][bx];
            let lin_g_adj = lin_g_orig + err_g[y][bx];
            let lin_b_adj = lin_b_orig + err_b[y][bx];

            // 3. Convert back to sRGB for quantization bounds (clamp for valid LUT indices)
            let lin_r_clamped = lin_r_adj.clamp(0.0, 1.0);
            let lin_g_clamped = lin_g_adj.clamp(0.0, 1.0);
            let lin_b_clamped = lin_b_adj.clamp(0.0, 1.0);

            let srgb_r_adj = (linear_to_srgb_single(lin_r_clamped) * 255.0).clamp(0.0, 255.0);
            let srgb_g_adj = (linear_to_srgb_single(lin_g_clamped) * 255.0).clamp(0.0, 255.0);
            let srgb_b_adj = (linear_to_srgb_single(lin_b_clamped) * 255.0).clamp(0.0, 255.0);

            // 4. Get level index bounds
            let r_min = quant_r.floor_level(srgb_r_adj.floor() as u8);
            let r_max = quant_r.ceil_level((srgb_r_adj.ceil() as u8).min(255));

            let g_min = quant_g.floor_level(srgb_g_adj.floor() as u8);
            let g_max = quant_g.ceil_level((srgb_g_adj.ceil() as u8).min(255));

            let b_min = quant_b.floor_level(srgb_b_adj.floor() as u8);
            let b_max = quant_b.ceil_level((srgb_b_adj.ceil() as u8).min(255));

            // 5. Convert target to Lab (use unclamped for true distance)
            let lab_target = match space {
                PerceptualSpace::Lab => linear_rgb_to_lab(lin_r_adj, lin_g_adj, lin_b_adj),
                PerceptualSpace::OkLab => linear_rgb_to_oklab(lin_r_adj, lin_g_adj, lin_b_adj),
            };

            // 6. Search candidates
            let mut best_r_level = r_min;
            let mut best_g_level = g_min;
            let mut best_b_level = b_min;
            let mut best_dist = f32::INFINITY;

            for r_level in r_min..=r_max {
                for g_level in g_min..=g_max {
                    for b_level in b_min..=b_max {
                        let lab_candidate = if let Some(ref lut) = lab_lut {
                            // Same bit depths: use LUT
                            let n = quant_r.num_levels;
                            let lut_idx = r_level * n * n + g_level * n + b_level;
                            lut[lut_idx]
                        } else {
                            // Different bit depths: compute on-the-fly
                            let r_ext = quant_r.level_to_srgb(r_level);
                            let g_ext = quant_g.level_to_srgb(g_level);
                            let b_ext = quant_b.level_to_srgb(b_level);

                            let r_lin = linear_lut[r_ext as usize];
                            let g_lin = linear_lut[g_ext as usize];
                            let b_lin = linear_lut[b_ext as usize];

                            let (l, a, b_ch) = match space {
                                PerceptualSpace::Lab => linear_rgb_to_lab(r_lin, g_lin, b_lin),
                                PerceptualSpace::OkLab => linear_rgb_to_oklab(r_lin, g_lin, b_lin),
                            };
                            LabValue { l, a, b: b_ch }
                        };

                        let dl = lab_target.0 - lab_candidate.l;
                        let da = lab_target.1 - lab_candidate.a;
                        let db = lab_target.2 - lab_candidate.b;
                        let dist = dl * dl + da * da + db * db;

                        if dist < best_dist {
                            best_dist = dist;
                            best_r_level = r_level;
                            best_g_level = g_level;
                            best_b_level = b_level;
                        }
                    }
                }
            }

            // 7. Get extended values for output and error calculation
            let best_r = quant_r.level_to_srgb(best_r_level);
            let best_g = quant_g.level_to_srgb(best_g_level);
            let best_b = quant_b.level_to_srgb(best_b_level);

            // 8. Write output
            r_out[idx] = best_r;
            g_out[idx] = best_g;
            b_out[idx] = best_b;

            // 9. Compute error in Linear RGB
            let best_lin_r = linear_lut[best_r as usize];
            let best_lin_g = linear_lut[best_g as usize];
            let best_lin_b = linear_lut[best_b as usize];

            let err_r_val = lin_r_adj - best_lin_r;
            let err_g_val = lin_g_adj - best_lin_g;
            let err_b_val = lin_b_adj - best_lin_b;

            // 10. Diffuse error (Floyd-Steinberg kernel)
            // Right: 7/16
            err_r[y][bx + 1] += err_r_val * (7.0 / 16.0);
            err_g[y][bx + 1] += err_g_val * (7.0 / 16.0);
            err_b[y][bx + 1] += err_b_val * (7.0 / 16.0);

            // Bottom-left: 3/16
            err_r[y + 1][bx - 1] += err_r_val * (3.0 / 16.0);
            err_g[y + 1][bx - 1] += err_g_val * (3.0 / 16.0);
            err_b[y + 1][bx - 1] += err_b_val * (3.0 / 16.0);

            // Bottom: 5/16
            err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
            err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
            err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);

            // Bottom-right: 1/16
            err_r[y + 1][bx + 1] += err_r_val * (1.0 / 16.0);
            err_g[y + 1][bx + 1] += err_g_val * (1.0 / 16.0);
            err_b[y + 1][bx + 1] += err_b_val * (1.0 / 16.0);
        }
    }

    (r_out, g_out, b_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_dither_basic() {
        // Test that perceptual dithering produces valid output
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_out, g_out, b_out) = perceptual_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::Lab);

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
    }

    #[test]
    fn test_perceptual_dither_produces_valid_levels() {
        // With 2-bit depth, output should only contain 0, 85, 170, 255
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_out, g_out, b_out) = perceptual_dither_rgb(&r, &g, &b, 10, 10, 2, 2, 2, PerceptualSpace::Lab);

        let valid_levels = [0u8, 85, 170, 255];
        for &v in &r_out {
            assert!(valid_levels.contains(&v), "R channel produced invalid 2-bit value: {}", v);
        }
        for &v in &g_out {
            assert!(valid_levels.contains(&v), "G channel produced invalid 2-bit value: {}", v);
        }
        for &v in &b_out {
            assert!(valid_levels.contains(&v), "B channel produced invalid 2-bit value: {}", v);
        }
    }

    #[test]
    fn test_perceptual_dither_lab_vs_oklab() {
        // Lab and OkLab should produce different results
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| ((i + 33) % 100) as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| ((i + 66) % 100) as f32 * 2.55).collect();

        let (r_lab, g_lab, b_lab) = perceptual_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::Lab);
        let (r_oklab, g_oklab, b_oklab) = perceptual_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::OkLab);

        // Results should differ (different perceptual spaces have different gamut mappings)
        let lab_combined: Vec<u8> = r_lab.iter().chain(g_lab.iter()).chain(b_lab.iter()).copied().collect();
        let oklab_combined: Vec<u8> = r_oklab.iter().chain(g_oklab.iter()).chain(b_oklab.iter()).copied().collect();
        assert_ne!(lab_combined, oklab_combined);
    }

    #[test]
    fn test_perceptual_dither_neutral_gray() {
        // Neutral gray should dither to nearby levels without color shift
        let gray_val = 128.0f32;
        let r: Vec<f32> = vec![gray_val; 100];
        let g: Vec<f32> = vec![gray_val; 100];
        let b: Vec<f32> = vec![gray_val; 100];

        let (r_out, g_out, b_out) = perceptual_dither_rgb(&r, &g, &b, 10, 10, 5, 5, 5, PerceptualSpace::Lab);

        // For neutral gray input, output should remain relatively neutral
        // (R, G, B should be similar for each pixel)
        for i in 0..100 {
            let r_v = r_out[i] as i32;
            let g_v = g_out[i] as i32;
            let b_v = b_out[i] as i32;
            // Allow some dithering variation but channels should be close
            assert!((r_v - g_v).abs() <= 36, "Neutral gray has color shift: R={} G={}", r_v, g_v);
            assert!((g_v - b_v).abs() <= 36, "Neutral gray has color shift: G={} B={}", g_v, b_v);
        }
    }

    #[test]
    fn test_perceptual_dither_different_bit_depths() {
        // Test with different bit depths per channel (like RGB565)
        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();

        // 5-bit R and B, 6-bit G (RGB565 format)
        let (r_out, g_out, b_out) = perceptual_dither_rgb(&r, &g, &b, 10, 10, 5, 6, 5, PerceptualSpace::Lab);

        // Check that outputs are valid for their respective bit depths
        let valid_5bit: Vec<u8> = (0..32).map(|l| bit_replicate(l, 5)).collect();
        let valid_6bit: Vec<u8> = (0..64).map(|l| bit_replicate(l, 6)).collect();

        for &v in &r_out {
            assert!(valid_5bit.contains(&v), "R channel produced invalid 5-bit value: {}", v);
        }
        for &v in &g_out {
            assert!(valid_6bit.contains(&v), "G channel produced invalid 6-bit value: {}", v);
        }
        for &v in &b_out {
            assert!(valid_5bit.contains(&v), "B channel produced invalid 5-bit value: {}", v);
        }
    }
}
