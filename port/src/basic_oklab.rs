/// Basic Oklab histogram matching algorithm.
/// Analogous to basic_lab.rs but using the Oklab color space.

use crate::color::{linear_rgb_to_oklab_pixel, oklab_to_linear_rgb_pixel};
use crate::dither::{dither_with_mode, DitherMode};
use crate::histogram::{match_histogram, match_histogram_f32, AlignmentMode, InterpolationMode};
use crate::pixel::Pixel4;

// Oklab ranges:
// L: 0-1
// a/b: roughly -0.4 to +0.4, but can extend to ~-0.5 to +0.5 for saturated colors
// We use a safe range of -0.5 to +0.5 for a/b scaling

const OKLAB_AB_MIN: f32 = -0.5;
const OKLAB_AB_MAX: f32 = 0.5;
const OKLAB_AB_RANGE: f32 = OKLAB_AB_MAX - OKLAB_AB_MIN; // 1.0

/// Scale L channel: 0-1 -> 0-255
fn scale_l_to_255(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v * 255.0).collect()
}

/// Scale AB channels: -0.5..0.5 -> 0-255
fn scale_ab_to_255(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let a_scaled: Vec<f32> = a
        .iter()
        .map(|&v| (v - OKLAB_AB_MIN) / OKLAB_AB_RANGE * 255.0)
        .collect();
    let b_scaled: Vec<f32> = b
        .iter()
        .map(|&v| (v - OKLAB_AB_MIN) / OKLAB_AB_RANGE * 255.0)
        .collect();
    (a_scaled, b_scaled)
}

/// Reverse L scaling: 0-255 -> 0-1
fn scale_255_to_l(l: &[f32]) -> Vec<f32> {
    l.iter().map(|&v| v / 255.0).collect()
}

/// Reverse L scaling: uint8 -> 0-1
fn scale_uint8_to_l(l: &[u8]) -> Vec<f32> {
    l.iter().map(|&v| v as f32 / 255.0).collect()
}

/// Reverse AB scaling: 0-255 -> -0.5..0.5
fn scale_255_to_ab(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let a_oklab: Vec<f32> = a
        .iter()
        .map(|&v| v / 255.0 * OKLAB_AB_RANGE + OKLAB_AB_MIN)
        .collect();
    let b_oklab: Vec<f32> = b
        .iter()
        .map(|&v| v / 255.0 * OKLAB_AB_RANGE + OKLAB_AB_MIN)
        .collect();
    (a_oklab, b_oklab)
}

/// Reverse AB scaling: uint8 -> -0.5..0.5
fn scale_uint8_to_ab(a: &[u8], b: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let a_oklab: Vec<f32> = a
        .iter()
        .map(|&v| v as f32 / 255.0 * OKLAB_AB_RANGE + OKLAB_AB_MIN)
        .collect();
    let b_oklab: Vec<f32> = b
        .iter()
        .map(|&v| v as f32 / 255.0 * OKLAB_AB_RANGE + OKLAB_AB_MIN)
        .collect();
    (a_oklab, b_oklab)
}

/// Basic Oklab histogram matching - returns linear RGB as Pixel4 array
///
/// This is the core algorithm that performs histogram matching in Oklab space
/// and returns the result as linear RGB Pixel4 array (0-1 range).
///
/// Args:
///     input: Input image as linear RGB Pixel4 array (0-1 range)
///     reference: Reference image as linear RGB Pixel4 array (0-1 range)
///     input_width, input_height: Input image dimensions
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram quantization
///
/// Returns: Linear RGB Pixel4 array
pub fn color_correct_basic_oklab_linear(
    input: &[Pixel4],
    reference: &[Pixel4],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: DitherMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    let pixel_count = input.len();
    let ref_count = reference.len();

    // Convert input and reference to Oklab, extracting channels
    let mut in_l = Vec::with_capacity(pixel_count);
    let mut in_a = Vec::with_capacity(pixel_count);
    let mut in_b_ch = Vec::with_capacity(pixel_count);

    for &p in input {
        let oklab = linear_rgb_to_oklab_pixel(p);
        in_l.push(oklab[0]);
        in_a.push(oklab[1]);
        in_b_ch.push(oklab[2]);
    }

    let mut ref_l = Vec::with_capacity(ref_count);
    let mut ref_a = Vec::with_capacity(ref_count);
    let mut ref_b_ch = Vec::with_capacity(ref_count);

    for &p in reference {
        let oklab = linear_rgb_to_oklab_pixel(p);
        ref_l.push(oklab[0]);
        ref_a.push(oklab[1]);
        ref_b_ch.push(oklab[2]);
    }

    // Store original L if preserving luminosity
    let original_l = if keep_luminosity {
        in_l.clone()
    } else {
        Vec::new()
    };

    // Scale to 0-255 range
    let in_l_scaled = scale_l_to_255(&in_l);
    let (in_a_scaled, in_b_scaled) = scale_ab_to_255(&in_a, &in_b_ch);
    let ref_l_scaled = scale_l_to_255(&ref_l);
    let (ref_a_scaled, ref_b_scaled) = scale_ab_to_255(&ref_a, &ref_b_ch);

    if let Some(ref mut cb) = progress {
        cb(0.1);
    }

    // Match histograms
    // histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
    let (final_l, final_a, final_b) = if histogram_mode > 0 {
        // Use f32 histogram matching directly (no dithering/quantization)
        let align_mode = if histogram_mode == 2 {
            AlignmentMode::Midpoint
        } else {
            AlignmentMode::Endpoint
        };
        if keep_luminosity {
            let matched_a =
                match_histogram_f32(&in_a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, 0);
            let matched_b =
                match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 1);
            let (a_oklab, b_oklab) = scale_255_to_ab(&matched_a, &matched_b);
            (original_l, a_oklab, b_oklab)
        } else {
            let matched_l =
                match_histogram_f32(&in_l_scaled, &ref_l_scaled, InterpolationMode::Linear, align_mode, 0);
            let matched_a =
                match_histogram_f32(&in_a_scaled, &ref_a_scaled, InterpolationMode::Linear, align_mode, 1);
            let matched_b =
                match_histogram_f32(&in_b_scaled, &ref_b_scaled, InterpolationMode::Linear, align_mode, 2);
            let l_oklab = scale_255_to_l(&matched_l);
            let (a_oklab, b_oklab) = scale_255_to_ab(&matched_a, &matched_b);
            (l_oklab, a_oklab, b_oklab)
        }
    } else {
        // Use binned histogram matching with dithering
        // Each dither call gets a unique seed for deterministic but varied randomization
        let in_l_u8 = dither_with_mode(&in_l_scaled, input_width, input_height, histogram_dither_mode, 0);
        let in_a_u8 = dither_with_mode(&in_a_scaled, input_width, input_height, histogram_dither_mode, 1);
        let in_b_u8 = dither_with_mode(&in_b_scaled, input_width, input_height, histogram_dither_mode, 2);
        let ref_l_u8 = dither_with_mode(&ref_l_scaled, ref_width, ref_height, histogram_dither_mode, 3);
        let ref_a_u8 = dither_with_mode(&ref_a_scaled, ref_width, ref_height, histogram_dither_mode, 4);
        let ref_b_u8 = dither_with_mode(&ref_b_scaled, ref_width, ref_height, histogram_dither_mode, 5);

        if keep_luminosity {
            let matched_a = match_histogram(&in_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&in_b_u8, &ref_b_u8);
            let (a_oklab, b_oklab) = scale_uint8_to_ab(&matched_a, &matched_b);
            (original_l, a_oklab, b_oklab)
        } else {
            let matched_l = match_histogram(&in_l_u8, &ref_l_u8);
            let matched_a = match_histogram(&in_a_u8, &ref_a_u8);
            let matched_b = match_histogram(&in_b_u8, &ref_b_u8);
            let l_oklab = scale_uint8_to_l(&matched_l);
            let (a_oklab, b_oklab) = scale_uint8_to_ab(&matched_a, &matched_b);
            (l_oklab, a_oklab, b_oklab)
        }
    };

    if let Some(ref mut cb) = progress {
        cb(0.8);
    }

    // Convert Oklab back to linear RGB as Pixel4 array
    let mut result = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        let oklab_pixel = Pixel4::new(final_l[i], final_a[i], final_b[i], 0.0);
        result.push(oklab_to_linear_rgb_pixel(oklab_pixel));
    }

    if let Some(ref mut cb) = progress {
        cb(1.0);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_roundtrip() {
        let l = vec![0.5];
        let a = vec![0.0];
        let b = vec![0.0];

        let l_scaled = scale_l_to_255(&l);
        let (a_scaled, b_scaled) = scale_ab_to_255(&a, &b);

        // Test roundtrip through f32 scaling
        let l2 = scale_255_to_l(&l_scaled);
        let (a2, b2) = scale_255_to_ab(&a_scaled, &b_scaled);

        assert!((l2[0] - 0.5).abs() < 0.01);
        assert!((a2[0] - 0.0).abs() < 0.01);
        assert!((b2[0] - 0.0).abs() < 0.01);

        // Also test roundtrip through uint8 (via rounding)
        let l_u8 = vec![l_scaled[0].round() as u8];
        let a_u8 = vec![a_scaled[0].round() as u8];
        let b_u8 = vec![b_scaled[0].round() as u8];

        let l3 = scale_uint8_to_l(&l_u8);
        let (a3, b3) = scale_uint8_to_ab(&a_u8, &b_u8);

        assert!((l3[0] - 0.5).abs() < 0.01);
        assert!((a3[0] - 0.0).abs() < 0.01);
        assert!((b3[0] - 0.0).abs() < 0.01);
    }
}
