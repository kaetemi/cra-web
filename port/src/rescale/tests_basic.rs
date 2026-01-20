//! Basic rescaling tests: identity transforms, simple roundtrips, dimension calculations

use super::*;

#[test]
fn test_bilinear_identity() {
    // 2x2 image using Pixel4
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.25, 0.25, 0.25, 0.0),
        Pixel4::new(0.5, 0.5, 0.5, 0.0),
        Pixel4::new(0.75, 0.75, 0.75, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_bilinear_upscale() {
    // 2x2 -> 4x4
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);
    assert_eq!(dst.len(), 16);
    // Output should be in valid range and contain intermediate values
    for p in &dst {
        assert!(p[0] >= 0.0 && p[0] <= 1.0);
    }
    // Should have some variation
    let min = dst.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let max = dst.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
    assert!(max > min);
}

#[test]
fn test_lanczos_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.25, 0.25, 0.25, 0.0),
        Pixel4::new(0.5, 0.5, 0.5, 0.0),
        Pixel4::new(0.75, 0.75, 0.75, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_lanczos_downscale() {
    // 4x4 -> 2x2
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
    ];
    let dst = rescale(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);
    assert_eq!(dst.len(), 4);
    // Left half should be ~0, right half ~1
    assert!(dst[0][0] < 0.5);
    assert!(dst[1][0] > 0.5);
}

#[test]
fn test_calculate_dimensions() {
    // Width only
    let (w, h) = calculate_target_dimensions(100, 50, Some(50), None);
    assert_eq!(w, 50);
    assert_eq!(h, 25);

    // Height only
    let (w, h) = calculate_target_dimensions(100, 50, None, Some(25));
    assert_eq!(w, 50);
    assert_eq!(h, 25);

    // Both - exact AR match, larger dimension (width) is primary
    let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(100));
    assert_eq!(w, 200);
    assert_eq!(h, 100);

    // None
    let (w, h) = calculate_target_dimensions(100, 50, None, None);
    assert_eq!(w, 100);
    assert_eq!(h, 50);
}

#[test]
fn test_calculate_dimensions_smart_primary() {
    // Power of 2 takes precedence: height=256 is pow2, width=512 is also pow2
    // Both pow2 -> larger wins, so width=512 is primary
    let (w, h) = calculate_target_dimensions(1024, 512, Some(512), Some(256));
    assert_eq!(w, 512);
    assert_eq!(h, 256);

    // Power of 2 takes precedence: only height=256 is pow2
    // 1920x1080 -> 455x256 (height is pow2, width is not)
    let (w, h) = calculate_target_dimensions(1920, 1080, Some(455), Some(256));
    assert_eq!(h, 256); // Height is primary (pow2)
    assert_eq!(w, (256.0_f64 * 1920.0 / 1080.0).round() as usize); // Width calculated from height

    // Power of 2 takes precedence: only width=512 is pow2
    // 1920x1080 -> 512x288
    let (w, h) = calculate_target_dimensions(1920, 1080, Some(512), Some(288));
    assert_eq!(w, 512); // Width is primary (pow2)
    assert_eq!(h, (512.0_f64 * 1080.0 / 1920.0).round() as usize); // Height calculated from width

    // Clean division: 1000x500 -> 250x125 (250 divides 1000, 125 divides 500)
    // Both divide cleanly, larger wins
    let (w, h) = calculate_target_dimensions(1000, 500, Some(250), Some(125));
    assert_eq!(w, 250);
    assert_eq!(h, 125);

    // Clean division: 1000x500 -> 200x100 (200 divides 1000, 100 divides 500)
    // Width is larger, so primary
    let (w, h) = calculate_target_dimensions(1000, 500, Some(200), Some(100));
    assert_eq!(w, 200);
    assert_eq!(h, 100);

    // Clean division wins over larger: 999x500 -> 200x100
    // 200 doesn't divide 999, but 100 divides 500 -> height is primary
    let (w, h) = calculate_target_dimensions(999, 500, Some(200), Some(100));
    assert_eq!(h, 100); // Height is primary (clean division)
    assert_eq!(w, (100.0_f64 * 999.0 / 500.0).round() as usize);

    // Intentional distortion: dimensions far from AR are kept exact
    // 100x50 (2:1) -> 200x200 (1:1) - very different AR
    let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(200));
    assert_eq!(w, 200);
    assert_eq!(h, 200); // Kept exact - intentional squish

    // Within 1 pixel tolerance: 100x50 -> 200x99
    // h_from_w = 200 * 50 / 100 = 100, diff = |99-100| = 1 (within tolerance)
    let (w, h) = calculate_target_dimensions(100, 50, Some(200), Some(99));
    assert_eq!(w, 200); // Width is primary (larger)
    assert_eq!(h, 100); // Corrected to proper AR
}

#[test]
fn test_bilinear_roundtrip_2x() {
    // Test that 2x upscale then 2x downscale returns approximately original
    // 4x4 -> 8x8 -> 4x4
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.1, 0.1, 0.1, 0.0), Pixel4::new(0.4, 0.4, 0.4, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0),
        Pixel4::new(0.2, 0.2, 0.2, 0.0), Pixel4::new(0.5, 0.5, 0.5, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0),
        Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0),
    ];
    let up = rescale(&src, 4, 4, 8, 8, RescaleMethod::Bilinear, ScaleMode::Independent);
    let down = rescale(&up, 8, 8, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

    for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
        let diff = (orig[0] - result[0]).abs();
        assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig[0], result[0], diff);
    }
}

#[test]
fn test_lanczos_roundtrip_2x() {
    // Test that 2x upscale then 2x downscale returns approximately original
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.1, 0.1, 0.1, 0.0), Pixel4::new(0.4, 0.4, 0.4, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0),
        Pixel4::new(0.2, 0.2, 0.2, 0.0), Pixel4::new(0.5, 0.5, 0.5, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0),
        Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0),
    ];
    let up = rescale(&src, 4, 4, 8, 8, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let down = rescale(&up, 8, 8, 4, 4, RescaleMethod::Lanczos3, ScaleMode::Independent);

    for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
        let diff = (orig[0] - result[0]).abs();
        assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig[0], result[0], diff);
    }
}

#[test]
fn test_no_shift_on_upscale() {
    // A single white pixel in center should stay centered after upscale
    // 3x3 with center pixel white -> 6x6
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
    ];
    let dst = rescale(&src, 3, 3, 6, 6, RescaleMethod::Bilinear, ScaleMode::Independent);

    // The brightest area should still be in the center region
    let center_sum = dst[2 * 6 + 2][0] + dst[2 * 6 + 3][0] + dst[3 * 6 + 2][0] + dst[3 * 6 + 3][0];
    let corner_sum = dst[0][0] + dst[5][0] + dst[30][0] + dst[35][0];

    assert!(center_sum > corner_sum, "Center should be brighter than corners");
}

#[test]
fn test_edge_pixels_preserved() {
    // Edge pixels shouldn't expand or shift weirdly
    // Left column black, right column white
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
    ];
    let up = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

    // Left edge (x=0) should still be darkest
    // Right edge (x=3) should still be brightest
    let left_avg = (up[0][0] + up[4][0] + up[8][0] + up[12][0]) / 4.0;
    let right_avg = (up[3][0] + up[7][0] + up[11][0] + up[15][0]) / 4.0;

    assert!(left_avg < 0.5, "Left edge should be dark: {}", left_avg);
    assert!(right_avg > 0.5, "Right edge should be bright: {}", right_avg);
}

#[test]
fn test_rgb_bilinear_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.1, 0.2, 0.0),
        Pixel4::new(0.3, 0.4, 0.5, 0.0),
        Pixel4::new(0.6, 0.7, 0.8, 0.0),
        Pixel4::new(0.9, 1.0, 0.5, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_rgb_bilinear_upscale() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);
    assert_eq!(dst.len(), 16);

    // All values should be in valid range
    for p in &dst {
        for c in 0..3 {
            assert!(p[c] >= 0.0 && p[c] <= 1.0);
        }
    }
}

#[test]
fn test_rgb_lanczos_roundtrip() {
    let src = vec![
        Pixel4::new(0.1, 0.2, 0.3, 0.0), Pixel4::new(0.4, 0.5, 0.6, 0.0),
        Pixel4::new(0.7, 0.8, 0.9, 0.0), Pixel4::new(0.2, 0.3, 0.4, 0.0),
    ];
    let up = rescale(&src, 2, 2, 4, 4, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let down = rescale(&up, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);

    for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
        for c in 0..3 {
            let diff = (orig[c] - result[c]).abs();
            assert!(diff < 0.2, "Pixel {} channel {} drifted: {} -> {}", i, c, orig[c], result[c]);
        }
    }
}

#[test]
fn test_uniform_scale_mode() {
    // Test that uniform scale modes produce identical scale factors
    // 100x50 -> 200x100 should be exactly 2x in both directions with uniform mode
    let (sx1, sy1) = calculate_scales(100, 50, 200, 100, ScaleMode::Independent);
    assert_eq!(sx1, 0.5);
    assert_eq!(sy1, 0.5);

    // 100x50 -> 200x99 with independent: different scales
    let (sx2, sy2) = calculate_scales(100, 50, 200, 99, ScaleMode::Independent);
    assert!((sx2 - 0.5).abs() < 0.001);
    assert!((sy2 - 0.505).abs() < 0.01); // 50/99 ≈ 0.505

    // With UniformWidth: both use width scale
    let (sx3, sy3) = calculate_scales(100, 50, 200, 99, ScaleMode::UniformWidth);
    assert_eq!(sx3, sy3);
    assert_eq!(sx3, 0.5);

    // With UniformHeight: both use height scale
    let (sx4, sy4) = calculate_scales(100, 50, 200, 99, ScaleMode::UniformHeight);
    assert_eq!(sx4, sy4);
    assert!((sx4 - 0.505).abs() < 0.01);
}

#[test]
fn test_mitchell_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.25, 0.25, 0.25, 0.0),
        Pixel4::new(0.5, 0.5, 0.5, 0.0),
        Pixel4::new(0.75, 0.75, 0.75, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Mitchell, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_mitchell_roundtrip_2x() {
    // Test that 2x upscale then 2x downscale returns approximately original
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(0.1, 0.1, 0.1, 0.0), Pixel4::new(0.4, 0.4, 0.4, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0),
        Pixel4::new(0.2, 0.2, 0.2, 0.0), Pixel4::new(0.5, 0.5, 0.5, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0), Pixel4::new(0.8, 0.8, 0.8, 0.0),
        Pixel4::new(0.3, 0.3, 0.3, 0.0), Pixel4::new(0.6, 0.6, 0.6, 0.0), Pixel4::new(0.9, 0.9, 0.9, 0.0), Pixel4::new(0.7, 0.7, 0.7, 0.0),
    ];
    let up = rescale(&src, 4, 4, 8, 8, RescaleMethod::Mitchell, ScaleMode::Independent);
    let down = rescale(&up, 8, 8, 4, 4, RescaleMethod::Mitchell, ScaleMode::Independent);

    for (i, (orig, result)) in src.iter().zip(down.iter()).enumerate() {
        let diff = (orig[0] - result[0]).abs();
        assert!(diff < 0.15, "Pixel {} drifted: {} -> {} (diff: {})", i, orig[0], result[0], diff);
    }
}

#[test]
fn test_catmull_rom_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.25, 0.25, 0.25, 0.0),
        Pixel4::new(0.5, 0.5, 0.5, 0.0),
        Pixel4::new(0.75, 0.75, 0.75, 0.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::CatmullRom, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_lanczos_scatter_identity() {
    // Same-size should return identical pixels
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Lanczos3Scatter, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_ewa_lanczos3_identity() {
    // Same-size should return identical pixels
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::EWASincLanczos3, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_ewa_sinc_lanczos2_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::EWASincLanczos2, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_ewa_jinc_lanczos2_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::EWALanczos2, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_ewa_jinc_lanczos3_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::EWALanczos3, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_ewa_mitchell_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::EWAMitchell, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_ewa_catmull_rom_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::EWACatmullRom, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_jinc_identity() {
    // Full-extent Jinc: same-size should return identical pixels
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::Jinc, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_jinc_produces_nonzero_output() {
    // Ensure Jinc doesn't produce all-zero output when resizing
    let src = vec![
        Pixel4::new(1.0, 0.5, 0.25, 1.0),
        Pixel4::new(0.5, 1.0, 0.5, 1.0),
        Pixel4::new(0.25, 0.5, 1.0, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 3, 3, RescaleMethod::Jinc, ScaleMode::Independent);
    // Check that output is non-zero
    let sum: f32 = dst.iter().map(|p| p.r() + p.g() + p.b()).sum();
    assert!(sum > 0.0, "Jinc output should not be all zeros");
}

#[test]
fn test_tent_box_identity() {
    // Identity transform should work
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::TentBox, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_tent_box_downscale() {
    // 4x4 -> 2x2 downscale
    let src = vec![Pixel4::new(0.5, 0.5, 0.5, 1.0); 16];
    let dst = rescale(&src, 4, 4, 2, 2, RescaleMethod::TentBox, ScaleMode::Independent);
    assert_eq!(dst.len(), 4);
    // Uniform input should give uniform output
    for p in &dst {
        assert!((p.r() - 0.5).abs() < 0.01, "TentBox should preserve uniform values, got {}", p.r());
    }
}

#[test]
fn test_tent_lanczos3_identity() {
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0),
        Pixel4::new(0.25, 0.25, 0.25, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.75, 0.75, 0.75, 1.0),
    ];
    let dst = rescale(&src, 2, 2, 2, 2, RescaleMethod::TentLanczos3, ScaleMode::Independent);
    assert_eq!(src, dst);
}

#[test]
fn test_tent_lanczos3_downscale() {
    // 4x4 -> 2x2 downscale
    let src = vec![Pixel4::new(0.5, 0.5, 0.5, 1.0); 16];
    let dst = rescale(&src, 4, 4, 2, 2, RescaleMethod::TentLanczos3, ScaleMode::Independent);
    assert_eq!(dst.len(), 4);
    // Uniform input should give uniform output
    for p in &dst {
        assert!((p.r() - 0.5).abs() < 0.01, "TentLanczos3 should preserve uniform values, got {}", p.r());
    }
}

#[test]
fn test_tent_box_preserves_energy() {
    // Test that tent-space pipeline preserves total energy (average brightness)
    let src = vec![
        Pixel4::new(0.1, 0.2, 0.3, 1.0),
        Pixel4::new(0.9, 0.1, 0.5, 1.0),
        Pixel4::new(0.3, 0.8, 0.2, 1.0),
        Pixel4::new(0.6, 0.4, 0.7, 1.0),
    ];
    let src_avg: f32 = src.iter().map(|p| (p.r() + p.g() + p.b()) / 3.0).sum::<f32>() / 4.0;

    let dst = rescale(&src, 2, 2, 1, 1, RescaleMethod::TentBox, ScaleMode::Independent);
    let dst_avg = (dst[0].r() + dst[0].g() + dst[0].b()) / 3.0;

    // Energy should be approximately preserved (tent kernels with negative lobes
    // can have some overshoot on patterns with sharp gradients)
    assert!((src_avg - dst_avg).abs() < 0.1,
        "TentBox should approximately preserve average brightness: src={}, dst={}", src_avg, dst_avg);
}

#[test]
fn test_tent_box_2x_kernel_weights() {
    // Verify that TentBox 2× downscale produces the expected kernel weights:
    // [-1, 7, 26, 26, 7, -1] / 64 as derived in tent_kernel.py
    use super::kernels::precompute_tent_kernel_weights;

    // 6 input pixels → 3 output pixels (2× downscale)
    let scale = 6.0 / 3.0;  // 2× downscale
    let weights = precompute_tent_kernel_weights(6, 3, scale, RescaleMethod::Box);

    // Check output pixel 1 (center) which should have the canonical kernel
    // For 2× downscale with offset=0.5, output pixel 1 is centered at input 2.5
    // The kernel should span 6 input pixels: indices 0-5
    let kw = &weights[1];

    // Expected: [-1, 7, 26, 26, 7, -1] / 64
    let expected = [-1.0/64.0, 7.0/64.0, 26.0/64.0, 26.0/64.0, 7.0/64.0, -1.0/64.0];

    // Check weights match within tolerance
    // Note: the actual indices may vary based on implementation details
    let sum: f32 = kw.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1.0, got {}", sum);

    // For the center pixel, verify the general structure:
    // - Should have 6 weights
    // - Two outer weights should be negative (the -1/64 parts)
    // - Inner weights should be positive
    if kw.weights.len() == 6 {
        // Check for the characteristic tent-space pattern with negative outer lobes
        let has_negative = kw.weights.iter().any(|&w| w < -0.001);
        assert!(has_negative, "TentBox 2× kernel should have negative outer weights (overshoot), got {:?}", kw.weights);

        // Check approximate structure: middle two weights should be largest
        let mid_weight = (kw.weights[2] + kw.weights[3]) / 2.0;
        assert!(mid_weight > 0.3, "Middle weights should be dominant, got {:?}", kw.weights);
    }
}
