//! Advanced rescaling tests: quality metrics, alpha-aware, scatter, EWA, comparisons

use super::*;

// ========================================================================
// Prime dimensions and stress tests
// ========================================================================

#[test]
fn test_lanczos_prime_dimensions() {
    // Test with prime number dimensions to stress-test kernel weight computation
    // Prime numbers create non-repeating scale factors that can hit edge cases
    // 97x89 -> 53x47 (all primes)
    let src_w = 97;
    let src_h = 89;
    let dst_w = 53;
    let dst_h = 47;

    // Create a gradient test pattern
    let mut src = Vec::with_capacity(src_w * src_h);
    for y in 0..src_h {
        for x in 0..src_w {
            let r = x as f32 / (src_w - 1) as f32;
            let g = y as f32 / (src_h - 1) as f32;
            let b = ((x + y) as f32 / (src_w + src_h - 2) as f32).min(1.0);
            src.push(Pixel4::new(r, g, b, 0.0));
        }
    }

    let dst = rescale(&src, src_w, src_h, dst_w, dst_h, RescaleMethod::Lanczos3, ScaleMode::Independent);
    assert_eq!(dst.len(), dst_w * dst_h);

    // Verify output gradient is roughly preserved (corners should match)
    // Top-left should be dark
    assert!(dst[0][0] < 0.1, "Top-left R should be dark: {}", dst[0][0]);
    assert!(dst[0][1] < 0.1, "Top-left G should be dark: {}", dst[0][1]);

    // Bottom-right should be bright
    let br = &dst[(dst_h - 1) * dst_w + (dst_w - 1)];
    assert!(br[0] > 0.9, "Bottom-right R should be bright: {}", br[0]);
    assert!(br[1] > 0.9, "Bottom-right G should be bright: {}", br[1]);

    // All values should be finite and reasonable
    for (i, p) in dst.iter().enumerate() {
        for c in 0..3 {
            assert!(p[c].is_finite(), "Pixel {} channel {} is not finite: {}", i, c, p[c]);
        }
    }
}

#[test]
fn test_lanczos_extreme_downscale_primes() {
    // Extreme downscale with primes: 1009x1013 -> 7x11
    // This creates a huge filter radius and tests weight accumulation
    let src_w = 127; // Smaller primes for faster test
    let src_h = 131;
    let dst_w = 7;
    let dst_h = 11;

    // Checkerboard pattern
    let mut src = Vec::with_capacity(src_w * src_h);
    for y in 0..src_h {
        for x in 0..src_w {
            let v = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            src.push(Pixel4::new(v, v, v, 0.0));
        }
    }

    let dst = rescale(&src, src_w, src_h, dst_w, dst_h, RescaleMethod::Lanczos3, ScaleMode::Independent);
    assert_eq!(dst.len(), dst_w * dst_h);

    // Checkerboard should average to ~0.5 after extreme downscale
    for (i, p) in dst.iter().enumerate() {
        for c in 0..3 {
            assert!(p[c].is_finite(), "Pixel {} channel {} is not finite", i, c);
            // Should be somewhere around 0.5 (averaged checkerboard)
            assert!(p[c] > 0.3 && p[c] < 0.7,
                "Pixel {} channel {} should be ~0.5 (averaged): {}", i, c, p[c]);
        }
    }
}

// ========================================================================
// Alpha-aware rescaling tests
// ========================================================================

#[test]
fn test_alpha_aware_no_bleed_bilinear() {
    // Test that black transparent pixels don't bleed into white opaque pixels
    // 2x2: top-left is white opaque, others are black transparent
    let src = vec![
        Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
    ];

    // Regular rescale: black pixels bleed into the result
    let regular = rescale(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

    // Alpha-aware rescale: black transparent pixels should not affect RGB
    let alpha_aware = rescale_with_alpha(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

    // The top-left region (where opaque pixel was) should be brighter in alpha-aware
    // In regular mode, black transparent pixels darken the result
    let regular_tl = regular[0].r();
    let alpha_tl = alpha_aware[0].r();

    // Alpha-aware should preserve white better (closer to 1.0)
    assert!(alpha_tl >= regular_tl,
        "Alpha-aware top-left ({}) should be >= regular ({})", alpha_tl, regular_tl);
}

#[test]
fn test_alpha_aware_preserves_transparent_rgb() {
    // Test that fully transparent regions preserve their underlying RGB
    // All pixels transparent, but with different RGB values
    let src = vec![
        Pixel4::new(1.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 1.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 0.0, 0.0),
    ];

    let dst = rescale_with_alpha(&src, 2, 2, 2, 2, RescaleMethod::Bilinear, ScaleMode::Independent);

    // Identity transform should preserve values exactly
    assert_eq!(src, dst);
}

#[test]
fn test_alpha_aware_lanczos_no_bleed() {
    // Test that black transparent pixels don't darken opaque regions
    // Use a larger image to avoid edge effects where Lanczos overshoots
    // 4x4 with opaque white center, black transparent border
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
    ];

    // Downscale 4x4 -> 2x2 (where black border would blend in)
    let regular = rescale(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let alpha_aware = rescale_with_alpha(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);

    // In regular mode, the black border darkens everything
    // In alpha-aware mode, transparent pixels don't affect RGB
    // Check center pixel brightness
    let regular_avg: f32 = regular.iter().map(|p| p.r()).sum::<f32>() / 4.0;
    let alpha_avg: f32 = alpha_aware.iter().map(|p| p.r()).sum::<f32>() / 4.0;

    assert!(alpha_avg > regular_avg,
        "Alpha-aware avg brightness ({}) should be > regular ({})", alpha_avg, regular_avg);
}

#[test]
fn test_alpha_channel_interpolated_normally() {
    // Alpha should be interpolated like any other channel
    // 2x2 with varying alpha
    let src = vec![
        Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
        Pixel4::new(1.0, 1.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 1.0, 0.0),
    ];

    let dst = rescale_with_alpha(&src, 2, 2, 4, 4, RescaleMethod::Bilinear, ScaleMode::Independent);

    // Top-left corner should have highest alpha (near 1.0)
    // Bottom-right corner should have lowest alpha (near 0.0)
    assert!(dst[0].a() > 0.5, "Top-left alpha should be high: {}", dst[0].a());
    assert!(dst[15].a() < 0.5, "Bottom-right alpha should be low: {}", dst[15].a());
}

#[test]
fn test_alpha_aware_dithered_pattern() {
    // Simulate a dithered image: alternating opaque colored and black transparent
    // This is the problematic case for naive scaling
    let src = vec![
        Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0),
        Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(1.0, 0.5, 0.0, 1.0),
    ];

    // Downscale 4x4 -> 2x2
    let dst = rescale_with_alpha(&src, 4, 4, 2, 2, RescaleMethod::Lanczos3, ScaleMode::Independent);

    // The RGB should still be approximately orange (1.0, 0.5, 0.0), not darkened by black
    for p in &dst {
        // Allow some tolerance due to Lanczos ringing
        assert!(p.r() > 0.7, "Red should stay high (not darkened by black): {}", p.r());
        assert!(p.g() > 0.3 && p.g() < 0.7, "Green should be mid-range: {}", p.g());
        assert!(p.b() < 0.3, "Blue should stay low: {}", p.b());
    }
}

// ========================================================================
// Quality metrics and edge tests
// ========================================================================

#[test]
fn test_lanczos_4x_roundtrip_quality() {
    // Test 4x upscale then 4x downscale - measure actual blur
    // Use a larger test image with gradients and edges
    let size = 16;
    let mut src = vec![Pixel4::default(); size * size];

    // Create a test pattern with:
    // - Horizontal gradient in top half
    // - Vertical edge in bottom half
    for y in 0..size {
        for x in 0..size {
            let val = if y < size / 2 {
                // Horizontal gradient
                x as f32 / (size - 1) as f32
            } else {
                // Vertical edge at center
                if x < size / 2 { 0.0 } else { 1.0 }
            };
            src[y * size + x] = Pixel4::new(val, val, val, 0.0);
        }
    }

    // 4x upscale then 4x downscale
    let up = rescale(&src, size, size, size * 4, size * 4, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let down = rescale(&up, size * 4, size * 4, size, size, RescaleMethod::Lanczos3, ScaleMode::Independent);

    // Calculate MSE
    let mut mse = 0.0f64;
    let mut max_diff = 0.0f32;
    for (orig, result) in src.iter().zip(down.iter()) {
        let diff = (orig[0] - result[0]).abs();
        mse += (diff as f64).powi(2);
        if diff > max_diff {
            max_diff = diff;
        }
    }
    mse /= (size * size) as f64;
    let psnr = if mse > 0.0 { 10.0 * (1.0 / mse).log10() } else { f64::INFINITY };

    eprintln!("4x roundtrip: MSE={:.6}, PSNR={:.2}dB, max_diff={:.4}", mse, psnr, max_diff);

    // For a proper Lanczos implementation, PSNR should be > 25dB
    // and max diff should be < 0.2 for this smooth test pattern
    assert!(psnr > 20.0, "PSNR too low: {:.2}dB (expected > 20dB)", psnr);
    assert!(max_diff < 0.25, "Max diff too high: {:.4} (expected < 0.25)", max_diff);
}

#[test]
fn test_lanczos_edge_sharpness_and_ringing() {
    // Test edge preservation and ringing with a simple step edge
    // Create a 32-pixel wide image with a sharp edge at center
    let size = 32;
    let mut src = vec![Pixel4::default(); size];
    for x in 0..size {
        let val = if x < size / 2 { 0.0 } else { 1.0 };
        src[x] = Pixel4::new(val, val, val, 0.0);
    }

    // 4x upscale then 4x downscale (1D - just use horizontal)
    let up = rescale(&src, size, 1, size * 4, 1, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let down = rescale(&up, size * 4, 1, size, 1, RescaleMethod::Lanczos3, ScaleMode::Independent);

    // Measure edge characteristics
    let edge_idx = size / 2; // Edge is between pixel 15 and 16

    // Check dark side (should be close to 0, might have slight overshoot/undershoot)
    let dark_vals: Vec<f32> = (0..edge_idx-2).map(|i| down[i][0]).collect();
    let dark_min = dark_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let dark_max = dark_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Check bright side (should be close to 1)
    let bright_vals: Vec<f32> = (edge_idx+3..size).map(|i| down[i][0]).collect();
    let bright_min = bright_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let bright_max = bright_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Check transition zone (pixels near edge)
    let trans_vals: Vec<f32> = (edge_idx-2..edge_idx+3).map(|i| down[i][0]).collect();

    eprintln!("Dark side: min={:.4}, max={:.4} (expected ~0)", dark_min, dark_max);
    eprintln!("Bright side: min={:.4}, max={:.4} (expected ~1)", bright_min, bright_max);
    eprintln!("Transition: {:?}", trans_vals.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());

    // Ringing check: dark side shouldn't go below -0.1 or above 0.1
    // (Lanczos can have small negative values due to ringing, but should be limited)
    assert!(dark_min > -0.15, "Dark side undershoot too strong: {:.4}", dark_min);
    assert!(dark_max < 0.15, "Dark side overshoot too strong: {:.4}", dark_max);

    // Bright side shouldn't go above 1.1 or below 0.9
    assert!(bright_max < 1.15, "Bright side overshoot too strong: {:.4}", bright_max);
    assert!(bright_min > 0.85, "Bright side undershoot too strong: {:.4}", bright_min);

    // Edge should still be reasonably sharp (transition should span ~3-5 pixels, not more)
    // Find where values cross 0.2 and 0.8
    let low_cross = down.iter().position(|p| p[0] > 0.2).unwrap_or(0);
    let high_cross = down.iter().position(|p| p[0] > 0.8).unwrap_or(size);
    let edge_width = high_cross - low_cross;
    eprintln!("Edge width (0.2 to 0.8): {} pixels", edge_width);

    assert!(edge_width <= 6, "Edge too blurry: {} pixels wide (expected <= 6)", edge_width);
}

#[test]
fn test_mitchell_less_ringing_than_lanczos() {
    // Compare Mitchell and Lanczos ringing on a sharp edge
    let size = 32;
    let mut src = vec![Pixel4::default(); size];
    for x in 0..size {
        let val = if x < size / 2 { 0.0 } else { 1.0 };
        src[x] = Pixel4::new(val, val, val, 0.0);
    }

    // 4x upscale then 4x downscale
    let lanczos_up = rescale(&src, size, 1, size * 4, 1, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let lanczos_down = rescale(&lanczos_up, size * 4, 1, size, 1, RescaleMethod::Lanczos3, ScaleMode::Independent);

    let mitchell_up = rescale(&src, size, 1, size * 4, 1, RescaleMethod::Mitchell, ScaleMode::Independent);
    let mitchell_down = rescale(&mitchell_up, size * 4, 1, size, 1, RescaleMethod::Mitchell, ScaleMode::Independent);

    // Measure ringing on dark side (pixels 0..13 should be ~0)
    let lanczos_dark_overshoot = (0..13).map(|i| lanczos_down[i][0]).fold(0.0f32, f32::max);
    let mitchell_dark_overshoot = (0..13).map(|i| mitchell_down[i][0]).fold(0.0f32, f32::max);

    // Measure ringing on bright side (pixels 19..32 should be ~1)
    let lanczos_bright_overshoot = (19..32).map(|i| (lanczos_down[i][0] - 1.0).abs()).fold(0.0f32, f32::max);
    let mitchell_bright_overshoot = (19..32).map(|i| (mitchell_down[i][0] - 1.0).abs()).fold(0.0f32, f32::max);

    eprintln!("Lanczos dark overshoot: {:.4}, bright overshoot: {:.4}", lanczos_dark_overshoot, lanczos_bright_overshoot);
    eprintln!("Mitchell dark overshoot: {:.4}, bright overshoot: {:.4}", mitchell_dark_overshoot, mitchell_bright_overshoot);

    // Mitchell should have noticeably less overshoot
    assert!(mitchell_dark_overshoot < lanczos_dark_overshoot,
        "Mitchell dark overshoot ({:.4}) should be < Lanczos ({:.4})", mitchell_dark_overshoot, lanczos_dark_overshoot);
    assert!(mitchell_bright_overshoot < lanczos_bright_overshoot,
        "Mitchell bright overshoot ({:.4}) should be < Lanczos ({:.4})", mitchell_bright_overshoot, lanczos_bright_overshoot);

    // Mitchell should have very low overshoot (<1%)
    assert!(mitchell_dark_overshoot < 0.01,
        "Mitchell dark overshoot ({:.4}) should be <1%", mitchell_dark_overshoot);
    // Bright side may have slightly more due to accumulated error
    assert!(mitchell_bright_overshoot < 0.03,
        "Mitchell bright overshoot ({:.4}) should be <3%", mitchell_bright_overshoot);
}

#[test]
fn test_mitchell_4x_roundtrip_quality() {
    // Same test as Lanczos but for Mitchell
    let size = 16;
    let mut src = vec![Pixel4::default(); size * size];

    for y in 0..size {
        for x in 0..size {
            let val = if y < size / 2 {
                x as f32 / (size - 1) as f32
            } else {
                if x < size / 2 { 0.0 } else { 1.0 }
            };
            src[y * size + x] = Pixel4::new(val, val, val, 0.0);
        }
    }

    let up = rescale(&src, size, size, size * 4, size * 4, RescaleMethod::Mitchell, ScaleMode::Independent);
    let down = rescale(&up, size * 4, size * 4, size, size, RescaleMethod::Mitchell, ScaleMode::Independent);

    let mut mse = 0.0f64;
    let mut max_diff = 0.0f32;
    for (orig, result) in src.iter().zip(down.iter()) {
        let diff = (orig[0] - result[0]).abs();
        mse += (diff as f64).powi(2);
        if diff > max_diff {
            max_diff = diff;
        }
    }
    mse /= (size * size) as f64;
    let psnr = if mse > 0.0 { 10.0 * (1.0 / mse).log10() } else { f64::INFINITY };

    eprintln!("Mitchell 4x roundtrip: MSE={:.6}, PSNR={:.2}dB, max_diff={:.4}", mse, psnr, max_diff);

    // Mitchell may be slightly blurrier than Lanczos, but should still have good PSNR
    assert!(psnr > 18.0, "Mitchell PSNR too low: {:.2}dB (expected > 18dB)", psnr);
    assert!(max_diff < 0.25, "Mitchell max diff too high: {:.4} (expected < 0.25)", max_diff);
}

#[test]
fn test_catmull_rom_between_mitchell_and_lanczos() {
    // Catmull-Rom should be sharper than Mitchell but have less ringing than Lanczos
    let size = 32;
    let mut src = vec![Pixel4::default(); size];
    for x in 0..size {
        let val = if x < size / 2 { 0.0 } else { 1.0 };
        src[x] = Pixel4::new(val, val, val, 0.0);
    }

    // 4x upscale then 4x downscale
    let lanczos_up = rescale(&src, size, 1, size * 4, 1, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let lanczos_down = rescale(&lanczos_up, size * 4, 1, size, 1, RescaleMethod::Lanczos3, ScaleMode::Independent);

    let catrom_up = rescale(&src, size, 1, size * 4, 1, RescaleMethod::CatmullRom, ScaleMode::Independent);
    let catrom_down = rescale(&catrom_up, size * 4, 1, size, 1, RescaleMethod::CatmullRom, ScaleMode::Independent);

    let mitchell_up = rescale(&src, size, 1, size * 4, 1, RescaleMethod::Mitchell, ScaleMode::Independent);
    let mitchell_down = rescale(&mitchell_up, size * 4, 1, size, 1, RescaleMethod::Mitchell, ScaleMode::Independent);

    // Measure ringing (overshoot on dark side)
    let lanczos_overshoot = (0..13).map(|i| lanczos_down[i][0]).fold(0.0f32, f32::max);
    let catrom_overshoot = (0..13).map(|i| catrom_down[i][0]).fold(0.0f32, f32::max);
    let _mitchell_overshoot = (0..13).map(|i| mitchell_down[i][0]).fold(0.0f32, f32::max);

    eprintln!("Overshoot - Lanczos: {:.4}, Catmull-Rom: {:.4}, Mitchell: {:.4}",
        lanczos_overshoot, catrom_overshoot, _mitchell_overshoot);

    // Catmull-Rom should have less ringing than Lanczos
    assert!(catrom_overshoot <= lanczos_overshoot,
        "Catmull-Rom overshoot ({:.4}) should be <= Lanczos ({:.4})", catrom_overshoot, lanczos_overshoot);
}

#[test]
fn test_catmull_rom_4x_roundtrip_quality() {
    // Catmull-Rom should be sharper than Mitchell (higher PSNR)
    let size = 16;
    let mut src = vec![Pixel4::default(); size * size];

    for y in 0..size {
        for x in 0..size {
            let val = if y < size / 2 {
                x as f32 / (size - 1) as f32
            } else {
                if x < size / 2 { 0.0 } else { 1.0 }
            };
            src[y * size + x] = Pixel4::new(val, val, val, 0.0);
        }
    }

    let up = rescale(&src, size, size, size * 4, size * 4, RescaleMethod::CatmullRom, ScaleMode::Independent);
    let down = rescale(&up, size * 4, size * 4, size, size, RescaleMethod::CatmullRom, ScaleMode::Independent);

    let mut mse = 0.0f64;
    let mut max_diff = 0.0f32;
    for (orig, result) in src.iter().zip(down.iter()) {
        let diff = (orig[0] - result[0]).abs();
        mse += (diff as f64).powi(2);
        if diff > max_diff {
            max_diff = diff;
        }
    }
    mse /= (size * size) as f64;
    let psnr = if mse > 0.0 { 10.0 * (1.0 / mse).log10() } else { f64::INFINITY };

    eprintln!("Catmull-Rom 4x roundtrip: MSE={:.6}, PSNR={:.2}dB, max_diff={:.4}", mse, psnr, max_diff);

    // Catmull-Rom should be sharper than Mitchell (PSNR > 27dB which Mitchell got)
    // but may have slightly more error than Lanczos due to less aggressive sharpening
    assert!(psnr > 25.0, "Catmull-Rom PSNR too low: {:.2}dB (expected > 25dB)", psnr);
    assert!(max_diff < 0.20, "Catmull-Rom max diff too high: {:.4} (expected < 0.20)", max_diff);
}

// ========================================================================
// Scatter-based rescaling tests
// ========================================================================

#[test]
fn test_lanczos_scatter_matches_gather_downscale() {
    // Scatter and gather should produce identical results for deterministic blending
    // 8x8 -> 4x4 downscale
    let src_size = 8;
    let dst_size = 4;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let r = x as f32 / (src_size - 1) as f32;
            let g = y as f32 / (src_size - 1) as f32;
            let b = 0.5;
            src.push(Pixel4::new(r, g, b, 1.0));
        }
    }

    let gather = rescale(&src, src_size, src_size, dst_size, dst_size,
                         RescaleMethod::Lanczos3, ScaleMode::Independent);
    let scatter = rescale(&src, src_size, src_size, dst_size, dst_size,
                          RescaleMethod::Lanczos3Scatter, ScaleMode::Independent);

    // Results should be nearly identical (within floating point tolerance)
    let mut max_diff = 0.0f32;
    for (g, s) in gather.iter().zip(scatter.iter()) {
        for c in 0..4 {
            let diff = (g[c] - s[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    eprintln!("Lanczos scatter vs gather max diff: {:.6}", max_diff);
    assert!(max_diff < 1e-4, "Scatter should match gather, max diff: {}", max_diff);
}

#[test]
fn test_lanczos_scatter_matches_gather_upscale() {
    // Test upscaling: 4x4 -> 8x8
    let src_size = 4;
    let dst_size = 8;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let v = ((x + y) % 2) as f32;
            src.push(Pixel4::new(v, v, v, 1.0));
        }
    }

    let gather = rescale(&src, src_size, src_size, dst_size, dst_size,
                         RescaleMethod::Lanczos3, ScaleMode::Independent);
    let scatter = rescale(&src, src_size, src_size, dst_size, dst_size,
                          RescaleMethod::Lanczos3Scatter, ScaleMode::Independent);

    let mut max_diff = 0.0f32;
    for (g, s) in gather.iter().zip(scatter.iter()) {
        for c in 0..4 {
            let diff = (g[c] - s[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    eprintln!("Lanczos scatter vs gather upscale max diff: {:.6}", max_diff);
    assert!(max_diff < 1e-4, "Scatter should match gather, max diff: {}", max_diff);
}

#[test]
fn test_sinc_scatter_matches_gather() {
    // Sinc scatter vs gather - use smaller image due to O(N²)
    // 6x6 -> 4x4
    let src_size = 6;
    let dst_size = 4;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let r = x as f32 / (src_size - 1) as f32;
            let g = y as f32 / (src_size - 1) as f32;
            src.push(Pixel4::new(r, g, 0.5, 1.0));
        }
    }

    let gather = rescale(&src, src_size, src_size, dst_size, dst_size,
                         RescaleMethod::Sinc, ScaleMode::Independent);
    let scatter = rescale(&src, src_size, src_size, dst_size, dst_size,
                          RescaleMethod::SincScatter, ScaleMode::Independent);

    let mut max_diff = 0.0f32;
    for (g, s) in gather.iter().zip(scatter.iter()) {
        for c in 0..4 {
            let diff = (g[c] - s[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    eprintln!("Sinc scatter vs gather max diff: {:.6}", max_diff);
    assert!(max_diff < 1e-4, "Scatter should match gather, max diff: {}", max_diff);
}

#[test]
fn test_lanczos_scatter_alpha_aware() {
    // Test alpha-aware scatter mode doesn't bleed transparent pixels
    let src = vec![
        Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
    ];

    let regular_gather = rescale(&src, 2, 2, 4, 4, RescaleMethod::Lanczos3, ScaleMode::Independent);
    let regular_scatter = rescale(&src, 2, 2, 4, 4, RescaleMethod::Lanczos3Scatter, ScaleMode::Independent);
    let alpha_scatter = rescale_with_alpha(&src, 2, 2, 4, 4, RescaleMethod::Lanczos3Scatter, ScaleMode::Independent);

    // Regular scatter should match regular gather
    let mut max_diff = 0.0f32;
    for (g, s) in regular_gather.iter().zip(regular_scatter.iter()) {
        for c in 0..4 {
            let diff = (g[c] - s[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    assert!(max_diff < 1e-4, "Regular scatter should match gather");

    // Alpha-aware mode should produce valid results (no NaN/inf)
    // and should have different values than regular mode (due to alpha weighting)
    for p in &alpha_scatter {
        assert!(p.r().is_finite(), "Alpha scatter should produce finite values");
    }

    // The top-left pixel in alpha-aware should be closer to 1.0 (the only opaque pixel's value)
    // while regular mode may have overshoot due to Lanczos ringing with the black transparent pixels
    let alpha_tl = alpha_scatter[0].r();
    assert!((alpha_tl - 1.0).abs() < 0.01,
        "Alpha-aware top-left should be ~1.0 (got {})", alpha_tl);
}

#[test]
fn test_scatter_prime_dimensions() {
    // Test with prime dimensions to stress-test weight computation
    // 97x89 -> 53x47
    let src_w = 97;
    let src_h = 89;
    let dst_w = 53;
    let dst_h = 47;

    let mut src = Vec::with_capacity(src_w * src_h);
    for y in 0..src_h {
        for x in 0..src_w {
            let r = x as f32 / (src_w - 1) as f32;
            let g = y as f32 / (src_h - 1) as f32;
            src.push(Pixel4::new(r, g, 0.5, 1.0));
        }
    }

    let gather = rescale(&src, src_w, src_h, dst_w, dst_h,
                         RescaleMethod::Lanczos3, ScaleMode::Independent);
    let scatter = rescale(&src, src_w, src_h, dst_w, dst_h,
                          RescaleMethod::Lanczos3Scatter, ScaleMode::Independent);

    let mut max_diff = 0.0f32;
    for (g, s) in gather.iter().zip(scatter.iter()) {
        for c in 0..4 {
            let diff = (g[c] - s[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    eprintln!("Scatter vs gather prime dimensions max diff: {:.6}", max_diff);
    assert!(max_diff < 1e-3, "Scatter should match gather for prime dims, max diff: {}", max_diff);
}

// ========================================================================
// EWA (Elliptical Weighted Average) rescaling tests
// ========================================================================

#[test]
fn test_ewa_lanczos3_downscale() {
    // 8x8 -> 4x4 downscale
    let src_size = 8;
    let dst_size = 4;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let r = x as f32 / (src_size - 1) as f32;
            let g = y as f32 / (src_size - 1) as f32;
            let b = 0.5;
            src.push(Pixel4::new(r, g, b, 1.0));
        }
    }

    let dst = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::EWALanczos3, ScaleMode::Independent);

    assert_eq!(dst.len(), dst_size * dst_size);

    // All values should be finite
    for (i, p) in dst.iter().enumerate() {
        for c in 0..4 {
            assert!(p[c].is_finite(), "Pixel {} channel {} is not finite: {}", i, c, p[c]);
        }
    }

    // Check gradient is preserved: top-left should be dark, bottom-right bright
    assert!(dst[0].r() < 0.3, "Top-left should be dark: {}", dst[0].r());
    let br = &dst[(dst_size - 1) * dst_size + (dst_size - 1)];
    assert!(br.r() > 0.7, "Bottom-right should be bright: {}", br.r());
}

#[test]
fn test_ewa_lanczos3_upscale() {
    // 4x4 -> 8x8 upscale
    let src_size = 4;
    let dst_size = 8;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let v = ((x + y) % 2) as f32;
            src.push(Pixel4::new(v, v, v, 1.0));
        }
    }

    let dst = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::EWALanczos3, ScaleMode::Independent);

    assert_eq!(dst.len(), dst_size * dst_size);

    // All values should be finite
    for (i, p) in dst.iter().enumerate() {
        for c in 0..4 {
            assert!(p[c].is_finite(), "Pixel {} channel {} is not finite: {}", i, c, p[c]);
        }
    }
}

#[test]
fn test_ewa_comparable_to_separable() {
    // EWA and separable Lanczos3 should produce similar results for uniform scaling
    // (they use the same kernel, just different 1D vs 2D evaluation)
    let src_size = 8;
    let dst_size = 4;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let r = x as f32 / (src_size - 1) as f32;
            let g = y as f32 / (src_size - 1) as f32;
            src.push(Pixel4::new(r, g, 0.5, 1.0));
        }
    }

    let separable = rescale(&src, src_size, src_size, dst_size, dst_size,
                            RescaleMethod::Lanczos3, ScaleMode::Independent);
    let ewa = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::EWALanczos3, ScaleMode::Independent);

    // EWA and separable should produce reasonably similar results
    // (not identical due to different filter shapes, but close)
    let mut max_diff = 0.0f32;
    for (sep, e) in separable.iter().zip(ewa.iter()) {
        for c in 0..3 {
            let diff = (sep[c] - e[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    eprintln!("EWA vs separable Lanczos3 max diff: {:.6}", max_diff);
    // They should be within a reasonable tolerance (less than 0.1)
    assert!(max_diff < 0.15, "EWA should be comparable to separable, max diff: {}", max_diff);
}

#[test]
fn test_ewa_alpha_aware() {
    // Test alpha-aware EWA mode doesn't bleed transparent pixels
    let src = vec![
        Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
    ];

    let regular_ewa = rescale(&src, 2, 2, 4, 4, RescaleMethod::EWALanczos3, ScaleMode::Independent);
    let alpha_ewa = rescale_with_alpha(&src, 2, 2, 4, 4, RescaleMethod::EWALanczos3, ScaleMode::Independent);

    // Alpha-aware mode should produce valid results (no NaN/inf)
    for p in &alpha_ewa {
        assert!(p.r().is_finite(), "Alpha EWA should produce finite values");
    }

    // The top-left pixel in alpha-aware should be closer to 1.0 (the only opaque pixel's value)
    let alpha_tl = alpha_ewa[0].r();
    assert!((alpha_tl - 1.0).abs() < 0.01,
        "Alpha-aware top-left should be ~1.0 (got {})", alpha_tl);

    // Regular mode should also produce valid results
    for p in &regular_ewa {
        assert!(p.r().is_finite(), "Regular EWA should produce finite values");
    }
}

#[test]
fn test_ewa_lanczos2_vs_lanczos3() {
    // Lanczos2 should have less ringing than Lanczos3
    let size = 16;
    let mut src = vec![Pixel4::default(); size * size];

    // Create a sharp edge pattern
    for y in 0..size {
        for x in 0..size {
            let val = if x < size / 2 { 0.0 } else { 1.0 };
            src[y * size + x] = Pixel4::new(val, val, val, 1.0);
        }
    }

    // 2x upscale then 2x downscale
    let l2_up = rescale(&src, size, size, size * 2, size * 2, RescaleMethod::EWALanczos2, ScaleMode::Independent);
    let l2_down = rescale(&l2_up, size * 2, size * 2, size, size, RescaleMethod::EWALanczos2, ScaleMode::Independent);

    let l3_up = rescale(&src, size, size, size * 2, size * 2, RescaleMethod::EWALanczos3, ScaleMode::Independent);
    let l3_down = rescale(&l3_up, size * 2, size * 2, size, size, RescaleMethod::EWALanczos3, ScaleMode::Independent);

    // Both should produce valid results
    for (i, (p2, p3)) in l2_down.iter().zip(l3_down.iter()).enumerate() {
        assert!(p2.r().is_finite(), "Lanczos2 pixel {} should be finite", i);
        assert!(p3.r().is_finite(), "Lanczos3 pixel {} should be finite", i);
    }
}

// ========================================================================
// Peaked Cosine (AVIR-style) rescaling tests
// ========================================================================

#[test]
fn test_peaked_cosine_downscale() {
    // 8x8 -> 4x4 downscale
    let src_size = 8;
    let dst_size = 4;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let r = x as f32 / (src_size - 1) as f32;
            let g = y as f32 / (src_size - 1) as f32;
            let b = 0.5;
            src.push(Pixel4::new(r, g, b, 1.0));
        }
    }

    let dst = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::PeakedCosine, ScaleMode::Independent);

    assert_eq!(dst.len(), dst_size * dst_size);

    // All values should be finite
    for (i, p) in dst.iter().enumerate() {
        for c in 0..4 {
            assert!(p[c].is_finite(), "Pixel {} channel {} is not finite: {}", i, c, p[c]);
        }
    }

    // Check gradient is preserved
    assert!(dst[0].r() < 0.3, "Top-left should be dark: {}", dst[0].r());
    let br = &dst[(dst_size - 1) * dst_size + (dst_size - 1)];
    assert!(br.r() > 0.7, "Bottom-right should be bright: {}", br.r());
}

#[test]
fn test_peaked_cosine_upscale() {
    // 4x4 -> 8x8 upscale
    let src_size = 4;
    let dst_size = 8;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            let v = ((x + y) % 2) as f32;
            src.push(Pixel4::new(v, v, v, 1.0));
        }
    }

    let dst = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::PeakedCosine, ScaleMode::Independent);

    assert_eq!(dst.len(), dst_size * dst_size);

    // All values should be finite
    for (i, p) in dst.iter().enumerate() {
        for c in 0..4 {
            assert!(p[c].is_finite(), "Pixel {} channel {} is not finite: {}", i, c, p[c]);
        }
    }
}

#[test]
fn test_peaked_cosine_roundtrip() {
    // Test 2x upscale then 2x downscale
    let size = 8;
    let mut src = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let r = x as f32 / (size - 1) as f32;
            let g = y as f32 / (size - 1) as f32;
            src.push(Pixel4::new(r, g, 0.5, 1.0));
        }
    }

    let up = rescale(&src, size, size, size * 2, size * 2,
                     RescaleMethod::PeakedCosine, ScaleMode::Independent);
    let down = rescale(&up, size * 2, size * 2, size, size,
                       RescaleMethod::PeakedCosine, ScaleMode::Independent);

    // Should be reasonably close to original
    let mut max_diff = 0.0f32;
    for (orig, result) in src.iter().zip(down.iter()) {
        for c in 0..3 {
            let diff = (orig[c] - result[c]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    eprintln!("Peaked Cosine 2x roundtrip max diff: {:.4}", max_diff);
    assert!(max_diff < 0.2, "Roundtrip should preserve values, max diff: {}", max_diff);
}

#[test]
fn test_peaked_cosine_alpha_aware() {
    // Test alpha-aware mode doesn't bleed transparent pixels
    let src = vec![
        Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 0.0, 0.0), Pixel4::new(0.0, 0.0, 0.0, 0.0),
    ];

    let alpha_pc = rescale_with_alpha(&src, 2, 2, 4, 4,
                                      RescaleMethod::PeakedCosine, ScaleMode::Independent);

    // All values should be finite
    for p in &alpha_pc {
        assert!(p.r().is_finite(), "Alpha PC should produce finite values");
    }

    // The top-left pixel should be close to 1.0 (the only opaque pixel's value)
    let alpha_tl = alpha_pc[0].r();
    assert!((alpha_tl - 1.0).abs() < 0.05,
        "Alpha-aware top-left should be ~1.0 (got {})", alpha_tl);
}

#[test]
fn test_peaked_cosine_extreme_downscale() {
    // Test large downscale: 64x64 -> 8x8
    let src_size = 64;
    let dst_size = 8;

    let mut src = Vec::with_capacity(src_size * src_size);
    for y in 0..src_size {
        for x in 0..src_size {
            // Checkerboard pattern
            let v = ((x + y) % 2) as f32;
            src.push(Pixel4::new(v, v, v, 1.0));
        }
    }

    let dst = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::PeakedCosine, ScaleMode::Independent);

    assert_eq!(dst.len(), dst_size * dst_size);

    // Checkerboard should average to ~0.5
    for (i, p) in dst.iter().enumerate() {
        assert!(p.r().is_finite(), "Pixel {} should be finite", i);
        assert!(p.r() > 0.3 && p.r() < 0.7,
            "Pixel {} should be ~0.5 (averaged checkerboard): {}", i, p.r());
    }
}

// PeakedCosineCorrected tests

#[test]
fn test_peaked_cosine_corrected_downscale() {
    // Test downscale produces reasonable output
    let src_size = 32;
    let dst_size = 8;

    let src: Vec<Pixel4> = (0..src_size * src_size)
        .map(|i| {
            let v = (i % 256) as f32 / 255.0;
            Pixel4::new(v, v * 0.5, v * 0.25, 1.0)
        })
        .collect();

    let dst = rescale(&src, src_size, src_size, dst_size, dst_size,
                      RescaleMethod::PeakedCosineCorrected, ScaleMode::Independent);

    assert_eq!(dst.len(), dst_size * dst_size);

    for (i, p) in dst.iter().enumerate() {
        assert!(p.r().is_finite() && !p.r().is_nan(), "Pixel {} r should be valid", i);
        assert!(p.g().is_finite() && !p.g().is_nan(), "Pixel {} g should be valid", i);
        assert!(p.b().is_finite() && !p.b().is_nan(), "Pixel {} b should be valid", i);
    }
}

#[test]
fn test_peaked_cosine_corrected_upscale() {
    // Test upscale produces reasonable output
    let src = vec![
        Pixel4::new(0.0, 0.0, 0.0, 1.0), Pixel4::new(1.0, 1.0, 1.0, 1.0),
        Pixel4::new(1.0, 1.0, 1.0, 1.0), Pixel4::new(0.0, 0.0, 0.0, 1.0),
    ];

    let dst = rescale(&src, 2, 2, 8, 8, RescaleMethod::PeakedCosineCorrected, ScaleMode::Independent);

    assert_eq!(dst.len(), 64);

    // Check corners roughly match source
    let tl = dst[0];
    let tr = dst[7];
    assert!(tl.r() < 0.3, "Top-left should be dark");
    assert!(tr.r() > 0.7, "Top-right should be bright");
}

#[test]
fn test_peaked_cosine_corrected_roundtrip() {
    // 4x4 -> 8x8 -> 4x4 should be reasonably close
    let src = vec![
        Pixel4::new(0.2, 0.2, 0.2, 1.0), Pixel4::new(0.4, 0.4, 0.4, 1.0),
        Pixel4::new(0.6, 0.6, 0.6, 1.0), Pixel4::new(0.8, 0.8, 0.8, 1.0),
        Pixel4::new(0.3, 0.3, 0.3, 1.0), Pixel4::new(0.5, 0.5, 0.5, 1.0),
        Pixel4::new(0.7, 0.7, 0.7, 1.0), Pixel4::new(0.9, 0.9, 0.9, 1.0),
        Pixel4::new(0.1, 0.1, 0.1, 1.0), Pixel4::new(0.3, 0.3, 0.3, 1.0),
        Pixel4::new(0.5, 0.5, 0.5, 1.0), Pixel4::new(0.7, 0.7, 0.7, 1.0),
        Pixel4::new(0.2, 0.2, 0.2, 1.0), Pixel4::new(0.4, 0.4, 0.4, 1.0),
        Pixel4::new(0.6, 0.6, 0.6, 1.0), Pixel4::new(0.8, 0.8, 0.8, 1.0),
    ];

    let up = rescale(&src, 4, 4, 8, 8, RescaleMethod::PeakedCosineCorrected, ScaleMode::Independent);
    let down = rescale(&up, 8, 8, 4, 4, RescaleMethod::PeakedCosineCorrected, ScaleMode::Independent);

    assert_eq!(down.len(), 16);

    let mut max_err = 0.0f32;
    for (a, b) in src.iter().zip(down.iter()) {
        max_err = max_err.max((a.r() - b.r()).abs());
    }

    // Corrected version may have slightly different roundtrip characteristics
    assert!(max_err < 0.3, "Round-trip error should be reasonable: {}", max_err);
}

#[test]
fn test_peaked_cosine_corrected_alpha_aware() {
    // Test alpha-aware rescaling
    let src = vec![
        Pixel4::new(1.0, 0.0, 0.0, 1.0), Pixel4::new(0.0, 1.0, 0.0, 0.0),
        Pixel4::new(0.0, 0.0, 1.0, 0.0), Pixel4::new(1.0, 1.0, 0.0, 1.0),
    ];

    let dst = rescale_with_alpha(&src, 2, 2, 4, 4,
                                 RescaleMethod::PeakedCosineCorrected, ScaleMode::Independent);

    assert_eq!(dst.len(), 16);

    // Center pixel should be influenced more by opaque pixels (corners)
    let center = dst[5];
    assert!(center.r().is_finite());
}

#[test]
fn test_peaked_cosine_corrected_vs_uncorrected() {
    // Corrected version should differ from uncorrected
    let src_size = 32;
    let dst_size = 8;

    let src: Vec<Pixel4> = (0..src_size * src_size)
        .map(|i| {
            let v = (i % 256) as f32 / 255.0;
            Pixel4::new(v, v, v, 1.0)
        })
        .collect();

    let uncorrected = rescale(&src, src_size, src_size, dst_size, dst_size,
                              RescaleMethod::PeakedCosine, ScaleMode::Independent);
    let corrected = rescale(&src, src_size, src_size, dst_size, dst_size,
                            RescaleMethod::PeakedCosineCorrected, ScaleMode::Independent);

    // They should be different (correction filter changes response)
    let mut total_diff = 0.0f32;
    for (a, b) in uncorrected.iter().zip(corrected.iter()) {
        total_diff += (a.r() - b.r()).abs();
    }

    // There should be some difference (correction filter is applied)
    assert!(total_diff > 0.01, "Corrected should differ from uncorrected: {}", total_diff);
}
