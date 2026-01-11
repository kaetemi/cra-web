/// Error diffusion dithering implementations.
/// Supports Floyd-Steinberg and Jarvis-Judice-Ninke algorithms.

/// Dithering mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitherMode {
    /// Floyd-Steinberg: Standard left-to-right scanning on all rows (default)
    #[default]
    Standard,
    /// Floyd-Steinberg: Serpentine scanning (alternating direction each row)
    /// Reduces diagonal banding artifacts
    Serpentine,
    /// Jarvis-Judice-Ninke: Standard left-to-right scanning
    /// Larger kernel (3 rows) produces smoother results but slower
    JarvisStandard,
    /// Jarvis-Judice-Ninke: Serpentine scanning
    /// Combines larger kernel with alternating scan direction
    JarvisSerpentine,
}

/// Floyd-Steinberg dithering with linear buffer and overflow padding.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///
/// Returns:
///     flat array of u8 values, same length as input
pub fn floyd_steinberg_dither(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    floyd_steinberg_dither_with_mode(img, width, height, DitherMode::Standard)
}

/// Dithering with selectable algorithm and scanning mode.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     mode: DitherMode variant selecting algorithm and scan pattern
///
/// Returns:
///     flat array of u8 values, same length as input
pub fn floyd_steinberg_dither_with_mode(
    img: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
) -> Vec<u8> {
    match mode {
        DitherMode::Standard => floyd_steinberg_standard(img, width, height),
        DitherMode::Serpentine => floyd_steinberg_serpentine(img, width, height),
        DitherMode::JarvisStandard => jarvis_judice_ninke_standard(img, width, height),
        DitherMode::JarvisSerpentine => jarvis_judice_ninke_serpentine(img, width, height),
    }
}

/// Standard Floyd-Steinberg dithering (left-to-right on all rows)
fn floyd_steinberg_standard(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    let len = width * height;

    // Allocate buffer with overflow padding
    let mut buf = vec![0.0f32; len + width + 2];
    buf[..len].copy_from_slice(img);

    for i in 0..len {
        let old = buf[i];
        let new = old.round();
        buf[i] = new;
        let err = old - new;

        // Distribute error to neighbors
        // Overflow writes hit padding, which we discard
        buf[i + 1] += err * (7.0 / 16.0);
        buf[i + width - 1] += err * (3.0 / 16.0);
        buf[i + width] += err * (5.0 / 16.0);
        buf[i + width + 1] += err * (1.0 / 16.0);
    }

    // Clamp and convert to u8, discard padding
    buf[..len]
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect()
}

/// Serpentine Floyd-Steinberg dithering (alternating direction each row)
/// This reduces diagonal banding artifacts by reversing scan direction on odd rows
fn floyd_steinberg_serpentine(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    let len = width * height;

    // Use 2D buffer for easier coordinate handling
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();

    // Add a padding row at the bottom
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        let is_reverse = y % 2 == 1;

        if is_reverse {
            // Right-to-left scanning
            for x in (0..width).rev() {
                let old = buf[y][x];
                let new = old.round();
                buf[y][x] = new;
                let err = old - new;

                // Distribute error to neighbors (mirrored for reverse direction)
                // Left neighbor (was right)
                if x > 0 {
                    buf[y][x - 1] += err * (7.0 / 16.0);
                }
                // Bottom-right (was bottom-left)
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (3.0 / 16.0);
                }
                // Bottom
                buf[y + 1][x] += err * (5.0 / 16.0);
                // Bottom-left (was bottom-right)
                if x > 0 {
                    buf[y + 1][x - 1] += err * (1.0 / 16.0);
                }
            }
        } else {
            // Left-to-right scanning (standard)
            for x in 0..width {
                let old = buf[y][x];
                let new = old.round();
                buf[y][x] = new;
                let err = old - new;

                // Distribute error to neighbors
                // Right neighbor
                if x + 1 < width {
                    buf[y][x + 1] += err * (7.0 / 16.0);
                }
                // Bottom-left
                if x > 0 {
                    buf[y + 1][x - 1] += err * (3.0 / 16.0);
                }
                // Bottom
                buf[y + 1][x] += err * (5.0 / 16.0);
                // Bottom-right
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (1.0 / 16.0);
                }
            }
        }
    }

    // Flatten, clamp, and convert to u8
    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0) as u8))
        .collect()
}

/// Jarvis-Judice-Ninke dithering (standard left-to-right scanning)
/// Uses a larger 3-row kernel for smoother gradients than Floyd-Steinberg
///
/// Kernel (divided by 48):
///         *   7   5
///     3   5   7   5   3
///     1   3   5   3   1
fn jarvis_judice_ninke_standard(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    // Use 2D buffer for easier coordinate handling with 3-row kernel
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();

    // Add 2 padding rows at the bottom for the 3-row kernel
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        for x in 0..width {
            let old = buf[y][x];
            let new = old.round();
            buf[y][x] = new;
            let err = old - new;

            // Distribute error using JJN kernel (divided by 48)
            // Row 0 (current row): * 7 5
            if x + 1 < width {
                buf[y][x + 1] += err * (7.0 / 48.0);
            }
            if x + 2 < width {
                buf[y][x + 2] += err * (5.0 / 48.0);
            }

            // Row 1: 3 5 7 5 3
            if x >= 2 {
                buf[y + 1][x - 2] += err * (3.0 / 48.0);
            }
            if x >= 1 {
                buf[y + 1][x - 1] += err * (5.0 / 48.0);
            }
            buf[y + 1][x] += err * (7.0 / 48.0);
            if x + 1 < width {
                buf[y + 1][x + 1] += err * (5.0 / 48.0);
            }
            if x + 2 < width {
                buf[y + 1][x + 2] += err * (3.0 / 48.0);
            }

            // Row 2: 1 3 5 3 1
            if x >= 2 {
                buf[y + 2][x - 2] += err * (1.0 / 48.0);
            }
            if x >= 1 {
                buf[y + 2][x - 1] += err * (3.0 / 48.0);
            }
            buf[y + 2][x] += err * (5.0 / 48.0);
            if x + 1 < width {
                buf[y + 2][x + 1] += err * (3.0 / 48.0);
            }
            if x + 2 < width {
                buf[y + 2][x + 2] += err * (1.0 / 48.0);
            }
        }
    }

    // Flatten, clamp, and convert to u8
    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0) as u8))
        .collect()
}

/// Jarvis-Judice-Ninke dithering with serpentine scanning
/// Alternates scan direction each row to reduce diagonal artifacts
fn jarvis_judice_ninke_serpentine(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    // Use 2D buffer for easier coordinate handling
    let mut buf: Vec<Vec<f32>> = (0..height)
        .map(|y| {
            let start = y * width;
            img[start..start + width].to_vec()
        })
        .collect();

    // Add 2 padding rows at the bottom for the 3-row kernel
    buf.push(vec![0.0f32; width]);
    buf.push(vec![0.0f32; width]);

    for y in 0..height {
        let is_reverse = y % 2 == 1;

        if is_reverse {
            // Right-to-left scanning
            for x in (0..width).rev() {
                let old = buf[y][x];
                let new = old.round();
                buf[y][x] = new;
                let err = old - new;

                // Distribute error using mirrored JJN kernel
                // Row 0 (current row): 5 7 *
                if x >= 1 {
                    buf[y][x - 1] += err * (7.0 / 48.0);
                }
                if x >= 2 {
                    buf[y][x - 2] += err * (5.0 / 48.0);
                }

                // Row 1: 3 5 7 5 3 (mirrored)
                if x + 2 < width {
                    buf[y + 1][x + 2] += err * (3.0 / 48.0);
                }
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (5.0 / 48.0);
                }
                buf[y + 1][x] += err * (7.0 / 48.0);
                if x >= 1 {
                    buf[y + 1][x - 1] += err * (5.0 / 48.0);
                }
                if x >= 2 {
                    buf[y + 1][x - 2] += err * (3.0 / 48.0);
                }

                // Row 2: 1 3 5 3 1 (mirrored)
                if x + 2 < width {
                    buf[y + 2][x + 2] += err * (1.0 / 48.0);
                }
                if x + 1 < width {
                    buf[y + 2][x + 1] += err * (3.0 / 48.0);
                }
                buf[y + 2][x] += err * (5.0 / 48.0);
                if x >= 1 {
                    buf[y + 2][x - 1] += err * (3.0 / 48.0);
                }
                if x >= 2 {
                    buf[y + 2][x - 2] += err * (1.0 / 48.0);
                }
            }
        } else {
            // Left-to-right scanning (same as standard)
            for x in 0..width {
                let old = buf[y][x];
                let new = old.round();
                buf[y][x] = new;
                let err = old - new;

                // Row 0: * 7 5
                if x + 1 < width {
                    buf[y][x + 1] += err * (7.0 / 48.0);
                }
                if x + 2 < width {
                    buf[y][x + 2] += err * (5.0 / 48.0);
                }

                // Row 1: 3 5 7 5 3
                if x >= 2 {
                    buf[y + 1][x - 2] += err * (3.0 / 48.0);
                }
                if x >= 1 {
                    buf[y + 1][x - 1] += err * (5.0 / 48.0);
                }
                buf[y + 1][x] += err * (7.0 / 48.0);
                if x + 1 < width {
                    buf[y + 1][x + 1] += err * (5.0 / 48.0);
                }
                if x + 2 < width {
                    buf[y + 1][x + 2] += err * (3.0 / 48.0);
                }

                // Row 2: 1 3 5 3 1
                if x >= 2 {
                    buf[y + 2][x - 2] += err * (1.0 / 48.0);
                }
                if x >= 1 {
                    buf[y + 2][x - 1] += err * (3.0 / 48.0);
                }
                buf[y + 2][x] += err * (5.0 / 48.0);
                if x + 1 < width {
                    buf[y + 2][x + 1] += err * (3.0 / 48.0);
                }
                if x + 2 < width {
                    buf[y + 2][x + 2] += err * (1.0 / 48.0);
                }
            }
        }
    }

    // Flatten, clamp, and convert to u8
    buf[..height]
        .iter()
        .flat_map(|row| row.iter().map(|&v| v.clamp(0.0, 255.0) as u8))
        .collect()
}

/// Dither multiple channels and interleave them
/// channels: Vec of channel data, each scaled to 0-255
/// Returns interleaved u8 data
#[allow(dead_code)]
pub fn dither_channel_stack(channels: &[Vec<f32>], width: usize, height: usize) -> Vec<u8> {
    let num_channels = channels.len();
    let pixels = width * height;
    let mut result = vec![0u8; pixels * num_channels];

    // Dither each channel
    let dithered: Vec<Vec<u8>> = channels
        .iter()
        .map(|ch| floyd_steinberg_dither(ch, width, height))
        .collect();

    // Interleave
    for i in 0..pixels {
        for (ch, dithered_ch) in dithered.iter().enumerate() {
            result[i * num_channels + ch] = dithered_ch[i];
        }
    }

    result
}

/// Dither RGB float image (0-1 range) to uint8
#[allow(dead_code)]
pub fn dither_rgb(rgb: &[f32], width: usize, height: usize) -> Vec<u8> {
    let pixels = width * height;

    // Extract and scale channels
    let channels: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            (0..pixels)
                .map(|i| (rgb[i * 3 + ch] * 255.0).clamp(0.0, 255.0))
                .collect()
        })
        .collect();

    dither_channel_stack(&channels, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dither_exact_values() {
        // Values that are already integers should stay the same
        let img = vec![0.0, 128.0, 255.0, 64.0];
        let result = floyd_steinberg_dither(&img, 2, 2);
        // Note: due to error diffusion, exact matches aren't guaranteed
        // but for a simple case like this, corners should be close
        assert!(result[0] == 0 || result[0] == 1);
        assert!(result[2] == 254 || result[2] == 255);
    }

    #[test]
    fn test_dither_clamping() {
        // Values outside 0-255 should be clamped
        let img = vec![-10.0, 300.0];
        let result = floyd_steinberg_dither(&img, 2, 1);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 255);
    }

    #[test]
    fn test_serpentine_dither() {
        // Test that serpentine mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = floyd_steinberg_dither_with_mode(&img, 4, 4, DitherMode::Serpentine);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v <= 255);
        }
    }

    #[test]
    fn test_serpentine_vs_standard() {
        // Serpentine and standard should produce different results on larger images
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let standard = floyd_steinberg_dither_with_mode(&img, 10, 10, DitherMode::Standard);
        let serpentine = floyd_steinberg_dither_with_mode(&img, 10, 10, DitherMode::Serpentine);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        // They should produce different results (not identical)
        assert_ne!(standard, serpentine);
    }

    #[test]
    fn test_jarvis_standard() {
        // Test that JJN standard mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = floyd_steinberg_dither_with_mode(&img, 4, 4, DitherMode::JarvisStandard);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_jarvis_serpentine() {
        // Test that JJN serpentine mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = floyd_steinberg_dither_with_mode(&img, 4, 4, DitherMode::JarvisSerpentine);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_jarvis_vs_floyd_steinberg() {
        // JJN and Floyd-Steinberg should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let fs_standard = floyd_steinberg_dither_with_mode(&img, 10, 10, DitherMode::Standard);
        let jjn_standard = floyd_steinberg_dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard);
        assert_eq!(fs_standard.len(), 100);
        assert_eq!(jjn_standard.len(), 100);
        // They should produce different results
        assert_ne!(fs_standard, jjn_standard);
    }

    #[test]
    fn test_jarvis_serpentine_vs_standard() {
        // JJN serpentine and standard should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let standard = floyd_steinberg_dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard);
        let serpentine = floyd_steinberg_dither_with_mode(&img, 10, 10, DitherMode::JarvisSerpentine);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        // They should produce different results
        assert_ne!(standard, serpentine);
    }
}
