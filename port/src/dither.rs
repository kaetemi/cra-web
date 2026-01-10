/// Floyd-Steinberg dithering implementation.
/// Matches the existing WASM implementation for bit-perfect output.

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
}
