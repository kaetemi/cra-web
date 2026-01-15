//! Image decoding with precise 16-bit and ICC profile support
//!
//! This module provides image decoding that preserves full precision:
//! - 16-bit images are decoded to f32 without 8-bit intermediate
//! - ICC profiles are extracted and can be applied via moxcms
//!
//! This allows the WASM build to bypass browser Canvas API limitations.

use image::{DynamicImage, GenericImageView, ImageDecoder, ImageReader};
use moxcms::{ColorProfile, Layout, ToneReprCurve, TransformOptions};
use std::io::Cursor;

/// Decode image from raw bytes to normalized f32 RGB (0-1 range)
/// Returns: [width as f32, height as f32, has_icc (0.0/1.0), is_16bit (0.0/1.0), ...pixel_data]
/// Pixel data is interleaved RGB f32 in 0-1 range
pub fn decode_image_to_f32(file_bytes: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = Cursor::new(file_bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| format!("Failed to detect image format: {}", e))?;

    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    // Extract ICC profile before decoding
    let has_icc = decoder.icc_profile().ok().flatten().is_some();

    let img = DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("Failed to decode image: {}", e))?;

    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    // Check if 16-bit source
    let is_16bit = matches!(
        img.color(),
        image::ColorType::Rgb16 | image::ColorType::Rgba16 | image::ColorType::L16 | image::ColorType::La16
    );

    // Convert to normalized f32 (0-1 range)
    let pixels: Vec<[f32; 3]> = if is_16bit {
        let rgb16 = img.to_rgb16();
        let data = rgb16.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 3] as f32 / 65535.0,
                data[i * 3 + 1] as f32 / 65535.0,
                data[i * 3 + 2] as f32 / 65535.0,
            ])
            .collect()
    } else {
        let rgb8 = img.to_rgb8();
        let data = rgb8.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 3] as f32 / 255.0,
                data[i * 3 + 1] as f32 / 255.0,
                data[i * 3 + 2] as f32 / 255.0,
            ])
            .collect()
    };

    // Build result: [width, height, has_icc, is_16bit, ...pixels]
    let mut result = Vec::with_capacity(4 + pixel_count * 3);
    result.push(width as f32);
    result.push(height as f32);
    result.push(if has_icc { 1.0 } else { 0.0 });
    result.push(if is_16bit { 1.0 } else { 0.0 });

    for pixel in pixels {
        result.push(pixel[0]);
        result.push(pixel[1]);
        result.push(pixel[2]);
    }

    Ok(result)
}

/// Extract ICC profile from image file bytes
pub fn extract_icc_profile(file_bytes: &[u8]) -> Option<Vec<u8>> {
    let cursor = Cursor::new(file_bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .ok()?;

    let mut decoder = reader.into_decoder().ok()?;
    decoder.icc_profile().ok().flatten()
}

/// Check if ICC profile is effectively sRGB
pub fn is_profile_srgb(icc_bytes: &[u8]) -> bool {
    let src_profile = match ColorProfile::new_from_slice(icc_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let srgb = ColorProfile::new_srgb();

    // Create transform from profile to sRGB
    let transform = match src_profile.create_transform_8bit(
        Layout::Rgb,
        &srgb,
        Layout::Rgb,
        TransformOptions::default(),
    ) {
        Ok(t) => t,
        Err(_) => return false,
    };

    // Test colors to verify identity transform
    let test_colors: &[[u8; 3]] = &[
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [128, 128, 128],
    ];

    for color in test_colors {
        let input: Vec<u8> = color.to_vec();
        let mut output = vec![0u8; 3];

        if transform.transform(&input, &mut output).is_err() {
            return false;
        }

        let max_diff = color
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (*a as i16 - *b as i16).abs())
            .max()
            .unwrap_or(0);

        if max_diff > 1 {
            return false;
        }
    }

    true
}

/// Create linear sRGB profile (sRGB primaries with gamma 1.0)
fn make_linear_srgb_profile() -> ColorProfile {
    let mut profile = ColorProfile::new_srgb();
    let linear_curve = ToneReprCurve::Parametric(vec![1.0]);
    profile.red_trc = Some(linear_curve.clone());
    profile.green_trc = Some(linear_curve.clone());
    profile.blue_trc = Some(linear_curve);
    profile
}

/// Transform image from ICC profile to linear sRGB
/// Input: interleaved RGB f32 (0-1 range)
/// Output: interleaved RGB f32 (0-1 range, linear sRGB)
pub fn transform_icc_to_linear_srgb(
    pixels: &[f32],
    width: usize,
    height: usize,
    icc_bytes: &[u8],
) -> Result<Vec<f32>, String> {
    let src_profile = ColorProfile::new_from_slice(icc_bytes)
        .map_err(|e| format!("Failed to parse ICC profile: {:?}", e))?;

    let linear_srgb = make_linear_srgb_profile();

    let transform = src_profile
        .create_transform_f32(
            Layout::Rgb,
            &linear_srgb,
            Layout::Rgb,
            TransformOptions::default(),
        )
        .map_err(|e| format!("Failed to create transform: {:?}", e))?;

    let pixel_count = width * height;
    let mut output = vec![0.0f32; pixel_count * 3];

    // Transform row by row
    let row_size = width * 3;
    for (src_row, dst_row) in pixels
        .chunks_exact(row_size)
        .zip(output.chunks_exact_mut(row_size))
    {
        transform
            .transform(src_row, dst_row)
            .map_err(|e| format!("Transform failed: {:?}", e))?;
    }

    Ok(output)
}
