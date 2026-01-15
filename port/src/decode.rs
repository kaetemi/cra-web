//! Image decoding with precise 16-bit and ICC profile support
//!
//! This module provides image decoding that preserves full precision:
//! - 16-bit images are decoded to f32 without 8-bit intermediate
//! - ICC profiles are extracted and can be applied via moxcms
//!
//! Used by both WASM and CLI for consistent behavior.

use image::{ColorType, DynamicImage, GenericImageView, ImageDecoder, ImageReader};
use moxcms::{ColorProfile, Layout, ToneReprCurve, TransformOptions};
use std::io::{BufRead, Cursor, Seek};
use std::path::Path;

// ============================================================================
// Image Loading
// ============================================================================

/// Result of decoding an image
pub struct DecodedImage {
    pub image: DynamicImage,
    pub icc_profile: Option<Vec<u8>>,
    pub is_16bit: bool,
}

/// Load image from file path with ICC profile extraction
pub fn load_image_from_path<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    let reader = ImageReader::open(path.as_ref())
        .map_err(|e| format!("Failed to open {}: {}", path.as_ref().display(), e))?
        .with_guessed_format()
        .map_err(|e| format!("Failed to detect format: {}", e))?;

    load_from_reader(reader)
}

/// Load image from byte slice with ICC profile extraction
pub fn load_image_from_bytes(data: &[u8]) -> Result<DecodedImage, String> {
    let cursor = Cursor::new(data);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| format!("Failed to detect format: {}", e))?;

    load_from_reader(reader)
}

/// Internal: load from ImageReader
fn load_from_reader<R: BufRead + Seek>(reader: ImageReader<R>) -> Result<DecodedImage, String> {
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    // Extract ICC profile before decoding (supported for PNG, JPEG, AVIF)
    let icc_profile = decoder.icc_profile().ok().flatten();

    let image = DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("Failed to decode image: {}", e))?;

    let is_16bit = matches!(
        image.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    Ok(DecodedImage {
        image,
        icc_profile,
        is_16bit,
    })
}

// ============================================================================
// Pixel Conversion
// ============================================================================

/// Convert DynamicImage to normalized f32 RGB (0-1 range)
/// Handles both 8-bit and 16-bit sources with appropriate precision
pub fn image_to_f32_normalized(img: &DynamicImage) -> Vec<[f32; 3]> {
    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    let is_16bit = matches!(
        img.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    if is_16bit {
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
    }
}

/// Convert DynamicImage to interleaved f32 RGB (0-1 range)
pub fn image_to_f32_interleaved(img: &DynamicImage) -> Vec<f32> {
    image_to_f32_normalized(img)
        .into_iter()
        .flat_map(|[r, g, b]| [r, g, b])
        .collect()
}

/// Convert DynamicImage directly to f32 sRGB 0-255 scale (no 0-1 intermediate)
/// Returns array of [r, g, b] for CLI compatibility.
/// - 8-bit: u8 as f32 (direct cast)
/// - 16-bit: u16 * 255.0 / 65535.0 directly to 0-255
pub fn image_to_f32_srgb_255_pixels(img: &DynamicImage) -> Vec<[f32; 3]> {
    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    let is_16bit = matches!(
        img.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    if is_16bit {
        // 16-bit: scale from 0-65535 to 0-255
        let rgb16 = img.to_rgb16();
        let data = rgb16.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 3] as f32 * 255.0 / 65535.0,
                data[i * 3 + 1] as f32 * 255.0 / 65535.0,
                data[i * 3 + 2] as f32 * 255.0 / 65535.0,
            ])
            .collect()
    } else {
        // 8-bit: u8 directly as f32 (no arithmetic needed)
        let rgb8 = img.to_rgb8();
        let data = rgb8.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 3] as f32,
                data[i * 3 + 1] as f32,
                data[i * 3 + 2] as f32,
            ])
            .collect()
    }
}

/// Convert DynamicImage directly to interleaved f32 sRGB 0-255 scale (for WASM)
/// - 8-bit: u8 as f32 (direct cast)
/// - 16-bit: u16 * 255.0 / 65535.0 directly to 0-255
pub fn image_to_f32_srgb_255(img: &DynamicImage) -> Vec<f32> {
    image_to_f32_srgb_255_pixels(img)
        .into_iter()
        .flat_map(|[r, g, b]| [r, g, b])
        .collect()
}

// ============================================================================
// ICC Profile Handling
// ============================================================================

/// Create a linear sRGB profile (sRGB primaries with gamma 1.0)
pub fn make_linear_srgb_profile() -> ColorProfile {
    let mut profile = ColorProfile::new_srgb();
    let linear_curve = ToneReprCurve::Parametric(vec![1.0]);
    profile.red_trc = Some(linear_curve.clone());
    profile.green_trc = Some(linear_curve.clone());
    profile.blue_trc = Some(linear_curve);
    profile
}

/// Check if ICC profile is effectively sRGB by comparing test colors
pub fn is_profile_srgb(icc_bytes: &[u8]) -> bool {
    is_profile_srgb_impl(icc_bytes, false)
}

/// Check if ICC profile is effectively sRGB (with optional verbose output)
pub fn is_profile_srgb_verbose(icc_bytes: &[u8], verbose: bool) -> bool {
    is_profile_srgb_impl(icc_bytes, verbose)
}

fn is_profile_srgb_impl(icc_bytes: &[u8], verbose: bool) -> bool {
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
        [192, 64, 32],
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
            if verbose {
                eprintln!(
                    "  Profile differs from sRGB: {:?} -> {:?} (diff={})",
                    color, output, max_diff
                );
            }
            return false;
        }
    }

    true
}

/// Transform image from ICC profile to linear sRGB (interleaved f32)
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

/// Transform image from ICC profile to linear sRGB (array of [f32; 3])
/// Input: array of [r, g, b] f32 (0-1 range)
/// Output: array of [r, g, b] f32 (0-1 range, linear sRGB)
pub fn transform_icc_to_linear_srgb_pixels(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    icc_bytes: &[u8],
) -> Result<Vec<[f32; 3]>, String> {
    // Flatten to interleaved
    let interleaved: Vec<f32> = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();

    let result = transform_icc_to_linear_srgb(&interleaved, width, height, icc_bytes)?;

    // Convert back to array format
    Ok(result
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}

// ============================================================================
// WASM-specific functions
// ============================================================================

/// Decode image from raw bytes for WASM
/// Returns: [width as f32, height as f32, has_icc (0.0/1.0), is_16bit (0.0/1.0), ...pixel_data]
/// Pixel data is interleaved RGB f32 in 0-1 range
pub fn decode_image_to_f32(file_bytes: &[u8]) -> Result<Vec<f32>, String> {
    let decoded = load_image_from_bytes(file_bytes)?;
    let (width, height) = decoded.image.dimensions();

    let pixels = image_to_f32_interleaved(&decoded.image);

    // Build result: [width, height, has_icc, is_16bit, ...pixels]
    let pixel_count = (width * height) as usize;
    let mut result = Vec::with_capacity(4 + pixel_count * 3);
    result.push(width as f32);
    result.push(height as f32);
    result.push(if decoded.icc_profile.is_some() { 1.0 } else { 0.0 });
    result.push(if decoded.is_16bit { 1.0 } else { 0.0 });
    result.extend(pixels);

    Ok(result)
}

/// Decode image from raw bytes directly to sRGB f32 0-255 scale
/// This avoids the 0-1 intermediate when no color processing is needed.
/// Returns: [width as f32, height as f32, has_icc (0.0/1.0), is_16bit (0.0/1.0), ...pixel_data]
/// Pixel data is interleaved RGB f32 in 0-255 range
pub fn decode_image_to_srgb_255(file_bytes: &[u8]) -> Result<Vec<f32>, String> {
    let decoded = load_image_from_bytes(file_bytes)?;
    let (width, height) = decoded.image.dimensions();
    let pixel_count = (width * height) as usize;

    // Convert directly to f32 0-255 without 0-1 intermediate
    let pixels = image_to_f32_srgb_255(&decoded.image);

    // Build result: [width, height, has_icc, is_16bit, ...pixels]
    let mut result = Vec::with_capacity(4 + pixel_count * 3);
    result.push(width as f32);
    result.push(height as f32);
    result.push(if decoded.icc_profile.is_some() { 1.0 } else { 0.0 });
    result.push(if decoded.is_16bit { 1.0 } else { 0.0 });
    result.extend(pixels);

    Ok(result)
}

/// Extract ICC profile from image file bytes
pub fn extract_icc_profile(file_bytes: &[u8]) -> Option<Vec<u8>> {
    load_image_from_bytes(file_bytes).ok()?.icc_profile
}
