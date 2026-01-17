//! Image decoding with precise 16-bit and ICC/CICP profile support
//!
//! This module provides image decoding that preserves full precision:
//! - 16-bit images are decoded to f32 without 8-bit intermediate
//! - ICC profiles are extracted and can be applied via moxcms
//! - CICP (Coding-Independent Code Points) for authoritative color space detection
//!
//! Used by both WASM and CLI for consistent behavior.

use image::{ColorType, DynamicImage, GenericImageView, ImageDecoder, ImageFormat, ImageReader};
use image::metadata::{Cicp, CicpColorPrimaries, CicpTransferCharacteristics, CicpVideoFullRangeFlag};
use moxcms::{CicpProfile as MoxcmsCicpProfile, CicpColorPrimaries as MoxcmsPrimaries, ColorProfile, Layout, LocalizableString, MatrixCoefficients, ProfileText, ToneReprCurve, TransferCharacteristics, TransformOptions};
use std::io::{BufRead, Cursor, Seek};
use std::path::Path;

use crate::sfi;

// ============================================================================
// Image Loading
// ============================================================================

/// Result of decoding an image
pub struct DecodedImage {
    pub image: DynamicImage,
    pub icc_profile: Option<Vec<u8>>,
    pub cicp: Cicp,
    pub format: Option<ImageFormat>,
    pub is_16bit: bool,
    pub is_f32: bool,
    pub has_alpha: bool,
    /// Whether this image has premultiplied alpha.
    /// For safetensors, this comes from metadata. For other formats, it's the format default.
    pub is_premultiplied_alpha: bool,
}

impl DecodedImage {
    /// Check if this image has premultiplied alpha that needs un-premultiplying.
    /// For safetensors, this comes from the alpha_premultiplied metadata.
    /// For other formats, it's the format default (e.g., EXR uses premultiplied by default).
    pub fn is_format_premultiplied_default(&self) -> bool {
        self.is_premultiplied_alpha
    }
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
    // Capture the format before consuming the reader
    let format = reader.format();

    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    // Extract ICC profile before decoding (supported for PNG, JPEG, AVIF)
    let icc_profile = decoder.icc_profile().ok().flatten();

    let image = DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("Failed to decode image: {}", e))?;

    // Extract CICP from the decoded image
    // Note: Currently defaults to sRGB as decoders don't populate this yet,
    // but when image crate adds decoder CICP extraction, this will automatically work
    let cicp = image.color_space();

    let is_16bit = matches!(
        image.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    let is_f32 = matches!(image.color(), ColorType::Rgb32F | ColorType::Rgba32F);

    let has_alpha = matches!(
        image.color(),
        ColorType::La8 | ColorType::Rgba8 | ColorType::La16 | ColorType::Rgba16 | ColorType::Rgba32F
    );

    // EXR uses premultiplied alpha by default
    let is_premultiplied_alpha = matches!(format, Some(ImageFormat::OpenExr));

    Ok(DecodedImage {
        image,
        icc_profile,
        cicp,
        format,
        is_16bit,
        is_f32,
        has_alpha,
        is_premultiplied_alpha,
    })
}

// ============================================================================
// Pixel Conversion
// ============================================================================

/// Convert DynamicImage to normalized f32 with variable channel count
/// channels: 3 for RGB (alpha discarded), 4 for RGBA
/// Handles 8-bit, 16-bit, and f32 sources with appropriate precision
pub fn image_to_f32_normalized_channels(img: &DynamicImage, channels: usize) -> Vec<f32> {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    if channels == 4 {
        image_to_f32_normalized_rgba(img)
            .into_iter()
            .flat_map(|[r, g, b, a]| [r, g, b, a])
            .collect()
    } else {
        image_to_f32_normalized(img)
            .into_iter()
            .flat_map(|[r, g, b]| [r, g, b])
            .collect()
    }
}

/// Convert DynamicImage to normalized f32 RGB (0-1 range)
/// Handles 8-bit, 16-bit, and f32 sources with appropriate precision
pub fn image_to_f32_normalized(img: &DynamicImage) -> Vec<[f32; 3]> {
    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    let is_16bit = matches!(
        img.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    let is_f32 = matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F);

    if is_f32 {
        // f32 images: already in float format, typically 0-1 range (may be HDR with values > 1.0)
        let rgb32f = img.to_rgb32f();
        let data = rgb32f.as_raw();
        (0..pixel_count)
            .map(|i| [data[i * 3], data[i * 3 + 1], data[i * 3 + 2]])
            .collect()
    } else if is_16bit {
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

/// Convert DynamicImage to normalized f32 RGBA (0-1 range)
/// Handles 8-bit, 16-bit, and f32 sources with appropriate precision
/// Alpha channel is preserved (linear, no gamma conversion needed)
pub fn image_to_f32_normalized_rgba(img: &DynamicImage) -> Vec<[f32; 4]> {
    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    let is_16bit = matches!(
        img.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    let is_f32 = matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F);

    if is_f32 {
        // f32 images: already in float format
        let rgba32f = img.to_rgba32f();
        let data = rgba32f.as_raw();
        (0..pixel_count)
            .map(|i| [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]])
            .collect()
    } else if is_16bit {
        let rgba16 = img.to_rgba16();
        let data = rgba16.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 4] as f32 / 65535.0,
                data[i * 4 + 1] as f32 / 65535.0,
                data[i * 4 + 2] as f32 / 65535.0,
                data[i * 4 + 3] as f32 / 65535.0,
            ])
            .collect()
    } else {
        let rgba8 = img.to_rgba8();
        let data = rgba8.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 4] as f32 / 255.0,
                data[i * 4 + 1] as f32 / 255.0,
                data[i * 4 + 2] as f32 / 255.0,
                data[i * 4 + 3] as f32 / 255.0,
            ])
            .collect()
    }
}

/// Convert DynamicImage to interleaved f32 RGBA (0-1 range)
pub fn image_to_f32_interleaved_rgba(img: &DynamicImage) -> Vec<f32> {
    image_to_f32_normalized_rgba(img)
        .into_iter()
        .flat_map(|[r, g, b, a]| [r, g, b, a])
        .collect()
}

/// Convert DynamicImage to sRGB f32 0-255 scale with variable channel count
/// channels: 3 for RGB, 4 for RGBA (alpha also in 0-255 range)
/// - 8-bit: u8 as f32 (direct cast)
/// - 16-bit: u16 * 255.0 / 65535.0 directly to 0-255
/// - f32: assumed 0-1 range, scaled to 0-255 (clamped for HDR)
pub fn image_to_f32_srgb_255_channels(img: &DynamicImage, channels: usize) -> Vec<f32> {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    if channels == 4 {
        image_to_f32_srgb_255_pixels_rgba(img)
            .into_iter()
            .flat_map(|[r, g, b, a]| [r, g, b, a])
            .collect()
    } else {
        image_to_f32_srgb_255_pixels(img)
            .into_iter()
            .flat_map(|[r, g, b]| [r, g, b])
            .collect()
    }
}

/// Convert DynamicImage directly to f32 sRGB 0-255 scale (no 0-1 intermediate)
/// Returns array of [r, g, b] for CLI compatibility.
/// - 8-bit: u8 as f32 (direct cast)
/// - 16-bit: u16 * 255.0 / 65535.0 directly to 0-255
/// - f32: assumed 0-1 range, scaled to 0-255 (clamped for HDR)
pub fn image_to_f32_srgb_255_pixels(img: &DynamicImage) -> Vec<[f32; 3]> {
    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    let is_16bit = matches!(
        img.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    let is_f32 = matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F);

    if is_f32 {
        // f32: scale from 0-1 to 0-255, clamp for HDR (this path is for direct output)
        let rgb32f = img.to_rgb32f();
        let data = rgb32f.as_raw();
        (0..pixel_count)
            .map(|i| [
                (data[i * 3] * 255.0).clamp(0.0, 255.0),
                (data[i * 3 + 1] * 255.0).clamp(0.0, 255.0),
                (data[i * 3 + 2] * 255.0).clamp(0.0, 255.0),
            ])
            .collect()
    } else if is_16bit {
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

/// Convert DynamicImage directly to f32 sRGBA 0-255 scale (no 0-1 intermediate)
/// Returns array of [r, g, b, a] with alpha also in 0-255 range.
/// - 8-bit: u8 as f32 (direct cast)
/// - 16-bit: u16 * 255.0 / 65535.0 directly to 0-255
/// - f32: assumed 0-1 range, scaled to 0-255 (RGB clamped for HDR, alpha clamped)
pub fn image_to_f32_srgb_255_pixels_rgba(img: &DynamicImage) -> Vec<[f32; 4]> {
    let (width, height) = img.dimensions();
    let pixel_count = (width * height) as usize;

    let is_16bit = matches!(
        img.color(),
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    let is_f32 = matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F);

    if is_f32 {
        // f32: scale from 0-1 to 0-255, clamp for HDR (this path is for direct output)
        let rgba32f = img.to_rgba32f();
        let data = rgba32f.as_raw();
        (0..pixel_count)
            .map(|i| [
                (data[i * 4] * 255.0).clamp(0.0, 255.0),
                (data[i * 4 + 1] * 255.0).clamp(0.0, 255.0),
                (data[i * 4 + 2] * 255.0).clamp(0.0, 255.0),
                (data[i * 4 + 3] * 255.0).clamp(0.0, 255.0),
            ])
            .collect()
    } else if is_16bit {
        // 16-bit: scale from 0-65535 to 0-255
        let rgba16 = img.to_rgba16();
        let data = rgba16.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 4] as f32 * 255.0 / 65535.0,
                data[i * 4 + 1] as f32 * 255.0 / 65535.0,
                data[i * 4 + 2] as f32 * 255.0 / 65535.0,
                data[i * 4 + 3] as f32 * 255.0 / 65535.0,
            ])
            .collect()
    } else {
        // 8-bit: u8 directly as f32 (no arithmetic needed)
        let rgba8 = img.to_rgba8();
        let data = rgba8.as_raw();
        (0..pixel_count)
            .map(|i| [
                data[i * 4] as f32,
                data[i * 4 + 1] as f32,
                data[i * 4 + 2] as f32,
                data[i * 4 + 3] as f32,
            ])
            .collect()
    }
}

/// Convert DynamicImage directly to interleaved f32 sRGBA 0-255 scale (for WASM)
pub fn image_to_f32_srgb_255_rgba(img: &DynamicImage) -> Vec<f32> {
    image_to_f32_srgb_255_pixels_rgba(img)
        .into_iter()
        .flat_map(|[r, g, b, a]| [r, g, b, a])
        .collect()
}

// ============================================================================
// CICP (Coding-Independent Code Points) Handling
// ============================================================================

/// Check if CICP indicates standard sRGB color space.
/// sRGB = BT.709/sRGB primaries + sRGB transfer function
/// Note: We only check primaries and transfer - decoded RGB images always have
/// Identity matrix and FullRange (the decoder handles YCbCr→RGB conversion).
pub fn is_cicp_srgb(cicp: &Cicp) -> bool {
    cicp.primaries == CicpColorPrimaries::SRgb
        && cicp.transfer == CicpTransferCharacteristics::SRgb
}

/// Check if CICP indicates linear sRGB color space.
/// Linear sRGB = BT.709/sRGB primaries + Linear transfer
pub fn is_cicp_linear_srgb(cicp: &Cicp) -> bool {
    cicp.primaries == CicpColorPrimaries::SRgb
        && cicp.transfer == CicpTransferCharacteristics::Linear
}

/// Check if CICP is unspecified (cannot determine color space from CICP alone).
/// This means we should fall back to ICC profile or assume sRGB.
pub fn is_cicp_unspecified(cicp: &Cicp) -> bool {
    cicp.primaries == CicpColorPrimaries::Unspecified
        || cicp.transfer == CicpTransferCharacteristics::Unspecified
}

/// Check if CICP indicates a non-sRGB color space that requires conversion.
/// Returns true if CICP is specified (not unspecified) and not sRGB/linear-sRGB.
/// Examples: Display P3, BT.2020, Adobe RGB, etc.
pub fn is_cicp_needs_conversion(cicp: &Cicp) -> bool {
    !is_cicp_unspecified(cicp) && !is_cicp_srgb(cicp) && !is_cicp_linear_srgb(cicp)
}

/// Get a human-readable description of the CICP color space.
pub fn cicp_description(cicp: &Cicp) -> String {
    if is_cicp_srgb(cicp) {
        "sRGB".to_string()
    } else if is_cicp_linear_srgb(cicp) {
        "Linear sRGB".to_string()
    } else if is_cicp_unspecified(cicp) {
        "Unspecified".to_string()
    } else {
        format!(
            "CICP(primaries={:?}, transfer={:?}, matrix={:?}, range={:?})",
            cicp.primaries, cicp.transfer, cicp.matrix, cicp.full_range
        )
    }
}

// ============================================================================
// CICP to moxcms Profile Conversion
// ============================================================================

/// Convert image crate CicpColorPrimaries to moxcms CicpColorPrimaries
fn map_cicp_primaries(primaries: CicpColorPrimaries) -> MoxcmsPrimaries {
    match primaries {
        CicpColorPrimaries::SRgb => MoxcmsPrimaries::Bt709,
        CicpColorPrimaries::Unspecified => MoxcmsPrimaries::Unspecified,
        CicpColorPrimaries::RgbM => MoxcmsPrimaries::Bt470M,
        CicpColorPrimaries::RgbB => MoxcmsPrimaries::Bt470Bg,
        CicpColorPrimaries::Bt601 => MoxcmsPrimaries::Bt601,
        CicpColorPrimaries::Rgb240m => MoxcmsPrimaries::Smpte240,
        CicpColorPrimaries::GenericFilm => MoxcmsPrimaries::GenericFilm,
        CicpColorPrimaries::Rgb2020 => MoxcmsPrimaries::Bt2020,
        CicpColorPrimaries::Xyz => MoxcmsPrimaries::Xyz,
        CicpColorPrimaries::SmpteRp431 => MoxcmsPrimaries::Smpte431,
        CicpColorPrimaries::SmpteRp432 => MoxcmsPrimaries::Smpte432,
        CicpColorPrimaries::Industry22 => MoxcmsPrimaries::Ebu3213,
        _ => MoxcmsPrimaries::Unspecified, // Unknown/future values
    }
}

/// Convert image crate CicpTransferCharacteristics to moxcms TransferCharacteristics
fn map_cicp_transfer(transfer: CicpTransferCharacteristics) -> TransferCharacteristics {
    match transfer {
        CicpTransferCharacteristics::Bt709 => TransferCharacteristics::Bt709,
        CicpTransferCharacteristics::Unspecified => TransferCharacteristics::Unspecified,
        CicpTransferCharacteristics::Bt470M => TransferCharacteristics::Bt470M,
        CicpTransferCharacteristics::Bt470BG => TransferCharacteristics::Bt470Bg,
        CicpTransferCharacteristics::Bt601 => TransferCharacteristics::Bt601,
        CicpTransferCharacteristics::Smpte240m => TransferCharacteristics::Smpte240,
        CicpTransferCharacteristics::Linear => TransferCharacteristics::Linear,
        CicpTransferCharacteristics::Log100 => TransferCharacteristics::Log100,
        CicpTransferCharacteristics::LogSqrt => TransferCharacteristics::Log100sqrt10,
        CicpTransferCharacteristics::Iec61966_2_4 => TransferCharacteristics::Iec61966,
        CicpTransferCharacteristics::Bt1361 => TransferCharacteristics::Bt1361,
        CicpTransferCharacteristics::SRgb => TransferCharacteristics::Srgb,
        CicpTransferCharacteristics::Bt2020_10bit => TransferCharacteristics::Bt202010bit,
        CicpTransferCharacteristics::Bt2020_12bit => TransferCharacteristics::Bt202012bit,
        CicpTransferCharacteristics::Smpte2084 => TransferCharacteristics::Smpte2084,
        CicpTransferCharacteristics::Smpte428 => TransferCharacteristics::Smpte428,
        CicpTransferCharacteristics::Bt2100Hlg => TransferCharacteristics::Hlg,
        _ => TransferCharacteristics::Unspecified, // Unknown/future values
    }
}

/// Create a moxcms ColorProfile from image crate CICP metadata.
/// Returns None if CICP is unspecified.
pub fn cicp_to_color_profile(cicp: &Cicp) -> Option<ColorProfile> {
    // Check for unspecified values first
    if cicp.primaries == CicpColorPrimaries::Unspecified
        || cicp.transfer == CicpTransferCharacteristics::Unspecified
    {
        return None;
    }

    // Map to moxcms types
    let cicp_profile = MoxcmsCicpProfile {
        color_primaries: map_cicp_primaries(cicp.primaries),
        transfer_characteristics: map_cicp_transfer(cicp.transfer),
        // For decoded RGB, matrix is always Identity (decoder handles YCbCr→RGB)
        matrix_coefficients: MatrixCoefficients::Identity,
        // Decoded RGB is always full range
        full_range: true,
    };

    Some(ColorProfile::new_from_cicp(cicp_profile))
}

/// Check if we can create a valid moxcms profile from CICP.
/// Returns true if CICP has specified primaries and transfer characteristics.
pub fn can_use_cicp(cicp: &Cicp) -> bool {
    cicp.primaries != CicpColorPrimaries::Unspecified
        && cicp.transfer != CicpTransferCharacteristics::Unspecified
}

/// Transform image from CICP color space to linear sRGB (interleaved f32)
/// Input: interleaved RGB f32 (0-1 range)
/// Output: interleaved RGB f32 (linear sRGB, may be outside 0-1 for wide gamut inputs)
pub fn transform_cicp_to_linear_srgb(
    pixels: &[f32],
    width: usize,
    height: usize,
    cicp: &Cicp,
) -> Result<Vec<f32>, String> {
    let src_profile = cicp_to_color_profile(cicp)
        .ok_or_else(|| format!("Cannot create profile from CICP: {}", cicp_description(cicp)))?;

    let linear_srgb = make_linear_srgb_profile();

    // Enable extended range to preserve out-of-gamut colors for wide gamut inputs
    let options = TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };

    let transform = src_profile
        .create_transform_f32(
            Layout::Rgb,
            &linear_srgb,
            Layout::Rgb,
            options,
        )
        .map_err(|e| format!("Failed to create CICP transform: {:?}", e))?;

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
            .map_err(|e| format!("CICP transform failed: {:?}", e))?;
    }

    Ok(output)
}

/// Transform image from CICP color space to linear sRGB (array of [f32; 3])
/// Input: array of [r, g, b] f32 (0-1 range)
/// Output: array of [r, g, b] f32 (linear sRGB, may be outside 0-1 for wide gamut inputs)
pub fn transform_cicp_to_linear_srgb_pixels(
    pixels: &[[f32; 4]],
    width: usize,
    height: usize,
    cicp: &Cicp,
) -> Result<Vec<[f32; 3]>, String> {
    // Extract RGB (strip alpha for transform)
    let interleaved: Vec<f32> = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();

    let result = transform_cicp_to_linear_srgb(&interleaved, width, height, cicp)?;

    // Convert back to array format
    Ok(result
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}

// ============================================================================
// ICC Profile Handling
// ============================================================================

/// Create a linear sRGB profile (sRGB primaries with linear TRC)
pub fn make_linear_srgb_profile() -> ColorProfile {
    let mut profile = ColorProfile::new_srgb();
    // Empty LUT = identity/linear (matches moxcms convention)
    let linear_curve = ToneReprCurve::Lut(vec![]);
    profile.red_trc = Some(linear_curve.clone());
    profile.green_trc = Some(linear_curve.clone());
    profile.blue_trc = Some(linear_curve);
    // Update CICP to reflect linear transfer (BT.709 primaries + linear)
    // Note: moxcms uses Bt709 matrix + limited range for sRGB profiles (broadcast convention)
    profile.cicp = Some(MoxcmsCicpProfile {
        color_primaries: MoxcmsPrimaries::Bt709,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Bt709,
        full_range: false,
    });
    // Update description to reflect linear
    profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
        "en".to_string(),
        "US".to_string(),
        "Linear sRGB".to_string(),
    )]));
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
    // Use default options (clamped) for this simple sRGB check
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
/// Output: interleaved RGB f32 (linear sRGB, may be outside 0-1 for wide gamut inputs)
pub fn transform_icc_to_linear_srgb(
    pixels: &[f32],
    width: usize,
    height: usize,
    icc_bytes: &[u8],
) -> Result<Vec<f32>, String> {
    let src_profile = ColorProfile::new_from_slice(icc_bytes)
        .map_err(|e| format!("Failed to parse ICC profile: {:?}", e))?;

    let linear_srgb = make_linear_srgb_profile();

    // Enable extended range to preserve out-of-gamut colors for wide gamut inputs
    // (e.g., Display P3, BT.2020 colors outside sRGB gamut)
    let options = TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };

    let transform = src_profile
        .create_transform_f32(
            Layout::Rgb,
            &linear_srgb,
            Layout::Rgb,
            options,
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
/// Output: array of [r, g, b] f32 (linear sRGB, may be outside 0-1 for wide gamut inputs)
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

/// Image metadata without pixel data
pub struct ImageMetadata {
    pub width: u32,
    pub height: u32,
    pub has_icc: bool,
    pub is_16bit: bool,
    pub is_f32: bool,
    pub has_alpha: bool,
}

/// Get image metadata and ICC profile without decoding pixels
/// This is much faster than full decode for just getting dimensions/profile info
pub fn get_metadata_and_icc(file_bytes: &[u8]) -> Result<(ImageMetadata, Option<Vec<u8>>), String> {
    let cursor = Cursor::new(file_bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| format!("Failed to detect format: {}", e))?;

    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    // Get ICC profile and dimensions from decoder (no pixel decode needed)
    let icc_profile = decoder.icc_profile().ok().flatten();
    let (width, height) = decoder.dimensions();
    let color_type = decoder.color_type();

    let is_16bit = matches!(
        color_type,
        ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
    );

    let is_f32 = matches!(color_type, ColorType::Rgb32F | ColorType::Rgba32F);

    let has_alpha = matches!(
        color_type,
        ColorType::La8 | ColorType::Rgba8 | ColorType::La16 | ColorType::Rgba16 | ColorType::Rgba32F
    );

    let metadata = ImageMetadata {
        width,
        height,
        has_icc: icc_profile.is_some(),
        is_16bit,
        is_f32,
        has_alpha,
    };

    Ok((metadata, icc_profile))
}

/// Extract ICC profile from image file bytes (without decoding pixels)
pub fn extract_icc_profile(file_bytes: &[u8]) -> Option<Vec<u8>> {
    get_metadata_and_icc(file_bytes).ok()?.1
}

// ============================================================================
// SFI (Safetensors Floating-point Image) Format Support
// ============================================================================

/// Check if the data is in SFI (safetensors) format
pub fn is_safetensors_format(data: &[u8]) -> bool {
    sfi::is_sfi_format(data)
}

/// Load a safetensors image file and return as DecodedImage
/// The returned image is already in linear sRGB (if transfer was sRGB, it's been decoded)
pub fn load_safetensors_image(data: &[u8]) -> Result<DecodedImage, String> {
    let sfi_image = sfi::read_sfi(data)
        .map_err(|e| format!("Failed to read safetensors: {}", e))?;

    // Convert SfiImage pixels to DynamicImage
    let width = sfi_image.width;
    let height = sfi_image.height;
    let pixel_count = (width * height) as usize;

    // Build CICP based on SFI metadata
    // SFI files are already converted to linear during read, so we report linear transfer
    let cicp = Cicp {
        primaries: CicpColorPrimaries::SRgb, // SFI srgb primaries = sRGB/BT.709
        transfer: CicpTransferCharacteristics::Linear, // Already decoded to linear
        matrix: image::metadata::CicpMatrixCoefficients::Identity,
        full_range: CicpVideoFullRangeFlag::FullRange,
    };

    // Create DynamicImage from pixels
    // SfiImage.pixels are already linear, so we store as Rgba32F
    let mut rgba_data = Vec::with_capacity(pixel_count * 4);
    for p in &sfi_image.pixels {
        rgba_data.push(p[0]);
        rgba_data.push(p[1]);
        rgba_data.push(p[2]);
        rgba_data.push(p[3]);
    }

    let img_buffer = image::ImageBuffer::<image::Rgba<f32>, Vec<f32>>::from_raw(
        width, height, rgba_data
    ).ok_or_else(|| "Failed to create image buffer from SFI data".to_string())?;

    let image = DynamicImage::ImageRgba32F(img_buffer);

    Ok(DecodedImage {
        image,
        icc_profile: None, // SFI uses metadata, not ICC
        cicp,
        format: None, // No ImageFormat for safetensors
        is_16bit: false,
        is_f32: true,
        has_alpha: sfi_image.has_alpha,
        is_premultiplied_alpha: sfi_image.metadata.alpha_premultiplied,
    })
}

/// Load image from bytes, automatically detecting SFI format
pub fn load_image_from_bytes_auto(data: &[u8]) -> Result<DecodedImage, String> {
    if is_safetensors_format(data) {
        load_safetensors_image(data)
    } else {
        load_image_from_bytes(data)
    }
}

/// Load image from path, automatically detecting SFI format
pub fn load_image_from_path_auto<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    let data = std::fs::read(path.as_ref())
        .map_err(|e| format!("Failed to read {}: {}", path.as_ref().display(), e))?;
    load_image_from_bytes_auto(&data)
}

/// Load a raw binary image file using provided metadata
///
/// This function decodes packed binary formats (RGB565, RGB888, L8, ARGB8888, etc.)
/// and converts them to standard RGBA u8 format using bit replication.
///
/// Raw files are always assumed to be sRGB. No color space conversion is applied.
pub fn load_raw_image<P: AsRef<Path>>(
    path: P,
    metadata: &crate::binary_format::RawImageMetadata,
) -> Result<DecodedImage, String> {
    let data = std::fs::read(path.as_ref())
        .map_err(|e| format!("Failed to read {}: {}", path.as_ref().display(), e))?;

    let decoded = crate::binary_format::decode_raw_image(&data, metadata)?;

    // Convert RGBA u8 data to DynamicImage
    let img = image::RgbaImage::from_raw(
        decoded.width as u32,
        decoded.height as u32,
        decoded.pixels,
    )
    .ok_or_else(|| "Failed to create image from raw data".to_string())?;

    // Create unspecified CICP (raw files don't have color metadata)
    let cicp = Cicp {
        primaries: CicpColorPrimaries::Unspecified,
        transfer: CicpTransferCharacteristics::Unspecified,
        matrix: image::metadata::CicpMatrixCoefficients::Identity,
        full_range: CicpVideoFullRangeFlag::FullRange,
    };

    Ok(DecodedImage {
        image: DynamicImage::ImageRgba8(img),
        icc_profile: None, // Raw files don't have embedded ICC profiles
        cicp,
        format: None, // Not a standard image format
        is_16bit: false, // Output is always 8-bit
        is_f32: false,
        has_alpha: decoded.has_alpha,
        is_premultiplied_alpha: false, // Raw files are not premultiplied
    })
}
