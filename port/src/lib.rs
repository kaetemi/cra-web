/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// All WASM-exported functions use the `_wasm` suffix.
/// Buffer types (BufferF32x4, BufferF32, BufferU8) are pure opaque data containers.
/// Width/height are passed as parameters, not stored in buffers.

use wasm_bindgen::prelude::*;
use image::GenericImageView;

pub mod basic_lab;
pub mod basic_oklab;
pub mod basic_rgb;
pub mod buffer;
pub mod color;
mod color_distance;
pub mod colorspace_derived;
pub mod correction;
pub mod cra_lab;
pub mod cra_rgb;
pub mod dither;
pub mod format;
mod histogram;
pub mod output;
pub mod pixel;
pub mod rescale;
mod rotation;
pub mod supersample;
pub mod tiled_lab;
mod tiling;

// Re-export format submodules at crate root for backwards compatibility
pub use format::binary as binary_format;
pub use format::decode;
pub use format::sfi;

use buffer::{BufferF32x4, BufferF32, BufferU8};
use dither::common::{DitherMode, PerceptualSpace};
use pixel::Pixel4;

// ============================================================================
// Helper functions for enum conversion
// ============================================================================

fn dither_mode_from_u8(mode: u8) -> DitherMode {
    match mode {
        1 => DitherMode::Serpentine,
        2 => DitherMode::JarvisStandard,
        3 => DitherMode::JarvisSerpentine,
        4 => DitherMode::MixedStandard,
        5 => DitherMode::MixedSerpentine,
        6 => DitherMode::MixedRandom,
        7 => DitherMode::None,
        _ => DitherMode::Standard,
    }
}

fn perceptual_space_from_u8(space: u8) -> PerceptualSpace {
    match space {
        0 => PerceptualSpace::LabCIE76,
        2 => PerceptualSpace::LabCIE94,
        3 => PerceptualSpace::LabCIEDE2000,
        4 => PerceptualSpace::LinearRGB,
        5 => PerceptualSpace::YCbCr,
        6 => PerceptualSpace::Srgb,
        7 => PerceptualSpace::YCbCrBt601,
        8 => PerceptualSpace::OkLabLr,
        9 => PerceptualSpace::OkLabHeavyChroma,
        _ => PerceptualSpace::OkLab, // 1 or any other value defaults to OkLab
    }
}

fn rescale_method_from_u8(method: u8) -> rescale::RescaleMethod {
    match method {
        1 => rescale::RescaleMethod::Lanczos3,
        2 => rescale::RescaleMethod::Mitchell,
        3 => rescale::RescaleMethod::CatmullRom,
        4 => rescale::RescaleMethod::Sinc,
        5 => rescale::RescaleMethod::Lanczos3Scatter,
        6 => rescale::RescaleMethod::SincScatter,
        7 => rescale::RescaleMethod::Lanczos2,
        8 => rescale::RescaleMethod::EWASincLanczos2,
        9 => rescale::RescaleMethod::EWASincLanczos3,
        12 => rescale::RescaleMethod::EWALanczos2,
        13 => rescale::RescaleMethod::EWALanczos3,
        14 => rescale::RescaleMethod::EWAMitchell,
        15 => rescale::RescaleMethod::EWACatmullRom,
        16 => rescale::RescaleMethod::Jinc,
        17 => rescale::RescaleMethod::StochasticJinc,
        18 => rescale::RescaleMethod::StochasticJincScatter,
        19 => rescale::RescaleMethod::StochasticJincScatterNormalized,
        20 => rescale::RescaleMethod::Box,
        21 => rescale::RescaleMethod::EWALanczos3Sharp,
        22 => rescale::RescaleMethod::EWALanczos4Sharpest,
        29 => rescale::RescaleMethod::BilinearIterative,
        33 => rescale::RescaleMethod::HybridLanczos3,
        _ => rescale::RescaleMethod::Bilinear,
    }
}

fn scale_mode_from_u8(mode: u8) -> rescale::ScaleMode {
    match mode {
        1 => rescale::ScaleMode::UniformWidth,
        2 => rescale::ScaleMode::UniformHeight,
        _ => rescale::ScaleMode::Independent,
    }
}

fn tent_mode_from_u8(tent_mode: u8) -> rescale::TentMode {
    match tent_mode {
        0 => rescale::TentMode::Off,
        1 => rescale::TentMode::SampleToSample,
        2 => rescale::TentMode::Prescale,
        _ => rescale::TentMode::Off,
    }
}

fn correction_method_from_u8(method: u8, luminosity_flag: bool) -> dither::common::ColorCorrectionMethod {
    use dither::common::ColorCorrectionMethod;
    match method {
        0 => ColorCorrectionMethod::BasicLab { keep_luminosity: luminosity_flag },
        1 => ColorCorrectionMethod::BasicRgb,
        2 => ColorCorrectionMethod::BasicOklab { keep_luminosity: luminosity_flag },
        3 => ColorCorrectionMethod::CraLab { keep_luminosity: luminosity_flag },
        4 => ColorCorrectionMethod::CraRgb { use_perceptual: luminosity_flag },
        5 => ColorCorrectionMethod::CraOklab { keep_luminosity: luminosity_flag },
        6 => ColorCorrectionMethod::TiledLab { tiled_luminosity: luminosity_flag },
        _ => ColorCorrectionMethod::TiledOklab { tiled_luminosity: luminosity_flag },
    }
}

fn histogram_mode_from_u8(mode: u8) -> dither::common::HistogramMode {
    use dither::common::HistogramMode;
    match mode {
        0 => HistogramMode::Binned,
        2 => HistogramMode::MidpointAligned,
        _ => HistogramMode::EndpointAligned,
    }
}

// ============================================================================
// Image Decoding - Returns BufferF32x4
// ============================================================================

/// Loaded image - holds decoded pixels (u8/u16) and ICC profile
/// Converts to appropriate f32 format on demand (like CLI's DecodedImage pattern)
#[wasm_bindgen]
pub struct LoadedImage {
    // Store the DynamicImage for on-demand conversion
    image: image::DynamicImage,
    icc_profile: Option<Vec<u8>>,
    cicp: image::metadata::Cicp,
    width: u32,
    height: u32,
    is_16bit: bool,
    is_f32: bool,
    has_alpha: bool,
    has_non_srgb_icc: bool,
    is_exr: bool,
    // Whether this image has premultiplied alpha (from metadata or format default)
    is_premultiplied_alpha: bool,
    // CICP flags (authoritative color space indicators)
    is_cicp_srgb: bool,
    is_cicp_linear: bool,
    is_cicp_needs_conversion: bool,
}

#[wasm_bindgen]
impl LoadedImage {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn is_16bit(&self) -> bool {
        self.is_16bit
    }

    #[wasm_bindgen(getter)]
    pub fn is_f32(&self) -> bool {
        self.is_f32
    }

    #[wasm_bindgen(getter)]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    #[wasm_bindgen(getter)]
    pub fn has_non_srgb_icc(&self) -> bool {
        self.has_non_srgb_icc
    }

    #[wasm_bindgen(getter)]
    pub fn is_exr(&self) -> bool {
        self.is_exr
    }

    /// Check if this image has premultiplied alpha that needs un-premultiplying.
    /// For safetensors, this comes from metadata. For other formats, it's the format default.
    #[wasm_bindgen(getter)]
    pub fn is_format_premultiplied_default(&self) -> bool {
        self.is_premultiplied_alpha
    }

    /// CICP indicates standard sRGB color space (authoritative check).
    /// sRGB = BT.709 primaries + sRGB transfer + Identity matrix + Full range
    #[wasm_bindgen(getter)]
    pub fn is_cicp_srgb(&self) -> bool {
        self.is_cicp_srgb
    }

    /// CICP indicates linear sRGB color space (authoritative check).
    /// Linear sRGB = BT.709 primaries + Linear transfer + Identity matrix + Full range
    #[wasm_bindgen(getter)]
    pub fn is_cicp_linear(&self) -> bool {
        self.is_cicp_linear
    }

    /// CICP indicates a non-sRGB color space that requires conversion.
    /// Returns true if CICP is specified (not unspecified) and not sRGB/linear-sRGB.
    #[wasm_bindgen(getter)]
    pub fn is_cicp_needs_conversion(&self) -> bool {
        self.is_cicp_needs_conversion
    }

    /// Get ICC profile bytes (only if has_non_srgb_icc is true)
    pub fn get_icc_profile(&self) -> Option<Vec<u8>> {
        self.icc_profile.clone()
    }

    /// Convert to normalized f32 (0-1) for linear processing path (RGB only)
    /// Alpha channel is set to 1.0 (fully opaque)
    /// Call this when needs_linear is true
    pub fn to_normalized_buffer(&self) -> BufferF32x4 {
        let normalized = decode::image_to_f32_normalized(&self.image);
        let pixels: Vec<Pixel4> = normalized.into_iter()
            .map(|[r, g, b]| Pixel4::new(r, g, b, 1.0))
            .collect();
        BufferF32x4::new(pixels)
    }

    /// Convert to normalized f32 (0-1) for linear processing path (RGBA with alpha)
    /// Call this when needs_linear is true and image has alpha
    pub fn to_normalized_buffer_rgba(&self) -> BufferF32x4 {
        let normalized = decode::image_to_f32_normalized_rgba(&self.image);
        let pixels: Vec<Pixel4> = normalized.into_iter()
            .map(|[r, g, b, a]| Pixel4::new(r, g, b, a))
            .collect();
        BufferF32x4::new(pixels)
    }

    /// Convert directly to sRGB f32 (0-255) for dither-only path (RGB only)
    /// Alpha channel is set to 255.0 (fully opaque)
    /// Call this when needs_linear is false (no resize, no grayscale, no ICC)
    pub fn to_srgb_255_buffer(&self) -> BufferF32x4 {
        let srgb_pixels = decode::image_to_f32_srgb_255_pixels(&self.image);
        let pixels: Vec<Pixel4> = srgb_pixels.into_iter()
            .map(|[r, g, b]| Pixel4::new(r, g, b, 255.0))
            .collect();
        BufferF32x4::new(pixels)
    }

    /// Convert directly to sRGB f32 (0-255) for dither-only path (RGBA with alpha)
    /// Call this when needs_linear is false and image has alpha
    pub fn to_srgb_255_buffer_rgba(&self) -> BufferF32x4 {
        let srgb_pixels = decode::image_to_f32_srgb_255_pixels_rgba(&self.image);
        let pixels: Vec<Pixel4> = srgb_pixels.into_iter()
            .map(|[r, g, b, a]| Pixel4::new(r, g, b, a))
            .collect();
        BufferF32x4::new(pixels)
    }
}

/// Load image once - returns LoadedImage that can be converted to either format
/// Single decode, then call to_normalized_buffer() or to_srgb_255_buffer()
#[wasm_bindgen]
pub fn load_image_wasm(file_bytes: Vec<u8>) -> Result<LoadedImage, JsValue> {
    let decoded = decode::load_image_from_bytes(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let (width, height) = decoded.image.dimensions();

    // Check CICP first (authoritative color space indicators)
    let is_cicp_srgb = decode::is_cicp_srgb(&decoded.cicp);
    let is_cicp_linear = decode::is_cicp_linear_srgb(&decoded.cicp);
    let is_cicp_needs_conversion = decode::is_cicp_needs_conversion(&decoded.cicp);

    // Check if ICC is non-sRGB (only if CICP doesn't indicate sRGB/linear)
    // CICP takes precedence over ICC as it's more authoritative
    let has_non_srgb_icc = if is_cicp_srgb || is_cicp_linear {
        // CICP says it's sRGB or linear, trust that over ICC
        false
    } else {
        decoded.icc_profile.as_ref()
            .map(|icc| !icc.is_empty() && !decode::is_profile_srgb(icc))
            .unwrap_or(false)
    };

    // Check if format is EXR (for backwards compatibility flag)
    let is_exr = matches!(decoded.format, Some(image::ImageFormat::OpenExr));

    Ok(LoadedImage {
        image: decoded.image,
        icc_profile: if has_non_srgb_icc { decoded.icc_profile } else { None },
        cicp: decoded.cicp,
        width,
        height,
        is_16bit: decoded.is_16bit,
        is_f32: decoded.is_f32,
        has_alpha: decoded.has_alpha,
        has_non_srgb_icc,
        is_exr,
        is_premultiplied_alpha: decoded.is_premultiplied_alpha,
        is_cicp_srgb,
        is_cicp_linear,
        is_cicp_needs_conversion,
    })
}

/// Get decode metadata without the pixel data (fast - no pixel decoding)
/// Returns: [width, height, has_icc (0/1), is_16bit (0/1), is_f32 (0/1), has_alpha (0/1)]
#[wasm_bindgen]
pub fn decode_metadata_wasm(file_bytes: Vec<u8>) -> Result<Vec<f32>, JsValue> {
    let (metadata, _) = decode::get_metadata_and_icc(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(vec![
        metadata.width as f32,
        metadata.height as f32,
        if metadata.has_icc { 1.0 } else { 0.0 },
        if metadata.is_16bit { 1.0 } else { 0.0 },
        if metadata.is_f32 { 1.0 } else { 0.0 },
        if metadata.has_alpha { 1.0 } else { 0.0 },
    ])
}

/// Get metadata and check if ICC profile is non-sRGB (single parse)
/// Returns: [width, height, has_non_srgb_icc (0/1), is_16bit (0/1), is_f32 (0/1), has_alpha (0/1)]
/// This combines decode_metadata + extract_icc + is_icc_srgb check in one call
#[wasm_bindgen]
pub fn decode_metadata_with_icc_check_wasm(file_bytes: Vec<u8>) -> Result<Vec<f32>, JsValue> {
    let (metadata, icc_profile) = decode::get_metadata_and_icc(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let has_non_srgb_icc = match icc_profile {
        Some(ref icc) if !icc.is_empty() => !decode::is_profile_srgb(icc),
        _ => false,
    };

    Ok(vec![
        metadata.width as f32,
        metadata.height as f32,
        if has_non_srgb_icc { 1.0 } else { 0.0 },
        if metadata.is_16bit { 1.0 } else { 0.0 },
        if metadata.is_f32 { 1.0 } else { 0.0 },
        if metadata.has_alpha { 1.0 } else { 0.0 },
    ])
}

/// Create BufferF32x4 from interleaved f32 data with variable channel count
/// channels: 3 for RGB (alpha defaults to 1.0), 4 for RGBA
#[wasm_bindgen]
pub fn create_buffer_from_interleaved_wasm(data: Vec<f32>, pixel_count: usize, channels: usize) -> Result<BufferF32x4, JsValue> {
    if channels != 3 && channels != 4 {
        return Err(JsValue::from_str("channels must be 3 or 4"));
    }
    if data.len() != pixel_count * channels {
        return Err(JsValue::from_str(&format!(
            "Data length {} doesn't match pixel_count*channels = {}",
            data.len(), pixel_count * channels
        )));
    }

    let pixels = pixel::interleaved_f32_to_pixels(&data, channels);
    Ok(BufferF32x4::new(pixels))
}

/// Create BufferF32x4 from interleaved RGBA f32 data (values 0-1)
#[wasm_bindgen]
pub fn create_buffer_from_rgba_wasm(data: Vec<f32>, pixel_count: usize) -> Result<BufferF32x4, JsValue> {
    create_buffer_from_interleaved_wasm(data, pixel_count, 4)
}

/// Create BufferF32x4 from interleaved RGB f32 data (values 0-1)
/// Alpha channel defaults to 1.0 (fully opaque)
#[wasm_bindgen]
pub fn create_buffer_from_rgb_wasm(data: Vec<f32>, pixel_count: usize) -> Result<BufferF32x4, JsValue> {
    create_buffer_from_interleaved_wasm(data, pixel_count, 3)
}

// ============================================================================
// SFI (Safetensors Floating-point Image) Format Support
// ============================================================================

/// Check if file data is in SFI (safetensors) format
#[wasm_bindgen]
pub fn is_sfi_format_wasm(file_bytes: &[u8]) -> bool {
    decode::is_safetensors_format(file_bytes)
}

/// Load SFI (safetensors) image file, returns LoadedImage
/// The image is automatically converted to linear sRGB.
#[wasm_bindgen]
pub fn load_sfi_wasm(file_bytes: Vec<u8>) -> Result<LoadedImage, JsValue> {
    let decoded = decode::load_safetensors_image(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let (width, height) = image::GenericImageView::dimensions(&decoded.image);

    Ok(LoadedImage {
        image: decoded.image,
        icc_profile: None,
        cicp: decoded.cicp,
        width,
        height,
        is_16bit: false,
        is_f32: true,
        has_alpha: decoded.has_alpha,
        has_non_srgb_icc: false,
        is_exr: false,
        is_premultiplied_alpha: decoded.is_premultiplied_alpha,
        is_cicp_srgb: false,  // SFI files are already linear
        is_cicp_linear: true, // Reported as linear
        is_cicp_needs_conversion: false,
    })
}

/// Write linear RGB buffer to SFI F32 format
/// Input: BufferF32x4 containing linear RGB pixels (0-1 range)
/// Output: Vec<u8> containing the safetensors file bytes
#[wasm_bindgen]
pub fn write_sfi_f32_wasm(buf: &BufferF32x4, width: u32, height: u32, include_alpha: bool) -> Vec<u8> {
    sfi::write_sfi_f32(buf.as_slice(), width, height, include_alpha, sfi::SfiTransfer::Linear)
}

/// Write linear RGB buffer to SFI F16 format
/// Input: BufferF32x4 containing linear RGB pixels (0-1 range)
/// Output: Vec<u8> containing the safetensors file bytes
/// Note: F16 uses round-to-nearest. For optimal precision, error diffusion
/// specific to F16 precision is recommended but not yet implemented.
#[wasm_bindgen]
pub fn write_sfi_f16_wasm(buf: &BufferF32x4, width: u32, height: u32, include_alpha: bool) -> Vec<u8> {
    sfi::write_sfi_f16(buf.as_slice(), width, height, include_alpha, sfi::SfiTransfer::Linear)
}

// ============================================================================
// ICC Profile Handling
// ============================================================================

/// Transform image from ICC profile to linear sRGB (in-place)
/// Alpha channel is preserved unchanged (ICC profiles only apply to RGB).
#[wasm_bindgen]
pub fn transform_icc_to_linear_srgb_wasm(buf: &mut BufferF32x4, width: usize, height: usize, icc_bytes: Vec<u8>) -> Result<(), JsValue> {
    let pixels = buf.as_slice();

    // Extract RGB (ICC profile applies only to RGB, not alpha)
    let interleaved: Vec<f32> = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();

    let result = decode::transform_icc_to_linear_srgb(&interleaved, width, height, &icc_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    // Rebuild pixels preserving original alpha values
    let pixel_count = width * height;
    let new_pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(result[i * 3], result[i * 3 + 1], result[i * 3 + 2], pixels[i][3]))
        .collect();

    *buf = BufferF32x4::new(new_pixels);
    Ok(())
}

/// Transform image from CICP color space to linear sRGB (in-place)
/// Alpha channel is preserved unchanged (color profiles only apply to RGB).
/// Uses the CICP stored in the LoadedImage.
#[wasm_bindgen]
pub fn transform_cicp_to_linear_srgb_wasm(buf: &mut BufferF32x4, width: usize, height: usize, loaded_image: &LoadedImage) -> Result<(), JsValue> {
    let pixels = buf.as_slice();

    // Extract RGBA for transform (CICP transform takes [f32; 4])
    let rgba: Vec<[f32; 4]> = pixels.iter().map(|p| [p[0], p[1], p[2], p[3]]).collect();

    let result = decode::transform_cicp_to_linear_srgb_pixels(&rgba, width, height, &loaded_image.cicp)
        .map_err(|e| JsValue::from_str(&e))?;

    // Rebuild pixels preserving original alpha values
    let pixel_count = width * height;
    let new_pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(result[i][0], result[i][1], result[i][2], pixels[i][3]))
        .collect();

    *buf = BufferF32x4::new(new_pixels);
    Ok(())
}

// ============================================================================
// Color Space Conversions (in-place on BufferF32x4)
// ============================================================================

/// Convert sRGB (0-1) to linear RGB (0-1) in-place
/// RGB channels are gamma-decoded; alpha channel passes through unchanged (already linear)
#[wasm_bindgen]
pub fn srgb_to_linear_wasm(buf: &mut BufferF32x4) {
    color::srgb_to_linear_inplace(buf.as_mut_slice());
}

/// Convert linear RGB (0-1) to sRGB (0-1) in-place
/// RGB channels are gamma-encoded; alpha channel passes through unchanged (already linear)
#[wasm_bindgen]
pub fn linear_to_srgb_wasm(buf: &mut BufferF32x4) {
    color::linear_to_srgb_inplace(buf.as_mut_slice());
}

/// Normalize from 0-255 to 0-1 in-place (all 4 channels including alpha)
#[wasm_bindgen]
pub fn normalize_wasm(buf: &mut BufferF32x4) {
    color::normalize_inplace(buf.as_mut_slice());
}

/// Denormalize from 0-1 to 0-255 in-place (all 4 channels including alpha, clamps to 0-255)
#[wasm_bindgen]
pub fn denormalize_clamped_wasm(buf: &mut BufferF32x4) {
    color::denormalize_inplace_clamped(buf.as_mut_slice());
}

/// Un-premultiply alpha in-place (divide RGB by alpha)
/// This should be called after converting to linear space for images with premultiplied alpha
/// (like EXR files which have premultiplied alpha by default)
#[wasm_bindgen]
pub fn unpremultiply_alpha_wasm(buf: &mut BufferF32x4) {
    pixel::unpremultiply_alpha_inplace(buf.as_mut_slice());
}

/// Premultiply alpha in-place (multiply RGB by alpha)
/// This is the inverse of unpremultiply_alpha_wasm
#[wasm_bindgen]
pub fn premultiply_alpha_wasm(buf: &mut BufferF32x4) {
    pixel::premultiply_alpha_inplace(buf.as_mut_slice());
}

// ============================================================================
// Grayscale Conversion
// ============================================================================

/// Convert linear RGB to linear grayscale (luminance)
#[wasm_bindgen]
pub fn rgb_to_grayscale_wasm(buf: &BufferF32x4) -> BufferF32 {
    let gray: Vec<f32> = buf.as_slice().iter()
        .map(|p| color::linear_rgb_to_luminance(p[0], p[1], p[2]))
        .collect();
    BufferF32::new(gray)
}

/// Extract alpha channel from RGBA buffer
/// Returns alpha values in 0-1 range (same as input)
#[wasm_bindgen]
pub fn extract_alpha_wasm(buf: &BufferF32x4) -> BufferF32 {
    let alpha: Vec<f32> = buf.as_slice().iter()
        .map(|p| p[3])
        .collect();
    BufferF32::new(alpha)
}

/// Convert linear grayscale to sRGB grayscale (gamma encode) in-place
#[wasm_bindgen]
pub fn gray_linear_to_srgb_wasm(buf: &mut BufferF32) {
    for v in buf.as_mut_slice().iter_mut() {
        *v = color::linear_to_srgb_single(*v);
    }
}

/// Denormalize grayscale from 0-1 to 0-255 in-place
#[wasm_bindgen]
pub fn gray_denormalize_wasm(buf: &mut BufferF32) {
    for v in buf.as_mut_slice().iter_mut() {
        *v *= 255.0;
    }
}

// ============================================================================
// Tonemapping
// ============================================================================

/// Apply ACES tonemapping to linear RGB buffer in-place
/// Compresses HDR values to SDR range [0, 1]
#[wasm_bindgen]
pub fn tonemap_aces_wasm(buf: &mut BufferF32x4) {
    color::tonemap_aces_inplace(buf.as_mut_slice());
}

/// Apply inverse ACES tonemapping to linear RGB buffer in-place
/// Expands SDR values to approximate HDR range
#[wasm_bindgen]
pub fn tonemap_aces_inverse_wasm(buf: &mut BufferF32x4) {
    color::tonemap_aces_inverse_inplace(buf.as_mut_slice());
}

/// Apply ACES tonemapping to grayscale buffer in-place
#[wasm_bindgen]
pub fn gray_tonemap_aces_wasm(buf: &mut BufferF32) {
    for v in buf.as_mut_slice().iter_mut() {
        *v = color::tonemap_aces_single(*v);
    }
}

/// Apply inverse ACES tonemapping to grayscale buffer in-place
#[wasm_bindgen]
pub fn gray_tonemap_aces_inverse_wasm(buf: &mut BufferF32) {
    for v in buf.as_mut_slice().iter_mut() {
        *v = color::tonemap_aces_inverse_single(*v);
    }
}

// ============================================================================
// Color Correction
// ============================================================================

/// Apply color correction
/// Input/output are linear RGB (0-1) in Pixel4 format
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn color_correct_wasm(
    input: &BufferF32x4,
    reference: &BufferF32x4,
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    method: u8,
    luminosity_flag: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    colorspace_aware_histogram: bool,
    histogram_distance_space: u8,
) -> BufferF32x4 {
    use correction::HistogramOptions;

    let histogram_options = HistogramOptions {
        mode: histogram_mode_from_u8(histogram_mode),
        dither_mode: dither_mode_from_u8(histogram_dither_mode),
        colorspace_aware: colorspace_aware_histogram,
        colorspace_aware_space: perceptual_space_from_u8(histogram_distance_space),
    };

    let result = correction::color_correct(
        input.as_slice(),
        reference.as_slice(),
        input_width,
        input_height,
        ref_width,
        ref_height,
        correction_method_from_u8(method, luminosity_flag),
        histogram_options,
        None,
    );

    BufferF32x4::new(result)
}

/// Apply color correction with progress callback
/// Input/output are linear RGB (0-1) in Pixel4 format
/// Progress callback is called with values 0.0 to 1.0
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn color_correct_with_progress_wasm(
    input: &BufferF32x4,
    reference: &BufferF32x4,
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    method: u8,
    luminosity_flag: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    colorspace_aware_histogram: bool,
    histogram_distance_space: u8,
    progress_callback: &js_sys::Function,
) -> BufferF32x4 {
    use correction::HistogramOptions;

    let histogram_options = HistogramOptions {
        mode: histogram_mode_from_u8(histogram_mode),
        dither_mode: dither_mode_from_u8(histogram_dither_mode),
        colorspace_aware: colorspace_aware_histogram,
        colorspace_aware_space: perceptual_space_from_u8(histogram_distance_space),
    };

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = correction::color_correct(
        input.as_slice(),
        reference.as_slice(),
        input_width,
        input_height,
        ref_width,
        ref_height,
        correction_method_from_u8(method, luminosity_flag),
        histogram_options,
        Some(&mut progress_fn),
    );

    BufferF32x4::new(result)
}

// ============================================================================
// Rescaling
// ============================================================================

/// Rescale image to new dimensions
/// method: 0=Bilinear, 1=Lanczos3
/// scale_mode: 0=Independent, 1=UniformWidth, 2=UniformHeight
#[wasm_bindgen]
pub fn rescale_rgb_wasm(
    buf: &BufferF32x4,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
) -> BufferF32x4 {
    let result = rescale::rescale(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
    );
    BufferF32x4::new(result)
}

/// Rescale with progress callback
#[wasm_bindgen]
pub fn rescale_rgb_with_progress_wasm(
    buf: &BufferF32x4,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
    progress_callback: &js_sys::Function,
) -> BufferF32x4 {
    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = rescale::rescale_with_progress(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
        Some(&mut progress_fn),
    );

    BufferF32x4::new(result)
}

/// Calculate target dimensions preserving aspect ratio
/// Automatically enforces uniform scaling when dimensions are within 1 pixel of uniform AR.
#[wasm_bindgen]
pub fn calculate_dimensions_wasm(
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> Vec<u32> {
    let tw = if target_width == 0 { None } else { Some(target_width as usize) };
    let th = if target_height == 0 { None } else { Some(target_height as usize) };
    let (w, h) = rescale::calculate_target_dimensions(src_width as usize, src_height as usize, tw, th);
    vec![w as u32, h as u32]
}

/// Calculate target dimensions with explicit control over uniform scaling
/// When force_exact is true, uses exact dimensions without automatic uniform scaling adjustment.
#[wasm_bindgen]
pub fn calculate_dimensions_exact_wasm(
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
    force_exact: bool,
) -> Vec<u32> {
    let tw = if target_width == 0 { None } else { Some(target_width as usize) };
    let th = if target_height == 0 { None } else { Some(target_height as usize) };
    let (w, h) = rescale::calculate_target_dimensions_exact(src_width as usize, src_height as usize, tw, th, force_exact);
    vec![w as u32, h as u32]
}

/// Alpha-aware rescale for RGBA images
/// Unlike regular rescaling, this weights RGB by alpha during interpolation
/// to prevent transparent pixels from bleeding their color into opaque regions.
/// method: 0=Bilinear, 1=Lanczos3
/// scale_mode: 0=Independent, 1=UniformWidth, 2=UniformHeight
#[wasm_bindgen]
pub fn rescale_rgb_alpha_wasm(
    buf: &BufferF32x4,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
) -> BufferF32x4 {
    let result = rescale::rescale_with_alpha(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
    );
    BufferF32x4::new(result)
}

/// Alpha-aware rescale with progress callback
#[wasm_bindgen]
pub fn rescale_rgb_alpha_with_progress_wasm(
    buf: &BufferF32x4,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
    progress_callback: &js_sys::Function,
) -> BufferF32x4 {
    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = rescale::rescale_with_alpha_progress(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
        Some(&mut progress_fn),
    );

    BufferF32x4::new(result)
}

/// Rescale with tent_mode support for supersampling
/// tent_mode: 0=Off, 1=SampleToSample (for tent-volume), 2=Prescale (tent-to-box)
#[wasm_bindgen]
pub fn rescale_rgb_tent_with_progress_wasm(
    buf: &BufferF32x4,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
    tent_mode: u8,
    progress_callback: &js_sys::Function,
) -> BufferF32x4 {
    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = rescale::rescale_with_progress_tent(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
        tent_mode_from_u8(tent_mode),
        Some(&mut progress_fn),
    );

    BufferF32x4::new(result)
}

/// Alpha-aware rescale with tent_mode support for supersampling
/// tent_mode: 0=Off, 1=SampleToSample (for tent-volume), 2=Prescale (tent-to-box)
#[wasm_bindgen]
pub fn rescale_rgb_alpha_tent_with_progress_wasm(
    buf: &BufferF32x4,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
    tent_mode: u8,
    progress_callback: &js_sys::Function,
) -> BufferF32x4 {
    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = rescale::rescale_with_alpha_progress_tent(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
        tent_mode_from_u8(tent_mode),
        Some(&mut progress_fn),
    );

    BufferF32x4::new(result)
}

// ============================================================================
// Supersampling (Tent Volume)
// ============================================================================

/// Expand box-space image to tent-space for supersampling
/// Returns expanded buffer and new dimensions as [width, height]
#[wasm_bindgen]
pub fn tent_expand_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
) -> TentExpandResult {
    let (expanded, new_width, new_height) = supersample::tent_expand(buf.as_slice(), width, height);
    TentExpandResult {
        buffer: BufferF32x4::new(expanded),
        width: new_width as u32,
        height: new_height as u32,
    }
}

/// Result of tent expansion containing buffer and new dimensions
#[wasm_bindgen]
pub struct TentExpandResult {
    buffer: BufferF32x4,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl TentExpandResult {
    #[wasm_bindgen(getter)]
    pub fn buffer(&self) -> BufferF32x4 {
        self.buffer.clone_buffer()
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Contract tent-space image back to box-space after supersampled processing
/// Returns contracted buffer and new dimensions as [width, height]
#[wasm_bindgen]
pub fn tent_contract_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
) -> TentContractResult {
    let (contracted, new_width, new_height) = supersample::tent_contract(buf.as_slice(), width, height);
    TentContractResult {
        buffer: BufferF32x4::new(contracted),
        width: new_width as u32,
        height: new_height as u32,
    }
}

/// Result of tent contraction containing buffer and new dimensions
#[wasm_bindgen]
pub struct TentContractResult {
    buffer: BufferF32x4,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl TentContractResult {
    #[wasm_bindgen(getter)]
    pub fn buffer(&self) -> BufferF32x4 {
        self.buffer.clone_buffer()
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Calculate tent-space target dimensions for supersampling
/// Given a target box-space size, returns the tent-space dimensions needed
#[wasm_bindgen]
pub fn supersample_target_dimensions_wasm(width: u32, height: u32) -> Vec<u32> {
    let (w, h) = supersample::supersample_target_dimensions(width as usize, height as usize);
    vec![w as u32, h as u32]
}

/// Expand box-space image to tent-space using Lanczos3 constraint
/// Peaks are adjusted so that Lanczos3 interpolation at each peak returns the original value.
/// Returns expanded buffer and new dimensions.
#[wasm_bindgen]
pub fn tent_expand_lanczos_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
) -> TentExpandResult {
    let (expanded, new_width, new_height) = supersample::tent_expand_lanczos(buf.as_slice(), width, height);
    TentExpandResult {
        buffer: BufferF32x4::new(expanded),
        width: new_width as u32,
        height: new_height as u32,
    }
}

/// Contract tent-space image back to box-space using Lanczos3 interpolation
/// Applies Lanczos3 at each peak position to recover original values.
/// Use with tent_expand_lanczos_wasm for a fully reversible pipeline.
#[wasm_bindgen]
pub fn tent_contract_lanczos_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
) -> TentContractResult {
    let (contracted, new_width, new_height) = supersample::tent_contract_lanczos(buf.as_slice(), width, height);
    TentContractResult {
        buffer: BufferF32x4::new(contracted),
        width: new_width as u32,
        height: new_height as u32,
    }
}

// ============================================================================
// Dithering
// ============================================================================

/// Dither RGB image and return quantized u8 RGB data directly
/// Input: BufferF32x4 with sRGB 0-255 values
/// Output: BufferU8 with interleaved RGB u8 values (3 bytes per pixel)
#[wasm_bindgen]
pub fn dither_rgb_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: u8,
    mode: u8,
    space: u8,
    seed: u32,
) -> BufferU8 {
    use dither::common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode, alpha_mode: None },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
            alpha_mode: None,
        },
    };

    let result_u8 = output::dither_output_rgb(
        buf.as_slice(),
        width,
        height,
        bits_r,
        bits_g,
        bits_b,
        output_technique,
        seed,
        None,
    );

    BufferU8::new(result_u8)
}

/// Dither grayscale image (expects sRGB 0-255 in f32 format)
/// Returns BufferU8 with quantized values
#[wasm_bindgen]
pub fn dither_gray_wasm(
    buf: &BufferF32,
    width: usize,
    height: usize,
    bits: u8,
    technique: u8,
    mode: u8,
    space: u8,
    seed: u32,
) -> BufferU8 {
    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let result = if technique >= 2 {
        // Colorspace-aware dithering
        dither::luminosity::colorspace_aware_dither_gray_with_mode(
            buf.as_slice(),
            width,
            height,
            bits,
            perceptual_space,
            dither_mode,
            seed,
            None,
        )
    } else if technique == 1 {
        // Per-channel dithering
        dither::dither_with_mode_bits(buf.as_slice(), width, height, dither_mode, seed, bits, None)
    } else {
        // No dithering - just quantize (round to nearest level)
        let levels = 1u32 << bits;
        let max_level = (levels - 1) as f32;
        buf.as_slice().iter()
            .map(|&v| {
                let level = ((v / 255.0) * max_level + 0.5) as u8;
                dither::common::bit_replicate(level.min(max_level as u8), bits)
            })
            .collect()
    };

    BufferU8::new(result)
}

/// Dither RGB image with progress callback
/// Input: BufferF32x4 with sRGB 0-255 values
/// Output: BufferU8 with interleaved RGB u8 values (3 bytes per pixel)
/// Progress callback is called after each row with progress (0.0 to 1.0)
#[wasm_bindgen]
pub fn dither_rgb_with_progress_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: u8,
    mode: u8,
    space: u8,
    seed: u32,
    progress_callback: &js_sys::Function,
) -> BufferU8 {
    use dither::common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode, alpha_mode: None },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
            alpha_mode: None,
        },
    };

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result_u8 = output::dither_output_rgb(
        buf.as_slice(),
        width,
        height,
        bits_r,
        bits_g,
        bits_b,
        output_technique,
        seed,
        Some(&mut progress_fn),
    );

    BufferU8::new(result_u8)
}

/// Dither grayscale image with progress callback (expects sRGB 0-255 in f32 format)
/// Returns BufferU8 with quantized values
/// Progress callback is called after each row with progress (0.0 to 1.0)
#[wasm_bindgen]
pub fn dither_gray_with_progress_wasm(
    buf: &BufferF32,
    width: usize,
    height: usize,
    bits: u8,
    technique: u8,
    mode: u8,
    space: u8,
    seed: u32,
    progress_callback: &js_sys::Function,
) -> BufferU8 {
    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = if technique >= 2 {
        // Colorspace-aware dithering with progress
        dither::luminosity::colorspace_aware_dither_gray_with_mode(
            buf.as_slice(),
            width,
            height,
            bits,
            perceptual_space,
            dither_mode,
            seed,
            Some(&mut progress_fn),
        )
    } else if technique == 1 {
        // Per-channel dithering - no progress support
        dither::dither_with_mode_bits(buf.as_slice(), width, height, dither_mode, seed, bits, None)
    } else {
        // No dithering - just quantize (round to nearest level)
        let levels = 1u32 << bits;
        let max_level = (levels - 1) as f32;
        buf.as_slice().iter()
            .map(|&v| {
                let level = ((v / 255.0) * max_level + 0.5) as u8;
                dither::common::bit_replicate(level.min(max_level as u8), bits)
            })
            .collect()
    };

    BufferU8::new(result)
}

/// Dither LA (Luminosity+Alpha) image with progress callback
/// Input: Two BufferF32 - one for grayscale (sRGB 0-255), one for alpha (0-255)
/// Output: BufferU8 with interleaved LA u8 values (2 bytes per pixel: L, A)
/// Uses alpha-aware error propagation: transparent pixels pass error through.
/// alpha_mode: Separate dither mode for alpha channel (255 = use same as mode)
/// Progress callback is called after each row with progress (0.0 to 1.0)
#[wasm_bindgen]
pub fn dither_la_with_progress_wasm(
    gray_buf: &BufferF32,
    alpha_buf: &BufferF32,
    width: usize,
    height: usize,
    bits_l: u8,
    bits_a: u8,
    technique: u8,
    mode: u8,
    alpha_mode: u8,
    space: u8,
    seed: u32,
    progress_callback: &js_sys::Function,
) -> BufferU8 {
    use dither::common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);
    let alpha_dither_mode = if alpha_mode == 255 {
        None
    } else {
        Some(dither_mode_from_u8(alpha_mode))
    };

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode, alpha_mode: alpha_dither_mode },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
            alpha_mode: alpha_dither_mode,
        },
    };

    let result_u8 = output::dither_output_la(
        gray_buf.as_slice(),
        alpha_buf.as_slice(),
        width,
        height,
        bits_l,
        bits_a,
        output_technique,
        seed,
        Some(&mut progress_fn),
    );

    BufferU8::new(result_u8)
}

/// Dither RGBA image and return quantized u8 RGBA data directly
/// Input: BufferF32x4 with sRGB 0-255 values (alpha also 0-255)
/// Output: BufferU8 with interleaved RGBA u8 values (4 bytes per pixel)
/// When bits_a > 0: Alpha channel is dithered with alpha-aware RGB error propagation.
/// When bits_a == 0: Alpha is stripped and RGB-only dithering is used (output alpha = 255).
/// alpha_mode: Separate dither mode for alpha channel (255 = use same as mode)
#[wasm_bindgen]
pub fn dither_rgba_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    bits_a: u8,
    technique: u8,
    mode: u8,
    alpha_mode: u8,
    space: u8,
    seed: u32,
) -> BufferU8 {
    use dither::common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);
    let alpha_dither_mode = if alpha_mode == 255 {
        None
    } else {
        Some(dither_mode_from_u8(alpha_mode))
    };

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode, alpha_mode: alpha_dither_mode },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
            alpha_mode: alpha_dither_mode,
        },
    };

    let result_u8 = output::dither_output_rgba(
        buf.as_slice(),
        width,
        height,
        bits_r,
        bits_g,
        bits_b,
        bits_a,
        output_technique,
        seed,
        None,
    );

    BufferU8::new(result_u8)
}

/// Dither RGBA image with progress callback
/// Input: BufferF32x4 with sRGB 0-255 values (alpha also 0-255)
/// Output: BufferU8 with interleaved RGBA u8 values (4 bytes per pixel)
/// When bits_a > 0: Alpha channel is dithered with alpha-aware RGB error propagation.
/// When bits_a == 0: Alpha is stripped and RGB-only dithering is used (output alpha = 255).
/// alpha_mode: Separate dither mode for alpha channel (255 = use same as mode)
/// Progress callback is called after each row with progress (0.0 to 1.0)
#[wasm_bindgen]
pub fn dither_rgba_with_progress_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    bits_a: u8,
    technique: u8,
    mode: u8,
    alpha_mode: u8,
    space: u8,
    seed: u32,
    progress_callback: &js_sys::Function,
) -> BufferU8 {
    use dither::common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);
    let alpha_dither_mode = if alpha_mode == 255 {
        None
    } else {
        Some(dither_mode_from_u8(alpha_mode))
    };

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode, alpha_mode: alpha_dither_mode },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
            alpha_mode: alpha_dither_mode,
        },
    };

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result_u8 = output::dither_output_rgba(
        buf.as_slice(),
        width,
        height,
        bits_r,
        bits_g,
        bits_b,
        bits_a,
        output_technique,
        seed,
        Some(&mut progress_fn),
    );

    BufferU8::new(result_u8)
}

// ============================================================================
// Paletted Dithering (Palette-based dithering with integrated alpha-RGB distance)
// ============================================================================

/// Generate the web-safe 216-color palette (6×6×6 RGB cube)
/// Returns RGBA tuples as (r, g, b, 255) - all colors are fully opaque
fn generate_websafe_palette() -> Vec<(u8, u8, u8, u8)> {
    const LEVELS: [u8; 6] = [0, 51, 102, 153, 204, 255];
    let mut colors = Vec::with_capacity(216);
    for &r in &LEVELS {
        for &g in &LEVELS {
            for &b in &LEVELS {
                colors.push((r, g, b, 255));
            }
        }
    }
    colors
}

/// Generate the CGA 5153 monitor palette (16 colors)
/// Hardware-accurate palette based on actual IBM 5153 monitor voltage normalization
fn generate_cga_5153_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 00: Black
        (0x00, 0x00, 0xC4, 255), // 01: Blue
        (0x00, 0xC4, 0x00, 255), // 02: Green
        (0x00, 0xC4, 0xC4, 255), // 03: Cyan
        (0xC4, 0x00, 0x00, 255), // 04: Red
        (0xC4, 0x00, 0xC4, 255), // 05: Magenta
        (0xC4, 0x7E, 0x00, 255), // 06: Brown (dark yellow)
        (0xC4, 0xC4, 0xC4, 255), // 07: Light gray
        (0x4E, 0x4E, 0x4E, 255), // 08: Dark gray
        (0x4E, 0x4E, 0xDC, 255), // 09: Light blue
        (0x4E, 0xDC, 0x4E, 255), // 10: Light green
        (0x4E, 0xF3, 0xF3, 255), // 11: Light cyan
        (0xDC, 0x4E, 0x4E, 255), // 12: Light red
        (0xF3, 0x4E, 0xF3, 255), // 13: Light magenta
        (0xF3, 0xF3, 0x4E, 255), // 14: Yellow
        (0xFF, 0xFF, 0xFF, 255), // 15: White
    ]
}

/// Generate the CGA BIOS/EGA canonical palette (16 colors)
/// The "fake" standard palette commonly used in emulators and documentation
fn generate_cga_bios_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 00: Black
        (0x00, 0x00, 0xAA, 255), // 01: Blue
        (0x00, 0xAA, 0x00, 255), // 02: Green
        (0x00, 0xAA, 0xAA, 255), // 03: Cyan
        (0xAA, 0x00, 0x00, 255), // 04: Red
        (0xAA, 0x00, 0xAA, 255), // 05: Magenta
        (0xAA, 0x55, 0x00, 255), // 06: Brown (dark yellow)
        (0xAA, 0xAA, 0xAA, 255), // 07: Light gray
        (0x55, 0x55, 0x55, 255), // 08: Dark gray
        (0x55, 0x55, 0xFF, 255), // 09: Light blue
        (0x55, 0xFF, 0x55, 255), // 10: Light green
        (0x55, 0xFF, 0xFF, 255), // 11: Light cyan
        (0xFF, 0x55, 0x55, 255), // 12: Light red
        (0xFF, 0x55, 0xFF, 255), // 13: Light magenta
        (0xFF, 0xFF, 0x55, 255), // 14: Yellow
        (0xFF, 0xFF, 0xFF, 255), // 15: White
    ]
}

/// Generate the CGA Palette 1 (4 colors)
/// The cyan/magenta high-intensity palette used in CGA graphics mode 4/5
fn generate_cga_palette1_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 0: Black
        (0x55, 0xFF, 0xFF, 255), // 1: Cyan (85, 255, 255)
        (0xFF, 0x55, 0xFF, 255), // 2: Magenta (255, 85, 255)
        (0xFF, 0xFF, 0xFF, 255), // 3: White
    ]
}

/// Generate the CGA Palette 1 with 5153 monitor colors (4 colors)
/// Hardware-accurate palette based on actual IBM 5153 monitor measurements
fn generate_cga_palette1_5153_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 0: Black
        (0x4E, 0xF3, 0xF3, 255), // 1: Light cyan (78, 243, 243)
        (0xF3, 0x4E, 0xF3, 255), // 2: Light magenta (243, 78, 243)
        (0xFF, 0xFF, 0xFF, 255), // 3: White
    ]
}

/// Generate palette colors based on palette type
/// 0 = WebSafe (216 colors), 1 = CGA 5153 (16 colors), 2 = CGA BIOS (16 colors),
/// 3 = CGA Palette 1 (4 colors), 4 = CGA Palette 1 5153 (4 colors)
fn generate_palette(palette_type: u8) -> Vec<(u8, u8, u8, u8)> {
    match palette_type {
        1 => generate_cga_5153_palette(),
        2 => generate_cga_bios_palette(),
        3 => generate_cga_palette1_palette(),
        4 => generate_cga_palette1_5153_palette(),
        _ => generate_websafe_palette(), // Default to web-safe
    }
}

/// Dither RGBA image to palette with integrated alpha-RGB distance metric
/// Input: BufferF32x4 with sRGB 0-255 values (alpha also 0-255)
/// Output: BufferU8 with interleaved RGBA u8 values (4 bytes per pixel)
///
/// Uses the integrated distance metric: sqrt(alpha_dist² + (rgb_dist × alpha)²)
/// This weighs down RGB errors where pixels are less visible (low alpha).
///
/// palette_type: 0 = web-safe (216 colors), 1 = CGA 5153 (16 colors), 2 = CGA BIOS (16 colors)
/// mode: Dither mode (0=none, 1=fs-standard, 2=fs-serpentine, etc.)
/// space: Perceptual space for RGB distance (0=OkLab, etc.)
/// use_ghost_entries: Whether to use ghost entries for gamut mapping (recommended: true)
#[wasm_bindgen]
pub fn dither_paletted_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
    palette_type: u8,
    mode: u8,
    space: u8,
    seed: u32,
    use_ghost_entries: bool,
) -> BufferU8 {
    use dither::paletted::{DitherPalette, paletted_dither_rgba_gamut_mapped};
    use color::interleave_rgba_u8;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    // Generate palette based on type
    let palette_colors = generate_palette(palette_type);

    let palette = DitherPalette::new(&palette_colors, perceptual_space);

    // Extract channels from input buffer
    let pixels = buf.as_slice();
    let r_channel: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
    let g_channel: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
    let b_channel: Vec<f32> = pixels.iter().map(|p| p[2]).collect();
    let a_channel: Vec<f32> = pixels.iter().map(|p| p[3]).collect();

    // Perform gamut-mapped paletted dithering
    let (r_out, g_out, b_out, a_out) = paletted_dither_rgba_gamut_mapped(
        &r_channel, &g_channel, &b_channel, &a_channel,
        width, height,
        &palette,
        dither_mode,
        seed,
        use_ghost_entries,
        true, // overshoot_penalty enabled by default
        None,
    );

    // Interleave to RGBA
    let interleaved = interleave_rgba_u8(&r_out, &g_out, &b_out, &a_out);

    BufferU8::new(interleaved)
}

/// Dither RGBA image to palette with progress callback
/// Input: BufferF32x4 with sRGB 0-255 values (alpha also 0-255)
/// Output: BufferU8 with interleaved RGBA u8 values (4 bytes per pixel)
///
/// palette_type: 0 = web-safe (216 colors), 1 = CGA 5153 (16 colors), 2 = CGA BIOS (16 colors)
/// mode: Dither mode (0=none, 1=fs-standard, 2=fs-serpentine, etc.)
/// space: Perceptual space for RGB distance (0=OkLab, etc.)
/// use_ghost_entries: Whether to use ghost entries for gamut mapping (recommended: true)
#[wasm_bindgen]
pub fn dither_paletted_with_progress_wasm(
    buf: &BufferF32x4,
    width: usize,
    height: usize,
    palette_type: u8,
    mode: u8,
    space: u8,
    seed: u32,
    use_ghost_entries: bool,
    progress_callback: &js_sys::Function,
) -> BufferU8 {
    use dither::paletted::{DitherPalette, paletted_dither_rgba_gamut_mapped};
    use color::interleave_rgba_u8;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    // Generate palette based on type
    let palette_colors = generate_palette(palette_type);

    let palette = DitherPalette::new(&palette_colors, perceptual_space);

    // Extract channels from input buffer
    let pixels = buf.as_slice();
    let r_channel: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
    let g_channel: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
    let b_channel: Vec<f32> = pixels.iter().map(|p| p[2]).collect();
    let a_channel: Vec<f32> = pixels.iter().map(|p| p[3]).collect();

    // Setup progress callback
    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    // Perform gamut-mapped paletted dithering
    let (r_out, g_out, b_out, a_out) = paletted_dither_rgba_gamut_mapped(
        &r_channel, &g_channel, &b_channel, &a_channel,
        width, height,
        &palette,
        dither_mode,
        seed,
        use_ghost_entries,
        true, // overshoot_penalty enabled by default
        Some(&mut progress_fn),
    );

    // Interleave to RGBA
    let interleaved = interleave_rgba_u8(&r_out, &g_out, &b_out, &a_out);

    BufferU8::new(interleaved)
}

// ============================================================================
// Binary Format Encoding (takes final u8 data)
// ============================================================================

#[wasm_bindgen]
pub fn is_valid_format_wasm(format: &str) -> bool {
    binary_format::is_valid_format(format)
}

#[wasm_bindgen]
pub fn format_supports_binary_wasm(format: &str) -> bool {
    binary_format::format_supports_binary(format)
}

#[wasm_bindgen]
pub fn format_supports_palettized_wasm(format: &str) -> bool {
    binary_format::ColorFormat::parse(format)
        .map(|f| binary_format::supports_palettized_png(&f))
        .unwrap_or(false)
}

#[wasm_bindgen]
pub fn format_is_rgb666_wasm(format: &str) -> bool {
    binary_format::ColorFormat::parse(format)
        .map(|f| f.is_rgb666())
        .unwrap_or(false)
}

#[wasm_bindgen]
pub fn format_total_bits_wasm(format: &str) -> u8 {
    binary_format::format_total_bits(format).unwrap_or(0)
}

#[wasm_bindgen]
pub fn format_is_grayscale_wasm(format: &str) -> bool {
    binary_format::format_is_grayscale(format)
}

/// Encode interleaved RGB data to row-aligned binary format
/// Input: Interleaved RGB u8 data (RGBRGB..., 3 bytes per pixel)
#[wasm_bindgen]
pub fn encode_rgb_row_aligned_wasm(
    rgb_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    binary_format::encode_rgb_row_aligned(&rgb_data, width, height, bits_r, bits_g, bits_b)
}

#[wasm_bindgen]
pub fn encode_gray_row_aligned_wasm(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_gray_row_aligned(&gray_data, width, height, bits)
}

#[wasm_bindgen]
pub fn is_valid_stride_wasm(stride: usize) -> bool {
    binary_format::is_valid_stride(stride)
}

/// Encode interleaved RGB data to row-aligned binary format with configurable stride
/// Input: Interleaved RGB u8 data (RGBRGB..., 3 bytes per pixel)
#[wasm_bindgen]
pub fn encode_rgb_row_aligned_stride_wasm(
    rgb_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_rgb_row_aligned_stride(
        &rgb_data, width, height, bits_r, bits_g, bits_b,
        stride, binary_format::StrideFill::from_u8(fill)
    )
}

#[wasm_bindgen]
pub fn encode_gray_row_aligned_stride_wasm(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_gray_row_aligned_stride(
        &gray_data, width, height, bits, stride, binary_format::StrideFill::from_u8(fill)
    )
}

/// Encode interleaved LA data to row-aligned binary format
/// Input: Interleaved LA u8 data (LALA..., 2 bytes per pixel, L first then A)
/// Output: Row-aligned LA binary data with hardware ordering (A in MSB, L in LSB)
#[wasm_bindgen]
pub fn encode_la_row_aligned_stride_wasm(
    la_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_l: u8,
    bits_a: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_la_row_aligned_stride(
        &la_data, width, height, bits_l, bits_a, stride, binary_format::StrideFill::from_u8(fill)
    )
}

#[wasm_bindgen]
/// Encode a single channel from interleaved data to row-aligned binary format
/// Input: Interleaved u8 data (e.g., RGBRGB... or RGBARGBA...)
/// num_channels: Number of channels in the interleaved data (3 for RGB, 4 for RGBA)
/// channel: Which channel to extract (0=R, 1=G, 2=B, 3=A)
pub fn encode_channel_row_aligned_stride_wasm(
    interleaved_data: Vec<u8>,
    width: usize,
    height: usize,
    num_channels: u8,
    channel: u8,
    bits: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_channel_from_interleaved_row_aligned_stride(
        &interleaved_data, width, height, num_channels as usize, channel as usize, bits, stride, binary_format::StrideFill::from_u8(fill)
    )
}

/// Encode interleaved RGBA data to row-aligned ARGB binary format
/// Input: Interleaved RGBA u8 data (RGBARGBA..., 4 bytes per pixel)
/// Output: Row-aligned ARGB binary data with hardware ordering (A in MSB, R, G, B toward LSB)
#[wasm_bindgen]
pub fn encode_argb_row_aligned_stride_wasm(
    rgba_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_a: u8,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_argb_row_aligned_stride(
        &rgba_data, width, height, bits_a, bits_r, bits_g, bits_b,
        stride, binary_format::StrideFill::from_u8(fill)
    )
}

// ============================================================================
// PNG Encoding (returns bytes, not file)
// ============================================================================

/// Encode RGB u8 data to PNG bytes
/// Input: Interleaved RGB u8 data (RGBRGB..., 3 bytes per pixel)
/// Output: PNG file bytes
#[wasm_bindgen]
pub fn encode_png_rgb_wasm(data: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    use image::{ImageBuffer, Rgb};
    use std::io::Cursor;

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, data)
        .ok_or_else(|| JsValue::from_str("Failed to create RGB image buffer"))?;

    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&format!("Failed to encode PNG: {}", e)))?;

    Ok(bytes)
}

/// Encode RGBA u8 data to PNG bytes
/// Input: Interleaved RGBA u8 data (RGBARGBA..., 4 bytes per pixel)
/// Output: PNG file bytes
#[wasm_bindgen]
pub fn encode_png_rgba_wasm(data: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    use image::{ImageBuffer, Rgba};
    use std::io::Cursor;

    let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, data)
        .ok_or_else(|| JsValue::from_str("Failed to create RGBA image buffer"))?;

    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&format!("Failed to encode PNG: {}", e)))?;

    Ok(bytes)
}

/// Encode grayscale u8 data to PNG bytes
/// Input: Grayscale u8 data (1 byte per pixel)
/// Output: PNG file bytes
#[wasm_bindgen]
pub fn encode_png_gray_wasm(data: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    use image::{ImageBuffer, Luma};
    use std::io::Cursor;

    let img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, data)
        .ok_or_else(|| JsValue::from_str("Failed to create grayscale image buffer"))?;

    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&format!("Failed to encode PNG: {}", e)))?;

    Ok(bytes)
}

/// Encode grayscale+alpha u8 data to PNG bytes
/// Input: Interleaved LA u8 data (LALA..., 2 bytes per pixel)
/// Output: PNG file bytes
#[wasm_bindgen]
pub fn encode_png_gray_alpha_wasm(data: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    use image::{ImageBuffer, LumaA};
    use std::io::Cursor;

    let img: ImageBuffer<LumaA<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, data)
        .ok_or_else(|| JsValue::from_str("Failed to create grayscale+alpha image buffer"))?;

    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&format!("Failed to encode PNG: {}", e)))?;

    Ok(bytes)
}

/// Encode interleaved u8 data to palettized PNG bytes
/// Input: Interleaved u8 data (format depends on color format type)
///   - Grayscale (L): 1 byte per pixel
///   - Grayscale+Alpha (LA): 2 bytes per pixel (L, A)
///   - RGB: 3 bytes per pixel (R, G, B)
///   - RGBA/ARGB: 4 bytes per pixel (R, G, B, A)
/// Output: PNG file bytes with palette containing bit-replicated colors
/// Only works for formats with ≤8 bits per pixel
#[wasm_bindgen]
pub fn encode_palettized_png_wasm(
    interleaved_data: Vec<u8>,
    format: &str,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsValue> {
    let color_format = binary_format::ColorFormat::parse(format)
        .map_err(|e| JsValue::from_str(&e))?;

    binary_format::encode_palettized_png(
        &interleaved_data,
        width as usize,
        height as usize,
        &color_format,
    )
    .map_err(|e| JsValue::from_str(&e))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_buffer(data: &[u8]) -> BufferF32x4 {
        let pixels: Vec<Pixel4> = data.chunks(3)
            .map(|c| Pixel4::new(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 0.0))
            .collect();
        BufferF32x4::new(pixels)
    }

    #[test]
    fn test_color_space_roundtrip() {
        let mut buf = create_test_buffer(&[128, 64, 32, 200, 100, 50]);

        srgb_to_linear_wasm(&mut buf);
        linear_to_srgb_wasm(&mut buf);

        let pixels = buf.as_slice();
        // Values should be approximately the same after roundtrip
        assert!((pixels[0][0] - 128.0/255.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_denormalize() {
        let pixels = vec![Pixel4::new(128.0, 64.0, 32.0, 0.0)];
        let mut buf = BufferF32x4::new(pixels);

        normalize_wasm(&mut buf);
        let p = buf.as_slice();
        assert!((p[0][0] - 128.0/255.0).abs() < 0.001);

        denormalize_clamped_wasm(&mut buf);
        let p = buf.as_slice();
        assert!((p[0][0] - 128.0).abs() < 0.01);
    }

    #[test]
    fn test_rescale() {
        let pixels = vec![
            Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(1.0, 1.0, 1.0, 0.0),
            Pixel4::new(0.0, 0.0, 0.0, 0.0),
            Pixel4::new(1.0, 1.0, 1.0, 0.0),
        ];
        let buf = BufferF32x4::new(pixels);

        let result = rescale_rgb_wasm(&buf, 2, 2, 4, 4, 0, 0);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_color_correct_smoke() {
        let mut input = create_test_buffer(&[128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150]);
        let mut reference = create_test_buffer(&[255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0]);

        // Convert to linear
        srgb_to_linear_wasm(&mut input);
        srgb_to_linear_wasm(&mut reference);

        let result = color_correct_wasm(&input, &reference, 2, 2, 2, 2, 0, false, 0, 4, false, 0);
        assert_eq!(result.len(), 4);
    }
}
