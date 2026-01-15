/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// All WASM-exported functions use the `_wasm` suffix.
/// Buffer types (BufferF32x4, BufferF32, BufferU8) are pure opaque data containers.
/// Width/height are passed as parameters, not stored in buffers.

use wasm_bindgen::prelude::*;
use js_sys;
use image::GenericImageView;

pub mod basic_lab;
pub mod basic_oklab;
pub mod basic_rgb;
pub mod binary_format;
pub mod buffer;
pub mod color;
mod color_distance;
pub mod colorspace_derived;
pub mod correction;
pub mod cra_lab;
pub mod cra_rgb;
pub mod dither;
pub mod dither_colorspace_aware;
pub mod dither_colorspace_lab;
pub mod dither_colorspace_luminosity;
pub mod dither_common;
mod histogram;
pub mod output;
pub mod pixel;
pub mod rescale;
mod rotation;
pub mod tiled_lab;
mod tiling;

pub mod decode;

use buffer::{BufferF32x4, BufferF32, BufferU8};
use dither_common::{DitherMode, PerceptualSpace};
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
        _ => PerceptualSpace::OkLab,
    }
}

fn rescale_method_from_u8(method: u8) -> rescale::RescaleMethod {
    match method {
        1 => rescale::RescaleMethod::Lanczos3,
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

fn correction_method_from_u8(method: u8, luminosity_flag: bool) -> dither_common::ColorCorrectionMethod {
    use dither_common::ColorCorrectionMethod;
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

fn histogram_mode_from_u8(mode: u8) -> dither_common::HistogramMode {
    use dither_common::HistogramMode;
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
    width: u32,
    height: u32,
    is_16bit: bool,
    has_non_srgb_icc: bool,
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
    pub fn has_non_srgb_icc(&self) -> bool {
        self.has_non_srgb_icc
    }

    /// Get ICC profile bytes (only if has_non_srgb_icc is true)
    pub fn get_icc_profile(&self) -> Option<Vec<u8>> {
        self.icc_profile.clone()
    }

    /// Convert to normalized f32 (0-1) for linear processing path
    /// Call this when needs_linear is true
    pub fn to_normalized_buffer(&self) -> BufferF32x4 {
        let normalized = decode::image_to_f32_normalized(&self.image);
        let pixels: Vec<Pixel4> = normalized.into_iter()
            .map(|[r, g, b]| Pixel4::new(r, g, b, 0.0))
            .collect();
        BufferF32x4::new(pixels)
    }

    /// Convert directly to sRGB f32 (0-255) for dither-only path
    /// Call this when needs_linear is false (no resize, no grayscale, no ICC)
    pub fn to_srgb_255_buffer(&self) -> BufferF32x4 {
        let srgb_pixels = decode::image_to_f32_srgb_255_pixels(&self.image);
        let pixels: Vec<Pixel4> = srgb_pixels.into_iter()
            .map(|[r, g, b]| Pixel4::new(r, g, b, 0.0))
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

    // Check if ICC is non-sRGB
    let has_non_srgb_icc = decoded.icc_profile.as_ref()
        .map(|icc| !icc.is_empty() && !decode::is_profile_srgb(icc))
        .unwrap_or(false);

    Ok(LoadedImage {
        image: decoded.image,
        icc_profile: if has_non_srgb_icc { decoded.icc_profile } else { None },
        width,
        height,
        is_16bit: decoded.is_16bit,
        has_non_srgb_icc,
    })
}

/// Get decode metadata without the pixel data (fast - no pixel decoding)
/// Returns: [width, height, has_icc (0/1), is_16bit (0/1)]
#[wasm_bindgen]
pub fn decode_metadata_wasm(file_bytes: Vec<u8>) -> Result<Vec<f32>, JsValue> {
    let (metadata, _) = decode::get_metadata_and_icc(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(vec![
        metadata.width as f32,
        metadata.height as f32,
        if metadata.has_icc { 1.0 } else { 0.0 },
        if metadata.is_16bit { 1.0 } else { 0.0 },
    ])
}

/// Get metadata and check if ICC profile is non-sRGB (single parse)
/// Returns: [width, height, has_non_srgb_icc (0/1), is_16bit (0/1)]
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
    ])
}

/// Create BufferF32x4 from interleaved RGBA f32 data (values 0-1)
#[wasm_bindgen]
pub fn create_buffer_from_rgba_wasm(data: Vec<f32>, pixel_count: usize) -> Result<BufferF32x4, JsValue> {
    if data.len() != pixel_count * 4 {
        return Err(JsValue::from_str(&format!(
            "Data length {} doesn't match pixel_count*4 = {}",
            data.len(), pixel_count * 4
        )));
    }

    let pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]))
        .collect();

    Ok(BufferF32x4::new(pixels))
}

/// Create BufferF32x4 from interleaved RGB f32 data (values 0-1)
#[wasm_bindgen]
pub fn create_buffer_from_rgb_wasm(data: Vec<f32>, pixel_count: usize) -> Result<BufferF32x4, JsValue> {
    if data.len() != pixel_count * 3 {
        return Err(JsValue::from_str(&format!(
            "Data length {} doesn't match pixel_count*3 = {}",
            data.len(), pixel_count * 3
        )));
    }

    let pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.0))
        .collect();

    Ok(BufferF32x4::new(pixels))
}

// ============================================================================
// ICC Profile Handling
// ============================================================================

/// Transform image from ICC profile to linear sRGB (in-place)
#[wasm_bindgen]
pub fn transform_icc_to_linear_srgb_wasm(buf: &mut BufferF32x4, width: usize, height: usize, icc_bytes: Vec<u8>) -> Result<(), JsValue> {
    let pixels = buf.as_slice();
    let interleaved: Vec<f32> = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();

    let result = decode::transform_icc_to_linear_srgb(&interleaved, width, height, &icc_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let pixel_count = width * height;
    let new_pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(result[i * 3], result[i * 3 + 1], result[i * 3 + 2], 0.0))
        .collect();

    *buf = BufferF32x4::new(new_pixels);
    Ok(())
}

// ============================================================================
// Color Space Conversions (in-place on BufferF32x4)
// ============================================================================

/// Convert sRGB (0-1) to linear RGB (0-1) in-place
#[wasm_bindgen]
pub fn srgb_to_linear_wasm(buf: &mut BufferF32x4) {
    color::srgb_to_linear_inplace(buf.as_mut_slice());
}

/// Convert linear RGB (0-1) to sRGB (0-1) in-place
#[wasm_bindgen]
pub fn linear_to_srgb_wasm(buf: &mut BufferF32x4) {
    color::linear_to_srgb_inplace(buf.as_mut_slice());
}

/// Normalize from 0-255 to 0-1 in-place
#[wasm_bindgen]
pub fn normalize_wasm(buf: &mut BufferF32x4) {
    color::normalize_inplace(buf.as_mut_slice());
}

/// Denormalize from 0-1 to 0-255 in-place
#[wasm_bindgen]
pub fn denormalize_wasm(buf: &mut BufferF32x4) {
    color::denormalize_inplace(buf.as_mut_slice());
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

/// Rescale grayscale image
#[wasm_bindgen]
pub fn rescale_gray_wasm(
    buf: &BufferF32,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
) -> BufferF32 {
    let result = rescale::rescale_channel_uniform(
        buf.as_slice(),
        src_width,
        src_height,
        dst_width,
        dst_height,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
    );
    BufferF32::new(result)
}

/// Calculate target dimensions preserving aspect ratio
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
    use dither_common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
        },
    };

    let result_u8 = output::dither_output_interleaved(
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
        dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode(
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
                dither_common::bit_replicate(level.min(max_level as u8), bits)
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
    use dither_common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
        },
    };

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result_u8 = output::dither_output_interleaved(
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
        dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode(
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
                dither_common::bit_replicate(level.min(max_level as u8), bits)
            })
            .collect()
    };

    BufferU8::new(result)
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

#[wasm_bindgen]
pub fn encode_rgb_packed_wasm(
    r_data: Vec<u8>,
    g_data: Vec<u8>,
    b_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    fill: u8,
) -> Vec<u8> {
    let fill_mode = if fill == 0 {
        binary_format::StrideFill::Black
    } else {
        binary_format::StrideFill::Repeat
    };
    binary_format::encode_rgb_packed(&r_data, &g_data, &b_data, width, height, bits_r, bits_g, bits_b, fill_mode)
}

#[wasm_bindgen]
pub fn encode_rgb_row_aligned_wasm(
    r_data: Vec<u8>,
    g_data: Vec<u8>,
    b_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    binary_format::encode_rgb_row_aligned(&r_data, &g_data, &b_data, width, height, bits_r, bits_g, bits_b)
}

#[wasm_bindgen]
pub fn encode_gray_packed_wasm(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_gray_packed(&gray_data, width, height, bits)
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
pub fn encode_channel_packed_wasm(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_channel_packed(&channel_data, width, height, bits)
}

#[wasm_bindgen]
pub fn encode_channel_row_aligned_wasm(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_channel_row_aligned(&channel_data, width, height, bits)
}

#[wasm_bindgen]
pub fn is_valid_stride_wasm(stride: usize) -> bool {
    binary_format::is_valid_stride(stride)
}

#[wasm_bindgen]
pub fn encode_rgb_row_aligned_stride_wasm(
    r_data: Vec<u8>,
    g_data: Vec<u8>,
    b_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_rgb_row_aligned_stride(
        &r_data, &g_data, &b_data, width, height, bits_r, bits_g, bits_b,
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

#[wasm_bindgen]
pub fn encode_channel_row_aligned_stride_wasm(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_channel_row_aligned_stride(
        &channel_data, width, height, bits, stride, binary_format::StrideFill::from_u8(fill)
    )
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

        denormalize_wasm(&mut buf);
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
