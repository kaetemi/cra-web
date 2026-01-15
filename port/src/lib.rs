/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// This crate provides WASM-compatible implementations of various color correction
/// algorithms, ported from the original Python scripts.
///
/// All image processing uses ImageBuffer/GrayBuffer as opaque handles to avoid
/// copying data across the WASM boundary at every function call.

use wasm_bindgen::prelude::*;
use js_sys;

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

use buffer::{ImageBuffer, GrayBuffer};
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
// Image Decoding - Returns ImageBuffer
// ============================================================================

/// Decode image from raw file bytes to ImageBuffer (Pixel4 format, normalized 0-1)
/// The buffer is ready for srgb_to_linear conversion.
#[wasm_bindgen]
pub fn decode_image(file_bytes: Vec<u8>) -> Result<ImageBuffer, JsValue> {
    let result = decode::decode_image_to_f32(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    // Result format: [width, height, has_icc, is_16bit, ...pixels]
    if result.len() < 4 {
        return Err(JsValue::from_str("Invalid decode result"));
    }

    let width = result[0] as u32;
    let height = result[1] as u32;
    let pixel_data = &result[4..];
    let pixel_count = (width as usize) * (height as usize);

    if pixel_data.len() != pixel_count * 3 {
        return Err(JsValue::from_str("Pixel data size mismatch"));
    }

    // Convert to Pixel4
    let pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(pixel_data[i * 3], pixel_data[i * 3 + 1], pixel_data[i * 3 + 2], 0.0))
        .collect();

    Ok(ImageBuffer::from_pixel4(pixels, width, height))
}

/// Decode image directly to sRGB 0-255 scale (for dither-only paths)
#[wasm_bindgen]
pub fn decode_image_srgb_255(file_bytes: Vec<u8>) -> Result<ImageBuffer, JsValue> {
    let result = decode::decode_image_to_srgb_255(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    if result.len() < 4 {
        return Err(JsValue::from_str("Invalid decode result"));
    }

    let width = result[0] as u32;
    let height = result[1] as u32;
    let pixel_data = &result[4..];
    let pixel_count = (width as usize) * (height as usize);

    if pixel_data.len() != pixel_count * 3 {
        return Err(JsValue::from_str("Pixel data size mismatch"));
    }

    let pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(pixel_data[i * 3], pixel_data[i * 3 + 1], pixel_data[i * 3 + 2], 0.0))
        .collect();

    Ok(ImageBuffer::from_pixel4(pixels, width, height))
}

/// Create ImageBuffer from interleaved RGBA f32 data (values 0-1)
/// Useful for creating buffers from canvas-extracted pixel data
#[wasm_bindgen]
pub fn create_buffer_from_rgba(data: Vec<f32>, width: u32, height: u32) -> Result<ImageBuffer, JsValue> {
    let pixel_count = (width as usize) * (height as usize);
    if data.len() != pixel_count * 4 {
        return Err(JsValue::from_str(&format!(
            "Data length {} doesn't match {}x{}x4 = {}",
            data.len(), width, height, pixel_count * 4
        )));
    }

    let pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]))
        .collect();

    Ok(ImageBuffer::from_pixel4(pixels, width, height))
}

/// Create ImageBuffer from interleaved RGB f32 data (values 0-1)
#[wasm_bindgen]
pub fn create_buffer_from_rgb(data: Vec<f32>, width: u32, height: u32) -> Result<ImageBuffer, JsValue> {
    let pixel_count = (width as usize) * (height as usize);
    if data.len() != pixel_count * 3 {
        return Err(JsValue::from_str(&format!(
            "Data length {} doesn't match {}x{}x3 = {}",
            data.len(), width, height, pixel_count * 3
        )));
    }

    let pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.0))
        .collect();

    Ok(ImageBuffer::from_pixel4(pixels, width, height))
}

/// Get decode metadata without the pixel data
/// Returns: [width, height, has_icc (0/1), is_16bit (0/1)]
#[wasm_bindgen]
pub fn decode_metadata(file_bytes: Vec<u8>) -> Result<Vec<f32>, JsValue> {
    let result = decode::decode_image_to_f32(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    if result.len() < 4 {
        return Err(JsValue::from_str("Invalid decode result"));
    }

    Ok(vec![result[0], result[1], result[2], result[3]])
}

/// Extract ICC profile from file bytes
#[wasm_bindgen]
pub fn extract_icc_profile(file_bytes: Vec<u8>) -> Vec<u8> {
    decode::extract_icc_profile(&file_bytes).unwrap_or_default()
}

/// Check if ICC profile is sRGB
#[wasm_bindgen]
pub fn is_icc_profile_srgb(icc_bytes: Vec<u8>) -> bool {
    decode::is_profile_srgb(&icc_bytes)
}

/// Transform image from ICC profile to linear sRGB (in-place)
#[wasm_bindgen]
pub fn transform_icc_to_linear_srgb(buf: &mut ImageBuffer, icc_bytes: Vec<u8>) -> Result<(), JsValue> {
    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    let (width, height) = buf.dimensions();

    // Extract interleaved for transform
    let interleaved: Vec<f32> = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();

    let result = decode::transform_icc_to_linear_srgb(
        &interleaved, width as usize, height as usize, &icc_bytes
    ).map_err(|e| JsValue::from_str(&e))?;

    // Convert back to Pixel4
    let pixel_count = (width as usize) * (height as usize);
    let new_pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(result[i * 3], result[i * 3 + 1], result[i * 3 + 2], 0.0))
        .collect();

    buf.set_pixel4(new_pixels);
    Ok(())
}

// ============================================================================
// Color Space Conversions (in-place on ImageBuffer)
// ============================================================================

/// Convert sRGB (0-1) to linear RGB (0-1) in-place
#[wasm_bindgen]
pub fn srgb_to_linear(buf: &mut ImageBuffer) -> Result<(), JsValue> {
    let pixels = buf.as_pixel4_mut()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;
    color::srgb_to_linear_inplace(pixels);
    Ok(())
}

/// Convert linear RGB (0-1) to sRGB (0-1) in-place
#[wasm_bindgen]
pub fn linear_to_srgb(buf: &mut ImageBuffer) -> Result<(), JsValue> {
    let pixels = buf.as_pixel4_mut()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;
    color::linear_to_srgb_inplace(pixels);
    Ok(())
}

/// Normalize from 0-255 to 0-1 in-place
#[wasm_bindgen]
pub fn normalize(buf: &mut ImageBuffer) -> Result<(), JsValue> {
    let pixels = buf.as_pixel4_mut()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;
    color::normalize_inplace(pixels);
    Ok(())
}

/// Denormalize from 0-1 to 0-255 in-place
#[wasm_bindgen]
pub fn denormalize(buf: &mut ImageBuffer) -> Result<(), JsValue> {
    let pixels = buf.as_pixel4_mut()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;
    color::denormalize_inplace(pixels);
    Ok(())
}

// ============================================================================
// Grayscale Conversion
// ============================================================================

/// Convert linear RGB to linear grayscale (luminance)
#[wasm_bindgen]
pub fn rgb_to_grayscale(buf: &ImageBuffer) -> Result<GrayBuffer, JsValue> {
    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    let gray: Vec<f32> = pixels.iter()
        .map(|p| color::linear_rgb_to_luminance(p[0], p[1], p[2]))
        .collect();

    let (w, h) = buf.dimensions();
    Ok(GrayBuffer::from_f32_internal(gray, w, h))
}

/// Convert linear grayscale to sRGB grayscale (gamma encode) in-place
#[wasm_bindgen]
pub fn gray_linear_to_srgb(buf: &mut GrayBuffer) -> Result<(), JsValue> {
    let data = buf.as_f32_mut()
        .ok_or_else(|| JsValue::from_str("Buffer must be f32 format"))?;

    for v in data.iter_mut() {
        *v = color::linear_to_srgb_single(*v);
    }
    Ok(())
}

/// Denormalize grayscale from 0-1 to 0-255 in-place
#[wasm_bindgen]
pub fn gray_denormalize(buf: &mut GrayBuffer) -> Result<(), JsValue> {
    let data = buf.as_f32_mut()
        .ok_or_else(|| JsValue::from_str("Buffer must be f32 format"))?;

    for v in data.iter_mut() {
        *v *= 255.0;
    }
    Ok(())
}

// ============================================================================
// Color Correction
// ============================================================================

/// Apply color correction
/// Input/output are linear RGB (0-1) in Pixel4 format
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn color_correct(
    input: &ImageBuffer,
    reference: &ImageBuffer,
    method: u8,
    luminosity_flag: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    colorspace_aware_histogram: bool,
    histogram_distance_space: u8,
) -> Result<ImageBuffer, JsValue> {
    use correction::HistogramOptions;

    let input_pixels = input.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Input must be pixel4 format"))?;
    let ref_pixels = reference.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Reference must be pixel4 format"))?;

    let (iw, ih) = input.dimensions();
    let (rw, rh) = reference.dimensions();

    let histogram_options = HistogramOptions {
        mode: histogram_mode_from_u8(histogram_mode),
        dither_mode: dither_mode_from_u8(histogram_dither_mode),
        colorspace_aware: colorspace_aware_histogram,
        colorspace_aware_space: perceptual_space_from_u8(histogram_distance_space),
    };

    let result = correction::color_correct(
        input_pixels,
        ref_pixels,
        iw as usize,
        ih as usize,
        rw as usize,
        rh as usize,
        correction_method_from_u8(method, luminosity_flag),
        histogram_options,
    );

    Ok(ImageBuffer::from_pixel4(result, iw, ih))
}

// ============================================================================
// Rescaling
// ============================================================================

/// Rescale image to new dimensions
/// method: 0=Bilinear, 1=Lanczos3
/// scale_mode: 0=Independent, 1=UniformWidth, 2=UniformHeight
#[wasm_bindgen]
pub fn rescale_rgb(
    buf: &ImageBuffer,
    dst_width: u32,
    dst_height: u32,
    method: u8,
    scale_mode: u8,
) -> Result<ImageBuffer, JsValue> {
    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    let (sw, sh) = buf.dimensions();

    let result = rescale::rescale(
        pixels,
        sw as usize,
        sh as usize,
        dst_width as usize,
        dst_height as usize,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
    );

    Ok(ImageBuffer::from_pixel4(result, dst_width, dst_height))
}

/// Rescale with progress callback
#[wasm_bindgen]
pub fn rescale_rgb_with_progress(
    buf: &ImageBuffer,
    dst_width: u32,
    dst_height: u32,
    method: u8,
    scale_mode: u8,
    progress_callback: &js_sys::Function,
) -> Result<ImageBuffer, JsValue> {
    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    let (sw, sh) = buf.dimensions();

    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();
    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    let result = rescale::rescale_with_progress(
        pixels,
        sw as usize,
        sh as usize,
        dst_width as usize,
        dst_height as usize,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
        Some(&mut progress_fn),
    );

    Ok(ImageBuffer::from_pixel4(result, dst_width, dst_height))
}

/// Rescale grayscale image
#[wasm_bindgen]
pub fn rescale_gray(
    buf: &GrayBuffer,
    dst_width: u32,
    dst_height: u32,
    method: u8,
    scale_mode: u8,
) -> Result<GrayBuffer, JsValue> {
    let data = buf.as_f32()
        .ok_or_else(|| JsValue::from_str("Buffer must be f32 format"))?;

    let (sw, sh) = buf.dimensions();

    let result = rescale::rescale_channel_uniform(
        data,
        sw as usize,
        sh as usize,
        dst_width as usize,
        dst_height as usize,
        rescale_method_from_u8(method),
        scale_mode_from_u8(scale_mode),
    );

    Ok(GrayBuffer::from_f32_internal(result, dst_width, dst_height))
}

/// Calculate target dimensions preserving aspect ratio
#[wasm_bindgen]
pub fn calculate_dimensions(
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

/// Dither RGB image (expects sRGB 0-255 in Pixel4 format)
/// Returns ImageBuffer with quantized values in sRGB 0-255
#[wasm_bindgen]
pub fn dither_rgb(
    buf: &ImageBuffer,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: u8,
    mode: u8,
    space: u8,
    seed: u32,
) -> Result<ImageBuffer, JsValue> {
    use dither_common::OutputTechnique;

    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    let (w, h) = buf.dimensions();
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
        pixels,
        w as usize,
        h as usize,
        bits_r,
        bits_g,
        bits_b,
        output_technique,
        seed,
    );

    // Convert u8 back to Pixel4 with 0-255 values
    let pixel_count = (w as usize) * (h as usize);
    let result_pixels: Vec<Pixel4> = (0..pixel_count)
        .map(|i| Pixel4::new(
            result_u8[i * 3] as f32,
            result_u8[i * 3 + 1] as f32,
            result_u8[i * 3 + 2] as f32,
            0.0
        ))
        .collect();

    Ok(ImageBuffer::from_pixel4(result_pixels, w, h))
}

/// Dither grayscale image (expects sRGB 0-255 in f32 format)
/// Returns GrayBuffer with u8 quantized values
#[wasm_bindgen]
pub fn dither_gray(
    buf: &GrayBuffer,
    bits: u8,
    mode: u8,
    space: u8,
    seed: u32,
) -> Result<GrayBuffer, JsValue> {
    let data = buf.as_f32()
        .ok_or_else(|| JsValue::from_str("Buffer must be f32 format"))?;

    let (w, h) = buf.dimensions();
    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let result = dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode(
        data,
        w as usize,
        h as usize,
        bits,
        perceptual_space,
        dither_mode,
        seed,
    );

    Ok(GrayBuffer::from_u8_internal(result, w, h))
}

// ============================================================================
// Output Extraction
// ============================================================================

/// Extract u8 RGB from Pixel4 buffer (values should already be 0-255)
#[wasm_bindgen]
pub fn to_u8_rgb(buf: &ImageBuffer) -> Result<Vec<u8>, JsValue> {
    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    Ok(pixel::pixels_to_srgb_u8(pixels))
}

/// Extract u8 RGBA from Pixel4 buffer (values should already be 0-255)
#[wasm_bindgen]
pub fn to_u8_rgba(buf: &ImageBuffer) -> Result<Vec<u8>, JsValue> {
    let pixels = buf.as_pixel4()
        .ok_or_else(|| JsValue::from_str("Buffer must be pixel4 format"))?;

    Ok(pixel::pixels_to_srgb_u8_rgba(pixels))
}

/// Extract u8 from GrayBuffer
#[wasm_bindgen]
pub fn gray_to_u8(buf: &GrayBuffer) -> Vec<u8> {
    match buf.as_u8() {
        Some(data) => data.clone(),
        None => match buf.as_f32() {
            Some(data) => data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect(),
            None => Vec::new(),
        }
    }
}

// ============================================================================
// Binary Format Encoding (takes final u8 data)
// ============================================================================

#[wasm_bindgen]
pub fn is_valid_format(format: &str) -> bool {
    binary_format::is_valid_format(format)
}

#[wasm_bindgen]
pub fn format_supports_binary(format: &str) -> bool {
    binary_format::format_supports_binary(format)
}

#[wasm_bindgen]
pub fn format_is_rgb666(format: &str) -> bool {
    binary_format::ColorFormat::parse(format)
        .map(|f| f.is_rgb666())
        .unwrap_or(false)
}

#[wasm_bindgen]
pub fn format_total_bits(format: &str) -> u8 {
    binary_format::format_total_bits(format).unwrap_or(0)
}

#[wasm_bindgen]
pub fn format_is_grayscale(format: &str) -> bool {
    binary_format::format_is_grayscale(format)
}

#[wasm_bindgen]
pub fn encode_rgb_packed(
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
pub fn encode_rgb_row_aligned(
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
pub fn encode_gray_packed(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_gray_packed(&gray_data, width, height, bits)
}

#[wasm_bindgen]
pub fn encode_gray_row_aligned(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_gray_row_aligned(&gray_data, width, height, bits)
}

#[wasm_bindgen]
pub fn encode_channel_packed(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_channel_packed(&channel_data, width, height, bits)
}

#[wasm_bindgen]
pub fn encode_channel_row_aligned(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_channel_row_aligned(&channel_data, width, height, bits)
}

#[wasm_bindgen]
pub fn is_valid_stride(stride: usize) -> bool {
    binary_format::is_valid_stride(stride)
}

#[wasm_bindgen]
pub fn encode_rgb_row_aligned_stride(
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
pub fn encode_gray_row_aligned_stride(
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
pub fn encode_channel_row_aligned_stride(
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

    fn create_test_buffer(data: &[u8], w: u32, h: u32) -> ImageBuffer {
        let pixels: Vec<Pixel4> = data.chunks(3)
            .map(|c| Pixel4::new(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 0.0))
            .collect();
        ImageBuffer::from_pixel4(pixels, w, h)
    }

    #[test]
    fn test_color_space_roundtrip() {
        let mut buf = create_test_buffer(&[128, 64, 32, 200, 100, 50], 2, 1);

        srgb_to_linear(&mut buf).unwrap();
        linear_to_srgb(&mut buf).unwrap();

        let pixels = buf.as_pixel4().unwrap();
        // Values should be approximately the same after roundtrip
        assert!((pixels[0][0] - 128.0/255.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_denormalize() {
        let pixels = vec![Pixel4::new(128.0, 64.0, 32.0, 0.0)];
        let mut buf = ImageBuffer::from_pixel4(pixels, 1, 1);

        normalize(&mut buf).unwrap();
        let p = buf.as_pixel4().unwrap();
        assert!((p[0][0] - 128.0/255.0).abs() < 0.001);

        denormalize(&mut buf).unwrap();
        let p = buf.as_pixel4().unwrap();
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
        let buf = ImageBuffer::from_pixel4(pixels, 2, 2);

        let result = rescale_rgb(&buf, 4, 4, 0, 0).unwrap();
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
    }

    #[test]
    fn test_color_correct_smoke() {
        let input = create_test_buffer(&[128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150], 2, 2);
        let reference = create_test_buffer(&[255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0], 2, 2);

        // Convert to linear
        let mut input_linear = input.clone_buffer();
        srgb_to_linear(&mut input_linear).unwrap();
        let mut ref_linear = reference.clone_buffer();
        srgb_to_linear(&mut ref_linear).unwrap();

        let result = color_correct(&input_linear, &ref_linear, 0, false, 0, 4, false, 0).unwrap();
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 2);
    }
}
