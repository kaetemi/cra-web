/// WASM-friendly image buffer types with opaque handles
///
/// These buffers allow efficient transfer between JS and WASM without
/// copying at every function boundary. Buffers hold raw data - no colorspace
/// or format tracking beyond the pixel data type.

use wasm_bindgen::prelude::*;
use crate::pixel::Pixel4;

/// Pixel data storage - holds the raw buffer data
#[derive(Clone)]
pub enum PixelData {
    /// Raw u8 bytes (sRGB, grayscale, etc.)
    U8(Vec<u8>),
    /// f32 channel data (linear RGB, normalized, etc.)
    F32(Vec<f32>),
    /// SIMD-friendly Pixel4 array (16-byte aligned)
    Pixel4(Vec<Pixel4>),
}

/// Opaque image buffer handle for WASM
/// Holds either u8, f32, or Pixel4 data plus dimensions
#[wasm_bindgen]
pub struct ImageBuffer {
    data: PixelData,
    width: u32,
    height: u32,
    channels: u8,
}

#[wasm_bindgen]
impl ImageBuffer {
    // ========================================================================
    // Constructors - create buffers from JS typed arrays
    // ========================================================================

    /// Create buffer from u8 array (e.g., sRGB image data)
    #[wasm_bindgen(constructor)]
    pub fn new_u8(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Result<ImageBuffer, JsValue> {
        let expected = (width as usize) * (height as usize) * (channels as usize);
        if data.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Data length {} doesn't match {}x{}x{} = {}",
                data.len(), width, height, channels, expected
            )));
        }
        Ok(ImageBuffer {
            data: PixelData::U8(data),
            width,
            height,
            channels,
        })
    }

    /// Create buffer from f32 array (e.g., linear RGB)
    #[wasm_bindgen]
    pub fn from_f32(data: Vec<f32>, width: u32, height: u32, channels: u8) -> Result<ImageBuffer, JsValue> {
        let expected = (width as usize) * (height as usize) * (channels as usize);
        if data.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Data length {} doesn't match {}x{}x{} = {}",
                data.len(), width, height, channels, expected
            )));
        }
        Ok(ImageBuffer {
            data: PixelData::F32(data),
            width,
            height,
            channels,
        })
    }

    /// Create buffer with Pixel4 data (SIMD-optimized format)
    /// For RGB/RGBA data: channels should be 3 or 4, but internal storage is Pixel4
    #[wasm_bindgen]
    pub fn from_pixel4_interleaved(data: Vec<f32>, width: u32, height: u32) -> Result<ImageBuffer, JsValue> {
        let pixel_count = (width as usize) * (height as usize);
        if data.len() != pixel_count * 4 {
            return Err(JsValue::from_str(&format!(
                "Data length {} doesn't match {}x{}x4 = {}",
                data.len(), width, height, pixel_count * 4
            )));
        }

        // Convert interleaved f32 to Pixel4
        let pixels: Vec<Pixel4> = (0..pixel_count)
            .map(|i| Pixel4::new(data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]))
            .collect();

        Ok(ImageBuffer {
            data: PixelData::Pixel4(pixels),
            width,
            height,
            channels: 4,
        })
    }

    /// Create empty Pixel4 buffer with given dimensions
    #[wasm_bindgen]
    pub fn new_pixel4(width: u32, height: u32) -> ImageBuffer {
        let pixel_count = (width as usize) * (height as usize);
        ImageBuffer {
            data: PixelData::Pixel4(vec![Pixel4::default(); pixel_count]),
            width,
            height,
            channels: 4,
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get the buffer format as string: "u8", "f32", or "pixel4"
    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        match &self.data {
            PixelData::U8(_) => "u8".to_string(),
            PixelData::F32(_) => "f32".to_string(),
            PixelData::Pixel4(_) => "pixel4".to_string(),
        }
    }

    /// Get total pixel count
    #[wasm_bindgen(getter)]
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    // ========================================================================
    // Data extraction - get data back to JS
    // ========================================================================

    /// Get u8 data (returns empty if not u8 format)
    #[wasm_bindgen]
    pub fn get_u8_data(&self) -> Vec<u8> {
        match &self.data {
            PixelData::U8(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Get f32 data (returns empty if not f32 format)
    #[wasm_bindgen]
    pub fn get_f32_data(&self) -> Vec<f32> {
        match &self.data {
            PixelData::F32(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Get Pixel4 data as interleaved f32 RGBA (returns empty if not pixel4 format)
    #[wasm_bindgen]
    pub fn get_pixel4_interleaved(&self) -> Vec<f32> {
        match &self.data {
            PixelData::Pixel4(pixels) => {
                let mut result = Vec::with_capacity(pixels.len() * 4);
                for p in pixels {
                    result.push(p[0]);
                    result.push(p[1]);
                    result.push(p[2]);
                    result.push(p[3]);
                }
                result
            }
            _ => Vec::new(),
        }
    }

    /// Get Pixel4 data as interleaved f32 RGB (ignoring alpha, returns empty if not pixel4)
    #[wasm_bindgen]
    pub fn get_pixel4_rgb_interleaved(&self) -> Vec<f32> {
        match &self.data {
            PixelData::Pixel4(pixels) => {
                let mut result = Vec::with_capacity(pixels.len() * 3);
                for p in pixels {
                    result.push(p[0]);
                    result.push(p[1]);
                    result.push(p[2]);
                }
                result
            }
            _ => Vec::new(),
        }
    }

    // ========================================================================
    // Format conversion (lossless transforms only)
    // ========================================================================

    /// Convert 3-channel f32 interleaved to Pixel4 format (padding with 0.0)
    #[wasm_bindgen]
    pub fn f32_rgb_to_pixel4(&self) -> Result<ImageBuffer, JsValue> {
        match &self.data {
            PixelData::F32(data) => {
                if self.channels != 3 {
                    return Err(JsValue::from_str("Expected 3-channel f32 data"));
                }
                let pixel_count = (self.width as usize) * (self.height as usize);
                let pixels: Vec<Pixel4> = (0..pixel_count)
                    .map(|i| Pixel4::new(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.0))
                    .collect();
                Ok(ImageBuffer {
                    data: PixelData::Pixel4(pixels),
                    width: self.width,
                    height: self.height,
                    channels: 4,
                })
            }
            _ => Err(JsValue::from_str("Buffer is not f32 format")),
        }
    }

    /// Convert Pixel4 to 3-channel f32 interleaved (dropping alpha)
    #[wasm_bindgen]
    pub fn pixel4_to_f32_rgb(&self) -> Result<ImageBuffer, JsValue> {
        match &self.data {
            PixelData::Pixel4(pixels) => {
                let mut data = Vec::with_capacity(pixels.len() * 3);
                for p in pixels {
                    data.push(p[0]);
                    data.push(p[1]);
                    data.push(p[2]);
                }
                Ok(ImageBuffer {
                    data: PixelData::F32(data),
                    width: self.width,
                    height: self.height,
                    channels: 3,
                })
            }
            _ => Err(JsValue::from_str("Buffer is not pixel4 format")),
        }
    }

    // ========================================================================
    // Clone for when you need to keep the original
    // ========================================================================

    #[wasm_bindgen]
    pub fn clone_buffer(&self) -> ImageBuffer {
        ImageBuffer {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            channels: self.channels,
        }
    }
}

// ============================================================================
// Non-WASM internal API for Rust code
// ============================================================================

impl ImageBuffer {
    /// Get mutable reference to Pixel4 data (for internal processing)
    pub fn as_pixel4_mut(&mut self) -> Option<&mut Vec<Pixel4>> {
        match &mut self.data {
            PixelData::Pixel4(pixels) => Some(pixels),
            _ => None,
        }
    }

    /// Get immutable reference to Pixel4 data
    pub fn as_pixel4(&self) -> Option<&Vec<Pixel4>> {
        match &self.data {
            PixelData::Pixel4(pixels) => Some(pixels),
            _ => None,
        }
    }

    /// Get mutable reference to f32 data
    pub fn as_f32_mut(&mut self) -> Option<&mut Vec<f32>> {
        match &mut self.data {
            PixelData::F32(data) => Some(data),
            _ => None,
        }
    }

    /// Get immutable reference to f32 data
    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match &self.data {
            PixelData::F32(data) => Some(data),
            _ => None,
        }
    }

    /// Get mutable reference to u8 data
    pub fn as_u8_mut(&mut self) -> Option<&mut Vec<u8>> {
        match &mut self.data {
            PixelData::U8(data) => Some(data),
            _ => None,
        }
    }

    /// Get immutable reference to u8 data
    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match &self.data {
            PixelData::U8(data) => Some(data),
            _ => None,
        }
    }

    /// Create from existing Pixel4 vec (internal use)
    pub fn from_pixel4_vec(pixels: Vec<Pixel4>, width: u32, height: u32) -> Self {
        ImageBuffer {
            data: PixelData::Pixel4(pixels),
            width,
            height,
            channels: 4,
        }
    }

    /// Create from existing f32 vec (internal use)
    pub fn from_f32_vec(data: Vec<f32>, width: u32, height: u32, channels: u8) -> Self {
        ImageBuffer {
            data: PixelData::F32(data),
            width,
            height,
            channels,
        }
    }

    /// Create from existing u8 vec (internal use)
    pub fn from_u8_vec(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        ImageBuffer {
            data: PixelData::U8(data),
            width,
            height,
            channels,
        }
    }

    /// Take ownership of Pixel4 data (internal use)
    pub fn take_pixel4(self) -> Option<Vec<Pixel4>> {
        match self.data {
            PixelData::Pixel4(pixels) => Some(pixels),
            _ => None,
        }
    }

    /// Take ownership of f32 data (internal use)
    pub fn take_f32(self) -> Option<Vec<f32>> {
        match self.data {
            PixelData::F32(data) => Some(data),
            _ => None,
        }
    }

    /// Take ownership of u8 data (internal use)
    pub fn take_u8(self) -> Option<Vec<u8>> {
        match self.data {
            PixelData::U8(data) => Some(data),
            _ => None,
        }
    }

    /// Get dimensions as tuple
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Grayscale buffer for single-channel operations
#[wasm_bindgen]
pub struct GrayBuffer {
    data: GrayData,
    width: u32,
    height: u32,
}

#[derive(Clone)]
pub enum GrayData {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

#[wasm_bindgen]
impl GrayBuffer {
    /// Create buffer from u8 array
    #[wasm_bindgen(constructor)]
    pub fn new_u8(data: Vec<u8>, width: u32, height: u32) -> Result<GrayBuffer, JsValue> {
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Data length {} doesn't match {}x{} = {}",
                data.len(), width, height, expected
            )));
        }
        Ok(GrayBuffer {
            data: GrayData::U8(data),
            width,
            height,
        })
    }

    /// Create buffer from f32 array
    #[wasm_bindgen]
    pub fn from_f32(data: Vec<f32>, width: u32, height: u32) -> Result<GrayBuffer, JsValue> {
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Data length {} doesn't match {}x{} = {}",
                data.len(), width, height, expected
            )));
        }
        Ok(GrayBuffer {
            data: GrayData::F32(data),
            width,
            height,
        })
    }

    /// Create empty f32 buffer
    #[wasm_bindgen]
    pub fn new_f32(width: u32, height: u32) -> GrayBuffer {
        let pixel_count = (width as usize) * (height as usize);
        GrayBuffer {
            data: GrayData::F32(vec![0.0; pixel_count]),
            width,
            height,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        match &self.data {
            GrayData::U8(_) => "u8".to_string(),
            GrayData::F32(_) => "f32".to_string(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Get u8 data
    #[wasm_bindgen]
    pub fn get_u8_data(&self) -> Vec<u8> {
        match &self.data {
            GrayData::U8(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Get f32 data
    #[wasm_bindgen]
    pub fn get_f32_data(&self) -> Vec<f32> {
        match &self.data {
            GrayData::F32(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub fn clone_buffer(&self) -> GrayBuffer {
        GrayBuffer {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
        }
    }
}

// Non-WASM API for GrayBuffer
impl GrayBuffer {
    pub fn as_f32_mut(&mut self) -> Option<&mut Vec<f32>> {
        match &mut self.data {
            GrayData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match &self.data {
            GrayData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut Vec<u8>> {
        match &mut self.data {
            GrayData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match &self.data {
            GrayData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn from_f32_vec(data: Vec<f32>, width: u32, height: u32) -> Self {
        GrayBuffer {
            data: GrayData::F32(data),
            width,
            height,
        }
    }

    pub fn from_u8_vec(data: Vec<u8>, width: u32, height: u32) -> Self {
        GrayBuffer {
            data: GrayData::U8(data),
            width,
            height,
        }
    }

    pub fn take_f32(self) -> Option<Vec<f32>> {
        match self.data {
            GrayData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn take_u8(self) -> Option<Vec<u8>> {
        match self.data {
            GrayData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

// ============================================================================
// WASM Buffer Operations - Color conversion, correction, dithering, rescale
// ============================================================================

use crate::color;
use crate::dither::DitherMode;
use crate::dither_common::PerceptualSpace;
use crate::rescale::{RescaleMethod, ScaleMode};

/// Convert u8 mode integer to DitherMode enum
fn dither_mode_from_u8(mode: u8) -> DitherMode {
    match mode {
        0 => DitherMode::Standard,
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

/// Convert u8 to PerceptualSpace
fn perceptual_space_from_u8(space: u8) -> PerceptualSpace {
    match space {
        0 => PerceptualSpace::LabCIE76,
        1 => PerceptualSpace::LabCIE94,
        2 => PerceptualSpace::LabCIEDE2000,
        3 => PerceptualSpace::OkLab,
        _ => PerceptualSpace::OkLab,
    }
}

/// Convert u8 to RescaleMethod
fn rescale_method_from_u8(method: u8) -> RescaleMethod {
    match method {
        0 => RescaleMethod::Bilinear,
        1 => RescaleMethod::Lanczos3,
        _ => RescaleMethod::Bilinear,
    }
}

/// Convert u8 to ScaleMode
fn scale_mode_from_u8(mode: u8) -> ScaleMode {
    match mode {
        0 => ScaleMode::Independent,
        1 => ScaleMode::UniformWidth,
        2 => ScaleMode::UniformHeight,
        _ => ScaleMode::Independent,
    }
}

// ============================================================================
// ImageBuffer operations
// ============================================================================

#[wasm_bindgen]
impl ImageBuffer {
    // ========================================================================
    // Color space conversions (in-place for Pixel4 data)
    // ========================================================================

    /// Convert sRGB to linear RGB (in-place, Pixel4 format required)
    #[wasm_bindgen]
    pub fn srgb_to_linear(&mut self) -> Result<(), JsValue> {
        match &mut self.data {
            PixelData::Pixel4(pixels) => {
                color::srgb_to_linear_inplace(pixels);
                Ok(())
            }
            _ => Err(JsValue::from_str("srgb_to_linear requires pixel4 format")),
        }
    }

    /// Convert linear RGB to sRGB (in-place, Pixel4 format required)
    #[wasm_bindgen]
    pub fn linear_to_srgb(&mut self) -> Result<(), JsValue> {
        match &mut self.data {
            PixelData::Pixel4(pixels) => {
                color::linear_to_srgb_inplace(pixels);
                Ok(())
            }
            _ => Err(JsValue::from_str("linear_to_srgb requires pixel4 format")),
        }
    }

    /// Normalize pixel values from 0-255 to 0-1 range (in-place)
    #[wasm_bindgen]
    pub fn normalize(&mut self) -> Result<(), JsValue> {
        match &mut self.data {
            PixelData::Pixel4(pixels) => {
                color::normalize_inplace(pixels);
                Ok(())
            }
            _ => Err(JsValue::from_str("normalize requires pixel4 format")),
        }
    }

    /// Denormalize pixel values from 0-1 to 0-255 range (in-place)
    #[wasm_bindgen]
    pub fn denormalize(&mut self) -> Result<(), JsValue> {
        match &mut self.data {
            PixelData::Pixel4(pixels) => {
                color::denormalize_inplace(pixels);
                Ok(())
            }
            _ => Err(JsValue::from_str("denormalize requires pixel4 format")),
        }
    }

    // ========================================================================
    // Color correction (returns new buffer)
    // ========================================================================

    /// Apply color correction (requires Pixel4 in linear RGB, 0-1 range)
    /// method: 0=BasicLab, 1=BasicRgb, 2=BasicOklab, 3=CraLab, 4=CraRgb, 5=CraOklab, 6=TiledLab, 7=TiledOklab
    /// luminosity_flag: method-specific flag (keep_luminosity for Lab methods, use_perceptual for RGB)
    #[wasm_bindgen]
    pub fn color_correct(
        &self,
        reference: &ImageBuffer,
        method: u8,
        luminosity_flag: bool,
    ) -> Result<ImageBuffer, JsValue> {
        use crate::correction::{color_correct, HistogramOptions};
        use crate::dither_common::ColorCorrectionMethod;

        let src_pixels = self.as_pixel4()
            .ok_or_else(|| JsValue::from_str("Source must be pixel4 format"))?;
        let ref_pixels = reference.as_pixel4()
            .ok_or_else(|| JsValue::from_str("Reference must be pixel4 format"))?;

        let cc_method = match method {
            0 => ColorCorrectionMethod::BasicLab { keep_luminosity: luminosity_flag },
            1 => ColorCorrectionMethod::BasicRgb,
            2 => ColorCorrectionMethod::BasicOklab { keep_luminosity: luminosity_flag },
            3 => ColorCorrectionMethod::CraLab { keep_luminosity: luminosity_flag },
            4 => ColorCorrectionMethod::CraRgb { use_perceptual: luminosity_flag },
            5 => ColorCorrectionMethod::CraOklab { keep_luminosity: luminosity_flag },
            6 => ColorCorrectionMethod::TiledLab { tiled_luminosity: luminosity_flag },
            _ => ColorCorrectionMethod::TiledOklab { tiled_luminosity: luminosity_flag },
        };

        let result = color_correct(
            src_pixels,
            ref_pixels,
            self.width as usize,
            self.height as usize,
            reference.width as usize,
            reference.height as usize,
            cc_method,
            HistogramOptions::default(),
        );

        Ok(ImageBuffer::from_pixel4_vec(result, self.width, self.height))
    }

    // ========================================================================
    // Dithering (returns new buffer with quantized values)
    // ========================================================================

    /// Dither RGB image (requires Pixel4 in sRGB 0-255 range)
    /// Returns buffer with quantized values in sRGB 0-255 range
    /// bits: bit depth per channel (1-8)
    /// mode: 0=Standard, 1=Serpentine, 2=JarvisStd, 3=JarvisSerpentine, 4-6=Mixed, 7=None
    #[wasm_bindgen]
    pub fn dither_rgb(
        &self,
        bits: u8,
        mode: u8,
        seed: u32,
    ) -> Result<ImageBuffer, JsValue> {
        let pixels = self.as_pixel4()
            .ok_or_else(|| JsValue::from_str("dither_rgb requires pixel4 format"))?;

        let dither_mode = dither_mode_from_u8(mode);
        let height = self.height as usize;

        // Extract RGB channels
        let r_scaled: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
        let g_scaled: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
        let b_scaled: Vec<f32> = pixels.iter().map(|p| p[2]).collect();

        // Apply dithering to each channel
        let r_u8 = crate::dither::dither_with_mode_bits(&r_scaled, self.width as usize, height, dither_mode, seed, bits);
        let g_u8 = crate::dither::dither_with_mode_bits(&g_scaled, self.width as usize, height, dither_mode, seed.wrapping_add(1), bits);
        let b_u8 = crate::dither::dither_with_mode_bits(&b_scaled, self.width as usize, height, dither_mode, seed.wrapping_add(2), bits);

        // Reconstruct Pixel4 array with dithered values
        let result: Vec<Pixel4> = (0..pixels.len())
            .map(|i| Pixel4::new(r_u8[i] as f32, g_u8[i] as f32, b_u8[i] as f32, pixels[i][3]))
            .collect();

        Ok(ImageBuffer::from_pixel4_vec(result, self.width, self.height))
    }

    /// Colorspace-aware dithering (requires Pixel4 in sRGB 0-255 range)
    /// Returns buffer with quantized values in sRGB 0-255 range
    /// perceptual_space: 0=CIE76, 1=CIE94, 2=CIEDE2000, 3=OkLab
    #[wasm_bindgen]
    pub fn dither_colorspace_aware(
        &self,
        bits: u8,
        mode: u8,
        perceptual_space: u8,
        seed: u32,
    ) -> Result<ImageBuffer, JsValue> {
        let pixels = self.as_pixel4()
            .ok_or_else(|| JsValue::from_str("dither requires pixel4 format"))?;

        let dither_mode = dither_mode_from_u8(mode);
        let space = perceptual_space_from_u8(perceptual_space);
        let height = self.height as usize;

        // Extract RGB channels
        let r_scaled: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
        let g_scaled: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
        let b_scaled: Vec<f32> = pixels.iter().map(|p| p[2]).collect();

        // Apply colorspace-aware dithering
        let (r_u8, g_u8, b_u8) = crate::dither_colorspace_aware::colorspace_aware_dither_rgb_with_mode(
            &r_scaled, &g_scaled, &b_scaled,
            self.width as usize, height,
            bits, bits, bits,
            space, dither_mode, seed
        );

        // Reconstruct Pixel4 array with dithered values
        let result: Vec<Pixel4> = (0..pixels.len())
            .map(|i| Pixel4::new(r_u8[i] as f32, g_u8[i] as f32, b_u8[i] as f32, pixels[i][3]))
            .collect();

        Ok(ImageBuffer::from_pixel4_vec(result, self.width, self.height))
    }

    // ========================================================================
    // Rescaling (returns new buffer with new dimensions)
    // ========================================================================

    /// Rescale image to new dimensions
    /// method: 0=Bilinear, 1=Lanczos3
    /// scale_mode: 0=Independent, 1=UniformWidth, 2=UniformHeight
    #[wasm_bindgen]
    pub fn rescale_to(
        &self,
        dst_width: u32,
        dst_height: u32,
        method: u8,
        scale_mode: u8,
    ) -> Result<ImageBuffer, JsValue> {
        let pixels = self.as_pixel4()
            .ok_or_else(|| JsValue::from_str("rescale requires pixel4 format"))?;

        let rescale_method = rescale_method_from_u8(method);
        let mode = scale_mode_from_u8(scale_mode);

        let result = crate::rescale::rescale(
            pixels,
            self.width as usize,
            self.height as usize,
            dst_width as usize,
            dst_height as usize,
            rescale_method,
            mode,
        );

        Ok(ImageBuffer::from_pixel4_vec(result, dst_width, dst_height))
    }

    // ========================================================================
    // Output format conversion
    // ========================================================================

    /// Convert Pixel4 (sRGB 0-255) to u8 RGB output
    #[wasm_bindgen]
    pub fn to_u8_rgb(&self) -> Result<ImageBuffer, JsValue> {
        let pixels = self.as_pixel4()
            .ok_or_else(|| JsValue::from_str("to_u8_rgb requires pixel4 format"))?;

        let data = crate::pixel::pixels_to_srgb_u8(pixels);
        Ok(ImageBuffer::from_u8_vec(data, self.width, self.height, 3))
    }

    /// Convert Pixel4 (sRGB 0-255) to u8 RGBA output
    #[wasm_bindgen]
    pub fn to_u8_rgba(&self) -> Result<ImageBuffer, JsValue> {
        let pixels = self.as_pixel4()
            .ok_or_else(|| JsValue::from_str("to_u8_rgba requires pixel4 format"))?;

        let data = crate::pixel::pixels_to_srgb_u8_rgba(pixels);
        Ok(ImageBuffer::from_u8_vec(data, self.width, self.height, 4))
    }

    /// Create Pixel4 buffer from u8 RGB input (values kept as 0-255)
    #[wasm_bindgen]
    pub fn from_u8_rgb_to_pixel4(&self) -> Result<ImageBuffer, JsValue> {
        let data = self.as_u8()
            .ok_or_else(|| JsValue::from_str("from_u8_rgb requires u8 format"))?;

        if self.channels != 3 {
            return Err(JsValue::from_str("Expected 3-channel RGB data"));
        }

        let pixels = crate::pixel::srgb_u8_to_pixels(data);
        Ok(ImageBuffer::from_pixel4_vec(pixels, self.width, self.height))
    }

    /// Create Pixel4 buffer from u8 RGBA input (values kept as 0-255)
    #[wasm_bindgen]
    pub fn from_u8_rgba_to_pixel4(&self) -> Result<ImageBuffer, JsValue> {
        let data = self.as_u8()
            .ok_or_else(|| JsValue::from_str("from_u8_rgba requires u8 format"))?;

        if self.channels != 4 {
            return Err(JsValue::from_str("Expected 4-channel RGBA data"));
        }

        let pixels = crate::pixel::srgb_u8_rgba_to_pixels(data);
        Ok(ImageBuffer::from_pixel4_vec(pixels, self.width, self.height))
    }
}

// ============================================================================
// GrayBuffer operations
// ============================================================================

#[wasm_bindgen]
impl GrayBuffer {
    /// Dither grayscale image (requires f32 in 0-255 range)
    /// Returns u8 output
    #[wasm_bindgen]
    pub fn dither(
        &self,
        bits: u8,
        mode: u8,
        seed: u32,
    ) -> Result<GrayBuffer, JsValue> {
        let data = self.as_f32()
            .ok_or_else(|| JsValue::from_str("dither requires f32 format"))?;

        let dither_mode = dither_mode_from_u8(mode);
        let height = self.height as usize;

        let result = crate::dither::dither_with_mode_bits(
            data, self.width as usize, height, dither_mode, seed, bits
        );

        Ok(GrayBuffer::from_u8_vec(result, self.width, self.height))
    }

    /// Colorspace-aware grayscale dithering
    /// Returns u8 output
    #[wasm_bindgen]
    pub fn dither_perceptual(
        &self,
        bits: u8,
        mode: u8,
        perceptual_space: u8,
        seed: u32,
    ) -> Result<GrayBuffer, JsValue> {
        let data = self.as_f32()
            .ok_or_else(|| JsValue::from_str("dither requires f32 format"))?;

        let dither_mode = dither_mode_from_u8(mode);
        let space = perceptual_space_from_u8(perceptual_space);
        let height = self.height as usize;

        let result = crate::dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode(
            data, self.width as usize, height, bits, space, dither_mode, seed
        );

        Ok(GrayBuffer::from_u8_vec(result, self.width, self.height))
    }

    /// Convert f32 (0-255) to u8 output
    #[wasm_bindgen]
    pub fn to_u8(&self) -> Result<GrayBuffer, JsValue> {
        let data = self.as_f32()
            .ok_or_else(|| JsValue::from_str("to_u8 requires f32 format"))?;

        let result: Vec<u8> = data.iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();

        Ok(GrayBuffer::from_u8_vec(result, self.width, self.height))
    }

    /// Create f32 buffer from u8 input (values converted to 0-255 f32)
    #[wasm_bindgen]
    pub fn u8_to_f32(&self) -> Result<GrayBuffer, JsValue> {
        let data = self.as_u8()
            .ok_or_else(|| JsValue::from_str("u8_to_f32 requires u8 format"))?;

        let result: Vec<f32> = data.iter().map(|&v| v as f32).collect();

        Ok(GrayBuffer::from_f32_vec(result, self.width, self.height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imagebuffer_u8() {
        let data = vec![0u8; 12]; // 2x2 RGB
        let buf = ImageBuffer::new_u8(data.clone(), 2, 2, 3).unwrap();
        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);
        assert_eq!(buf.channels(), 3);
        assert_eq!(buf.format(), "u8");
        assert_eq!(buf.get_u8_data(), data);
    }

    #[test]
    fn test_imagebuffer_f32() {
        let data = vec![0.5f32; 12]; // 2x2 RGB
        let buf = ImageBuffer::from_f32(data.clone(), 2, 2, 3).unwrap();
        assert_eq!(buf.format(), "f32");
        assert_eq!(buf.get_f32_data(), data);
    }

    #[test]
    fn test_imagebuffer_pixel4() {
        let buf = ImageBuffer::new_pixel4(2, 2);
        assert_eq!(buf.format(), "pixel4");
        assert_eq!(buf.pixel_count(), 4);

        let interleaved = buf.get_pixel4_interleaved();
        assert_eq!(interleaved.len(), 16); // 4 pixels * 4 channels
    }

    #[test]
    fn test_f32_to_pixel4_conversion() {
        let data = vec![1.0, 0.5, 0.25, 0.75, 0.5, 0.25]; // 2 RGB pixels
        let buf = ImageBuffer::from_f32(data, 2, 1, 3).unwrap();

        let pixel4_buf = buf.f32_rgb_to_pixel4().unwrap();
        assert_eq!(pixel4_buf.format(), "pixel4");

        let rgb = pixel4_buf.get_pixel4_rgb_interleaved();
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb[0], 1.0);
        assert_eq!(rgb[1], 0.5);
        assert_eq!(rgb[2], 0.25);
    }

    #[test]
    fn test_graybuffer() {
        let data = vec![128u8; 4]; // 2x2
        let buf = GrayBuffer::new_u8(data.clone(), 2, 2).unwrap();
        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);
        assert_eq!(buf.format(), "u8");
        assert_eq!(buf.get_u8_data(), data);
    }

    #[test]
    fn test_pixel4_alignment() {
        let buf = ImageBuffer::new_pixel4(100, 100);
        let pixels = buf.as_pixel4().unwrap();

        // Verify Pixel4 alignment
        for p in pixels.iter() {
            let ptr = p as *const Pixel4 as usize;
            assert_eq!(ptr % 16, 0, "Pixel4 should be 16-byte aligned");
        }
    }
}
