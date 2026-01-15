/// Opaque image buffer types for WASM
///
/// These buffers hold image data without copying at every function boundary.
/// ImageBuffer can contain u8, f32, or Pixel4 data depending on the processing stage.
/// All processing happens through separate WASM functions that operate on these buffers.

use wasm_bindgen::prelude::*;
use crate::pixel::Pixel4;

// ============================================================================
// Pixel data storage
// ============================================================================

/// Pixel data storage variants
#[derive(Clone)]
pub enum PixelData {
    /// Raw u8 bytes (sRGB, etc.)
    U8(Vec<u8>),
    /// f32 channel data (linear RGB, normalized, etc.)
    F32(Vec<f32>),
    /// SIMD-friendly Pixel4 array (16-byte aligned)
    Pixel4(Vec<Pixel4>),
}

/// Grayscale data storage variants
#[derive(Clone)]
pub enum GrayData {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

// ============================================================================
// ImageBuffer - opaque RGB/RGBA image container
// ============================================================================

/// Opaque image buffer handle for WASM
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
    // Constructors
    // ========================================================================

    /// Create buffer from u8 array
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Result<ImageBuffer, JsValue> {
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

    /// Create buffer from f32 array
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

    /// Get the buffer format: "u8", "f32", or "pixel4"
    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        match &self.data {
            PixelData::U8(_) => "u8".to_string(),
            PixelData::F32(_) => "f32".to_string(),
            PixelData::Pixel4(_) => "pixel4".to_string(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    // ========================================================================
    // Data extraction (for final output to JS)
    // ========================================================================

    /// Get u8 data (empty if not u8 format)
    #[wasm_bindgen]
    pub fn get_u8(&self) -> Vec<u8> {
        match &self.data {
            PixelData::U8(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Get f32 data (empty if not f32 format)
    #[wasm_bindgen]
    pub fn get_f32(&self) -> Vec<f32> {
        match &self.data {
            PixelData::F32(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Get Pixel4 as interleaved RGBA f32 (empty if not pixel4 format)
    #[wasm_bindgen]
    pub fn get_rgba(&self) -> Vec<f32> {
        match &self.data {
            PixelData::Pixel4(pixels) => {
                pixels.iter().flat_map(|p| [p[0], p[1], p[2], p[3]]).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Get Pixel4 as interleaved RGB f32, dropping alpha (empty if not pixel4)
    #[wasm_bindgen]
    pub fn get_rgb(&self) -> Vec<f32> {
        match &self.data {
            PixelData::Pixel4(pixels) => {
                pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Clone the buffer
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
// Internal Rust API (not exported to WASM)
// ============================================================================

impl ImageBuffer {
    pub fn as_pixel4(&self) -> Option<&Vec<Pixel4>> {
        match &self.data {
            PixelData::Pixel4(pixels) => Some(pixels),
            _ => None,
        }
    }

    pub fn as_pixel4_mut(&mut self) -> Option<&mut Vec<Pixel4>> {
        match &mut self.data {
            PixelData::Pixel4(pixels) => Some(pixels),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match &self.data {
            PixelData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut Vec<f32>> {
        match &mut self.data {
            PixelData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match &self.data {
            PixelData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut Vec<u8>> {
        match &mut self.data {
            PixelData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn from_pixel4(pixels: Vec<Pixel4>, width: u32, height: u32) -> Self {
        ImageBuffer {
            data: PixelData::Pixel4(pixels),
            width,
            height,
            channels: 4,
        }
    }

    pub fn from_f32_internal(data: Vec<f32>, width: u32, height: u32, channels: u8) -> Self {
        ImageBuffer {
            data: PixelData::F32(data),
            width,
            height,
            channels,
        }
    }

    pub fn from_u8_internal(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        ImageBuffer {
            data: PixelData::U8(data),
            width,
            height,
            channels,
        }
    }

    pub fn take_pixel4(self) -> Option<Vec<Pixel4>> {
        match self.data {
            PixelData::Pixel4(pixels) => Some(pixels),
            _ => None,
        }
    }

    pub fn take_f32(self) -> Option<Vec<f32>> {
        match self.data {
            PixelData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn take_u8(self) -> Option<Vec<u8>> {
        match self.data {
            PixelData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn set_pixel4(&mut self, pixels: Vec<Pixel4>) {
        self.data = PixelData::Pixel4(pixels);
        self.channels = 4;
    }

    pub fn set_f32(&mut self, data: Vec<f32>, channels: u8) {
        self.data = PixelData::F32(data);
        self.channels = channels;
    }

    pub fn set_u8(&mut self, data: Vec<u8>, channels: u8) {
        self.data = PixelData::U8(data);
        self.channels = channels;
    }
}

// ============================================================================
// GrayBuffer - opaque grayscale image container
// ============================================================================

#[wasm_bindgen]
pub struct GrayBuffer {
    data: GrayData,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl GrayBuffer {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Result<GrayBuffer, JsValue> {
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

    #[wasm_bindgen]
    pub fn get_u8(&self) -> Vec<u8> {
        match &self.data {
            GrayData::U8(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub fn get_f32(&self) -> Vec<f32> {
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

// Internal Rust API for GrayBuffer
impl GrayBuffer {
    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match &self.data {
            GrayData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut Vec<f32>> {
        match &mut self.data {
            GrayData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match &self.data {
            GrayData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut Vec<u8>> {
        match &mut self.data {
            GrayData::U8(data) => Some(data),
            _ => None,
        }
    }

    pub fn from_f32_internal(data: Vec<f32>, width: u32, height: u32) -> Self {
        GrayBuffer {
            data: GrayData::F32(data),
            width,
            height,
        }
    }

    pub fn from_u8_internal(data: Vec<u8>, width: u32, height: u32) -> Self {
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

    pub fn set_f32(&mut self, data: Vec<f32>) {
        self.data = GrayData::F32(data);
    }

    pub fn set_u8(&mut self, data: Vec<u8>) {
        self.data = GrayData::U8(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imagebuffer_u8() {
        let data = vec![0u8; 12]; // 2x2 RGB
        let buf = ImageBuffer::new(data.clone(), 2, 2, 3).unwrap();
        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);
        assert_eq!(buf.channels(), 3);
        assert_eq!(buf.format(), "u8");
        assert_eq!(buf.get_u8(), data);
    }

    #[test]
    fn test_imagebuffer_f32() {
        let data = vec![0.5f32; 12]; // 2x2 RGB
        let buf = ImageBuffer::from_f32(data.clone(), 2, 2, 3).unwrap();
        assert_eq!(buf.format(), "f32");
        assert_eq!(buf.get_f32(), data);
    }

    #[test]
    fn test_imagebuffer_pixel4() {
        let pixels = vec![Pixel4::new(1.0, 0.5, 0.25, 0.0); 4];
        let buf = ImageBuffer::from_pixel4(pixels, 2, 2);
        assert_eq!(buf.format(), "pixel4");
        assert_eq!(buf.pixel_count(), 4);

        let rgba = buf.get_rgba();
        assert_eq!(rgba.len(), 16);
        assert_eq!(rgba[0], 1.0);
        assert_eq!(rgba[1], 0.5);
        assert_eq!(rgba[2], 0.25);
    }

    #[test]
    fn test_graybuffer() {
        let data = vec![128u8; 4]; // 2x2
        let buf = GrayBuffer::new(data.clone(), 2, 2).unwrap();
        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);
        assert_eq!(buf.format(), "u8");
        assert_eq!(buf.get_u8(), data);
    }

    #[test]
    fn test_pixel4_alignment() {
        let pixels = vec![Pixel4::default(); 100];
        let buf = ImageBuffer::from_pixel4(pixels, 10, 10);
        let pixels = buf.as_pixel4().unwrap();

        for p in pixels.iter() {
            let ptr = p as *const Pixel4 as usize;
            assert_eq!(ptr % 16, 0, "Pixel4 should be 16-byte aligned");
        }
    }
}
