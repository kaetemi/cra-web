/// Opaque buffer types for WASM
///
/// These are simple data containers without any semantic meaning.
/// Width/height/channels are managed by the caller, not stored here.
/// These exist purely to avoid copying data across the WASM boundary.

use wasm_bindgen::prelude::*;
use crate::pixel::Pixel4;

// ============================================================================
// BufferF32x4 - Pixel4 (SIMD-friendly, 16-byte aligned) storage
// ============================================================================

#[wasm_bindgen]
pub struct BufferF32x4(Vec<Pixel4>);

#[wasm_bindgen]
impl BufferF32x4 {
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[wasm_bindgen]
    pub fn clone_buffer(&self) -> BufferF32x4 {
        BufferF32x4(self.0.clone())
    }
}

impl BufferF32x4 {
    pub fn new(data: Vec<Pixel4>) -> Self {
        BufferF32x4(data)
    }

    pub fn as_slice(&self) -> &[Pixel4] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [Pixel4] {
        &mut self.0
    }

    pub fn into_inner(self) -> Vec<Pixel4> {
        self.0
    }
}

// ============================================================================
// BufferF32 - f32 channel data storage
// ============================================================================

#[wasm_bindgen]
pub struct BufferF32(Vec<f32>);

#[wasm_bindgen]
impl BufferF32 {
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[wasm_bindgen]
    pub fn clone_buffer(&self) -> BufferF32 {
        BufferF32(self.0.clone())
    }
}

impl BufferF32 {
    pub fn new(data: Vec<f32>) -> Self {
        BufferF32(data)
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
}

// ============================================================================
// BufferU8 - u8 byte data storage
// ============================================================================

#[wasm_bindgen]
pub struct BufferU8(Vec<u8>);

#[wasm_bindgen]
impl BufferU8 {
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[wasm_bindgen]
    pub fn clone_buffer(&self) -> BufferU8 {
        BufferU8(self.0.clone())
    }

    /// Extract as Vec<u8> for JS consumption
    #[wasm_bindgen]
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.clone()
    }
}

impl BufferU8 {
    pub fn new(data: Vec<u8>) -> Self {
        BufferU8(data)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.0
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_f32x4() {
        let pixels = vec![Pixel4::new(1.0, 0.5, 0.25, 0.0); 4];
        let buf = BufferF32x4::new(pixels);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.as_slice()[0][0], 1.0);
    }

    #[test]
    fn test_buffer_f32() {
        let data = vec![0.5f32; 12];
        let buf = BufferF32::new(data);
        assert_eq!(buf.len(), 12);
        assert_eq!(buf.as_slice()[0], 0.5);
    }

    #[test]
    fn test_buffer_u8() {
        let data = vec![128u8; 10];
        let buf = BufferU8::new(data);
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.as_slice()[0], 128);
    }

    #[test]
    fn test_pixel4_alignment() {
        let pixels = vec![Pixel4::default(); 100];
        let buf = BufferF32x4::new(pixels);
        for p in buf.as_slice().iter() {
            let ptr = p as *const Pixel4 as usize;
            assert_eq!(ptr % 16, 0, "Pixel4 should be 16-byte aligned");
        }
    }
}
