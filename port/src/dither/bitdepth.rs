/// Bit depth quantization utilities for dithering.
///
/// Provides:
/// - `bit_replicate`: Extend n-bit values to 8 bits
/// - `build_linear_lut`: Pre-computed sRGB to linear conversion LUT
/// - `QuantLevelParams`: Quantization level lookup tables

use crate::color::srgb_to_linear_single;

/// Extend n-bit value to 8 bits by repeating the bit pattern.
/// e.g., 3-bit value ABC becomes ABCABCAB
#[inline]
pub fn bit_replicate(value: u8, bits: u8) -> u8 {
    if bits == 8 {
        return value;
    }
    let mut result: u16 = 0;
    let mut shift = 8i8;
    while shift > 0 {
        shift -= bits as i8;
        if shift >= 0 {
            result |= (value as u16) << shift;
        } else {
            // Partial bits at the end
            result |= (value as u16) >> (-shift);
        }
    }
    result as u8
}

/// Pre-computed LUT for sRGB to linear conversion (256 entries).
/// Maps each sRGB byte value (0-255) to its linear RGB equivalent (0.0-1.0).
pub fn build_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = srgb_to_linear_single(i as f32 / 255.0);
    }
    lut
}

/// Quantization level parameters for n-bit output.
///
/// Maps 8-bit sRGB input values to quantization level indices,
/// supporting floor/ceil lookup for error diffusion dithering.
/// Uses bit replication to extend n-bit levels to full 8-bit range.
pub struct QuantLevelParams {
    /// Number of quantization levels (2^bits)
    pub num_levels: usize,
    /// Level index -> extended sRGB value (0-255)
    pub level_values: Vec<u8>,
    /// sRGB value -> floor level index
    lut_floor_level: [u8; 256],
    /// sRGB value -> ceil level index
    lut_ceil_level: [u8; 256],
}

impl QuantLevelParams {
    /// Create quantization parameters for given bit depth (1-8).
    pub fn new(bits: u8) -> Self {
        debug_assert!(bits >= 1 && bits <= 8, "bits must be 1-8");
        let num_levels = 1usize << bits;
        let max_idx = num_levels - 1;
        let shift = 8 - bits;

        let level_values: Vec<u8> = (0..num_levels)
            .map(|l| bit_replicate(l as u8, bits))
            .collect();

        let mut lut_floor_level = [0u8; 256];
        let mut lut_ceil_level = [0u8; 256];

        for v in 0..256u16 {
            let trunc_idx = (v as u8 >> shift) as usize;
            let trunc_val = level_values[trunc_idx];

            let (floor_idx, ceil_idx) = if trunc_val == v as u8 {
                (trunc_idx, trunc_idx)
            } else if trunc_val < v as u8 {
                let ceil = if trunc_idx < max_idx {
                    trunc_idx + 1
                } else {
                    trunc_idx
                };
                (trunc_idx, ceil)
            } else {
                let floor = if trunc_idx > 0 { trunc_idx - 1 } else { trunc_idx };
                (floor, trunc_idx)
            };

            lut_floor_level[v as usize] = floor_idx as u8;
            lut_ceil_level[v as usize] = ceil_idx as u8;
        }

        Self {
            num_levels,
            level_values,
            lut_floor_level,
            lut_ceil_level,
        }
    }

    /// Get the floor level index for an sRGB value.
    #[inline]
    pub fn floor_level(&self, srgb_value: u8) -> usize {
        self.lut_floor_level[srgb_value as usize] as usize
    }

    /// Get the ceil level index for an sRGB value.
    #[inline]
    pub fn ceil_level(&self, srgb_value: u8) -> usize {
        self.lut_ceil_level[srgb_value as usize] as usize
    }

    /// Get the sRGB value for a level index.
    #[inline]
    pub fn level_to_srgb(&self, level: usize) -> u8 {
        self.level_values[level]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_replicate() {
        // 1-bit: 0 -> 0x00, 1 -> 0xFF
        assert_eq!(bit_replicate(0, 1), 0x00);
        assert_eq!(bit_replicate(1, 1), 0xFF);

        // 2-bit: 0 -> 0x00, 1 -> 0x55, 2 -> 0xAA, 3 -> 0xFF
        assert_eq!(bit_replicate(0, 2), 0x00);
        assert_eq!(bit_replicate(1, 2), 0x55);
        assert_eq!(bit_replicate(2, 2), 0xAA);
        assert_eq!(bit_replicate(3, 2), 0xFF);

        // 4-bit: 0 -> 0x00, 15 -> 0xFF
        assert_eq!(bit_replicate(0, 4), 0x00);
        assert_eq!(bit_replicate(15, 4), 0xFF);
        assert_eq!(bit_replicate(8, 4), 0x88);

        // 8-bit: identity
        assert_eq!(bit_replicate(0, 8), 0);
        assert_eq!(bit_replicate(128, 8), 128);
        assert_eq!(bit_replicate(255, 8), 255);
    }

    #[test]
    fn test_quant_level_params() {
        // 2-bit quantization: 4 levels (0x00, 0x55, 0xAA, 0xFF)
        let params = QuantLevelParams::new(2);
        assert_eq!(params.num_levels, 4);
        assert_eq!(params.level_values, vec![0x00, 0x55, 0xAA, 0xFF]);

        // Check floor/ceil for value between levels
        assert_eq!(params.floor_level(0x60), 1); // floor of 0x60 is level 1 (0x55)
        assert_eq!(params.ceil_level(0x60), 2);  // ceil of 0x60 is level 2 (0xAA)

        // Check exact match
        assert_eq!(params.floor_level(0x55), 1);
        assert_eq!(params.ceil_level(0x55), 1);
    }

    #[test]
    fn test_build_linear_lut() {
        let lut = build_linear_lut();
        assert_eq!(lut.len(), 256);
        assert_eq!(lut[0], 0.0);
        assert!((lut[255] - 1.0).abs() < 1e-6);
        // Mid-gray should be around 0.2 (sRGB 128 -> ~0.215)
        assert!(lut[128] > 0.2 && lut[128] < 0.23);
    }
}
