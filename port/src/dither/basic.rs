/// Error diffusion dithering implementations.
/// Supports Floyd-Steinberg, Jarvis-Judice-Ninke, and Mixed algorithms.

// Re-export color space aware dithering from dedicated module
pub use super::rgb::{
    colorspace_aware_dither_rgb,
    colorspace_aware_dither_rgb_with_mode,
};

// Re-export common types for backwards compatibility
pub use super::common::DitherMode;

// Import shared utilities from common
use super::common::{
    apply_single_channel_kernel, bit_replicate, lowbias32, lowbias32_old, wang_hash, FloydSteinberg,
    JarvisJudiceNinke, NoneKernel, Ostromoukhov, SingleChannelKernel,
};

/// Quantization parameters for reduced bit depth dithering.
/// Pre-computed to avoid repeated calculations in the hot loop.
#[derive(Debug, Clone, Copy)]
struct QuantParams {
    /// Number of bits (1-8)
    #[allow(dead_code)]
    bits: u8,
    /// Maximum level value (2^bits - 1), e.g., 31 for 5 bits, 255 for 8 bits
    #[allow(dead_code)]
    max_level: f32,
    /// LUT mapping each uint8 input to the nearest bit-replicated level
    lut_nearest: [f32; 256],
    /// LUT mapping each uint8 input to the floor (largest level <= input)
    lut_floor: [f32; 256],
    /// LUT mapping each uint8 input to the ceil (smallest level >= input)
    lut_ceil: [f32; 256],
}

impl QuantParams {
    /// Create quantization parameters for given bit depth.
    /// Bit depth of 8 gives standard 256-level quantization (round to integers).
    /// Lower values give fewer levels, e.g., 3 bits = 8 levels.
    #[inline]
    fn new(bits: u8) -> Self {
        debug_assert!(bits >= 1 && bits <= 8, "bits must be 1-8");
        let levels = 1u32 << bits;
        let max_level = (levels - 1) as f32;
        let max_idx = (levels - 1) as usize;
        let shift = 8 - bits;

        // Pre-compute bit-replicated values for each level
        let level_values: Vec<u8> = (0..levels)
            .map(|l| Self::bit_replicate_local(l as u8, bits))
            .collect();

        let mut lut_nearest = [0.0f32; 256];
        let mut lut_floor = [0.0f32; 256];
        let mut lut_ceil = [0.0f32; 256];

        for v in 0..256u16 {
            // Use bit truncation to find a nearby level
            let trunc_idx = (v as u8 >> shift) as usize;
            let trunc_val = level_values[trunc_idx];

            let (floor_val, ceil_val) = if trunc_val == v as u8 {
                (trunc_val, trunc_val)
            } else if trunc_val < v as u8 {
                // trunc is floor, ceil is trunc+1
                let ceil = if trunc_idx < max_idx { level_values[trunc_idx + 1] } else { trunc_val };
                (trunc_val, ceil)
            } else {
                // trunc is ceil, floor is trunc-1
                let floor = if trunc_idx > 0 { level_values[trunc_idx - 1] } else { trunc_val };
                (floor, trunc_val)
            };

            // Nearest: closer of floor/ceil (ties go to floor)
            let dist_floor = v.abs_diff(floor_val as u16);
            let dist_ceil = v.abs_diff(ceil_val as u16);
            let nearest_val = if dist_floor <= dist_ceil { floor_val } else { ceil_val };

            lut_floor[v as usize] = floor_val as f32;
            lut_ceil[v as usize] = ceil_val as f32;
            lut_nearest[v as usize] = nearest_val as f32;
        }

        Self { bits, max_level, lut_nearest, lut_floor, lut_ceil }
    }

    /// Extend n-bit value to 8 bits by repeating the bit pattern.
    /// e.g., 3-bit value ABC becomes ABCABCAB
    /// Delegates to the shared bit_replicate function from dither_common.
    #[inline]
    fn bit_replicate_local(value: u8, bits: u8) -> u8 {
        bit_replicate(value, bits)
    }

    /// Quantize a value to the nearest level and return the bit-replicated value.
    /// Input and output are in 0-255 range.
    #[inline]
    fn quantize(&self, value: f32) -> f32 {
        self.lut_nearest[value.round().clamp(0.0, 255.0) as usize]
    }

    /// Quantize a value to the floor level (largest level <= input).
    #[inline]
    #[allow(dead_code)]
    fn quantize_floor(&self, value: f32) -> f32 {
        self.lut_floor[value.round().clamp(0.0, 255.0) as usize]
    }

    /// Quantize a value to the ceil level (smallest level >= input).
    #[inline]
    #[allow(dead_code)]
    fn quantize_ceil(&self, value: f32) -> f32 {
        self.lut_ceil[value.round().clamp(0.0, 255.0) as usize]
    }

    /// Create default 8-bit quantization (standard rounding to integers)
    #[inline]
    fn default_8bit() -> Self {
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            lut[i] = i as f32;
        }
        Self {
            bits: 8,
            max_level: 255.0,
            lut_nearest: lut,
            lut_floor: lut,
            lut_ceil: lut,
        }
    }
}

// ============================================================================
// Buffer helpers
// ============================================================================

/// Create a padded buffer with edge seeding for normalized edge dithering.
///
/// Buffer structure:
/// ```text
/// [overshoot] [seeding] [real image] [seeding] [overshoot]
/// ```
///
/// - Seeding areas are filled with duplicated edge pixels
/// - Overshoot areas are zeroed (catch error diffusion)
/// - Real image data is copied to the center
#[inline]
fn create_seeded_buffer(
    img: &[f32],
    width: usize,
    height: usize,
    reach: usize,
) -> Vec<Vec<f32>> {
    // Total padding: overshoot (reach) + seeding (reach) on each side horizontally
    // Top: seeding only (reach), Bottom: overshoot only (reach)
    let total_left = reach * 2;
    let total_right = reach * 2;
    let total_top = reach;
    let total_bottom = reach;

    let buf_width = total_left + width + total_right;
    let buf_height = total_top + height + total_bottom;
    let mut buf = vec![vec![0.0f32; buf_width]; buf_height];

    // Copy real image data
    for y in 0..height {
        for x in 0..width {
            buf[y + total_top][x + total_left] = img[y * width + x];
        }
    }

    // Fill seeding areas with duplicated edge pixels
    // Seeding columns start at `reach` (after overshoot) and span `reach` columns
    let seed_left_start = reach;  // after left overshoot
    let seed_right_start = total_left + width;  // after real image

    // Fill left seeding columns (duplicate first column of real image)
    for y in 0..height {
        let first_col_val = img[y * width];  // first pixel of this row
        for sx in 0..reach {
            buf[y + total_top][seed_left_start + sx] = first_col_val;
        }
    }

    // Fill right seeding columns (duplicate last column of real image)
    for y in 0..height {
        let last_col_val = img[y * width + (width - 1)];  // last pixel of this row
        for sx in 0..reach {
            buf[y + total_top][seed_right_start + sx] = last_col_val;
        }
    }

    // Fill top seeding rows (duplicate first row, including seeding columns)
    if reach > 0 {
        let first_row_left_seed = img[0];  // first pixel of first row
        let first_row_right_seed = img[width - 1];  // last pixel of first row

        for sy in 0..reach {
            // Left seeding area of top seeding row
            for sx in 0..reach {
                buf[sy][seed_left_start + sx] = first_row_left_seed;
            }
            // Real image area of top seeding row (copy first row)
            for x in 0..width {
                buf[sy][x + total_left] = img[x];
            }
            // Right seeding area of top seeding row
            for sx in 0..reach {
                buf[sy][seed_right_start + sx] = first_row_right_seed;
            }
        }
    }

    buf
}

/// Extract real pixels from seeded buffer, clamp, and convert to u8.
#[inline]
fn extract_result_seeded(
    buf: &[Vec<f32>],
    width: usize,
    height: usize,
    reach: usize,
) -> Vec<u8> {
    let total_left = reach * 2;
    let total_top = reach;

    let mut result = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            result.push(buf[y + total_top][x + total_left].clamp(0.0, 255.0).round() as u8);
        }
    }
    result
}

// ============================================================================
// Generic dithering implementations
// ============================================================================

/// Generic standard (left-to-right) dithering with any kernel.
/// Processes seeding rows/columns plus real image, all left-to-right.
fn dither_standard<K: SingleChannelKernel>(
    img: &[f32],
    width: usize,
    height: usize,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let reach = K::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    // Processing area: seeding rows + real rows, seeding cols + real cols + seeding cols
    let process_height = reach + height;  // top seeding + real
    let process_width = reach + width + reach;  // left seeding + real + right seeding
    let bx_start = reach;  // skip left overshoot

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let old = buf[y][bx];
            let new = quant.quantize(old);
            buf[y][bx] = new;
            let err = old - new;
            K::apply_ltr(&mut buf, bx, y, err, old);
        }
        if let Some(ref mut cb) = progress {
            // Progress based on real image rows (after seeding)
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Generic serpentine dithering with any kernel.
/// Alternates scan direction each row to reduce diagonal banding.
/// Processes seeding rows/columns plus real image.
fn dither_serpentine<K: SingleChannelKernel>(
    img: &[f32],
    width: usize,
    height: usize,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let reach = K::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    // Processing area: seeding rows + real rows, seeding cols + real cols + seeding cols
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;  // skip left overshoot

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let old = buf[y][bx];
                let new = quant.quantize(old);
                buf[y][bx] = new;
                let err = old - new;
                K::apply_rtl(&mut buf, bx, y, err, old);
            }
        } else {
            // Left-to-right on even rows
            for px in 0..process_width {
                let bx = bx_start + px;
                let old = buf[y][bx];
                let new = quant.quantize(old);
                buf[y][bx] = new;
                let err = old - new;
                K::apply_ltr(&mut buf, bx, y, err, old);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

// ============================================================================
// Mixed dithering (runtime kernel selection)
// ============================================================================

/// Process a single pixel: quantize and return error.
#[inline]
fn process_pixel(buf: &mut [Vec<f32>], bx: usize, y: usize, quant: &QuantParams) -> f32 {
    let old = buf[y][bx];
    let new = quant.quantize(old);
    buf[y][bx] = new;
    old - new
}

/// Apply kernel based on runtime selection.
#[inline]
fn apply_mixed_kernel(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, use_jjn: bool, is_rtl: bool) {
    apply_single_channel_kernel(buf, bx, y, err, use_jjn, is_rtl);
}

/// Mixed dithering with standard left-to-right scanning.
/// Randomly selects between Floyd-Steinberg and Jarvis-Judice-Ninke kernels per pixel.
/// Uses lowbias32 hash for better spectral properties.
fn mixed_dither_standard(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32(seed);
    // Use JJN reach (larger) to accommodate both kernels
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    // Processing area: seeding rows + real rows, seeding cols + real cols + seeding cols
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let err = process_pixel(&mut buf, bx, y, &quant);
            // Use original image coordinates for hash (adjusted for seeding offset)
            let img_x = px.saturating_sub(reach);
            let img_y = y.saturating_sub(reach);
            let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            let use_jjn = pixel_hash & 1 == 1;
            apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Mixed dithering with standard scanning using legacy wang_hash.
/// Kept for comparison testing; prefer mixed_dither_standard for production.
fn mixed_dither_wang_standard(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let err = process_pixel(&mut buf, bx, y, &quant);
            let img_x = px.saturating_sub(reach);
            let img_y = y.saturating_sub(reach);
            let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            let use_jjn = pixel_hash & 1 == 1;
            apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Mixed dithering with serpentine scanning.
/// Randomly selects kernel per pixel, alternates scan direction each row.
/// Uses lowbias32 hash for better spectral properties.
fn mixed_dither_serpentine(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32(seed);
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, true);
            }
        } else {
            // Left-to-right on even rows
            for px in 0..process_width {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Mixed dithering with serpentine scanning using legacy wang_hash.
/// Kept for comparison testing; prefer mixed_dither_serpentine for production.
fn mixed_dither_wang_serpentine(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = wang_hash((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Mixed dithering with standard scanning using original lowbias32_old.
/// Kept for comparison testing; default boon uses improved lowbias32.
fn mixed_dither_lowbias_old_standard(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32_old(seed);
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let err = process_pixel(&mut buf, bx, y, &quant);
            let img_x = px.saturating_sub(reach);
            let img_y = y.saturating_sub(reach);
            let pixel_hash = lowbias32_old((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            let use_jjn = pixel_hash & 1 == 1;
            apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Mixed dithering with serpentine scanning using original lowbias32_old.
/// Kept for comparison testing; default boon uses improved lowbias32.
fn mixed_dither_lowbias_old_serpentine(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32_old(seed);
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = lowbias32_old((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let pixel_hash = lowbias32_old((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Mixed dithering with random scan direction per row.
/// Randomly selects kernel per pixel AND scan direction per row.
/// Uses lowbias32 hash for better spectral properties.
fn mixed_dither_random(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32(seed);
    let reach = <JarvisJudiceNinke as SingleChannelKernel>::REACH;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        let img_y = y.saturating_sub(reach);
        let row_hash = lowbias32((img_y as u32) ^ hashed_seed);
        if row_hash & 1 == 1 {
            // Right-to-left (randomly selected)
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, true);
            }
        } else {
            // Left-to-right (randomly selected)
            for px in 0..process_width {
                let bx = bx_start + px;
                let err = process_pixel(&mut buf, bx, y, &quant);
                let img_x = px.saturating_sub(reach);
                let pixel_hash = lowbias32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                let use_jjn = pixel_hash & 1 == 1;
                apply_mixed_kernel(&mut buf, bx, y, err, use_jjn, false);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

// ============================================================================
// Zhou-Fang dithering (threshold modulation + variable coefficients)
// ============================================================================

use super::kernels::{apply_zhou_fang_ltr, apply_zhou_fang_rtl, zhou_fang_threshold};

/// Zhou-Fang dithering with standard left-to-right scanning.
/// Uses threshold modulation to break up "worm" patterns in mid-tones.
fn zhou_fang_standard(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);
    // Same reach as Floyd-Steinberg (3 neighbors)
    let reach = 1usize;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let old = buf[y][bx];

            // Get input level for threshold modulation
            let level = ((old + 0.5) as i32).clamp(0, 255);

            // Compute modulated threshold based on position
            let img_x = px.saturating_sub(reach);
            let img_y = y.saturating_sub(reach);
            let threshold = zhou_fang_threshold(level, img_x, img_y, hashed_seed);

            // Quantize using modulated threshold between floor and ceil
            let floor_val = quant.quantize_floor(old);
            let ceil_val = quant.quantize_ceil(old);
            let new = if floor_val == ceil_val {
                floor_val
            } else {
                // Interpolation position between floor and ceil
                let frac = (old - floor_val) / (ceil_val - floor_val);
                if frac > threshold { ceil_val } else { floor_val }
            };
            buf[y][bx] = new;

            let err = old - new;
            apply_zhou_fang_ltr(&mut buf, bx, y, err, level);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Zhou-Fang dithering with serpentine scanning.
/// Uses threshold modulation with alternating scan direction each row.
fn zhou_fang_serpentine(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = wang_hash(seed);
    let reach = 1usize;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let old = buf[y][bx];
                let level = ((old + 0.5) as i32).clamp(0, 255);

                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let threshold = zhou_fang_threshold(level, img_x, img_y, hashed_seed);

                let floor_val = quant.quantize_floor(old);
                let ceil_val = quant.quantize_ceil(old);
                let new = if floor_val == ceil_val {
                    floor_val
                } else {
                    let frac = (old - floor_val) / (ceil_val - floor_val);
                    if frac > threshold { ceil_val } else { floor_val }
                };
                buf[y][bx] = new;

                let err = old - new;
                apply_zhou_fang_rtl(&mut buf, bx, y, err, level);
            }
        } else {
            // Left-to-right on even rows
            for px in 0..process_width {
                let bx = bx_start + px;
                let old = buf[y][bx];
                let level = ((old + 0.5) as i32).clamp(0, 255);

                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let threshold = zhou_fang_threshold(level, img_x, img_y, hashed_seed);

                let floor_val = quant.quantize_floor(old);
                let ceil_val = quant.quantize_ceil(old);
                let new = if floor_val == ceil_val {
                    floor_val
                } else {
                    let frac = (old - floor_val) / (ceil_val - floor_val);
                    if frac > threshold { ceil_val } else { floor_val }
                };
                buf[y][bx] = new;

                let err = old - new;
                apply_zhou_fang_ltr(&mut buf, bx, y, err, level);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

// ============================================================================
// Ulichney dithering (Floyd-Steinberg with random threshold modulation)
// ============================================================================

/// Generate a random threshold in [0, 1] for Ulichney dithering.
/// Uses lowbias32 hash for position-based randomization.
#[inline]
fn ulichney_threshold(x: usize, y: usize, seed: u32) -> f32 {
    let hash = lowbias32((x as u32) ^ ((y as u32) << 16) ^ seed);
    // Map to [0, 1] range (this gives uniform distribution)
    (hash as f32) / (u32::MAX as f32)
}

/// Quantize with threshold modulation for Ulichney dithering.
/// For 8-bit output, uses floor/ceil of the f32 value directly.
/// threshold should be in [0, 1] range.
#[inline]
fn ulichney_quantize(quant: &QuantParams, value: f32, threshold: f32) -> f32 {
    // For 8-bit, we need to use the actual f32 floor/ceil
    if quant.bits == 8 {
        let floor_val = value.floor();
        let ceil_val = value.ceil();
        if floor_val == ceil_val {
            floor_val.clamp(0.0, 255.0)
        } else {
            let frac = value - floor_val;
            if frac > threshold {
                ceil_val.clamp(0.0, 255.0)
            } else {
                floor_val.clamp(0.0, 255.0)
            }
        }
    } else {
        // For reduced bit depths, use the LUT-based floor/ceil
        let floor_val = quant.quantize_floor(value);
        let ceil_val = quant.quantize_ceil(value);
        if floor_val == ceil_val {
            floor_val
        } else {
            let frac = (value - floor_val) / (ceil_val - floor_val);
            if frac > threshold { ceil_val } else { floor_val }
        }
    }
}

/// Ulichney dithering with standard left-to-right scanning.
/// Uses Floyd-Steinberg kernel with random threshold modulation (±0.5).
/// The threshold for choosing between floor and ceil is randomized per-pixel.
fn ulichney_standard(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32(seed);
    // Same reach as Floyd-Steinberg
    let reach = 1usize;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let old = buf[y][bx];

            // Compute randomized threshold based on position
            let img_x = px.saturating_sub(reach);
            let img_y = y.saturating_sub(reach);
            let threshold = ulichney_threshold(img_x, img_y, hashed_seed);

            // Quantize using modulated threshold
            let new = ulichney_quantize(&quant, old, threshold);
            buf[y][bx] = new;

            let err = old - new;
            FloydSteinberg::apply_ltr(&mut buf, bx, y, err, old);
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

/// Ulichney dithering with serpentine scanning.
/// Uses Floyd-Steinberg kernel with random threshold modulation (±0.5).
fn ulichney_serpentine(
    img: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    quant: QuantParams,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let hashed_seed = lowbias32(seed);
    let reach = 1usize;
    let mut buf = create_seeded_buffer(img, width, height, reach);

    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left on odd rows
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let old = buf[y][bx];

                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let threshold = ulichney_threshold(img_x, img_y, hashed_seed);

                let new = ulichney_quantize(&quant, old, threshold);
                buf[y][bx] = new;

                let err = old - new;
                FloydSteinberg::apply_rtl(&mut buf, bx, y, err, old);
            }
        } else {
            // Left-to-right on even rows
            for px in 0..process_width {
                let bx = bx_start + px;
                let old = buf[y][bx];

                let img_x = px.saturating_sub(reach);
                let img_y = y.saturating_sub(reach);
                let threshold = ulichney_threshold(img_x, img_y, hashed_seed);

                let new = ulichney_quantize(&quant, old, threshold);
                buf[y][bx] = new;

                let err = old - new;
                FloydSteinberg::apply_ltr(&mut buf, bx, y, err, old);
            }
        }
        if let Some(ref mut cb) = progress {
            if y >= reach {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }

    extract_result_seeded(&buf, width, height, reach)
}

// ============================================================================
// Public API
// ============================================================================

/// Floyd-Steinberg dithering with linear buffer and overflow padding.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///
/// Returns:
///     flat array of u8 values, same length as input
pub fn floyd_steinberg_dither(img: &[f32], width: usize, height: usize) -> Vec<u8> {
    dither_with_mode(img, width, height, DitherMode::Standard, 0)
}

/// Floyd-Steinberg dithering with configurable bit depth.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     bits: output bit depth (1-8), controls number of quantization levels
///
/// Returns:
///     flat array of u8 values, same length as input.
///     Lower bit depths are represented in uint8 by extending bits through repetition.
pub fn floyd_steinberg_dither_bits(img: &[f32], width: usize, height: usize, bits: u8) -> Vec<u8> {
    dither_with_mode_bits(img, width, height, DitherMode::Standard, 0, bits, None)
}

/// Dithering with selectable algorithm and scanning mode (standard 8-bit depth).
///
/// This is the standard function used internally by histogram processing.
/// For research with configurable bit depth, use `dither_with_mode_bits`.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     mode: DitherMode variant selecting algorithm and scan pattern
///     seed: random seed for mixed modes (ignored for non-mixed modes)
///
/// Returns:
///     flat array of u8 values, same length as input
pub fn dither_with_mode(
    img: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    dither_with_mode_bits(img, width, height, mode, seed, 8, None)
}

/// Dithering with selectable algorithm, scanning mode, and bit depth.
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     width: image width
///     height: image height
///     mode: DitherMode variant selecting algorithm and scan pattern
///     seed: random seed for mixed modes (ignored for non-mixed modes)
///     bits: output bit depth (1-8), controls number of quantization levels
///     progress: optional callback called after each row with progress (0.0 to 1.0)
///
/// Returns:
///     flat array of u8 values, same length as input.
///     Lower bit depths are represented in uint8 by extending bits through repetition.
pub fn dither_with_mode_bits(
    img: &[f32],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
    bits: u8,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let quant = if bits == 8 {
        QuantParams::default_8bit()
    } else {
        QuantParams::new(bits.clamp(1, 8))
    };

    match mode {
        DitherMode::None => dither_standard::<NoneKernel>(img, width, height, quant, progress),
        DitherMode::Standard => dither_standard::<FloydSteinberg>(img, width, height, quant, progress),
        DitherMode::Serpentine => dither_serpentine::<FloydSteinberg>(img, width, height, quant, progress),
        DitherMode::JarvisStandard => dither_standard::<JarvisJudiceNinke>(img, width, height, quant, progress),
        DitherMode::JarvisSerpentine => dither_serpentine::<JarvisJudiceNinke>(img, width, height, quant, progress),
        DitherMode::MixedStandard => mixed_dither_standard(img, width, height, seed, quant, progress),
        DitherMode::MixedSerpentine => mixed_dither_serpentine(img, width, height, seed, quant, progress),
        DitherMode::MixedRandom => mixed_dither_random(img, width, height, seed, quant, progress),
        DitherMode::MixedWangStandard => mixed_dither_wang_standard(img, width, height, seed, quant, progress),
        DitherMode::MixedWangSerpentine => mixed_dither_wang_serpentine(img, width, height, seed, quant, progress),
        DitherMode::MixedLowbiasOldStandard => mixed_dither_lowbias_old_standard(img, width, height, seed, quant, progress),
        DitherMode::MixedLowbiasOldSerpentine => mixed_dither_lowbias_old_serpentine(img, width, height, seed, quant, progress),
        DitherMode::OstromoukhovStandard => dither_standard::<Ostromoukhov>(img, width, height, quant, progress),
        DitherMode::OstromoukhovSerpentine => dither_serpentine::<Ostromoukhov>(img, width, height, quant, progress),
        DitherMode::ZhouFangStandard => zhou_fang_standard(img, width, height, seed, quant, progress),
        DitherMode::ZhouFangSerpentine => zhou_fang_serpentine(img, width, height, seed, quant, progress),
        DitherMode::UlichneyStandard => ulichney_standard(img, width, height, seed, quant, progress),
        DitherMode::UlichneySerpentine => ulichney_serpentine(img, width, height, seed, quant, progress),
    }
}

// ============================================================================
// Pixel4 convenience wrappers
// ============================================================================

use crate::color::interleave_rgb_u8;
use crate::pixel::{pixels_to_channels, Pixel4};

/// Dither Pixel4 array (sRGB 0-255 range) to interleaved u8 with selectable mode.
///
/// This is a convenience wrapper that extracts channels, dithers each independently,
/// and interleaves the result. For higher quality color-aware dithering, use
/// `colorspace_aware_dither_rgb_with_mode` instead.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range
///     width, height: image dimensions
///     mode: dither algorithm and scan pattern
///     seed: random seed for mixed modes
///
/// Returns:
///     Interleaved RGB u8 data (RGBRGB...)
pub fn dither_rgb_with_mode(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    mode: DitherMode,
    seed: u32,
) -> Vec<u8> {
    let (r, g, b) = pixels_to_channels(pixels);
    let r_u8 = dither_with_mode(&r, width, height, mode, seed);
    let g_u8 = dither_with_mode(&g, width, height, mode, seed.wrapping_add(1));
    let b_u8 = dither_with_mode(&b, width, height, mode, seed.wrapping_add(2));
    interleave_rgb_u8(&r_u8, &g_u8, &b_u8)
}

/// Dither Pixel4 array to separate RGB channels with selectable mode and bit depth.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range
///     width, height: image dimensions
///     bits_r, bits_g, bits_b: output bit depth per channel (1-8)
///     mode: dither algorithm and scan pattern
///     seed: random seed for mixed modes
///
/// Returns:
///     Tuple of (R, G, B) u8 vectors
pub fn dither_rgb_channels_with_mode(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    mode: DitherMode,
    seed: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (r, g, b) = pixels_to_channels(pixels);
    let r_u8 = dither_with_mode_bits(&r, width, height, mode, seed, bits_r, None);
    let g_u8 = dither_with_mode_bits(&g, width, height, mode, seed.wrapping_add(1), bits_g, None);
    let b_u8 = dither_with_mode_bits(&b, width, height, mode, seed.wrapping_add(2), bits_b, None);
    (r_u8, g_u8, b_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dither_exact_values() {
        // Values that are already integers should stay the same
        let img = vec![0.0, 128.0, 255.0, 64.0];
        let result = floyd_steinberg_dither(&img, 2, 2);
        // Note: due to error diffusion, exact matches aren't guaranteed
        // but for a simple case like this, corners should be close
        assert!(result[0] == 0 || result[0] == 1);
        assert!(result[2] == 254 || result[2] == 255);
    }

    #[test]
    fn test_dither_clamping() {
        // Values outside 0-255 should be clamped
        let img = vec![-10.0, 300.0];
        let result = floyd_steinberg_dither(&img, 2, 1);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 255);
    }

    #[test]
    fn test_serpentine_dither() {
        // Test that serpentine mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = dither_with_mode(&img, 4, 4, DitherMode::Serpentine, 0);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_serpentine_vs_standard() {
        // Serpentine and standard should produce different results on larger images
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let standard = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::Serpentine, 0);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        // They should produce different results (not identical)
        assert_ne!(standard, serpentine);
    }

    #[test]
    fn test_jarvis_standard() {
        // Test that JJN standard mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = dither_with_mode(&img, 4, 4, DitherMode::JarvisStandard, 0);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_jarvis_serpentine() {
        // Test that JJN serpentine mode produces valid output
        let img = vec![127.5; 16]; // 4x4 gray image
        let result = dither_with_mode(&img, 4, 4, DitherMode::JarvisSerpentine, 0);
        assert_eq!(result.len(), 16);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_jarvis_vs_floyd_steinberg() {
        // JJN and Floyd-Steinberg should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let fs_standard = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let jjn_standard = dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard, 0);
        assert_eq!(fs_standard.len(), 100);
        assert_eq!(jjn_standard.len(), 100);
        // They should produce different results
        assert_ne!(fs_standard, jjn_standard);
    }

    #[test]
    fn test_jarvis_serpentine_vs_standard() {
        // JJN serpentine and standard should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect(); // 10x10 gradient
        let standard = dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard, 0);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::JarvisSerpentine, 0);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        // They should produce different results
        assert_ne!(standard, serpentine);
    }

    #[test]
    fn test_mixed_standard() {
        // Test that mixed standard mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        assert_eq!(result.len(), 100);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_mixed_serpentine() {
        // Test that mixed serpentine mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::MixedSerpentine, 42);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_mixed_random() {
        // Test that mixed random mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::MixedRandom, 42);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_mixed_vs_pure_algorithms() {
        // Mixed modes should produce different results from pure FS and JJN
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let fs = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let jjn = dither_with_mode(&img, 10, 10, DitherMode::JarvisStandard, 0);
        let mixed = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        assert_eq!(fs.len(), 100);
        assert_eq!(jjn.len(), 100);
        assert_eq!(mixed.len(), 100);
        // Mixed should differ from both pure algorithms
        assert_ne!(fs, mixed);
        assert_ne!(jjn, mixed);
    }

    #[test]
    fn test_mixed_deterministic_with_same_seed() {
        // Same seed should produce identical results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result1 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        let result2 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        assert_eq!(result1.len(), 100);
        assert_eq!(result2.len(), 100);
        // Same seed should produce identical results
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_mixed_different_seeds_produce_different_results() {
        // Different seeds should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result1 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        let result2 = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 99);
        assert_eq!(result1.len(), 100);
        assert_eq!(result2.len(), 100);
        // Different seeds should produce different results
        assert_ne!(result1, result2);
    }

    #[test]
    fn test_mixed_all_modes_produce_different_results() {
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let standard = dither_with_mode(&img, 10, 10, DitherMode::MixedStandard, 42);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::MixedSerpentine, 42);
        let random = dither_with_mode(&img, 10, 10, DitherMode::MixedRandom, 42);
        // All three mixed modes should produce different results
        assert_ne!(standard, serpentine);
        assert_ne!(standard, random);
        assert_ne!(serpentine, random);
    }

    // Bit depth tests

    #[test]
    fn test_quant_params_8bit() {
        // 8-bit should behave like normal rounding
        let quant = QuantParams::default_8bit();
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(127.5), 128.0);
        assert_eq!(quant.quantize(255.0), 255.0);
        assert_eq!(quant.quantize(100.4), 100.0);
        assert_eq!(quant.quantize(100.6), 101.0);
    }

    #[test]
    fn test_quant_params_1bit() {
        // 1-bit = 2 levels: 0 and 255
        let quant = QuantParams::new(1);
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(127.0), 0.0);   // < 127.5 rounds to 0
        assert_eq!(quant.quantize(128.0), 255.0); // > 127.5 rounds to 255
        assert_eq!(quant.quantize(255.0), 255.0);
    }

    #[test]
    fn test_quant_params_2bit() {
        // 2-bit = 4 levels: 0, 85, 170, 255
        let quant = QuantParams::new(2);
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(42.0), 0.0);    // rounds to 0
        assert_eq!(quant.quantize(43.0), 85.0);   // rounds to 85
        assert_eq!(quant.quantize(85.0), 85.0);
        assert_eq!(quant.quantize(127.0), 85.0);  // rounds to 85
        assert_eq!(quant.quantize(128.0), 170.0); // rounds to 170
        assert_eq!(quant.quantize(170.0), 170.0);
        assert_eq!(quant.quantize(212.0), 170.0); // rounds to 170
        assert_eq!(quant.quantize(213.0), 255.0); // rounds to 255
        assert_eq!(quant.quantize(255.0), 255.0);
    }

    #[test]
    fn test_quant_params_3bit() {
        // 3-bit = 8 levels with bit-replicated values:
        // 000→00000000=0, 001→00100100=36, 010→01001001=73, 011→01101101=109
        // 100→10010010=146, 101→10110110=182, 110→11011011=219, 111→11111111=255
        let quant = QuantParams::new(3);
        assert_eq!(quant.quantize(0.0), 0.0);
        assert_eq!(quant.quantize(255.0), 255.0);
        // Check that 127.5 quantizes to nearest level (level 3 or 4)
        // 127.5 * 7 / 255 = 3.5, rounds to level 4
        // Level 4 (100) → 10010010 = 146
        assert_eq!(quant.quantize(127.5), 146.0);
    }

    #[test]
    fn test_quantization_uses_bit_replication() {
        // Verify that quantization produces correct bit-replicated values
        // For 5 bits, value 3: should be 00011 → 00011000 = 24 (not 25 from linear scaling)
        let quant_5bit = QuantParams::new(5);
        // Input value that rounds to level 3: level = round(v * 31 / 255)
        // For level 3: v ≈ 3 * 255 / 31 ≈ 24.68, so input around 24-25 should give level 3
        let quantized = quant_5bit.quantize(24.68);
        assert_eq!(quantized, 24.0, "5-bit level 3 should be 24 (bit replication), not 25");

        // Verify some known bit replication values
        // 3-bit: 101 (5) → 10110110 = 182
        let quant_3bit = QuantParams::new(3);
        let level_5_input = 5.0 * 255.0 / 7.0; // ≈ 182.14
        assert_eq!(quant_3bit.quantize(level_5_input), 182.0);

        // 2-bit: all values should match (2 divides 8 evenly)
        let quant_2bit = QuantParams::new(2);
        assert_eq!(quant_2bit.quantize(0.0), 0.0);     // 00 → 00000000
        assert_eq!(quant_2bit.quantize(85.0), 85.0);   // 01 → 01010101
        assert_eq!(quant_2bit.quantize(170.0), 170.0); // 10 → 10101010
        assert_eq!(quant_2bit.quantize(255.0), 255.0); // 11 → 11111111

        // 4-bit: all values should match (4 divides 8 evenly)
        let quant_4bit = QuantParams::new(4);
        assert_eq!(quant_4bit.quantize(0.0), 0.0);     // 0000 → 00000000
        assert_eq!(quant_4bit.quantize(17.0), 17.0);   // 0001 → 00010001
        assert_eq!(quant_4bit.quantize(255.0), 255.0); // 1111 → 11111111
    }

    #[test]
    fn test_bit_replicate_correctness() {
        // Verify the bit_replicate function produces correct values
        // 1-bit
        assert_eq!(bit_replicate(0, 1), 0b00000000);
        assert_eq!(bit_replicate(1, 1), 0b11111111);

        // 2-bit
        assert_eq!(bit_replicate(0, 2), 0b00000000);
        assert_eq!(bit_replicate(1, 2), 0b01010101); // 85
        assert_eq!(bit_replicate(2, 2), 0b10101010); // 170
        assert_eq!(bit_replicate(3, 2), 0b11111111); // 255

        // 3-bit: ABC → ABCABCAB
        assert_eq!(bit_replicate(0b000, 3), 0b00000000); // 0
        assert_eq!(bit_replicate(0b001, 3), 0b00100100); // 36
        assert_eq!(bit_replicate(0b101, 3), 0b10110110); // 182
        assert_eq!(bit_replicate(0b111, 3), 0b11111111); // 255

        // 4-bit: ABCD → ABCDABCD
        assert_eq!(bit_replicate(0b0001, 4), 0b00010001); // 17
        assert_eq!(bit_replicate(0b1010, 4), 0b10101010); // 170

        // 5-bit: ABCDE → ABCDEABC
        assert_eq!(bit_replicate(0b00011, 5), 0b00011000); // 24
        assert_eq!(bit_replicate(0b11111, 5), 0b11111111); // 255
    }

    #[test]
    fn test_dither_1bit_produces_only_0_and_255() {
        // With 1-bit depth, output should only contain 0 and 255
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = floyd_steinberg_dither_bits(&img, 10, 10, 1);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 0 || v == 255, "1-bit dither should only produce 0 or 255, got {}", v);
        }
    }

    #[test]
    fn test_dither_2bit_produces_valid_levels() {
        // With 2-bit depth, output should only contain 0, 85, 170, 255
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = floyd_steinberg_dither_bits(&img, 10, 10, 2);
        assert_eq!(result.len(), 100);
        let valid_levels = [0u8, 85, 170, 255];
        for &v in &result {
            assert!(valid_levels.contains(&v), "2-bit dither should only produce 0, 85, 170, or 255, got {}", v);
        }
    }

    #[test]
    fn test_dither_3bit_produces_valid_levels() {
        // With 3-bit depth, output should only contain 8 specific levels (bit-replicated)
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = floyd_steinberg_dither_bits(&img, 10, 10, 3);
        assert_eq!(result.len(), 100);
        // 3-bit = 8 levels with bit replication: 0, 36, 73, 109, 146, 182, 219, 255
        let valid_levels: Vec<u8> = (0..8).map(|i| bit_replicate(i, 3)).collect();
        for &v in &result {
            assert!(valid_levels.contains(&v), "3-bit dither should only produce valid 3-bit levels, got {}", v);
        }
    }

    #[test]
    fn test_dither_different_bit_depths_produce_different_results() {
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result_8bit = floyd_steinberg_dither_bits(&img, 10, 10, 8);
        let result_4bit = floyd_steinberg_dither_bits(&img, 10, 10, 4);
        let result_2bit = floyd_steinberg_dither_bits(&img, 10, 10, 2);
        let result_1bit = floyd_steinberg_dither_bits(&img, 10, 10, 1);
        // Different bit depths should produce different results
        assert_ne!(result_8bit, result_4bit);
        assert_ne!(result_4bit, result_2bit);
        assert_ne!(result_2bit, result_1bit);
    }

    #[test]
    fn test_dither_8bit_matches_default() {
        // 8-bit should produce the same result as the default function
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result_default = floyd_steinberg_dither(&img, 10, 10);
        let result_8bit = floyd_steinberg_dither_bits(&img, 10, 10, 8);
        assert_eq!(result_default, result_8bit);
    }

    #[test]
    fn test_dither_with_mode_bits_parameter() {
        // Test that dither_with_mode_bits correctly uses the bits parameter
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result_8bit = dither_with_mode_bits(&img, 10, 10, DitherMode::Standard, 0, 8, None);
        let result_2bit = dither_with_mode_bits(&img, 10, 10, DitherMode::Standard, 0, 2, None);
        assert_ne!(result_8bit, result_2bit);
        // 2-bit should only contain valid levels
        let valid_levels = [0u8, 85, 170, 255];
        for &v in &result_2bit {
            assert!(valid_levels.contains(&v));
        }
    }

    #[test]
    fn test_dither_bits_all_modes() {
        // Test that bit depth works with all dithering modes
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let modes = [
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
            DitherMode::UlichneyStandard,
            DitherMode::UlichneySerpentine,
        ];
        let valid_levels = [0u8, 85, 170, 255]; // 2-bit levels
        for mode in modes {
            let result = dither_with_mode_bits(&img, 10, 10, mode, 42, 2, None);
            assert_eq!(result.len(), 100);
            for &v in &result {
                assert!(valid_levels.contains(&v), "Mode {:?} produced invalid 2-bit value: {}", mode, v);
            }
        }
    }

    #[test]
    fn test_ulichney_standard() {
        // Test that Ulichney standard mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 42);
        assert_eq!(result.len(), 100);
        // All values should be valid uint8
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_ulichney_serpentine() {
        // Test that Ulichney serpentine mode produces valid output
        let img = vec![127.5; 100]; // 10x10 gray image
        let result = dither_with_mode(&img, 10, 10, DitherMode::UlichneySerpentine, 42);
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!(v == 127 || v == 128);
        }
    }

    #[test]
    fn test_ulichney_vs_standard_fs() {
        // Ulichney should produce different results from standard Floyd-Steinberg
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let fs = dither_with_mode(&img, 10, 10, DitherMode::Standard, 0);
        let ulichney = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 42);
        assert_eq!(fs.len(), 100);
        assert_eq!(ulichney.len(), 100);
        // They should produce different results due to threshold modulation
        assert_ne!(fs, ulichney);
    }

    #[test]
    fn test_ulichney_deterministic_with_same_seed() {
        // Same seed should produce identical results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result1 = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 42);
        let result2 = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 42);
        assert_eq!(result1.len(), 100);
        assert_eq!(result2.len(), 100);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_ulichney_different_seeds_produce_different_results() {
        // Different seeds should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result1 = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 42);
        let result2 = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 99);
        assert_eq!(result1.len(), 100);
        assert_eq!(result2.len(), 100);
        assert_ne!(result1, result2);
    }

    #[test]
    fn test_ulichney_serpentine_vs_standard() {
        // Ulichney serpentine and standard should produce different results
        let img: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let standard = dither_with_mode(&img, 10, 10, DitherMode::UlichneyStandard, 42);
        let serpentine = dither_with_mode(&img, 10, 10, DitherMode::UlichneySerpentine, 42);
        assert_eq!(standard.len(), 100);
        assert_eq!(serpentine.len(), 100);
        assert_ne!(standard, serpentine);
    }
}
