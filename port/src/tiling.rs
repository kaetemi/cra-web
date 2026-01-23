/// Tiling utilities for localized color correction.
/// Implements overlapping block processing with Hamming window blending.

use std::f32::consts::PI;

/// A block definition with slices for input and reference images
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub y_start: usize,
    pub y_end: usize,
    pub x_start: usize,
    pub x_end: usize,
    pub ref_y_start: usize,
    pub ref_y_end: usize,
    pub ref_x_start: usize,
    pub ref_x_end: usize,
}

/// Generate overlapping blocks for tiled processing.
/// Creates a 9x9 grid of tiles, then generates 8x8 blocks where each
/// block spans 2x2 tiles, resulting in 50% overlap between adjacent blocks.
/// Edge blocks are extended to cover the full image dimensions.
pub fn generate_tile_blocks(
    height: usize,
    width: usize,
    ref_height: usize,
    ref_width: usize,
) -> Vec<Block> {
    let tile_h = height / 9;
    let tile_w = width / 9;
    let tile_h_ref = ref_height / 9;
    let tile_w_ref = ref_width / 9;

    let mut blocks = Vec::with_capacity(64);

    for i in 0..8 {
        for j in 0..8 {
            let y_start = i * tile_h;
            let x_start = j * tile_w;
            let ref_y_start = i * tile_h_ref;
            let ref_x_start = j * tile_w_ref;

            // Extend edge blocks to cover full image
            let y_end = if i == 7 { height } else { y_start + 2 * tile_h };
            let x_end = if j == 7 { width } else { x_start + 2 * tile_w };
            let ref_y_end = if i == 7 {
                ref_height
            } else {
                ref_y_start + 2 * tile_h_ref
            };
            let ref_x_end = if j == 7 {
                ref_width
            } else {
                ref_x_start + 2 * tile_w_ref
            };

            blocks.push(Block {
                y_start,
                y_end,
                x_start,
                x_end,
                ref_y_start,
                ref_y_end,
                ref_x_start,
                ref_x_end,
            });
        }
    }

    blocks
}

/// Compute Hamming window value
#[inline]
fn hamming(n: usize, total: usize) -> f32 {
    0.54 - 0.46 * (2.0 * PI * n as f32 / (total - 1) as f32).cos()
}

/// Create 2D Hamming window for smooth block blending
pub fn create_hamming_weights(height: usize, width: usize) -> Vec<f32> {
    let mut weights = vec![0.0f32; height * width];

    for y in 0..height {
        let y_weight = hamming(y, height);
        for x in 0..width {
            let x_weight = hamming(x, width);
            weights[y * width + x] = y_weight * x_weight;
        }
    }

    weights
}

/// Extract a block from a single-channel image
pub fn extract_block_single(
    img: &[f32],
    width: usize,
    y_start: usize,
    y_end: usize,
    x_start: usize,
    x_end: usize,
) -> Vec<f32> {
    let block_h = y_end - y_start;
    let block_w = x_end - x_start;
    let mut block = vec![0.0f32; block_h * block_w];

    for y in 0..block_h {
        for x in 0..block_w {
            block[y * block_w + x] = img[(y_start + y) * width + (x_start + x)];
        }
    }

    block
}

/// Accumulate a weighted block into the result image (single channel)
pub fn accumulate_block_single(
    result: &mut [f32],
    weight_acc: &mut [f32],
    width: usize,
    block: &[f32],
    weights: &[f32],
    y_start: usize,
    y_end: usize,
    x_start: usize,
    x_end: usize,
) {
    let block_h = y_end - y_start;
    let block_w = x_end - x_start;

    for y in 0..block_h {
        for x in 0..block_w {
            let block_idx = y * block_w + x;
            let img_idx = (y_start + y) * width + (x_start + x);

            result[img_idx] += block[block_idx] * weights[block_idx];
            weight_acc[img_idx] += weights[block_idx];
        }
    }
}

/// Normalize accumulated results by weights
pub fn normalize_accumulated(result: &mut [f32], weights: &[f32]) {
    for (val, &w) in result.iter_mut().zip(weights.iter()) {
        if w > 1e-7 {
            *val /= w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_blocks() {
        let blocks = generate_tile_blocks(900, 900, 900, 900);
        assert_eq!(blocks.len(), 64);

        // Check first block
        assert_eq!(blocks[0].y_start, 0);
        assert_eq!(blocks[0].x_start, 0);

        // Check last block covers the end
        let last = &blocks[63];
        assert_eq!(last.y_end, 900);
        assert_eq!(last.x_end, 900);
    }

    #[test]
    fn test_hamming_weights() {
        let weights = create_hamming_weights(10, 10);
        assert_eq!(weights.len(), 100);

        // Center should have higher weight than edges
        let center = weights[5 * 10 + 5];
        let corner = weights[0];
        assert!(center > corner);
    }
}
