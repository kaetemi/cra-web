use wasm_bindgen::prelude::*;

/// Floyd-Steinberg dithering with 2D padded buffer.
/// Uses padding to avoid bounds checks: 1 col left, 1 col right, 1 row bottom.
#[wasm_bindgen]
pub fn floyd_steinberg_dither(img: Vec<f32>, width: usize, height: usize) -> Vec<u8> {
    let pad_left = 1;
    let pad_right = 1;
    let pad_bottom = 1;
    let buf_width = width + pad_left + pad_right;
    let buf_height = height + pad_bottom;

    // Create padded buffer and copy image data
    let mut buf = vec![vec![0.0f32; buf_width]; buf_height];
    for y in 0..height {
        for x in 0..width {
            buf[y][x + pad_left] = img[y * width + x];
        }
    }

    // Process only real pixels, error diffusion writes to padding are safe
    for y in 0..height {
        for x in 0..width {
            let bx = x + pad_left;
            let old = buf[y][bx];
            let new = old.round();
            buf[y][bx] = new;
            let err = old - new;

            // FS kernel:   * 7
            //            3 5 1
            buf[y][bx + 1] += err * (7.0 / 16.0);
            buf[y + 1][bx - 1] += err * (3.0 / 16.0);
            buf[y + 1][bx] += err * (5.0 / 16.0);
            buf[y + 1][bx + 1] += err * (1.0 / 16.0);
        }
    }

    // Extract real pixels, clamp, and convert to u8
    let mut result = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            result.push(buf[y][x + pad_left].clamp(0.0, 255.0).round() as u8);
        }
    }
    result
}
