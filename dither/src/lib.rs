use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn floyd_steinberg_dither(img: Vec<f32>, w: usize, h: usize) -> Vec<u8> {
    let len = w * h;

    // Allocate buffer with overflow padding
    let mut buf = vec![0.0f32; len + w + 2];
    buf[..len].copy_from_slice(&img);

    for i in 0..len {
        let old = buf[i];
        let new = old.round();
        buf[i] = new;
        let err = old - new;

        // Distribute error to neighbors
        // Overflow writes hit padding, which we discard
        buf[i + 1] += err * (7.0 / 16.0);
        buf[i + w - 1] += err * (3.0 / 16.0);
        buf[i + w] += err * (5.0 / 16.0);
        buf[i + w + 1] += err * (1.0 / 16.0);
    }

    // Clamp and convert to u8, discard padding
    buf[..len]
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect()
}
