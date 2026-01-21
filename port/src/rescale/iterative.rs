//! Iterative downscaling methods
//!
//! This module implements iterative (mipmap-style) downscaling where large
//! scale factors are broken into multiple 2× passes for better quality.

use crate::pixel::Pixel4;
use super::{ScaleMode, TentMode};
use super::bilinear;

/// Calculate the pass schedule for iterative downscaling.
///
/// For a scale factor N, this calculates intermediate dimensions for
/// floor(log2(N)) passes of 2× downscaling, followed by a final pass
/// to reach the exact target dimensions.
///
/// Returns a vector of (width, height) tuples for each pass.
fn calculate_iterative_passes(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<(usize, usize)> {
    let mut passes = Vec::new();

    // Use the larger scale factor to determine passes
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;
    let scale = scale_x.max(scale_y);

    if scale <= 1.0 {
        // Upscaling or identity - just return target dimensions
        passes.push((dst_width, dst_height));
        return passes;
    }

    // Calculate number of 2× passes
    // For scale N, we want floor(log2(N)) passes of 2×
    let num_2x_passes = (scale.log2().floor() as usize).max(0);

    let mut current_w = src_width;
    let mut current_h = src_height;

    for i in 0..num_2x_passes {
        // Calculate target for this 2× pass
        // For the last 2× pass, we need to ensure we don't overshoot the final target
        let next_w = (current_w + 1) / 2;  // Ceiling division for safety
        let next_h = (current_h + 1) / 2;

        // If this would overshoot the final target, stop 2× passes
        if next_w < dst_width || next_h < dst_height {
            break;
        }

        // If this is the last pass and we'd exactly hit the target, use target
        if i == num_2x_passes - 1 && next_w == dst_width && next_h == dst_height {
            passes.push((dst_width, dst_height));
            current_w = dst_width;
            current_h = dst_height;
        } else {
            passes.push((next_w, next_h));
            current_w = next_w;
            current_h = next_h;
        }
    }

    // Add final pass to reach exact target dimensions if not already there
    if current_w != dst_width || current_h != dst_height {
        passes.push((dst_width, dst_height));
    }

    passes
}

/// Iterative bilinear downscaling for Pixel4 images (mipmap-style).
///
/// For downscaling by factor N, iteratively applies 2× bilinear for each power of 2,
/// then applies the remaining factor. This produces mipmap-quality downscaling
/// which is often better than a single pass for large scale factors.
pub fn rescale_bilinear_iterative_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate the pass schedule
    let passes = calculate_iterative_passes(src_width, src_height, dst_width, dst_height);

    if passes.is_empty() {
        // Identity
        if let Some(ref mut cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    let total_passes = passes.len();
    let mut current = src.to_vec();
    let mut current_w = src_width;
    let mut current_h = src_height;

    for (pass_idx, &(next_w, next_h)) in passes.iter().enumerate() {
        // Skip if dimensions unchanged
        if next_w == current_w && next_h == current_h {
            continue;
        }

        // Apply bilinear downscale
        let next = bilinear::rescale_bilinear_pixels(
            &current, current_w, current_h,
            next_w, next_h,
            scale_mode,
            TentMode::Off,
            None,
        );

        current = next;
        current_w = next_w;
        current_h = next_h;

        // Report progress
        if let Some(ref mut cb) = progress {
            cb((pass_idx + 1) as f32 / total_passes as f32);
        }
    }

    current
}

/// Alpha-aware iterative bilinear downscaling for Pixel4 images.
pub fn rescale_bilinear_iterative_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate the pass schedule
    let passes = calculate_iterative_passes(src_width, src_height, dst_width, dst_height);

    if passes.is_empty() {
        // Identity
        if let Some(ref mut cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    let total_passes = passes.len();
    let mut current = src.to_vec();
    let mut current_w = src_width;
    let mut current_h = src_height;

    for (pass_idx, &(next_w, next_h)) in passes.iter().enumerate() {
        // Skip if dimensions unchanged
        if next_w == current_w && next_h == current_h {
            continue;
        }

        // Apply bilinear downscale (alpha-aware)
        let next = bilinear::rescale_bilinear_alpha_pixels(
            &current, current_w, current_h,
            next_w, next_h,
            scale_mode,
            TentMode::Off,
            None,
        );

        current = next;
        current_w = next_w;
        current_h = next_h;

        // Report progress
        if let Some(ref mut cb) = progress {
            cb((pass_idx + 1) as f32 / total_passes as f32);
        }
    }

    current
}
