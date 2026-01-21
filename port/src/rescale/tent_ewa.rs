//! 2D Tent-Space EWA resampling and iterative downscaling
//!
//! This module implements 2D (non-separable) tent-space resampling using
//! precomputed kernel weights. Unlike the separable TentBox/TentLanczos3
//! methods, these use the full 2D tent-space kernel.
//!
//! Available methods:
//! - **Tent2DBox**: 2D tent-space with box filter integration
//! - **Tent2DLanczos3Jinc**: 2D tent-space with EWA Lanczos3-jinc kernel
//! - **TentBoxIterative**: Iterative 2× separable tent-box downscaling
//! - **Tent2DBoxIterative**: Iterative 2× 2D tent-box downscaling
//! - **BilinearIterative**: Iterative 2× bilinear downscaling (mipmap-style)
//! - **IterativeTentVolume**: Iterative scaling with explicit tent_expand/contract steps

use crate::pixel::Pixel4;
use crate::supersample::{tent_expand, tent_contract};
use super::{RescaleMethod, ScaleMode, TentMode, calculate_scales};
use super::kernels::precompute_tent_2d_kernel_weights;
use super::separable;
use super::bilinear;

/// 2D Tent-space resampling for Pixel4 images using precomputed kernel weights.
///
/// This applies the full 2D tent-space kernel (expand → resample → contract)
/// in a single pass using precomputed weights.
pub fn rescale_tent_2d_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate scales
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Determine if using EWA or box kernel
    let is_ewa = matches!(method, RescaleMethod::Tent2DLanczos3Jinc);

    // Precompute all 2D kernel weights
    let all_weights = precompute_tent_2d_kernel_weights(
        src_height, src_width, dst_height, dst_width,
        scale_y, scale_x, is_ewa
    );

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let row_weights = &all_weights[dst_y];

        for dst_x in 0..dst_width {
            let kw = &row_weights[dst_x];

            let mut sum = Pixel4::default();

            if kw.weights.is_empty() || kw.weights.iter().all(|&w| w.abs() < 1e-10) {
                // Fallback: use nearest pixel
                let idx = kw.fallback_y * src_width + kw.fallback_x;
                dst[dst_y * dst_width + dst_x] = src[idx];
                continue;
            }

            // Apply 2D kernel weights
            for ky in 0..kw.height {
                let src_y = kw.start_y + ky;
                if src_y >= src_height {
                    continue;
                }

                for kx in 0..kw.width {
                    let src_x = kw.start_x + kx;
                    if src_x >= src_width {
                        continue;
                    }

                    let weight = kw.weights[ky * kw.width + kx];
                    if weight.abs() < 1e-10 {
                        continue;
                    }

                    let pixel = src[src_y * src_width + src_x];
                    sum = sum + pixel * weight;
                }
            }

            dst[dst_y * dst_width + dst_x] = sum;
        }

        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

/// Alpha-aware 2D tent-space resampling for Pixel4 images.
/// RGB channels are weighted by alpha to prevent transparent pixel color bleeding.
pub fn rescale_tent_2d_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate scales
    let (scale_x, scale_y) = calculate_scales(
        src_width, src_height, dst_width, dst_height, scale_mode
    );

    // Determine if using EWA or box kernel
    let is_ewa = matches!(method, RescaleMethod::Tent2DLanczos3Jinc);

    // Precompute all 2D kernel weights
    let all_weights = precompute_tent_2d_kernel_weights(
        src_height, src_width, dst_height, dst_width,
        scale_y, scale_x, is_ewa
    );

    let mut dst = vec![Pixel4::default(); dst_width * dst_height];

    for dst_y in 0..dst_height {
        let row_weights = &all_weights[dst_y];

        for dst_x in 0..dst_width {
            let kw = &row_weights[dst_x];

            if kw.weights.is_empty() || kw.weights.iter().all(|&w| w.abs() < 1e-10) {
                // Fallback: use nearest pixel
                let idx = kw.fallback_y * src_width + kw.fallback_x;
                dst[dst_y * dst_width + dst_x] = src[idx];
                continue;
            }

            // Alpha-aware accumulation
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_a = 0.0f32;
            let mut sum_alpha_weight = 0.0f32;
            let mut weight_sum = 0.0f32;
            // Fallback: unweighted RGB sum
            let mut sum_r_unweighted = 0.0f32;
            let mut sum_g_unweighted = 0.0f32;
            let mut sum_b_unweighted = 0.0f32;

            // Apply 2D kernel weights
            for ky in 0..kw.height {
                let src_y = kw.start_y + ky;
                if src_y >= src_height {
                    continue;
                }

                for kx in 0..kw.width {
                    let src_x = kw.start_x + kx;
                    if src_x >= src_width {
                        continue;
                    }

                    let weight = kw.weights[ky * kw.width + kx];
                    if weight.abs() < 1e-10 {
                        continue;
                    }

                    let p = src[src_y * src_width + src_x];
                    let alpha = p.a();
                    let aw = weight * alpha;

                    sum_r += aw * p.r();
                    sum_g += aw * p.g();
                    sum_b += aw * p.b();
                    sum_a += weight * alpha;
                    sum_alpha_weight += aw;
                    weight_sum += weight;

                    sum_r_unweighted += weight * p.r();
                    sum_g_unweighted += weight * p.g();
                    sum_b_unweighted += weight * p.b();
                }
            }

            // Normalize
            let out_a = if weight_sum.abs() > 1e-8 {
                sum_a / weight_sum
            } else {
                0.0
            };

            let (out_r, out_g, out_b) = if sum_alpha_weight.abs() > 1e-8 {
                let inv_aw = 1.0 / sum_alpha_weight;
                (sum_r * inv_aw, sum_g * inv_aw, sum_b * inv_aw)
            } else if weight_sum.abs() > 1e-8 {
                let inv_w = 1.0 / weight_sum;
                (sum_r_unweighted * inv_w, sum_g_unweighted * inv_w, sum_b_unweighted * inv_w)
            } else {
                // Fallback: nearest neighbor
                let p = src[kw.fallback_y * src_width + kw.fallback_x];
                (p.r(), p.g(), p.b())
            };

            dst[dst_y * dst_width + dst_x] = Pixel4::new(out_r, out_g, out_b, out_a);
        }

        if let Some(ref mut cb) = progress {
            cb((dst_y + 1) as f32 / dst_height as f32);
        }
    }

    dst
}

// ============================================================================
// Iterative Tent-Box Downscaling
// ============================================================================

/// Calculate iterative 2× downscale passes for a given scale factor.
///
/// Returns a vector of (intermediate_width, intermediate_height) for each pass,
/// ending with the final target dimensions.
///
/// Example for 8× downscale (1024→128):
///   - Pass 1: 1024 → 512 (2×)
///   - Pass 2: 512 → 256 (2×)
///   - Pass 3: 256 → 128 (2×)
///
/// Example for 6× downscale (1024→171, but with rounding 1024→170):
///   - Pass 1: 1024 → 512 (2×)
///   - Pass 2: 512 → 170 (3.01×, or final target)
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

/// Iterative tent-box downscaling for Pixel4 images.
///
/// For downscaling by factor N, iteratively applies 2× tent-box for each power of 2,
/// then applies the remaining factor. This can produce better quality than a single
/// large-factor kernel because the 2× tent-box kernel is optimal.
///
/// When `use_2d` is true, uses the full 2D tent-space kernel (Tent2DBox).
/// When `use_2d` is false, uses the separable tent-space kernel (TentBox).
pub fn rescale_tent_iterative_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    use_2d: bool,
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

        // Apply tent-box downscale
        let next = if use_2d {
            rescale_tent_2d_pixels(
                &current, current_w, current_h,
                next_w, next_h,
                RescaleMethod::Tent2DBox, scale_mode,
                None,
            )
        } else {
            separable::rescale_kernel_pixels(
                &current, current_w, current_h,
                next_w, next_h,
                RescaleMethod::TentBox, scale_mode,
                TentMode::Off, None,
            )
        };

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

/// Alpha-aware iterative tent-box downscaling for Pixel4 images.
pub fn rescale_tent_iterative_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    use_2d: bool,
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

        // Apply tent-box downscale (alpha-aware)
        let next = if use_2d {
            rescale_tent_2d_alpha_pixels(
                &current, current_w, current_h,
                next_w, next_h,
                RescaleMethod::Tent2DBox, scale_mode,
                None,
            )
        } else {
            separable::rescale_kernel_alpha_pixels(
                &current, current_w, current_h,
                next_w, next_h,
                RescaleMethod::TentBox, scale_mode,
                TentMode::Off, None,
            )
        };

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

// ============================================================================
// Iterative Bilinear Downscaling
// ============================================================================

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

// ============================================================================
// Iterative Tent Volume Scaling (explicit tent_expand/contract)
// ============================================================================

/// Calculate iterative 2× scaling passes for tent volume method.
///
/// Returns a vector of (intermediate_width, intermediate_height) in box-space,
/// ending with the final target dimensions.
///
/// For downscaling, halves dimensions iteratively.
/// For upscaling, doubles dimensions iteratively.
fn calculate_iterative_tent_volume_passes(
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

    if scale > 1.0 {
        // Downscaling: halve iteratively
        let num_halves = (scale.log2().floor() as usize).max(0);

        let mut current_w = src_width;
        let mut current_h = src_height;

        for _ in 0..num_halves {
            let next_w = (current_w + 1) / 2;
            let next_h = (current_h + 1) / 2;

            // Stop if we'd overshoot the target
            if next_w < dst_width || next_h < dst_height {
                break;
            }

            passes.push((next_w, next_h));
            current_w = next_w;
            current_h = next_h;
        }

        // Final pass to exact dimensions
        if current_w != dst_width || current_h != dst_height {
            passes.push((dst_width, dst_height));
        }
    } else if scale < 1.0 {
        // Upscaling: double iteratively
        let inverse_scale = 1.0 / scale;
        let num_doubles = (inverse_scale.log2().floor() as usize).max(0);

        let mut current_w = src_width;
        let mut current_h = src_height;

        for _ in 0..num_doubles {
            let next_w = current_w * 2;
            let next_h = current_h * 2;

            // Stop if we'd overshoot the target
            if next_w > dst_width || next_h > dst_height {
                break;
            }

            passes.push((next_w, next_h));
            current_w = next_w;
            current_h = next_h;
        }

        // Final pass to exact dimensions
        if current_w != dst_width || current_h != dst_height {
            passes.push((dst_width, dst_height));
        }
    } else {
        // Identity - just return target
        if src_width != dst_width || src_height != dst_height {
            passes.push((dst_width, dst_height));
        }
    }

    passes
}

/// Perform a single tent volume scaling step.
///
/// Steps:
/// 1. tent_expand: box (W, H) → tent (2W+1, 2H+1)
/// 2. Scale in tent space to target tent dimensions (box or bilinear)
/// 3. tent_contract: tent (2*dstW+1, 2*dstH+1) → box (dstW, dstH)
///
/// `use_bilinear`: if true, use bilinear; if false, use box filter
fn tent_volume_scale_step(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    use_bilinear: bool,
) -> Vec<Pixel4> {
    // Step 1: Expand to tent space
    let (tent_data, tent_w, tent_h) = tent_expand(src, src_width, src_height);

    // Calculate target tent dimensions
    let target_tent_w = dst_width * 2 + 1;
    let target_tent_h = dst_height * 2 + 1;

    // Step 2: Scale in tent space with sample-to-sample mapping (tent-to-tent)
    let scaled_tent = if use_bilinear {
        bilinear::rescale_bilinear_pixels(
            &tent_data, tent_w, tent_h,
            target_tent_w, target_tent_h,
            scale_mode,
            TentMode::SampleToSample,
            None,
        )
    } else {
        // Use box filter for tent-to-tent scaling
        separable::rescale_kernel_pixels(
            &tent_data, tent_w, tent_h,
            target_tent_w, target_tent_h,
            RescaleMethod::Box,
            scale_mode,
            TentMode::SampleToSample,
            None,
        )
    };

    // Step 3: Contract back to box space
    let (result, _, _) = tent_contract(&scaled_tent, target_tent_w, target_tent_h);

    result
}

/// Alpha-aware single tent volume scaling step.
fn tent_volume_scale_step_alpha(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    use_bilinear: bool,
) -> Vec<Pixel4> {
    // Step 1: Expand to tent space
    let (tent_data, tent_w, tent_h) = tent_expand(src, src_width, src_height);

    // Calculate target tent dimensions
    let target_tent_w = dst_width * 2 + 1;
    let target_tent_h = dst_height * 2 + 1;

    // Step 2: Scale in tent space with sample-to-sample mapping (tent-to-tent)
    let scaled_tent = if use_bilinear {
        bilinear::rescale_bilinear_alpha_pixels(
            &tent_data, tent_w, tent_h,
            target_tent_w, target_tent_h,
            scale_mode,
            TentMode::SampleToSample,
            None,
        )
    } else {
        // Use box filter for tent-to-tent scaling (alpha-aware)
        separable::rescale_kernel_alpha_pixels(
            &tent_data, tent_w, tent_h,
            target_tent_w, target_tent_h,
            RescaleMethod::Box,
            scale_mode,
            TentMode::SampleToSample,
            None,
        )
    };

    // Step 3: Contract back to box space
    let (result, _, _) = tent_contract(&scaled_tent, target_tent_w, target_tent_h);

    result
}

/// Iterative tent volume scaling for Pixel4 images.
///
/// Each iteration explicitly uses:
/// 1. tent_expand (box → tent)
/// 2. Scale in tent space (box or bilinear based on `use_bilinear`)
/// 3. tent_contract (tent → box)
///
/// This ensures volume preservation at every step through the tent representation.
///
/// `use_bilinear`: if true, use bilinear interpolation; if false, use box filter
pub fn rescale_iterative_tent_volume_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    use_bilinear: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate the pass schedule
    let passes = calculate_iterative_tent_volume_passes(src_width, src_height, dst_width, dst_height);

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

        // Apply tent volume scale step
        let next = tent_volume_scale_step(
            &current, current_w, current_h,
            next_w, next_h,
            scale_mode,
            use_bilinear,
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

/// Alpha-aware iterative tent volume scaling for Pixel4 images.
///
/// `use_bilinear`: if true, use bilinear interpolation; if false, use box filter
pub fn rescale_iterative_tent_volume_alpha_pixels(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
    use_bilinear: bool,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    // Calculate the pass schedule
    let passes = calculate_iterative_tent_volume_passes(src_width, src_height, dst_width, dst_height);

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

        // Apply tent volume scale step (alpha-aware)
        let next = tent_volume_scale_step_alpha(
            &current, current_w, current_h,
            next_w, next_h,
            scale_mode,
            use_bilinear,
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
