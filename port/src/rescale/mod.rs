//! Image rescaling module with Bilinear, Mitchell, Lanczos, and Sinc support
//!
//! Operates in linear RGB space for correct color blending during interpolation.
//!
//! # Module Structure
//! - `kernels`: Interpolation kernel functions and weight precomputation
//! - `bilinear`: Bilinear interpolation implementation
//! - `separable`: Separable 2-pass kernel rescaling (Mitchell, Lanczos, etc.)

mod kernels;
mod bilinear;
mod separable;
mod scatter;
mod ewa;
mod tent_ewa;

#[cfg(test)]
mod tests_basic;
#[cfg(test)]
mod tests_advanced;

use crate::pixel::Pixel4;

// Re-export kernel types for internal use
pub use kernels::KernelWeights;

/// Rescaling method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RescaleMethod {
    /// Bilinear interpolation - fast, good for moderate scaling
    Bilinear,
    /// Mitchell-Netravali (B=C=1/3) - soft, minimal ringing
    Mitchell,
    /// Catmull-Rom (B=0, C=0.5) - sharp interpolating spline, low ringing
    CatmullRom,
    /// Lanczos2 - good sharpness with less ringing than Lanczos3
    Lanczos2,
    /// Lanczos3 - good balance of sharpness and ringing
    Lanczos3,
    /// Pure Sinc (non-windowed) - theoretically ideal, uses full image extent
    /// WARNING: O(N²) - very slow for large images, severe ringing at edges
    Sinc,
    /// Lanczos3 with scatter-based accumulation (experimental)
    Lanczos3Scatter,
    /// Sinc with scatter-based accumulation (experimental)
    /// WARNING: O(N²) - very slow for large images
    SincScatter,
    /// Box filter: true area integration computing exact overlap between dest and source pixels.
    /// - Upscaling: nearest-neighbor (dest pixel smaller than source, samples one pixel)
    /// - Downscaling: proper area average (dest pixel covers multiple sources, weighted by overlap)
    Box,
    /// EWA (Elliptical Weighted Average) with radial sinc-Lanczos2 kernel
    /// Note: Uses 1D sinc applied radially, not true jinc-based EWA Lanczos
    EWASincLanczos2,
    /// EWA (Elliptical Weighted Average) with radial sinc-Lanczos3 kernel
    /// Note: Uses 1D sinc applied radially, not true jinc-based EWA Lanczos
    EWASincLanczos3,
    /// EWA (Elliptical Weighted Average) Lanczos2 with proper jinc-based kernel
    /// Uses jinc(r) * jinc(r/2) where jinc is the 2D analog of sinc (J1 Bessel)
    EWALanczos2,
    /// EWA (Elliptical Weighted Average) Lanczos3 with proper jinc-based kernel
    /// Uses jinc(r) * jinc(r/3) where jinc is the 2D analog of sinc (J1 Bessel)
    EWALanczos3,
    /// EWA Lanczos3 Sharp (3-lobe, sharpened to minimize 1D step response error)
    /// Uses blur=0.9812505837223707 (Robidoux optimization for ImageMagick)
    /// Better preserves horizontal/vertical lines
    EWALanczos3Sharp,
    /// EWA Lanczos4 Sharpest (4-lobe, sharpened to minimize total impulse response error)
    /// Uses blur=0.8845120932605005 (Robidoux optimization)
    /// Very sharp but rings more. Best for preserving hash patterns.
    EWALanczos4Sharpest,
    /// EWA (Elliptical Weighted Average) Mitchell-Netravali (B=C=1/3)
    /// 2D radial application of the Mitchell cubic kernel
    EWAMitchell,
    /// EWA (Elliptical Weighted Average) Catmull-Rom (B=0, C=0.5)
    /// 2D radial application of the Catmull-Rom cubic kernel
    EWACatmullRom,
    /// Pure Jinc (unwindowed) - the ideal 2D radially symmetric lowpass filter
    /// Uses J1(πr)/(πr) over full image extent via EWA sampling
    /// WARNING: O(N²) - extremely slow for large images, severe ringing at edges
    Jinc,
    /// Stochastic Jinc (gather) - probabilistic sampling with Gaussian selection
    /// Uses Lanczos5-sized window (radius 5.0) with Gaussian sampling distribution.
    /// Normalizes intake per destination pixel. Much faster than full Jinc.
    StochasticJinc,
    /// Stochastic Jinc with scatter-based accumulation
    /// Instead of normalizing intake per destination (gather), normalizes emission per source.
    /// Each source pixel distributes its value to destinations with weights summing to 1.0.
    /// Better energy conservation, may have different aliasing characteristics.
    StochasticJincScatter,
    /// Stochastic Jinc Scatter with destination normalization
    /// Like StochasticJincScatter but also normalizes by total weight received per destination.
    /// Makes brightness equivalent to gather while keeping scatter characteristics.
    StochasticJincScatterNormalized,
    /// Tent-space Box filter pipeline
    /// Full tent-space downscaling: box→tent expand, resample with box filter, tent→box contract.
    /// Produces equivalent direct kernel: [-1, 7, 26, 26, 7, -1]/64 for 2× downscale.
    /// Better quality than plain box filter due to volume-preserving tent representation.
    TentBox,
    /// Tent-space Lanczos3 filter pipeline
    /// Full tent-space downscaling: box→tent expand, resample with Lanczos3, tent→box contract.
    /// Combines tent-space's ringing-free surface with Lanczos3's sharpness.
    TentLanczos3,
    /// 2D Tent-space Box filter (non-separable)
    /// Full tent-space pipeline using true 2D box integration (area overlap).
    /// Unlike separable TentBox, this uses the exact 2D tent-space kernel.
    /// For 2× downscale, produces 6×6 kernel: sum=4096, negative lobes at corners.
    Tent2DBox,
    /// 2D Tent-space EWA Lanczos3 with jinc (non-separable)
    /// Full tent-space pipeline using EWA Lanczos3-jinc radial kernel.
    /// Combines tent-space volume preservation with proper 2D radial filtering.
    Tent2DLanczos3Jinc,
    /// Iterative Tent-space Box filter (separable)
    /// For downscaling by factor N, iteratively applies 2× tent-box for each power of 2,
    /// then applies the remaining factor. E.g., 8× = 2×→2×→2×, 6× = 2×→3×.
    /// This can produce better quality than a single large-factor kernel.
    TentBoxIterative,
    /// Iterative 2D Tent-space Box filter (non-separable)
    /// Same iterative approach as TentBoxIterative but using the full 2D kernel.
    Tent2DBoxIterative,
    /// Iterative Bilinear downscaling
    /// For downscaling by factor N, iteratively applies 2× bilinear for each power of 2,
    /// then applies the remaining factor. Useful for mipmap-style quality downscaling.
    BilinearIterative,
    /// Iterative Tent Volume scaling with explicit supersample steps (box inner filter)
    /// Each iteration: tent_expand → box scale in tent-space → tent_contract
    /// This explicitly uses the tent volume representation for each 2× step,
    /// ensuring volume preservation at every iteration.
    /// Final step scales to exact target size in tent-space before contracting.
    IterativeTentVolume,
    /// Iterative Tent Volume scaling with bilinear inner filter
    /// Each iteration: tent_expand → bilinear scale in tent-space → tent_contract
    /// Same as IterativeTentVolume but uses bilinear interpolation in tent-space.
    IterativeTentVolumeBilinear,
    /// Tent-space with Lanczos3 kernel constraint (instead of volume constraint)
    /// Uses tent_expand_lanczos (Lanczos3-constrained peaks) and tent_contract_lanczos.
    /// Fully reversible: Lanczos3 interpolation at peaks returns original values.
    /// Inner scaling uses box filter in tent-space.
    TentLanczos3Constraint,
}

/// Scale mode for aspect ratio preservation
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScaleMode {
    /// Independent X/Y scaling (default, can cause slight AR distortion)
    #[default]
    Independent,
    /// Uniform scaling based on width (width is primary dimension)
    UniformWidth,
    /// Uniform scaling based on height (height is primary dimension)
    UniformHeight,
}

/// Tent-space coordinate mapping mode
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TentMode {
    /// Standard box-to-box mapping (pixel centers to pixel centers)
    #[default]
    Off,
    /// Sample-to-sample mapping for tent-space (edge samples map to edge samples)
    /// Used when both source and destination are in tent-space.
    /// Maps position 0→0, position N-1→M-1.
    SampleToSample,
    /// Tent-to-box prescale mapping (integrates tent_contract into rescale)
    /// Source is tent-space (samples at integer positions), destination is box-space.
    /// Maps src tent sample 1 → dst pixel center 0.5, etc.
    /// The 0.5 pixel fringe is only on the src (tent) side, not the dst (box) side.
    Prescale,
}

impl RescaleMethod {
    pub fn from_str(s: &str) -> Option<RescaleMethod> {
        match s.to_lowercase().as_str() {
            "bilinear" | "linear" => Some(RescaleMethod::Bilinear),
            "mitchell" => Some(RescaleMethod::Mitchell),
            "catmull-rom" | "catmullrom" | "catrom" | "cubic" | "bicubic" => Some(RescaleMethod::CatmullRom),
            "lanczos2" => Some(RescaleMethod::Lanczos2),
            "lanczos" | "lanczos3" => Some(RescaleMethod::Lanczos3),
            "sinc" => Some(RescaleMethod::Sinc),
            "lanczos3-scatter" | "lanczos_scatter" => Some(RescaleMethod::Lanczos3Scatter),
            "sinc-scatter" | "sinc_scatter" => Some(RescaleMethod::SincScatter),
            "box" | "area" | "nearest" => Some(RescaleMethod::Box),
            "ewa-sinc-lanczos2" | "ewa_sinc_lanczos2" | "ewasinclanczos2" => Some(RescaleMethod::EWASincLanczos2),
            "ewa-sinc-lanczos3" | "ewa_sinc_lanczos3" | "ewasinclanczos3" | "ewa-sinc-lanczos" | "ewa_sinc_lanczos" => Some(RescaleMethod::EWASincLanczos3),
            "ewa-lanczos2" | "ewa_lanczos2" | "ewalanczos2" => Some(RescaleMethod::EWALanczos2),
            "ewa-lanczos3" | "ewa_lanczos3" | "ewalanczos3" | "ewa-lanczos" | "ewa_lanczos" => Some(RescaleMethod::EWALanczos3),
            "ewa-lanczos3-sharp" | "ewa_lanczos3_sharp" | "ewalanczos3sharp" | "ewa-lanczos-sharp" | "ewa_lanczos_sharp" | "lanczos3-sharp" | "lanczos3sharp" => Some(RescaleMethod::EWALanczos3Sharp),
            "ewa-lanczos4-sharpest" | "ewa_lanczos4_sharpest" | "ewalanczos4sharpest" | "lanczos4-sharpest" | "lanczos4sharpest" => Some(RescaleMethod::EWALanczos4Sharpest),
            "ewa-mitchell" | "ewa_mitchell" | "ewamitchell" => Some(RescaleMethod::EWAMitchell),
            "ewa-catmull-rom" | "ewa_catmull_rom" | "ewacatmullrom" | "ewa-catrom" | "ewa_catrom" => Some(RescaleMethod::EWACatmullRom),
            "jinc" => Some(RescaleMethod::Jinc),
            "stochastic-jinc" | "stochastic_jinc" | "stochasticjinc" | "stochastic" => Some(RescaleMethod::StochasticJinc),
            "stochastic-jinc-scatter" | "stochastic_jinc_scatter" | "stochasticjincscatter" | "stochastic-scatter" | "stochastic_scatter" => Some(RescaleMethod::StochasticJincScatter),
            "stochastic-jinc-scatter-normalized" | "stochastic_jinc_scatter_normalized" | "stochasticjincscatternormalized" | "stochastic-scatter-normalized" | "stochastic_scatter_normalized" => Some(RescaleMethod::StochasticJincScatterNormalized),
            "tent-box" | "tent_box" | "tentbox" => Some(RescaleMethod::TentBox),
            "tent-lanczos3" | "tent_lanczos3" | "tentlanczos3" | "tent-lanczos" | "tent_lanczos" | "tentlanczos" => Some(RescaleMethod::TentLanczos3),
            "tent-2d-box" | "tent_2d_box" | "tent2dbox" | "tent-2d" | "tent_2d" | "tent2d" => Some(RescaleMethod::Tent2DBox),
            "tent-2d-lanczos3-jinc" | "tent_2d_lanczos3_jinc" | "tent2dlanczos3jinc" | "tent-2d-lanczos" | "tent_2d_lanczos" | "tent2dlanczos" | "tent-2d-jinc" | "tent_2d_jinc" | "tent2djinc" => Some(RescaleMethod::Tent2DLanczos3Jinc),
            "tent-box-iterative" | "tent_box_iterative" | "tentboxiterative" | "tent-iterative" | "tent_iterative" | "tentiterative" => Some(RescaleMethod::TentBoxIterative),
            "tent-2d-box-iterative" | "tent_2d_box_iterative" | "tent2dboxiterative" | "tent-2d-iterative" | "tent_2d_iterative" | "tent2diterative" => Some(RescaleMethod::Tent2DBoxIterative),
            "bilinear-iterative" | "bilinear_iterative" | "bilineariterative" | "mipmap" | "mip" => Some(RescaleMethod::BilinearIterative),
            "iterative-tent-volume" | "iterative_tent_volume" | "iterativetentvolume" | "tent-volume-iterative" | "tent_volume_iterative" | "tentvolume" | "tent-vol" | "tent_vol" | "iterative-tent-volume-box" | "iterative_tent_volume_box" => Some(RescaleMethod::IterativeTentVolume),
            "iterative-tent-volume-bilinear" | "iterative_tent_volume_bilinear" | "tentvolume-bilinear" | "tent-vol-bilinear" | "tent_vol_bilinear" => Some(RescaleMethod::IterativeTentVolumeBilinear),
            "tent-lanczos3-constraint" | "tent_lanczos3_constraint" | "tentlanczos3constraint" | "tent-l3c" | "tent_l3c" | "tl3c" => Some(RescaleMethod::TentLanczos3Constraint),
            _ => None,
        }
    }

    /// Get the kernel radius for this method (0 = full image extent or scale-dependent)
    pub fn base_radius(&self) -> f32 {
        match self {
            RescaleMethod::Bilinear | RescaleMethod::BilinearIterative => 1.0,
            RescaleMethod::Mitchell | RescaleMethod::EWAMitchell => 2.0,
            RescaleMethod::CatmullRom | RescaleMethod::EWACatmullRom => 2.0,
            RescaleMethod::Lanczos2 | RescaleMethod::EWASincLanczos2 | RescaleMethod::EWALanczos2 => 2.0,
            RescaleMethod::Lanczos3 | RescaleMethod::Lanczos3Scatter | RescaleMethod::EWASincLanczos3 | RescaleMethod::EWALanczos3 | RescaleMethod::EWALanczos3Sharp => 3.0,
            RescaleMethod::EWALanczos4Sharpest => 4.0,
            RescaleMethod::Sinc | RescaleMethod::SincScatter | RescaleMethod::Jinc | RescaleMethod::StochasticJinc | RescaleMethod::StochasticJincScatter | RescaleMethod::StochasticJincScatterNormalized => 0.0, // Special: uses full image extent
            RescaleMethod::Box | RescaleMethod::TentBox | RescaleMethod::Tent2DBox | RescaleMethod::TentBoxIterative | RescaleMethod::Tent2DBoxIterative | RescaleMethod::IterativeTentVolume | RescaleMethod::IterativeTentVolumeBilinear | RescaleMethod::TentLanczos3Constraint => 1.0,  // Not used; Box has its own precompute that calculates radius from scale
            RescaleMethod::TentLanczos3 | RescaleMethod::Tent2DLanczos3Jinc => 3.0,  // Uses Lanczos3 internally
        }
    }

    /// Returns true if this method uses full image extent (O(N²))
    pub fn is_full_extent(&self) -> bool {
        matches!(self, RescaleMethod::Sinc | RescaleMethod::SincScatter | RescaleMethod::Jinc | RescaleMethod::StochasticJinc | RescaleMethod::StochasticJincScatter | RescaleMethod::StochasticJincScatterNormalized)
    }

    /// Returns true if this is a scatter-based method
    pub fn is_scatter(&self) -> bool {
        matches!(self, RescaleMethod::Lanczos3Scatter | RescaleMethod::SincScatter | RescaleMethod::StochasticJincScatter | RescaleMethod::StochasticJincScatterNormalized)
    }

    /// Returns true if this is an EWA (Elliptical Weighted Average) method
    pub fn is_ewa(&self) -> bool {
        matches!(self, RescaleMethod::EWASincLanczos2 | RescaleMethod::EWASincLanczos3 |
                       RescaleMethod::EWALanczos2 | RescaleMethod::EWALanczos3 |
                       RescaleMethod::EWALanczos3Sharp | RescaleMethod::EWALanczos4Sharpest |
                       RescaleMethod::EWAMitchell | RescaleMethod::EWACatmullRom |
                       RescaleMethod::Jinc | RescaleMethod::StochasticJinc | RescaleMethod::StochasticJincScatter | RescaleMethod::StochasticJincScatterNormalized)
    }

    /// Get the underlying kernel method for scatter/EWA variants
    pub fn kernel_method(&self) -> RescaleMethod {
        match self {
            RescaleMethod::Lanczos3Scatter => RescaleMethod::Lanczos3,
            RescaleMethod::SincScatter => RescaleMethod::Sinc,
            RescaleMethod::EWASincLanczos2 | RescaleMethod::EWALanczos2 => RescaleMethod::Lanczos2,
            RescaleMethod::EWASincLanczos3 | RescaleMethod::EWALanczos3 => RescaleMethod::Lanczos3,
            // Sharpened variants are self-contained (blur-adjusted jinc)
            RescaleMethod::EWALanczos3Sharp => RescaleMethod::EWALanczos3Sharp,
            RescaleMethod::EWALanczos4Sharpest => RescaleMethod::EWALanczos4Sharpest,
            RescaleMethod::EWAMitchell => RescaleMethod::Mitchell,
            RescaleMethod::EWACatmullRom => RescaleMethod::CatmullRom,
            RescaleMethod::StochasticJinc | RescaleMethod::StochasticJincScatter | RescaleMethod::StochasticJincScatterNormalized => RescaleMethod::Jinc,
            RescaleMethod::TentBox | RescaleMethod::Tent2DBox | RescaleMethod::TentBoxIterative | RescaleMethod::Tent2DBoxIterative | RescaleMethod::IterativeTentVolume | RescaleMethod::TentLanczos3Constraint => RescaleMethod::Box,
            RescaleMethod::IterativeTentVolumeBilinear => RescaleMethod::Bilinear,
            RescaleMethod::TentLanczos3 => RescaleMethod::Lanczos3,
            RescaleMethod::Tent2DLanczos3Jinc => RescaleMethod::EWALanczos3,  // Uses jinc-based Lanczos3
            RescaleMethod::BilinearIterative => RescaleMethod::Bilinear,
            other => *other,
        }
    }

    /// Returns true if this is a tent-space pipeline method
    pub fn is_tent_pipeline(&self) -> bool {
        matches!(self, RescaleMethod::TentBox | RescaleMethod::TentLanczos3 | RescaleMethod::Tent2DBox | RescaleMethod::Tent2DLanczos3Jinc | RescaleMethod::TentBoxIterative | RescaleMethod::Tent2DBoxIterative | RescaleMethod::IterativeTentVolume | RescaleMethod::IterativeTentVolumeBilinear | RescaleMethod::TentLanczos3Constraint)
    }

    /// Returns true if this is a 2D (non-separable) tent-space method
    pub fn is_tent_2d(&self) -> bool {
        matches!(self, RescaleMethod::Tent2DBox | RescaleMethod::Tent2DLanczos3Jinc | RescaleMethod::Tent2DBoxIterative)
    }

    /// Returns true if this is an iterative tent-space method
    pub fn is_tent_iterative(&self) -> bool {
        matches!(self, RescaleMethod::TentBoxIterative | RescaleMethod::Tent2DBoxIterative)
    }

    /// Returns true if this is any iterative downscaling method
    pub fn is_iterative(&self) -> bool {
        matches!(self, RescaleMethod::TentBoxIterative | RescaleMethod::Tent2DBoxIterative | RescaleMethod::BilinearIterative | RescaleMethod::IterativeTentVolume | RescaleMethod::IterativeTentVolumeBilinear)
    }

}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate scale factors based on scale mode
pub fn calculate_scales(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    scale_mode: ScaleMode,
) -> (f32, f32) {
    match scale_mode {
        ScaleMode::Independent => {
            (src_width as f32 / dst_width as f32,
             src_height as f32 / dst_height as f32)
        }
        ScaleMode::UniformWidth => {
            let scale = src_width as f32 / dst_width as f32;
            (scale, scale)
        }
        ScaleMode::UniformHeight => {
            let scale = src_height as f32 / dst_height as f32;
            (scale, scale)
        }
    }
}

/// Check if a number is a power of 2
#[inline]
fn is_power_of_2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Calculate target dimensions preserving aspect ratio
///
/// When both width and height are specified, checks if they're within 1 pixel
/// of uniform aspect ratio. If so, picks the "best" primary dimension:
/// 1. Power of 2 takes precedence
/// 2. Clean division of source dimension takes precedence
/// 3. Otherwise, largest dimension is primary
///
/// Set `force_exact` to true to skip the automatic uniform scaling detection
/// and use the exact dimensions provided (even if they cause slight distortion).
pub fn calculate_target_dimensions(
    src_width: usize,
    src_height: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
) -> (usize, usize) {
    calculate_target_dimensions_exact(src_width, src_height, target_width, target_height, false)
}

/// Calculate target dimensions with explicit control over uniform scaling
///
/// When `force_exact` is false (default), automatically adjusts dimensions to
/// preserve aspect ratio if they're within 1 pixel of uniform scaling.
///
/// When `force_exact` is true, uses the exact dimensions provided without
/// any automatic adjustment, allowing intentional non-uniform scaling.
pub fn calculate_target_dimensions_exact(
    src_width: usize,
    src_height: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
    force_exact: bool,
) -> (usize, usize) {
    match (target_width, target_height) {
        (Some(w), Some(h)) => {
            // If force_exact, skip all automatic adjustment
            if force_exact {
                return (w, h);
            }

            // Calculate what uniform AR would give us from each dimension
            let h_from_w = (w as f64 * src_height as f64 / src_width as f64).round() as usize;
            let w_from_h = (h as f64 * src_width as f64 / src_height as f64).round() as usize;

            // Check if both dimensions are within 1 pixel of uniform AR
            let h_close = (h as isize - h_from_w as isize).abs() <= 1;
            let w_close = (w as isize - w_from_h as isize).abs() <= 1;

            if h_close || w_close {
                // Pick the best primary dimension
                let width_is_pow2 = is_power_of_2(w);
                let height_is_pow2 = is_power_of_2(h);
                let width_divides = src_width % w == 0;
                let height_divides = src_height % h == 0;

                let use_width_as_primary = if width_is_pow2 && !height_is_pow2 {
                    true
                } else if height_is_pow2 && !width_is_pow2 {
                    false
                } else if width_divides && !height_divides {
                    true
                } else if height_divides && !width_divides {
                    false
                } else {
                    // Default: use larger dimension as primary
                    w >= h
                };

                if use_width_as_primary {
                    let aspect = src_height as f64 / src_width as f64;
                    (w, (w as f64 * aspect).round() as usize)
                } else {
                    let aspect = src_width as f64 / src_height as f64;
                    ((h as f64 * aspect).round() as usize, h)
                }
            } else {
                // Dimensions are too different from AR - use exact values (intentional distortion)
                (w, h)
            }
        }
        (Some(w), None) => {
            let aspect = src_height as f64 / src_width as f64;
            (w, (w as f64 * aspect).round() as usize)
        }
        (None, Some(h)) => {
            let aspect = src_width as f64 / src_height as f64;
            ((h as f64 * aspect).round() as usize, h)
        }
        (None, None) => (src_width, src_height),
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Rescale Pixel4 image (SIMD-friendly, linear space)
pub fn rescale(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> Vec<Pixel4> {
    rescale_with_progress_tent(src, src_width, src_height, dst_width, dst_height, method, scale_mode, TentMode::Off, None)
}

/// Rescale Pixel4 image with optional progress callback (SIMD-friendly, linear space)
/// Progress callback receives 0.0-1.0 (0.0 before first row, 1.0 after last row)
pub fn rescale_with_progress(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    rescale_with_progress_tent(src, src_width, src_height, dst_width, dst_height, method, scale_mode, TentMode::Off, progress)
}

/// Rescale Pixel4 image with tent-mode support for supersampling
///
/// `tent_mode` controls coordinate mapping:
/// - `TentMode::Off`: Standard box-to-box mapping (pixel centers to pixel centers)
/// - `TentMode::SampleToSample`: Sample-to-sample mapping for tent-space (position 0→0, N-1→M-1)
/// - `TentMode::Prescale`: Tent-to-box prescale (integrates tent_contract into rescale)
pub fn rescale_with_progress_tent(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    tent_mode: TentMode,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    if src_width == dst_width && src_height == dst_height && tent_mode != TentMode::Prescale {
        if let Some(cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => bilinear::rescale_bilinear_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress),
        RescaleMethod::Mitchell | RescaleMethod::CatmullRom |
        RescaleMethod::Lanczos2 | RescaleMethod::Lanczos3 |
        RescaleMethod::Sinc | RescaleMethod::Box => {
            separable::rescale_kernel_pixels(src, src_width, src_height, dst_width, dst_height, method, scale_mode, tent_mode, progress)
        }
        RescaleMethod::Lanczos3Scatter | RescaleMethod::SincScatter => {
            scatter::rescale_scatter_pixels(src, src_width, src_height, dst_width, dst_height, method, scale_mode, tent_mode, progress)
        }
        RescaleMethod::EWASincLanczos2 | RescaleMethod::EWASincLanczos3 |
        RescaleMethod::EWALanczos2 | RescaleMethod::EWALanczos3 |
        RescaleMethod::EWALanczos3Sharp | RescaleMethod::EWALanczos4Sharpest |
        RescaleMethod::EWAMitchell | RescaleMethod::EWACatmullRom |
        RescaleMethod::Jinc => {
            ewa::rescale_ewa_pixels(src, src_width, src_height, dst_width, dst_height, method, scale_mode, tent_mode, progress)
        }
        RescaleMethod::StochasticJinc => {
            ewa::rescale_stochastic_jinc_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress)
        }
        RescaleMethod::StochasticJincScatter => {
            ewa::rescale_stochastic_jinc_scatter_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress)
        }
        RescaleMethod::StochasticJincScatterNormalized => {
            ewa::rescale_stochastic_jinc_scatter_normalized_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress)
        }
        RescaleMethod::TentBox | RescaleMethod::TentLanczos3 => {
            // Tent-space pipeline as equivalent direct separable kernel.
            // The kernel weights are precomputed to incorporate:
            // 1. Box → Tent expansion (volume-preserving)
            // 2. Resampling in tent space
            // 3. Tent → Box contraction
            // This is more efficient than actually expanding/contracting the image.
            separable::rescale_kernel_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                method, scale_mode,
                TentMode::Off,  // Not using tent_mode - weights already handle tent operations
                progress,
            )
        }
        RescaleMethod::Tent2DBox | RescaleMethod::Tent2DLanczos3Jinc => {
            // 2D Tent-space pipeline using non-separable kernel.
            // Uses precomputed 2D kernel weights that incorporate the full
            // expand → resample → contract pipeline in a single convolution.
            tent_ewa::rescale_tent_2d_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                method, scale_mode,
                progress,
            )
        }
        RescaleMethod::TentBoxIterative => {
            // Iterative 2× tent-box downscaling using separable kernels.
            // Applies 2× downscales iteratively, then final factor.
            tent_ewa::rescale_tent_iterative_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                false,  // separable
                progress,
            )
        }
        RescaleMethod::Tent2DBoxIterative => {
            // Iterative 2× tent-box downscaling using 2D kernels.
            // Applies 2× downscales iteratively, then final factor.
            tent_ewa::rescale_tent_iterative_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                true,  // 2D
                progress,
            )
        }
        RescaleMethod::BilinearIterative => {
            // Iterative 2× bilinear downscaling (mipmap-style).
            tent_ewa::rescale_bilinear_iterative_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                progress,
            )
        }
        RescaleMethod::IterativeTentVolume => {
            // Iterative tent volume scaling with box filter in tent-space.
            tent_ewa::rescale_iterative_tent_volume_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                false,  // use_bilinear = false (box)
                progress,
            )
        }
        RescaleMethod::IterativeTentVolumeBilinear => {
            // Iterative tent volume scaling with bilinear filter in tent-space.
            tent_ewa::rescale_iterative_tent_volume_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                true,  // use_bilinear = true
                progress,
            )
        }
        RescaleMethod::TentLanczos3Constraint => {
            // Tent-space with Lanczos3 constraint (instead of volume constraint).
            tent_ewa::rescale_tent_lanczos3_constraint_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                progress,
            )
        }
    }
}

/// Alpha-aware rescale for RGBA images
///
/// Unlike regular rescaling which treats all 4 channels equally, this function
/// weights RGB contributions by alpha during interpolation. This prevents
/// transparent pixels (which often have arbitrary RGB values, e.g., black in
/// dithered images) from bleeding their color into opaque regions.
///
/// Behavior:
/// - Opaque regions: RGB interpolated normally (alpha weights are ~1.0)
/// - Mixed regions: opaque pixels dominate RGB (weighted by their alpha)
/// - Fully transparent regions: RGB interpolated normally (fallback preserves underlying color)
/// - Alpha channel: always interpolated normally
pub fn rescale_with_alpha(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
) -> Vec<Pixel4> {
    rescale_with_alpha_progress_tent(src, src_width, src_height, dst_width, dst_height, method, scale_mode, TentMode::Off, None)
}

/// Alpha-aware rescale with progress callback
pub fn rescale_with_alpha_progress(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    rescale_with_alpha_progress_tent(src, src_width, src_height, dst_width, dst_height, method, scale_mode, TentMode::Off, progress)
}

/// Alpha-aware rescale with tent-mode support for supersampling
pub fn rescale_with_alpha_progress_tent(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: RescaleMethod,
    scale_mode: ScaleMode,
    tent_mode: TentMode,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<Pixel4> {
    if src_width == dst_width && src_height == dst_height && tent_mode != TentMode::Prescale {
        if let Some(cb) = progress {
            cb(1.0);
        }
        return src.to_vec();
    }

    match method {
        RescaleMethod::Bilinear => bilinear::rescale_bilinear_alpha_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress),
        RescaleMethod::Mitchell | RescaleMethod::CatmullRom |
        RescaleMethod::Lanczos2 | RescaleMethod::Lanczos3 |
        RescaleMethod::Sinc | RescaleMethod::Box => {
            separable::rescale_kernel_alpha_pixels(src, src_width, src_height, dst_width, dst_height, method, scale_mode, tent_mode, progress)
        }
        RescaleMethod::Lanczos3Scatter | RescaleMethod::SincScatter => {
            scatter::rescale_scatter_alpha_pixels(src, src_width, src_height, dst_width, dst_height, method, scale_mode, tent_mode, progress)
        }
        RescaleMethod::EWASincLanczos2 | RescaleMethod::EWASincLanczos3 |
        RescaleMethod::EWALanczos2 | RescaleMethod::EWALanczos3 |
        RescaleMethod::EWALanczos3Sharp | RescaleMethod::EWALanczos4Sharpest |
        RescaleMethod::EWAMitchell | RescaleMethod::EWACatmullRom |
        RescaleMethod::Jinc => {
            ewa::rescale_ewa_alpha_pixels(src, src_width, src_height, dst_width, dst_height, method, scale_mode, tent_mode, progress)
        }
        RescaleMethod::StochasticJinc => {
            ewa::rescale_stochastic_jinc_alpha_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress)
        }
        RescaleMethod::StochasticJincScatter => {
            ewa::rescale_stochastic_jinc_scatter_alpha_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress)
        }
        RescaleMethod::StochasticJincScatterNormalized => {
            ewa::rescale_stochastic_jinc_scatter_normalized_alpha_pixels(src, src_width, src_height, dst_width, dst_height, scale_mode, tent_mode, progress)
        }
        RescaleMethod::TentBox | RescaleMethod::TentLanczos3 => {
            // Tent-space pipeline as equivalent direct separable kernel (alpha-aware).
            // The kernel weights incorporate expand → resample → contract operations.
            separable::rescale_kernel_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                method, scale_mode,
                TentMode::Off,  // Not using tent_mode - weights already handle tent operations
                progress,
            )
        }
        RescaleMethod::Tent2DBox | RescaleMethod::Tent2DLanczos3Jinc => {
            // 2D Tent-space pipeline using non-separable kernel (alpha-aware).
            tent_ewa::rescale_tent_2d_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                method, scale_mode,
                progress,
            )
        }
        RescaleMethod::TentBoxIterative => {
            // Iterative 2× tent-box downscaling using separable kernels (alpha-aware).
            tent_ewa::rescale_tent_iterative_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                false,  // separable
                progress,
            )
        }
        RescaleMethod::Tent2DBoxIterative => {
            // Iterative 2× tent-box downscaling using 2D kernels (alpha-aware).
            tent_ewa::rescale_tent_iterative_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                true,  // 2D
                progress,
            )
        }
        RescaleMethod::BilinearIterative => {
            // Iterative 2× bilinear downscaling (alpha-aware).
            tent_ewa::rescale_bilinear_iterative_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                progress,
            )
        }
        RescaleMethod::IterativeTentVolume => {
            // Iterative tent volume scaling with box filter (alpha-aware).
            tent_ewa::rescale_iterative_tent_volume_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                false,  // use_bilinear = false (box)
                progress,
            )
        }
        RescaleMethod::IterativeTentVolumeBilinear => {
            // Iterative tent volume scaling with bilinear filter (alpha-aware).
            tent_ewa::rescale_iterative_tent_volume_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                true,  // use_bilinear = true
                progress,
            )
        }
        RescaleMethod::TentLanczos3Constraint => {
            // Tent-space with Lanczos3 constraint (alpha-aware).
            tent_ewa::rescale_tent_lanczos3_constraint_alpha_pixels(
                src, src_width, src_height,
                dst_width, dst_height,
                scale_mode,
                progress,
            )
        }
    }
}
