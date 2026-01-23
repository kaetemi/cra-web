/// Common types and utilities shared across dithering implementations.
///
/// Provides:
/// - `PerceptualSpace`: Enum for selecting color space and distance metric
/// - `DitherMode`: Enum for selecting dithering algorithm and scanning mode
/// - `linear_rgb_to_perceptual`: Convert linear RGB to perceptual space coordinates
/// - `bit_replicate`: Extend n-bit values to 8 bits
/// - `wang_hash`: Deterministic hash for random number generation

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_rgb_to_oklab_lr, linear_rgb_to_ycbcr,
    linear_rgb_to_ycbcr_bt601, linear_rgb_to_ycbcr_bt601_clamped, linear_rgb_to_ycbcr_clamped,
    linear_to_srgb_single,
};
use crate::colorspace_derived::f32 as cs;

// Re-export bit depth utilities for backwards compatibility
pub use super::bitdepth::{bit_replicate, build_linear_lut, QuantLevelParams};

/// Perceptual color space and distance metric for candidate selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerceptualSpace {
    /// CIELAB with CIE76 (ΔE*ab): Simple Euclidean distance
    /// Fast but perceptually non-uniform, over-weights chromaticity
    LabCIE76,
    /// CIELAB with CIE94: Weighted distance for perceptual uniformity
    /// Down-weights chromatic differences for saturated colors
    LabCIE94,
    /// CIELAB with CIEDE2000: Most accurate perceptual distance
    /// Includes corrections for blue region, lightness, and chroma
    LabCIEDE2000,
    /// OKLab color space with Euclidean distance
    /// Designed so Euclidean distance is perceptually uniform
    OkLab,
    /// OKLab with revised lightness (Lr) for better Munsell Value matching (default)
    /// Uses Ottosson's Lr formula which expands dark values compared to standard L
    /// Better for palettes where dark colors should stay distinct from grays
    #[default]
    OkLabLr,
    /// OKLab with heavy chroma weighting (×4) for dithering
    /// Penalizes chromatic differences more heavily, encouraging neutral (light/dark)
    /// oscillations rather than chromatic (complementary color) oscillations.
    /// Useful for limited palettes where CIE76-style chroma weighting works better.
    OkLabHeavyChroma,
    /// Linear RGB with Euclidean distance (NOT RECOMMENDED)
    /// Simple Euclidean distance in linear RGB space - not perceptually uniform,
    /// provided for testing and comparison purposes only
    LinearRGB,
    /// Y'CbCr with Euclidean distance (NOT RECOMMENDED)
    /// Luma-chroma separation using BT.709 coefficients on gamma-encoded values.
    /// Not perceptually uniform - provided for testing and comparison purposes only
    YCbCr,
    /// sRGB with Euclidean distance (NOT RECOMMENDED)
    /// Simple Euclidean distance in gamma-encoded sRGB space.
    /// Not perceptually uniform - provided for testing and comparison purposes only
    Srgb,
    /// Y'CbCr BT.601 (legacy) with Euclidean distance (NOT RECOMMENDED)
    /// Luma-chroma separation using legacy BT.601 coefficients (0.299/0.587/0.114).
    /// This is the JPEG/ITU-T T.871 encoding, historically from NTSC 1953.
    /// Not perceptually uniform - provided for compatibility testing only
    YCbCrBt601,
}

/// Output dithering technique selection
///
/// Controls how RGB values are quantized to the target bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTechnique {
    /// No dithering - simple rounding/truncation
    /// Fastest but can cause visible banding
    None,
    /// Per-channel error diffusion dithering
    /// Each RGB channel is dithered independently with its own error buffer
    /// Fast and works well for most cases
    PerChannel {
        mode: DitherMode,
        /// Optional separate mode for alpha channel (defaults to mode if None)
        alpha_mode: Option<DitherMode>,
    },
    /// Colorspace-aware joint RGB dithering (default)
    /// Processes RGB channels together, selecting the quantized color that
    /// minimizes perceptual distance. Higher quality but slower.
    ColorspaceAware {
        mode: DitherMode,
        space: PerceptualSpace,
        /// Optional separate mode for alpha channel (defaults to mode if None)
        alpha_mode: Option<DitherMode>,
        /// Whether to apply gamut overshoot penalty (reduces color fringing)
        overshoot_penalty: bool,
    },
}

impl Default for OutputTechnique {
    fn default() -> Self {
        OutputTechnique::ColorspaceAware {
            mode: DitherMode::default(),
            space: PerceptualSpace::default(),
            alpha_mode: None,
            overshoot_penalty: true,
        }
    }
}

/// Color correction method selection
///
/// Selects the algorithm and color space for histogram matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorCorrectionMethod {
    /// Basic histogram matching in LAB color space
    /// Matches L, a, b channels independently
    BasicLab {
        /// If true, preserve original L channel (luminosity)
        keep_luminosity: bool,
    },
    /// Basic histogram matching in RGB color space
    /// Matches R, G, B channels independently
    BasicRgb,
    /// Basic histogram matching in OKLab color space
    /// Matches L, a, b channels independently
    /// OKLab provides better perceptual uniformity than LAB
    BasicOklab {
        /// If true, preserve original L channel (luminosity)
        keep_luminosity: bool,
    },
    /// Chroma Rotation Averaging in LAB color space
    /// Rotates the AB chroma plane at multiple angles, performs histogram
    /// matching at each rotation, then averages the results
    CraLab {
        /// If true, preserve original L channel (luminosity)
        keep_luminosity: bool,
    },
    /// Chroma Rotation Averaging in RGB color space
    /// Rotates the RGB cube around the neutral gray axis (1,1,1)
    CraRgb {
        /// If true, use perceptual weighting for rotation averaging
        use_perceptual: bool,
    },
    /// Chroma Rotation Averaging in OKLab color space
    /// Like CRA LAB but uses OKLab for better perceptual uniformity
    CraOklab {
        /// If true, preserve original L channel (luminosity)
        keep_luminosity: bool,
    },
    /// Tiled CRA in LAB color space
    /// Divides image into overlapping tiles with Hamming window blending
    TiledLab {
        /// If true, process L channel per-tile before global match
        tiled_luminosity: bool,
    },
    /// Tiled CRA in OKLab color space
    /// Divides image into overlapping tiles with Hamming window blending
    TiledOklab {
        /// If true, process L channel per-tile before global match
        tiled_luminosity: bool,
    },
}

impl Default for ColorCorrectionMethod {
    fn default() -> Self {
        ColorCorrectionMethod::BasicOklab {
            keep_luminosity: false,
        }
    }
}

/// Histogram mode for color correction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistogramMode {
    /// uint8 binned histogram (256 bins)
    /// Fastest, but quantizes to 256 levels
    Binned,
    /// f32 endpoint-aligned quantile matching
    /// Preserves exact min/max of reference
    #[default]
    EndpointAligned,
    /// f32 midpoint-aligned quantile matching
    /// More statistically correct, doesn't force range expansion
    MidpointAligned,
}

/// Dithering mode selection for color-space aware dithering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitherMode {
    /// No error diffusion - each pixel quantized independently
    /// Produces banding but useful as a baseline or for testing
    None,
    /// Floyd-Steinberg: Standard left-to-right scanning (default)
    #[default]
    Standard,
    /// Floyd-Steinberg: Serpentine scanning (alternating direction each row)
    /// Reduces diagonal banding artifacts
    Serpentine,
    /// Jarvis-Judice-Ninke: Standard left-to-right scanning
    /// Larger kernel (3 rows) produces smoother results but slower
    JarvisStandard,
    /// Jarvis-Judice-Ninke: Serpentine scanning
    /// Combines larger kernel with alternating scan direction
    JarvisSerpentine,
    /// Mixed: Randomly selects between FS and JJN kernels per-pixel
    /// Standard left-to-right scanning
    MixedStandard,
    /// Mixed: Randomly selects between FS and JJN kernels per-pixel
    /// Serpentine scanning (alternating direction each row)
    MixedSerpentine,
    /// Mixed: Randomly selects between FS and JJN kernels per-pixel
    /// AND randomly selects scan direction per row
    MixedRandom,
    /// Ostromoukhov: Variable-coefficient kernel based on input intensity
    /// Standard left-to-right scanning
    OstromoukhovStandard,
    /// Ostromoukhov: Variable-coefficient kernel based on input intensity
    /// Serpentine scanning (alternating direction each row)
    OstromoukhovSerpentine,
}

/// Wang hash for deterministic randomization - excellent avalanche properties.
/// Each bit of input affects all bits of output.
#[inline]
pub fn wang_hash(mut x: u32) -> u32 {
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_mul(9);
    x = x ^ (x >> 4);
    x = x.wrapping_mul(0x27d4eb2d);
    x = x ^ (x >> 15);
    x
}

// ============================================================================
// Gamut overshoot penalty helpers
// ============================================================================

/// Calculate distance from a linear RGB point to the [0,1]³ RGB cube.
/// Returns 0.0 if the point is inside the cube.
///
/// Used for gamut overshoot penalty: when choosing a quantized color,
/// we penalize choices that would push the error diffusion outside the
/// representable color gamut.
#[inline]
pub fn rgb_cube_overshoot_distance(r: f32, g: f32, b: f32) -> f32 {
    // Clamp to [0, 1] cube
    let r_clamped = r.clamp(0.0, 1.0);
    let g_clamped = g.clamp(0.0, 1.0);
    let b_clamped = b.clamp(0.0, 1.0);

    // Euclidean distance from original to clamped
    let dr = r - r_clamped;
    let dg = g - g_clamped;
    let db = b - b_clamped;
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Calculate the gamut overshoot penalty multiplier.
///
/// Given the target linear RGB (with error already added) and a candidate
/// quantized linear RGB, computes the "opposing point" where error diffusion
/// would push neighboring pixels: opposing = 2*target - candidate.
///
/// If this opposing point is outside the [0,1]³ RGB cube, returns a penalty
/// multiplier > 1 that scales the perceptual distance, discouraging choices
/// that cause large unrecoverable errors.
///
/// Penalty formula: (overshoot_distance + 1)²
#[inline]
pub fn gamut_overshoot_penalty(
    target_r: f32, target_g: f32, target_b: f32,
    candidate_r: f32, candidate_g: f32, candidate_b: f32,
) -> f32 {
    // Calculate the opposing point: where error diffusion would push neighbors
    let opposing_r = 2.0 * target_r - candidate_r;
    let opposing_g = 2.0 * target_g - candidate_g;
    let opposing_b = 2.0 * target_b - candidate_b;

    // Calculate overshoot distance
    let overshoot = rgb_cube_overshoot_distance(opposing_r, opposing_g, opposing_b);

    // Penalty multiplier: (overshoot + 1)²
    let factor = overshoot + 1.0;
    factor * factor
}

/// Calculate the grayscale gamut overshoot penalty multiplier.
///
/// Similar to RGB version but for single-channel grayscale dithering.
/// The gamut is simply [0, 1] on the linear gray axis.
///
/// Penalty formula: (overshoot_distance + 1)²
#[inline]
pub fn gray_overshoot_penalty(target_gray: f32, candidate_gray: f32) -> f32 {
    // Calculate the opposing point
    let opposing = 2.0 * target_gray - candidate_gray;

    // Calculate overshoot (distance outside [0, 1])
    let overshoot = if opposing < 0.0 {
        -opposing
    } else if opposing > 1.0 {
        opposing - 1.0
    } else {
        0.0
    };

    // Penalty multiplier: (overshoot + 1)²
    let factor = overshoot + 1.0;
    factor * factor
}

// ============================================================================
// Perceptual space conversion helpers
// ============================================================================

/// Convert linear RGB to perceptual space coordinates (unclamped).
///
/// Use this for distance calculation where out-of-gamut values should be
/// preserved for accurate error measurement. Values outside [0,1] are passed
/// through to the conversion functions (except sRGB which requires clamping).
///
/// Returns (L/Y/R, a/Cb/G, b/Cr/B) depending on the color space.
#[inline]
pub fn linear_rgb_to_perceptual(space: PerceptualSpace, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    match space {
        PerceptualSpace::Srgb => {
            // sRGB requires clamping for valid gamma conversion
            let r_clamped = r.clamp(0.0, 1.0);
            let g_clamped = g.clamp(0.0, 1.0);
            let b_clamped = b.clamp(0.0, 1.0);
            (
                linear_to_srgb_single(r_clamped),
                linear_to_srgb_single(g_clamped),
                linear_to_srgb_single(b_clamped),
            )
        }
        PerceptualSpace::LinearRGB => (r, g, b),
        PerceptualSpace::YCbCr => linear_rgb_to_ycbcr(r, g, b),
        PerceptualSpace::YCbCrBt601 => linear_rgb_to_ycbcr_bt601(r, g, b),
        PerceptualSpace::LabCIE76 | PerceptualSpace::LabCIE94 | PerceptualSpace::LabCIEDE2000 => {
            linear_rgb_to_lab(r, g, b)
        }
        PerceptualSpace::OkLab | PerceptualSpace::OkLabHeavyChroma => linear_rgb_to_oklab(r, g, b),
        PerceptualSpace::OkLabLr => linear_rgb_to_oklab_lr(r, g, b),
    }
}

/// Convert linear RGB to perceptual space coordinates (clamped).
///
/// Use this for building LUTs or when the input must be valid (in-gamut).
/// All inputs are clamped to [0,1] before conversion.
///
/// Returns (L/Y/R, a/Cb/G, b/Cr/B) depending on the color space.
#[inline]
pub fn linear_rgb_to_perceptual_clamped(
    space: PerceptualSpace,
    r: f32,
    g: f32,
    b: f32,
) -> (f32, f32, f32) {
    let r_clamped = r.clamp(0.0, 1.0);
    let g_clamped = g.clamp(0.0, 1.0);
    let b_clamped = b.clamp(0.0, 1.0);

    match space {
        PerceptualSpace::Srgb => (
            linear_to_srgb_single(r_clamped),
            linear_to_srgb_single(g_clamped),
            linear_to_srgb_single(b_clamped),
        ),
        PerceptualSpace::LinearRGB => (r_clamped, g_clamped, b_clamped),
        PerceptualSpace::YCbCr => linear_rgb_to_ycbcr_clamped(r_clamped, g_clamped, b_clamped),
        PerceptualSpace::YCbCrBt601 => {
            linear_rgb_to_ycbcr_bt601_clamped(r_clamped, g_clamped, b_clamped)
        }
        PerceptualSpace::LabCIE76 | PerceptualSpace::LabCIE94 | PerceptualSpace::LabCIEDE2000 => {
            linear_rgb_to_lab(r_clamped, g_clamped, b_clamped)
        }
        PerceptualSpace::OkLab | PerceptualSpace::OkLabHeavyChroma => {
            linear_rgb_to_oklab(r_clamped, g_clamped, b_clamped)
        }
        PerceptualSpace::OkLabLr => linear_rgb_to_oklab_lr(r_clamped, g_clamped, b_clamped),
    }
}

// ============================================================================
// Lightness distance functions (for grayscale dithering)
// ============================================================================

/// Simple lightness distance squared (ΔL²).
/// For grayscale with CIE76/CIE94/OKLab, distance reduces to this
/// because a* = b* = 0 for neutral grays.
#[inline]
pub fn lightness_distance_sq(l1: f32, l2: f32) -> f32 {
    let dl = l1 - l2;
    dl * dl
}

/// CIEDE2000 lightness distance for grayscale.
/// Unlike CIE76/CIE94, CIEDE2000 uses a lightness weighting factor SL
/// that depends on the average lightness of the two colors.
/// This compensates for reduced human sensitivity in dark/light regions.
///
/// Formula: ΔE² = (ΔL / SL)²
/// Where: SL = 1 + (K2 × (L̄ - 50)²) / √(20 + (L̄ - 50)²)
///        L̄ = (L1 + L2) / 2
///        K2 = 0.015 (from CIE94, shared with CIEDE2000)
#[inline]
pub fn lightness_distance_ciede2000_sq(l1: f32, l2: f32) -> f32 {
    let dl = l1 - l2;
    let l_bar = (l1 + l2) / 2.0;
    let l_bar_minus_mid = l_bar - cs::CIEDE2000_SL_L_MIDPOINT;
    let l_bar_minus_mid_sq = l_bar_minus_mid * l_bar_minus_mid;
    let sl = 1.0
        + (cs::CIE94_K2 * l_bar_minus_mid_sq)
            / (cs::CIEDE2000_SL_DENOM_OFFSET + l_bar_minus_mid_sq).sqrt();
    let dl_term = dl / sl;
    dl_term * dl_term
}

/// Compute grayscale perceptual distance based on the selected space/metric.
/// For grayscale, chroma components are zero so distance reduces to lightness only.
#[inline]
pub fn perceptual_lightness_distance_sq(space: PerceptualSpace, l1: f32, l2: f32) -> f32 {
    match space {
        // CIE76 and CIE94 reduce to simple ΔL² for neutral grays (a=b=0)
        PerceptualSpace::LabCIE76 | PerceptualSpace::LabCIE94 => lightness_distance_sq(l1, l2),
        // CIEDE2000 uses SL weighting based on average lightness
        PerceptualSpace::LabCIEDE2000 => lightness_distance_ciede2000_sq(l1, l2),
        // All other spaces use simple Euclidean distance, which reduces to ΔL² for grays
        // OkLabHeavyChroma also reduces to ΔL² since there's no chroma component
        PerceptualSpace::OkLab
        | PerceptualSpace::OkLabLr
        | PerceptualSpace::OkLabHeavyChroma
        | PerceptualSpace::LinearRGB
        | PerceptualSpace::YCbCr
        | PerceptualSpace::YCbCrBt601
        | PerceptualSpace::Srgb => lightness_distance_sq(l1, l2),
    }
}

// Re-export kernel types from dedicated module
pub use super::kernels::{
    apply_mixed_kernel_rgb, apply_mixed_kernel_rgba, apply_single_channel_kernel, FloydSteinberg,
    JarvisJudiceNinke, NoneKernel, Ostromoukhov, RgbKernel, RgbaKernel, SingleChannelKernel,
};
