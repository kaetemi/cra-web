/// Common types and utilities shared across dithering implementations.
///
/// Provides:
/// - `PerceptualSpace`: Enum for selecting color space and distance metric
/// - `DitherMode`: Enum for selecting dithering algorithm and scanning mode
/// - `linear_rgb_to_perceptual`: Convert linear RGB to perceptual space coordinates
/// - `bit_replicate`: Extend n-bit values to 8 bits
/// - `wang_hash`: Deterministic hash for random number generation

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_rgb_to_ycbcr, linear_rgb_to_ycbcr_bt601,
    linear_rgb_to_ycbcr_bt601_clamped, linear_rgb_to_ycbcr_clamped, linear_to_srgb_single,
};

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
    /// OKLab color space with Euclidean distance (default)
    /// Designed so Euclidean distance is perceptually uniform
    #[default]
    OkLab,
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
    },
}

impl Default for OutputTechnique {
    fn default() -> Self {
        OutputTechnique::ColorspaceAware {
            mode: DitherMode::default(),
            space: PerceptualSpace::default(),
            alpha_mode: None,
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
        PerceptualSpace::OkLab => linear_rgb_to_oklab(r, g, b),
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
        PerceptualSpace::OkLab => linear_rgb_to_oklab(r_clamped, g_clamped, b_clamped),
    }
}

// ============================================================================
// Single-channel error diffusion kernel
// ============================================================================

/// Apply Floyd-Steinberg or Jarvis-Judice-Ninke error diffusion to a single channel.
///
/// This is used by mixed-mode dithering where each channel can use a different kernel.
/// The kernel is selected at runtime based on the `use_jjn` flag.
///
/// Args:
///     err: Error buffer for one channel (2D array indexed as err[y][x])
///     bx: Buffer x coordinate (includes padding for kernel reach)
///     y: Buffer y coordinate
///     err_val: Error value to diffuse
///     use_jjn: If true, use Jarvis-Judice-Ninke; if false, use Floyd-Steinberg
///     is_rtl: If true, diffuse right-to-left; if false, left-to-right
#[inline]
pub fn apply_single_channel_kernel(
    err: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_val: f32,
    use_jjn: bool,
    is_rtl: bool,
) {
    match (use_jjn, is_rtl) {
        (true, false) => {
            // JJN LTR
            err[y][bx + 1] += err_val * (7.0 / 48.0);
            err[y][bx + 2] += err_val * (5.0 / 48.0);
            err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
            err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx] += err_val * (7.0 / 48.0);
            err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
            err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
            err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx] += err_val * (5.0 / 48.0);
            err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
        }
        (true, true) => {
            // JJN RTL
            err[y][bx - 1] += err_val * (7.0 / 48.0);
            err[y][bx - 2] += err_val * (5.0 / 48.0);
            err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
            err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx] += err_val * (7.0 / 48.0);
            err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
            err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
            err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx] += err_val * (5.0 / 48.0);
            err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
        }
        (false, false) => {
            // FS LTR
            err[y][bx + 1] += err_val * (7.0 / 16.0);
            err[y + 1][bx - 1] += err_val * (3.0 / 16.0);
            err[y + 1][bx] += err_val * (5.0 / 16.0);
            err[y + 1][bx + 1] += err_val * (1.0 / 16.0);
        }
        (false, true) => {
            // FS RTL
            err[y][bx - 1] += err_val * (7.0 / 16.0);
            err[y + 1][bx + 1] += err_val * (3.0 / 16.0);
            err[y + 1][bx] += err_val * (5.0 / 16.0);
            err[y + 1][bx - 1] += err_val * (1.0 / 16.0);
        }
    }
}

/// Apply mixed-mode error diffusion to RGB channels (3 separate buffers).
///
/// Each channel independently selects between Floyd-Steinberg and Jarvis-Judice-Ninke
/// based on bits from the pixel_hash. This creates texture variety while maintaining
/// deterministic results from the seed.
///
/// Bit assignment: R=bit0, G=bit1, B=bit2
#[inline]
pub fn apply_mixed_kernel_rgb(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    pixel_hash: u32,
    is_rtl: bool,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;

    apply_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
}

/// Apply mixed-mode error diffusion to RGBA channels (4 separate buffers).
///
/// Each channel independently selects between Floyd-Steinberg and Jarvis-Judice-Ninke
/// based on bits from the pixel_hash. This creates texture variety while maintaining
/// deterministic results from the seed.
///
/// Bit assignment: R=bit0, G=bit1, B=bit2, A=bit3
#[inline]
pub fn apply_mixed_kernel_rgba(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    err_a_val: f32,
    pixel_hash: u32,
    is_rtl: bool,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;
    let use_jjn_a = pixel_hash & 8 != 0;

    apply_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
    apply_single_channel_kernel(err_a, bx, y, err_a_val, use_jjn_a, is_rtl);
}

// ============================================================================
// Compile-time kernel traits for monomorphization
// ============================================================================

/// Single-channel error diffusion kernel trait.
///
/// Enables compile-time dispatch for dithering algorithms. Each implementation
/// provides left-to-right and right-to-left variants for serpentine scanning.
///
/// Buffer structure with seeding (to normalize edge dithering):
/// ```text
/// [overshoot] [seeding] [real image] [seeding] [overshoot]
/// ```
/// - Seeding columns/rows: filled with duplicated edge pixels, ARE processed
/// - Overshoot: initialized to zero, catches error diffusion, NOT processed
pub trait SingleChannelKernel {
    /// Maximum reach of error diffusion in any direction.
    /// Floyd-Steinberg: 1, Jarvis-Judice-Ninke: 2, None: 0
    const REACH: usize;

    /// Apply kernel for left-to-right scanning.
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32);

    /// Apply kernel for right-to-left scanning (mirrored).
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32);

    // Derived constants for buffer layout
    /// Total left padding (overshoot + seeding)
    const TOTAL_LEFT: usize = Self::REACH * 2;
    /// Total right padding (seeding + overshoot)
    const TOTAL_RIGHT: usize = Self::REACH * 2;
    /// Total top padding (seeding only, no overshoot since error flows down)
    const TOTAL_TOP: usize = Self::REACH;
    /// Total bottom padding (overshoot only, no seeding since error comes from above)
    const TOTAL_BOTTOM: usize = Self::REACH;
    /// Offset from buffer edge to start of seeding area (= overshoot size)
    const SEED_OFFSET: usize = Self::REACH;
}

/// RGB (3-channel) error diffusion kernel trait.
///
/// Applies the same kernel pattern to three separate error buffers.
pub trait RgbKernel {
    /// Maximum reach of error diffusion in any direction.
    const REACH: usize;

    /// Apply kernel to RGB channels for left-to-right scanning.
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    );

    /// Apply kernel to RGB channels for right-to-left scanning.
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    );

    // Derived constants for buffer layout
    const TOTAL_LEFT: usize = Self::REACH * 2;
    const TOTAL_RIGHT: usize = Self::REACH * 2;
    const TOTAL_TOP: usize = Self::REACH;
    const TOTAL_BOTTOM: usize = Self::REACH;
    const SEED_OFFSET: usize = Self::REACH;
}

/// RGBA (4-channel) error diffusion kernel trait.
///
/// Applies the same kernel pattern to four separate error buffers.
pub trait RgbaKernel {
    /// Maximum reach of error diffusion in any direction.
    const REACH: usize;

    /// Apply kernel to RGBA channels for left-to-right scanning.
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        err_a: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        err_a_val: f32,
    );

    /// Apply kernel to RGBA channels for right-to-left scanning.
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        err_a: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        err_a_val: f32,
    );

    // Derived constants for buffer layout
    const TOTAL_LEFT: usize = Self::REACH * 2;
    const TOTAL_RIGHT: usize = Self::REACH * 2;
    const TOTAL_TOP: usize = Self::REACH;
    const TOTAL_BOTTOM: usize = Self::REACH;
    const SEED_OFFSET: usize = Self::REACH;
}

// ============================================================================
// Floyd-Steinberg kernel implementation
// ============================================================================

/// Floyd-Steinberg error diffusion kernel.
///
/// Compact 2-row kernel with good speed/quality trade-off.
/// Kernel weights (divided by 16):
/// ```text
///       * 7
///     3 5 1
/// ```
pub struct FloydSteinberg;

impl SingleChannelKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
        apply_single_channel_kernel(buf, bx, y, err, false, false);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
        apply_single_channel_kernel(buf, bx, y, err, false, true);
    }
}

impl RgbKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
    }
}

impl RgbaKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        err_a: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        err_a_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_ltr(err_a, bx, y, err_a_val);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        err_a: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        err_a_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_rtl(err_a, bx, y, err_a_val);
    }
}

// ============================================================================
// Jarvis-Judice-Ninke kernel implementation
// ============================================================================

/// Jarvis-Judice-Ninke error diffusion kernel.
///
/// Larger 3-row kernel produces smoother gradients than Floyd-Steinberg.
/// Kernel weights (divided by 48):
/// ```text
///         * 7 5
///     3 5 7 5 3
///     1 3 5 3 1
/// ```
pub struct JarvisJudiceNinke;

impl SingleChannelKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
        apply_single_channel_kernel(buf, bx, y, err, true, false);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
        apply_single_channel_kernel(buf, bx, y, err, true, true);
    }
}

impl RgbKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
    }
}

impl RgbaKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        err_a: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        err_a_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_ltr(err_a, bx, y, err_a_val);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        err_a: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        err_a_val: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_rtl(err_a, bx, y, err_a_val);
    }
}

// ============================================================================
// No-op kernel implementation
// ============================================================================

/// No-op kernel that discards error (no diffusion).
///
/// Each pixel is independently quantized to nearest level.
/// Produces banding but useful as a baseline for comparison.
pub struct NoneKernel;

impl SingleChannelKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(_buf: &mut [Vec<f32>], _bx: usize, _y: usize, _err: f32) {}

    #[inline]
    fn apply_rtl(_buf: &mut [Vec<f32>], _bx: usize, _y: usize, _err: f32) {}
}

impl RgbKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
    ) {
    }

    #[inline]
    fn apply_rtl(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
    ) {
    }
}

impl RgbaKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _err_a: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
        _err_a_val: f32,
    ) {
    }

    #[inline]
    fn apply_rtl(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _err_a: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
        _err_a_val: f32,
    ) {
    }
}
