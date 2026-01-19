/// Common types and utilities shared across dithering implementations.
///
/// Provides:
/// - `PerceptualSpace`: Enum for selecting color space and distance metric
/// - `DitherMode`: Enum for selecting dithering algorithm and scanning mode
/// - `bit_replicate`: Extend n-bit values to 8 bits
/// - `wang_hash`: Deterministic hash for random number generation

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
