//! Command-line argument definitions and type conversions

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use cra_wasm::dither::common::ColorCorrectionMethod;
use cra_wasm::dither::common::{DitherMode, HistogramMode as LibHistogramMode, PerceptualSpace};
use cra_wasm::dither::rgb::DitherMode as CSDitherMode;

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Histogram {
    /// No histogram matching (dither only)
    None,
    /// Basic LAB histogram matching
    BasicLab,
    /// Basic RGB histogram matching
    BasicRgb,
    /// Basic Oklab histogram matching (perceptually uniform)
    BasicOklab,
    /// CRA LAB - Chroma Rotation Averaging in LAB space
    CraLab,
    /// CRA RGB - Chroma Rotation Averaging in RGB space
    CraRgb,
    /// CRA Oklab - Chroma Rotation Averaging in Oklab space (perceptually uniform)
    CraOklab,
    /// Tiled LAB with overlapping blocks
    TiledLab,
    /// Tiled Oklab with overlapping blocks (perceptually uniform)
    TiledOklab,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DitherMethod {
    /// Floyd-Steinberg with standard left-to-right scanning
    FsStandard,
    /// Floyd-Steinberg with serpentine (alternating) scanning
    FsSerpentine,
    /// Jarvis-Judice-Ninke with standard scanning (larger kernel, smoother)
    JjnStandard,
    /// Jarvis-Judice-Ninke with serpentine scanning
    JjnSerpentine,
    /// Mixed: randomly selects FS or JJN per-pixel, standard scanning
    MixedStandard,
    /// Mixed: randomly selects FS or JJN per-pixel, serpentine scanning
    MixedSerpentine,
    /// Mixed: randomly selects kernel AND scan direction per-row
    MixedRandom,
    /// No error diffusion - each pixel quantized independently (produces banding)
    None,
}

impl DitherMethod {
    pub fn to_dither_mode(self) -> DitherMode {
        match self {
            DitherMethod::FsStandard => DitherMode::Standard,
            DitherMethod::FsSerpentine => DitherMode::Serpentine,
            DitherMethod::JjnStandard => DitherMode::JarvisStandard,
            DitherMethod::JjnSerpentine => DitherMode::JarvisSerpentine,
            DitherMethod::MixedStandard => DitherMode::MixedStandard,
            DitherMethod::MixedSerpentine => DitherMode::MixedSerpentine,
            DitherMethod::MixedRandom => DitherMode::MixedRandom,
            DitherMethod::None => DitherMode::None,
        }
    }

    pub fn to_cs_dither_mode(self) -> CSDitherMode {
        match self {
            DitherMethod::FsStandard => CSDitherMode::Standard,
            DitherMethod::FsSerpentine => CSDitherMode::Serpentine,
            DitherMethod::JjnStandard => CSDitherMode::JarvisStandard,
            DitherMethod::JjnSerpentine => CSDitherMode::JarvisSerpentine,
            DitherMethod::MixedStandard => CSDitherMode::MixedStandard,
            DitherMethod::MixedSerpentine => CSDitherMode::MixedSerpentine,
            DitherMethod::MixedRandom => CSDitherMode::MixedRandom,
            DitherMethod::None => CSDitherMode::None,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum HistogramMode {
    /// Binned uint8 histogram matching (256 bins, with dithering)
    #[default]
    Binned,
    /// F32 sort-based, endpoint-aligned (no quantization, preserves extremes)
    F32Endpoint,
    /// F32 sort-based, midpoint-aligned (no quantization, statistically correct)
    F32Midpoint,
}

impl HistogramMode {
    pub fn to_lib_mode(self) -> LibHistogramMode {
        match self {
            HistogramMode::Binned => LibHistogramMode::Binned,
            HistogramMode::F32Endpoint => LibHistogramMode::EndpointAligned,
            HistogramMode::F32Midpoint => LibHistogramMode::MidpointAligned,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ColorSpace {
    /// OKLab (perceptually uniform, recommended)
    #[default]
    Oklab,
    /// CIELAB with CIE76 (simple Euclidean distance)
    LabCie76,
    /// CIELAB with CIE94 (weighted distance)
    LabCie94,
    /// CIELAB with CIEDE2000 (most accurate)
    LabCiede2000,
    /// Linear RGB (not recommended, for testing only)
    LinearRgb,
    /// Y'CbCr (not recommended, for testing only)
    YCbCr,
    /// sRGB (not recommended, for testing only)
    Srgb,
}

impl ColorSpace {
    pub fn to_perceptual_space(self) -> PerceptualSpace {
        match self {
            ColorSpace::Oklab => PerceptualSpace::OkLab,
            ColorSpace::LabCie76 => PerceptualSpace::LabCIE76,
            ColorSpace::LabCie94 => PerceptualSpace::LabCIE94,
            ColorSpace::LabCiede2000 => PerceptualSpace::LabCIEDE2000,
            ColorSpace::LinearRgb => PerceptualSpace::LinearRGB,
            ColorSpace::YCbCr => PerceptualSpace::YCbCr,
            ColorSpace::Srgb => PerceptualSpace::Srgb,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq)]
pub enum InputColorProfile {
    /// Assume standard sRGB input, use builtin gamma functions
    Srgb,
    /// Assume linear RGB input (for normal maps, height maps, data textures)
    Linear,
    /// Auto-detect: check CICP, then ICC profile, use moxcms if non-sRGB (default)
    #[default]
    Auto,
    /// Always use embedded ICC profile via moxcms (even if sRGB)
    Icc,
    /// Always use CICP metadata via moxcms (ignore ICC profile)
    Cicp,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq)]
pub enum PremultipliedAlpha {
    /// Auto-detect based on format (only EXR has premultiplied alpha by default)
    #[default]
    Auto,
    /// Input has premultiplied alpha - un-premultiply after loading
    Yes,
    /// Input does not have premultiplied alpha - no conversion needed
    No,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ScaleMethod {
    /// Bilinear interpolation (fast, good for moderate scaling)
    Bilinear,
    /// Mitchell-Netravali (B=C=1/3): soft, minimal ringing
    Mitchell,
    /// Catmull-Rom (B=0, C=0.5): sharp, low ringing
    CatmullRom,
    /// Lanczos2: good sharpness, less ringing than Lanczos3
    Lanczos2,
    /// Lanczos3: good balance of sharpness and ringing
    Lanczos3,
    /// Pure Sinc (non-windowed): theoretically ideal, full image extent (SLOW, research only)
    Sinc,
    /// Lanczos3 with scatter-based accumulation (experimental)
    Lanczos3Scatter,
    /// Sinc with scatter-based accumulation (experimental, SLOW)
    SincScatter,
    /// Box filter: nearest-neighbor for upscaling, proper area average for downscaling
    Box,
    /// EWA Sinc-Lanczos2: radial sinc-based 2D kernel (faster than jinc)
    EwaSincLanczos2,
    /// EWA Sinc-Lanczos3: radial sinc-based 2D kernel (faster than jinc)
    EwaSincLanczos3,
    /// EWA Lanczos2: proper jinc-based 2D kernel (best quality, slower)
    EwaLanczos2,
    /// EWA Lanczos3: proper jinc-based 2D kernel (best quality, recommended)
    #[default]
    EwaLanczos3,
    /// EWA Lanczos3 Sharp: 3-lobe jinc sharpened to minimize 1D step response error (Robidoux)
    EwaLanczos3Sharp,
    /// EWA Lanczos4 Sharpest: 4-lobe jinc sharpened to minimize total impulse response error (Robidoux)
    EwaLanczos4Sharpest,
    /// EWA Mitchell: Mitchell-Netravali applied radially (soft, 2D)
    EwaMitchell,
    /// EWA Catmull-Rom: Catmull-Rom applied radially (sharp, 2D)
    EwaCatmullRom,
    /// Pure Jinc (unwindowed): 2D analog of sinc, full image extent (SLOW, research only)
    Jinc,
    /// Stochastic Jinc: jinc with Gaussian sampling, gather-based (experimental)
    StochasticJinc,
    /// Stochastic Jinc Scatter: jinc with Gaussian sampling, scatter-based (experimental)
    StochasticJincScatter,
    /// Stochastic Jinc Scatter Normalized: scatter with destination normalization
    StochasticJincScatterNormalized,
    /// Iterative Bilinear: mipmap-style 2× bilinear passes for power-of-2, then final factor
    BilinearIterative,
    /// Hybrid Lanczos3: separable Lanczos3 for bulk scaling (>=2x), EWA Lanczos3 for final 2x
    /// Combines speed of separable convolution with EWA quality for the final refinement
    HybridLanczos3,
}

impl ScaleMethod {
    pub fn to_rescale_method(self) -> cra_wasm::rescale::RescaleMethod {
        match self {
            ScaleMethod::Bilinear => cra_wasm::rescale::RescaleMethod::Bilinear,
            ScaleMethod::Mitchell => cra_wasm::rescale::RescaleMethod::Mitchell,
            ScaleMethod::CatmullRom => cra_wasm::rescale::RescaleMethod::CatmullRom,
            ScaleMethod::Lanczos2 => cra_wasm::rescale::RescaleMethod::Lanczos2,
            ScaleMethod::Lanczos3 => cra_wasm::rescale::RescaleMethod::Lanczos3,
            ScaleMethod::Sinc => cra_wasm::rescale::RescaleMethod::Sinc,
            ScaleMethod::Lanczos3Scatter => cra_wasm::rescale::RescaleMethod::Lanczos3Scatter,
            ScaleMethod::SincScatter => cra_wasm::rescale::RescaleMethod::SincScatter,
            ScaleMethod::Box => cra_wasm::rescale::RescaleMethod::Box,
            ScaleMethod::EwaSincLanczos2 => cra_wasm::rescale::RescaleMethod::EWASincLanczos2,
            ScaleMethod::EwaSincLanczos3 => cra_wasm::rescale::RescaleMethod::EWASincLanczos3,
            ScaleMethod::EwaLanczos2 => cra_wasm::rescale::RescaleMethod::EWALanczos2,
            ScaleMethod::EwaLanczos3 => cra_wasm::rescale::RescaleMethod::EWALanczos3,
            ScaleMethod::EwaLanczos3Sharp => cra_wasm::rescale::RescaleMethod::EWALanczos3Sharp,
            ScaleMethod::EwaLanczos4Sharpest => cra_wasm::rescale::RescaleMethod::EWALanczos4Sharpest,
            ScaleMethod::EwaMitchell => cra_wasm::rescale::RescaleMethod::EWAMitchell,
            ScaleMethod::EwaCatmullRom => cra_wasm::rescale::RescaleMethod::EWACatmullRom,
            ScaleMethod::Jinc => cra_wasm::rescale::RescaleMethod::Jinc,
            ScaleMethod::StochasticJinc => cra_wasm::rescale::RescaleMethod::StochasticJinc,
            ScaleMethod::StochasticJincScatter => cra_wasm::rescale::RescaleMethod::StochasticJincScatter,
            ScaleMethod::StochasticJincScatterNormalized => cra_wasm::rescale::RescaleMethod::StochasticJincScatterNormalized,
            ScaleMethod::BilinearIterative => cra_wasm::rescale::RescaleMethod::BilinearIterative,
            ScaleMethod::HybridLanczos3 => cra_wasm::rescale::RescaleMethod::HybridLanczos3,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum Tonemapping {
    /// ACES filmic tonemapping (forward: HDR to SDR)
    Aces,
    /// ACES inverse tonemapping (reverse: SDR to HDR approximation)
    AcesInverse,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
pub enum Supersample {
    /// No supersampling (default)
    #[default]
    None,
    /// Tent-volume supersampling: expands to (2N+1)×(2M+1), adjusts pixel values
    /// using pyramid tent volume matching, then contracts back after processing.
    /// Preserves total light energy during resizing operations.
    TentVolume,
    /// Tent-volume prescale: expands to tent-space, then rescales directly to
    /// final dimensions with integrated contraction. If no resize is specified,
    /// uses box filter to input size. More efficient than TentVolume.
    TentVolumePrescale,
}

/// Build ColorCorrectionMethod from CLI arguments
pub fn build_correction_method(
    histogram: Histogram,
    keep_luminosity: bool,
    tiled_luminosity: bool,
    use_perceptual: bool,
) -> Option<ColorCorrectionMethod> {
    match histogram {
        Histogram::None => None,
        Histogram::BasicLab => Some(ColorCorrectionMethod::BasicLab { keep_luminosity }),
        Histogram::BasicRgb => Some(ColorCorrectionMethod::BasicRgb),
        Histogram::BasicOklab => Some(ColorCorrectionMethod::BasicOklab { keep_luminosity }),
        Histogram::CraLab => Some(ColorCorrectionMethod::CraLab { keep_luminosity }),
        Histogram::CraRgb => Some(ColorCorrectionMethod::CraRgb { use_perceptual }),
        Histogram::CraOklab => Some(ColorCorrectionMethod::CraOklab { keep_luminosity }),
        Histogram::TiledLab => Some(ColorCorrectionMethod::TiledLab { tiled_luminosity }),
        Histogram::TiledOklab => Some(ColorCorrectionMethod::TiledOklab { tiled_luminosity }),
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum StrideFillArg {
    /// Fill padding with black (zeros)
    #[default]
    Black,
    /// Repeat the last pixel to fill padding
    Repeat,
}

impl StrideFillArg {
    pub fn to_stride_fill(self) -> cra_wasm::binary_format::StrideFill {
        match self {
            StrideFillArg::Black => cra_wasm::binary_format::StrideFill::Black,
            StrideFillArg::Repeat => cra_wasm::binary_format::StrideFill::Repeat,
        }
    }
}

/// Safetensors output data format
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum SafetensorsFormat {
    /// 32-bit floating point (highest precision)
    #[default]
    Fp32,
    /// 16-bit floating point (IEEE half precision)
    Fp16,
    /// Brain floating point (16-bit, same exponent range as FP32)
    Bf16,
}

/// Safetensors transfer function
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq)]
pub enum SafetensorsTransfer {
    /// Auto: linear if processing path, sRGB if non-processing path
    Auto,
    /// Linear light (no gamma curve)
    Linear,
    /// sRGB transfer function (gamma-encoded, most common)
    #[default]
    Srgb,
}

// ============================================================================
// Command Line Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "cra")]
#[command(author, version, about = "CRA - Unified Color Correction and Dithering Tool", long_about = None)]
pub struct Args {
    /// Input image path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Input metadata for raw binary files (JSON format)
    ///
    /// Required fields: format, width, height
    /// Optional fields: stride (default: 0 = packed)
    ///
    /// Raw files are always assumed to be sRGB.
    ///
    /// Example: '{"format": "RGB565", "width": 128, "height": 64}'
    /// Supported formats: RGB888, RGB565, RGB332, L8, L4, L2, L1, ARGB8888, ARGB1555, etc.
    #[arg(long, value_name = "JSON")]
    pub input_metadata: Option<String>,

    /// Input color profile handling
    #[arg(long, value_enum, default_value_t = InputColorProfile::Auto)]
    pub input_profile: InputColorProfile,

    /// Input premultiplied alpha handling (auto: only EXR is premultiplied by default)
    #[arg(long, value_enum, default_value_t = PremultipliedAlpha::Auto)]
    pub input_premultiplied_alpha: PremultipliedAlpha,

    /// Reference image path (optional - required for histogram matching methods)
    #[arg(short, long)]
    pub r#ref: Option<PathBuf>,

    /// Reference color profile handling
    #[arg(long, value_enum, default_value_t = InputColorProfile::Auto)]
    pub ref_profile: InputColorProfile,

    /// Output PNG image path (optional)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output raw binary file path (optional) - respects --stride for row alignment
    #[arg(long)]
    pub output_raw: Option<PathBuf>,

    /// Disable palettized PNG output for formats with ≤8 bits per pixel
    /// By default, PNG output uses indexed color (palette) when the format supports it,
    /// which produces smaller files. Use this flag to force regular RGB/grayscale PNG output.
    #[arg(long)]
    pub no_palettized_output: bool,

    /// Output raw binary for red channel only (optional) - respects --stride
    #[arg(long)]
    pub output_raw_r: Option<PathBuf>,

    /// Output raw binary for green channel only (optional) - respects --stride
    #[arg(long)]
    pub output_raw_g: Option<PathBuf>,

    /// Output raw binary for blue channel only (optional) - respects --stride
    #[arg(long)]
    pub output_raw_b: Option<PathBuf>,

    /// Output raw binary for alpha channel only (optional) - respects --stride
    /// Requires ARGB format (e.g., ARGB8888, ARGB1555) and input with alpha channel
    #[arg(long)]
    pub output_raw_a: Option<PathBuf>,

    /// Output metadata JSON file path (optional)
    #[arg(long)]
    pub output_meta: Option<PathBuf>,

    /// Output safetensors file path (optional)
    /// Writes the image as floating-point before dithering.
    #[arg(long)]
    pub output_safetensors: Option<PathBuf>,

    /// Safetensors output transfer function
    #[arg(long, value_enum, default_value_t = SafetensorsTransfer::Srgb)]
    pub safetensors_transfer: SafetensorsTransfer,

    /// Safetensors output data format
    #[arg(long, value_enum, default_value_t = SafetensorsFormat::Fp32)]
    pub safetensors_format: SafetensorsFormat,

    /// Strip alpha channel from safetensors output
    #[arg(long)]
    pub safetensors_no_alpha: bool,

    /// Dithering method for safetensors FP16/BF16 output quantization (FP32 ignores this)
    #[arg(long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    pub safetensors_dither: DitherMethod,

    /// Perceptual space for safetensors dithering distance metric
    #[arg(long, value_enum, default_value_t = ColorSpace::Oklab)]
    pub safetensors_distance_space: ColorSpace,

    /// Output format: RGB, ARGB, L with bit counts.
    /// RGB: RGB8 (=RGB888), RGB332, RGB565, RGB888
    /// ARGB: ARGB8 (=ARGB8888), ARGB4 (=ARGB4444), ARGB1555, ARGB4444, ARGB8888
    /// Grayscale: L1, L2, L4, L8
    /// Single digit means same for all channels.
    /// Default: ARGB8888 if input has alpha, RGB888 otherwise.
    #[arg(short, long)]
    pub format: Option<String>,

    /// Histogram matching method (default: none if no reference, cra-oklab if reference provided)
    #[arg(long, value_enum)]
    pub histogram: Option<Histogram>,

    /// Preserve original luminosity (L channel) - applies to basic-lab, basic-oklab, cra-lab, cra-oklab
    #[arg(long)]
    pub keep_luminosity: bool,

    /// Process L channel per-tile before global match - applies to tiled-lab, tiled-oklab
    #[arg(long)]
    pub tiled_luminosity: bool,

    /// Use perceptual weighting - applies to cra-rgb
    #[arg(long)]
    pub perceptual: bool,

    /// Histogram matching mode
    #[arg(long, value_enum, default_value_t = HistogramMode::Binned)]
    pub histogram_mode: HistogramMode,

    /// Dithering method for final output quantization
    #[arg(long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    pub output_dither: DitherMethod,

    /// Dithering method for alpha channel output quantization (defaults to same as --output-dither)
    /// Only applies when output format includes alpha (ARGB, LA)
    #[arg(long, value_enum)]
    pub output_alpha_dither: Option<DitherMethod>,

    /// Dithering method for histogram processing (only used with --histogram-mode=binned)
    #[arg(long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    pub histogram_dither: DitherMethod,

    /// Disable colorspace-aware dithering for histogram quantization (use per-channel instead)
    #[arg(long)]
    pub no_colorspace_aware_histogram: bool,

    /// Perceptual space for colorspace-aware histogram dithering distance metric
    #[arg(long, value_enum, default_value_t = ColorSpace::Oklab)]
    pub histogram_distance_space: ColorSpace,

    /// Disable colorspace-aware dithering for final RGB output (use per-channel instead)
    #[arg(long)]
    pub no_colorspace_aware_output: bool,

    /// Perceptual space for output dithering distance metric (default: oklab for RGB, lab-cie94 for grayscale)
    #[arg(long, value_enum)]
    pub output_distance_space: Option<ColorSpace>,

    /// Random seed for mixed dithering modes
    #[arg(short, long, default_value_t = 12345)]
    pub seed: u32,

    /// Row stride alignment in bytes for row-aligned binary output (power of 2, 1-128)
    #[arg(long, default_value_t = 1)]
    pub stride: usize,

    /// How to fill stride padding bytes
    #[arg(long, value_enum, default_value_t = StrideFillArg::Black)]
    pub stride_fill: StrideFillArg,

    /// Resize image to this width (preserves aspect ratio)
    #[arg(long)]
    pub width: Option<u32>,

    /// Resize image to this height (preserves aspect ratio)
    #[arg(long)]
    pub height: Option<u32>,

    /// Scaling method for resize operations
    #[arg(long, value_enum, default_value_t = ScaleMethod::EwaLanczos3)]
    pub scale_method: ScaleMethod,

    /// Supersampling method for resize operations
    ///
    /// tent-volume: Before resizing, expands image to (2N+1)×(2M+1) with linear
    /// interpolation between pixels. Adjusts carried-over mid pixels using pyramid
    /// tent volume matching. After processing, contracts back using tent volumes.
    /// The resizer targets (2*target+1) dimensions and uses second-pixel-from-edge
    /// midpoints as scaling corners.
    #[arg(long, value_enum, default_value_t = Supersample::None)]
    pub supersample: Supersample,

    /// Input tonemapping: applied before histogram matching (color correction)
    #[arg(long, value_enum)]
    pub input_tonemapping: Option<Tonemapping>,

    /// Output tonemapping: applied after grayscale conversion (for grayscale output) or to RGB (for RGB output)
    #[arg(long, value_enum)]
    pub tonemapping: Option<Tonemapping>,

    /// Exposure adjustment (linear multiplier, applied before tonemapping)
    /// Values > 1.0 brighten, < 1.0 darken. Example: 2.0 = +1 stop, 0.5 = -1 stop
    #[arg(long)]
    pub exposure: Option<f32>,

    /// Disable automatic uniform scaling detection
    ///
    /// By default, when both --width and --height are specified and result in nearly
    /// uniform scaling (within 1 pixel of preserving aspect ratio), the tool automatically
    /// enforces uniform scaling. Use this flag to disable that behavior and use the
    /// exact dimensions specified, even if it causes slight aspect ratio distortion.
    #[arg(long)]
    pub non_uniform: bool,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Show progress bar during processing
    #[arg(long)]
    pub progress: bool,
}
