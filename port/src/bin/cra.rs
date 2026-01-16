//! CRA - Unified Color Correction and Dithering CLI
//!
//! A unified command-line tool for color correction and error diffusion dithering.
//! Pipeline: input sRGB -> linear RGB -> optional resize -> optional processing -> dither to sRGB -> output
//!
//! All processing occurs in linear RGB space for correct color math.

use clap::{Parser, ValueEnum};
use image::{ColorType, DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgba};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use cra_wasm::binary_format::{
    encode_channel_from_interleaved_row_aligned_stride, encode_gray_packed,
    encode_gray_row_aligned_stride, encode_rgb_packed, encode_rgb_row_aligned_stride,
    is_valid_stride, ColorFormat, StrideFill,
};
use cra_wasm::color::{linear_pixels_to_grayscale, linear_to_srgb_single, srgb_to_linear_single};
use cra_wasm::correction::{color_correct, HistogramOptions};
use cra_wasm::decode::{
    can_use_cicp, cicp_description, image_to_f32_normalized_rgba, image_to_f32_srgb_255_pixels_rgba,
    is_cicp_linear_srgb, is_cicp_needs_conversion, is_cicp_srgb, is_cicp_unspecified,
    is_profile_srgb_verbose, load_image_from_path, transform_cicp_to_linear_srgb_pixels,
    transform_icc_to_linear_srgb_pixels,
};
use cra_wasm::dither_rgb::DitherMode as CSDitherMode;
use cra_wasm::dither_luminosity::colorspace_aware_dither_gray_with_mode;
use cra_wasm::dither_common::{
    ColorCorrectionMethod, DitherMode, HistogramMode as LibHistogramMode, OutputTechnique,
    PerceptualSpace,
};
use cra_wasm::color::{denormalize_inplace_clamped, linear_to_srgb_inplace};
use cra_wasm::output::{dither_output_rgb, dither_output_rgba};
use cra_wasm::pixel::{Pixel4, unpremultiply_alpha_inplace};

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Histogram {
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
enum DitherMethod {
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
    fn to_dither_mode(self) -> DitherMode {
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

    fn to_cs_dither_mode(self) -> CSDitherMode {
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
enum HistogramMode {
    /// Binned uint8 histogram matching (256 bins, with dithering)
    #[default]
    Binned,
    /// F32 sort-based, endpoint-aligned (no quantization, preserves extremes)
    F32Endpoint,
    /// F32 sort-based, midpoint-aligned (no quantization, statistically correct)
    F32Midpoint,
}

impl HistogramMode {
    fn to_lib_mode(self) -> LibHistogramMode {
        match self {
            HistogramMode::Binned => LibHistogramMode::Binned,
            HistogramMode::F32Endpoint => LibHistogramMode::EndpointAligned,
            HistogramMode::F32Midpoint => LibHistogramMode::MidpointAligned,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum ColorSpace {
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
}

impl ColorSpace {
    fn to_perceptual_space(self) -> PerceptualSpace {
        match self {
            ColorSpace::Oklab => PerceptualSpace::OkLab,
            ColorSpace::LabCie76 => PerceptualSpace::LabCIE76,
            ColorSpace::LabCie94 => PerceptualSpace::LabCIE94,
            ColorSpace::LabCiede2000 => PerceptualSpace::LabCIEDE2000,
            ColorSpace::LinearRgb => PerceptualSpace::LinearRGB,
            ColorSpace::YCbCr => PerceptualSpace::YCbCr,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq)]
enum InputColorProfile {
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
enum PremultipliedAlpha {
    /// Auto-detect based on format (only EXR has premultiplied alpha by default)
    #[default]
    Auto,
    /// Input has premultiplied alpha - un-premultiply after loading
    Yes,
    /// Input does not have premultiplied alpha - no conversion needed
    No,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum ScaleMethod {
    /// Bilinear interpolation (fast, good for moderate scaling)
    Bilinear,
    /// Lanczos3 (high quality, recommended for significant down/upscaling)
    #[default]
    Lanczos,
}

impl ScaleMethod {
    fn to_rescale_method(self) -> cra_wasm::rescale::RescaleMethod {
        match self {
            ScaleMethod::Bilinear => cra_wasm::rescale::RescaleMethod::Bilinear,
            ScaleMethod::Lanczos => cra_wasm::rescale::RescaleMethod::Lanczos3,
        }
    }
}

/// Build ColorCorrectionMethod from CLI arguments
fn build_correction_method(
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
enum StrideFillArg {
    /// Fill padding with black (zeros)
    #[default]
    Black,
    /// Repeat the last pixel to fill padding
    Repeat,
}

impl StrideFillArg {
    fn to_stride_fill(self) -> StrideFill {
        match self {
            StrideFillArg::Black => StrideFill::Black,
            StrideFillArg::Repeat => StrideFill::Repeat,
        }
    }
}

// ============================================================================
// Progress Bar
// ============================================================================

/// Print a progress bar to stderr (overwrites the current line)
fn print_progress(label: &str, progress: f32) {
    const BAR_WIDTH: usize = 30;
    let filled = (progress * BAR_WIDTH as f32).round() as usize;
    let empty = BAR_WIDTH.saturating_sub(filled);
    eprint!(
        "\r{}: [{}{}] {:3}%",
        label,
        "=".repeat(filled),
        " ".repeat(empty),
        (progress * 100.0).round() as u32
    );
    let _ = std::io::stderr().flush();
}

/// Clear the progress bar line
fn clear_progress() {
    eprint!("\r{}\r", " ".repeat(60));
    let _ = std::io::stderr().flush();
}

// ============================================================================
// Command Line Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "cra")]
#[command(author, version, about = "CRA - Unified Color Correction and Dithering Tool", long_about = None)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    input: PathBuf,

    /// Input color profile handling
    #[arg(long, value_enum, default_value_t = InputColorProfile::Auto)]
    input_profile: InputColorProfile,

    /// Input premultiplied alpha handling (auto: only EXR is premultiplied by default)
    #[arg(long, value_enum, default_value_t = PremultipliedAlpha::Auto)]
    input_premultiplied_alpha: PremultipliedAlpha,

    /// Reference image path (optional - required for histogram matching methods)
    #[arg(short, long)]
    r#ref: Option<PathBuf>,

    /// Reference color profile handling
    #[arg(long, value_enum, default_value_t = InputColorProfile::Auto)]
    ref_profile: InputColorProfile,

    /// Output PNG image path (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output raw binary file path (optional) - respects --stride for row alignment
    #[arg(long)]
    output_raw: Option<PathBuf>,

    /// Output raw binary for red channel only (optional) - respects --stride
    #[arg(long)]
    output_raw_r: Option<PathBuf>,

    /// Output raw binary for green channel only (optional) - respects --stride
    #[arg(long)]
    output_raw_g: Option<PathBuf>,

    /// Output raw binary for blue channel only (optional) - respects --stride
    #[arg(long)]
    output_raw_b: Option<PathBuf>,

    /// Output metadata JSON file path (optional)
    #[arg(long)]
    output_meta: Option<PathBuf>,

    /// Output format: RGB with bit counts (e.g., RGB8, RGB332, RGB565, RGB888) or L with bits (e.g., L4, L8).
    /// Single digit means same for all channels (RGB8 = RGB888, RGB5 = RGB555).
    #[arg(short, long, default_value = "RGB888")]
    format: String,

    /// Histogram matching method (default: none if no reference, cra-oklab if reference provided)
    #[arg(long, value_enum)]
    histogram: Option<Histogram>,

    /// Preserve original luminosity (L channel) - applies to basic-lab, basic-oklab, cra-lab, cra-oklab
    #[arg(long)]
    keep_luminosity: bool,

    /// Process L channel per-tile before global match - applies to tiled-lab, tiled-oklab
    #[arg(long)]
    tiled_luminosity: bool,

    /// Use perceptual weighting - applies to cra-rgb
    #[arg(long)]
    perceptual: bool,

    /// Histogram matching mode
    #[arg(long, value_enum, default_value_t = HistogramMode::Binned)]
    histogram_mode: HistogramMode,

    /// Dithering method for final output quantization
    #[arg(long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    output_dither: DitherMethod,

    /// Dithering method for histogram processing (only used with --histogram-mode=binned)
    #[arg(long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    histogram_dither: DitherMethod,

    /// Disable colorspace-aware dithering for histogram quantization (use per-channel instead)
    #[arg(long)]
    no_colorspace_aware_histogram: bool,

    /// Perceptual space for colorspace-aware histogram dithering distance metric
    #[arg(long, value_enum, default_value_t = ColorSpace::Oklab)]
    histogram_distance_space: ColorSpace,

    /// Disable colorspace-aware dithering for final RGB output (use per-channel instead)
    #[arg(long)]
    no_colorspace_aware_output: bool,

    /// Perceptual space for output dithering distance metric (default: oklab for RGB, lab-cie94 for grayscale)
    #[arg(long, value_enum)]
    output_distance_space: Option<ColorSpace>,

    /// Random seed for mixed dithering modes
    #[arg(short, long, default_value_t = 12345)]
    seed: u32,

    /// Row stride alignment in bytes for row-aligned binary output (power of 2, 1-128)
    #[arg(long, default_value_t = 1)]
    stride: usize,

    /// How to fill stride padding bytes
    #[arg(long, value_enum, default_value_t = StrideFillArg::Black)]
    stride_fill: StrideFillArg,

    /// Resize image to this width (preserves aspect ratio)
    #[arg(long)]
    width: Option<u32>,

    /// Resize image to this height (preserves aspect ratio)
    #[arg(long)]
    height: Option<u32>,

    /// Scaling method for resize operations
    #[arg(long, value_enum, default_value_t = ScaleMethod::Lanczos)]
    scale_method: ScaleMethod,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Show progress bar during processing
    #[arg(long)]
    progress: bool,
}

// ============================================================================
// Linear RGB Image Processing
// ============================================================================

/// Convert ICC pixels to Pixel4 format using shared decode module
/// Takes RGBA input, strips to RGB for ICC transform, merges alpha back
fn convert_icc_to_linear_pixels(
    input_pixels: &[[f32; 4]],
    width: u32,
    height: u32,
    icc_profile: &[u8],
    verbose: bool,
) -> Result<(Vec<Pixel4>, u32, u32), String> {
    // ICC transform only handles RGB - strip alpha, transform, merge back
    let rgb_only: Vec<[f32; 3]> = input_pixels.iter().map(|p| [p[0], p[1], p[2]]).collect();

    let result = transform_icc_to_linear_srgb_pixels(
        &rgb_only,
        width as usize,
        height as usize,
        icc_profile,
    )?;

    if verbose {
        eprintln!("  Converted via ICC profile to linear sRGB (float path)");
    }

    // Merge transformed RGB with original alpha
    let pixels: Vec<Pixel4> = result
        .into_iter()
        .zip(input_pixels.iter())
        .map(|([r, g, b], orig)| Pixel4([r, g, b, orig[3]]))
        .collect();

    Ok((pixels, width, height))
}

/// Convert CICP pixels to Pixel4 format using shared decode module
/// Takes RGBA input, strips to RGB for CICP transform, merges alpha back
fn convert_cicp_to_linear_pixels(
    input_pixels: &[[f32; 4]],
    width: u32,
    height: u32,
    cicp: &image::metadata::Cicp,
    verbose: bool,
) -> Result<(Vec<Pixel4>, u32, u32), String> {
    let result = transform_cicp_to_linear_srgb_pixels(
        input_pixels,
        width as usize,
        height as usize,
        cicp,
    )?;

    if verbose {
        eprintln!("  Converted via CICP to linear sRGB (float path)");
    }

    // Merge transformed RGB with original alpha
    let pixels: Vec<Pixel4> = result
        .into_iter()
        .zip(input_pixels.iter())
        .map(|([r, g, b], orig)| Pixel4([r, g, b, orig[3]]))
        .collect();

    Ok((pixels, width, height))
}

/// Determine effective color profile mode based on CICP and ICC profile detection.
/// Priority: CICP (authoritative) > ICC profile > CICP fallback > assume sRGB
fn determine_effective_profile(
    profile_mode: InputColorProfile,
    icc_profile: &Option<Vec<u8>>,
    cicp: &image::metadata::Cicp,
    verbose: bool,
) -> InputColorProfile {
    match profile_mode {
        InputColorProfile::Srgb => InputColorProfile::Srgb,
        InputColorProfile::Linear => InputColorProfile::Linear,
        InputColorProfile::Icc => {
            if icc_profile.is_some() {
                InputColorProfile::Icc
            } else {
                if verbose {
                    eprintln!("  No ICC profile found, falling back to sRGB");
                }
                InputColorProfile::Srgb
            }
        }
        InputColorProfile::Cicp => {
            if can_use_cicp(cicp) {
                if verbose {
                    eprintln!("  Using CICP: {}", cicp_description(cicp));
                }
                InputColorProfile::Cicp
            } else {
                if verbose {
                    eprintln!("  CICP not usable ({}), falling back to sRGB", cicp_description(cicp));
                }
                InputColorProfile::Srgb
            }
        }
        InputColorProfile::Auto => {
            // Check CICP first (authoritative, O(1) check)
            if is_cicp_srgb(cicp) {
                if verbose {
                    eprintln!("  CICP indicates sRGB (authoritative)");
                }
                return InputColorProfile::Srgb;
            }
            if is_cicp_linear_srgb(cicp) {
                if verbose {
                    eprintln!("  CICP indicates linear sRGB (authoritative)");
                }
                return InputColorProfile::Linear;
            }

            // Check if CICP indicates non-sRGB color space
            let cicp_needs_conversion = is_cicp_needs_conversion(cicp);

            // Fall back to ICC profile check (if CICP was unspecified or needs conversion)
            if let Some(icc) = icc_profile {
                if is_profile_srgb_verbose(icc, verbose) {
                    if verbose && is_cicp_unspecified(cicp) {
                        eprintln!("  CICP unspecified, ICC profile is sRGB-compatible");
                    }
                    InputColorProfile::Srgb
                } else {
                    if verbose {
                        if is_cicp_unspecified(cicp) {
                            eprintln!("  CICP unspecified, using non-sRGB ICC profile via moxcms");
                        } else {
                            eprintln!("  Using ICC profile for color conversion via moxcms");
                        }
                    }
                    InputColorProfile::Icc
                }
            } else if cicp_needs_conversion && can_use_cicp(cicp) {
                // No ICC profile, but CICP indicates non-sRGB and we can use it
                if verbose {
                    eprintln!("  No ICC profile, using CICP for conversion: {}", cicp_description(cicp));
                }
                InputColorProfile::Cicp
            } else {
                if verbose {
                    if is_cicp_unspecified(cicp) {
                        eprintln!("  CICP unspecified, no ICC profile, assuming sRGB");
                    } else if cicp_needs_conversion {
                        eprintln!("  CICP indicates non-sRGB but unsupported ({}), assuming sRGB", cicp_description(cicp));
                    } else {
                        eprintln!("  No ICC profile available, assuming sRGB");
                    }
                }
                InputColorProfile::Srgb
            }
        }
    }
}

/// Convert pre-loaded image to linear RGB channels (f32, 0-1 range)
/// Returns (pixels, width, height, has_alpha)
/// Always uses RGBA path internally - Pixel4 is float4 so no overhead.
/// If unpremultiply is true, un-premultiplies alpha after conversion to linear.
fn convert_to_linear(
    img: &DynamicImage,
    icc_profile: &Option<Vec<u8>>,
    cicp: &image::metadata::Cicp,
    profile_mode: InputColorProfile,
    unpremultiply: bool,
    verbose: bool,
) -> Result<(Vec<Pixel4>, u32, u32, bool), String> {
    let (width, height) = img.dimensions();
    let has_alpha = matches!(
        img.color(),
        ColorType::La8 | ColorType::Rgba8 | ColorType::La16 | ColorType::Rgba16 | ColorType::Rgba32F
    );

    if verbose {
        eprintln!("  Input profile mode: {:?}", profile_mode);
        eprintln!("  Dimensions: {}x{}", width, height);
        eprintln!("  Color type: {:?}", img.color());
        eprintln!("  Has alpha: {}", has_alpha);
        eprintln!("  CICP: {}", cicp_description(cicp));
        if unpremultiply {
            eprintln!("  Premultiplied alpha: yes (will un-premultiply)");
        }
        if let Some(icc) = icc_profile {
            eprintln!("  ICC profile: {} bytes", icc.len());
        } else {
            eprintln!("  ICC profile: none");
        }
    }

    let effective_mode = determine_effective_profile(profile_mode, icc_profile, cicp, verbose);

    if verbose {
        let is_16bit = matches!(
            img.color(),
            ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
        );
        let is_f32 = matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F);
        if is_f32 {
            eprintln!("  Using f32 precision path (values already float)");
        } else if is_16bit {
            eprintln!("  Converting 16-bit to float (dividing by 65535)");
        } else {
            eprintln!("  Converting 8-bit to float (dividing by 255)");
        }
    }

    // Always use RGBA path - Pixel4 is float4, no overhead
    // Images without alpha get alpha=1.0 from image crate's to_rgba*
    let normalized = image_to_f32_normalized_rgba(img);

    // Apply color space conversion based on effective mode
    let mut pixels = match effective_mode {
        InputColorProfile::Icc => {
            // ICC transform handles RGB extraction and alpha merge internally
            let icc = icc_profile.as_ref().expect("ICC mode requires profile");
            let (pixels, _, _) = convert_icc_to_linear_pixels(&normalized, width, height, icc, verbose)?;
            pixels
        }
        InputColorProfile::Cicp => {
            // CICP transform handles RGB extraction and alpha merge internally
            let (pixels, _, _) = convert_cicp_to_linear_pixels(&normalized, width, height, cicp, verbose)?;
            pixels
        }
        InputColorProfile::Linear => {
            // Already linear, just convert to Pixel4
            if verbose {
                eprintln!("  Input is linear, no gamma conversion");
            }
            normalized
                .into_iter()
                .map(|[r, g, b, a]| Pixel4([r, g, b, a]))
                .collect()
        }
        InputColorProfile::Srgb | InputColorProfile::Auto => {
            // sRGB input - apply gamma decode (alpha stays linear)
            if verbose {
                eprintln!("  Applying sRGB gamma decode");
            }
            normalized
                .into_iter()
                .map(|[r, g, b, a]| Pixel4([
                    srgb_to_linear_single(r),
                    srgb_to_linear_single(g),
                    srgb_to_linear_single(b),
                    a, // Alpha is already linear
                ]))
                .collect()
        }
    };

    // Un-premultiply alpha if needed (done in linear space)
    if unpremultiply && has_alpha {
        if verbose {
            eprintln!("  Un-premultiplying alpha (in linear space)");
        }
        unpremultiply_alpha_inplace(&mut pixels);
    }

    Ok((pixels, width, height, has_alpha))
}

/// Convert pre-loaded image to sRGB (f32, 0-255 range) - no color space conversion
/// Use when only dithering is needed (no resize, no color correction)
/// Returns (pixels, width, height, has_alpha)
/// Always uses RGBA path internally - Pixel4 is float4 so no overhead.
fn convert_to_srgb_255(img: &DynamicImage, verbose: bool) -> (Vec<Pixel4>, u32, u32, bool) {
    let (width, height) = img.dimensions();
    let has_alpha = matches!(
        img.color(),
        ColorType::La8 | ColorType::Rgba8 | ColorType::La16 | ColorType::Rgba16 | ColorType::Rgba32F
    );

    if verbose {
        eprintln!("  Dimensions: {}x{}", width, height);
        eprintln!("  Color type: {:?}", img.color());
        eprintln!("  Has alpha: {}", has_alpha);
        let is_16bit = matches!(
            img.color(),
            ColorType::Rgb16 | ColorType::Rgba16 | ColorType::L16 | ColorType::La16
        );
        let is_f32 = matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F);
        if is_f32 {
            eprintln!("  Using f32 precision path (scaling to 0-255, clamping HDR)");
        } else if is_16bit {
            eprintln!("  Using 16-bit precision path (scaling to 0-255)");
        } else {
            eprintln!("  Using 8-bit path");
        }
    }

    // Always use RGBA path - Pixel4 is float4, no overhead
    // Images without alpha get alpha=255.0 from image crate's to_rgba*
    let pixels: Vec<Pixel4> = image_to_f32_srgb_255_pixels_rgba(img)
        .into_iter()
        .map(|[r, g, b, a]| Pixel4([r, g, b, a]))
        .collect();
    (pixels, width, height, has_alpha)
}

/// Resize linear RGB image in linear space for correct color blending
fn resize_linear(
    pixels: &[Pixel4],
    src_width: u32,
    src_height: u32,
    target_width: Option<u32>,
    target_height: Option<u32>,
    method: cra_wasm::rescale::RescaleMethod,
    verbose: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Result<(Vec<Pixel4>, u32, u32), String> {
    use cra_wasm::rescale::{calculate_target_dimensions, rescale_with_progress, ScaleMode};

    let tw = target_width.map(|w| w as usize);
    let th = target_height.map(|h| h as usize);
    let (dst_width, dst_height) = calculate_target_dimensions(
        src_width as usize,
        src_height as usize,
        tw,
        th,
    );

    if dst_width == src_width as usize && dst_height == src_height as usize {
        return Ok((pixels.to_vec(), src_width, src_height));
    }

    if verbose {
        let method_name = match method {
            cra_wasm::rescale::RescaleMethod::Bilinear => "Bilinear",
            cra_wasm::rescale::RescaleMethod::Lanczos3 => "Lanczos3",
        };
        eprintln!(
            "Resizing in linear RGB ({}): {}x{} -> {}x{}",
            method_name, src_width, src_height, dst_width, dst_height
        );
    }

    // Rescale directly in linear space for correct color blending
    let dst_pixels = rescale_with_progress(
        pixels,
        src_width as usize, src_height as usize,
        dst_width, dst_height,
        method,
        ScaleMode::Independent,
        progress,
    );

    Ok((dst_pixels, dst_width as u32, dst_height as u32))
}

// ============================================================================
// Dithering
// ============================================================================

/// Build OutputTechnique from CLI options
fn build_output_technique(
    colorspace_aware: bool,
    mode: CSDitherMode,
    space: PerceptualSpace,
) -> OutputTechnique {
    if colorspace_aware {
        OutputTechnique::ColorspaceAware { mode, space }
    } else {
        OutputTechnique::PerChannel { mode }
    }
}

fn dither_grayscale(
    gray: &[f32],
    width: usize,
    height: usize,
    bits: u8,
    space: PerceptualSpace,
    mode: CSDitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    colorspace_aware_dither_gray_with_mode(gray, width, height, bits, space, mode, seed, progress)
}

/// Result of dithering operation
struct DitherResult {
    /// Interleaved output (grayscale, RGB, or RGBA)
    interleaved: Vec<u8>,
    /// True if this is grayscale data
    is_grayscale: bool,
    /// True if this is RGBA data (4 channels)
    has_alpha: bool,
}

/// Dither linear RGB pixels to the target format
#[allow(clippy::too_many_arguments)]
fn dither_pixels(
    pixels: Vec<Pixel4>,
    width: usize,
    height: usize,
    format: &cra_wasm::binary_format::ColorFormat,
    colorspace_aware: bool,
    dither_mode: CSDitherMode,
    colorspace: PerceptualSpace,
    seed: u32,
    has_alpha: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    if format.is_grayscale {
        // Step 1: Convert linear RGB to linear grayscale (luminance only)
        let linear_gray = linear_pixels_to_grayscale(&pixels);

        // Step 2: Convert linear to sRGB (gamma)
        // Step 3: Denormalize to 0-255
        let srgb_gray: Vec<f32> = linear_gray
            .iter()
            .map(|&l| linear_to_srgb_single(l) * 255.0)
            .collect();

        // Step 4: Dither grayscale (alpha not supported for grayscale output)
        let interleaved = dither_grayscale(
            &srgb_gray, width, height, format.bits_r, colorspace, dither_mode, seed, progress,
        );
        DitherResult { interleaved, is_grayscale: true, has_alpha: false }
    } else {
        let mut linear_pixels = pixels;

        // Convert linear RGB to sRGB 0-255 (alpha already in correct range)
        linear_to_srgb_inplace(&mut linear_pixels);
        denormalize_inplace_clamped(&mut linear_pixels);

        // Dither
        let technique = build_output_technique(colorspace_aware, dither_mode, colorspace);
        if has_alpha {
            let interleaved = dither_output_rgba(
                &linear_pixels,
                width, height,
                format.bits_r, format.bits_g, format.bits_b,
                technique,
                seed,
                progress,
            );
            DitherResult { interleaved, is_grayscale: false, has_alpha: true }
        } else {
            let interleaved = dither_output_rgb(
                &linear_pixels,
                width, height,
                format.bits_r, format.bits_g, format.bits_b,
                technique,
                seed,
                progress,
            );
            DitherResult { interleaved, is_grayscale: false, has_alpha: false }
        }
    }
}

/// Dither sRGB RGB pixels (0-255 range) directly to the target format
/// Use when no color correction, resize, or grayscale conversion is needed
/// (avoids linear conversion overhead)
#[allow(clippy::too_many_arguments)]
fn dither_pixels_srgb_rgb(
    pixels: Vec<Pixel4>,
    width: usize,
    height: usize,
    format: &cra_wasm::binary_format::ColorFormat,
    colorspace_aware: bool,
    dither_mode: CSDitherMode,
    colorspace: PerceptualSpace,
    seed: u32,
    has_alpha: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    debug_assert!(!format.is_grayscale, "Use linear path for grayscale");

    let technique = build_output_technique(colorspace_aware, dither_mode, colorspace);
    if has_alpha {
        let interleaved = dither_output_rgba(
            &pixels,
            width, height,
            format.bits_r, format.bits_g, format.bits_b,
            technique,
            seed,
            progress,
        );
        DitherResult { interleaved, is_grayscale: false, has_alpha: true }
    } else {
        let interleaved = dither_output_rgb(
            &pixels,
            width, height,
            format.bits_r, format.bits_g, format.bits_b,
            technique,
            seed,
            progress,
        );
        DitherResult { interleaved, is_grayscale: false, has_alpha: false }
    }
}

/// Encode dithered pixels to binary format
fn encode_binary(
    result: &DitherResult,
    format: &ColorFormat,
    width: usize,
    height: usize,
    row_aligned: bool,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    if result.is_grayscale {
        if row_aligned {
            encode_gray_row_aligned_stride(&result.interleaved, width, height, format.bits_r, stride, fill)
        } else {
            encode_gray_packed(&result.interleaved, width, height, format.bits_r)
        }
    } else {
        if row_aligned {
            encode_rgb_row_aligned_stride(
                &result.interleaved, width, height, format.bits_r, format.bits_g, format.bits_b, stride, fill,
            )
        } else {
            encode_rgb_packed(
                &result.interleaved, width, height, format.bits_r, format.bits_g, format.bits_b, fill,
            )
        }
    }
}

// ============================================================================
// PNG Output
// ============================================================================

fn save_png_grayscale(path: &PathBuf, data: &[u8], width: u32, height: u32) -> Result<(), String> {
    let img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, data.to_vec())
            .ok_or_else(|| "Failed to create grayscale image buffer".to_string())?;

    img.save(path)
        .map_err(|e| format!("Failed to save {}: {}", path.display(), e))?;

    Ok(())
}

fn save_png_rgb(path: &PathBuf, data: &[u8], width: u32, height: u32) -> Result<(), String> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, data.to_vec())
            .ok_or_else(|| "Failed to create RGB image buffer".to_string())?;

    img.save(path)
        .map_err(|e| format!("Failed to save {}: {}", path.display(), e))?;

    Ok(())
}

fn save_png_rgba(path: &PathBuf, data: &[u8], width: u32, height: u32) -> Result<(), String> {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, data.to_vec())
            .ok_or_else(|| "Failed to create RGBA image buffer".to_string())?;

    img.save(path)
        .map_err(|e| format!("Failed to save {}: {}", path.display(), e))?;

    Ok(())
}

// ============================================================================
// Metadata JSON
// ============================================================================

fn write_metadata(
    path: &PathBuf,
    args: &Args,
    format: &ColorFormat,
    histogram: Histogram,
    output_colorspace: ColorSpace,
    width: u32,
    height: u32,
    outputs: &[(String, PathBuf, usize)],
) -> Result<(), String> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"input\": \"{}\",\n", args.input.display()));
    if let Some(ref ref_path) = args.r#ref {
        json.push_str(&format!("  \"reference\": \"{}\",\n", ref_path.display()));
    }
    json.push_str(&format!("  \"histogram\": \"{:?}\",\n", histogram));
    json.push_str(&format!("  \"format\": \"{}\",\n", format.name));
    json.push_str(&format!("  \"width\": {},\n", width));
    json.push_str(&format!("  \"height\": {},\n", height));
    json.push_str(&format!("  \"is_grayscale\": {},\n", format.is_grayscale));
    json.push_str(&format!("  \"bits_per_pixel\": {},\n", format.total_bits));

    if !format.is_grayscale {
        json.push_str(&format!("  \"bits_r\": {},\n", format.bits_r));
        json.push_str(&format!("  \"bits_g\": {},\n", format.bits_g));
        json.push_str(&format!("  \"bits_b\": {},\n", format.bits_b));
    } else {
        json.push_str(&format!("  \"bits_l\": {},\n", format.bits_r));
    }

    json.push_str(&format!("  \"output_dither\": \"{:?}\",\n", args.output_dither));
    json.push_str(&format!("  \"output_colorspace\": \"{:?}\",\n", output_colorspace));
    json.push_str(&format!("  \"seed\": {},\n", args.seed));
    json.push_str(&format!("  \"stride\": {},\n", args.stride));
    json.push_str(&format!("  \"stride_fill\": \"{:?}\",\n", args.stride_fill));

    json.push_str("  \"outputs\": [\n");
    for (i, (output_type, output_path, size)) in outputs.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"type\": \"{}\",\n", output_type));
        json.push_str(&format!("      \"path\": \"{}\",\n", output_path.display()));
        json.push_str(&format!("      \"size_bytes\": {}\n", size));
        if i < outputs.len() - 1 {
            json.push_str("    },\n");
        } else {
            json.push_str("    }\n");
        }
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    let mut file =
        File::create(path).map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;

    file.write_all(json.as_bytes())
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), String> {
    let args = Args::parse();

    // Parse format string
    let format = ColorFormat::parse(&args.format)?;

    // Determine histogram method: user-specified, or default based on whether reference is provided
    let histogram = args.histogram.unwrap_or(if args.r#ref.is_some() {
        Histogram::CraOklab
    } else {
        Histogram::None
    });

    // Validate: histogram matching methods require a reference image
    let needs_reference = !matches!(histogram, Histogram::None);
    if needs_reference && args.r#ref.is_none() {
        return Err(format!(
            "Histogram {:?} requires a reference image. Use --ref <path> or --histogram none for dither-only mode.",
            histogram
        ));
    }

    // Determine output colorspace for dithering
    let output_colorspace = args.output_distance_space.unwrap_or(if format.is_grayscale {
        ColorSpace::LabCie94
    } else {
        ColorSpace::Oklab
    });

    let histogram_options = HistogramOptions {
        mode: args.histogram_mode.to_lib_mode(),
        dither_mode: args.histogram_dither.to_dither_mode(),
        colorspace_aware: !args.no_colorspace_aware_histogram,
        colorspace_aware_space: args.histogram_distance_space.to_perceptual_space(),
    };
    let output_dither_mode = args.output_dither.to_cs_dither_mode();

    // Build the correction method (None if histogram is None)
    let correction_method = build_correction_method(
        histogram,
        args.keep_luminosity,
        args.tiled_luminosity,
        args.perceptual,
    );

    if args.verbose {
        eprintln!("Histogram: {:?}", histogram);
        eprintln!("Format: {} ({} bits/pixel)", format.name, format.total_bits);
        if format.is_grayscale {
            eprintln!("  Grayscale: {} bits", format.bits_r);
        } else {
            eprintln!(
                "  RGB: {}+{}+{} bits",
                format.bits_r, format.bits_g, format.bits_b
            );
        }
        if needs_reference {
            eprintln!("Histogram mode: {:?}", args.histogram_mode);
        }
        eprintln!("Output dither: {:?}", args.output_dither);
        eprintln!(
            "Output colorspace: {:?}{}",
            output_colorspace,
            if args.output_distance_space.is_none() { " (default)" } else { "" }
        );
        eprintln!("Seed: {}", args.seed);
    }

    // Check if at least one output is specified
    let has_channel_output = args.output_raw_r.is_some()
        || args.output_raw_g.is_some()
        || args.output_raw_b.is_some();
    if args.output.is_none()
        && args.output_raw.is_none()
        && !has_channel_output
        && args.output_meta.is_none()
    {
        return Err(
            "No output specified. Use --output, --output-raw, --output-raw-r/g/b, or --output-meta"
                .to_string(),
        );
    }

    // Check binary output compatibility
    if args.output_raw.is_some() && !format.supports_binary() {
        return Err(format!(
            "Format {} ({} bits) does not support binary output. Binary output requires 1, 2, 4, 8, 16, 18 (RGB666), 24, or 32 bits per pixel.",
            format.name, format.total_bits
        ));
    }

    // Check channel output compatibility (requires RGB format, not grayscale)
    if has_channel_output && format.is_grayscale {
        return Err(
            "Separate channel outputs (--output-raw-r/g/b) require RGB format, not grayscale"
                .to_string(),
        );
    }

    // Validate stride
    if !is_valid_stride(args.stride) {
        return Err(format!(
            "Invalid stride {}. Must be a power of 2 between 1 and 128.",
            args.stride
        ));
    }

    // Pre-load input image with ICC profile (single file open)
    if args.verbose {
        eprintln!("Loading: {}", args.input.display());
    }
    let decoded_input = load_image_from_path(&args.input)?;

    // Determine if input has premultiplied alpha that needs un-premultiplying
    // (check before moving fields out of decoded_input)
    let needs_unpremultiply = match args.input_premultiplied_alpha {
        PremultipliedAlpha::Yes => true,
        PremultipliedAlpha::No => false,
        PremultipliedAlpha::Auto => {
            // Only EXR has premultiplied alpha by default
            decoded_input.is_format_premultiplied_default()
        }
    };

    let input_img = decoded_input.image;
    let input_icc = decoded_input.icc_profile;
    let input_cicp = decoded_input.cicp;

    if args.verbose && needs_unpremultiply {
        eprintln!("  Input has premultiplied alpha (will un-premultiply)");
    }

    // Determine if linear-space processing is needed
    // - Grayscale requires linear for correct luminance computation
    // - Non-sRGB input profiles require linear path for proper color handling
    // - Premultiplied alpha requires linear path for correct un-premultiplication
    let needs_resize = args.width.is_some() || args.height.is_some();

    // Check if color profile processing is actually needed (based on file contents, not just CLI flag)
    // Priority: CICP (authoritative) > ICC profile > CICP fallback
    let needs_profile_processing = match args.input_profile {
        InputColorProfile::Srgb => false,
        InputColorProfile::Linear => true, // User says input is linear, we need to skip gamma decode
        InputColorProfile::Auto => {
            // Check CICP first (authoritative)
            if is_cicp_srgb(&input_cicp) {
                false // CICP says sRGB, no conversion needed
            } else if is_cicp_linear_srgb(&input_cicp) {
                true // CICP says linear, need linear path
            } else if is_cicp_needs_conversion(&input_cicp) {
                true // CICP says non-sRGB, need conversion (via ICC or CICP)
            } else {
                // CICP unspecified, fall back to ICC check
                input_icc
                    .as_ref()
                    .map(|icc_data| !is_profile_srgb_verbose(icc_data, args.verbose))
                    .unwrap_or(false)
            }
        }
        InputColorProfile::Icc => {
            // Check if file has any ICC profile
            input_icc.is_some()
        }
        InputColorProfile::Cicp => {
            // Check if CICP is usable (not unspecified, not sRGB)
            can_use_cicp(&input_cicp) && !is_cicp_srgb(&input_cicp)
        }
    };

    let needs_linear = needs_reference || needs_resize || format.is_grayscale || needs_profile_processing || needs_unpremultiply;

    // Process image based on whether linear space is needed
    let (dither_result, width, height) = if needs_linear {
        // Linear RGB path: load -> resize -> color correct -> dither
        if args.verbose {
            eprintln!("Processing in linear RGB space...");
        }

        let (input_pixels, src_width, src_height, input_has_alpha) = convert_to_linear(&input_img, &input_icc, &input_cicp, args.input_profile, needs_unpremultiply, args.verbose)?;

        // Resize in linear RGB space
        let mut resize_progress = |p: f32| print_progress("Resize", p);
        let (input_pixels, width, height) = resize_linear(
            &input_pixels,
            src_width, src_height,
            args.width, args.height,
            args.scale_method.to_rescale_method(),
            args.verbose,
            if args.progress { Some(&mut resize_progress) } else { None },
        )?;
        if args.progress {
            clear_progress();
        }

        let width_usize = width as usize;
        let height_usize = height as usize;

        let pixels_to_dither = if needs_reference {
            // Load reference and apply color correction
            let ref_path = args.r#ref.as_ref().unwrap();
            // Load reference with specified profile handling (default: Auto detects ICC)
            if args.verbose {
                eprintln!("Loading: {}", ref_path.display());
            }
            let decoded_ref = load_image_from_path(ref_path)?;
            // Reference doesn't need alpha or un-premultiplying - we only use it for color matching
            let (ref_pixels, ref_width, ref_height, _) = convert_to_linear(&decoded_ref.image, &decoded_ref.icc_profile, &decoded_ref.cicp, args.ref_profile, false, args.verbose)?;
            let ref_width_usize = ref_width as usize;
            let ref_height_usize = ref_height as usize;

            let mut correction_progress = |p: f32| print_progress("Color Correct", p);
            let corrected = color_correct(
                &input_pixels,
                &ref_pixels,
                width_usize,
                height_usize,
                ref_width_usize,
                ref_height_usize,
                correction_method.expect("Method should not be None when reference is provided"),
                histogram_options,
                if args.progress { Some(&mut correction_progress) } else { None },
            );
            if args.progress {
                clear_progress();
            }
            corrected
        } else {
            input_pixels
        };

        let mut dither_progress = |p: f32| print_progress("Dither", p);
        let result = dither_pixels(
            pixels_to_dither,
            width_usize,
            height_usize,
            &format,
            !args.no_colorspace_aware_output,
            output_dither_mode,
            output_colorspace.to_perceptual_space(),
            args.seed,
            input_has_alpha,
            if args.progress { Some(&mut dither_progress) } else { None },
        );
        if args.progress {
            clear_progress();
        }

        (result, width, height)
    } else {
        // sRGB path: RGB dither-only (no resize, no color correction, no grayscale)
        // Avoids unnecessary sRGB -> linear -> sRGB conversion
        if args.verbose {
            eprintln!("Dithering RGB channels (sRGB path)...");
        }

        let (input_pixels, width, height, input_has_alpha) = convert_to_srgb_255(&input_img, args.verbose);

        let mut dither_progress = |p: f32| print_progress("Dither", p);
        let result = dither_pixels_srgb_rgb(
            input_pixels,
            width as usize,
            height as usize,
            &format,
            !args.no_colorspace_aware_output,
            output_dither_mode,
            output_colorspace.to_perceptual_space(),
            args.seed,
            input_has_alpha,
            if args.progress { Some(&mut dither_progress) } else { None },
        );
        if args.progress {
            clear_progress();
        }

        (result, width, height)
    };

    let width_usize = width as usize;
    let height_usize = height as usize;

    // Track outputs for metadata
    let mut outputs: Vec<(String, PathBuf, usize)> = Vec::new();

    // Write PNG output
    if let Some(ref png_path) = args.output {
        if args.verbose {
            eprintln!("Writing PNG: {}", png_path.display());
        }

        if format.is_grayscale {
            save_png_grayscale(png_path, &dither_result.interleaved, width, height)?;
        } else if dither_result.has_alpha {
            save_png_rgba(png_path, &dither_result.interleaved, width, height)?;
        } else {
            save_png_rgb(png_path, &dither_result.interleaved, width, height)?;
        }

        let size = std::fs::metadata(png_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        outputs.push(("png".to_string(), png_path.clone(), size));
    }

    // Write binary output (respects --stride setting, default 1 = packed)
    if let Some(ref bin_path) = args.output_raw {
        let fill = args.stride_fill.to_stride_fill();
        let row_aligned = args.stride > 1;

        if args.verbose {
            if row_aligned {
                eprintln!(
                    "Writing binary (stride={}, fill={:?}): {}",
                    args.stride, args.stride_fill, bin_path.display()
                );
            } else {
                eprintln!("Writing binary (packed): {}", bin_path.display());
            }
        }

        let bin_data = encode_binary(&dither_result, &format, width_usize, height_usize, row_aligned, args.stride, fill);

        let mut file = File::create(bin_path)
            .map_err(|e| format!("Failed to create {}: {}", bin_path.display(), e))?;
        file.write_all(&bin_data)
            .map_err(|e| format!("Failed to write {}: {}", bin_path.display(), e))?;

        let label = if row_aligned { "binary_row_aligned" } else { "binary_packed" };
        outputs.push((label.to_string(), bin_path.clone(), bin_data.len()));
    }

    // Write separate channel outputs (R, G, B) - encode directly from interleaved data
    let fill = args.stride_fill.to_stride_fill();
    let row_aligned = args.stride > 1;
    let needs_channel_output = args.output_raw_r.is_some() || args.output_raw_g.is_some() || args.output_raw_b.is_some();
    if needs_channel_output && !dither_result.is_grayscale {
        if let Some(ref path) = args.output_raw_r {
            if args.verbose {
                eprintln!("Writing red channel binary: {}", path.display());
            }
            let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                &dither_result.interleaved, width_usize, height_usize, 3, 0, format.bits_r, args.stride, fill,
            );
            let mut file = File::create(path)
                .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
            file.write_all(&bin_data)
                .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
            let label = if row_aligned { "binary_r_row_aligned" } else { "binary_r" };
            outputs.push((label.to_string(), path.clone(), bin_data.len()));
        }

        if let Some(ref path) = args.output_raw_g {
            if args.verbose {
                eprintln!("Writing green channel binary: {}", path.display());
            }
            let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                &dither_result.interleaved, width_usize, height_usize, 3, 1, format.bits_g, args.stride, fill,
            );
            let mut file = File::create(path)
                .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
            file.write_all(&bin_data)
                .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
            let label = if row_aligned { "binary_g_row_aligned" } else { "binary_g" };
            outputs.push((label.to_string(), path.clone(), bin_data.len()));
        }

        if let Some(ref path) = args.output_raw_b {
            if args.verbose {
                eprintln!("Writing blue channel binary: {}", path.display());
            }
            let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                &dither_result.interleaved, width_usize, height_usize, 3, 2, format.bits_b, args.stride, fill,
            );
            let mut file = File::create(path)
                .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
            file.write_all(&bin_data)
                .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
            let label = if row_aligned { "binary_b_row_aligned" } else { "binary_b" };
            outputs.push((label.to_string(), path.clone(), bin_data.len()));
        }
    }

    // Write metadata JSON
    if let Some(ref meta_path) = args.output_meta {
        if args.verbose {
            eprintln!("Writing metadata: {}", meta_path.display());
        }
        write_metadata(
            meta_path,
            &args,
            &format,
            histogram,
            output_colorspace,
            width,
            height,
            &outputs,
        )?;
    }

    if args.verbose {
        eprintln!("Done!");
    }

    Ok(())
}
