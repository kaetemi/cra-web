//! CRA - Unified Color Correction and Dithering CLI
//!
//! A unified command-line tool for color correction and error diffusion dithering.
//! Pipeline: image -> linear RGB -> optional resize -> optional color correction -> dithered output
//!
//! Supports:
//! - Color correction methods: basic-lab, basic-rgb, basic-oklab, cra-lab, cra-rgb, cra-oklab, tiled-lab, tiled-oklab
//! - RGB formats: RGB111, RGB332, RGB565, RGB888, etc. (parse bit counts from format string)
//! - Grayscale formats: L1, L2, L4, L8 (single channel)
//! - Multiple dithering algorithms: Floyd-Steinberg, Jarvis-Judice-Ninke, Mixed
//! - Multiple perceptual spaces: OKLab, CIELAB (CIE76, CIE94, CIEDE2000)
//! - Output formats: PNG, raw binary, row-padded binary
//! - Metadata JSON output with parameters and dimensions

use clap::{Parser, ValueEnum};
use image::{GenericImageView, ImageBuffer, Luma, Rgb};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use cra_wasm::binary_format::{
    encode_gray_packed, encode_gray_row_aligned_stride, encode_rgb_packed,
    encode_rgb_row_aligned_stride, is_valid_stride, ColorFormat, StrideFill,
};
use cra_wasm::dither::colorspace_aware_dither_rgb_with_mode;
use cra_wasm::dither_colorspace_aware::DitherMode as CSDitherMode;
use cra_wasm::dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode;
use cra_wasm::dither_common::PerceptualSpace;
use cra_wasm::{
    color_correct_basic_lab, color_correct_basic_oklab, color_correct_basic_rgb,
    color_correct_cra_lab, color_correct_cra_oklab, color_correct_cra_rgb,
    color_correct_tiled_lab, color_correct_tiled_oklab,
};

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Method {
    /// No color correction (dither only)
    None,
    /// Basic LAB histogram matching (color_correction_basic.py)
    BasicLab,
    /// Basic RGB histogram matching (color_correction_basic_rgb.py)
    BasicRgb,
    /// Basic Oklab histogram matching (perceptually uniform)
    BasicOklab,
    /// CRA LAB - Chroma Rotation Averaging in LAB space (color_correction_cra.py)
    CraLab,
    /// CRA RGB - Chroma Rotation Averaging in RGB space (color_correction_cra_rgb.py)
    CraRgb,
    /// CRA Oklab - Chroma Rotation Averaging in Oklab space (perceptually uniform)
    CraOklab,
    /// Tiled LAB with overlapping blocks (color_correction_tiled.py)
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
}

impl DitherMethod {
    fn to_u8(self) -> u8 {
        match self {
            DitherMethod::FsStandard => 0,
            DitherMethod::FsSerpentine => 1,
            DitherMethod::JjnStandard => 2,
            DitherMethod::JjnSerpentine => 3,
            DitherMethod::MixedStandard => 4,
            DitherMethod::MixedSerpentine => 5,
            DitherMethod::MixedRandom => 6,
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
    fn to_u8(self) -> u8 {
        match self {
            HistogramMode::Binned => 0,
            HistogramMode::F32Endpoint => 1,
            HistogramMode::F32Midpoint => 2,
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
    fn to_u8(self) -> u8 {
        match self {
            ColorSpace::LabCie76 => 0,
            ColorSpace::Oklab => 1,
            ColorSpace::LabCie94 => 2,
            ColorSpace::LabCiede2000 => 3,
            ColorSpace::LinearRgb => 4,
            ColorSpace::YCbCr => 5,
        }
    }

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
// Command Line Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "cra")]
#[command(author, version, about = "CRA - Unified Color Correction and Dithering Tool", long_about = None)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    input: PathBuf,

    /// Reference image path (optional - required for color correction methods)
    #[arg(short, long)]
    r#ref: Option<PathBuf>,

    /// Output PNG image path (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output raw binary file path (optional) - packed pixel data
    #[arg(long)]
    output_bin: Option<PathBuf>,

    /// Output row-aligned binary file path (optional) - each row padded to stride boundary
    #[arg(long)]
    output_bin_r: Option<PathBuf>,

    /// Output metadata JSON file path (optional)
    #[arg(long)]
    output_meta: Option<PathBuf>,

    /// Output format: RGB with bit counts (e.g., RGB332, RGB565, RGB888) or L with bits (e.g., L4, L8)
    #[arg(short, long, default_value = "RGB888")]
    format: String,

    /// Color correction method (default: none if no reference, cra-lab if reference provided)
    #[arg(short, long, value_enum)]
    method: Option<Method>,

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

    /// Use color-aware dithering for histogram quantization (only with --histogram-mode=binned)
    /// Applies to: cra-lab, cra-oklab, tiled-lab, tiled-oklab
    #[arg(long)]
    color_aware_histogram: bool,

    /// Perceptual space for color-aware histogram dithering distance metric
    #[arg(long, value_enum, default_value_t = ColorSpace::Oklab)]
    histogram_distance_space: ColorSpace,

    /// Use color-aware dithering for final RGB output (joint RGB processing)
    /// Applies to: cra-lab, cra-oklab, tiled-lab, tiled-oklab
    #[arg(long)]
    color_aware_output: bool,

    /// Perceptual space for output dithering distance metric (for both color correction and dither-only modes)
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

    /// Downscale image to this width (preserves aspect ratio)
    #[arg(long)]
    width: Option<u32>,

    /// Downscale image to this height (preserves aspect ratio)
    #[arg(long)]
    height: Option<u32>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

// ============================================================================
// Image Loading and Processing
// ============================================================================

fn load_image_rgb(path: &PathBuf, verbose: bool) -> Result<(Vec<u8>, u32, u32), String> {
    if verbose {
        eprintln!("Loading: {}", path.display());
    }

    let img = image::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();
    let data: Vec<u8> = rgb_img.as_raw().to_vec();

    if verbose {
        eprintln!("  Dimensions: {}x{}", width, height);
    }

    Ok((data, width, height))
}

fn downscale_image(
    data: &[u8],
    src_width: u32,
    src_height: u32,
    target_width: Option<u32>,
    target_height: Option<u32>,
    verbose: bool,
) -> Result<(Vec<u8>, u32, u32), String> {
    let (dst_width, dst_height) = match (target_width, target_height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => {
            let aspect = src_height as f64 / src_width as f64;
            (w, (w as f64 * aspect).round() as u32)
        }
        (None, Some(h)) => {
            let aspect = src_width as f64 / src_height as f64;
            ((h as f64 * aspect).round() as u32, h)
        }
        (None, None) => return Ok((data.to_vec(), src_width, src_height)),
    };

    if dst_width == src_width && dst_height == src_height {
        return Ok((data.to_vec(), src_width, src_height));
    }

    if verbose {
        eprintln!(
            "Resizing: {}x{} -> {}x{}",
            src_width, src_height, dst_width, dst_height
        );
    }

    let src_img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(src_width, src_height, data.to_vec())
            .ok_or_else(|| "Failed to create source image buffer".to_string())?;

    let resized = image::imageops::resize(
        &src_img,
        dst_width,
        dst_height,
        image::imageops::FilterType::Lanczos3,
    );

    Ok((resized.into_raw(), dst_width, dst_height))
}

fn rgb_to_grayscale(data: &[u8]) -> Vec<f32> {
    // Rec.709 luminance coefficients
    const R_COEF: f32 = 0.2126;
    const G_COEF: f32 = 0.7152;
    const B_COEF: f32 = 0.0722;

    let pixels = data.len() / 3;
    let mut gray = Vec::with_capacity(pixels);

    for i in 0..pixels {
        let r = data[i * 3] as f32;
        let g = data[i * 3 + 1] as f32;
        let b = data[i * 3 + 2] as f32;
        gray.push(r * R_COEF + g * G_COEF + b * B_COEF);
    }

    gray
}

fn split_channels(data: &[u8]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = data.len() / 3;
    let mut r = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);

    for i in 0..pixels {
        r.push(data[i * 3] as f32);
        g.push(data[i * 3 + 1] as f32);
        b.push(data[i * 3 + 2] as f32);
    }

    (r, g, b)
}

// ============================================================================
// Dithering (for dither-only mode)
// ============================================================================

fn dither_grayscale(
    gray: &[f32],
    width: usize,
    height: usize,
    bits: u8,
    space: PerceptualSpace,
    mode: CSDitherMode,
    seed: u32,
) -> Vec<u8> {
    colorspace_aware_dither_gray_with_mode(gray, width, height, bits, space, mode, seed)
}

fn dither_rgb(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    space: PerceptualSpace,
    mode: CSDitherMode,
    seed: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    colorspace_aware_dither_rgb_with_mode(
        r, g, b, width, height, bits_r, bits_g, bits_b, space, mode, seed,
    )
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

// ============================================================================
// Metadata JSON
// ============================================================================

fn write_metadata(
    path: &PathBuf,
    args: &Args,
    format: &ColorFormat,
    method: Method,
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
    json.push_str(&format!("  \"method\": \"{:?}\",\n", method));
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

    // Determine method: user-specified, or default based on whether reference is provided
    let method = args.method.unwrap_or(if args.r#ref.is_some() {
        Method::CraLab
    } else {
        Method::None
    });

    // Validate: color correction methods require a reference image
    let needs_reference = !matches!(method, Method::None);
    if needs_reference && args.r#ref.is_none() {
        return Err(format!(
            "Method {:?} requires a reference image. Use --ref <path> or --method none for dither-only mode.",
            method
        ));
    }

    // Determine output colorspace for dithering
    // Default: OKLab for RGB, CIE94 for grayscale
    let output_colorspace = args.output_distance_space.unwrap_or(if format.is_grayscale {
        ColorSpace::LabCie94
    } else {
        ColorSpace::Oklab
    });

    let histogram_mode = args.histogram_mode.to_u8();
    let histogram_dither = args.histogram_dither.to_u8();
    let output_dither = args.output_dither.to_u8();
    let histogram_distance_space = args.histogram_distance_space.to_u8();
    let output_distance_space_u8 = output_colorspace.to_u8();

    if args.verbose {
        eprintln!("Method: {:?}", method);
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
            if args.color_aware_output {
                eprintln!("Color-aware output: true");
            }
            if histogram_mode == 0 {
                eprintln!("Histogram dither: {:?}", args.histogram_dither);
                if args.color_aware_histogram {
                    eprintln!("Color-aware histogram: true");
                    eprintln!("Histogram distance space: {:?}", args.histogram_distance_space);
                }
            }
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
    if args.output.is_none()
        && args.output_bin.is_none()
        && args.output_bin_r.is_none()
        && args.output_meta.is_none()
    {
        return Err(
            "No output specified. Use --output, --output-bin, --output-bin-r, or --output-meta"
                .to_string(),
        );
    }

    // Check binary output compatibility
    if (args.output_bin.is_some() || args.output_bin_r.is_some()) && !format.supports_binary() {
        return Err(format!(
            "Format {} ({} bits) does not support binary output. Binary output requires 1, 2, 4, 8, 16, 18 (RGB666), 24, or 32 bits per pixel.",
            format.name, format.total_bits
        ));
    }

    // Validate stride
    if !is_valid_stride(args.stride) {
        return Err(format!(
            "Invalid stride {}. Must be a power of 2 between 1 and 128.",
            args.stride
        ));
    }

    // Load input image
    let (input_data, src_width, src_height) = load_image_rgb(&args.input, args.verbose)?;

    // Downscale if requested
    let (input_data, width, height) = downscale_image(
        &input_data,
        src_width,
        src_height,
        args.width,
        args.height,
        args.verbose,
    )?;

    let width_usize = width as usize;
    let height_usize = height as usize;

    if args.verbose {
        eprintln!("Processing...");
    }

    // Process image: color correction + dithering, or dithering only
    let dithered_gray: Option<Vec<u8>>;
    let dithered_r: Option<Vec<u8>>;
    let dithered_g: Option<Vec<u8>>;
    let dithered_b: Option<Vec<u8>>;
    let dithered_interleaved: Vec<u8>;

    if needs_reference {
        // Color correction path: uses the WASM library functions which handle
        // internal dithering and return sRGB888 output, then we apply final dithering
        let ref_path = args.r#ref.as_ref().unwrap();
        let (ref_data, ref_width, ref_height) = load_image_rgb(ref_path, args.verbose)?;

        // Apply color correction (returns sRGB888)
        let corrected = match method {
            Method::None => unreachable!(),
            Method::BasicLab => color_correct_basic_lab(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.keep_luminosity,
                histogram_mode,
                histogram_dither,
                output_dither,
            ),
            Method::BasicRgb => color_correct_basic_rgb(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                histogram_mode,
                histogram_dither,
                output_dither,
            ),
            Method::BasicOklab => color_correct_basic_oklab(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.keep_luminosity,
                histogram_mode,
                histogram_dither,
                output_dither,
            ),
            Method::CraLab => color_correct_cra_lab(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.keep_luminosity,
                histogram_mode,
                histogram_dither,
                args.color_aware_histogram,
                histogram_distance_space,
                output_dither,
                args.color_aware_output,
                output_distance_space_u8,
            ),
            Method::CraRgb => color_correct_cra_rgb(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.perceptual,
                histogram_mode,
                histogram_dither,
                output_dither,
            ),
            Method::CraOklab => color_correct_cra_oklab(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.keep_luminosity,
                histogram_mode,
                histogram_dither,
                args.color_aware_histogram,
                histogram_distance_space,
                output_dither,
                args.color_aware_output,
                output_distance_space_u8,
            ),
            Method::TiledLab => color_correct_tiled_lab(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.tiled_luminosity,
                histogram_mode,
                histogram_dither,
                args.color_aware_histogram,
                histogram_distance_space,
                output_dither,
                args.color_aware_output,
                output_distance_space_u8,
            ),
            Method::TiledOklab => color_correct_tiled_oklab(
                &input_data,
                width_usize,
                height_usize,
                &ref_data,
                ref_width as usize,
                ref_height as usize,
                args.tiled_luminosity,
                histogram_mode,
                histogram_dither,
                args.color_aware_histogram,
                histogram_distance_space,
                output_dither,
                args.color_aware_output,
                output_distance_space_u8,
            ),
        };

        // If format is RGB888, we're done - the color correction already dithered to 8-bit
        // Otherwise, we need to apply additional bit-depth dithering
        if format.is_grayscale {
            // Convert corrected RGB to grayscale and dither to target bit depth
            let gray = rgb_to_grayscale(&corrected);
            let gray_out = dither_grayscale(
                &gray,
                width_usize,
                height_usize,
                format.bits_r,
                output_colorspace.to_perceptual_space(),
                args.output_dither.to_cs_dither_mode(),
                args.seed,
            );
            dithered_interleaved = gray_out.clone();
            dithered_gray = Some(gray_out);
            dithered_r = None;
            dithered_g = None;
            dithered_b = None;
        } else if format.bits_r == 8 && format.bits_g == 8 && format.bits_b == 8 {
            // RGB888 - use corrected output directly
            let (r, g, b) = split_channels(&corrected);
            let r_u8: Vec<u8> = r.iter().map(|&v| v as u8).collect();
            let g_u8: Vec<u8> = g.iter().map(|&v| v as u8).collect();
            let b_u8: Vec<u8> = b.iter().map(|&v| v as u8).collect();
            dithered_interleaved = corrected;
            dithered_gray = None;
            dithered_r = Some(r_u8);
            dithered_g = Some(g_u8);
            dithered_b = Some(b_u8);
        } else {
            // Non-RGB888 - need additional dithering to target bit depth
            let (r, g, b) = split_channels(&corrected);
            let (r_out, g_out, b_out) = dither_rgb(
                &r,
                &g,
                &b,
                width_usize,
                height_usize,
                format.bits_r,
                format.bits_g,
                format.bits_b,
                output_colorspace.to_perceptual_space(),
                args.output_dither.to_cs_dither_mode(),
                args.seed,
            );

            let pixels = width_usize * height_usize;
            dithered_interleaved = {
                let mut out = Vec::with_capacity(pixels * 3);
                for i in 0..pixels {
                    out.push(r_out[i]);
                    out.push(g_out[i]);
                    out.push(b_out[i]);
                }
                out
            };
            dithered_gray = None;
            dithered_r = Some(r_out);
            dithered_g = Some(g_out);
            dithered_b = Some(b_out);
        }
    } else {
        // Dither-only path (no color correction)
        let perceptual_space = output_colorspace.to_perceptual_space();
        let cs_mode = args.output_dither.to_cs_dither_mode();

        if format.is_grayscale {
            if args.verbose {
                eprintln!("Converting to grayscale and dithering...");
            }
            let gray = rgb_to_grayscale(&input_data);
            let gray_out = dither_grayscale(
                &gray,
                width_usize,
                height_usize,
                format.bits_r,
                perceptual_space,
                cs_mode,
                args.seed,
            );
            dithered_interleaved = gray_out.clone();
            dithered_gray = Some(gray_out);
            dithered_r = None;
            dithered_g = None;
            dithered_b = None;
        } else {
            if args.verbose {
                eprintln!("Dithering RGB channels...");
            }
            let (r, g, b) = split_channels(&input_data);
            let (r_out, g_out, b_out) = dither_rgb(
                &r,
                &g,
                &b,
                width_usize,
                height_usize,
                format.bits_r,
                format.bits_g,
                format.bits_b,
                perceptual_space,
                cs_mode,
                args.seed,
            );

            let pixels = width_usize * height_usize;
            dithered_interleaved = {
                let mut out = Vec::with_capacity(pixels * 3);
                for i in 0..pixels {
                    out.push(r_out[i]);
                    out.push(g_out[i]);
                    out.push(b_out[i]);
                }
                out
            };
            dithered_gray = None;
            dithered_r = Some(r_out);
            dithered_g = Some(g_out);
            dithered_b = Some(b_out);
        }
    }

    // Track outputs for metadata
    let mut outputs: Vec<(String, PathBuf, usize)> = Vec::new();

    // Write PNG output
    if let Some(ref png_path) = args.output {
        if args.verbose {
            eprintln!("Writing PNG: {}", png_path.display());
        }

        if format.is_grayscale {
            save_png_grayscale(png_path, &dithered_interleaved, width, height)?;
        } else {
            save_png_rgb(png_path, &dithered_interleaved, width, height)?;
        }

        let size = std::fs::metadata(png_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        outputs.push(("png".to_string(), png_path.clone(), size));
    }

    // Write packed binary output
    if let Some(ref bin_path) = args.output_bin {
        if args.verbose {
            eprintln!("Writing binary (packed): {}", bin_path.display());
        }

        let fill = args.stride_fill.to_stride_fill();
        let bin_data = if format.is_grayscale {
            encode_gray_packed(
                dithered_gray.as_ref().unwrap(),
                width_usize,
                height_usize,
                format.bits_r,
            )
        } else {
            encode_rgb_packed(
                dithered_r.as_ref().unwrap(),
                dithered_g.as_ref().unwrap(),
                dithered_b.as_ref().unwrap(),
                width_usize,
                height_usize,
                format.bits_r,
                format.bits_g,
                format.bits_b,
                fill,
            )
        };

        let mut file = File::create(bin_path)
            .map_err(|e| format!("Failed to create {}: {}", bin_path.display(), e))?;
        file.write_all(&bin_data)
            .map_err(|e| format!("Failed to write {}: {}", bin_path.display(), e))?;

        outputs.push((
            "binary_packed".to_string(),
            bin_path.clone(),
            bin_data.len(),
        ));
    }

    // Write row-aligned binary output
    if let Some(ref bin_r_path) = args.output_bin_r {
        if args.verbose {
            eprintln!(
                "Writing binary (row-aligned, stride={}, fill={:?}): {}",
                args.stride,
                args.stride_fill,
                bin_r_path.display()
            );
        }

        let fill = args.stride_fill.to_stride_fill();
        let bin_data = if format.is_grayscale {
            encode_gray_row_aligned_stride(
                dithered_gray.as_ref().unwrap(),
                width_usize,
                height_usize,
                format.bits_r,
                args.stride,
                fill,
            )
        } else {
            encode_rgb_row_aligned_stride(
                dithered_r.as_ref().unwrap(),
                dithered_g.as_ref().unwrap(),
                dithered_b.as_ref().unwrap(),
                width_usize,
                height_usize,
                format.bits_r,
                format.bits_g,
                format.bits_b,
                args.stride,
                fill,
            )
        };

        let mut file = File::create(bin_r_path)
            .map_err(|e| format!("Failed to create {}: {}", bin_r_path.display(), e))?;
        file.write_all(&bin_data)
            .map_err(|e| format!("Failed to write {}: {}", bin_r_path.display(), e))?;

        outputs.push((
            "binary_row_aligned".to_string(),
            bin_r_path.clone(),
            bin_data.len(),
        ));
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
            method,
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
