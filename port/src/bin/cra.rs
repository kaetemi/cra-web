//! CRA Color Correction CLI
//!
//! Command-line interface for color correction algorithms,
//! matching the functionality of the Python scripts.

use clap::{Parser, ValueEnum};
use image::{GenericImageView, ImageBuffer, Rgb};
use std::path::PathBuf;

use cra_wasm::{
    color_correct_basic_lab, color_correct_basic_oklab, color_correct_basic_rgb,
    color_correct_cra_lab, color_correct_cra_oklab, color_correct_cra_rgb,
    color_correct_tiled_lab, color_correct_tiled_oklab,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Method {
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

#[derive(Parser, Debug)]
#[command(name = "cra")]
#[command(author, version, about = "CRA Color Correction Tool", long_about = None)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    input: PathBuf,

    /// Reference image path
    #[arg(short, long)]
    r#ref: PathBuf,

    /// Output image path
    #[arg(short, long)]
    output: PathBuf,

    /// Color correction method
    #[arg(short, long, value_enum, default_value_t = Method::CraLab)]
    method: Method,

    /// Preserve original luminosity (L channel) - applies to basic-lab, basic-oklab, cra-lab, cra-oklab
    #[arg(long)]
    keep_luminosity: bool,

    /// Process L channel per-tile before global match - applies to tiled-lab, tiled-oklab
    #[arg(long)]
    tiled_luminosity: bool,

    /// Use perceptual weighting - applies to cra-rgb
    #[arg(short, long)]
    perceptual: bool,

    /// Histogram matching mode
    #[arg(long, value_enum, default_value_t = HistogramMode::Binned)]
    histogram_mode: HistogramMode,

    /// Dithering method for final output quantization
    #[arg(long, value_enum, default_value_t = DitherMethod::FsStandard)]
    output_dither: DitherMethod,

    /// Dithering method for histogram processing (only used with --histogram-mode=binned)
    #[arg(long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    histogram_dither: DitherMethod,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn load_image_rgb(path: &PathBuf, verbose: bool) -> Result<(Vec<u8>, usize, usize), String> {
    if verbose {
        eprintln!("Loading: {}", path.display());
    }

    let img = image::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();

    // Convert to flat RGB array (RGBRGB...)
    let data: Vec<u8> = rgb_img.as_raw().to_vec();

    if verbose {
        eprintln!("  Dimensions: {}x{}", width, height);
    }

    Ok((data, width as usize, height as usize))
}

fn save_image_rgb(
    path: &PathBuf,
    data: &[u8],
    width: usize,
    height: usize,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        eprintln!("Saving: {}", path.display());
    }

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width as u32, height as u32, data.to_vec())
            .ok_or_else(|| "Failed to create image buffer".to_string())?;

    img.save(path)
        .map_err(|e| format!("Failed to save {}: {}", path.display(), e))?;

    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let histogram_mode = args.histogram_mode.to_u8();
    let histogram_dither = args.histogram_dither.to_u8();
    let output_dither = args.output_dither.to_u8();

    if args.verbose {
        eprintln!("Method: {:?}", args.method);
        eprintln!("Histogram mode: {:?}", args.histogram_mode);
        eprintln!("Output dither: {:?}", args.output_dither);
        if histogram_mode == 0 {
            eprintln!("Histogram dither: {:?}", args.histogram_dither);
        }
    }

    // Load images
    let (input_data, input_width, input_height) = load_image_rgb(&args.input, args.verbose)?;
    let (ref_data, ref_width, ref_height) = load_image_rgb(&args.r#ref, args.verbose)?;

    if args.verbose {
        eprintln!("Processing...");
    }

    // Apply color correction
    let output_data = match args.method {
        Method::BasicLab => color_correct_basic_lab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.keep_luminosity,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::BasicRgb => color_correct_basic_rgb(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::BasicOklab => color_correct_basic_oklab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.keep_luminosity,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::CraLab => color_correct_cra_lab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.keep_luminosity,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::CraRgb => color_correct_cra_rgb(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.perceptual,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::CraOklab => color_correct_cra_oklab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.keep_luminosity,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::TiledLab => color_correct_tiled_lab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.tiled_luminosity,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
        Method::TiledOklab => color_correct_tiled_oklab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.tiled_luminosity,
            histogram_mode,
            histogram_dither,
            output_dither,
        ),
    };

    // Save output
    save_image_rgb(&args.output, &output_data, input_width, input_height, args.verbose)?;

    if args.verbose {
        eprintln!("Done!");
    }

    Ok(())
}
