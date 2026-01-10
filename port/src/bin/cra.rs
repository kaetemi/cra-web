//! CRA Color Correction CLI
//!
//! Command-line interface for color correction algorithms,
//! matching the functionality of the Python scripts.

use clap::{Parser, ValueEnum};
use image::{GenericImageView, ImageBuffer, Rgb};
use std::path::PathBuf;

use cra_wasm::{
    color_correct_basic_lab, color_correct_basic_rgb, color_correct_cra_lab,
    color_correct_cra_rgb, color_correct_tiled_lab,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Method {
    /// Basic LAB histogram matching (color_correction_basic.py)
    BasicLab,
    /// Basic RGB histogram matching (color_correction_basic_rgb.py)
    BasicRgb,
    /// CRA LAB - Chroma Rotation Averaging in LAB space (color_correction_cra.py)
    CraLab,
    /// CRA RGB - Chroma Rotation Averaging in RGB space (color_correction_cra_rgb.py)
    CraRgb,
    /// Tiled LAB with overlapping blocks (color_correction_tiled.py)
    TiledLab,
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

    /// Preserve original luminosity (L channel) - applies to basic-lab and cra-lab
    #[arg(long)]
    keep_luminosity: bool,

    /// Process L channel per-tile before global match - applies to tiled-lab
    #[arg(long)]
    tiled_luminosity: bool,

    /// Use perceptual weighting - applies to cra-rgb
    #[arg(short, long)]
    perceptual: bool,

    /// Use f32 sort-based histogram matching (no quantization artifacts)
    #[arg(long)]
    f32_histogram: bool,

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

    if args.verbose {
        eprintln!("Method: {:?}", args.method);
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
            args.f32_histogram,
        ),
        Method::BasicRgb => color_correct_basic_rgb(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.f32_histogram,
        ),
        Method::CraLab => color_correct_cra_lab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.keep_luminosity,
            args.f32_histogram,
        ),
        Method::CraRgb => color_correct_cra_rgb(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.perceptual,
            args.f32_histogram,
        ),
        Method::TiledLab => color_correct_tiled_lab(
            &input_data,
            input_width,
            input_height,
            &ref_data,
            ref_width,
            ref_height,
            args.tiled_luminosity,
            args.f32_histogram,
        ),
    };

    // Save output
    save_image_rgb(&args.output, &output_data, input_width, input_height, args.verbose)?;

    if args.verbose {
        eprintln!("Done!");
    }

    Ok(())
}
