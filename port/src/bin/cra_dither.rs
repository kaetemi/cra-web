//! CRA Dithering CLI Tool
//!
//! Standalone command-line tool for error diffusion dithering with configurable
//! bit depths, multiple output formats, and perceptual color space support.
//!
//! Supports:
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

use cra_wasm::dither::colorspace_aware_dither_rgb_with_mode;
use cra_wasm::dither_colorspace_aware::DitherMode as CSDitherMode;
use cra_wasm::dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode;
use cra_wasm::dither_common::PerceptualSpace;

// ============================================================================
// Command Line Arguments
// ============================================================================

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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ColorSpace {
    /// OKLab color space (default, recommended)
    Oklab,
    /// CIELAB with CIE76 (simple Euclidean distance)
    LabCie76,
    /// CIELAB with CIE94 (weighted distance)
    LabCie94,
    /// CIELAB with CIEDE2000 (most accurate)
    LabCiede2000,
    /// Linear RGB (not recommended)
    LinearRgb,
    /// Y'CbCr (not recommended)
    Ycbcr,
}

impl ColorSpace {
    fn to_perceptual_space(self) -> PerceptualSpace {
        match self {
            ColorSpace::Oklab => PerceptualSpace::OkLab,
            ColorSpace::LabCie76 => PerceptualSpace::LabCIE76,
            ColorSpace::LabCie94 => PerceptualSpace::LabCIE94,
            ColorSpace::LabCiede2000 => PerceptualSpace::LabCIEDE2000,
            ColorSpace::LinearRgb => PerceptualSpace::LinearRGB,
            ColorSpace::Ycbcr => PerceptualSpace::YCbCr,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "cra_dither")]
#[command(author, version, about = "CRA Dithering Tool - Error diffusion dithering with configurable bit depth", long_about = None)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    input: PathBuf,

    /// Output format: RGB with bit counts (e.g., RGB565, RGB111, RGB888) or L with bits (e.g., L4, L8)
    #[arg(short, long, default_value = "RGB565")]
    format: String,

    /// Output PNG image path (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output raw binary file path (optional) - packed pixel data
    #[arg(long)]
    output_bin: Option<PathBuf>,

    /// Output row-aligned binary file path (optional) - each row padded to byte boundary
    #[arg(long)]
    output_bin_r: Option<PathBuf>,

    /// Output metadata JSON file path (optional)
    #[arg(long)]
    output_meta: Option<PathBuf>,

    /// Dithering method
    #[arg(short, long, value_enum, default_value_t = DitherMethod::MixedStandard)]
    method: DitherMethod,

    /// Perceptual color space for distance calculations
    #[arg(short, long, value_enum, default_value_t = ColorSpace::Oklab)]
    colorspace: ColorSpace,

    /// Random seed for mixed dithering modes
    #[arg(short, long, default_value_t = 12345)]
    seed: u32,

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
// Format Parsing
// ============================================================================

/// Parsed color format with bit depths per channel
#[derive(Debug, Clone)]
struct ColorFormat {
    /// Format name (e.g., "RGB565", "L4")
    name: String,
    /// Whether this is a grayscale format
    is_grayscale: bool,
    /// Bits per red channel (or grayscale)
    bits_r: u8,
    /// Bits per green channel (0 for grayscale)
    bits_g: u8,
    /// Bits per blue channel (0 for grayscale)
    bits_b: u8,
    /// Total bits per pixel
    total_bits: u8,
}

impl ColorFormat {
    /// Parse a format string like "RGB565", "RGB111", "L4", "L8", etc.
    fn parse(format: &str) -> Result<Self, String> {
        let format_upper = format.to_uppercase();

        // Grayscale formats: L1, L2, L4, L8
        if format_upper.starts_with('L') {
            let bits_str = &format_upper[1..];
            let bits: u8 = bits_str
                .parse()
                .map_err(|_| format!("Invalid grayscale format '{}': expected L followed by bit count (1-8)", format))?;

            if bits < 1 || bits > 8 {
                return Err(format!("Grayscale bits must be 1-8, got {}", bits));
            }

            return Ok(ColorFormat {
                name: format_upper,
                is_grayscale: true,
                bits_r: bits,
                bits_g: 0,
                bits_b: 0,
                total_bits: bits,
            });
        }

        // RGB formats: RGB565, RGB111, RGB332, RGB888, etc.
        if format_upper.starts_with("RGB") {
            let bits_str = &format_upper[3..];

            if bits_str.len() != 3 {
                return Err(format!(
                    "Invalid RGB format '{}': expected RGB followed by 3 digits (e.g., RGB565)",
                    format
                ));
            }

            let bits_r: u8 = bits_str[0..1]
                .parse()
                .map_err(|_| format!("Invalid red bit count in '{}'", format))?;
            let bits_g: u8 = bits_str[1..2]
                .parse()
                .map_err(|_| format!("Invalid green bit count in '{}'", format))?;
            let bits_b: u8 = bits_str[2..3]
                .parse()
                .map_err(|_| format!("Invalid blue bit count in '{}'", format))?;

            if bits_r < 1 || bits_r > 8 {
                return Err(format!("Red bits must be 1-8, got {}", bits_r));
            }
            if bits_g < 1 || bits_g > 8 {
                return Err(format!("Green bits must be 1-8, got {}", bits_g));
            }
            if bits_b < 1 || bits_b > 8 {
                return Err(format!("Blue bits must be 1-8, got {}", bits_b));
            }

            let total_bits = bits_r + bits_g + bits_b;

            return Ok(ColorFormat {
                name: format_upper,
                is_grayscale: false,
                bits_r,
                bits_g,
                bits_b,
                total_bits,
            });
        }

        Err(format!(
            "Unknown format '{}': expected RGB### (e.g., RGB565) or L# (e.g., L4)",
            format
        ))
    }

    /// Check if this format can be represented in a standard binary output
    /// Binary output is supported for formats that fit within power-of-2 sizes
    fn supports_binary(&self) -> bool {
        // Supported: formats where total bits is 1, 2, 4, 8, 16, 24, or 32
        matches!(self.total_bits, 1 | 2 | 4 | 8 | 16 | 24 | 32)
    }

    /// Get the number of bytes per pixel for binary output (rounded up)
    fn bytes_per_pixel(&self) -> usize {
        ((self.total_bits as usize) + 7) / 8
    }

    /// Get the number of pixels that fit in one byte (for sub-byte formats)
    fn pixels_per_byte(&self) -> usize {
        if self.total_bits >= 8 {
            1
        } else {
            8 / (self.total_bits as usize)
        }
    }
}

// ============================================================================
// Image Loading and Processing
// ============================================================================

fn load_image(path: &PathBuf, verbose: bool) -> Result<(Vec<u8>, u32, u32), String> {
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
    // Determine target dimensions
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
            "Downscaling: {}x{} -> {}x{}",
            src_width, src_height, dst_width, dst_height
        );
    }

    // Use the image crate's resize functionality
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
// Dithering
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
// Binary Output Encoding
// ============================================================================

/// Encode RGB pixel to packed binary format
/// Returns the raw bits as a u32 (caller should mask to appropriate bit width)
fn encode_rgb_pixel(r: u8, g: u8, b: u8, bits_r: u8, bits_g: u8, bits_b: u8) -> u32 {
    // Extract the significant bits from each channel
    // The dithered values are bit-replicated, so we need to extract the original N bits
    let r_val = (r >> (8 - bits_r)) as u32;
    let g_val = (g >> (8 - bits_g)) as u32;
    let b_val = (b >> (8 - bits_b)) as u32;

    // Pack as R,G,B from MSB to LSB
    let g_shift = bits_b;
    let r_shift = bits_b + bits_g;

    (r_val << r_shift) | (g_val << g_shift) | b_val
}

/// Encode grayscale pixel to packed binary format
fn encode_gray_pixel(l: u8, bits: u8) -> u32 {
    (l >> (8 - bits)) as u32
}

/// Write packed binary output (continuous bit stream, no row padding)
fn write_binary_packed(
    output: &mut Vec<u8>,
    dithered_data: &[u8],
    format: &ColorFormat,
    width: usize,
    height: usize,
) {
    let total_bits = format.total_bits as usize;
    let total_pixels = width * height;

    if total_bits >= 8 {
        // Byte-aligned or multi-byte pixels
        let bytes_per_pixel = format.bytes_per_pixel();
        output.reserve(total_pixels * bytes_per_pixel);

        if format.is_grayscale {
            for i in 0..total_pixels {
                let val = encode_gray_pixel(dithered_data[i], format.bits_r);
                // Write little-endian (LSB first) to match reference implementation
                for b in 0..bytes_per_pixel {
                    output.push(((val >> (b * 8)) & 0xFF) as u8);
                }
            }
        } else {
            for i in 0..total_pixels {
                let r = dithered_data[i * 3];
                let g = dithered_data[i * 3 + 1];
                let b = dithered_data[i * 3 + 2];
                let val = encode_rgb_pixel(r, g, b, format.bits_r, format.bits_g, format.bits_b);
                // Write little-endian (LSB first) to match reference implementation
                for byte_idx in 0..bytes_per_pixel {
                    output.push(((val >> (byte_idx * 8)) & 0xFF) as u8);
                }
            }
        }
    } else {
        // Sub-byte pixels - pack multiple pixels per byte
        let pixels_per_byte = format.pixels_per_byte();
        let total_bytes = (total_pixels + pixels_per_byte - 1) / pixels_per_byte;
        output.reserve(total_bytes);

        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        for i in 0..total_pixels {
            let val = if format.is_grayscale {
                encode_gray_pixel(dithered_data[i], format.bits_r)
            } else {
                let r = dithered_data[i * 3];
                let g = dithered_data[i * 3 + 1];
                let b = dithered_data[i * 3 + 2];
                encode_rgb_pixel(r, g, b, format.bits_r, format.bits_g, format.bits_b)
            };

            // Pack from MSB to LSB
            let shift = 8 - bits_in_byte - total_bits;
            current_byte |= (val as u8) << shift;
            bits_in_byte += total_bits;

            if bits_in_byte == 8 {
                output.push(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }

        // Flush remaining bits
        if bits_in_byte > 0 {
            output.push(current_byte);
        }
    }
}

/// Write row-aligned binary output (each row padded to byte boundary)
fn write_binary_row_aligned(
    output: &mut Vec<u8>,
    dithered_data: &[u8],
    format: &ColorFormat,
    width: usize,
    height: usize,
) {
    let total_bits = format.total_bits as usize;

    if total_bits >= 8 {
        // For byte-aligned formats, row alignment is automatic
        write_binary_packed(output, dithered_data, format, width, height);
        return;
    }

    // Sub-byte pixels - pack each row separately with padding
    let pixels_per_byte = format.pixels_per_byte();
    let bytes_per_row = (width + pixels_per_byte - 1) / pixels_per_byte;
    output.reserve(bytes_per_row * height);

    for y in 0..height {
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: usize = 0;

        for x in 0..width {
            let i = y * width + x;
            let val = if format.is_grayscale {
                encode_gray_pixel(dithered_data[i], format.bits_r)
            } else {
                let r = dithered_data[i * 3];
                let g = dithered_data[i * 3 + 1];
                let b = dithered_data[i * 3 + 2];
                encode_rgb_pixel(r, g, b, format.bits_r, format.bits_g, format.bits_b)
            };

            // Pack from MSB to LSB
            let shift = 8 - bits_in_byte - total_bits;
            current_byte |= (val as u8) << shift;
            bits_in_byte += total_bits;

            if bits_in_byte == 8 {
                output.push(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }

        // Flush remaining bits at end of row
        if bits_in_byte > 0 {
            output.push(current_byte);
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

// ============================================================================
// Metadata JSON
// ============================================================================

fn write_metadata(
    path: &PathBuf,
    args: &Args,
    format: &ColorFormat,
    width: u32,
    height: u32,
    outputs: &[(String, PathBuf, usize)], // (type, path, size_bytes)
) -> Result<(), String> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"input\": \"{}\",\n", args.input.display()));
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

    json.push_str(&format!("  \"dither_method\": \"{:?}\",\n", args.method));
    json.push_str(&format!("  \"colorspace\": \"{:?}\",\n", args.colorspace));
    json.push_str(&format!("  \"seed\": {},\n", args.seed));

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

    if args.verbose {
        eprintln!("Format: {} ({} bits/pixel)", format.name, format.total_bits);
        if format.is_grayscale {
            eprintln!("  Grayscale: {} bits", format.bits_r);
        } else {
            eprintln!(
                "  RGB: {}+{}+{} bits",
                format.bits_r, format.bits_g, format.bits_b
            );
        }
        eprintln!("Dither method: {:?}", args.method);
        eprintln!("Colorspace: {:?}", args.colorspace);
        eprintln!("Seed: {}", args.seed);
    }

    // Check if at least one output is specified
    if args.output.is_none()
        && args.output_bin.is_none()
        && args.output_bin_r.is_none()
        && args.output_meta.is_none()
    {
        return Err("No output specified. Use --output, --output-bin, --output-bin-r, or --output-meta".to_string());
    }

    // Check binary output compatibility
    if (args.output_bin.is_some() || args.output_bin_r.is_some()) && !format.supports_binary() {
        return Err(format!(
            "Format {} ({} bits) does not support binary output. Binary output requires 1, 2, 4, 8, 16, 24, or 32 bits per pixel.",
            format.name, format.total_bits
        ));
    }

    // Load input image
    let (data, src_width, src_height) = load_image(&args.input, args.verbose)?;

    // Downscale if requested
    let (data, width, height) =
        downscale_image(&data, src_width, src_height, args.width, args.height, args.verbose)?;

    let width_usize = width as usize;
    let height_usize = height as usize;

    // Perform dithering
    let dithered_data: Vec<u8>;
    let perceptual_space = args.colorspace.to_perceptual_space();
    let cs_mode = args.method.to_cs_dither_mode();

    if format.is_grayscale {
        if args.verbose {
            eprintln!("Converting to grayscale and dithering...");
        }
        let gray = rgb_to_grayscale(&data);
        dithered_data = dither_grayscale(
            &gray,
            width_usize,
            height_usize,
            format.bits_r,
            perceptual_space,
            cs_mode,
            args.seed,
        );
    } else {
        if args.verbose {
            eprintln!("Dithering RGB channels...");
        }
        let (r, g, b) = split_channels(&data);
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

        // Interleave channels
        let pixels = width_usize * height_usize;
        dithered_data = {
            let mut out = Vec::with_capacity(pixels * 3);
            for i in 0..pixels {
                out.push(r_out[i]);
                out.push(g_out[i]);
                out.push(b_out[i]);
            }
            out
        };
    }

    // Track outputs for metadata
    let mut outputs: Vec<(String, PathBuf, usize)> = Vec::new();

    // Write PNG output
    if let Some(ref png_path) = args.output {
        if args.verbose {
            eprintln!("Writing PNG: {}", png_path.display());
        }

        if format.is_grayscale {
            save_png_grayscale(png_path, &dithered_data, width, height)?;
        } else {
            save_png_rgb(png_path, &dithered_data, width, height)?;
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

        let mut bin_data = Vec::new();
        write_binary_packed(&mut bin_data, &dithered_data, &format, width_usize, height_usize);

        let mut file = File::create(bin_path)
            .map_err(|e| format!("Failed to create {}: {}", bin_path.display(), e))?;
        file.write_all(&bin_data)
            .map_err(|e| format!("Failed to write {}: {}", bin_path.display(), e))?;

        outputs.push(("binary_packed".to_string(), bin_path.clone(), bin_data.len()));
    }

    // Write row-aligned binary output
    if let Some(ref bin_r_path) = args.output_bin_r {
        if args.verbose {
            eprintln!("Writing binary (row-aligned): {}", bin_r_path.display());
        }

        let mut bin_data = Vec::new();
        write_binary_row_aligned(&mut bin_data, &dithered_data, &format, width_usize, height_usize);

        let mut file = File::create(bin_r_path)
            .map_err(|e| format!("Failed to create {}: {}", bin_r_path.display(), e))?;
        file.write_all(&bin_data)
            .map_err(|e| format!("Failed to write {}: {}", bin_r_path.display(), e))?;

        outputs.push(("binary_row_aligned".to_string(), bin_r_path.clone(), bin_data.len()));
    }

    // Write metadata JSON
    if let Some(ref meta_path) = args.output_meta {
        if args.verbose {
            eprintln!("Writing metadata: {}", meta_path.display());
        }
        write_metadata(meta_path, &args, &format, width, height, &outputs)?;
    }

    if args.verbose {
        eprintln!("Done!");
    }

    Ok(())
}
