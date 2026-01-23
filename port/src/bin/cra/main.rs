//! CRA - Unified Color Correction and Dithering CLI
//!
//! A unified command-line tool for color correction and error diffusion dithering.
//! Pipeline: input sRGB -> linear RGB -> optional resize -> optional processing -> dither to sRGB -> output
//!
//! All processing occurs in linear RGB space for correct color math.

mod args;

use args::*;
use clap::Parser;
use image::{ColorType, DynamicImage, GenericImageView, ImageBuffer, Luma, LumaA, Rgb, Rgba};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use cra_wasm::binary_format::{
    encode_argb_row_aligned_stride,
    encode_channel_from_interleaved_row_aligned_stride,
    encode_gray_row_aligned_stride, encode_la_row_aligned_stride,
    encode_palettized_png, encode_explicit_palette_png, encode_rgb_row_aligned_stride,
    encode_palettized_gif, encode_explicit_palette_gif, supports_gif,
    is_valid_stride, supports_palettized_png, ColorFormat, RawImageMetadata, StrideFill,
};
use cra_wasm::dither::paletted::{DitherPalette, paletted_dither_rgba_gamut_mapped};
use cra_wasm::sfi::{SfiTransfer, write_sfi_bf16, write_sfi_f16, write_sfi_f32, write_sfi_f16_channels, write_sfi_bf16_channels};
use cra_wasm::dither::fp16::{dither_rgb_f16_with_mode, dither_rgba_f16_with_mode, Fp16WorkingSpace};
use cra_wasm::dither::bf16::{dither_rgb_bf16_with_mode, dither_rgba_bf16_with_mode, Bf16WorkingSpace};
use cra_wasm::color::{linear_pixels_to_grayscale, linear_to_srgb_single, srgb_to_linear_single, tonemap_aces_single, tonemap_aces_inverse_single, tonemap_aces_inplace, tonemap_aces_inverse_inplace};
use cra_wasm::correction::{color_correct, HistogramOptions};
use cra_wasm::decode::{
    can_use_cicp, cicp_description, image_to_f32_normalized_rgba, image_to_f32_srgb_255_pixels_rgba,
    is_cicp_linear_srgb, is_cicp_needs_conversion, is_cicp_srgb, is_cicp_unspecified,
    is_profile_srgb_verbose, load_image_from_path, load_image_from_path_auto, load_raw_image,
    transform_cicp_to_linear_srgb_pixels, transform_icc_to_linear_srgb_pixels,
};
use cra_wasm::dither::rgb::DitherMode as CSDitherMode;
use cra_wasm::dither::luminosity::colorspace_aware_dither_gray_with_mode;
use cra_wasm::dither::basic::dither_with_mode_bits;
use cra_wasm::dither::common::{DitherMode, OutputTechnique, PerceptualSpace};
use cra_wasm::color::{denormalize_inplace_clamped, linear_to_srgb_inplace};
use cra_wasm::output::{dither_output_rgb, dither_output_rgba, dither_output_la};
use cra_wasm::pixel::{Pixel4, unpremultiply_alpha_inplace};
use cra_wasm::supersample::{tent_expand, tent_contract, supersample_target_dimensions};

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
// Palette Format Support
// ============================================================================

/// Known palette-based formats
#[derive(Debug, Clone, PartialEq)]
enum PaletteFormat {
    /// Web-safe 216-color palette (6×6×6 RGB cube)
    WebSafe,
    /// CGA 5153 monitor palette (16 colors, hardware-accurate voltage normalization)
    Cga5153,
    /// CGA BIOS/EGA canonical palette (16 colors, the "fake" standard palette)
    CgaBios,
    /// CGA Palette 1 (4 colors: black, cyan, magenta, white)
    CgaPalette1,
    /// CGA Palette 1 with 5153 monitor colors (4 colors)
    CgaPalette1_5153,
    /// Custom palette loaded from an input PNG file
    Input(String, Vec<(u8, u8, u8, u8)>),
}

impl PaletteFormat {
    /// Try to parse a format string as a palette format
    fn parse(format: &str) -> Option<Self> {
        match format.to_uppercase().as_str() {
            "PALETTE_WEBSAFE" => Some(PaletteFormat::WebSafe),
            "PALETTE_CGA_5153" => Some(PaletteFormat::Cga5153),
            "PALETTE_CGA_BIOS" => Some(PaletteFormat::CgaBios),
            "PALETTE_CGA_PALETTE1" => Some(PaletteFormat::CgaPalette1),
            "PALETTE_CGA_PALETTE1_5153" => Some(PaletteFormat::CgaPalette1_5153),
            _ => None,
        }
    }

    /// Get display name
    fn name(&self) -> String {
        match self {
            PaletteFormat::WebSafe => "PALETTE_WEBSAFE".to_string(),
            PaletteFormat::Cga5153 => "PALETTE_CGA_5153".to_string(),
            PaletteFormat::CgaBios => "PALETTE_CGA_BIOS".to_string(),
            PaletteFormat::CgaPalette1 => "PALETTE_CGA_PALETTE1".to_string(),
            PaletteFormat::CgaPalette1_5153 => "PALETTE_CGA_PALETTE1_5153".to_string(),
            PaletteFormat::Input(name, _) => name.clone(),
        }
    }

    /// Get number of colors in the palette
    fn color_count(&self) -> usize {
        match self {
            PaletteFormat::WebSafe => 216,
            PaletteFormat::Cga5153 => 16,
            PaletteFormat::CgaBios => 16,
            PaletteFormat::CgaPalette1 => 4,
            PaletteFormat::CgaPalette1_5153 => 4,
            PaletteFormat::Input(_, colors) => colors.len(),
        }
    }

    /// Get the palette colors
    fn colors(&self) -> Vec<(u8, u8, u8, u8)> {
        match self {
            PaletteFormat::WebSafe => generate_websafe_palette(),
            PaletteFormat::Cga5153 => generate_cga_5153_palette(),
            PaletteFormat::CgaBios => generate_cga_bios_palette(),
            PaletteFormat::CgaPalette1 => generate_cga_palette1_palette(),
            PaletteFormat::CgaPalette1_5153 => generate_cga_palette1_5153_palette(),
            PaletteFormat::Input(_, colors) => colors.clone(),
        }
    }
}

/// Generate the web-safe 216-color palette (6×6×6 RGB cube)
/// Values are: 0, 51, 102, 153, 204, 255 (hex: 00, 33, 66, 99, CC, FF)
fn generate_websafe_palette() -> Vec<(u8, u8, u8, u8)> {
    const LEVELS: [u8; 6] = [0, 51, 102, 153, 204, 255];
    let mut colors = Vec::with_capacity(216);
    for &r in &LEVELS {
        for &g in &LEVELS {
            for &b in &LEVELS {
                colors.push((r, g, b, 255)); // All opaque
            }
        }
    }
    colors
}

/// Generate the CGA 5153 monitor palette (16 colors)
/// Hardware-accurate palette based on actual IBM 5153 monitor voltage normalization (0 to 1.30V range)
/// This is what CGA graphics actually looked like on period-accurate hardware.
fn generate_cga_5153_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 00: Black
        (0x00, 0x00, 0xC4, 255), // 01: Blue
        (0x00, 0xC4, 0x00, 255), // 02: Green
        (0x00, 0xC4, 0xC4, 255), // 03: Cyan
        (0xC4, 0x00, 0x00, 255), // 04: Red
        (0xC4, 0x00, 0xC4, 255), // 05: Magenta
        (0xC4, 0x7E, 0x00, 255), // 06: Brown (dark yellow)
        (0xC4, 0xC4, 0xC4, 255), // 07: Light gray
        (0x4E, 0x4E, 0x4E, 255), // 08: Dark gray
        (0x4E, 0x4E, 0xDC, 255), // 09: Light blue
        (0x4E, 0xDC, 0x4E, 255), // 10: Light green
        (0x4E, 0xF3, 0xF3, 255), // 11: Light cyan
        (0xDC, 0x4E, 0x4E, 255), // 12: Light red
        (0xF3, 0x4E, 0xF3, 255), // 13: Light magenta
        (0xF3, 0xF3, 0x4E, 255), // 14: Yellow
        (0xFF, 0xFF, 0xFF, 255), // 15: White
    ]
}

/// Generate the CGA BIOS/EGA canonical palette (16 colors)
/// The "fake" standard palette commonly used in emulators and documentation.
/// Less accurate to actual CGA hardware but widely recognized.
fn generate_cga_bios_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 00: Black
        (0x00, 0x00, 0xAA, 255), // 01: Blue
        (0x00, 0xAA, 0x00, 255), // 02: Green
        (0x00, 0xAA, 0xAA, 255), // 03: Cyan
        (0xAA, 0x00, 0x00, 255), // 04: Red
        (0xAA, 0x00, 0xAA, 255), // 05: Magenta
        (0xAA, 0x55, 0x00, 255), // 06: Brown (dark yellow)
        (0xAA, 0xAA, 0xAA, 255), // 07: Light gray
        (0x55, 0x55, 0x55, 255), // 08: Dark gray
        (0x55, 0x55, 0xFF, 255), // 09: Light blue
        (0x55, 0xFF, 0x55, 255), // 10: Light green
        (0x55, 0xFF, 0xFF, 255), // 11: Light cyan
        (0xFF, 0x55, 0x55, 255), // 12: Light red
        (0xFF, 0x55, 0xFF, 255), // 13: Light magenta
        (0xFF, 0xFF, 0x55, 255), // 14: Yellow
        (0xFF, 0xFF, 0xFF, 255), // 15: White
    ]
}

/// Generate the CGA Palette 1 (4 colors)
/// The cyan/magenta high-intensity palette used in CGA graphics mode 4/5.
fn generate_cga_palette1_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 0: Black
        (0x55, 0xFF, 0xFF, 255), // 1: Cyan (85, 255, 255)
        (0xFF, 0x55, 0xFF, 255), // 2: Magenta (255, 85, 255)
        (0xFF, 0xFF, 0xFF, 255), // 3: White
    ]
}

/// Generate the CGA Palette 1 with 5153 monitor colors (4 colors)
/// Hardware-accurate palette based on actual IBM 5153 monitor measurements
fn generate_cga_palette1_5153_palette() -> Vec<(u8, u8, u8, u8)> {
    vec![
        (0x00, 0x00, 0x00, 255), // 0: Black
        (0x4E, 0xF3, 0xF3, 255), // 1: Light cyan (78, 243, 243)
        (0xF3, 0x4E, 0xF3, 255), // 2: Light magenta (243, 78, 243)
        (0xFF, 0xFF, 0xFF, 255), // 3: White
    ]
}

/// Extract palette from a PNG file
/// Returns (palette_name, colors) or an error message
fn extract_palette_from_png(path: &PathBuf) -> Result<(String, Vec<(u8, u8, u8, u8)>), String> {
    use png::ColorType;
    use std::fs::File;
    use std::io::BufReader;
    use std::collections::HashSet;

    let file = File::open(path)
        .map_err(|e| format!("Failed to open palette file {}: {}", path.display(), e))?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()
        .map_err(|e| format!("Failed to read PNG info from {}: {}", path.display(), e))?;
    let info = reader.info().clone();

    // Generate name from filename
    let name = path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| format!("PALETTE_INPUT_{}", s.to_uppercase().replace(['-', ' ', '.'], "_")))
        .unwrap_or_else(|| "PALETTE_INPUT".to_string());

    // Check if this is an indexed/paletted PNG
    if info.color_type == ColorType::Indexed {
        // Get the palette
        let palette = info.palette.as_ref()
            .ok_or_else(|| format!("PNG {} has no palette", path.display()))?;

        // Get the tRNS (transparency) chunk if present
        let trns = info.trns.as_ref().map(|t| t.as_ref());

        // Convert to our format (R, G, B, A)
        let num_colors = palette.len() / 3;
        if num_colors == 0 || num_colors > 256 {
            return Err(format!("Invalid palette size: {} colors", num_colors));
        }

        let mut colors = Vec::with_capacity(num_colors);
        for i in 0..num_colors {
            let r = palette[i * 3];
            let g = palette[i * 3 + 1];
            let b = palette[i * 3 + 2];
            let a = trns.and_then(|t| t.get(i).copied()).unwrap_or(255);
            colors.push((r, g, b, a));
        }

        return Ok((name, colors));
    }

    // Not an indexed PNG - extract unique colors from pixel data
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame_info = reader.next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG {}: {}", path.display(), e))?;

    let data = &buf[..frame_info.buffer_size()];

    // Extract unique colors in order of appearance
    let mut colors: Vec<(u8, u8, u8, u8)> = Vec::new();
    let mut seen: HashSet<(u8, u8, u8, u8)> = HashSet::new();

    match info.color_type {
        ColorType::Rgba => {
            for chunk in data.chunks_exact(4) {
                let color = (chunk[0], chunk[1], chunk[2], chunk[3]);
                if seen.insert(color) {
                    colors.push(color);
                    if colors.len() > 256 {
                        return Err(format!(
                            "PNG {} has more than 256 unique colors ({} found so far). Cannot use as palette source.",
                            path.display(), colors.len()
                        ));
                    }
                }
            }
        }
        ColorType::Rgb => {
            for chunk in data.chunks_exact(3) {
                let color = (chunk[0], chunk[1], chunk[2], 255);
                if seen.insert(color) {
                    colors.push(color);
                    if colors.len() > 256 {
                        return Err(format!(
                            "PNG {} has more than 256 unique colors ({} found so far). Cannot use as palette source.",
                            path.display(), colors.len()
                        ));
                    }
                }
            }
        }
        ColorType::GrayscaleAlpha => {
            for chunk in data.chunks_exact(2) {
                let color = (chunk[0], chunk[0], chunk[0], chunk[1]);
                if seen.insert(color) {
                    colors.push(color);
                    if colors.len() > 256 {
                        return Err(format!(
                            "PNG {} has more than 256 unique colors ({} found so far). Cannot use as palette source.",
                            path.display(), colors.len()
                        ));
                    }
                }
            }
        }
        ColorType::Grayscale => {
            for &g in data {
                let color = (g, g, g, 255);
                if seen.insert(color) {
                    colors.push(color);
                    if colors.len() > 256 {
                        return Err(format!(
                            "PNG {} has more than 256 unique colors ({} found so far). Cannot use as palette source.",
                            path.display(), colors.len()
                        ));
                    }
                }
            }
        }
        _ => {
            return Err(format!(
                "Unsupported color type {:?} for palette extraction from {}",
                info.color_type, path.display()
            ));
        }
    }

    if colors.is_empty() {
        return Err(format!("PNG {} contains no colors", path.display()));
    }

    Ok((name, colors))
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
/// When has_alpha is true, uses alpha-aware rescaling to prevent transparent pixels
/// from bleeding their color into opaque regions.
/// When force_exact is true, disables automatic uniform scaling detection.
/// When use_supersample is true, the resizer targets (2*target+1) dimensions for tent-volume supersampling.
fn resize_linear(
    pixels: &[Pixel4],
    src_width: u32,
    src_height: u32,
    target_width: Option<u32>,
    target_height: Option<u32>,
    method: cra_wasm::rescale::RescaleMethod,
    has_alpha: bool,
    force_exact: bool,
    tent_mode: cra_wasm::rescale::TentMode,
    verbose: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Result<(Vec<Pixel4>, u32, u32), String> {
    use cra_wasm::rescale::{calculate_target_dimensions_exact, rescale_with_progress_tent, rescale_with_alpha_progress_tent, ScaleMode, TentMode};

    let tw = target_width.map(|w| w as usize);
    let th = target_height.map(|h| h as usize);
    let (base_dst_width, base_dst_height) = calculate_target_dimensions_exact(
        src_width as usize,
        src_height as usize,
        tw,
        th,
        force_exact,
    );

    // When in SampleToSample mode (tent-volume), target dimensions are (2*requested+1) so contraction gives requested size
    // When in Prescale mode, target dimensions are the final requested size (contract integrated into rescale)
    let (dst_width, dst_height) = if tent_mode == TentMode::SampleToSample {
        supersample_target_dimensions(base_dst_width, base_dst_height)
    } else {
        (base_dst_width, base_dst_height)
    };

    if dst_width == src_width as usize && dst_height == src_height as usize {
        return Ok((pixels.to_vec(), src_width, src_height));
    }

    if verbose {
        let method_name = match method {
            cra_wasm::rescale::RescaleMethod::Bilinear => "Bilinear",
            cra_wasm::rescale::RescaleMethod::Mitchell => "Mitchell",
            cra_wasm::rescale::RescaleMethod::CatmullRom => "Catmull-Rom",
            cra_wasm::rescale::RescaleMethod::Lanczos2 => "Lanczos2",
            cra_wasm::rescale::RescaleMethod::Lanczos3 => "Lanczos3",
            cra_wasm::rescale::RescaleMethod::Sinc => "Sinc (full extent)",
            cra_wasm::rescale::RescaleMethod::Lanczos3Scatter => "Lanczos3 Scatter",
            cra_wasm::rescale::RescaleMethod::SincScatter => "Sinc Scatter (full extent)",
            cra_wasm::rescale::RescaleMethod::Box => "Box (area average)",
            cra_wasm::rescale::RescaleMethod::EWASincLanczos2 => "EWA Sinc-Lanczos2",
            cra_wasm::rescale::RescaleMethod::EWASincLanczos3 => "EWA Sinc-Lanczos3",
            cra_wasm::rescale::RescaleMethod::EWALanczos2 => "EWA Lanczos2 (jinc)",
            cra_wasm::rescale::RescaleMethod::EWALanczos3 => "EWA Lanczos3 (jinc)",
            cra_wasm::rescale::RescaleMethod::EWALanczos3Sharp => "EWA Lanczos3 Sharp (Robidoux)",
            cra_wasm::rescale::RescaleMethod::EWALanczos4Sharpest => "EWA Lanczos4 Sharpest (Robidoux)",
            cra_wasm::rescale::RescaleMethod::EWAMitchell => "EWA Mitchell",
            cra_wasm::rescale::RescaleMethod::EWACatmullRom => "EWA Catmull-Rom",
            cra_wasm::rescale::RescaleMethod::Jinc => "Jinc (full extent)",
            cra_wasm::rescale::RescaleMethod::StochasticJinc => "Stochastic Jinc",
            cra_wasm::rescale::RescaleMethod::StochasticJincScatter => "Stochastic Jinc Scatter",
            cra_wasm::rescale::RescaleMethod::StochasticJincScatterNormalized => "Stochastic Jinc Scatter (normalized)",
            cra_wasm::rescale::RescaleMethod::BilinearIterative => "Bilinear Iterative",
            cra_wasm::rescale::RescaleMethod::HybridLanczos3 => "Hybrid Lanczos3 (sep→EWA 2x)",
        };
        let alpha_note = if has_alpha { " (alpha-aware)" } else { "" };
        let ss_note = match tent_mode {
            TentMode::SampleToSample => " (supersample target)",
            TentMode::Prescale => " (prescale)",
            TentMode::Off => "",
        };
        eprintln!(
            "Resizing in linear RGB ({}{}{}): {}x{} -> {}x{}",
            method_name, alpha_note, ss_note, src_width, src_height, dst_width, dst_height
        );
    }

    // Use alpha-aware rescaling when image has alpha channel to prevent
    // transparent pixels from bleeding their color into opaque regions
    let dst_pixels = if has_alpha {
        rescale_with_alpha_progress_tent(
            pixels,
            src_width as usize, src_height as usize,
            dst_width, dst_height,
            method,
            ScaleMode::Independent,
            tent_mode,
            progress,
        )
    } else {
        rescale_with_progress_tent(
            pixels,
            src_width as usize, src_height as usize,
            dst_width, dst_height,
            method,
            ScaleMode::Independent,
            tent_mode,
            progress,
        )
    };

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
    alpha_mode: Option<CSDitherMode>,
    overshoot_penalty: bool,
) -> OutputTechnique {
    if colorspace_aware {
        OutputTechnique::ColorspaceAware { mode, space, alpha_mode, overshoot_penalty }
    } else {
        OutputTechnique::PerChannel { mode, alpha_mode }
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
    colorspace_aware: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    if colorspace_aware {
        colorspace_aware_dither_gray_with_mode(gray, width, height, bits, space, mode, seed, progress)
    } else {
        // Use basic per-channel dithering (supports Zhou-Fang threshold modulation)
        let basic_mode = match mode {
            CSDitherMode::Standard => DitherMode::Standard,
            CSDitherMode::Serpentine => DitherMode::Serpentine,
            CSDitherMode::JarvisStandard => DitherMode::JarvisStandard,
            CSDitherMode::JarvisSerpentine => DitherMode::JarvisSerpentine,
            CSDitherMode::MixedStandard => DitherMode::MixedStandard,
            CSDitherMode::MixedSerpentine => DitherMode::MixedSerpentine,
            CSDitherMode::MixedRandom => DitherMode::MixedRandom,
            CSDitherMode::OstromoukhovStandard => DitherMode::OstromoukhovStandard,
            CSDitherMode::OstromoukhovSerpentine => DitherMode::OstromoukhovSerpentine,
            CSDitherMode::ZhouFangStandard => DitherMode::ZhouFangStandard,
            CSDitherMode::ZhouFangSerpentine => DitherMode::ZhouFangSerpentine,
            CSDitherMode::None => DitherMode::None,
        };
        dither_with_mode_bits(gray, width, height, basic_mode, seed, bits, progress)
    }
}

/// Result of dithering operation
struct DitherResult {
    /// Interleaved output (grayscale, RGB, or RGBA)
    interleaved: Vec<u8>,
    /// True if this is grayscale data
    is_grayscale: bool,
    /// True if this is RGBA data (4 channels)
    has_alpha: bool,
    /// Palette indices (for paletted output modes)
    palette_indices: Option<Vec<u8>>,
    /// Palette colors as RGBA tuples (for paletted output modes)
    palette_colors: Option<Vec<(u8, u8, u8, u8)>>,
}

/// Dither linear RGB pixels to the target RGB format (grayscale handled separately in main)
#[allow(clippy::too_many_arguments)]
fn dither_pixels_rgb(
    pixels: Vec<Pixel4>,
    width: usize,
    height: usize,
    format: &cra_wasm::binary_format::ColorFormat,
    colorspace_aware: bool,
    dither_mode: CSDitherMode,
    alpha_mode: Option<CSDitherMode>,
    colorspace: PerceptualSpace,
    seed: u32,
    has_alpha: bool,
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    let mut linear_pixels = pixels;

    // Convert linear RGB to sRGB 0-255 (alpha already in correct range)
    linear_to_srgb_inplace(&mut linear_pixels);
    denormalize_inplace_clamped(&mut linear_pixels);

    // Dither
    let technique = build_output_technique(colorspace_aware, dither_mode, colorspace, alpha_mode, overshoot_penalty);
    if has_alpha {
        // Use bits_a from format: ARGB formats preserve alpha, RGB formats strip it (bits_a=0)
        let bits_a = format.bits_a;
        let interleaved = dither_output_rgba(
            &linear_pixels,
            width, height,
            format.bits_r, format.bits_g, format.bits_b, bits_a,
            technique,
            seed,
            progress,
        );
        // Output has alpha only if format supports it (bits_a > 0)
        DitherResult { interleaved, is_grayscale: false, has_alpha: bits_a > 0, palette_indices: None, palette_colors: None }
    } else {
        let interleaved = dither_output_rgb(
            &linear_pixels,
            width, height,
            format.bits_r, format.bits_g, format.bits_b,
            technique,
            seed,
            progress,
        );
        DitherResult { interleaved, is_grayscale: false, has_alpha: false, palette_indices: None, palette_colors: None }
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
    alpha_mode: Option<CSDitherMode>,
    colorspace: PerceptualSpace,
    seed: u32,
    has_alpha: bool,
    overshoot_penalty: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    debug_assert!(!format.is_grayscale, "Use linear path for grayscale");

    let technique = build_output_technique(colorspace_aware, dither_mode, colorspace, alpha_mode, overshoot_penalty);
    if has_alpha {
        // Use bits_a from format: ARGB formats preserve alpha, RGB formats strip it (bits_a=0)
        let bits_a = format.bits_a;
        let interleaved = dither_output_rgba(
            &pixels,
            width, height,
            format.bits_r, format.bits_g, format.bits_b, bits_a,
            technique,
            seed,
            progress,
        );
        // Output has alpha only if format supports it (bits_a > 0)
        DitherResult { interleaved, is_grayscale: false, has_alpha: bits_a > 0, palette_indices: None, palette_colors: None }
    } else {
        let interleaved = dither_output_rgb(
            &pixels,
            width, height,
            format.bits_r, format.bits_g, format.bits_b,
            technique,
            seed,
            progress,
        );
        DitherResult { interleaved, is_grayscale: false, has_alpha: false, palette_indices: None, palette_colors: None }
    }
}

/// Dither pixels using palette-based dithering with integrated alpha-RGB distance metric.
/// Takes linear RGB pixels and dithers to a fixed palette.
#[allow(clippy::too_many_arguments)]
fn dither_pixels_paletted(
    pixels: Vec<Pixel4>,
    width: usize,
    height: usize,
    palette_format: PaletteFormat,
    colorspace: PerceptualSpace,
    dither_mode: DitherMode,
    seed: u32,
    use_hull_tracing: bool,
    overshoot_penalty: bool,
    hull_error_decay: f32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    use cra_wasm::color::interleave_rgba_u8;
    use std::collections::HashMap;

    let mut linear_pixels = pixels;

    // Convert linear RGB to sRGB 0-255 (alpha stays linear 0-1, will be scaled)
    linear_to_srgb_inplace(&mut linear_pixels);
    denormalize_inplace_clamped(&mut linear_pixels);

    // Get palette colors (works for all palette types including Input)
    let palette_colors = palette_format.colors();

    // Create the dither palette with precomputed perceptual coordinates
    let palette = DitherPalette::new(&palette_colors, colorspace);

    // Extract channels from Pixel4
    let r_channel: Vec<f32> = linear_pixels.iter().map(|p| p[0]).collect();
    let g_channel: Vec<f32> = linear_pixels.iter().map(|p| p[1]).collect();
    let b_channel: Vec<f32> = linear_pixels.iter().map(|p| p[2]).collect();
    let a_channel: Vec<f32> = linear_pixels.iter().map(|p| p[3]).collect();

    // Perform gamut-mapped paletted dithering
    let (r_out, g_out, b_out, a_out) = paletted_dither_rgba_gamut_mapped(
        &r_channel, &g_channel, &b_channel, &a_channel,
        width, height,
        &palette,
        dither_mode,
        seed,
        use_hull_tracing,
        overshoot_penalty,
        hull_error_decay,
        progress,
    );

    // Build reverse lookup map from RGBA to palette index
    let color_to_index: HashMap<(u8, u8, u8, u8), u8> = palette_colors
        .iter()
        .enumerate()
        .map(|(i, &(r, g, b, a))| ((r, g, b, a), i as u8))
        .collect();

    // Convert output to palette indices
    let pixel_count = width * height;
    let palette_indices: Vec<u8> = (0..pixel_count)
        .map(|i| {
            let key = (r_out[i], g_out[i], b_out[i], a_out[i]);
            *color_to_index.get(&key).unwrap_or(&0)
        })
        .collect();

    // Interleave to RGBA
    let interleaved = interleave_rgba_u8(&r_out, &g_out, &b_out, &a_out);

    DitherResult {
        interleaved,
        is_grayscale: false,
        has_alpha: true, // Palette formats always output RGBA
        palette_indices: Some(palette_indices),
        palette_colors: Some(palette_colors),
    }
}

/// Dither sRGB pixels (0-255 range) using palette-based dithering.
/// Use when no color correction, resize, or grayscale conversion is needed.
#[allow(clippy::too_many_arguments)]
fn dither_pixels_srgb_paletted(
    pixels: Vec<Pixel4>,
    width: usize,
    height: usize,
    palette_format: PaletteFormat,
    colorspace: PerceptualSpace,
    dither_mode: DitherMode,
    seed: u32,
    use_hull_tracing: bool,
    overshoot_penalty: bool,
    hull_error_decay: f32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    use cra_wasm::color::interleave_rgba_u8;
    use std::collections::HashMap;

    // Get palette colors (works for all palette types including Input)
    let palette_colors = palette_format.colors();

    // Create the dither palette with precomputed perceptual coordinates
    let palette = DitherPalette::new(&palette_colors, colorspace);

    // Extract channels from Pixel4 (already sRGB 0-255)
    let r_channel: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
    let g_channel: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
    let b_channel: Vec<f32> = pixels.iter().map(|p| p[2]).collect();
    let a_channel: Vec<f32> = pixels.iter().map(|p| p[3]).collect();

    // Perform gamut-mapped paletted dithering
    let (r_out, g_out, b_out, a_out) = paletted_dither_rgba_gamut_mapped(
        &r_channel, &g_channel, &b_channel, &a_channel,
        width, height,
        &palette,
        dither_mode,
        seed,
        use_hull_tracing,
        overshoot_penalty,
        hull_error_decay,
        progress,
    );

    // Build reverse lookup map from RGBA to palette index
    let color_to_index: HashMap<(u8, u8, u8, u8), u8> = palette_colors
        .iter()
        .enumerate()
        .map(|(i, &(r, g, b, a))| ((r, g, b, a), i as u8))
        .collect();

    // Convert output to palette indices
    let pixel_count = width * height;
    let palette_indices: Vec<u8> = (0..pixel_count)
        .map(|i| {
            let key = (r_out[i], g_out[i], b_out[i], a_out[i]);
            *color_to_index.get(&key).unwrap_or(&0)
        })
        .collect();

    // Interleave to RGBA
    let interleaved = interleave_rgba_u8(&r_out, &g_out, &b_out, &a_out);

    DitherResult {
        interleaved,
        is_grayscale: false,
        has_alpha: true, // Palette formats always output RGBA
        palette_indices: Some(palette_indices),
        palette_colors: Some(palette_colors),
    }
}

/// Encode dithered pixels to binary format (always row-aligned)
/// Each row is byte-aligned at minimum, with optional stride padding for hardware alignment.
/// There is no use case for "packed" encoding where rows continue from the previous row
/// without byte alignment - such output would be impractical for embedded/hardware use.
fn encode_binary(
    result: &DitherResult,
    format: &ColorFormat,
    width: usize,
    height: usize,
    stride: usize,
    fill: StrideFill,
) -> Vec<u8> {
    if result.is_grayscale && result.has_alpha {
        // LA format: grayscale with alpha (Alpha in MSB, Luminosity in LSB)
        encode_la_row_aligned_stride(
            &result.interleaved, width, height, format.bits_r, format.bits_a, stride, fill,
        )
    } else if result.is_grayscale {
        // Pure grayscale format
        encode_gray_row_aligned_stride(&result.interleaved, width, height, format.bits_r, stride, fill)
    } else if result.has_alpha {
        // ARGB format (hardware ordering: A in MSB, R, G, B toward LSB)
        encode_argb_row_aligned_stride(
            &result.interleaved, width, height, format.bits_a, format.bits_r, format.bits_g, format.bits_b, stride, fill,
        )
    } else {
        // RGB format
        encode_rgb_row_aligned_stride(
            &result.interleaved, width, height, format.bits_r, format.bits_g, format.bits_b, stride, fill,
        )
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

fn save_png_grayscale_alpha(path: &PathBuf, data: &[u8], width: u32, height: u32) -> Result<(), String> {
    let img: ImageBuffer<LumaA<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, data.to_vec())
            .ok_or_else(|| "Failed to create grayscale+alpha image buffer".to_string())?;

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

/// Effective safetensors settings (resolved from args + context)
struct SafetensorsMetadata {
    format: SafetensorsFormat,
    transfer: SfiTransfer,
    has_alpha: bool,
    /// Dither method for FP16/BF16 quantization (None for FP32 - no dithering applied)
    dither: Option<DitherMethod>,
    /// Perceptual space for dithering distance metric (None for FP32 - no dithering applied)
    distance_space: Option<ColorSpace>,
}

fn write_metadata(
    path: &PathBuf,
    args: &Args,
    format: &ColorFormat,
    palette_format: Option<&PaletteFormat>,
    histogram: Histogram,
    output_colorspace: ColorSpace,
    width: u32,
    height: u32,
    outputs: &[(String, PathBuf, usize)],
    safetensors_meta: Option<&SafetensorsMetadata>,
    has_integer_output: bool,
    png_palettized: bool,
) -> Result<(), String> {
    // Get name from output filename (without extension)
    // Prefer raw output path, then first output, then input, then fallback to "image"
    let name = args.output_raw.as_ref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .or_else(|| outputs.first().and_then(|(_, p, _)| p.file_stem()).and_then(|s| s.to_str()))
        .or_else(|| args.input.file_stem().and_then(|s| s.to_str()))
        .unwrap_or("image");

    let mut json = String::new();
    json.push_str("{\n");

    // Basic fields
    json.push_str(&format!("  \"name\": \"{}\",\n", name));
    json.push_str("  \"type\": \"bitmap\",\n");
    json.push_str(&format!("  \"width\": {},\n", width));
    json.push_str(&format!("  \"height\": {},\n", height));

    // Integer format fields (only when integer output is present: PNG, raw, or channel outputs)
    if has_integer_output {
        // For palette formats, output "PALETTED8" as the format; otherwise use the format name
        let is_palette_format = palette_format.is_some();
        if let Some(ref pf) = palette_format {
            // LUT file size = 4 bytes (ARGB8888) per color
            let lut_file_size = pf.color_count() * 4;
            json.push_str(&format!("  \"lut_file_size\": {},\n", lut_file_size));
            json.push_str("  \"format\": \"PALETTED8\",\n");
            json.push_str("  \"bits_per_pixel\": 8,\n");
            // Also include the actual palette name for reference
            json.push_str(&format!("  \"palette\": \"{}\",\n", pf.name()));
        } else {
            json.push_str(&format!("  \"format\": \"{}\",\n", format.name));
            json.push_str(&format!("  \"bits_per_pixel\": {},\n", format.total_bits));
            if !format.is_grayscale {
                json.push_str(&format!("  \"bits_r\": {},\n", format.bits_r));
                json.push_str(&format!("  \"bits_g\": {},\n", format.bits_g));
                json.push_str(&format!("  \"bits_b\": {},\n", format.bits_b));
                if format.has_alpha {
                    json.push_str(&format!("  \"bits_a\": {},\n", format.bits_a));
                }
            } else {
                json.push_str(&format!("  \"bits_l\": {},\n", format.bits_r));
            }
        }
        json.push_str(&format!("  \"output_dither\": \"{:?}\",\n", args.output_dither));
        if let Some(alpha_dither) = args.output_alpha_dither {
            json.push_str(&format!("  \"output_alpha_dither\": \"{:?}\",\n", alpha_dither));
        }
        json.push_str(&format!("  \"output_distance_space\": \"{:?}\",\n", output_colorspace));

        // Palettized output (only when PNG output is present)
        if args.output.is_some() {
            json.push_str(&format!("  \"output_palettized\": {},\n", png_palettized));
        }

        // Raw-file-specific fields (only when raw output is present)
        if args.output_raw.is_some() {
            // Calculate stride and total_size for raw output
            // Stride = bytes per row (respects --stride alignment)
            // For palette formats, use 8 bits per pixel (palette indices)
            let bits_per_pixel = if is_palette_format { 8 } else { format.total_bits as usize };
            let packed_row_bits = width as usize * bits_per_pixel;
            let packed_row_bytes = (packed_row_bits + 7) / 8;
            let row_stride = if args.stride > 1 {
                // Align to stride
                ((packed_row_bytes + args.stride - 1) / args.stride) * args.stride
            } else {
                packed_row_bytes
            };
            let total_size = row_stride * height as usize;

            json.push_str(&format!("  \"stride_alignment\": {},\n", args.stride));
            json.push_str(&format!("  \"stride_fill\": \"{:?}\",\n", args.stride_fill));
            json.push_str("  \"compressed\": 0,\n");
            json.push_str(&format!("  \"stride\": {},\n", row_stride));
            json.push_str(&format!("  \"total_size\": {},\n", total_size));
        }
    }

    // Safetensors-specific fields (only when safetensors output is present)
    if let Some(sfi) = safetensors_meta {
        let transfer_str = match sfi.transfer {
            SfiTransfer::Linear => "linear",
            SfiTransfer::Srgb => "srgb",
            SfiTransfer::Unspecified => "unspecified",
        };
        json.push_str(&format!("  \"safetensors_format\": \"{:?}\",\n", sfi.format));
        json.push_str(&format!("  \"safetensors_transfer\": \"{}\",\n", transfer_str));
        json.push_str(&format!("  \"safetensors_has_alpha\": {},\n", sfi.has_alpha));
        // Dither fields only for FP16/BF16 (FP32 doesn't apply dithering)
        if let Some(dither) = sfi.dither {
            json.push_str(&format!("  \"safetensors_dither\": \"{:?}\",\n", dither));
        }
        if let Some(space) = sfi.distance_space {
            json.push_str(&format!("  \"safetensors_distance_space\": \"{:?}\",\n", space));
        }
    }

    // Common fields
    json.push_str(&format!("  \"histogram\": \"{:?}\",\n", histogram));
    json.push_str(&format!("  \"input\": \"{}\",\n", args.input.display()));
    if let Some(ref ref_path) = args.r#ref {
        json.push_str(&format!("  \"reference\": \"{}\",\n", ref_path.display()));
    }
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
// Safetensors Output Helper
// ============================================================================

/// Options for safetensors dithering
struct SafetensorsDitherOptions {
    dither_mode: DitherMode,
    perceptual_space: PerceptualSpace,
    seed: u32,
}

/// Prepare and write safetensors output with common metadata handling
/// Returns the SafetensorsMetadata for later use
#[allow(clippy::too_many_arguments)]
fn prepare_and_write_safetensors(
    sfi_path: &PathBuf,
    output_pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    args: &Args,
    default_transfer: SfiTransfer,
) -> Result<SafetensorsMetadata, String> {
    let transfer = match args.safetensors_transfer {
        SafetensorsTransfer::Auto => default_transfer,
        SafetensorsTransfer::Linear => SfiTransfer::Linear,
        SafetensorsTransfer::Srgb => SfiTransfer::Srgb,
    };
    let (dither, distance_space) = if matches!(args.safetensors_format, SafetensorsFormat::Fp32) {
        (None, None)
    } else {
        (Some(args.safetensors_dither), Some(args.safetensors_distance_space))
    };
    let meta = SafetensorsMetadata {
        format: args.safetensors_format,
        transfer,
        has_alpha: include_alpha,
        dither,
        distance_space,
    };
    let dither_opts = SafetensorsDitherOptions {
        dither_mode: args.safetensors_dither.to_dither_mode(),
        perceptual_space: args.safetensors_distance_space.to_perceptual_space(),
        seed: args.seed,
    };
    write_safetensors_output(
        sfi_path,
        output_pixels,
        width,
        height,
        include_alpha,
        transfer,
        args.safetensors_format,
        Some(&dither_opts),
        args.verbose,
    )?;
    Ok(meta)
}

/// Convert linear RGB pixels to safetensors output format (with optional sRGB conversion)
fn pixels_to_safetensors_format(pixels: &[Pixel4], transfer: SfiTransfer) -> Vec<Pixel4> {
    if transfer == SfiTransfer::Srgb {
        pixels.iter()
            .map(|p| Pixel4::new(
                linear_to_srgb_single(p[0]),
                linear_to_srgb_single(p[1]),
                linear_to_srgb_single(p[2]),
                p[3],
            ))
            .collect()
    } else {
        pixels.to_vec()
    }
}

/// Convert linear grayscale to Pixel4 format (R=G=B=L) for safetensors
fn grayscale_to_safetensors_format(gray: &[f32], alpha: Option<&[f32]>, transfer: SfiTransfer) -> Vec<Pixel4> {
    let alpha_iter = alpha.map(|a| a.iter()).into_iter().flatten().chain(std::iter::repeat(&1.0f32));
    if transfer == SfiTransfer::Srgb {
        gray.iter().zip(alpha_iter)
            .map(|(&l, &a)| {
                let srgb_l = linear_to_srgb_single(l);
                Pixel4::new(srgb_l, srgb_l, srgb_l, a)
            })
            .collect()
    } else {
        gray.iter().zip(alpha_iter)
            .map(|(&l, &a)| Pixel4::new(l, l, l, a))
            .collect()
    }
}

/// Convert sRGB 0-255 pixels to safetensors format (normalized 0-1)
fn srgb255_to_safetensors_format(pixels: &[Pixel4], transfer: SfiTransfer) -> Vec<Pixel4> {
    if transfer == SfiTransfer::Linear {
        // sRGB 0-255 → normalize → sRGB → linear
        pixels.iter()
            .map(|p| Pixel4::new(
                srgb_to_linear_single(p[0] / 255.0),
                srgb_to_linear_single(p[1] / 255.0),
                srgb_to_linear_single(p[2] / 255.0),
                p[3] / 255.0,
            ))
            .collect()
    } else {
        // sRGB 0-255 → normalize (stays sRGB)
        pixels.iter()
            .map(|p| Pixel4::new(p[0] / 255.0, p[1] / 255.0, p[2] / 255.0, p[3] / 255.0))
            .collect()
    }
}

/// Write safetensors output file (linear or sRGB FP32/FP16/BF16)
/// For FP16/BF16, applies error diffusion dithering for optimal precision.
#[allow(clippy::too_many_arguments)]
fn write_safetensors_output(
    path: &PathBuf,
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
    format: SafetensorsFormat,
    dither_opts: Option<&SafetensorsDitherOptions>,
    verbose: bool,
) -> Result<usize, String> {
    let transfer_name = match transfer {
        SfiTransfer::Linear => "linear",
        SfiTransfer::Srgb => "sRGB",
        SfiTransfer::Unspecified => "unspecified",
    };

    let format_name = match format {
        SafetensorsFormat::Fp32 => "FP32",
        SafetensorsFormat::Fp16 => "FP16",
        SafetensorsFormat::Bf16 => "BF16",
    };

    if verbose {
        eprintln!(
            "Safetensors output: {} ({}, {}, {})",
            path.display(),
            format_name,
            transfer_name,
            if include_alpha { "RGBA" } else { "RGB" }
        );
    }

    let width_usize = width as usize;
    let height_usize = height as usize;

    // Extract channels from Pixel4 array
    let r_channel: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
    let g_channel: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
    let b_channel: Vec<f32> = pixels.iter().map(|p| p[2]).collect();
    let a_channel: Vec<f32> = pixels.iter().map(|p| p[3]).collect();

    let sfi_data = match format {
        SafetensorsFormat::Fp32 => {
            // FP32 has enough precision - no dithering needed
            write_sfi_f32(pixels, width, height, include_alpha, transfer)
        }
        SafetensorsFormat::Fp16 => {
            // Determine working space based on transfer
            let working_space = match transfer {
                SfiTransfer::Linear => Fp16WorkingSpace::Linear,
                SfiTransfer::Srgb | SfiTransfer::Unspecified => Fp16WorkingSpace::Srgb,
            };

            if let Some(opts) = dither_opts {
                if verbose {
                    eprintln!("  Dithering to FP16 ({:?}, {:?})", opts.dither_mode, opts.perceptual_space);
                }

                if include_alpha {
                    let (r_out, g_out, b_out, a_out) = dither_rgba_f16_with_mode(
                        &r_channel, &g_channel, &b_channel, &a_channel,
                        width_usize, height_usize,
                        opts.perceptual_space,
                        working_space,
                        opts.dither_mode,
                        opts.seed,
                        None,
                    );
                    write_sfi_f16_channels(&r_out, &g_out, &b_out, Some(&a_out), width, height, transfer)
                } else {
                    let (r_out, g_out, b_out) = dither_rgb_f16_with_mode(
                        &r_channel, &g_channel, &b_channel,
                        width_usize, height_usize,
                        opts.perceptual_space,
                        working_space,
                        opts.dither_mode,
                        opts.seed,
                        None,
                    );
                    write_sfi_f16_channels(&r_out, &g_out, &b_out, None, width, height, transfer)
                }
            } else {
                // No dithering - use round-to-nearest
                if verbose {
                    eprintln!("  Note: FP16 output uses round-to-nearest (no dithering)");
                }
                write_sfi_f16(pixels, width, height, include_alpha, transfer)
            }
        }
        SafetensorsFormat::Bf16 => {
            // Determine working space based on transfer
            let working_space = match transfer {
                SfiTransfer::Linear => Bf16WorkingSpace::Linear,
                SfiTransfer::Srgb | SfiTransfer::Unspecified => Bf16WorkingSpace::Srgb,
            };

            if let Some(opts) = dither_opts {
                if verbose {
                    eprintln!("  Dithering to BF16 ({:?}, {:?})", opts.dither_mode, opts.perceptual_space);
                }

                if include_alpha {
                    let (r_out, g_out, b_out, a_out) = dither_rgba_bf16_with_mode(
                        &r_channel, &g_channel, &b_channel, &a_channel,
                        width_usize, height_usize,
                        opts.perceptual_space,
                        working_space,
                        opts.dither_mode,
                        opts.seed,
                        None,
                    );
                    write_sfi_bf16_channels(&r_out, &g_out, &b_out, Some(&a_out), width, height, transfer)
                } else {
                    let (r_out, g_out, b_out) = dither_rgb_bf16_with_mode(
                        &r_channel, &g_channel, &b_channel,
                        width_usize, height_usize,
                        opts.perceptual_space,
                        working_space,
                        opts.dither_mode,
                        opts.seed,
                        None,
                    );
                    write_sfi_bf16_channels(&r_out, &g_out, &b_out, None, width, height, transfer)
                }
            } else {
                // No dithering - use round-to-nearest
                if verbose {
                    eprintln!("  Note: BF16 output uses round-to-nearest (no dithering)");
                }
                write_sfi_bf16(pixels, width, height, include_alpha, transfer)
            }
        }
    };

    let data_len = sfi_data.len();

    let mut file = File::create(path)
        .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
    file.write_all(&sfi_data)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;

    if verbose {
        eprintln!("  Written {} bytes", data_len);
    }

    Ok(data_len)
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), String> {
    let args = Args::parse();

    // Check for input palette file first (takes priority over --format)
    let palette_format = if let Some(ref palette_path) = args.input_palette {
        let (name, colors) = extract_palette_from_png(palette_path)?;
        if args.verbose {
            eprintln!("Loaded palette from {}: {} colors", palette_path.display(), colors.len());
        }
        if args.format.is_some() {
            eprintln!("Warning: --format is ignored when --input-palette is specified");
        }
        Some(PaletteFormat::Input(name, colors))
    } else {
        // Check for palette-based formats (e.g., PALETTE_WEBSAFE)
        args.format.as_ref().and_then(|f| PaletteFormat::parse(f))
    };

    // Parse format string if explicitly provided and not a palette format
    // If user specified a format, parse it now to detect grayscale for needs_linear
    let explicit_format = if palette_format.is_some() {
        None // Palette formats don't use ColorFormat
    } else {
        args.format.as_ref().map(|f| ColorFormat::parse(f)).transpose()?
    };
    // For needs_linear calculation: assume non-grayscale if format not specified (default is RGB/ARGB)
    // Palette formats are always RGB (not grayscale)
    let is_grayscale_format = explicit_format.as_ref().map(|f| f.is_grayscale).unwrap_or(false);

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

    let histogram_options = HistogramOptions {
        mode: args.histogram_mode.to_lib_mode(),
        dither_mode: args.histogram_dither.to_dither_mode(),
        colorspace_aware: !args.no_colorspace_aware_histogram,
        colorspace_aware_space: args.histogram_distance_space.to_perceptual_space(),
    };
    let output_dither_mode = args.output_dither.to_cs_dither_mode();
    let output_alpha_mode = args.output_alpha_dither.map(|m| m.to_cs_dither_mode());

    // Warn about Zhou-Fang limitations
    let is_zhou_fang = matches!(
        output_dither_mode,
        CSDitherMode::ZhouFangStandard | CSDitherMode::ZhouFangSerpentine
    );
    if is_zhou_fang && !args.no_colorspace_aware_output {
        eprintln!("Warning: Zhou-Fang threshold modulation only works with --no-colorspace-aware-output.");
        eprintln!("         With colorspace-aware dithering, Zhou-Fang falls back to Ostromoukhov (variable coefficients only).");
    }

    // Build the correction method (None if histogram is None)
    let correction_method = build_correction_method(
        histogram,
        args.keep_luminosity,
        args.tiled_luminosity,
        args.perceptual,
    );

    // Check if at least one output is specified
    let has_channel_output = args.output_raw_r.is_some()
        || args.output_raw_g.is_some()
        || args.output_raw_b.is_some()
        || args.output_raw_a.is_some();
    // Integer output = PNG, raw binary, or channel outputs (requires dithering)
    let has_integer_output = args.output.is_some()
        || args.output_raw.is_some()
        || has_channel_output;
    if !has_integer_output
        && args.output_meta.is_none()
        && args.output_safetensors.is_none()
    {
        return Err(
            "No output specified. Use --output, --output-raw, --output-raw-r/g/b/a, --output-meta, or --output-safetensors"
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
    // Uses auto-detection to support SFI (safetensors) format
    // Or raw binary format if --input-metadata is provided
    if args.verbose {
        eprintln!("Loading: {}", args.input.display());
    }
    let decoded_input = if let Some(ref metadata_json) = args.input_metadata {
        // Parse metadata and load as raw binary file
        let raw_metadata = RawImageMetadata::from_json(metadata_json)
            .map_err(|e| format!("Invalid --input-metadata: {}", e))?;
        if args.verbose {
            eprintln!("  Raw format: {} ({}x{}, stride={})",
                raw_metadata.format, raw_metadata.width, raw_metadata.height, raw_metadata.stride);
        }
        load_raw_image(&args.input, &raw_metadata)?
    } else {
        load_image_from_path_auto(&args.input)?
    };

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

    // Detect if input has alpha channel (before conversion)
    let input_image_has_alpha = matches!(
        input_img.color(),
        ColorType::La8 | ColorType::Rgba8 | ColorType::La16 | ColorType::Rgba16 | ColorType::Rgba32F
    );

    // Finalize format: use explicit format if provided, otherwise apply alpha-aware default
    // For palette formats, we create a synthetic RGBA8888 format for output handling
    let format = if let Some(ref pf) = palette_format {
        if args.verbose {
            eprintln!("  Palette format: {} ({} colors)", pf.name(), pf.color_count());
        }
        // Palette formats output as RGBA8888
        ColorFormat::parse("ARGB8888").expect("ARGB8888 should always parse")
    } else {
        match explicit_format {
            Some(f) => f,
            None => {
                // Default: ARGB8888 if input has alpha, RGB888 otherwise
                let default_format = if input_image_has_alpha { "ARGB8888" } else { "RGB888" };
                if args.verbose {
                    eprintln!("  Format: {} (default, based on input alpha)", default_format);
                }
                ColorFormat::parse(default_format).expect("Default format should always parse")
            }
        }
    };

    // Determine output colorspace for dithering
    // Palette formats always use colorspace-aware dithering (required for distance metric)
    let output_colorspace = args.output_distance_space.unwrap_or(if format.is_grayscale {
        ColorSpace::LabCie94
    } else {
        ColorSpace::Oklab
    });

    // Now perform format-dependent validation
    if args.verbose && args.format.is_some() && palette_format.is_none() {
        eprintln!("Format: {} ({} bits/pixel)", format.name, format.total_bits);
        if format.is_grayscale {
            eprintln!("  Grayscale: {} bits", format.bits_r);
        } else if format.has_alpha {
            eprintln!(
                "  ARGB: {}+{}+{}+{} bits",
                format.bits_a, format.bits_r, format.bits_g, format.bits_b
            );
        } else {
            eprintln!(
                "  RGB: {}+{}+{} bits",
                format.bits_r, format.bits_g, format.bits_b
            );
        }
    }

    // Palette format validation
    if palette_format.is_some() {
        // Palette formats don't support separate channel outputs
        if args.output_raw_r.is_some() || args.output_raw_g.is_some() || args.output_raw_b.is_some() || args.output_raw_a.is_some() {
            return Err("Palette formats do not support separate channel outputs. Use --output-raw for palette indices.".to_string());
        }
        // Warn about ignored options
        if args.no_colorspace_aware_output {
            eprintln!("Warning: --no-colorspace-aware-output is ignored for palette formats (always uses colorspace-aware dithering)");
        }
        if args.output_alpha_dither.is_some() {
            eprintln!("Warning: --output-alpha-dither is ignored for palette formats (alpha is integrated into main dithering)");
        }
        if is_zhou_fang {
            eprintln!("Warning: Zhou-Fang falls back to Ostromoukhov for palette formats (threshold modulation not applicable).");
        }
    } else {
        // --output-raw-palette is only valid for palette formats
        if args.output_raw_palette.is_some() {
            return Err("--output-raw-palette is only valid for palette output formats (e.g., --format PALETTE_WEBSAFE)".to_string());
        }
    }

    // Check binary output compatibility (non-palette formats)
    if palette_format.is_none() && args.output_raw.is_some() && !format.supports_binary() {
        return Err(format!(
            "Format {} ({} bits) does not support binary output. Binary output requires 1, 2, 4, 8, 16, 18 (RGB666), 24, or 32 bits per pixel.",
            format.name, format.total_bits
        ));
    }

    // Check channel output compatibility (requires RGB/ARGB format, not grayscale)
    if (args.output_raw_r.is_some() || args.output_raw_g.is_some() || args.output_raw_b.is_some()) && format.is_grayscale {
        return Err(
            "Separate channel outputs (--output-raw-r/g/b) require RGB/ARGB format, not grayscale"
                .to_string(),
        );
    }

    // Check alpha channel output compatibility (requires ARGB format)
    if args.output_raw_a.is_some() && !format.has_alpha {
        return Err(
            "Alpha channel output (--output-raw-a) requires ARGB format (e.g., ARGB8888, ARGB1555)"
                .to_string(),
        );
    }

    if args.verbose && needs_unpremultiply {
        eprintln!("  Input has premultiplied alpha (will un-premultiply)");
    }

    if args.verbose {
        eprintln!("Histogram: {:?}", histogram);
        if needs_reference {
            eprintln!("Histogram mode: {:?}", args.histogram_mode);
        }
        eprintln!("Output dither: {:?}", args.output_dither);
        if let Some(alpha_dither) = args.output_alpha_dither {
            if palette_format.is_none() {
                eprintln!("Output alpha dither: {:?}", alpha_dither);
            }
        }
        eprintln!(
            "Output distance space: {:?}{}",
            output_colorspace,
            if args.output_distance_space.is_none() { " (default)" } else { "" }
        );
        eprintln!("Seed: {}", args.seed);
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

    let needs_tonemapping = args.tonemapping.is_some() || args.input_tonemapping.is_some();
    let needs_exposure = args.exposure.is_some();
    let needs_linear = needs_reference || needs_resize || is_grayscale_format
        || needs_profile_processing || needs_unpremultiply || needs_tonemapping || needs_exposure;

    // Track effective safetensors settings for metadata
    let mut safetensors_meta: Option<SafetensorsMetadata> = None;

    // Process image based on whether linear space is needed
    let (dither_result, width, height) = if needs_linear {
        // Linear RGB path: load -> resize -> color correct -> safetensors -> dither
        if args.verbose {
            eprintln!("Processing in linear RGB space...");
        }

        let (input_pixels, src_width, src_height, original_has_alpha) = convert_to_linear(&input_img, &input_icc, &input_cicp, args.input_profile, needs_unpremultiply, args.verbose)?;

        // Determine supersampling mode
        let (tent_mode, needs_expansion) = match args.supersample {
            Supersample::TentVolume if needs_resize => (cra_wasm::rescale::TentMode::SampleToSample, true),
            Supersample::TentVolumePrescale => (cra_wasm::rescale::TentMode::Prescale, true),
            _ => (cra_wasm::rescale::TentMode::Off, false),
        };

        // Apply tent-space expansion before input tonemapping (if enabled)
        let (input_pixels, expanded_width, expanded_height) = if needs_expansion {
            if args.verbose {
                eprintln!("Tent-space supersampling (volume): expanding {}x{} -> {}x{}",
                    src_width, src_height, src_width * 2 + 1, src_height * 2 + 1);
            }
            let (expanded, w, h) = tent_expand(&input_pixels, src_width as usize, src_height as usize);
            (expanded, w as u32, h as u32)
        } else {
            (input_pixels, src_width, src_height)
        };

        // Apply input tonemapping in tent-space (if specified)
        let input_pixels = if let Some(tm) = args.input_tonemapping {
            let mut p = input_pixels;
            match tm {
                Tonemapping::Aces => tonemap_aces_inplace(&mut p),
                Tonemapping::AcesInverse => tonemap_aces_inverse_inplace(&mut p),
            }
            p
        } else {
            input_pixels
        };

        // Histogram processing doesn't support alpha yet - discard alpha when using reference
        let input_has_alpha = if needs_reference && original_has_alpha {
            if args.verbose {
                eprintln!("  Note: Alpha channel discarded (histogram processing does not support alpha)");
            }
            false
        } else {
            original_has_alpha
        };

        // Resize in linear RGB space (use alpha-aware rescaling if image has alpha)
        let effective_method = args.scale_method.to_rescale_method();
        let mut resize_progress = |p: f32| print_progress("Resize", p);
        let (input_pixels, width, height) = resize_linear(
            &input_pixels,
            expanded_width, expanded_height,
            args.width, args.height,
            effective_method,
            original_has_alpha,
            args.non_uniform,
            tent_mode,
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

        // Process differently based on output format (grayscale vs RGB)
        let result = if is_grayscale_format {
            // GRAYSCALE PATH: Convert to grayscale, then tonemapping, then tent_contract, then safetensors, then dither

            // Step 1: Convert linear RGB to linear grayscale
            let linear_gray = linear_pixels_to_grayscale(&pixels_to_dither);

            // Step 2: Extract alpha if needed (before we lose access to pixels_to_dither)
            let alpha: Option<Vec<f32>> = if input_has_alpha && format.bits_a > 0 {
                Some(pixels_to_dither.iter().map(|p| p[3]).collect())
            } else {
                None
            };

            // Step 3: Apply exposure (linear multiplier)
            let linear_gray = if let Some(exp) = args.exposure {
                linear_gray.iter().map(|&l| l * exp).collect()
            } else {
                linear_gray
            };

            // Step 4: Apply tonemapping to grayscale
            let linear_gray = if let Some(tm) = args.tonemapping {
                match tm {
                    Tonemapping::Aces => linear_gray.iter().map(|&l| tonemap_aces_single(l)).collect(),
                    Tonemapping::AcesInverse => linear_gray.iter().map(|&l| tonemap_aces_inverse_single(l)).collect(),
                }
            } else {
                linear_gray
            };

            // Step 5: Apply tent-space contraction after tonemapping (only for SampleToSample mode)
            // Prescale mode integrates contraction into rescale, so no explicit contract needed
            let (linear_gray, alpha, width, height) = if tent_mode == cra_wasm::rescale::TentMode::SampleToSample {
                if args.verbose {
                    eprintln!("Tent-space supersampling (volume): contracting {}x{} -> {}x{}",
                        width, height, (width - 1) / 2, (height - 1) / 2);
                }
                // Contract grayscale by reconstructing as single-channel Pixel4
                let gray_pixels: Vec<Pixel4> = linear_gray.iter()
                    .zip(alpha.as_ref().map(|a| a.iter()).into_iter().flatten().chain(std::iter::repeat(&1.0f32)))
                    .map(|(&l, &a)| Pixel4::new(l, l, l, a))
                    .collect();
                let (contracted, w, h) = tent_contract(&gray_pixels, width_usize, height_usize);
                let new_gray: Vec<f32> = contracted.iter().map(|p| p[0]).collect();
                let new_alpha: Option<Vec<f32>> = if alpha.is_some() {
                    Some(contracted.iter().map(|p| p[3]).collect())
                } else {
                    None
                };
                (new_gray, new_alpha, w as u32, h as u32)
            } else {
                (linear_gray, alpha, width, height)
            };
            let width_usize = width as usize;
            let height_usize = height as usize;

            // Step 6: Write safetensors output (grayscale as R=G=B=L)
            if let Some(ref sfi_path) = args.output_safetensors {
                let include_alpha = !args.safetensors_no_alpha && alpha.is_some();
                let transfer = match args.safetensors_transfer {
                    SafetensorsTransfer::Auto => SfiTransfer::Linear,
                    SafetensorsTransfer::Linear => SfiTransfer::Linear,
                    SafetensorsTransfer::Srgb => SfiTransfer::Srgb,
                };
                let output_pixels = grayscale_to_safetensors_format(&linear_gray, alpha.as_deref(), transfer);
                safetensors_meta = Some(prepare_and_write_safetensors(
                    sfi_path, &output_pixels, width, height, include_alpha, &args, SfiTransfer::Linear,
                )?);
            }

            // Step 7: Dither grayscale
            if has_integer_output {
                let mut dither_progress = |p: f32| print_progress("Dither", p);

                // Convert linear to sRGB and denormalize to 0-255
                let srgb_gray: Vec<f32> = linear_gray.iter()
                    .map(|&l| linear_to_srgb_single(l) * 255.0)
                    .collect();

                let result = if let Some(ref alpha) = alpha {
                    // LA format: grayscale with alpha
                    let alpha_255: Vec<f32> = alpha.iter().map(|&a| a * 255.0).collect();
                    let technique = build_output_technique(!args.no_colorspace_aware_output, output_dither_mode, output_colorspace.to_perceptual_space(), output_alpha_mode, !args.no_overshoot_penalty);
                    let interleaved = dither_output_la(
                        &srgb_gray, &alpha_255, width_usize, height_usize, format.bits_r, format.bits_a,
                        technique, args.seed, if args.progress { Some(&mut dither_progress) } else { None },
                    );
                    DitherResult { interleaved, is_grayscale: true, has_alpha: true, palette_indices: None, palette_colors: None }
                } else {
                    // Pure grayscale
                    let interleaved = dither_grayscale(
                        &srgb_gray, width_usize, height_usize, format.bits_r,
                        output_colorspace.to_perceptual_space(), output_dither_mode, args.seed,
                        !args.no_colorspace_aware_output,
                        if args.progress { Some(&mut dither_progress) } else { None },
                    );
                    DitherResult { interleaved, is_grayscale: true, has_alpha: false, palette_indices: None, palette_colors: None }
                };

                if args.progress {
                    clear_progress();
                }
                Some(result)
            } else {
                None
            }
        } else {
            // RGB PATH: Exposure, tonemapping, tent_contract, then safetensors, then dither

            // Step 1: Apply exposure (linear multiplier)
            let pixels_to_dither = if let Some(exp) = args.exposure {
                pixels_to_dither.iter()
                    .map(|p| Pixel4::new(p[0] * exp, p[1] * exp, p[2] * exp, p[3]))
                    .collect()
            } else {
                pixels_to_dither
            };

            // Step 2: Apply tonemapping to RGB
            let pixels_to_dither = if let Some(tm) = args.tonemapping {
                let mut p = pixels_to_dither;
                match tm {
                    Tonemapping::Aces => tonemap_aces_inplace(&mut p),
                    Tonemapping::AcesInverse => tonemap_aces_inverse_inplace(&mut p),
                }
                p
            } else {
                pixels_to_dither
            };

            // Step 3: Apply tent-space contraction after tonemapping (only for SampleToSample mode)
            // Prescale mode integrates contraction into rescale, so no explicit contract needed
            let (pixels_to_dither, width, height) = if tent_mode == cra_wasm::rescale::TentMode::SampleToSample {
                if args.verbose {
                    eprintln!("Tent-space supersampling (volume): contracting {}x{} -> {}x{}",
                        width, height, (width - 1) / 2, (height - 1) / 2);
                }
                let (contracted, w, h) = tent_contract(&pixels_to_dither, width_usize, height_usize);
                (contracted, w as u32, h as u32)
            } else {
                (pixels_to_dither, width, height)
            };
            let width_usize = width as usize;
            let height_usize = height as usize;

            // Step 4: Write safetensors output
            if let Some(ref sfi_path) = args.output_safetensors {
                let include_alpha = !args.safetensors_no_alpha && input_has_alpha;
                let transfer = match args.safetensors_transfer {
                    SafetensorsTransfer::Auto => SfiTransfer::Linear,
                    SafetensorsTransfer::Linear => SfiTransfer::Linear,
                    SafetensorsTransfer::Srgb => SfiTransfer::Srgb,
                };
                let output_pixels = pixels_to_safetensors_format(&pixels_to_dither, transfer);
                safetensors_meta = Some(prepare_and_write_safetensors(
                    sfi_path, &output_pixels, width, height, include_alpha, &args, SfiTransfer::Linear,
                )?);
            }

            // Step 5: Dither RGB (or palette)
            if has_integer_output {
                let mut dither_progress = |p: f32| print_progress("Dither", p);
                let result = if let Some(ref pf) = palette_format {
                    // Palette dithering with integrated alpha-RGB distance
                    dither_pixels_paletted(
                        pixels_to_dither,
                        width_usize,
                        height_usize,
                        pf.clone(),
                        output_colorspace.to_perceptual_space(),
                        args.output_dither.to_dither_mode(),
                        args.seed,
                        !args.no_hull_tracing,
                        !args.no_overshoot_penalty,
                        args.hull_error_decay.unwrap_or(1.0),
                        if args.progress { Some(&mut dither_progress) } else { None },
                    )
                } else {
                    // Standard RGB dithering
                    dither_pixels_rgb(
                        pixels_to_dither,
                        width_usize,
                        height_usize,
                        &format,
                        !args.no_colorspace_aware_output,
                        output_dither_mode,
                        output_alpha_mode,
                        output_colorspace.to_perceptual_space(),
                        args.seed,
                        input_has_alpha,
                        !args.no_overshoot_penalty,
                        if args.progress { Some(&mut dither_progress) } else { None },
                    )
                };
                if args.progress {
                    clear_progress();
                }
                Some(result)
            } else {
                None
            }
        };

        (result, width, height)
    } else {
        // sRGB path: RGB dither-only (no resize, no color correction, no grayscale)
        // Avoids unnecessary sRGB -> linear -> sRGB conversion
        if args.verbose {
            eprintln!("Dithering RGB channels (sRGB path)...");
        }

        let (input_pixels, width, height, input_has_alpha) = convert_to_srgb_255(&input_img, args.verbose);

        // Write safetensors output (sRGB path) - normalize 0-255 to 0-1 range
        // Data is sRGB 0-255, auto resolves to sRGB transfer
        if let Some(ref sfi_path) = args.output_safetensors {
            let include_alpha = !args.safetensors_no_alpha && input_has_alpha;
            let transfer = match args.safetensors_transfer {
                SafetensorsTransfer::Auto => SfiTransfer::Srgb,
                SafetensorsTransfer::Linear => SfiTransfer::Linear,
                SafetensorsTransfer::Srgb => SfiTransfer::Srgb,
            };
            let output_pixels = srgb255_to_safetensors_format(&input_pixels, transfer);
            safetensors_meta = Some(prepare_and_write_safetensors(
                sfi_path, &output_pixels, width, height, include_alpha, &args, SfiTransfer::Srgb,
            )?);
        }

        // Only perform dithering if integer output is needed
        let result = if has_integer_output {
            let mut dither_progress = |p: f32| print_progress("Dither", p);
            let result = if let Some(ref pf) = palette_format {
                // Palette dithering with integrated alpha-RGB distance
                dither_pixels_srgb_paletted(
                    input_pixels,
                    width as usize,
                    height as usize,
                    pf.clone(),
                    output_colorspace.to_perceptual_space(),
                    args.output_dither.to_dither_mode(),
                    args.seed,
                    !args.no_hull_tracing,
                    !args.no_overshoot_penalty,
                    args.hull_error_decay.unwrap_or(1.0),
                    if args.progress { Some(&mut dither_progress) } else { None },
                )
            } else {
                // Standard RGB dithering
                dither_pixels_srgb_rgb(
                    input_pixels,
                    width as usize,
                    height as usize,
                    &format,
                    !args.no_colorspace_aware_output,
                    output_dither_mode,
                    output_alpha_mode,
                    output_colorspace.to_perceptual_space(),
                    args.seed,
                    input_has_alpha,
                    !args.no_overshoot_penalty,
                    if args.progress { Some(&mut dither_progress) } else { None },
                )
            };
            if args.progress {
                clear_progress();
            }
            Some(result)
        } else {
            None
        };

        (result, width, height)
    };

    let width_usize = width as usize;
    let height_usize = height as usize;

    // Track outputs for metadata
    let mut outputs: Vec<(String, PathBuf, usize)> = Vec::new();

    // Write integer outputs (PNG, raw binary, channel outputs) - only if dithering was performed
    if let Some(dither_result) = dither_result {
        // If format requests alpha but input had none, inject opaque alpha channel
        let dither_result = if format.has_alpha && !dither_result.has_alpha && !dither_result.is_grayscale {
            if args.verbose {
                eprintln!("Injecting opaque alpha channel (input had no alpha)");
            }
            // Expand RGB (3 bytes/pixel) to RGBA (4 bytes/pixel) with alpha = 255
            let rgb = &dither_result.interleaved;
            let pixel_count = width_usize * height_usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                rgba.push(rgb[i * 3]);     // R
                rgba.push(rgb[i * 3 + 1]); // G
                rgba.push(rgb[i * 3 + 2]); // B
                rgba.push(255);            // A (fully opaque)
            }
            DitherResult {
                interleaved: rgba,
                is_grayscale: false,
                has_alpha: true,
                palette_indices: dither_result.palette_indices,
                palette_colors: dither_result.palette_colors,
            }
        } else {
            dither_result
        };

        // Write PNG output
        // Use palettized PNG for explicit palette formats or formats ≤8 bits
        if let Some(ref png_path) = args.output {
            // Check if we have explicit palette data (from paletted dithering)
            let has_explicit_palette = dither_result.palette_indices.is_some()
                && dither_result.palette_colors.is_some();
            let use_palettized = supports_palettized_png(&format) && !args.no_palettized_output;

            if args.verbose {
                if has_explicit_palette {
                    eprintln!("Writing PNG (palettized, {} colors): {}",
                        dither_result.palette_colors.as_ref().map(|p| p.len()).unwrap_or(0),
                        png_path.display());
                } else if use_palettized {
                    eprintln!("Writing PNG (palettized): {}", png_path.display());
                } else {
                    eprintln!("Writing PNG: {}", png_path.display());
                }
            }

            if has_explicit_palette {
                // Use explicit palette PNG for true paletted output (CGA, web-safe, etc.)
                let png_data = encode_explicit_palette_png(
                    dither_result.palette_indices.as_ref().unwrap(),
                    width_usize,
                    height_usize,
                    dither_result.palette_colors.as_ref().unwrap(),
                )?;
                let mut file = File::create(png_path)
                    .map_err(|e| format!("Failed to create {}: {}", png_path.display(), e))?;
                file.write_all(&png_data)
                    .map_err(|e| format!("Failed to write {}: {}", png_path.display(), e))?;
            } else if use_palettized {
                // Use palettized PNG for smaller file size (formats ≤8 bits)
                let png_data = encode_palettized_png(
                    &dither_result.interleaved,
                    width_usize,
                    height_usize,
                    &format,
                )?;
                let mut file = File::create(png_path)
                    .map_err(|e| format!("Failed to create {}: {}", png_path.display(), e))?;
                file.write_all(&png_data)
                    .map_err(|e| format!("Failed to write {}: {}", png_path.display(), e))?;
            } else if format.is_grayscale && format.has_alpha {
                // LA format - grayscale with alpha
                save_png_grayscale_alpha(png_path, &dither_result.interleaved, width, height)?;
            } else if format.is_grayscale {
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

        // Write GIF output (paletted formats only)
        if let Some(ref gif_path) = args.output_gif {
            // Check if we have explicit palette data (from paletted dithering)
            let has_explicit_palette = dither_result.palette_indices.is_some()
                && dither_result.palette_colors.is_some();
            let use_palettized_gif = supports_gif(&format);

            if has_explicit_palette {
                if args.verbose {
                    eprintln!("Writing GIF (palettized, {} colors): {}",
                        dither_result.palette_colors.as_ref().map(|p| p.len()).unwrap_or(0),
                        gif_path.display());
                }
                // Use explicit palette GIF for true paletted output (CGA, web-safe, etc.)
                let gif_data = encode_explicit_palette_gif(
                    dither_result.palette_indices.as_ref().unwrap(),
                    width_usize,
                    height_usize,
                    dither_result.palette_colors.as_ref().unwrap(),
                )?;
                let mut file = File::create(gif_path)
                    .map_err(|e| format!("Failed to create {}: {}", gif_path.display(), e))?;
                file.write_all(&gif_data)
                    .map_err(|e| format!("Failed to write {}: {}", gif_path.display(), e))?;
            } else if use_palettized_gif {
                if args.verbose {
                    eprintln!("Writing GIF (palettized): {}", gif_path.display());
                }
                // Use palettized GIF for formats ≤8 bits
                let gif_data = encode_palettized_gif(
                    &dither_result.interleaved,
                    width_usize,
                    height_usize,
                    &format,
                )?;
                let mut file = File::create(gif_path)
                    .map_err(|e| format!("Failed to create {}: {}", gif_path.display(), e))?;
                file.write_all(&gif_data)
                    .map_err(|e| format!("Failed to write {}: {}", gif_path.display(), e))?;
            } else {
                return Err(format!(
                    "GIF output requires paletted format (≤8 bits per pixel). \
                    Format {} has {} bits. Use --format with a lower bit depth or use --input-palette.",
                    format.name, format.total_bits
                ));
            }

            let size = std::fs::metadata(gif_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            outputs.push(("gif".to_string(), gif_path.clone(), size));
        }

        // Write binary output (always row-aligned, respects --stride for additional alignment)
        if let Some(ref bin_path) = args.output_raw {
            let fill = args.stride_fill.to_stride_fill();

            // For palette formats, output palette indices; otherwise output encoded binary
            let bin_data = if let Some(ref indices) = dither_result.palette_indices {
                if args.verbose {
                    if args.stride > 1 {
                        eprintln!(
                            "Writing palette indices (row-aligned, stride={}, fill={:?}): {}",
                            args.stride, args.stride_fill, bin_path.display()
                        );
                    } else {
                        eprintln!("Writing palette indices (row-aligned): {}", bin_path.display());
                    }
                }
                // Encode palette indices with row alignment (8 bits per pixel)
                encode_channel_from_interleaved_row_aligned_stride(
                    indices, width_usize, height_usize, 1, 0, 8, args.stride, fill,
                )
            } else {
                if args.verbose {
                    if args.stride > 1 {
                        eprintln!(
                            "Writing binary (row-aligned, stride={}, fill={:?}): {}",
                            args.stride, args.stride_fill, bin_path.display()
                        );
                    } else {
                        eprintln!("Writing binary (row-aligned): {}", bin_path.display());
                    }
                }
                encode_binary(&dither_result, &format, width_usize, height_usize, args.stride, fill)
            };

            let mut file = File::create(bin_path)
                .map_err(|e| format!("Failed to create {}: {}", bin_path.display(), e))?;
            file.write_all(&bin_data)
                .map_err(|e| format!("Failed to write {}: {}", bin_path.display(), e))?;

            outputs.push(("binary".to_string(), bin_path.clone(), bin_data.len()));
        }

        // Write palette colors as ARGB8888 binary (for palette formats only)
        if let Some(ref palette_path) = args.output_raw_palette {
            if let Some(ref colors) = dither_result.palette_colors {
                if args.verbose {
                    eprintln!("Writing palette ({} colors, ARGB8888): {}", colors.len(), palette_path.display());
                }
                // Write palette as ARGB8888 (4 bytes per color: A, R, G, B)
                let mut palette_data = Vec::with_capacity(colors.len() * 4);
                for &(r, g, b, a) in colors {
                    palette_data.push(a);
                    palette_data.push(r);
                    palette_data.push(g);
                    palette_data.push(b);
                }
                let mut file = File::create(palette_path)
                    .map_err(|e| format!("Failed to create {}: {}", palette_path.display(), e))?;
                file.write_all(&palette_data)
                    .map_err(|e| format!("Failed to write {}: {}", palette_path.display(), e))?;
                outputs.push(("palette".to_string(), palette_path.clone(), palette_data.len()));
            }
        }

        // Write separate channel outputs (R, G, B, A) - encode directly from interleaved data
        let fill = args.stride_fill.to_stride_fill();
        let needs_rgb_channel_output = args.output_raw_r.is_some() || args.output_raw_g.is_some() || args.output_raw_b.is_some();
        let num_channels = if dither_result.has_alpha { 4 } else { 3 };

        if needs_rgb_channel_output && !dither_result.is_grayscale {
            if let Some(ref path) = args.output_raw_r {
                if args.verbose {
                    eprintln!("Writing red channel binary: {}", path.display());
                }
                let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                    &dither_result.interleaved, width_usize, height_usize, num_channels, 0, format.bits_r, args.stride, fill,
                );
                let mut file = File::create(path)
                    .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
                file.write_all(&bin_data)
                    .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
                outputs.push(("binary_r".to_string(), path.clone(), bin_data.len()));
            }

            if let Some(ref path) = args.output_raw_g {
                if args.verbose {
                    eprintln!("Writing green channel binary: {}", path.display());
                }
                let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                    &dither_result.interleaved, width_usize, height_usize, num_channels, 1, format.bits_g, args.stride, fill,
                );
                let mut file = File::create(path)
                    .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
                file.write_all(&bin_data)
                    .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
                outputs.push(("binary_g".to_string(), path.clone(), bin_data.len()));
            }

            if let Some(ref path) = args.output_raw_b {
                if args.verbose {
                    eprintln!("Writing blue channel binary: {}", path.display());
                }
                let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                    &dither_result.interleaved, width_usize, height_usize, num_channels, 2, format.bits_b, args.stride, fill,
                );
                let mut file = File::create(path)
                    .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
                file.write_all(&bin_data)
                    .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
                outputs.push(("binary_b".to_string(), path.clone(), bin_data.len()));
            }
        }

        // Write alpha channel output (requires ARGB format and input with alpha)
        if let Some(ref path) = args.output_raw_a {
            if dither_result.has_alpha {
                if args.verbose {
                    eprintln!("Writing alpha channel binary: {}", path.display());
                }
                let bin_data = encode_channel_from_interleaved_row_aligned_stride(
                    &dither_result.interleaved, width_usize, height_usize, num_channels, 3, format.bits_a, args.stride, fill,
                );
                let mut file = File::create(path)
                    .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
                file.write_all(&bin_data)
                    .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
                outputs.push(("binary_a".to_string(), path.clone(), bin_data.len()));
            } else {
                eprintln!("Warning: --output-raw-a specified but input image has no alpha channel, skipping");
            }
        }
    }

    // Write metadata JSON
    if let Some(ref meta_path) = args.output_meta {
        if args.verbose {
            eprintln!("Writing metadata: {}", meta_path.display());
        }
        // Determine if PNG output used palettized encoding
        // Explicit palette formats (CGA, web-safe) always use palettized PNG
        let png_palettized = args.output.is_some()
            && (palette_format.is_some() || (supports_palettized_png(&format) && !args.no_palettized_output));
        write_metadata(
            meta_path,
            &args,
            &format,
            palette_format.as_ref(),
            histogram,
            output_colorspace,
            width,
            height,
            &outputs,
            safetensors_meta.as_ref(),
            has_integer_output,
            png_palettized,
        )?;
    }

    if args.verbose {
        eprintln!("Done!");
    }

    Ok(())
}
