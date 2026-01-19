//! CRA - Unified Color Correction and Dithering CLI
//!
//! A unified command-line tool for color correction and error diffusion dithering.
//! Pipeline: input sRGB -> linear RGB -> optional resize -> optional processing -> dither to sRGB -> output
//!
//! All processing occurs in linear RGB space for correct color math.

mod args;

use args::*;
use clap::Parser;
use image::{ColorType, DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgba};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use cra_wasm::binary_format::{
    encode_argb_packed, encode_argb_row_aligned_stride,
    encode_channel_from_interleaved_row_aligned_stride, encode_gray_packed,
    encode_gray_row_aligned_stride, encode_la_packed, encode_la_row_aligned_stride,
    encode_rgb_packed, encode_rgb_row_aligned_stride,
    is_valid_stride, ColorFormat, StrideFill, RawImageMetadata,
};
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
use cra_wasm::dither::common::{DitherMode, OutputTechnique, PerceptualSpace};
use cra_wasm::color::{denormalize_inplace_clamped, linear_to_srgb_inplace};
use cra_wasm::output::{dither_output_rgb, dither_output_rgba, dither_output_la};
use cra_wasm::pixel::{Pixel4, unpremultiply_alpha_inplace};

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
fn resize_linear(
    pixels: &[Pixel4],
    src_width: u32,
    src_height: u32,
    target_width: Option<u32>,
    target_height: Option<u32>,
    method: cra_wasm::rescale::RescaleMethod,
    has_alpha: bool,
    force_exact: bool,
    verbose: bool,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Result<(Vec<Pixel4>, u32, u32), String> {
    use cra_wasm::rescale::{calculate_target_dimensions_exact, rescale_with_progress, rescale_with_alpha_progress, ScaleMode};

    let tw = target_width.map(|w| w as usize);
    let th = target_height.map(|h| h as usize);
    let (dst_width, dst_height) = calculate_target_dimensions_exact(
        src_width as usize,
        src_height as usize,
        tw,
        th,
        force_exact,
    );

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
        };
        let alpha_note = if has_alpha { " (alpha-aware)" } else { "" };
        eprintln!(
            "Resizing in linear RGB ({}{}): {}x{} -> {}x{}",
            method_name, alpha_note, src_width, src_height, dst_width, dst_height
        );
    }

    // Use alpha-aware rescaling when image has alpha channel to prevent
    // transparent pixels from bleeding their color into opaque regions
    let dst_pixels = if has_alpha {
        rescale_with_alpha_progress(
            pixels,
            src_width as usize, src_height as usize,
            dst_width, dst_height,
            method,
            ScaleMode::Independent,
            progress,
        )
    } else {
        rescale_with_progress(
            pixels,
            src_width as usize, src_height as usize,
            dst_width, dst_height,
            method,
            ScaleMode::Independent,
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
) -> OutputTechnique {
    if colorspace_aware {
        OutputTechnique::ColorspaceAware { mode, space, alpha_mode }
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
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    let mut linear_pixels = pixels;

    // Convert linear RGB to sRGB 0-255 (alpha already in correct range)
    linear_to_srgb_inplace(&mut linear_pixels);
    denormalize_inplace_clamped(&mut linear_pixels);

    // Dither
    let technique = build_output_technique(colorspace_aware, dither_mode, colorspace, alpha_mode);
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
        DitherResult { interleaved, is_grayscale: false, has_alpha: bits_a > 0 }
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
    progress: Option<&mut dyn FnMut(f32)>,
) -> DitherResult {
    debug_assert!(!format.is_grayscale, "Use linear path for grayscale");

    let technique = build_output_technique(colorspace_aware, dither_mode, colorspace, alpha_mode);
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
        DitherResult { interleaved, is_grayscale: false, has_alpha: bits_a > 0 }
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
    if result.is_grayscale && result.has_alpha {
        // LA format: grayscale with alpha (Alpha in MSB, Luminosity in LSB)
        if row_aligned {
            encode_la_row_aligned_stride(
                &result.interleaved, width, height, format.bits_r, format.bits_a, stride, fill,
            )
        } else {
            encode_la_packed(
                &result.interleaved, width, height, format.bits_r, format.bits_a,
            )
        }
    } else if result.is_grayscale {
        // Pure grayscale format
        if row_aligned {
            encode_gray_row_aligned_stride(&result.interleaved, width, height, format.bits_r, stride, fill)
        } else {
            encode_gray_packed(&result.interleaved, width, height, format.bits_r)
        }
    } else if result.has_alpha {
        // ARGB format (hardware ordering: A in MSB, R, G, B toward LSB)
        if row_aligned {
            encode_argb_row_aligned_stride(
                &result.interleaved, width, height, format.bits_a, format.bits_r, format.bits_g, format.bits_b, stride, fill,
            )
        } else {
            encode_argb_packed(
                &result.interleaved, width, height, format.bits_a, format.bits_r, format.bits_g, format.bits_b,
            )
        }
    } else {
        // RGB format
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
    histogram: Histogram,
    output_colorspace: ColorSpace,
    width: u32,
    height: u32,
    outputs: &[(String, PathBuf, usize)],
    safetensors_meta: Option<&SafetensorsMetadata>,
    has_integer_output: bool,
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
        json.push_str(&format!("  \"output_dither\": \"{:?}\",\n", args.output_dither));
        if let Some(alpha_dither) = args.output_alpha_dither {
            json.push_str(&format!("  \"output_alpha_dither\": \"{:?}\",\n", alpha_dither));
        }
        json.push_str(&format!("  \"output_colorspace\": \"{:?}\",\n", output_colorspace));

        // Raw-file-specific fields (only when raw output is present)
        if args.output_raw.is_some() {
            // Calculate stride and total_size for raw output
            // Stride = bytes per row (respects --stride alignment)
            let bits_per_pixel = format.total_bits as usize;
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

    // Parse format string if explicitly provided; otherwise defer to apply alpha-aware default later
    // If user specified a format, parse it now to detect grayscale for needs_linear
    let explicit_format = args.format.as_ref().map(|f| ColorFormat::parse(f)).transpose()?;
    // For needs_linear calculation: assume non-grayscale if format not specified (default is RGB/ARGB)
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
    let format = match explicit_format {
        Some(f) => f,
        None => {
            // Default: ARGB8888 if input has alpha, RGB888 otherwise
            let default_format = if input_image_has_alpha { "ARGB8888" } else { "RGB888" };
            if args.verbose {
                eprintln!("  Format: {} (default, based on input alpha)", default_format);
            }
            ColorFormat::parse(default_format).expect("Default format should always parse")
        }
    };

    // Determine output colorspace for dithering
    let output_colorspace = args.output_distance_space.unwrap_or(if format.is_grayscale {
        ColorSpace::LabCie94
    } else {
        ColorSpace::Oklab
    });

    // Now perform format-dependent validation
    if args.verbose && args.format.is_some() {
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

    // Check binary output compatibility
    if args.output_raw.is_some() && !format.supports_binary() {
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
            eprintln!("Output alpha dither: {:?}", alpha_dither);
        }
        eprintln!(
            "Output colorspace: {:?}{}",
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

        // Apply input tonemapping before resize (if specified)
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
        let mut resize_progress = |p: f32| print_progress("Resize", p);
        let (input_pixels, width, height) = resize_linear(
            &input_pixels,
            src_width, src_height,
            args.width, args.height,
            args.scale_method.to_rescale_method(),
            original_has_alpha,
            args.non_uniform,
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
            // GRAYSCALE PATH: Convert to grayscale, then tonemapping, then safetensors, then dither

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

            // Step 5: Write safetensors output (grayscale as R=G=B=L)
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

            // Step 6: Dither grayscale
            if has_integer_output {
                let mut dither_progress = |p: f32| print_progress("Dither", p);

                // Convert linear to sRGB and denormalize to 0-255
                let srgb_gray: Vec<f32> = linear_gray.iter()
                    .map(|&l| linear_to_srgb_single(l) * 255.0)
                    .collect();

                let result = if let Some(ref alpha) = alpha {
                    // LA format: grayscale with alpha
                    let alpha_255: Vec<f32> = alpha.iter().map(|&a| a * 255.0).collect();
                    let technique = build_output_technique(!args.no_colorspace_aware_output, output_dither_mode, output_colorspace.to_perceptual_space(), output_alpha_mode);
                    let interleaved = dither_output_la(
                        &srgb_gray, &alpha_255, width_usize, height_usize, format.bits_r, format.bits_a,
                        technique, args.seed, if args.progress { Some(&mut dither_progress) } else { None },
                    );
                    DitherResult { interleaved, is_grayscale: true, has_alpha: true }
                } else {
                    // Pure grayscale
                    let interleaved = dither_grayscale(
                        &srgb_gray, width_usize, height_usize, format.bits_r,
                        output_colorspace.to_perceptual_space(), output_dither_mode, args.seed,
                        if args.progress { Some(&mut dither_progress) } else { None },
                    );
                    DitherResult { interleaved, is_grayscale: true, has_alpha: false }
                };

                if args.progress {
                    clear_progress();
                }
                Some(result)
            } else {
                None
            }
        } else {
            // RGB PATH: Exposure, tonemapping, then safetensors, then dither

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

            // Step 3: Write safetensors output
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

            // Step 4: Dither RGB
            if has_integer_output {
                let mut dither_progress = |p: f32| print_progress("Dither", p);
                let result = dither_pixels_rgb(
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
                    if args.progress { Some(&mut dither_progress) } else { None },
                );
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
            let result = dither_pixels_srgb_rgb(
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
                if args.progress { Some(&mut dither_progress) } else { None },
            );
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
            }
        } else {
            dither_result
        };

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

        // Write separate channel outputs (R, G, B, A) - encode directly from interleaved data
        let fill = args.stride_fill.to_stride_fill();
        let row_aligned = args.stride > 1;
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
                let label = if row_aligned { "binary_r_row_aligned" } else { "binary_r" };
                outputs.push((label.to_string(), path.clone(), bin_data.len()));
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
                let label = if row_aligned { "binary_g_row_aligned" } else { "binary_g" };
                outputs.push((label.to_string(), path.clone(), bin_data.len()));
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
                let label = if row_aligned { "binary_b_row_aligned" } else { "binary_b" };
                outputs.push((label.to_string(), path.clone(), bin_data.len()));
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
                let label = if row_aligned { "binary_a_row_aligned" } else { "binary_a" };
                outputs.push((label.to_string(), path.clone(), bin_data.len()));
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
        write_metadata(
            meta_path,
            &args,
            &format,
            histogram,
            output_colorspace,
            width,
            height,
            &outputs,
            safetensors_meta.as_ref(),
            has_integer_output,
        )?;
    }

    if args.verbose {
        eprintln!("Done!");
    }

    Ok(())
}
