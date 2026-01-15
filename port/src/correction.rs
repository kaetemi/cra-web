/// Unified color correction API.
///
/// This module provides a single entry point for all color correction methods,
/// unifying the various algorithms (Basic, CRA, Tiled) and color spaces
/// (RGB, LAB, OKLab) into a consistent interface.
///
/// The primary API is `color_correct` which takes a `ColorCorrectionMethod` enum
/// to select the algorithm and method-specific options.

use crate::basic_lab;
use crate::basic_oklab;
use crate::basic_rgb;
use crate::cra_lab;
use crate::cra_rgb;
use crate::dither_common::{ColorCorrectionMethod, DitherMode, HistogramMode, PerceptualSpace};
use crate::pixel::Pixel4;
use crate::tiled_lab;

/// Options for histogram matching during color correction
#[derive(Debug, Clone, Copy)]
pub struct HistogramOptions {
    /// Histogram matching mode (binned, endpoint-aligned, or midpoint-aligned)
    pub mode: HistogramMode,
    /// Dithering mode for histogram quantization
    pub dither_mode: DitherMode,
    /// Enable colorspace-aware histogram dithering (CRA/Tiled methods only)
    pub colorspace_aware: bool,
    /// Perceptual space for colorspace-aware histogram dithering
    pub colorspace_aware_space: PerceptualSpace,
}

impl Default for HistogramOptions {
    fn default() -> Self {
        Self {
            mode: HistogramMode::default(),
            dither_mode: DitherMode::MixedStandard, // Default for histogram processing
            colorspace_aware: true,
            colorspace_aware_space: PerceptualSpace::default(),
        }
    }
}

impl HistogramOptions {
    /// Convert HistogramMode to the u8 format used by internal functions
    fn mode_as_u8(&self) -> u8 {
        match self.mode {
            HistogramMode::Binned => 0,
            HistogramMode::EndpointAligned => 1,
            HistogramMode::MidpointAligned => 2,
        }
    }
}

/// Perform color correction using the specified method.
///
/// This is the unified entry point for all color correction algorithms.
/// It takes linear RGB input and returns linear RGB output.
///
/// # Arguments
/// * `input` - Input image as linear RGB Pixel4 array (0-1 range)
/// * `reference` - Reference image as linear RGB Pixel4 array (0-1 range)
/// * `input_width`, `input_height` - Input image dimensions
/// * `ref_width`, `ref_height` - Reference image dimensions
/// * `method` - Color correction method selection
/// * `histogram_options` - Histogram matching options
///
/// # Returns
/// Output image as linear RGB Pixel4 array (0-1 range)
#[allow(clippy::too_many_arguments)]
pub fn color_correct(
    input: &[Pixel4],
    reference: &[Pixel4],
    input_width: usize,
    input_height: usize,
    ref_width: usize,
    ref_height: usize,
    method: ColorCorrectionMethod,
    histogram_options: HistogramOptions,
) -> Vec<Pixel4> {
    let hist_mode = histogram_options.mode_as_u8();
    let hist_dither = histogram_options.dither_mode;
    let colorspace_aware = histogram_options.colorspace_aware;
    let colorspace_aware_space = histogram_options.colorspace_aware_space;

    match method {
        ColorCorrectionMethod::BasicLab { keep_luminosity } => {
            basic_lab::color_correct_basic_lab_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                keep_luminosity,
                hist_mode,
                hist_dither,
            )
        }
        ColorCorrectionMethod::BasicRgb => {
            basic_rgb::color_correct_basic_rgb_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                hist_mode,
                hist_dither,
            )
        }
        ColorCorrectionMethod::BasicOklab { keep_luminosity } => {
            basic_oklab::color_correct_basic_oklab_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                keep_luminosity,
                hist_mode,
                hist_dither,
            )
        }
        ColorCorrectionMethod::CraLab { keep_luminosity } => {
            cra_lab::color_correct_cra_lab_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                keep_luminosity,
                hist_mode,
                hist_dither,
                colorspace_aware,
                colorspace_aware_space,
            )
        }
        ColorCorrectionMethod::CraRgb { use_perceptual } => {
            cra_rgb::color_correct_cra_rgb_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                use_perceptual,
                hist_mode,
                hist_dither,
            )
        }
        ColorCorrectionMethod::CraOklab { keep_luminosity } => {
            cra_lab::color_correct_cra_oklab_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                keep_luminosity,
                hist_mode,
                hist_dither,
                colorspace_aware,
                colorspace_aware_space,
            )
        }
        ColorCorrectionMethod::TiledLab { tiled_luminosity } => {
            tiled_lab::color_correct_tiled_lab_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                tiled_luminosity,
                hist_mode,
                hist_dither,
                colorspace_aware,
                colorspace_aware_space,
            )
        }
        ColorCorrectionMethod::TiledOklab { tiled_luminosity } => {
            tiled_lab::color_correct_tiled_oklab_linear(
                input,
                reference,
                input_width,
                input_height,
                ref_width,
                ref_height,
                tiled_luminosity,
                hist_mode,
                hist_dither,
                colorspace_aware,
                colorspace_aware_space,
            )
        }
    }
}
