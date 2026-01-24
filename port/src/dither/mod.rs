//! Dithering and error diffusion module
//!
//! Provides various dithering algorithms for quantizing images from high bit-depth
//! to lower bit-depth while maintaining visual quality through error diffusion.
//!
//! # Module Structure
//! - `common`: Shared types (DitherMode, PerceptualSpace, etc.) and utility functions
//! - `basic`: Basic single-channel and naive RGB dithering
//! - `lab`: LAB colorspace dithering
//! - `luminosity`: Grayscale/luminosity colorspace-aware dithering
//! - `luminosity_alpha`: Grayscale with alpha colorspace-aware dithering
//! - `rgb`: Colorspace-aware RGB dithering
//! - `rgba`: Colorspace-aware RGBA dithering
//! - `paletted`: Palette-based RGBA dithering with integrated alpha-RGB distance
//! - `palette_hull`: Convex hull computation for palettes in linear RGB space
//! - `palette_projection`: Hull tracing and gamut mapping for palette dithering
//! - `fp16`: FP16 (half-precision float) dithering
//! - `bf16`: BF16 (brain float) dithering
//! - `kernels`: Error diffusion kernel implementations
//! - `bitdepth`: Bit depth quantization utilities (bit_replicate, linear LUT, quant params)

pub mod kernels;
pub mod bitdepth;
pub mod common;
pub mod basic;
pub mod lab;
pub mod luminosity;
pub mod luminosity_alpha;
pub mod rgb;
pub mod rgba;
pub mod paletted;
pub mod palette_hull;
pub mod palette_projection;
pub mod fp16;
pub mod bf16;

// Re-export common types at the dither module level for convenience
pub use bitdepth::{bit_replicate, build_linear_lut, QuantLevelParams};
pub use common::{
    lowbias32,
    wang_hash,
    ColorCorrectionMethod,
    DitherMode,
    HistogramMode,
    OutputTechnique,
    PerceptualSpace,
};

// Re-export core dithering functions
pub use basic::{
    dither_rgb_channels_with_mode,
    dither_rgb_with_mode,
    dither_with_mode,
    dither_with_mode_bits,
    floyd_steinberg_dither,
    floyd_steinberg_dither_bits,
};

// Re-export LAB dithering
pub use lab::{
    lab_space_dither,
    lab_space_dither_with_mode,
    LabQuantParams,
    LabQuantSpace,
};

// Re-export luminosity dithering
pub use luminosity::{
    colorspace_aware_dither_gray,
    colorspace_aware_dither_gray_with_mode,
};

// Re-export luminosity+alpha dithering
pub use luminosity_alpha::{
    colorspace_aware_dither_gray_alpha,
    colorspace_aware_dither_gray_alpha_with_mode,
};

// Re-export RGB dithering
pub use rgb::{
    colorspace_aware_dither_rgb,
    colorspace_aware_dither_rgb_channels,
    colorspace_aware_dither_rgb_interleaved,
    colorspace_aware_dither_rgb_with_mode,
};

// Re-export RGBA dithering
pub use rgba::{
    colorspace_aware_dither_rgba,
    colorspace_aware_dither_rgba_channels,
    colorspace_aware_dither_rgba_interleaved,
    colorspace_aware_dither_rgba_with_mode,
};

// Re-export paletted dithering
pub use paletted::{
    paletted_dither_rgba,
    paletted_dither_rgba_channels,
    paletted_dither_rgba_gamut_mapped,
    paletted_dither_rgba_interleaved,
    paletted_dither_rgba_with_mode,
    paletted_dither_to_indices,
    DitherPalette,
};

// Re-export palette hull types
pub use palette_hull::{
    HullPlane,
    PaletteHull,
    EPSILON as HULL_EPSILON,
};

// Re-export palette projection types
pub use palette_projection::ExtendedPalette;

// Re-export FP16 dithering
pub use fp16::{
    dither_rgb_f16_with_mode,
    dither_rgba_f16,
    dither_rgba_f16_with_mode,
    Fp16WorkingSpace,
};

// Re-export BF16 dithering
pub use bf16::{
    dither_rgb_bf16_with_mode,
    dither_rgba_bf16,
    dither_rgba_bf16_with_mode,
    Bf16WorkingSpace,
};
