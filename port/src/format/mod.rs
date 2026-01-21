//! Image format handling modules.
//!
//! This module contains:
//! - `decode`: General image decoding with ICC/CICP profile support
//! - `sfi`: Safetensors Floating-point Image format support
//! - `binary`: Raw binary format encoding/decoding
//! - `color_format`: Color format definition and parsing
//! - `rgb666`: RGB666 special format encoding/decoding
//! - `palettized_png`: Palettized PNG output for low-bit-depth formats

pub mod binary;
pub mod color_format;
pub mod decode;
pub mod palettized_png;
pub mod rgb666;
pub mod sfi;

// Re-export commonly used types at the format level
pub use binary::{RawImageMetadata, DecodedRawImage, StrideFill};
pub use color_format::ColorFormat;
pub use decode::{DecodedImage, ImageMetadata};
pub use sfi::{SfiImage, SfiMetadata, SfiError, SfiTransfer, SfiDtype};
