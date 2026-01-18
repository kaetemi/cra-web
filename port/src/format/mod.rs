//! Image format handling modules.
//!
//! This module contains:
//! - `decode`: General image decoding with ICC/CICP profile support
//! - `sfi`: Safetensors Floating-point Image format support
//! - `binary`: Raw binary format encoding/decoding

pub mod binary;
pub mod decode;
pub mod sfi;

// Re-export commonly used types at the format level
pub use binary::{ColorFormat, RawImageMetadata, DecodedRawImage, StrideFill};
pub use decode::{DecodedImage, ImageMetadata};
pub use sfi::{SfiImage, SfiMetadata, SfiError, SfiTransfer, SfiDtype};
