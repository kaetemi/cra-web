//! SFI (Safetensors Floating-point Image) format support
//!
//! Implements Level 2 conformance of the SFI specification:
//! - Read and write F32 tensors
//! - Support F16 tensors
//! - Support unspecified/srgb primaries
//! - Support unspecified/linear/srgb transfer
//! - Support RGB and RGBA channels
//! - Support HWC and CHW dimension order

use half::{bf16, f16};
use serde_json::{json, Map, Value};

use crate::color::{linear_to_srgb_single, srgb_to_linear_single};
use crate::pixel::Pixel4;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur when reading/writing SFI files
#[derive(Debug)]
pub enum SfiError {
    /// File is too short to contain valid SFI header
    FileTooShort,
    /// Invalid header size (corrupted or not safetensors)
    InvalidHeaderSize,
    /// JSON parse error in header
    JsonParseError(String),
    /// Missing required metadata field
    MissingMetadata(String),
    /// Invalid metadata value
    InvalidMetadata(String),
    /// Tensor not found
    TensorNotFound(String),
    /// Invalid tensor dtype
    InvalidDtype(String),
    /// Invalid tensor shape
    InvalidShape(String),
    /// Data size mismatch
    DataSizeMismatch { expected: usize, actual: usize },
    /// Unsupported feature
    Unsupported(String),
}

impl std::fmt::Display for SfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SfiError::FileTooShort => write!(f, "File too short for safetensors format"),
            SfiError::InvalidHeaderSize => write!(f, "Invalid header size"),
            SfiError::JsonParseError(e) => write!(f, "JSON parse error: {}", e),
            SfiError::MissingMetadata(field) => write!(f, "Missing required metadata: {}", field),
            SfiError::InvalidMetadata(msg) => write!(f, "Invalid metadata: {}", msg),
            SfiError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            SfiError::InvalidDtype(dtype) => write!(f, "Invalid or unsupported dtype: {}", dtype),
            SfiError::InvalidShape(msg) => write!(f, "Invalid tensor shape: {}", msg),
            SfiError::DataSizeMismatch { expected, actual } => {
                write!(f, "Data size mismatch: expected {} bytes, got {}", expected, actual)
            }
            SfiError::Unsupported(msg) => write!(f, "Unsupported feature: {}", msg),
        }
    }
}

impl std::error::Error for SfiError {}

// ============================================================================
// Metadata Types
// ============================================================================

/// Color primaries identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SfiPrimaries {
    #[default]
    Unspecified,
    Srgb,
}

impl SfiPrimaries {
    fn from_str(s: &str) -> Result<Self, SfiError> {
        match s {
            "unspecified" => Ok(SfiPrimaries::Unspecified),
            "srgb" => Ok(SfiPrimaries::Srgb),
            _ => Err(SfiError::InvalidMetadata(format!("Unknown primaries: {}", s))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            SfiPrimaries::Unspecified => "unspecified",
            SfiPrimaries::Srgb => "srgb",
        }
    }
}

/// Transfer characteristics identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SfiTransfer {
    Unspecified,
    #[default]
    Linear,
    Srgb,
}

impl SfiTransfer {
    fn from_str(s: &str) -> Result<Self, SfiError> {
        match s {
            "unspecified" => Ok(SfiTransfer::Unspecified),
            "linear" => Ok(SfiTransfer::Linear),
            "srgb" => Ok(SfiTransfer::Srgb),
            _ => Err(SfiError::InvalidMetadata(format!("Unknown transfer: {}", s))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            SfiTransfer::Unspecified => "unspecified",
            SfiTransfer::Linear => "linear",
            SfiTransfer::Srgb => "srgb",
        }
    }
}

/// Channel configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SfiChannels {
    #[default]
    Rgb,
    Rgba,
}

impl SfiChannels {
    fn from_str(s: &str) -> Result<Self, SfiError> {
        match s {
            "RGB" => Ok(SfiChannels::Rgb),
            "RGBA" => Ok(SfiChannels::Rgba),
            _ => Err(SfiError::InvalidMetadata(format!("Unknown channels: {}", s))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            SfiChannels::Rgb => "RGB",
            SfiChannels::Rgba => "RGBA",
        }
    }

    fn count(&self) -> usize {
        match self {
            SfiChannels::Rgb => 3,
            SfiChannels::Rgba => 4,
        }
    }
}

/// Tensor dimension order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SfiDimensionOrder {
    #[default]
    Hwc,
    Chw,
}

impl SfiDimensionOrder {
    fn from_str(s: &str) -> Result<Self, SfiError> {
        match s {
            "HWC" => Ok(SfiDimensionOrder::Hwc),
            "CHW" => Ok(SfiDimensionOrder::Chw),
            _ => Err(SfiError::InvalidMetadata(format!("Unknown dimension_order: {}", s))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            SfiDimensionOrder::Hwc => "HWC",
            SfiDimensionOrder::Chw => "CHW",
        }
    }
}

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfiDtype {
    F32,
    F16,
    BF16,
}

impl SfiDtype {
    fn from_str(s: &str) -> Result<Self, SfiError> {
        match s {
            "F32" => Ok(SfiDtype::F32),
            "F16" => Ok(SfiDtype::F16),
            "BF16" => Ok(SfiDtype::BF16),
            _ => Err(SfiError::InvalidDtype(s.to_string())),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            SfiDtype::F32 => "F32",
            SfiDtype::F16 => "F16",
            SfiDtype::BF16 => "BF16",
        }
    }

    fn bytes_per_element(&self) -> usize {
        match self {
            SfiDtype::F32 => 4,
            SfiDtype::F16 | SfiDtype::BF16 => 2,
        }
    }
}

/// SFI image metadata
#[derive(Debug, Clone, Default)]
pub struct SfiMetadata {
    pub primaries: SfiPrimaries,
    pub transfer: SfiTransfer,
    pub channels: SfiChannels,
    pub dimension_order: SfiDimensionOrder,
    pub alpha_premultiplied: bool,
}

// ============================================================================
// SFI Image Type
// ============================================================================

/// Loaded SFI image with pixel data and metadata
#[derive(Debug)]
pub struct SfiImage {
    /// Linear RGB(A) pixel data as Pixel4 (alpha=1.0 for RGB)
    pub pixels: Vec<Pixel4>,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Whether the image has alpha channel
    pub has_alpha: bool,
    /// Original dtype (F16 or F32)
    pub dtype: SfiDtype,
    /// Image metadata
    pub metadata: SfiMetadata,
}

// ============================================================================
// Format Detection
// ============================================================================

/// Check if data is a valid SFI (safetensors) format file.
/// Validates header size and checks for SFI metadata marker.
pub fn is_sfi_format(data: &[u8]) -> bool {
    // Need at least 8 bytes for header size
    if data.len() < 8 {
        return false;
    }

    // Read header size (u64 LE)
    let header_size = u64::from_le_bytes([
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
    ]) as usize;

    // Sanity check: header size should be reasonable
    if header_size == 0 || header_size > 100_000_000 || 8 + header_size > data.len() {
        return false;
    }

    // Parse JSON header to verify it's valid and contains SFI metadata
    let json_bytes = &data[8..8 + header_size];
    let json_str = match std::str::from_utf8(json_bytes) {
        Ok(s) => s,
        Err(_) => return false,
    };

    let parsed: Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return false,
    };

    // Check for __metadata__ with format = "sfi"
    if let Some(metadata) = parsed.get("__metadata__") {
        if let Some(format) = metadata.get("format") {
            return format.as_str() == Some("sfi");
        }
    }

    false
}

// ============================================================================
// Reading SFI Files
// ============================================================================

/// Read an SFI file and return the decoded image, converting to linear RGB.
/// This is the common case for image processing pipelines.
pub fn read_sfi(data: &[u8]) -> Result<SfiImage, SfiError> {
    read_sfi_with_transfer(data, SfiTransfer::Linear)
}

/// Read an SFI file and return the decoded image with specified output transfer.
/// - `target_transfer`: The desired output transfer function (Linear or Srgb)
pub fn read_sfi_with_transfer(data: &[u8], target_transfer: SfiTransfer) -> Result<SfiImage, SfiError> {
    // Read header size
    if data.len() < 8 {
        return Err(SfiError::FileTooShort);
    }

    let header_size = u64::from_le_bytes([
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
    ]) as usize;

    if header_size == 0 || 8 + header_size > data.len() {
        return Err(SfiError::InvalidHeaderSize);
    }

    // Parse JSON header
    let json_bytes = &data[8..8 + header_size];
    let json_str = std::str::from_utf8(json_bytes)
        .map_err(|e| SfiError::JsonParseError(e.to_string()))?;

    let parsed: Value = serde_json::from_str(json_str)
        .map_err(|e| SfiError::JsonParseError(e.to_string()))?;

    // Extract __metadata__
    let meta = parsed.get("__metadata__")
        .ok_or_else(|| SfiError::MissingMetadata("__metadata__".to_string()))?;

    // Verify format
    let format = meta.get("format")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SfiError::MissingMetadata("format".to_string()))?;
    if format != "sfi" {
        return Err(SfiError::InvalidMetadata(format!("Expected format 'sfi', got '{}'", format)));
    }

    // Parse metadata fields
    let primaries_str = meta.get("primaries")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SfiError::MissingMetadata("primaries".to_string()))?;
    let primaries = SfiPrimaries::from_str(primaries_str)?;

    let transfer_str = meta.get("transfer")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SfiError::MissingMetadata("transfer".to_string()))?;
    let transfer = SfiTransfer::from_str(transfer_str)?;

    let channels_str = meta.get("channels")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SfiError::MissingMetadata("channels".to_string()))?;
    let channels = SfiChannels::from_str(channels_str)?;

    let dim_order_str = meta.get("dimension_order")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SfiError::MissingMetadata("dimension_order".to_string()))?;
    let dimension_order = SfiDimensionOrder::from_str(dim_order_str)?;

    let alpha_premultiplied = meta.get("alpha_premultiplied")
        .and_then(|v| v.as_str())
        .map(|s| s == "true")
        .unwrap_or(false);

    let metadata = SfiMetadata {
        primaries,
        transfer,
        channels,
        dimension_order,
        alpha_premultiplied,
    };

    // Find the "image" tensor
    let tensor_info = parsed.get("image")
        .ok_or_else(|| SfiError::TensorNotFound("image".to_string()))?;

    // Parse dtype
    let dtype_str = tensor_info.get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SfiError::MissingMetadata("dtype".to_string()))?;
    let dtype = SfiDtype::from_str(dtype_str)?;

    // Parse shape
    let shape = tensor_info.get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| SfiError::MissingMetadata("shape".to_string()))?;

    if shape.len() != 3 {
        return Err(SfiError::InvalidShape(format!("Expected 3 dimensions, got {}", shape.len())));
    }

    let dims: Vec<usize> = shape.iter()
        .map(|v| v.as_u64().map(|n| n as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| SfiError::InvalidShape("Invalid dimension values".to_string()))?;

    // Parse dimension order to get H, W, C
    let (height, width, num_channels) = match dimension_order {
        SfiDimensionOrder::Hwc => (dims[0], dims[1], dims[2]),
        SfiDimensionOrder::Chw => (dims[1], dims[2], dims[0]),
    };

    // Validate channel count matches metadata
    if num_channels != channels.count() {
        return Err(SfiError::InvalidShape(format!(
            "Channel count mismatch: shape has {} channels, metadata says {}",
            num_channels, channels.as_str()
        )));
    }

    // Parse data offsets
    let offsets = tensor_info.get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| SfiError::MissingMetadata("data_offsets".to_string()))?;

    if offsets.len() != 2 {
        return Err(SfiError::InvalidShape("data_offsets must have 2 elements".to_string()));
    }

    let start = offsets[0].as_u64().ok_or_else(|| SfiError::InvalidShape("Invalid data_offsets".to_string()))? as usize;
    let end = offsets[1].as_u64().ok_or_else(|| SfiError::InvalidShape("Invalid data_offsets".to_string()))? as usize;

    // Calculate expected data size
    let pixel_count = height * width;
    let expected_bytes = pixel_count * num_channels * dtype.bytes_per_element();
    let actual_bytes = end - start;

    if expected_bytes != actual_bytes {
        return Err(SfiError::DataSizeMismatch {
            expected: expected_bytes,
            actual: actual_bytes,
        });
    }

    // Get tensor data
    let data_start = 8 + header_size + start;
    let data_end = 8 + header_size + end;

    if data_end > data.len() {
        return Err(SfiError::FileTooShort);
    }

    let tensor_data = &data[data_start..data_end];

    // Read pixels based on dtype and dimension order
    let has_alpha = channels == SfiChannels::Rgba;
    let pixels = read_pixels(tensor_data, width, height, dtype, dimension_order, num_channels, transfer, target_transfer)?;

    Ok(SfiImage {
        pixels,
        width: width as u32,
        height: height as u32,
        has_alpha,
        dtype,
        metadata,
    })
}

/// Read pixel data from tensor bytes, converting from source to target transfer
fn read_pixels(
    data: &[u8],
    width: usize,
    height: usize,
    dtype: SfiDtype,
    dimension_order: SfiDimensionOrder,
    num_channels: usize,
    source_transfer: SfiTransfer,
    target_transfer: SfiTransfer,
) -> Result<Vec<Pixel4>, SfiError> {
    let pixel_count = width * height;
    let mut pixels = Vec::with_capacity(pixel_count);

    // Determine conversion needed
    let needs_srgb_to_linear = source_transfer == SfiTransfer::Srgb && target_transfer == SfiTransfer::Linear;
    let needs_linear_to_srgb = source_transfer == SfiTransfer::Linear && target_transfer == SfiTransfer::Srgb;

    match dtype {
        SfiDtype::F32 => {
            let float_data: Vec<f32> = data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            for i in 0..pixel_count {
                let (r, g, b, a) = read_pixel_at(&float_data, i, width, height, dimension_order, num_channels);

                let (r, g, b) = if needs_srgb_to_linear {
                    (srgb_to_linear_single(r), srgb_to_linear_single(g), srgb_to_linear_single(b))
                } else if needs_linear_to_srgb {
                    (linear_to_srgb_single(r), linear_to_srgb_single(g), linear_to_srgb_single(b))
                } else {
                    (r, g, b)
                };

                pixels.push(Pixel4::new(r, g, b, a));
            }
        }
        SfiDtype::F16 => {
            let f16_data: Vec<f16> = data.chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            for i in 0..pixel_count {
                let (r, g, b, a) = read_pixel_at_f16(&f16_data, i, width, height, dimension_order, num_channels);

                let (r, g, b) = if needs_srgb_to_linear {
                    (srgb_to_linear_single(r), srgb_to_linear_single(g), srgb_to_linear_single(b))
                } else if needs_linear_to_srgb {
                    (linear_to_srgb_single(r), linear_to_srgb_single(g), linear_to_srgb_single(b))
                } else {
                    (r, g, b)
                };

                pixels.push(Pixel4::new(r, g, b, a));
            }
        }
        SfiDtype::BF16 => {
            let bf16_data: Vec<bf16> = data.chunks_exact(2)
                .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            for i in 0..pixel_count {
                let (r, g, b, a) = read_pixel_at_bf16(&bf16_data, i, width, height, dimension_order, num_channels);

                let (r, g, b) = if needs_srgb_to_linear {
                    (srgb_to_linear_single(r), srgb_to_linear_single(g), srgb_to_linear_single(b))
                } else if needs_linear_to_srgb {
                    (linear_to_srgb_single(r), linear_to_srgb_single(g), linear_to_srgb_single(b))
                } else {
                    (r, g, b)
                };

                pixels.push(Pixel4::new(r, g, b, a));
            }
        }
    }

    Ok(pixels)
}

/// Read a single pixel at index i from f32 data
fn read_pixel_at(
    data: &[f32],
    pixel_idx: usize,
    width: usize,
    height: usize,
    dim_order: SfiDimensionOrder,
    num_channels: usize,
) -> (f32, f32, f32, f32) {
    let y = pixel_idx / width;
    let x = pixel_idx % width;

    match dim_order {
        SfiDimensionOrder::Hwc => {
            let base = (y * width + x) * num_channels;
            let r = data[base];
            let g = data[base + 1];
            let b = data[base + 2];
            let a = if num_channels == 4 { data[base + 3] } else { 1.0 };
            (r, g, b, a)
        }
        SfiDimensionOrder::Chw => {
            let plane_size = height * width;
            let offset = y * width + x;
            let r = data[offset];
            let g = data[plane_size + offset];
            let b = data[2 * plane_size + offset];
            let a = if num_channels == 4 { data[3 * plane_size + offset] } else { 1.0 };
            (r, g, b, a)
        }
    }
}

/// Read a single pixel at index i from f16 data
fn read_pixel_at_f16(
    data: &[f16],
    pixel_idx: usize,
    width: usize,
    height: usize,
    dim_order: SfiDimensionOrder,
    num_channels: usize,
) -> (f32, f32, f32, f32) {
    let y = pixel_idx / width;
    let x = pixel_idx % width;

    match dim_order {
        SfiDimensionOrder::Hwc => {
            let base = (y * width + x) * num_channels;
            let r = data[base].to_f32();
            let g = data[base + 1].to_f32();
            let b = data[base + 2].to_f32();
            let a = if num_channels == 4 { data[base + 3].to_f32() } else { 1.0 };
            (r, g, b, a)
        }
        SfiDimensionOrder::Chw => {
            let plane_size = height * width;
            let offset = y * width + x;
            let r = data[offset].to_f32();
            let g = data[plane_size + offset].to_f32();
            let b = data[2 * plane_size + offset].to_f32();
            let a = if num_channels == 4 { data[3 * plane_size + offset].to_f32() } else { 1.0 };
            (r, g, b, a)
        }
    }
}

/// Read a single pixel at index i from bf16 data
fn read_pixel_at_bf16(
    data: &[bf16],
    pixel_idx: usize,
    width: usize,
    height: usize,
    dim_order: SfiDimensionOrder,
    num_channels: usize,
) -> (f32, f32, f32, f32) {
    let y = pixel_idx / width;
    let x = pixel_idx % width;

    match dim_order {
        SfiDimensionOrder::Hwc => {
            let base = (y * width + x) * num_channels;
            let r = data[base].to_f32();
            let g = data[base + 1].to_f32();
            let b = data[base + 2].to_f32();
            let a = if num_channels == 4 { data[base + 3].to_f32() } else { 1.0 };
            (r, g, b, a)
        }
        SfiDimensionOrder::Chw => {
            let plane_size = height * width;
            let offset = y * width + x;
            let r = data[offset].to_f32();
            let g = data[plane_size + offset].to_f32();
            let b = data[2 * plane_size + offset].to_f32();
            let a = if num_channels == 4 { data[3 * plane_size + offset].to_f32() } else { 1.0 };
            (r, g, b, a)
        }
    }
}

// ============================================================================
// Writing SFI Files
// ============================================================================

/// Write linear RGB pixels to SFI F32 format
pub fn write_sfi_f32(
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
) -> Vec<u8> {
    write_sfi_internal(pixels, width, height, include_alpha, transfer, SfiDtype::F32, false)
}

/// Write linear RGB pixels to SFI F16 format.
pub fn write_sfi_f16(
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
) -> Vec<u8> {
    write_sfi_internal(pixels, width, height, include_alpha, transfer, SfiDtype::F16, false)
}

/// Write linear RGB pixels to SFI BF16 format.
pub fn write_sfi_bf16(
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
) -> Vec<u8> {
    write_sfi_internal(pixels, width, height, include_alpha, transfer, SfiDtype::BF16, false)
}

/// Write linear RGB pixels to SFI F16 format with alpha premultiply flag.
pub fn write_sfi_f16_premultiplied(
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
    alpha_premultiplied: bool,
) -> Vec<u8> {
    write_sfi_internal(pixels, width, height, include_alpha, transfer, SfiDtype::F16, alpha_premultiplied)
}

/// Write linear RGB pixels to SFI F32 format with alpha premultiply flag.
pub fn write_sfi_f32_premultiplied(
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
    alpha_premultiplied: bool,
) -> Vec<u8> {
    write_sfi_internal(pixels, width, height, include_alpha, transfer, SfiDtype::F32, alpha_premultiplied)
}

/// Write pre-dithered F16 RGB(A) channels to SFI format.
/// Takes separate channel data as f16 slices (already dithered).
pub fn write_sfi_f16_channels(
    r_channel: &[f16],
    g_channel: &[f16],
    b_channel: &[f16],
    a_channel: Option<&[f16]>,
    width: u32,
    height: u32,
    transfer: SfiTransfer,
) -> Vec<u8> {
    write_sfi_channels_internal(
        r_channel, g_channel, b_channel, a_channel,
        width, height, transfer, SfiDtype::F16, false,
    )
}

/// Write pre-dithered BF16 RGB(A) channels to SFI format.
/// Takes separate channel data as bf16 slices (already dithered).
pub fn write_sfi_bf16_channels(
    r_channel: &[bf16],
    g_channel: &[bf16],
    b_channel: &[bf16],
    a_channel: Option<&[bf16]>,
    width: u32,
    height: u32,
    transfer: SfiTransfer,
) -> Vec<u8> {
    write_sfi_bf16_channels_internal(
        r_channel, g_channel, b_channel, a_channel,
        width, height, transfer, false,
    )
}

fn write_sfi_channels_internal(
    r_channel: &[f16],
    g_channel: &[f16],
    b_channel: &[f16],
    a_channel: Option<&[f16]>,
    width: u32,
    height: u32,
    transfer: SfiTransfer,
    dtype: SfiDtype,
    alpha_premultiplied: bool,
) -> Vec<u8> {
    let include_alpha = a_channel.is_some();
    let num_channels = if include_alpha { 4 } else { 3 };
    let channels = if include_alpha { SfiChannels::Rgba } else { SfiChannels::Rgb };
    let pixel_count = (width * height) as usize;
    let element_size = dtype.bytes_per_element();
    let tensor_bytes = pixel_count * num_channels * element_size;

    // Build metadata JSON
    let mut metadata = Map::new();
    metadata.insert("format".to_string(), json!("sfi"));
    metadata.insert("version".to_string(), json!("1.0"));
    metadata.insert("primaries".to_string(), json!("srgb"));
    metadata.insert("transfer".to_string(), json!(transfer.as_str()));
    metadata.insert("channels".to_string(), json!(channels.as_str()));
    metadata.insert("dimension_order".to_string(), json!("HWC"));
    if include_alpha {
        metadata.insert("alpha_premultiplied".to_string(), json!(if alpha_premultiplied { "true" } else { "false" }));
    }

    // Build tensor info
    let shape = if include_alpha {
        json!([height, width, 4])
    } else {
        json!([height, width, 3])
    };

    let mut tensor_info = Map::new();
    tensor_info.insert("dtype".to_string(), json!(dtype.as_str()));
    tensor_info.insert("shape".to_string(), shape);
    tensor_info.insert("data_offsets".to_string(), json!([0, tensor_bytes]));

    // Build header object
    let mut header = Map::new();
    header.insert("__metadata__".to_string(), Value::Object(metadata));
    header.insert("image".to_string(), Value::Object(tensor_info));

    // Serialize header to JSON
    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Calculate total output size
    let total_size = 8 + header_bytes.len() + tensor_bytes;
    let mut output = Vec::with_capacity(total_size);

    // Write header size (u64 LE)
    output.extend_from_slice(&header_size.to_le_bytes());

    // Write header JSON
    output.extend_from_slice(header_bytes);

    // Write tensor data (already dithered, just interleave and write)
    for i in 0..pixel_count {
        output.extend_from_slice(&r_channel[i].to_le_bytes());
        output.extend_from_slice(&g_channel[i].to_le_bytes());
        output.extend_from_slice(&b_channel[i].to_le_bytes());
        if let Some(a) = a_channel {
            output.extend_from_slice(&a[i].to_le_bytes());
        }
    }

    output
}

fn write_sfi_bf16_channels_internal(
    r_channel: &[bf16],
    g_channel: &[bf16],
    b_channel: &[bf16],
    a_channel: Option<&[bf16]>,
    width: u32,
    height: u32,
    transfer: SfiTransfer,
    alpha_premultiplied: bool,
) -> Vec<u8> {
    let include_alpha = a_channel.is_some();
    let num_channels = if include_alpha { 4 } else { 3 };
    let channels = if include_alpha { SfiChannels::Rgba } else { SfiChannels::Rgb };
    let pixel_count = (width * height) as usize;
    let element_size = SfiDtype::BF16.bytes_per_element();
    let tensor_bytes = pixel_count * num_channels * element_size;

    // Build metadata JSON
    let mut metadata = Map::new();
    metadata.insert("format".to_string(), json!("sfi"));
    metadata.insert("version".to_string(), json!("1.0"));
    metadata.insert("primaries".to_string(), json!("srgb"));
    metadata.insert("transfer".to_string(), json!(transfer.as_str()));
    metadata.insert("channels".to_string(), json!(channels.as_str()));
    metadata.insert("dimension_order".to_string(), json!("HWC"));
    if include_alpha {
        metadata.insert("alpha_premultiplied".to_string(), json!(if alpha_premultiplied { "true" } else { "false" }));
    }

    // Build tensor info
    let shape = if include_alpha {
        json!([height, width, 4])
    } else {
        json!([height, width, 3])
    };

    let mut tensor_info = Map::new();
    tensor_info.insert("dtype".to_string(), json!(SfiDtype::BF16.as_str()));
    tensor_info.insert("shape".to_string(), shape);
    tensor_info.insert("data_offsets".to_string(), json!([0, tensor_bytes]));

    // Build header object
    let mut header = Map::new();
    header.insert("__metadata__".to_string(), Value::Object(metadata));
    header.insert("image".to_string(), Value::Object(tensor_info));

    // Serialize header to JSON
    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Calculate total output size
    let total_size = 8 + header_bytes.len() + tensor_bytes;
    let mut output = Vec::with_capacity(total_size);

    // Write header size (u64 LE)
    output.extend_from_slice(&header_size.to_le_bytes());

    // Write header JSON
    output.extend_from_slice(header_bytes);

    // Write tensor data (already dithered, just interleave and write)
    for i in 0..pixel_count {
        output.extend_from_slice(&r_channel[i].to_le_bytes());
        output.extend_from_slice(&g_channel[i].to_le_bytes());
        output.extend_from_slice(&b_channel[i].to_le_bytes());
        if let Some(a) = a_channel {
            output.extend_from_slice(&a[i].to_le_bytes());
        }
    }

    output
}

fn write_sfi_internal(
    pixels: &[Pixel4],
    width: u32,
    height: u32,
    include_alpha: bool,
    transfer: SfiTransfer,
    dtype: SfiDtype,
    alpha_premultiplied: bool,
) -> Vec<u8> {
    let num_channels = if include_alpha { 4 } else { 3 };
    let channels = if include_alpha { SfiChannels::Rgba } else { SfiChannels::Rgb };
    let pixel_count = (width * height) as usize;
    let element_size = dtype.bytes_per_element();
    let tensor_bytes = pixel_count * num_channels * element_size;

    // Build metadata JSON
    let mut metadata = Map::new();
    metadata.insert("format".to_string(), json!("sfi"));
    metadata.insert("version".to_string(), json!("1.0"));
    metadata.insert("primaries".to_string(), json!("srgb"));
    metadata.insert("transfer".to_string(), json!(transfer.as_str()));
    metadata.insert("channels".to_string(), json!(channels.as_str()));
    metadata.insert("dimension_order".to_string(), json!("HWC"));
    if include_alpha {
        metadata.insert("alpha_premultiplied".to_string(), json!(if alpha_premultiplied { "true" } else { "false" }));
    }

    // Build tensor info
    let shape = if include_alpha {
        json!([height, width, 4])
    } else {
        json!([height, width, 3])
    };

    let mut tensor_info = Map::new();
    tensor_info.insert("dtype".to_string(), json!(dtype.as_str()));
    tensor_info.insert("shape".to_string(), shape);
    tensor_info.insert("data_offsets".to_string(), json!([0, tensor_bytes]));

    // Build header object
    let mut header = Map::new();
    header.insert("__metadata__".to_string(), Value::Object(metadata));
    header.insert("image".to_string(), Value::Object(tensor_info));

    // Serialize header to JSON
    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Calculate total output size
    let total_size = 8 + header_bytes.len() + tensor_bytes;
    let mut output = Vec::with_capacity(total_size);

    // Write header size (u64 LE)
    output.extend_from_slice(&header_size.to_le_bytes());

    // Write header JSON
    output.extend_from_slice(header_bytes);

    // Write tensor data (no colorspace conversion - caller provides data in correct space)
    // Only dtype conversion is performed here
    match dtype {
        SfiDtype::F32 => {
            for pixel in pixels.iter() {
                output.extend_from_slice(&pixel[0].to_le_bytes());
                output.extend_from_slice(&pixel[1].to_le_bytes());
                output.extend_from_slice(&pixel[2].to_le_bytes());
                if include_alpha {
                    output.extend_from_slice(&pixel[3].to_le_bytes());
                }
            }
        }
        SfiDtype::F16 => {
            for pixel in pixels.iter() {
                // Convert to f16 using round-to-nearest (default for half crate)
                output.extend_from_slice(&f16::from_f32(pixel[0]).to_le_bytes());
                output.extend_from_slice(&f16::from_f32(pixel[1]).to_le_bytes());
                output.extend_from_slice(&f16::from_f32(pixel[2]).to_le_bytes());
                if include_alpha {
                    output.extend_from_slice(&f16::from_f32(pixel[3]).to_le_bytes());
                }
            }
        }
        SfiDtype::BF16 => {
            for pixel in pixels.iter() {
                // Convert to bf16 using round-to-nearest (default for half crate)
                output.extend_from_slice(&bf16::from_f32(pixel[0]).to_le_bytes());
                output.extend_from_slice(&bf16::from_f32(pixel[1]).to_le_bytes());
                output.extend_from_slice(&bf16::from_f32(pixel[2]).to_le_bytes());
                if include_alpha {
                    output.extend_from_slice(&bf16::from_f32(pixel[3]).to_le_bytes());
                }
            }
        }
    }

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pixels() -> Vec<Pixel4> {
        vec![
            Pixel4::new(0.0, 0.0, 0.0, 1.0),
            Pixel4::new(1.0, 0.0, 0.0, 1.0),
            Pixel4::new(0.0, 1.0, 0.0, 1.0),
            Pixel4::new(0.0, 0.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn test_write_and_read_f32_rgb() {
        let pixels = create_test_pixels();
        let data = write_sfi_f32(&pixels, 2, 2, false, SfiTransfer::Linear);

        assert!(is_sfi_format(&data));

        let image = read_sfi(&data).unwrap();
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert!(!image.has_alpha);
        assert_eq!(image.dtype, SfiDtype::F32);

        // Check pixels
        for (orig, loaded) in pixels.iter().zip(image.pixels.iter()) {
            assert!((orig[0] - loaded[0]).abs() < 1e-6);
            assert!((orig[1] - loaded[1]).abs() < 1e-6);
            assert!((orig[2] - loaded[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_and_read_f32_rgba() {
        let mut pixels = create_test_pixels();
        pixels[0][3] = 0.5; // Add some alpha variation

        let data = write_sfi_f32(&pixels, 2, 2, true, SfiTransfer::Linear);

        let image = read_sfi(&data).unwrap();
        assert!(image.has_alpha);

        // Check alpha is preserved
        assert!((image.pixels[0][3] - 0.5).abs() < 1e-6);
        assert!((image.pixels[1][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_write_and_read_f16() {
        let pixels = create_test_pixels();
        let data = write_sfi_f16(&pixels, 2, 2, false, SfiTransfer::Linear);

        let image = read_sfi(&data).unwrap();
        assert_eq!(image.dtype, SfiDtype::F16);

        // F16 has lower precision
        for (orig, loaded) in pixels.iter().zip(image.pixels.iter()) {
            assert!((orig[0] - loaded[0]).abs() < 0.001);
            assert!((orig[1] - loaded[1]).abs() < 0.001);
            assert!((orig[2] - loaded[2]).abs() < 0.001);
        }
    }

    #[test]
    fn test_write_and_read_bf16() {
        let pixels = create_test_pixels();
        let data = write_sfi_bf16(&pixels, 2, 2, false, SfiTransfer::Linear);

        let image = read_sfi(&data).unwrap();
        assert_eq!(image.dtype, SfiDtype::BF16);

        // BF16 has lower precision (7 mantissa bits vs 10 for F16)
        for (orig, loaded) in pixels.iter().zip(image.pixels.iter()) {
            assert!((orig[0] - loaded[0]).abs() < 0.01);
            assert!((orig[1] - loaded[1]).abs() < 0.01);
            assert!((orig[2] - loaded[2]).abs() < 0.01);
        }
    }

    #[test]
    fn test_srgb_transfer_read_converts_to_linear() {
        // Write sRGB data (0.5 in sRGB space) - write does NOT convert
        let srgb_pixels = vec![
            Pixel4::new(0.5, 0.5, 0.5, 1.0), // 0.5 in sRGB space
        ];

        let data = write_sfi_f32(&srgb_pixels, 1, 1, false, SfiTransfer::Srgb);

        // Read back - should convert sRGB to linear
        let image = read_sfi(&data).unwrap();

        // sRGB 0.5 → linear is approximately 0.214
        let expected_linear = srgb_to_linear_single(0.5);
        assert!((image.pixels[0][0] - expected_linear).abs() < 1e-5);
    }

    #[test]
    fn test_linear_transfer_no_conversion() {
        // Write linear data - write does NOT convert
        let linear_pixels = vec![
            Pixel4::new(0.5, 0.5, 0.5, 1.0), // 0.5 in linear space
        ];

        let data = write_sfi_f32(&linear_pixels, 1, 1, false, SfiTransfer::Linear);

        // Read back - no conversion needed for linear
        let image = read_sfi(&data).unwrap();

        // Should get back exactly what we wrote
        assert!((image.pixels[0][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_format() {
        let not_sfi = b"not a valid safetensors file";
        assert!(!is_sfi_format(not_sfi));
    }

    #[test]
    fn test_too_short() {
        let too_short = &[0u8; 4];
        assert!(!is_sfi_format(too_short));
    }
}
