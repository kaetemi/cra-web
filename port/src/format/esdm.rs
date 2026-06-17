//! ESD Core `.esdm` binary metadata sidecar encoding (BMP type).
//!
//! The `.esdm` file is a fixed-layout binary sidecar placed alongside an asset
//! file to provide metadata the asset data itself does not contain (dimensions,
//! EVE bitmap format, stride, palette size, ...). It is consumed at runtime by
//! ESD Core's `Esd_LoadResourceEx()`. This is distinct from the human-readable
//! resource-descriptor JSON written by `--output-meta` (which targets EVE Screen
//! Editor at authoring time).
//!
//! Only the BMP type is produced here, describing an uncompressed raw bitmap
//! (`compression = ESD_RESOURCE_RAW`). All multi-byte fields are little-endian.
//! See EXT_IMAGE_FORMATS.md for the full layout.

use super::color_format::ColorFormat;

/// `"BMP"` signature, NUL-terminated ASCII, little-endian (`0x00504D42`).
const SIGNATURE_BMP: u32 = 0x0050_4D42;
/// Metadata format version.
const VERSION: u8 = 1;
/// Total size of a BMP `.esdm` structure (12-byte header + 44-byte type data).
const BMP_SIZE: usize = 56;
/// `ESD_RESOURCE_RAW`: uncompressed asset data.
const COMPRESSION_RAW: u8 = 0;
/// EVE `PALETTEDARGB8` bitmap format value (BT820, 256-entry ARGB8888 palette).
const EVE_FORMAT_PALETTEDARGB8: u32 = 21;
/// Default palette sidecar filename suffix when none can be derived.
const DEFAULT_PALETTE_FILE_EXT: &str = ".pal.raw";
/// `paletteFileExt` field width in bytes.
const PALETTE_FILE_EXT_LEN: usize = 10;

/// Map a CRA [`ColorFormat`] to its EVE bitmap format enum value.
///
/// Returns `None` for bit-depth combinations that have no EVE equivalent
/// (CRA accepts arbitrary per-channel bit depths; EVE only defines a fixed set).
pub fn eve_bitmap_format(fmt: &ColorFormat) -> Option<u32> {
    let (r, g, b, a) = (fmt.bits_r, fmt.bits_g, fmt.bits_b, fmt.bits_a);
    if fmt.is_grayscale {
        if fmt.has_alpha {
            match (r, a) {
                (1, 1) => Some(24), // LA1
                (2, 2) => Some(25), // LA2
                (4, 4) => Some(26), // LA4
                (8, 8) => Some(27), // LA8
                _ => None,
            }
        } else {
            match r {
                1 => Some(1),  // L1
                2 => Some(17), // L2
                4 => Some(2),  // L4
                8 => Some(3),  // L8
                _ => None,
            }
        }
    } else if fmt.has_alpha {
        match (a, r, g, b) {
            (1, 5, 5, 5) => Some(0),  // ARGB1555
            (2, 2, 2, 2) => Some(5),  // ARGB2 (ARGB2222)
            (4, 4, 4, 4) => Some(6),  // ARGB4 (ARGB4444)
            (6, 6, 6, 6) => Some(23), // ARGB6
            (8, 8, 8, 8) => Some(20), // ARGB8 (ARGB8888)
            _ => None,
        }
    } else {
        match (r, g, b) {
            (3, 3, 2) => Some(4),  // RGB332
            (5, 6, 5) => Some(7),  // RGB565
            (6, 6, 6) => Some(22), // RGB6 (RGB666)
            (8, 8, 8) => Some(19), // RGB8 (RGB888)
            _ => None,
        }
    }
}

/// Fields needed to encode a BMP-type `.esdm` sidecar.
struct BmpFields {
    width: i32,
    height: i32,
    stride: i32,
    eve_format: u32,
    raw_size: u32,
    ext_len: u8,
    palette_size: u16,
    palette_file_ext: [u8; PALETTE_FILE_EXT_LEN],
    cells: u16,
    swizzle: u16,
}

/// Convert a suffix string into the fixed 10-byte NUL-padded `paletteFileExt`
/// field (truncated if longer than the field).
fn palette_ext_bytes(suffix: &str) -> [u8; PALETTE_FILE_EXT_LEN] {
    let mut out = [0u8; PALETTE_FILE_EXT_LEN];
    let bytes = suffix.as_bytes();
    let n = bytes.len().min(PALETTE_FILE_EXT_LEN);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

/// Encode the fixed 56-byte BMP `.esdm` layout (little-endian).
fn encode(f: &BmpFields) -> Vec<u8> {
    let mut buf = vec![0u8; BMP_SIZE];
    // Common header (12 bytes)
    buf[0..4].copy_from_slice(&SIGNATURE_BMP.to_le_bytes());
    buf[4] = VERSION;
    buf[5] = BMP_SIZE as u8;
    buf[6] = COMPRESSION_RAW;
    buf[7] = f.ext_len;
    buf[8..12].copy_from_slice(&f.raw_size.to_le_bytes());
    // BMP type-specific (44 bytes)
    buf[12..16].copy_from_slice(&f.width.to_le_bytes());
    buf[16..20].copy_from_slice(&f.height.to_le_bytes());
    buf[20..24].copy_from_slice(&f.stride.to_le_bytes());
    buf[24..28].copy_from_slice(&f.eve_format.to_le_bytes());
    buf[28..30].copy_from_slice(&f.palette_size.to_le_bytes());
    buf[30..40].copy_from_slice(&f.palette_file_ext);
    // 40..52 addtlResExt (12 bytes) left NUL.
    buf[52..54].copy_from_slice(&f.cells.to_le_bytes());
    buf[54..56].copy_from_slice(&f.swizzle.to_le_bytes());
    buf
}

/// Encode a BMP `.esdm` sidecar for a packed (non-paletted) raw bitmap.
///
/// - `stride` is the row stride in bytes of the raw file.
/// - `raw_size` is the total uncompressed byte size of the raw file.
/// - `ext_len` is the length of the raw file's extension including the dot
///   (e.g. 4 for `.raw`).
///
/// Returns `Err` if the format has no EVE bitmap format equivalent.
pub fn encode_esdm_bmp_from_format(
    fmt: &ColorFormat,
    width: u32,
    height: u32,
    stride: u32,
    raw_size: u32,
    ext_len: u8,
) -> Result<Vec<u8>, String> {
    let eve_format = eve_bitmap_format(fmt).ok_or_else(|| {
        format!(
            "Format {} ({} bpp) has no EVE bitmap format equivalent; cannot write .esdm metadata",
            fmt.name, fmt.total_bits
        )
    })?;
    Ok(encode(&BmpFields {
        width: width as i32,
        height: height as i32,
        stride: stride as i32,
        eve_format,
        raw_size,
        ext_len,
        palette_size: 0,
        palette_file_ext: [0u8; PALETTE_FILE_EXT_LEN],
        cells: 1,
        swizzle: 0,
    }))
}

/// Length of a filename's extension including the leading dot (`".bin"` -> 4).
/// Returns 0 when there is no extension (no dot, leading-dot only, or trailing dot).
pub fn extension_ext_len(filename: &str) -> u8 {
    match filename.rfind('.') {
        Some(i) if i > 0 && i + 1 < filename.len() => (filename.len() - i).min(u8::MAX as usize) as u8,
        _ => 0,
    }
}

/// Derive the `.esdm` `extLen` and `paletteFileExt` for a paletted bitmap from
/// the common prefix of the bitmap and palette filenames.
///
/// The base is the longest common prefix trimmed of trailing separators
/// (`_`, `-`, `.`); `extLen` is the bitmap's remaining suffix length and the
/// returned string is the palette's remaining suffix — so that
/// `palette = bitmap[..len - extLen] + paletteFileExt`. For example
/// `blahblah_index.bin` + `blahblah_lut.raw` -> base `blahblah`, extLen 10
/// (`_index.bin`), `_lut.raw`.
///
/// Falls back to the bitmap's plain extension and an empty palette suffix (which
/// the encoder turns into [`DEFAULT_PALETTE_FILE_EXT`]) when there is no usable
/// shared prefix or no palette filename.
pub fn derive_palette_naming(bitmap_name: &str, palette_name: Option<&str>) -> (u8, String) {
    let fallback = (extension_ext_len(bitmap_name), String::new());
    let Some(palette_name) = palette_name else {
        return fallback;
    };
    // Longest common prefix, tracked on char boundaries.
    let mut common = 0usize;
    for ((i, bc), pc) in bitmap_name.char_indices().zip(palette_name.chars()) {
        if bc != pc {
            break;
        }
        common = i + bc.len_utf8();
    }
    let base = bitmap_name[..common].trim_end_matches(['_', '-', '.']).len();
    if base == 0 || base >= bitmap_name.len() || base >= palette_name.len() {
        return fallback;
    }
    let ext_len = (bitmap_name.len() - base).min(u8::MAX as usize) as u8;
    (ext_len, palette_name[base..].to_string())
}

/// Encode a BMP `.esdm` sidecar for paletted output (8-bit indices into an
/// ARGB8888 palette, `PALETTEDARGB8`). `palette_count` is the number of colors;
/// the palette sidecar size is `palette_count * 4` bytes.
///
/// `palette_file_ext` is the palette sidecar's filename suffix (e.g. `.pal.raw`);
/// an empty string falls back to [`DEFAULT_PALETTE_FILE_EXT`].
pub fn encode_esdm_bmp_paletted(
    palette_count: u32,
    width: u32,
    height: u32,
    stride: u32,
    raw_size: u32,
    ext_len: u8,
    palette_file_ext: &str,
) -> Vec<u8> {
    let ext = if palette_file_ext.is_empty() {
        DEFAULT_PALETTE_FILE_EXT
    } else {
        palette_file_ext
    };
    encode(&BmpFields {
        width: width as i32,
        height: height as i32,
        stride: stride as i32,
        eve_format: EVE_FORMAT_PALETTEDARGB8,
        raw_size,
        ext_len,
        palette_size: (palette_count.saturating_mul(4)).min(u16::MAX as u32) as u16,
        palette_file_ext: palette_ext_bytes(ext),
        cells: 1,
        swizzle: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(name: &str) -> ColorFormat {
        ColorFormat::parse(name).unwrap()
    }

    #[test]
    fn eve_format_mapping() {
        assert_eq!(eve_bitmap_format(&fmt("RGB565")), Some(7));
        assert_eq!(eve_bitmap_format(&fmt("RGB332")), Some(4));
        assert_eq!(eve_bitmap_format(&fmt("RGB888")), Some(19));
        assert_eq!(eve_bitmap_format(&fmt("RGB666")), Some(22));
        assert_eq!(eve_bitmap_format(&fmt("ARGB1555")), Some(0));
        assert_eq!(eve_bitmap_format(&fmt("ARGB4444")), Some(6));
        assert_eq!(eve_bitmap_format(&fmt("ARGB8888")), Some(20));
        assert_eq!(eve_bitmap_format(&fmt("L1")), Some(1));
        assert_eq!(eve_bitmap_format(&fmt("L2")), Some(17));
        assert_eq!(eve_bitmap_format(&fmt("L4")), Some(2));
        assert_eq!(eve_bitmap_format(&fmt("L8")), Some(3));
        assert_eq!(eve_bitmap_format(&fmt("LA8")), Some(27));
        // No EVE equivalent for an arbitrary bit combination.
        assert_eq!(eve_bitmap_format(&fmt("RGB444")), None);
    }

    #[test]
    fn header_layout_rgb565() {
        // 128x64 RGB565, 2 bpp -> stride 256, raw_size 256*64.
        let stride = 256u32;
        let raw_size = stride * 64;
        let bytes =
            encode_esdm_bmp_from_format(&fmt("RGB565"), 128, 64, stride, raw_size, 4).unwrap();

        assert_eq!(bytes.len(), 56);
        // Signature "BMP\0".
        assert_eq!(&bytes[0..4], &[0x42, 0x4D, 0x50, 0x00]);
        assert_eq!(bytes[4], 1); // version
        assert_eq!(bytes[5], 56); // size
        assert_eq!(bytes[6], 0); // compression = RAW
        assert_eq!(bytes[7], 4); // extLen (".raw")
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), raw_size);
        assert_eq!(i32::from_le_bytes(bytes[12..16].try_into().unwrap()), 128);
        assert_eq!(i32::from_le_bytes(bytes[16..20].try_into().unwrap()), 64);
        assert_eq!(i32::from_le_bytes(bytes[20..24].try_into().unwrap()), 256);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 7); // RGB565
        assert_eq!(u16::from_le_bytes(bytes[28..30].try_into().unwrap()), 0); // paletteSize
        assert_eq!(u16::from_le_bytes(bytes[52..54].try_into().unwrap()), 1); // cells
        assert_eq!(u16::from_le_bytes(bytes[54..56].try_into().unwrap()), 0); // swizzle
    }

    #[test]
    fn paletted_layout() {
        let bytes = encode_esdm_bmp_paletted(16, 320, 200, 320, 320 * 200, 4, ".pal.raw");
        assert_eq!(bytes.len(), 56);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 21); // PALETTEDARGB8
        assert_eq!(u16::from_le_bytes(bytes[28..30].try_into().unwrap()), 64); // 16 * 4 bytes
        // paletteFileExt at offset 30, NUL-padded to 10 bytes.
        assert_eq!(&bytes[30..38], b".pal.raw");
        assert_eq!(&bytes[38..40], &[0, 0]);
    }

    #[test]
    fn paletted_default_ext_and_truncation() {
        // Empty suffix falls back to the default.
        let d = encode_esdm_bmp_paletted(4, 8, 8, 8, 64, 4, "");
        assert_eq!(&d[30..38], b".pal.raw");
        // Over-long suffix is truncated to the 10-byte field.
        let t = encode_esdm_bmp_paletted(4, 8, 8, 8, 64, 4, ".abcdefghijklmnop.raw");
        assert_eq!(&t[30..40], b".abcdefghi");
    }

    #[test]
    fn palette_naming_common_prefix() {
        // ESD converter convention: base "blahblah", role suffixes keep the separator.
        assert_eq!(
            derive_palette_naming("blahblah_index.bin", Some("blahblah_lut.raw")),
            (10, "_lut.raw".to_string())
        );
        // Simple shared-stem case.
        assert_eq!(
            derive_palette_naming("foo.raw", Some("foo.pal.raw")),
            (4, ".pal.raw".to_string())
        );
        assert_eq!(
            derive_palette_naming("img.raw", Some("img.lut.raw")),
            (4, ".lut.raw".to_string())
        );
        // No palette -> plain extension, empty suffix (encoder default).
        assert_eq!(derive_palette_naming("bar.raw", None), (4, String::new()));
        // No shared prefix -> fallback.
        assert_eq!(
            derive_palette_naming("a.raw", Some("palette.raw")),
            (4, String::new())
        );
    }

    #[test]
    fn extension_lengths() {
        assert_eq!(extension_ext_len("foo.raw"), 4);
        assert_eq!(extension_ext_len("blahblah_index.bin"), 4);
        assert_eq!(extension_ext_len("noext"), 0);
        assert_eq!(extension_ext_len(".hidden"), 0);
        assert_eq!(extension_ext_len("trailing."), 0);
    }

    #[test]
    fn unmappable_format_errors() {
        let stride = 2u32 * 16;
        assert!(encode_esdm_bmp_from_format(&fmt("RGB444"), 16, 16, stride, stride * 16, 4).is_err());
    }
}
