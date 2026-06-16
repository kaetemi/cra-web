//! LVGL binary image format (`.bin`) encoding.
//!
//! Produces LVGL's native binary image: a 12-byte header followed by an optional
//! palette (indexed formats), the pixel data, and an optional separate alpha
//! plane (`RGB565A8`). See EXT_IMAGE_FORMATS.md for the full layout.
//!
//! CRA already packs pixels little-endian with the most-significant channel in
//! the high bits, and sub-byte formats MSB-first — which is exactly LVGL's
//! memory convention (BGR/BGRA, little-endian RGB565, MSB-first indices). So for
//! most color formats the LVGL pixel plane is byte-identical to CRA's raw-binary
//! encoding; only indexed, `RGB565A8`, `RGB565_SWAPPED`, `XRGB8888` and the
//! premultiplied variant need bespoke packing.
//!
//! LVGL output is always byte-packed (`stride` = natural row stride, alignment 1)
//! and ignores the EVE-oriented `--stride`; the stride is written into the header.

use super::binary::{self, StrideFill};
use super::color_format::ColorFormat;

/// `LV_IMAGE_HEADER_MAGIC`.
const MAGIC: u8 = 0x19;
/// `LV_IMAGE_FLAGS_PREMULTIPLIED`.
const FLAG_PREMULTIPLIED: u16 = 0x0001;

/// LVGL color formats (`lv_color_format_t`) that CRA can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LvglColorFormat {
    L8,
    I1,
    I2,
    I4,
    I8,
    A1,
    A2,
    A4,
    A8,
    Rgb888,
    Argb8888,
    Xrgb8888,
    Rgb565,
    Rgb565a8,
    Al88,
    Argb1555,
    Argb4444,
    Argb2222,
    Argb8888Premultiplied,
    Rgb565Swapped,
}

impl LvglColorFormat {
    /// The `cf` enum value stored in the header.
    pub fn cf_value(self) -> u8 {
        use LvglColorFormat::*;
        match self {
            L8 => 0x06,
            I1 => 0x07,
            I2 => 0x08,
            I4 => 0x09,
            I8 => 0x0A,
            A1 => 0x0B,
            A2 => 0x0C,
            A4 => 0x0D,
            A8 => 0x0E,
            Rgb888 => 0x0F,
            Argb8888 => 0x10,
            Xrgb8888 => 0x11,
            Rgb565 => 0x12,
            Rgb565a8 => 0x14,
            Al88 => 0x15,
            Argb1555 => 0x16,
            Argb4444 => 0x17,
            Argb2222 => 0x18,
            Argb8888Premultiplied => 0x1A,
            Rgb565Swapped => 0x1B,
        }
    }

    /// Canonical uppercase name.
    pub fn name(self) -> &'static str {
        use LvglColorFormat::*;
        match self {
            L8 => "L8",
            I1 => "I1",
            I2 => "I2",
            I4 => "I4",
            I8 => "I8",
            A1 => "A1",
            A2 => "A2",
            A4 => "A4",
            A8 => "A8",
            Rgb888 => "RGB888",
            Argb8888 => "ARGB8888",
            Xrgb8888 => "XRGB8888",
            Rgb565 => "RGB565",
            Rgb565a8 => "RGB565A8",
            Al88 => "AL88",
            Argb1555 => "ARGB1555",
            Argb4444 => "ARGB4444",
            Argb2222 => "ARGB2222",
            Argb8888Premultiplied => "ARGB8888_PREMULTIPLIED",
            Rgb565Swapped => "RGB565_SWAPPED",
        }
    }

    /// Parse a color format by name (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        use LvglColorFormat::*;
        match s.to_uppercase().as_str() {
            "L8" => Some(L8),
            "I1" => Some(I1),
            "I2" => Some(I2),
            "I4" => Some(I4),
            "I8" => Some(I8),
            "A1" => Some(A1),
            "A2" => Some(A2),
            "A4" => Some(A4),
            "A8" => Some(A8),
            "RGB888" => Some(Rgb888),
            "ARGB8888" => Some(Argb8888),
            "XRGB8888" => Some(Xrgb8888),
            "RGB565" => Some(Rgb565),
            "RGB565A8" => Some(Rgb565a8),
            "AL88" => Some(Al88),
            "ARGB1555" => Some(Argb1555),
            "ARGB4444" => Some(Argb4444),
            "ARGB2222" => Some(Argb2222),
            "ARGB8888_PREMULTIPLIED" | "ARGB8888PREMULTIPLIED" => Some(Argb8888Premultiplied),
            "RGB565_SWAPPED" | "RGB565SWAPPED" => Some(Rgb565Swapped),
            _ => None,
        }
    }
}

/// Dithered pixel data to wrap into an LVGL `.bin`, mirroring the layout of the
/// CLI's dither result. `interleaved` is RGB (3/px), RGBA (4/px), grayscale
/// (1/px) or LA (2/px) per `is_grayscale`/`has_alpha`; channel bytes are
/// bit-replicated u8 at the indicated bit depths.
pub struct LvglSource<'a> {
    pub interleaved: &'a [u8],
    pub is_grayscale: bool,
    pub has_alpha: bool,
    pub bits_r: u8,
    pub bits_g: u8,
    pub bits_b: u8,
    pub bits_a: u8,
    pub palette_indices: Option<&'a [u8]>,
    pub palette_colors: Option<&'a [(u8, u8, u8, u8)]>,
}

/// Choose the canonical LVGL color format for a CRA [`ColorFormat`].
///
/// `paletted_colors` is the palette color count for paletted output (selects the
/// smallest indexed format that fits). Returns `None` for layouts with no
/// canonical LVGL mapping (the caller can still request one explicitly).
pub fn default_cf_for_format(
    fmt: &ColorFormat,
    paletted_colors: Option<usize>,
) -> Option<LvglColorFormat> {
    use LvglColorFormat::*;
    if let Some(n) = paletted_colors {
        return Some(if n <= 2 {
            I1
        } else if n <= 4 {
            I2
        } else if n <= 16 {
            I4
        } else {
            I8
        });
    }
    if fmt.is_grayscale {
        if fmt.has_alpha {
            // Only L8+A8 has a direct LVGL equivalent (AL88).
            if fmt.bits_r == 8 && fmt.bits_a == 8 {
                Some(Al88)
            } else {
                None
            }
        } else {
            // LVGL has no sub-byte luminance; 1/2/4-bit grayscale maps to alpha.
            match fmt.bits_r {
                8 => Some(L8),
                1 => Some(A1),
                2 => Some(A2),
                4 => Some(A4),
                _ => None,
            }
        }
    } else if fmt.has_alpha {
        match (fmt.bits_a, fmt.bits_r, fmt.bits_g, fmt.bits_b) {
            (1, 5, 5, 5) => Some(Argb1555),
            (4, 4, 4, 4) => Some(Argb4444),
            (2, 2, 2, 2) => Some(Argb2222),
            (8, 8, 8, 8) => Some(Argb8888),
            (8, 5, 6, 5) => Some(Rgb565a8),
            _ => None,
        }
    } else {
        match (fmt.bits_r, fmt.bits_g, fmt.bits_b) {
            (8, 8, 8) => Some(Rgb888),
            (5, 6, 5) => Some(Rgb565),
            _ => None,
        }
    }
}

/// Validate that a CRA [`ColorFormat`]'s bit layout can feed the given LVGL `cf`,
/// without needing pixel data. Used for fail-fast CLI validation before dithering.
///
/// `paletted_colors` is the palette color count when the output is paletted.
pub fn check_compatible(
    cf: LvglColorFormat,
    fmt: &ColorFormat,
    paletted_colors: Option<usize>,
) -> Result<(), String> {
    use LvglColorFormat::*;
    let rgb = |br, bg, bb| {
        !fmt.is_grayscale && !fmt.has_alpha && fmt.bits_r == br && fmt.bits_g == bg && fmt.bits_b == bb
    };
    let argb = |ba, br, bg, bb| {
        !fmt.is_grayscale
            && fmt.has_alpha
            && fmt.bits_a == ba
            && fmt.bits_r == br
            && fmt.bits_g == bg
            && fmt.bits_b == bb
    };
    let gray = |bits| fmt.is_grayscale && !fmt.has_alpha && fmt.bits_r == bits;

    let ok = match cf {
        L8 | A8 => gray(8),
        A1 => gray(1),
        A2 => gray(2),
        A4 => gray(4),
        Rgb888 => rgb(8, 8, 8),
        Rgb565 | Rgb565Swapped => rgb(5, 6, 5),
        Argb1555 => argb(1, 5, 5, 5),
        Argb4444 => argb(4, 4, 4, 4),
        Argb2222 => argb(2, 2, 2, 2),
        Argb8888 | Argb8888Premultiplied => argb(8, 8, 8, 8),
        Xrgb8888 => !fmt.is_grayscale && fmt.bits_r == 8 && fmt.bits_g == 8 && fmt.bits_b == 8,
        Al88 => fmt.is_grayscale && fmt.has_alpha && fmt.bits_r == 8 && fmt.bits_a == 8,
        Rgb565a8 => argb(8, 5, 6, 5),
        I1 | I2 | I4 | I8 => {
            let bits = match cf {
                I1 => 1,
                I2 => 2,
                I4 => 4,
                _ => 8,
            };
            return match paletted_colors {
                None => Err(format!(
                    "LVGL {} requires paletted output (--format PALETTE_* or --input-palette)",
                    cf.name()
                )),
                Some(n) if n > (1usize << bits) => Err(format!(
                    "LVGL {} supports up to {} palette entries; palette has {}",
                    cf.name(),
                    1usize << bits,
                    n
                )),
                Some(_) => Ok(()),
            };
        }
    };

    if ok {
        Ok(())
    } else {
        Err(format!(
            "LVGL format {} is not compatible with the dither output format {}",
            cf.name(),
            fmt.name
        ))
    }
}

/// Encode an LVGL `.bin` for the given color format and dithered source.
///
/// Validates that `src`'s bit layout is compatible with `cf` (the LVGL output
/// reinterprets a single dither pass rather than re-dithering).
pub fn encode_lvgl_bin(
    cf: LvglColorFormat,
    width: usize,
    height: usize,
    src: &LvglSource,
) -> Result<Vec<u8>, String> {
    use LvglColorFormat::*;

    if width == 0 || height == 0 {
        return Err("LVGL .bin requires non-zero dimensions".to_string());
    }
    if width > u16::MAX as usize || height > u16::MAX as usize {
        return Err(format!(
            "LVGL .bin dimensions are limited to {}x{}; got {}x{}",
            u16::MAX,
            u16::MAX,
            width,
            height
        ));
    }

    let mut flags: u16 = 0;

    // Build the post-header body (palette + pixels + optional alpha plane) and
    // the row stride to record in the header.
    let (body, stride) = match cf {
        L8 | A8 => {
            require_gray(src, 8, cf)?;
            let plane = binary::encode_gray_row_aligned_stride(
                src.interleaved, width, height, 8, 1, StrideFill::Black,
            );
            (plane, width)
        }
        A1 | A2 | A4 => {
            let bits = match cf {
                A1 => 1,
                A2 => 2,
                _ => 4,
            };
            require_gray(src, bits, cf)?;
            let plane = binary::encode_gray_row_aligned_stride(
                src.interleaved, width, height, bits, 1, StrideFill::Black,
            );
            (plane, (width * bits as usize + 7) / 8)
        }
        Rgb888 => {
            require_rgb(src, 8, 8, 8, cf)?;
            let plane = binary::encode_rgb_row_aligned_stride(
                src.interleaved, width, height, 8, 8, 8, 1, StrideFill::Black,
            );
            (plane, width * 3)
        }
        Rgb565 => {
            require_rgb(src, 5, 6, 5, cf)?;
            let plane = binary::encode_rgb_row_aligned_stride(
                src.interleaved, width, height, 5, 6, 5, 1, StrideFill::Black,
            );
            (plane, width * 2)
        }
        Rgb565Swapped => {
            require_rgb(src, 5, 6, 5, cf)?;
            let mut plane = binary::encode_rgb_row_aligned_stride(
                src.interleaved, width, height, 5, 6, 5, 1, StrideFill::Black,
            );
            for px in plane.chunks_exact_mut(2) {
                px.swap(0, 1);
            }
            (plane, width * 2)
        }
        Argb1555 => {
            require_argb(src, 1, 5, 5, 5, cf)?;
            let plane = binary::encode_argb_row_aligned_stride(
                src.interleaved, width, height, 1, 5, 5, 5, 1, StrideFill::Black,
            );
            (plane, width * 2)
        }
        Argb4444 => {
            require_argb(src, 4, 4, 4, 4, cf)?;
            let plane = binary::encode_argb_row_aligned_stride(
                src.interleaved, width, height, 4, 4, 4, 4, 1, StrideFill::Black,
            );
            (plane, width * 2)
        }
        Argb2222 => {
            require_argb(src, 2, 2, 2, 2, cf)?;
            let plane = binary::encode_argb_row_aligned_stride(
                src.interleaved, width, height, 2, 2, 2, 2, 1, StrideFill::Black,
            );
            (plane, width)
        }
        Argb8888 => {
            require_argb(src, 8, 8, 8, 8, cf)?;
            let plane = binary::encode_argb_row_aligned_stride(
                src.interleaved, width, height, 8, 8, 8, 8, 1, StrideFill::Black,
            );
            (plane, width * 4)
        }
        Argb8888Premultiplied => {
            require_argb(src, 8, 8, 8, 8, cf)?;
            let premul = premultiply_rgba(src.interleaved);
            let plane = binary::encode_argb_row_aligned_stride(
                &premul, width, height, 8, 8, 8, 8, 1, StrideFill::Black,
            );
            flags |= FLAG_PREMULTIPLIED;
            (plane, width * 4)
        }
        Xrgb8888 => {
            // Accepts RGB888 or ARGB8888 dither output; alpha byte forced to 0xFF.
            if src.is_grayscale || src.bits_r != 8 || src.bits_g != 8 || src.bits_b != 8 {
                return Err(format!(
                    "LVGL {} requires RGB888 or ARGB8888 dither output (--format RGB888 or ARGB8888)",
                    cf.name()
                ));
            }
            (build_xrgb(src, width, height), width * 4)
        }
        Al88 => {
            require_la(src, 8, 8, cf)?;
            let plane = binary::encode_la_row_aligned_stride(
                src.interleaved, width, height, 8, 8, 1, StrideFill::Black,
            );
            (plane, width * 2)
        }
        Rgb565a8 => {
            // From CRA --format ARGB8565: RGB dithered to 565, alpha to 8 bits.
            require_argb(src, 8, 5, 6, 5, cf)?;
            let (rgb565, a8) = split_rgb565a8(src.interleaved, width, height);
            let mut body = rgb565;
            body.extend_from_slice(&a8);
            (body, width * 2)
        }
        I1 | I2 | I4 | I8 => {
            let bits: u8 = match cf {
                I1 => 1,
                I2 => 2,
                I4 => 4,
                _ => 8,
            };
            let indices = src.palette_indices.ok_or_else(|| {
                format!(
                    "LVGL {} requires paletted output (--format PALETTE_* or --input-palette)",
                    cf.name()
                )
            })?;
            let colors = src.palette_colors.ok_or_else(|| {
                format!("LVGL {} requires a palette", cf.name())
            })?;
            let entries = 1usize << bits;
            if colors.len() > entries {
                return Err(format!(
                    "LVGL {} supports up to {} palette entries; palette has {}",
                    cf.name(),
                    entries,
                    colors.len()
                ));
            }
            // Palette: `entries` lv_color32_t in BGRA order, zero-padded.
            let mut body = Vec::with_capacity(entries * 4 + indices.len());
            for i in 0..entries {
                let (r, g, b, a) = colors.get(i).copied().unwrap_or((0, 0, 0, 0));
                body.push(b);
                body.push(g);
                body.push(r);
                body.push(a);
            }
            let plane = pack_indices(indices, width, height, bits);
            body.extend_from_slice(&plane);
            (body, (width * bits as usize + 7) / 8)
        }
    };

    if stride > u16::MAX as usize {
        return Err(format!("LVGL .bin row stride {} exceeds {}", stride, u16::MAX));
    }

    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&lvgl_header(
        cf.cf_value(),
        flags,
        width as u16,
        height as u16,
        stride as u16,
    ));
    out.extend_from_slice(&body);
    Ok(out)
}

/// Build the 12-byte LVGL image header (little-endian).
fn lvgl_header(cf: u8, flags: u16, width: u16, height: u16, stride: u16) -> [u8; 12] {
    let w0: u32 = (MAGIC as u32) | ((cf as u32) << 8) | ((flags as u32) << 16);
    let w1: u32 = (width as u32) | ((height as u32) << 16);
    let w2: u32 = stride as u32; // upper 16 bits reserved (0)
    let mut h = [0u8; 12];
    h[0..4].copy_from_slice(&w0.to_le_bytes());
    h[4..8].copy_from_slice(&w1.to_le_bytes());
    h[8..12].copy_from_slice(&w2.to_le_bytes());
    h
}

fn require_gray(src: &LvglSource, bits: u8, cf: LvglColorFormat) -> Result<(), String> {
    if src.is_grayscale && !src.has_alpha && src.bits_r == bits {
        Ok(())
    } else {
        Err(format!(
            "LVGL {} requires {}-bit grayscale dither output (--format L{})",
            cf.name(),
            bits,
            bits
        ))
    }
}

fn require_la(src: &LvglSource, bits_l: u8, bits_a: u8, cf: LvglColorFormat) -> Result<(), String> {
    if src.is_grayscale && src.has_alpha && src.bits_r == bits_l && src.bits_a == bits_a {
        Ok(())
    } else {
        Err(format!(
            "LVGL {} requires LA{}{} dither output (--format LA{}{})",
            cf.name(),
            bits_l,
            bits_a,
            bits_l,
            bits_a
        ))
    }
}

fn require_rgb(
    src: &LvglSource,
    br: u8,
    bg: u8,
    bb: u8,
    cf: LvglColorFormat,
) -> Result<(), String> {
    if !src.is_grayscale && !src.has_alpha && src.bits_r == br && src.bits_g == bg && src.bits_b == bb
    {
        Ok(())
    } else {
        Err(format!(
            "LVGL {} requires RGB{}{}{} dither output (--format RGB{}{}{})",
            cf.name(),
            br,
            bg,
            bb,
            br,
            bg,
            bb
        ))
    }
}

fn require_argb(
    src: &LvglSource,
    ba: u8,
    br: u8,
    bg: u8,
    bb: u8,
    cf: LvglColorFormat,
) -> Result<(), String> {
    if !src.is_grayscale
        && src.has_alpha
        && src.bits_a == ba
        && src.bits_r == br
        && src.bits_g == bg
        && src.bits_b == bb
    {
        Ok(())
    } else {
        Err(format!(
            "LVGL {} requires ARGB{}{}{}{} dither output (--format ARGB{}{}{}{})",
            cf.name(),
            ba,
            br,
            bg,
            bb,
            ba,
            br,
            bg,
            bb
        ))
    }
}

/// Straight-alpha RGBA -> premultiplied RGBA (alpha unchanged).
fn premultiply_rgba(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for px in data.chunks_exact(4) {
        let a = px[3] as u32;
        let m = |c: u8| ((c as u32 * a + 127) / 255) as u8;
        out.push(m(px[0]));
        out.push(m(px[1]));
        out.push(m(px[2]));
        out.push(px[3]);
    }
    out
}

/// Build a BGRX (X=0xFF) plane from RGB888 or ARGB8888 interleaved data.
fn build_xrgb(src: &LvglSource, width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 4);
    if src.has_alpha {
        for px in src.interleaved.chunks_exact(4) {
            out.push(px[2]);
            out.push(px[1]);
            out.push(px[0]);
            out.push(0xFF);
        }
    } else {
        for px in src.interleaved.chunks_exact(3) {
            out.push(px[2]);
            out.push(px[1]);
            out.push(px[0]);
            out.push(0xFF);
        }
    }
    out
}

/// Split ARGB8565 (RGBA interleaved, RGB dithered to 565, A to 8) into a packed
/// RGB565 plane followed by a separate A8 plane.
fn split_rgb565a8(rgba: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let mut rgb565 = Vec::with_capacity(width * height * 2);
    let mut a8 = Vec::with_capacity(width * height);
    for px in rgba.chunks_exact(4) {
        let val = binary::encode_rgb_pixel(px[0], px[1], px[2], 5, 6, 5);
        rgb565.push((val & 0xFF) as u8);
        rgb565.push(((val >> 8) & 0xFF) as u8);
        a8.push(px[3]);
    }
    (rgb565, a8)
}

/// Pack raw palette indices into `bits`-wide values, MSB-first within each byte,
/// row by row (byte-aligned rows, no extra padding).
fn pack_indices(indices: &[u8], width: usize, height: usize, bits: u8) -> Vec<u8> {
    if bits == 8 {
        return indices.to_vec();
    }
    let mask: u8 = ((1u16 << bits) - 1) as u8;
    let row_bytes = (width * bits as usize + 7) / 8;
    let mut out = Vec::with_capacity(row_bytes * height);
    for y in 0..height {
        let mut cur: u8 = 0;
        let mut filled: usize = 0;
        for x in 0..width {
            let idx = indices[y * width + x] & mask;
            let shift = 8 - filled - bits as usize;
            cur |= idx << shift;
            filled += bits as usize;
            if filled == 8 {
                out.push(cur);
                cur = 0;
                filled = 0;
            }
        }
        if filled > 0 {
            out.push(cur);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_src<'a>(data: &'a [u8], br: u8, bg: u8, bb: u8) -> LvglSource<'a> {
        LvglSource {
            interleaved: data,
            is_grayscale: false,
            has_alpha: false,
            bits_r: br,
            bits_g: bg,
            bits_b: bb,
            bits_a: 0,
            palette_indices: None,
            palette_colors: None,
        }
    }

    fn argb_src<'a>(data: &'a [u8], ba: u8, br: u8, bg: u8, bb: u8) -> LvglSource<'a> {
        LvglSource {
            interleaved: data,
            is_grayscale: false,
            has_alpha: true,
            bits_r: br,
            bits_g: bg,
            bits_b: bb,
            bits_a: ba,
            palette_indices: None,
            palette_colors: None,
        }
    }

    #[test]
    fn header_rgb565() {
        // 2x1 RGB565.
        let data = [255, 0, 0, 0, 0, 255]; // red, blue
        let bin = encode_lvgl_bin(LvglColorFormat::Rgb565, 2, 1, &rgb_src(&data, 5, 6, 5)).unwrap();
        assert_eq!(bin.len(), 12 + 2 * 2);
        // Word 0: magic | cf<<8.
        assert_eq!(bin[0], 0x19);
        assert_eq!(bin[1], 0x12); // RGB565
        assert_eq!(u16::from_le_bytes([bin[2], bin[3]]), 0); // flags
        assert_eq!(u16::from_le_bytes([bin[4], bin[5]]), 2); // width
        assert_eq!(u16::from_le_bytes([bin[6], bin[7]]), 1); // height
        assert_eq!(u16::from_le_bytes([bin[8], bin[9]]), 4); // stride = 2px * 2 bytes
        // Red pixel -> R=0x1F<<11 = 0xF800 -> LE [0x00, 0xF8].
        assert_eq!(bin[12], 0x00);
        assert_eq!(bin[13], 0xF8);
    }

    #[test]
    fn argb8888_is_bgra() {
        let data = [0x11, 0x22, 0x33, 0x44]; // R,G,B,A
        let bin =
            encode_lvgl_bin(LvglColorFormat::Argb8888, 1, 1, &argb_src(&data, 8, 8, 8, 8)).unwrap();
        assert_eq!(bin[1], 0x10);
        // BGRA in memory.
        assert_eq!(&bin[12..16], &[0x33, 0x22, 0x11, 0x44]);
    }

    #[test]
    fn rgb565_swapped_swaps_bytes() {
        let data = [255, 0, 0]; // red
        let plain =
            encode_lvgl_bin(LvglColorFormat::Rgb565, 1, 1, &rgb_src(&data, 5, 6, 5)).unwrap();
        let swapped =
            encode_lvgl_bin(LvglColorFormat::Rgb565Swapped, 1, 1, &rgb_src(&data, 5, 6, 5)).unwrap();
        assert_eq!(swapped[1], 0x1B);
        assert_eq!(plain[12], swapped[13]);
        assert_eq!(plain[13], swapped[12]);
    }

    #[test]
    fn xrgb_forces_opaque() {
        let data = [0x11, 0x22, 0x33]; // RGB888
        let bin = encode_lvgl_bin(LvglColorFormat::Xrgb8888, 1, 1, &rgb_src(&data, 8, 8, 8)).unwrap();
        assert_eq!(bin[1], 0x11);
        assert_eq!(&bin[12..16], &[0x33, 0x22, 0x11, 0xFF]); // BGRX
    }

    #[test]
    fn premultiplied_sets_flag() {
        let data = [0xFF, 0xFF, 0xFF, 0x80]; // white, 50% alpha
        let bin = encode_lvgl_bin(
            LvglColorFormat::Argb8888Premultiplied,
            1,
            1,
            &argb_src(&data, 8, 8, 8, 8),
        )
        .unwrap();
        assert_eq!(bin[1], 0x1A);
        assert_eq!(u16::from_le_bytes([bin[2], bin[3]]), FLAG_PREMULTIPLIED);
        // 0xFF * 0x80 / 255 = 128.
        assert_eq!(&bin[12..16], &[0x80, 0x80, 0x80, 0x80]);
    }

    #[test]
    fn indexed_i4_palette_and_packing() {
        let indices = [0u8, 1, 2, 15];
        let colors = vec![(0u8, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255)];
        let src = LvglSource {
            interleaved: &[],
            is_grayscale: false,
            has_alpha: false,
            bits_r: 0,
            bits_g: 0,
            bits_b: 0,
            bits_a: 0,
            palette_indices: Some(&indices),
            palette_colors: Some(&colors),
        };
        let bin = encode_lvgl_bin(LvglColorFormat::I4, 4, 1, &src).unwrap();
        assert_eq!(bin[1], 0x09);
        // 16 palette entries (BGRA) + ceil(4*4/8)=2 index bytes.
        assert_eq!(bin.len(), 12 + 16 * 4 + 2);
        // First palette entry (0,0,0,255) -> BGRA.
        assert_eq!(&bin[12..16], &[0, 0, 0, 255]);
        // Second entry (255,0,0,255) -> B=0,G=0,R=255,A=255.
        assert_eq!(&bin[16..20], &[0, 0, 255, 255]);
        // Indices 0,1 -> 0x01 ; 2,15 -> 0x2F (MSB-first nibbles).
        let idx_start = 12 + 16 * 4;
        assert_eq!(bin[idx_start], 0x01);
        assert_eq!(bin[idx_start + 1], 0x2F);
    }

    #[test]
    fn rgb565a8_two_planes() {
        // 1x1 ARGB8565.
        let data = [255, 0, 0, 0x80]; // red, 50% alpha
        let bin = encode_lvgl_bin(LvglColorFormat::Rgb565a8, 1, 1, &argb_src(&data, 8, 5, 6, 5))
            .unwrap();
        assert_eq!(bin[1], 0x14);
        assert_eq!(u16::from_le_bytes([bin[8], bin[9]]), 2); // stride = RGB565 plane
        assert_eq!(bin.len(), 12 + 2 + 1); // 565 plane + a8 plane
        assert_eq!(bin[12], 0x00);
        assert_eq!(bin[13], 0xF8); // red 565
        assert_eq!(bin[14], 0x80); // alpha plane
    }

    #[test]
    fn validation_rejects_mismatched_format() {
        let data = [255, 0, 0]; // RGB888-shaped
        let err = encode_lvgl_bin(LvglColorFormat::Rgb565, 1, 1, &rgb_src(&data, 8, 8, 8));
        assert!(err.is_err());
    }

    #[test]
    fn default_cf_mapping() {
        let f = |n: &str| ColorFormat::parse(n).unwrap();
        assert_eq!(default_cf_for_format(&f("RGB565"), None), Some(LvglColorFormat::Rgb565));
        assert_eq!(default_cf_for_format(&f("ARGB8888"), None), Some(LvglColorFormat::Argb8888));
        assert_eq!(default_cf_for_format(&f("L8"), None), Some(LvglColorFormat::L8));
        assert_eq!(default_cf_for_format(&f("LA8"), None), Some(LvglColorFormat::Al88));
        assert_eq!(default_cf_for_format(&f("ARGB8565"), None), Some(LvglColorFormat::Rgb565a8));
        assert_eq!(default_cf_for_format(&f("RGB888"), Some(16)), Some(LvglColorFormat::I4));
        assert_eq!(default_cf_for_format(&f("RGB888"), Some(2)), Some(LvglColorFormat::I1));
    }
}
