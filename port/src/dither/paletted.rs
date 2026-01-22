/// Palette-based RGBA dithering with integrated alpha-RGB distance metric.
///
/// Unlike the standard RGBA dithering which pre-dithers alpha separately, this variant:
/// - Takes an arbitrary palette of up to 256 RGBA colors
/// - Integrates alpha into the main distance metric
/// - Uses combined distance: sqrt(alpha_dist² + (rgb_perceptual_dist × alpha_factor)²)
///
/// This approach weighs down RGB distance where pixels are less visible (low alpha),
/// making the dithering focus on alpha accuracy for transparent regions and RGB
/// accuracy for opaque regions.

use crate::color::{
    linear_rgb_to_lab, linear_rgb_to_oklab, linear_rgb_to_ycbcr, linear_rgb_to_ycbcr_clamped,
    linear_to_srgb_single, srgb_to_linear_single,
};
use crate::color_distance::{
    is_lab_space, is_linear_rgb_space, is_srgb_space, is_ycbcr_space, perceptual_distance_sq,
};
use super::common::{wang_hash, DitherMode, PerceptualSpace};

// ============================================================================
// Palette structures
// ============================================================================

/// A single palette entry with precomputed values for efficient distance calculation.
#[derive(Clone, Copy)]
struct PaletteEntry {
    /// sRGB output values (0-255)
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    /// Linear RGB values for error calculation
    lin_r: f32,
    lin_g: f32,
    lin_b: f32,
    lin_a: f32,
    /// Perceptual space coordinates (L/a/b or similar depending on space)
    perc_l: f32,
    perc_a: f32,
    perc_b: f32,
}

/// Precomputed palette for efficient dithering.
pub struct DitherPalette {
    entries: Vec<PaletteEntry>,
    space: PerceptualSpace,
}

impl DitherPalette {
    /// Create a new dither palette from RGBA colors.
    ///
    /// Args:
    ///     colors: Slice of (R, G, B, A) tuples in sRGB 0-255 range
    ///     space: Perceptual color space for distance calculation
    ///
    /// Panics if more than 256 colors are provided.
    pub fn new(colors: &[(u8, u8, u8, u8)], space: PerceptualSpace) -> Self {
        assert!(colors.len() <= 256, "Palette cannot exceed 256 colors");
        assert!(!colors.is_empty(), "Palette cannot be empty");

        let entries: Vec<PaletteEntry> = colors.iter().map(|&(r, g, b, a)| {
            // Convert to linear for error calculation
            let lin_r = srgb_to_linear_single(r as f32 / 255.0);
            let lin_g = srgb_to_linear_single(g as f32 / 255.0);
            let lin_b = srgb_to_linear_single(b as f32 / 255.0);
            let lin_a = a as f32 / 255.0; // Alpha is already linear

            // Convert to perceptual space
            let (perc_l, perc_a, perc_b) = if is_srgb_space(space) {
                (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
            } else if is_linear_rgb_space(space) {
                (lin_r, lin_g, lin_b)
            } else if is_ycbcr_space(space) {
                linear_rgb_to_ycbcr_clamped(lin_r, lin_g, lin_b)
            } else if is_lab_space(space) {
                linear_rgb_to_lab(lin_r, lin_g, lin_b)
            } else {
                linear_rgb_to_oklab(lin_r, lin_g, lin_b)
            };

            PaletteEntry {
                r, g, b, a,
                lin_r, lin_g, lin_b, lin_a,
                perc_l, perc_a, perc_b,
            }
        }).collect();

        Self { entries, space }
    }

    /// Create a palette from interleaved RGBA u8 data.
    pub fn from_rgba_bytes(data: &[u8], space: PerceptualSpace) -> Self {
        assert!(data.len() % 4 == 0, "RGBA data must be multiple of 4 bytes");
        let colors: Vec<(u8, u8, u8, u8)> = data.chunks_exact(4)
            .map(|c| (c[0], c[1], c[2], c[3]))
            .collect();
        Self::new(&colors, space)
    }

    /// Number of colors in the palette.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if palette is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================================
// Dither Kernel trait and implementations
// ============================================================================

trait DitherKernel {
    const REACH: usize;

    fn apply_ltr(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>], err_a: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32, err_a_val: f32,
    );

    fn apply_rtl(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>], err_a: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32, err_a_val: f32,
    );
}

struct FloydSteinberg;

impl DitherKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>], err_a: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32, err_a_val: f32,
    ) {
        // 7/16 to right
        err_r[y][bx + 1] += err_r_val * (7.0 / 16.0);
        err_g[y][bx + 1] += err_g_val * (7.0 / 16.0);
        err_b[y][bx + 1] += err_b_val * (7.0 / 16.0);
        err_a[y][bx + 1] += err_a_val * (7.0 / 16.0);

        // 3/16 to bottom-left
        err_r[y + 1][bx - 1] += err_r_val * (3.0 / 16.0);
        err_g[y + 1][bx - 1] += err_g_val * (3.0 / 16.0);
        err_b[y + 1][bx - 1] += err_b_val * (3.0 / 16.0);
        err_a[y + 1][bx - 1] += err_a_val * (3.0 / 16.0);

        // 5/16 to bottom
        err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
        err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
        err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);
        err_a[y + 1][bx] += err_a_val * (5.0 / 16.0);

        // 1/16 to bottom-right
        err_r[y + 1][bx + 1] += err_r_val * (1.0 / 16.0);
        err_g[y + 1][bx + 1] += err_g_val * (1.0 / 16.0);
        err_b[y + 1][bx + 1] += err_b_val * (1.0 / 16.0);
        err_a[y + 1][bx + 1] += err_a_val * (1.0 / 16.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>], err_a: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32, err_a_val: f32,
    ) {
        err_r[y][bx - 1] += err_r_val * (7.0 / 16.0);
        err_g[y][bx - 1] += err_g_val * (7.0 / 16.0);
        err_b[y][bx - 1] += err_b_val * (7.0 / 16.0);
        err_a[y][bx - 1] += err_a_val * (7.0 / 16.0);

        err_r[y + 1][bx + 1] += err_r_val * (3.0 / 16.0);
        err_g[y + 1][bx + 1] += err_g_val * (3.0 / 16.0);
        err_b[y + 1][bx + 1] += err_b_val * (3.0 / 16.0);
        err_a[y + 1][bx + 1] += err_a_val * (3.0 / 16.0);

        err_r[y + 1][bx] += err_r_val * (5.0 / 16.0);
        err_g[y + 1][bx] += err_g_val * (5.0 / 16.0);
        err_b[y + 1][bx] += err_b_val * (5.0 / 16.0);
        err_a[y + 1][bx] += err_a_val * (5.0 / 16.0);

        err_r[y + 1][bx - 1] += err_r_val * (1.0 / 16.0);
        err_g[y + 1][bx - 1] += err_g_val * (1.0 / 16.0);
        err_b[y + 1][bx - 1] += err_b_val * (1.0 / 16.0);
        err_a[y + 1][bx - 1] += err_a_val * (1.0 / 16.0);
    }
}

struct JarvisJudiceNinke;

impl DitherKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>], err_a: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32, err_a_val: f32,
    ) {
        // Row 0
        err_r[y][bx + 1] += err_r_val * (7.0 / 48.0);
        err_g[y][bx + 1] += err_g_val * (7.0 / 48.0);
        err_b[y][bx + 1] += err_b_val * (7.0 / 48.0);
        err_a[y][bx + 1] += err_a_val * (7.0 / 48.0);
        err_r[y][bx + 2] += err_r_val * (5.0 / 48.0);
        err_g[y][bx + 2] += err_g_val * (5.0 / 48.0);
        err_b[y][bx + 2] += err_b_val * (5.0 / 48.0);
        err_a[y][bx + 2] += err_a_val * (5.0 / 48.0);

        // Row 1
        err_r[y + 1][bx - 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx - 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx - 2] += err_b_val * (3.0 / 48.0);
        err_a[y + 1][bx - 2] += err_a_val * (3.0 / 48.0);
        err_r[y + 1][bx - 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx - 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx - 1] += err_b_val * (5.0 / 48.0);
        err_a[y + 1][bx - 1] += err_a_val * (5.0 / 48.0);
        err_r[y + 1][bx] += err_r_val * (7.0 / 48.0);
        err_g[y + 1][bx] += err_g_val * (7.0 / 48.0);
        err_b[y + 1][bx] += err_b_val * (7.0 / 48.0);
        err_a[y + 1][bx] += err_a_val * (7.0 / 48.0);
        err_r[y + 1][bx + 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx + 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx + 1] += err_b_val * (5.0 / 48.0);
        err_a[y + 1][bx + 1] += err_a_val * (5.0 / 48.0);
        err_r[y + 1][bx + 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx + 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx + 2] += err_b_val * (3.0 / 48.0);
        err_a[y + 1][bx + 2] += err_a_val * (3.0 / 48.0);

        // Row 2
        err_r[y + 2][bx - 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx - 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx - 2] += err_b_val * (1.0 / 48.0);
        err_a[y + 2][bx - 2] += err_a_val * (1.0 / 48.0);
        err_r[y + 2][bx - 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx - 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx - 1] += err_b_val * (3.0 / 48.0);
        err_a[y + 2][bx - 1] += err_a_val * (3.0 / 48.0);
        err_r[y + 2][bx] += err_r_val * (5.0 / 48.0);
        err_g[y + 2][bx] += err_g_val * (5.0 / 48.0);
        err_b[y + 2][bx] += err_b_val * (5.0 / 48.0);
        err_a[y + 2][bx] += err_a_val * (5.0 / 48.0);
        err_r[y + 2][bx + 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx + 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx + 1] += err_b_val * (3.0 / 48.0);
        err_a[y + 2][bx + 1] += err_a_val * (3.0 / 48.0);
        err_r[y + 2][bx + 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx + 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx + 2] += err_b_val * (1.0 / 48.0);
        err_a[y + 2][bx + 2] += err_a_val * (1.0 / 48.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>], err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>], err_a: &mut [Vec<f32>],
        bx: usize, y: usize,
        err_r_val: f32, err_g_val: f32, err_b_val: f32, err_a_val: f32,
    ) {
        // Row 0
        err_r[y][bx - 1] += err_r_val * (7.0 / 48.0);
        err_g[y][bx - 1] += err_g_val * (7.0 / 48.0);
        err_b[y][bx - 1] += err_b_val * (7.0 / 48.0);
        err_a[y][bx - 1] += err_a_val * (7.0 / 48.0);
        err_r[y][bx - 2] += err_r_val * (5.0 / 48.0);
        err_g[y][bx - 2] += err_g_val * (5.0 / 48.0);
        err_b[y][bx - 2] += err_b_val * (5.0 / 48.0);
        err_a[y][bx - 2] += err_a_val * (5.0 / 48.0);

        // Row 1
        err_r[y + 1][bx + 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx + 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx + 2] += err_b_val * (3.0 / 48.0);
        err_a[y + 1][bx + 2] += err_a_val * (3.0 / 48.0);
        err_r[y + 1][bx + 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx + 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx + 1] += err_b_val * (5.0 / 48.0);
        err_a[y + 1][bx + 1] += err_a_val * (5.0 / 48.0);
        err_r[y + 1][bx] += err_r_val * (7.0 / 48.0);
        err_g[y + 1][bx] += err_g_val * (7.0 / 48.0);
        err_b[y + 1][bx] += err_b_val * (7.0 / 48.0);
        err_a[y + 1][bx] += err_a_val * (7.0 / 48.0);
        err_r[y + 1][bx - 1] += err_r_val * (5.0 / 48.0);
        err_g[y + 1][bx - 1] += err_g_val * (5.0 / 48.0);
        err_b[y + 1][bx - 1] += err_b_val * (5.0 / 48.0);
        err_a[y + 1][bx - 1] += err_a_val * (5.0 / 48.0);
        err_r[y + 1][bx - 2] += err_r_val * (3.0 / 48.0);
        err_g[y + 1][bx - 2] += err_g_val * (3.0 / 48.0);
        err_b[y + 1][bx - 2] += err_b_val * (3.0 / 48.0);
        err_a[y + 1][bx - 2] += err_a_val * (3.0 / 48.0);

        // Row 2
        err_r[y + 2][bx + 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx + 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx + 2] += err_b_val * (1.0 / 48.0);
        err_a[y + 2][bx + 2] += err_a_val * (1.0 / 48.0);
        err_r[y + 2][bx + 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx + 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx + 1] += err_b_val * (3.0 / 48.0);
        err_a[y + 2][bx + 1] += err_a_val * (3.0 / 48.0);
        err_r[y + 2][bx] += err_r_val * (5.0 / 48.0);
        err_g[y + 2][bx] += err_g_val * (5.0 / 48.0);
        err_b[y + 2][bx] += err_b_val * (5.0 / 48.0);
        err_a[y + 2][bx] += err_a_val * (5.0 / 48.0);
        err_r[y + 2][bx - 1] += err_r_val * (3.0 / 48.0);
        err_g[y + 2][bx - 1] += err_g_val * (3.0 / 48.0);
        err_b[y + 2][bx - 1] += err_b_val * (3.0 / 48.0);
        err_a[y + 2][bx - 1] += err_a_val * (3.0 / 48.0);
        err_r[y + 2][bx - 2] += err_r_val * (1.0 / 48.0);
        err_g[y + 2][bx - 2] += err_g_val * (1.0 / 48.0);
        err_b[y + 2][bx - 2] += err_b_val * (1.0 / 48.0);
        err_a[y + 2][bx - 2] += err_a_val * (1.0 / 48.0);
    }
}

struct NoneKernel;

impl DitherKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(
        _err_r: &mut [Vec<f32>], _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>], _err_a: &mut [Vec<f32>],
        _bx: usize, _y: usize,
        _err_r_val: f32, _err_g_val: f32, _err_b_val: f32, _err_a_val: f32,
    ) {}

    #[inline]
    fn apply_rtl(
        _err_r: &mut [Vec<f32>], _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>], _err_a: &mut [Vec<f32>],
        _bx: usize, _y: usize,
        _err_r_val: f32, _err_g_val: f32, _err_b_val: f32, _err_a_val: f32,
    ) {}
}

// ============================================================================
// Mixed kernel helpers
// ============================================================================

#[inline]
fn apply_single_channel_kernel(
    err: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_val: f32,
    use_jjn: bool,
    is_rtl: bool,
) {
    match (use_jjn, is_rtl) {
        (true, false) => {
            // JJN LTR
            err[y][bx + 1] += err_val * (7.0 / 48.0);
            err[y][bx + 2] += err_val * (5.0 / 48.0);
            err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
            err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx] += err_val * (7.0 / 48.0);
            err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
            err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
            err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx] += err_val * (5.0 / 48.0);
            err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
        }
        (true, true) => {
            // JJN RTL
            err[y][bx - 1] += err_val * (7.0 / 48.0);
            err[y][bx - 2] += err_val * (5.0 / 48.0);
            err[y + 1][bx + 2] += err_val * (3.0 / 48.0);
            err[y + 1][bx + 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx] += err_val * (7.0 / 48.0);
            err[y + 1][bx - 1] += err_val * (5.0 / 48.0);
            err[y + 1][bx - 2] += err_val * (3.0 / 48.0);
            err[y + 2][bx + 2] += err_val * (1.0 / 48.0);
            err[y + 2][bx + 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx] += err_val * (5.0 / 48.0);
            err[y + 2][bx - 1] += err_val * (3.0 / 48.0);
            err[y + 2][bx - 2] += err_val * (1.0 / 48.0);
        }
        (false, false) => {
            // FS LTR
            err[y][bx + 1] += err_val * (7.0 / 16.0);
            err[y + 1][bx - 1] += err_val * (3.0 / 16.0);
            err[y + 1][bx] += err_val * (5.0 / 16.0);
            err[y + 1][bx + 1] += err_val * (1.0 / 16.0);
        }
        (false, true) => {
            // FS RTL
            err[y][bx - 1] += err_val * (7.0 / 16.0);
            err[y + 1][bx + 1] += err_val * (3.0 / 16.0);
            err[y + 1][bx] += err_val * (5.0 / 16.0);
            err[y + 1][bx - 1] += err_val * (1.0 / 16.0);
        }
    }
}

#[inline]
fn apply_mixed_kernel(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    err_a_val: f32,
    pixel_hash: u32,
    is_rtl: bool,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;
    let use_jjn_a = pixel_hash & 8 != 0;

    apply_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
    apply_single_channel_kernel(err_a, bx, y, err_a_val, use_jjn_a, is_rtl);
}

// ============================================================================
// Edge seeding helpers
// ============================================================================

#[inline]
fn get_seeding_rgba(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    px: usize,
    py: usize,
    reach: usize,
) -> (f32, f32, f32, f32) {
    let img_y = if py < reach { 0 } else { py - reach };
    let img_x = if px < reach {
        0
    } else if px >= reach + width {
        width - 1
    } else {
        px - reach
    };
    let idx = img_y * width + img_x;
    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
}

// ============================================================================
// Dithering context and pixel processing
// ============================================================================

/// Context for paletted dithering
struct DitherContextPaletted<'a> {
    palette: &'a DitherPalette,
}

/// Compute the integrated alpha-RGB distance for palette matching.
///
/// The distance metric is: sqrt(alpha_distance² + (rgb_perceptual_distance × alpha_factor)²)
///
/// This weighs down RGB errors for less visible (low alpha) pixels, making the dithering
/// prioritize alpha accuracy for transparent regions and RGB accuracy for opaque regions.
///
/// The alpha_factor is the target alpha value (after error adjustment), meaning:
/// - For fully transparent targets (α=0), only alpha distance matters
/// - For fully opaque targets (α=1), full RGB perceptual distance applies
/// - For semi-transparent targets, RGB distance is proportionally reduced
#[inline]
fn integrated_distance_sq(
    space: PerceptualSpace,
    target_perc_l: f32, target_perc_a: f32, target_perc_b: f32, target_alpha: f32,
    entry: &PaletteEntry,
) -> f32 {
    // Alpha distance (linear, 0-1 range)
    let alpha_diff = target_alpha - entry.lin_a;
    let alpha_dist_sq = alpha_diff * alpha_diff;

    // RGB perceptual distance
    let rgb_dist_sq = perceptual_distance_sq(
        space,
        target_perc_l, target_perc_a, target_perc_b,
        entry.perc_l, entry.perc_a, entry.perc_b,
    );

    // Scale RGB distance by alpha factor (target alpha determines visibility)
    // Using target alpha because that's the "intended" visibility of the pixel
    let alpha_factor = target_alpha.clamp(0.0, 1.0);
    let weighted_rgb_dist_sq = rgb_dist_sq * alpha_factor * alpha_factor;

    // Combined distance: sqrt(alpha_dist² + (rgb_dist × alpha)²)
    // We return squared distance for efficiency; caller can take sqrt if needed
    alpha_dist_sq + weighted_rgb_dist_sq
}

/// Process a single pixel with integrated alpha-RGB distance metric.
///
/// Returns (best_r, best_g, best_b, best_a, err_r, err_g, err_b, err_a)
#[inline]
fn process_pixel_paletted(
    ctx: &DitherContextPaletted,
    srgb_r_in: f32,
    srgb_g_in: f32,
    srgb_b_in: f32,
    alpha_in: f32,
    err_r: &[Vec<f32>],
    err_g: &[Vec<f32>],
    err_b: &[Vec<f32>],
    err_a: &[Vec<f32>],
    bx: usize,
    y: usize,
) -> (u8, u8, u8, u8, f32, f32, f32, f32) {
    // 1. Read accumulated error
    let err_r_in = err_r[y][bx];
    let err_g_in = err_g[y][bx];
    let err_b_in = err_b[y][bx];
    let err_a_in = err_a[y][bx];

    // 2. Convert input to linear space
    let srgb_r = srgb_r_in / 255.0;
    let srgb_g = srgb_g_in / 255.0;
    let srgb_b = srgb_b_in / 255.0;
    let alpha = alpha_in / 255.0; // Alpha is already linear

    let lin_r_orig = srgb_to_linear_single(srgb_r);
    let lin_g_orig = srgb_to_linear_single(srgb_g);
    let lin_b_orig = srgb_to_linear_single(srgb_b);

    // 3. Add accumulated error
    let lin_r_adj = lin_r_orig + err_r_in;
    let lin_g_adj = lin_g_orig + err_g_in;
    let lin_b_adj = lin_b_orig + err_b_in;
    let alpha_adj = alpha + err_a_in;

    // 4. Convert to perceptual space for distance calculation
    // Clamp for valid color space conversion
    let lin_r_clamped = lin_r_adj.clamp(0.0, 1.0);
    let lin_g_clamped = lin_g_adj.clamp(0.0, 1.0);
    let lin_b_clamped = lin_b_adj.clamp(0.0, 1.0);
    let alpha_clamped = alpha_adj.clamp(0.0, 1.0);

    let (target_perc_l, target_perc_a, target_perc_b) = if is_srgb_space(ctx.palette.space) {
        let srgb_r_adj = linear_to_srgb_single(lin_r_clamped);
        let srgb_g_adj = linear_to_srgb_single(lin_g_clamped);
        let srgb_b_adj = linear_to_srgb_single(lin_b_clamped);
        (srgb_r_adj, srgb_g_adj, srgb_b_adj)
    } else if is_linear_rgb_space(ctx.palette.space) {
        (lin_r_clamped, lin_g_clamped, lin_b_clamped)
    } else if is_ycbcr_space(ctx.palette.space) {
        linear_rgb_to_ycbcr(lin_r_adj, lin_g_adj, lin_b_adj)
    } else if is_lab_space(ctx.palette.space) {
        linear_rgb_to_lab(lin_r_clamped, lin_g_clamped, lin_b_clamped)
    } else {
        linear_rgb_to_oklab(lin_r_clamped, lin_g_clamped, lin_b_clamped)
    };

    // 5. Find best palette entry using integrated distance
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;

    for (idx, entry) in ctx.palette.entries.iter().enumerate() {
        let dist = integrated_distance_sq(
            ctx.palette.space,
            target_perc_l, target_perc_a, target_perc_b, alpha_clamped,
            entry,
        );

        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    let best = &ctx.palette.entries[best_idx];

    // 6. Compute error in linear space
    let err_r_val = lin_r_adj - best.lin_r;
    let err_g_val = lin_g_adj - best.lin_g;
    let err_b_val = lin_b_adj - best.lin_b;
    let err_a_val = alpha_adj - best.lin_a;

    (best.r, best.g, best.b, best.a, err_r_val, err_g_val, err_b_val, err_a_val)
}

// ============================================================================
// Generic scan pattern implementations
// ============================================================================

#[inline]
fn dither_standard_paletted<K: DitherKernel>(
    ctx: &DitherContextPaletted,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    a_out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for bx in bx_start..bx_start + process_width {
            let px = bx - bx_start;
            let in_real_image = y >= reach && px >= reach && px < reach + width;

            let (r_val, g_val, b_val, a_val) = if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
            } else {
                get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
            };

            let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            K::apply_ltr(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_serpentine_paletted<K: DitherKernel>(
    ctx: &DitherContextPaletted,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    a_out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            // Right-to-left
            for bx in (bx_start..bx_start + process_width).rev() {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                K::apply_rtl(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val);
            }
        } else {
            // Left-to-right
            for bx in bx_start..bx_start + process_width {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                K::apply_ltr(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val);
            }
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_standard_paletted(
    ctx: &DitherContextPaletted,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    a_out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        for bx in bx_start..bx_start + process_width {
            let px = bx - bx_start;
            let in_real_image = y >= reach && px >= reach && px < reach + width;

            let (r_val, g_val, b_val, a_val) = if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
            } else {
                get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
            };

            let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
            apply_mixed_kernel(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_serpentine_paletted(
    ctx: &DitherContextPaletted,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    a_out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        if y % 2 == 1 {
            for bx in (bx_start..bx_start + process_width).rev() {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
            }
        } else {
            for bx in bx_start..bx_start + process_width {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
            }
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_random_paletted(
    ctx: &DitherContextPaletted,
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    err_a: &mut [Vec<f32>],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
    a_out: &mut [u8],
    width: usize,
    height: usize,
    reach: usize,
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    for y in 0..process_height {
        let row_hash = wang_hash((y as u32) ^ hashed_seed);
        let is_rtl = row_hash & 1 == 1;

        if is_rtl {
            for bx in (bx_start..bx_start + process_width).rev() {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
            }
        } else {
            for bx in bx_start..bx_start + process_width {
                let px = bx - bx_start;
                let in_real_image = y >= reach && px >= reach && px < reach + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let pixel_hash = wang_hash((px as u32) ^ ((y as u32) << 16) ^ hashed_seed);
                apply_mixed_kernel(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
            }
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Palette-based RGBA dithering with integrated alpha-RGB distance metric.
///
/// This is the simplified API that uses Floyd-Steinberg with standard scanning.
/// For other algorithms and scan patterns, use `paletted_dither_rgba_with_mode`.
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     palette: Precomputed palette (up to 256 RGBA colors)
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as u8
pub fn paletted_dither_rgba(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    palette: &DitherPalette,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    paletted_dither_rgba_with_mode(
        r_channel, g_channel, b_channel, a_channel,
        width, height,
        palette,
        DitherMode::Standard,
        0,
        None,
    )
}

/// Palette-based RGBA dithering with selectable algorithm and scanning mode.
///
/// Uses integrated alpha-RGB distance metric:
///     distance = sqrt(alpha_dist² + (rgb_perceptual_dist × alpha_factor)²)
///
/// This weighs down RGB errors for less visible (low alpha) pixels, making the
/// dithering prioritize alpha accuracy for transparent regions and RGB accuracy
/// for opaque regions.
///
/// Note: Unlike the standard RGBA dithering, this variant does NOT support
/// separate alpha dithering mode - alpha is always integrated into the main
/// distance metric.
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     palette: Precomputed palette (up to 256 RGBA colors)
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as u8
pub fn paletted_dither_rgba_with_mode(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    palette: &DitherPalette,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = width * height;

    let ctx = DitherContextPaletted { palette };

    // Use JJN reach for all modes (largest kernel)
    let reach = JarvisJudiceNinke::REACH;
    let buf_width = reach * 4 + width;
    let buf_height = reach * 2 + height;

    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_a: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

    let mut r_out = vec![0u8; pixels];
    let mut g_out = vec![0u8; pixels];
    let mut b_out = vec![0u8; pixels];
    let mut a_out = vec![0u8; pixels];

    let hashed_seed = wang_hash(seed);

    match mode {
        DitherMode::None => {
            dither_standard_paletted::<NoneKernel>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_paletted::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_paletted::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_paletted::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_paletted::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedStandard => {
            dither_mixed_standard_paletted(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine => {
            dither_mixed_serpentine_paletted(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_paletted(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
    }

    (r_out, g_out, b_out, a_out)
}

// ============================================================================
// Pixel4 convenience wrappers
// ============================================================================

use crate::color::interleave_rgba_u8;
use crate::pixel::{pixels_to_channels_rgba, Pixel4};

/// Palette-based dither for Pixel4 array (sRGB 0-255 range) to separate RGBA channels.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range (including alpha)
///     width, height: image dimensions
///     palette: Precomputed palette (up to 256 RGBA colors)
///     mode: dither algorithm and scan pattern
///     seed: random seed for mixed modes
///     progress: optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     Tuple of (R, G, B, A) u8 vectors
pub fn paletted_dither_rgba_channels(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    palette: &DitherPalette,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (r, g, b, a) = pixels_to_channels_rgba(pixels);
    paletted_dither_rgba_with_mode(
        &r, &g, &b, &a,
        width, height,
        palette, mode, seed, progress,
    )
}

/// Palette-based dither for Pixel4 array to interleaved RGBA u8.
///
/// Args:
///     pixels: Pixel4 array with values in sRGB 0-255 range (including alpha)
///     width, height: image dimensions
///     palette: Precomputed palette (up to 256 RGBA colors)
///     mode: dither algorithm and scan pattern
///     seed: random seed for mixed modes
///     progress: optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     Interleaved RGBA u8 data (RGBARGBA...)
pub fn paletted_dither_rgba_interleaved(
    pixels: &[Pixel4],
    width: usize,
    height: usize,
    palette: &DitherPalette,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let (r_u8, g_u8, b_u8, a_u8) = paletted_dither_rgba_channels(
        pixels, width, height, palette, mode, seed, progress,
    );
    interleave_rgba_u8(&r_u8, &g_u8, &b_u8, &a_u8)
}

/// Palette-based dither returning palette indices instead of RGBA values.
///
/// This is useful for generating indexed image formats (GIF, PNG8, etc.)
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     palette: Precomputed palette (up to 256 RGBA colors)
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     Vector of palette indices (u8), one per pixel
pub fn paletted_dither_to_indices(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    palette: &DitherPalette,
    mode: DitherMode,
    seed: u32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    paletted_dither_to_indices_impl(
        r_channel, g_channel, b_channel, a_channel,
        width, height, palette, mode, seed, progress,
    )
}

/// Internal implementation for index-based dithering
fn paletted_dither_to_indices_impl(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    palette: &DitherPalette,
    mode: DitherMode,
    seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) -> Vec<u8> {
    let pixels = width * height;

    let ctx = DitherContextPaletted { palette };

    let reach = JarvisJudiceNinke::REACH;
    let buf_width = reach * 4 + width;
    let buf_height = reach * 2 + height;

    let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
    let mut err_a: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

    let mut indices = vec![0u8; pixels];

    let _hashed_seed = wang_hash(seed);

    // We need a custom dithering loop that returns indices
    // Use Floyd-Steinberg standard as the default implementation
    let process_height = reach + height;
    let process_width = reach + width + reach;
    let bx_start = reach;

    match mode {
        DitherMode::None | DitherMode::Standard => {
            for y in 0..process_height {
                for bx in bx_start..bx_start + process_width {
                    let px = bx - bx_start;
                    let in_real_image = y >= reach && px >= reach && px < reach + width;

                    let (r_val, g_val, b_val, a_val) = if in_real_image {
                        let img_x = px - reach;
                        let img_y = y - reach;
                        let idx = img_y * width + img_x;
                        (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                    } else {
                        get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                    };

                    let (best_idx, err_r_val, err_g_val, err_b_val, err_a_val) =
                        process_pixel_paletted_index(&ctx, r_val, g_val, b_val, a_val, &err_r, &err_g, &err_b, &err_a, bx, y);

                    if in_real_image {
                        let img_x = px - reach;
                        let img_y = y - reach;
                        let idx = img_y * width + img_x;
                        indices[idx] = best_idx;
                    }

                    if matches!(mode, DitherMode::Standard) {
                        FloydSteinberg::apply_ltr(&mut err_r, &mut err_g, &mut err_b, &mut err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val);
                    }
                }
                if y >= reach {
                    if let Some(ref mut cb) = progress {
                        cb((y - reach + 1) as f32 / height as f32);
                    }
                }
            }
        }
        DitherMode::Serpentine => {
            for y in 0..process_height {
                if y % 2 == 1 {
                    for bx in (bx_start..bx_start + process_width).rev() {
                        let px = bx - bx_start;
                        let in_real_image = y >= reach && px >= reach && px < reach + width;

                        let (r_val, g_val, b_val, a_val) = if in_real_image {
                            let img_x = px - reach;
                            let img_y = y - reach;
                            let idx = img_y * width + img_x;
                            (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                        } else {
                            get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                        };

                        let (best_idx, err_r_val, err_g_val, err_b_val, err_a_val) =
                            process_pixel_paletted_index(&ctx, r_val, g_val, b_val, a_val, &err_r, &err_g, &err_b, &err_a, bx, y);

                        if in_real_image {
                            let img_x = px - reach;
                            let img_y = y - reach;
                            let idx = img_y * width + img_x;
                            indices[idx] = best_idx;
                        }

                        FloydSteinberg::apply_rtl(&mut err_r, &mut err_g, &mut err_b, &mut err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val);
                    }
                } else {
                    for bx in bx_start..bx_start + process_width {
                        let px = bx - bx_start;
                        let in_real_image = y >= reach && px >= reach && px < reach + width;

                        let (r_val, g_val, b_val, a_val) = if in_real_image {
                            let img_x = px - reach;
                            let img_y = y - reach;
                            let idx = img_y * width + img_x;
                            (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                        } else {
                            get_seeding_rgba(r_channel, g_channel, b_channel, a_channel, width, px, y, reach)
                        };

                        let (best_idx, err_r_val, err_g_val, err_b_val, err_a_val) =
                            process_pixel_paletted_index(&ctx, r_val, g_val, b_val, a_val, &err_r, &err_g, &err_b, &err_a, bx, y);

                        if in_real_image {
                            let img_x = px - reach;
                            let img_y = y - reach;
                            let idx = img_y * width + img_x;
                            indices[idx] = best_idx;
                        }

                        FloydSteinberg::apply_ltr(&mut err_r, &mut err_g, &mut err_b, &mut err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val);
                    }
                }
                if y >= reach {
                    if let Some(ref mut cb) = progress {
                        cb((y - reach + 1) as f32 / height as f32);
                    }
                }
            }
        }
        _ => {
            // For other modes, fall back to RGBA dithering and look up indices
            let (r_out, g_out, b_out, a_out) = paletted_dither_rgba_with_mode(
                r_channel, g_channel, b_channel, a_channel,
                width, height, palette, mode, seed, progress,
            );

            // Map RGBA back to indices
            for i in 0..pixels {
                for (idx, entry) in palette.entries.iter().enumerate() {
                    if entry.r == r_out[i] && entry.g == g_out[i] && entry.b == b_out[i] && entry.a == a_out[i] {
                        indices[i] = idx as u8;
                        break;
                    }
                }
            }
        }
    }

    indices
}

/// Process pixel and return index instead of RGBA values
#[inline]
fn process_pixel_paletted_index(
    ctx: &DitherContextPaletted,
    srgb_r_in: f32,
    srgb_g_in: f32,
    srgb_b_in: f32,
    alpha_in: f32,
    err_r: &[Vec<f32>],
    err_g: &[Vec<f32>],
    err_b: &[Vec<f32>],
    err_a: &[Vec<f32>],
    bx: usize,
    y: usize,
) -> (u8, f32, f32, f32, f32) {
    let err_r_in = err_r[y][bx];
    let err_g_in = err_g[y][bx];
    let err_b_in = err_b[y][bx];
    let err_a_in = err_a[y][bx];

    let srgb_r = srgb_r_in / 255.0;
    let srgb_g = srgb_g_in / 255.0;
    let srgb_b = srgb_b_in / 255.0;
    let alpha = alpha_in / 255.0;

    let lin_r_orig = srgb_to_linear_single(srgb_r);
    let lin_g_orig = srgb_to_linear_single(srgb_g);
    let lin_b_orig = srgb_to_linear_single(srgb_b);

    let lin_r_adj = lin_r_orig + err_r_in;
    let lin_g_adj = lin_g_orig + err_g_in;
    let lin_b_adj = lin_b_orig + err_b_in;
    let alpha_adj = alpha + err_a_in;

    let lin_r_clamped = lin_r_adj.clamp(0.0, 1.0);
    let lin_g_clamped = lin_g_adj.clamp(0.0, 1.0);
    let lin_b_clamped = lin_b_adj.clamp(0.0, 1.0);
    let alpha_clamped = alpha_adj.clamp(0.0, 1.0);

    let (target_perc_l, target_perc_a, target_perc_b) = if is_srgb_space(ctx.palette.space) {
        let srgb_r_adj = linear_to_srgb_single(lin_r_clamped);
        let srgb_g_adj = linear_to_srgb_single(lin_g_clamped);
        let srgb_b_adj = linear_to_srgb_single(lin_b_clamped);
        (srgb_r_adj, srgb_g_adj, srgb_b_adj)
    } else if is_linear_rgb_space(ctx.palette.space) {
        (lin_r_clamped, lin_g_clamped, lin_b_clamped)
    } else if is_ycbcr_space(ctx.palette.space) {
        linear_rgb_to_ycbcr(lin_r_adj, lin_g_adj, lin_b_adj)
    } else if is_lab_space(ctx.palette.space) {
        linear_rgb_to_lab(lin_r_clamped, lin_g_clamped, lin_b_clamped)
    } else {
        linear_rgb_to_oklab(lin_r_clamped, lin_g_clamped, lin_b_clamped)
    };

    let mut best_idx = 0u8;
    let mut best_dist = f32::INFINITY;

    for (idx, entry) in ctx.palette.entries.iter().enumerate() {
        let dist = integrated_distance_sq(
            ctx.palette.space,
            target_perc_l, target_perc_a, target_perc_b, alpha_clamped,
            entry,
        );

        if dist < best_dist {
            best_dist = dist;
            best_idx = idx as u8;
        }
    }

    let best = &ctx.palette.entries[best_idx as usize];

    let err_r_val = lin_r_adj - best.lin_r;
    let err_g_val = lin_g_adj - best.lin_g;
    let err_b_val = lin_b_adj - best.lin_b;
    let err_a_val = alpha_adj - best.lin_a;

    (best_idx, err_r_val, err_g_val, err_b_val, err_a_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_creation() {
        let colors = vec![
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);
        assert_eq!(palette.len(), 5);
    }

    #[test]
    fn test_paletted_dither_basic() {
        let colors = vec![
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            (128, 128, 128, 255),
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);

        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let a: Vec<f32> = vec![255.0; 100];

        let (r_out, g_out, b_out, a_out) = paletted_dither_rgba(
            &r, &g, &b, &a, 10, 10, &palette
        );

        assert_eq!(r_out.len(), 100);
        assert_eq!(g_out.len(), 100);
        assert_eq!(b_out.len(), 100);
        assert_eq!(a_out.len(), 100);

        // All output values should be from the palette
        for &v in &r_out {
            assert!(v == 0 || v == 128 || v == 255, "Output R not in palette: {}", v);
        }
    }

    #[test]
    fn test_transparent_pixel_prioritizes_alpha() {
        // Palette with different colors but same alpha
        let colors = vec![
            (255, 0, 0, 0),    // Transparent red
            (0, 0, 255, 255),  // Opaque blue
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);

        // Input: transparent green - should prefer transparent red due to alpha match
        let r = vec![0.0; 1];
        let g = vec![255.0; 1];
        let b = vec![0.0; 1];
        let a = vec![0.0; 1]; // Fully transparent

        let (r_out, _g_out, _b_out, a_out) = paletted_dither_rgba(
            &r, &g, &b, &a, 1, 1, &palette
        );

        // Should choose transparent red because alpha match is prioritized
        // when input alpha is 0
        assert_eq!(a_out[0], 0, "Should pick transparent color");
        assert_eq!(r_out[0], 255, "Should pick red (transparent)");
    }

    #[test]
    fn test_opaque_pixel_prioritizes_color() {
        // Palette with different colors
        let colors = vec![
            (255, 0, 0, 255),  // Opaque red
            (0, 255, 0, 255),  // Opaque green
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);

        // Input: opaque green-ish color
        let r = vec![50.0; 1];
        let g = vec![200.0; 1];
        let b = vec![50.0; 1];
        let a = vec![255.0; 1];

        let (_r_out, g_out, _b_out, _a_out) = paletted_dither_rgba(
            &r, &g, &b, &a, 1, 1, &palette
        );

        // Should choose green because it's closer in RGB
        assert_eq!(g_out[0], 255, "Should pick green");
    }

    #[test]
    fn test_palette_index_output() {
        let colors = vec![
            (0, 0, 0, 255),
            (255, 255, 255, 255),
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);

        let r = vec![0.0, 255.0];
        let g = vec![0.0, 255.0];
        let b = vec![0.0, 255.0];
        let a = vec![255.0, 255.0];

        let indices = paletted_dither_to_indices(
            &r, &g, &b, &a, 2, 1, &palette,
            DitherMode::None, 0, None
        );

        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0, "Black should map to index 0");
        assert_eq!(indices[1], 1, "White should map to index 1");
    }

    #[test]
    fn test_all_dither_modes() {
        let colors = vec![
            (0, 0, 0, 255),
            (128, 128, 128, 255),
            (255, 255, 255, 255),
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);

        let r: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let g: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let a: Vec<f32> = vec![255.0; 100];

        let modes = [
            DitherMode::None,
            DitherMode::Standard,
            DitherMode::Serpentine,
            DitherMode::JarvisStandard,
            DitherMode::JarvisSerpentine,
            DitherMode::MixedStandard,
            DitherMode::MixedSerpentine,
            DitherMode::MixedRandom,
        ];

        for mode in modes {
            let (r_out, g_out, b_out, a_out) = paletted_dither_rgba_with_mode(
                &r, &g, &b, &a, 10, 10, &palette, mode, 42, None
            );

            assert_eq!(r_out.len(), 100, "Mode {:?} failed", mode);
            assert_eq!(g_out.len(), 100, "Mode {:?} failed", mode);
            assert_eq!(b_out.len(), 100, "Mode {:?} failed", mode);
            assert_eq!(a_out.len(), 100, "Mode {:?} failed", mode);
        }
    }
}
