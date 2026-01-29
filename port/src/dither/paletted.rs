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

use crate::color::srgb_to_linear_single;
use crate::color_distance::perceptual_distance_sq;
use super::common::{
    apply_mixed_kernel_rgba, linear_rgb_to_perceptual, linear_rgb_to_perceptual_clamped,
    triple32, wang_hash, DitherMode, FloydSteinberg, JarvisJudiceNinke, NoneKernel, Ostromoukhov,
    PerceptualSpace, RgbaKernel,
};
use super::kernels::{apply_h2_kernel_rgba, apply_adaptive_kernel_rgba, H2_REACH, H2_SEED};
use super::palette_hull::EPSILON as HULL_EPSILON;
use super::palette_projection::ExtendedPalette;

// ============================================================================
// Palette structures
// ============================================================================

/// A single palette entry with precomputed values for efficient distance calculation.
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Debug)]
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

            // Convert to perceptual space (clamped since palette values are in-gamut)
            let (perc_l, perc_a, perc_b) =
                linear_rgb_to_perceptual_clamped(space, lin_r, lin_g, lin_b);

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

    /// Extract linear RGB points from palette entries.
    /// Used for convex hull computation.
    pub fn linear_rgb_points(&self) -> Vec<[f32; 3]> {
        self.entries
            .iter()
            .map(|e| [e.lin_r, e.lin_g, e.lin_b])
            .collect()
    }

    /// Get sRGB values for a palette entry by index.
    pub fn get_srgb(&self, idx: usize) -> (u8, u8, u8, u8) {
        let e = &self.entries[idx];
        (e.r, e.g, e.b, e.a)
    }

    /// Get linear RGB values for a palette entry by index.
    pub fn get_linear_rgb(&self, idx: usize) -> (f32, f32, f32, f32) {
        let e = &self.entries[idx];
        (e.lin_r, e.lin_g, e.lin_b, e.lin_a)
    }

    /// Get perceptual space coordinates for a palette entry by index.
    pub fn get_perceptual(&self, idx: usize) -> (f32, f32, f32) {
        let e = &self.entries[idx];
        (e.perc_l, e.perc_a, e.perc_b)
    }

    /// Get the perceptual space used by this palette.
    pub fn space(&self) -> PerceptualSpace {
        self.space
    }
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

/// Extended context for paletted dithering with gamut mapping.
/// Supports hull clamping and optional hull tracing for keeping error diffusion
/// within the palette's gamut.
struct ExtendedDitherContext<'a> {
    extended: &'a ExtendedPalette,
    use_hull_tracing: bool,
    overshoot_penalty: bool,
    /// Error decay factor (0.0-1.0) applied when selected color is farther than hull.
    /// 1.0 = no decay (default), lower values reduce error accumulation.
    hull_error_decay: f32,
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
    // Alpha always gets error applied
    let alpha_adj = alpha + err_a_in;

    // For RGB, skip error application for fully transparent pixels.
    // This produces cleaner RGB output (useful if alpha is later stripped).
    // Error diffusion still works since the error term is alpha-weighted anyway.
    let (lin_r_adj, lin_g_adj, lin_b_adj) = if alpha_adj <= 0.0 {
        (lin_r_orig, lin_g_orig, lin_b_orig)
    } else {
        (lin_r_orig + err_r_in, lin_g_orig + err_g_in, lin_b_orig + err_b_in)
    };

    // 4. Convert to perceptual space for distance calculation
    // Use unclamped values for true distance (matching RGB/RGBA behavior)
    // Only clamp alpha since it's used as a weighting factor
    let alpha_clamped = alpha_adj.clamp(0.0, 1.0);
    let (target_perc_l, target_perc_a, target_perc_b) =
        linear_rgb_to_perceptual(ctx.palette.space, lin_r_adj, lin_g_adj, lin_b_adj);

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

    // 6. Compute alpha-aware error to diffuse
    // For RGB: error = (1 - α) × e_in + α × q_err
    // This ensures:
    //   - Fully transparent pixels (α=0) pass all accumulated error to neighbors
    //   - Fully opaque pixels (α=1) pass only quantization error
    //   - Semi-transparent pixels blend proportionally
    let q_err_r = lin_r_adj - best.lin_r;
    let q_err_g = lin_g_adj - best.lin_g;
    let q_err_b = lin_b_adj - best.lin_b;

    let alpha_factor = alpha_clamped;
    let one_minus_alpha = 1.0 - alpha_factor;
    let err_r_val = one_minus_alpha * err_r_in + alpha_factor * q_err_r;
    let err_g_val = one_minus_alpha * err_g_in + alpha_factor * q_err_g;
    let err_b_val = one_minus_alpha * err_b_in + alpha_factor * q_err_b;

    // Alpha uses simple quantization error (no visibility weighting for alpha itself)
    let err_a_val = alpha_adj - best.lin_a;

    (best.r, best.g, best.b, best.a, err_r_val, err_g_val, err_b_val, err_a_val)
}

/// Process a single pixel with extended palette (hull clamping + optional hull tracing).
///
/// Returns (best_r, best_g, best_b, best_a, err_r, err_g, err_b, err_a)
#[inline]
fn process_pixel_paletted_extended(
    ctx: &ExtendedDitherContext,
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
    let palette = ctx.extended.palette();
    let space = palette.space();

    // 1. Read accumulated error
    let err_r_in = err_r[y][bx];
    let err_g_in = err_g[y][bx];
    let err_b_in = err_b[y][bx];
    let err_a_in = err_a[y][bx];

    // 2. Convert input to linear space
    let srgb_r = srgb_r_in / 255.0;
    let srgb_g = srgb_g_in / 255.0;
    let srgb_b = srgb_b_in / 255.0;
    let alpha = alpha_in / 255.0;

    let lin_r_orig = srgb_to_linear_single(srgb_r);
    let lin_g_orig = srgb_to_linear_single(srgb_g);
    let lin_b_orig = srgb_to_linear_single(srgb_b);

    // 3. Add accumulated error
    let alpha_adj = alpha + err_a_in;

    let (lin_r_adj, lin_g_adj, lin_b_adj) = if alpha_adj <= 0.0 {
        (lin_r_orig, lin_g_orig, lin_b_orig)
    } else {
        (lin_r_orig + err_r_in, lin_g_orig + err_g_in, lin_b_orig + err_b_in)
    };

    // 4. ALWAYS clamp to hull (gamut mapping)
    let [lin_r_clamped, lin_g_clamped, lin_b_clamped] =
        ctx.extended.clamp_to_hull([lin_r_adj, lin_g_adj, lin_b_adj]);

    // 5. Convert to perceptual space for distance calculation
    let alpha_clamped = alpha_adj.clamp(0.0, 1.0);
    let (target_perc_l, target_perc_a, target_perc_b) =
        linear_rgb_to_perceptual(space, lin_r_clamped, lin_g_clamped, lin_b_clamped);

    // 6. Find best palette entry
    let best_idx = if ctx.use_hull_tracing {
        // Use hull-aware search (projects to edges/surfaces for boundary colors)
        ctx.extended.find_nearest_real([lin_r_clamped, lin_g_clamped, lin_b_clamped], ctx.overshoot_penalty)
    } else {
        // Direct search on palette entries only
        find_nearest_palette_entry(
            palette, space,
            target_perc_l, target_perc_a, target_perc_b, alpha_clamped,
        )
    };

    let (best_r, best_g, best_b, best_a) = palette.get_srgb(best_idx);
    let (best_lin_r, best_lin_g, best_lin_b, best_lin_a) = palette.get_linear_rgb(best_idx);

    // 7. Compute alpha-aware error to diffuse
    let q_err_r = lin_r_clamped - best_lin_r;
    let q_err_g = lin_g_clamped - best_lin_g;
    let q_err_b = lin_b_clamped - best_lin_b;

    // 7a. Apply hull error decay if selected color is significantly farther than hull boundary
    // This prevents error accumulation when palette is sparse near hull
    let decay = if ctx.hull_error_decay < 1.0 {
        // Distance from adjusted point to hull (how far we clamped)
        let dr_hull = lin_r_adj - lin_r_clamped;
        let dg_hull = lin_g_adj - lin_g_clamped;
        let db_hull = lin_b_adj - lin_b_clamped;
        let hull_dist_sq = dr_hull * dr_hull + dg_hull * dg_hull + db_hull * db_hull;

        // Distance from adjusted point to selected color
        let dr_color = lin_r_adj - best_lin_r;
        let dg_color = lin_g_adj - best_lin_g;
        let db_color = lin_b_adj - best_lin_b;
        let color_dist_sq = dr_color * dr_color + dg_color * dg_color + db_color * db_color;

        // Apply decay only when selected color is farther than hull by more than EPSILON
        // (Use squared epsilon since we're comparing squared distances)
        let epsilon_sq = HULL_EPSILON * HULL_EPSILON;
        if color_dist_sq > hull_dist_sq + epsilon_sq {
            ctx.hull_error_decay
        } else {
            1.0
        }
    } else {
        1.0
    };

    let alpha_factor = alpha_clamped;
    let one_minus_alpha = 1.0 - alpha_factor;
    let err_r_val = decay * (one_minus_alpha * err_r_in + alpha_factor * q_err_r);
    let err_g_val = decay * (one_minus_alpha * err_g_in + alpha_factor * q_err_g);
    let err_b_val = decay * (one_minus_alpha * err_b_in + alpha_factor * q_err_b);

    let err_a_val = alpha_adj - best_lin_a;

    (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val)
}

/// Find nearest palette entry using integrated alpha-RGB distance.
#[inline]
fn find_nearest_palette_entry(
    palette: &DitherPalette,
    space: PerceptualSpace,
    target_perc_l: f32,
    target_perc_a: f32,
    target_perc_b: f32,
    target_alpha: f32,
) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;

    for idx in 0..palette.len() {
        let (entry_perc_l, entry_perc_a, entry_perc_b) = palette.get_perceptual(idx);
        let (_, _, _, entry_lin_a) = palette.get_linear_rgb(idx);

        // Alpha distance
        let alpha_diff = target_alpha - entry_lin_a;
        let alpha_dist_sq = alpha_diff * alpha_diff;

        // RGB perceptual distance
        let rgb_dist_sq = perceptual_distance_sq(
            space,
            target_perc_l, target_perc_a, target_perc_b,
            entry_perc_l, entry_perc_a, entry_perc_b,
        );

        // Combined distance with alpha weighting
        let alpha_factor = target_alpha.clamp(0.0, 1.0);
        let weighted_rgb_dist_sq = rgb_dist_sq * alpha_factor * alpha_factor;
        let dist = alpha_dist_sq + weighted_rgb_dist_sq;

        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

// ============================================================================
// Generic scan pattern implementations
// ============================================================================

#[inline]
fn dither_standard_paletted<K: RgbaKernel>(
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

            K::apply_ltr(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_serpentine_paletted<K: RgbaKernel>(
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

                K::apply_rtl(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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

                K::apply_ltr(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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
            apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
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
// Extended scan pattern implementations (with hull clamping + optional hull tracing)
// ============================================================================

#[inline]
fn dither_standard_paletted_extended<K: RgbaKernel>(
    ctx: &ExtendedDitherContext,
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
                process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - reach;
                let img_y = y - reach;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            K::apply_ltr(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_serpentine_paletted_extended<K: RgbaKernel>(
    ctx: &ExtendedDitherContext,
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
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                K::apply_rtl(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - reach;
                    let img_y = y - reach;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                K::apply_ltr(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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
fn dither_mixed_standard_paletted_extended(
    ctx: &ExtendedDitherContext,
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
                process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

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
            apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
        }
        if y >= reach {
            if let Some(ref mut cb) = progress {
                cb((y - reach + 1) as f32 / height as f32);
            }
        }
    }
}

#[inline]
fn dither_mixed_serpentine_paletted_extended(
    ctx: &ExtendedDitherContext,
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
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
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
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
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
fn dither_mixed_random_paletted_extended(
    ctx: &ExtendedDitherContext,
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
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
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
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

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
                apply_mixed_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
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
// Mixed H2 (second-order kernel) paletted dithering
// ============================================================================

/// Get RGBA values for H2 processing coordinate, handling seeding area mapping.
#[inline]
fn get_seeding_rgba_h2(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    px: usize,
    py: usize,
    seed: usize,
) -> (f32, f32, f32, f32) {
    let img_y = if py < seed { 0 } else { py - seed };
    let img_x = if px < seed {
        0
    } else if px >= seed + width {
        width - 1
    } else {
        px - seed
    };
    let idx = img_y * width + img_x;
    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
}

/// Mixed H2 kernel dithering for paletted RGBA, LTR only.
#[inline]
fn dither_mixed_h2_standard_paletted(
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let in_real_image = y >= seed && px >= seed && px < seed + width;

            let (r_val, g_val, b_val, a_val) = if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
            } else {
                get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
            };

            let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            let img_x = px.wrapping_sub(seed);
            let img_y = y.wrapping_sub(seed);
            let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            apply_h2_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
        }
        if y >= seed {
            if let Some(ref mut cb) = progress {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

/// Mixed H2 kernel dithering for paletted RGBA, serpentine scanning.
fn dither_mixed_h2_serpentine_paletted(
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let in_real_image = y >= seed && px >= seed && px < seed + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let img_x = px.wrapping_sub(seed);
                let img_y = y.wrapping_sub(seed);
                let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_h2_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let in_real_image = y >= seed && px >= seed && px < seed + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let img_x = px.wrapping_sub(seed);
                let img_y = y.wrapping_sub(seed);
                let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_h2_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
            }
        }
        if y >= seed {
            if let Some(ref mut cb) = progress {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

/// Mixed H2 kernel dithering for extended paletted RGBA, LTR only.
#[inline]
fn dither_mixed_h2_standard_paletted_extended(
    ctx: &ExtendedDitherContext,
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let in_real_image = y >= seed && px >= seed && px < seed + width;

            let (r_val, g_val, b_val, a_val) = if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
            } else {
                get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
            };

            let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            let img_x = px.wrapping_sub(seed);
            let img_y = y.wrapping_sub(seed);
            let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
            apply_h2_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
        }
        if y >= seed {
            if let Some(ref mut cb) = progress {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

/// Mixed H2 kernel dithering for extended paletted RGBA, serpentine scanning.
fn dither_mixed_h2_serpentine_paletted_extended(
    ctx: &ExtendedDitherContext,
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
    hashed_seed: u32,
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        if y % 2 == 1 {
            for px in (0..process_width).rev() {
                let bx = bx_start + px;
                let in_real_image = y >= seed && px >= seed && px < seed + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let img_x = px.wrapping_sub(seed);
                let img_y = y.wrapping_sub(seed);
                let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_h2_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, true);
            }
        } else {
            for px in 0..process_width {
                let bx = bx_start + px;
                let in_real_image = y >= seed && px >= seed && px < seed + width;

                let (r_val, g_val, b_val, a_val) = if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
                } else {
                    get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
                };

                let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                    process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

                if in_real_image {
                    let img_x = px - seed;
                    let img_y = y - seed;
                    let idx = img_y * width + img_x;
                    r_out[idx] = best_r;
                    g_out[idx] = best_g;
                    b_out[idx] = best_b;
                    a_out[idx] = best_a;
                }

                let img_x = px.wrapping_sub(seed);
                let img_y = y.wrapping_sub(seed);
                let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);
                apply_h2_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, pixel_hash, false);
            }
        }
        if y >= seed {
            if let Some(ref mut cb) = progress {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

// ============================================================================
// Adaptive (gradient-blended 1st+2nd order) paletted dithering
// ============================================================================

/// Pre-compute alpha map from RGBA channel luminance gradients (for paletted).
fn compute_alpha_map_paletted(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut alpha_map = vec![1.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let lum = (0.2126 * r_channel[idx] + 0.7152 * g_channel[idx] + 0.0722 * b_channel[idx]) / 255.0;
            let gx = if x + 1 < width {
                let idx_r = y * width + x + 1;
                let lum_r = (0.2126 * r_channel[idx_r] + 0.7152 * g_channel[idx_r] + 0.0722 * b_channel[idx_r]) / 255.0;
                (lum_r - lum).abs()
            } else {
                0.0
            };
            let gy = if y + 1 < height {
                let idx_b = (y + 1) * width + x;
                let lum_b = (0.2126 * r_channel[idx_b] + 0.7152 * g_channel[idx_b] + 0.0722 * b_channel[idx_b]) / 255.0;
                (lum_b - lum).abs()
            } else {
                0.0
            };
            alpha_map[idx] = (1.0 - gx) * (1.0 - gy);
        }
    }
    alpha_map
}

/// Mixed adaptive dithering for paletted RGBA, always LTR.
#[inline]
fn dither_mixed_adaptive_paletted(
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
    hashed_seed: u32,
    grad_alpha_map: &[f32],
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let in_real_image = y >= seed && px >= seed && px < seed + width;

            let (r_val, g_val, b_val, a_val) = if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
            } else {
                get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
            };

            let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                process_pixel_paletted(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            let img_x = px.wrapping_sub(seed);
            let img_y = y.wrapping_sub(seed);
            let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);

            // Look up gradient alpha: use 1.0 for seed pixels
            let grad_alpha = if img_x < width && img_y < height {
                grad_alpha_map[img_y * width + img_x]
            } else {
                1.0
            };

            apply_adaptive_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, grad_alpha, pixel_hash);
        }
        if y >= seed {
            if let Some(ref mut cb) = progress {
                cb((y - seed + 1) as f32 / height as f32);
            }
        }
    }
}

/// Mixed adaptive dithering for extended paletted RGBA, always LTR.
#[inline]
fn dither_mixed_adaptive_paletted_extended(
    ctx: &ExtendedDitherContext,
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
    hashed_seed: u32,
    grad_alpha_map: &[f32],
    mut progress: Option<&mut dyn FnMut(f32)>,
) {
    let seed = H2_SEED;
    let reach = H2_REACH;

    let bx_start = reach;
    let process_width = seed + width + seed;
    let process_height = seed + height;

    for y in 0..process_height {
        for px in 0..process_width {
            let bx = bx_start + px;
            let in_real_image = y >= seed && px >= seed && px < seed + width;

            let (r_val, g_val, b_val, a_val) = if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                (r_channel[idx], g_channel[idx], b_channel[idx], a_channel[idx])
            } else {
                get_seeding_rgba_h2(r_channel, g_channel, b_channel, a_channel, width, px, y, seed)
            };

            let (best_r, best_g, best_b, best_a, err_r_val, err_g_val, err_b_val, err_a_val) =
                process_pixel_paletted_extended(ctx, r_val, g_val, b_val, a_val, err_r, err_g, err_b, err_a, bx, y);

            if in_real_image {
                let img_x = px - seed;
                let img_y = y - seed;
                let idx = img_y * width + img_x;
                r_out[idx] = best_r;
                g_out[idx] = best_g;
                b_out[idx] = best_b;
                a_out[idx] = best_a;
            }

            let img_x = px.wrapping_sub(seed);
            let img_y = y.wrapping_sub(seed);
            let pixel_hash = triple32((img_x as u32) ^ ((img_y as u32) << 16) ^ hashed_seed);

            // Look up gradient alpha: use 1.0 for seed pixels
            let grad_alpha = if img_x < width && img_y < height {
                grad_alpha_map[img_y * width + img_x]
            } else {
                1.0
            };

            apply_adaptive_kernel_rgba(err_r, err_g, err_b, err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, grad_alpha, pixel_hash);
        }
        if y >= seed {
            if let Some(ref mut cb) = progress {
                cb((y - seed + 1) as f32 / height as f32);
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

    // H2 needs different buffer dimensions (REACH=4, SEED=16), handle as early return
    if mode == DitherMode::MixedH2Standard || mode == DitherMode::MixedH2Serpentine {
        let h2_reach = H2_REACH;
        let h2_seed = H2_SEED;
        let buf_width = h2_reach + h2_seed + width + h2_seed + h2_reach;
        let buf_height = h2_seed + height + h2_reach;

        let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_a: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

        let mut r_out = vec![0u8; pixels];
        let mut g_out = vec![0u8; pixels];
        let mut b_out = vec![0u8; pixels];
        let mut a_out = vec![0u8; pixels];

        let hashed_seed = triple32(seed);

        if mode == DitherMode::MixedH2Serpentine {
            dither_mixed_h2_serpentine_paletted(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, hashed_seed, progress,
            );
        } else {
            dither_mixed_h2_standard_paletted(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, hashed_seed, progress,
            );
        }

        return (r_out, g_out, b_out, a_out);
    }

    // Adaptive needs same buffer dimensions as H2, handle as early return
    if mode == DitherMode::MixedAdaptive {
        let h2_reach = H2_REACH;
        let h2_seed = H2_SEED;
        let buf_width = h2_reach + h2_seed + width + h2_seed + h2_reach;
        let buf_height = h2_seed + height + h2_reach;

        let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_a: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

        let mut r_out = vec![0u8; pixels];
        let mut g_out = vec![0u8; pixels];
        let mut b_out = vec![0u8; pixels];
        let mut a_out = vec![0u8; pixels];

        let hashed_seed = triple32(seed);
        let grad_alpha_map = compute_alpha_map_paletted(r_channel, g_channel, b_channel, width, height);

        dither_mixed_adaptive_paletted(
            &ctx, r_channel, g_channel, b_channel, a_channel,
            &mut err_r, &mut err_g, &mut err_b, &mut err_a,
            &mut r_out, &mut g_out, &mut b_out, &mut a_out,
            width, height, hashed_seed, &grad_alpha_map, progress,
        );

        return (r_out, g_out, b_out, a_out);
    }

    // Use JJN reach for all modes (largest kernel)
    let reach = <JarvisJudiceNinke as RgbaKernel>::REACH;
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
        DitherMode::MixedStandard | DitherMode::MixedWangStandard | DitherMode::MixedLowbiasOldStandard => {
            dither_mixed_standard_paletted(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine | DitherMode::MixedWangSerpentine | DitherMode::MixedLowbiasOldSerpentine => {
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
        DitherMode::OstromoukhovStandard => {
            dither_standard_paletted::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_serpentine_paletted::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        // Zhou-Fang: fall back to Ostromoukhov for colorspace-aware dithering
        DitherMode::ZhouFangStandard => {
            dither_standard_paletted::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::ZhouFangSerpentine => {
            dither_serpentine_paletted::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        // Ulichney: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::UlichneyStandard | DitherMode::UlichneyWeightStandard => {
            dither_standard_paletted::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::UlichneySerpentine | DitherMode::UlichneyWeightSerpentine => {
            dither_serpentine_paletted::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        // FS+TPDF: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::FsTpdfStandard => {
            dither_standard_paletted::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::FsTpdfSerpentine => {
            dither_serpentine_paletted::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedH2Standard | DitherMode::MixedH2Serpentine | DitherMode::MixedAdaptive => {
            unreachable!("H2/Adaptive modes handled in early return above");
        }
    }

    (r_out, g_out, b_out, a_out)
}

/// Palette-based RGBA dithering with gamut mapping (hull clamping + optional hull tracing).
///
/// This extended version provides better handling of colors at palette gamut boundaries:
/// - Always clamps error-adjusted colors to within the palette's convex hull
/// - When `use_hull_tracing` is true, uses hull-aware search that considers
///   edge/surface projections for boundary color matching
///
/// Args:
///     r_channel, g_channel, b_channel, a_channel: Input channels as f32 in range [0, 255]
///     width, height: Image dimensions
///     palette: Precomputed palette (up to 256 RGBA colors)
///     mode: Dithering algorithm and scanning mode
///     seed: Random seed for mixed modes (ignored for non-mixed modes)
///     use_hull_tracing: If true, use hull-aware search for boundary colors
///     overshoot_penalty: If true, penalize choices that push error diffusion outside gamut
///     hull_error_decay: Error decay factor (0.0-1.0) when selected color is farther than hull.
///                       1.0 = no decay, lower values reduce error accumulation for sparse palettes.
///     progress: Optional callback called with progress (0.0 to 1.0)
///
/// Returns:
///     (r_out, g_out, b_out, a_out): Output channels as u8
pub fn paletted_dither_rgba_gamut_mapped(
    r_channel: &[f32],
    g_channel: &[f32],
    b_channel: &[f32],
    a_channel: &[f32],
    width: usize,
    height: usize,
    palette: &DitherPalette,
    mode: DitherMode,
    seed: u32,
    use_hull_tracing: bool,
    overshoot_penalty: bool,
    hull_error_decay: f32,
    progress: Option<&mut dyn FnMut(f32)>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = width * height;

    // Build extended palette with hull
    let extended = ExtendedPalette::new(palette.clone(), palette.space());

    let ctx = ExtendedDitherContext {
        extended: &extended,
        use_hull_tracing,
        overshoot_penalty,
        hull_error_decay,
    };

    // H2 needs different buffer dimensions (REACH=4, SEED=16), handle as early return
    if mode == DitherMode::MixedH2Standard || mode == DitherMode::MixedH2Serpentine {
        let h2_reach = H2_REACH;
        let h2_seed = H2_SEED;
        let buf_width = h2_reach + h2_seed + width + h2_seed + h2_reach;
        let buf_height = h2_seed + height + h2_reach;

        let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_a: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

        let mut r_out = vec![0u8; pixels];
        let mut g_out = vec![0u8; pixels];
        let mut b_out = vec![0u8; pixels];
        let mut a_out = vec![0u8; pixels];

        let hashed_seed = triple32(seed);

        if mode == DitherMode::MixedH2Serpentine {
            dither_mixed_h2_serpentine_paletted_extended(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, hashed_seed, progress,
            );
        } else {
            dither_mixed_h2_standard_paletted_extended(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, hashed_seed, progress,
            );
        }

        return (r_out, g_out, b_out, a_out);
    }

    // Adaptive needs same buffer dimensions as H2, handle as early return
    if mode == DitherMode::MixedAdaptive {
        let h2_reach = H2_REACH;
        let h2_seed = H2_SEED;
        let buf_width = h2_reach + h2_seed + width + h2_seed + h2_reach;
        let buf_height = h2_seed + height + h2_reach;

        let mut err_r: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_g: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_b: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];
        let mut err_a: Vec<Vec<f32>> = vec![vec![0.0f32; buf_width]; buf_height];

        let mut r_out = vec![0u8; pixels];
        let mut g_out = vec![0u8; pixels];
        let mut b_out = vec![0u8; pixels];
        let mut a_out = vec![0u8; pixels];

        let hashed_seed = triple32(seed);
        let grad_alpha_map = compute_alpha_map_paletted(r_channel, g_channel, b_channel, width, height);

        dither_mixed_adaptive_paletted_extended(
            &ctx, r_channel, g_channel, b_channel, a_channel,
            &mut err_r, &mut err_g, &mut err_b, &mut err_a,
            &mut r_out, &mut g_out, &mut b_out, &mut a_out,
            width, height, hashed_seed, &grad_alpha_map, progress,
        );

        return (r_out, g_out, b_out, a_out);
    }

    let reach = <JarvisJudiceNinke as RgbaKernel>::REACH;
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
            dither_standard_paletted_extended::<NoneKernel>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Standard => {
            dither_standard_paletted_extended::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::Serpentine => {
            dither_serpentine_paletted_extended::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisStandard => {
            dither_standard_paletted_extended::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::JarvisSerpentine => {
            dither_serpentine_paletted_extended::<JarvisJudiceNinke>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedStandard | DitherMode::MixedWangStandard | DitherMode::MixedLowbiasOldStandard => {
            dither_mixed_standard_paletted_extended(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedSerpentine | DitherMode::MixedWangSerpentine | DitherMode::MixedLowbiasOldSerpentine => {
            dither_mixed_serpentine_paletted_extended(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::MixedRandom => {
            dither_mixed_random_paletted_extended(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, hashed_seed, progress,
            );
        }
        DitherMode::OstromoukhovStandard => {
            dither_standard_paletted_extended::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::OstromoukhovSerpentine => {
            dither_serpentine_paletted_extended::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        // Zhou-Fang: fall back to Ostromoukhov for colorspace-aware dithering
        DitherMode::ZhouFangStandard => {
            dither_standard_paletted_extended::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::ZhouFangSerpentine => {
            dither_serpentine_paletted_extended::<Ostromoukhov>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        // Ulichney: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::UlichneyStandard | DitherMode::UlichneyWeightStandard => {
            dither_standard_paletted_extended::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::UlichneySerpentine | DitherMode::UlichneyWeightSerpentine => {
            dither_serpentine_paletted_extended::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        // FS+TPDF: fall back to Floyd-Steinberg for colorspace-aware dithering
        DitherMode::FsTpdfStandard => {
            dither_standard_paletted_extended::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::FsTpdfSerpentine => {
            dither_serpentine_paletted_extended::<FloydSteinberg>(
                &ctx, r_channel, g_channel, b_channel, a_channel,
                &mut err_r, &mut err_g, &mut err_b, &mut err_a,
                &mut r_out, &mut g_out, &mut b_out, &mut a_out,
                width, height, reach, progress,
            );
        }
        DitherMode::MixedH2Standard | DitherMode::MixedH2Serpentine | DitherMode::MixedAdaptive => {
            unreachable!("H2/Adaptive modes handled in early return above");
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

    let reach = <JarvisJudiceNinke as RgbaKernel>::REACH;
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
                        FloydSteinberg::apply_ltr(&mut err_r, &mut err_g, &mut err_b, &mut err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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

                        FloydSteinberg::apply_rtl(&mut err_r, &mut err_g, &mut err_b, &mut err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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

                        FloydSteinberg::apply_ltr(&mut err_r, &mut err_g, &mut err_b, &mut err_a, bx, y, err_r_val, err_g_val, err_b_val, err_a_val, r_val, g_val, b_val, a_val);
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

    // Alpha always gets error applied
    let alpha_adj = alpha + err_a_in;

    // For RGB, skip error application for fully transparent pixels
    let (lin_r_adj, lin_g_adj, lin_b_adj) = if alpha_adj <= 0.0 {
        (lin_r_orig, lin_g_orig, lin_b_orig)
    } else {
        (lin_r_orig + err_r_in, lin_g_orig + err_g_in, lin_b_orig + err_b_in)
    };

    // Use unclamped values for true distance (matching RGB/RGBA behavior)
    let alpha_clamped = alpha_adj.clamp(0.0, 1.0);
    let (target_perc_l, target_perc_a, target_perc_b) =
        linear_rgb_to_perceptual(ctx.palette.space, lin_r_adj, lin_g_adj, lin_b_adj);

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

    // Compute alpha-aware error to diffuse
    let q_err_r = lin_r_adj - best.lin_r;
    let q_err_g = lin_g_adj - best.lin_g;
    let q_err_b = lin_b_adj - best.lin_b;

    let alpha_factor = alpha_clamped;
    let one_minus_alpha = 1.0 - alpha_factor;
    let err_r_val = one_minus_alpha * err_r_in + alpha_factor * q_err_r;
    let err_g_val = one_minus_alpha * err_g_in + alpha_factor * q_err_g;
    let err_b_val = one_minus_alpha * err_b_in + alpha_factor * q_err_b;

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
