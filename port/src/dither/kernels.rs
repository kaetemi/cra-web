/// Error diffusion kernel implementations for dithering.
///
/// Provides:
/// - `SingleChannelKernel`, `RgbKernel`, `RgbaKernel`: Traits for compile-time kernel dispatch
/// - `FloydSteinberg`: Classic 2-row kernel with good speed/quality balance
/// - `JarvisJudiceNinke`: Larger 3-row kernel for smoother gradients
/// - `NoneKernel`: No-op kernel (no error diffusion)
/// - `apply_single_channel_kernel`: Runtime kernel selection for mixed-mode dithering
/// - `apply_mixed_kernel_rgb`, `apply_mixed_kernel_rgba`: Per-channel mixed-mode helpers

// ============================================================================
// Single-channel error diffusion kernel
// ============================================================================

/// Apply Floyd-Steinberg or Jarvis-Judice-Ninke error diffusion to a single channel.
///
/// This is used by mixed-mode dithering where each channel can use a different kernel.
/// The kernel is selected at runtime based on the `use_jjn` flag.
///
/// Args:
///     err: Error buffer for one channel (2D array indexed as err[y][x])
///     bx: Buffer x coordinate (includes padding for kernel reach)
///     y: Buffer y coordinate
///     err_val: Error value to diffuse
///     use_jjn: If true, use Jarvis-Judice-Ninke; if false, use Floyd-Steinberg
///     is_rtl: If true, diffuse right-to-left; if false, left-to-right
#[inline]
pub fn apply_single_channel_kernel(
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

/// Apply mixed-mode error diffusion to RGB channels (3 separate buffers).
///
/// Each channel independently selects between Floyd-Steinberg and Jarvis-Judice-Ninke
/// based on bits from the pixel_hash. This creates texture variety while maintaining
/// deterministic results from the seed.
///
/// Bit assignment: R=bit0, G=bit1, B=bit2
#[inline]
pub fn apply_mixed_kernel_rgb(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    pixel_hash: u32,
    is_rtl: bool,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;

    apply_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
}

/// Apply mixed-mode error diffusion to RGBA channels (4 separate buffers).
///
/// Each channel independently selects between Floyd-Steinberg and Jarvis-Judice-Ninke
/// based on bits from the pixel_hash. This creates texture variety while maintaining
/// deterministic results from the seed.
///
/// Bit assignment: R=bit0, G=bit1, B=bit2, A=bit3
#[inline]
pub fn apply_mixed_kernel_rgba(
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
// Compile-time kernel traits for monomorphization
// ============================================================================

/// Single-channel error diffusion kernel trait.
///
/// Enables compile-time dispatch for dithering algorithms. Each implementation
/// provides left-to-right and right-to-left variants for serpentine scanning.
///
/// Buffer structure with seeding (to normalize edge dithering):
/// ```text
/// [overshoot] [seeding] [real image] [seeding] [overshoot]
/// ```
/// - Seeding columns/rows: filled with duplicated edge pixels, ARE processed
/// - Overshoot: initialized to zero, catches error diffusion, NOT processed
pub trait SingleChannelKernel {
    /// Maximum reach of error diffusion in any direction.
    /// Floyd-Steinberg: 1, Jarvis-Judice-Ninke: 2, None: 0
    const REACH: usize;

    /// Apply kernel for left-to-right scanning.
    /// `orig` is the original pixel value (0-255 scale) before error adjustment,
    /// used by variable-coefficient kernels like Ostromoukhov.
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, orig: f32);

    /// Apply kernel for right-to-left scanning (mirrored).
    /// `orig` is the original pixel value (0-255 scale) before error adjustment,
    /// used by variable-coefficient kernels like Ostromoukhov.
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, orig: f32);

    // Derived constants for buffer layout
    /// Total left padding (overshoot + seeding)
    const TOTAL_LEFT: usize = Self::REACH * 2;
    /// Total right padding (seeding + overshoot)
    const TOTAL_RIGHT: usize = Self::REACH * 2;
    /// Total top padding (seeding only, no overshoot since error flows down)
    const TOTAL_TOP: usize = Self::REACH;
    /// Total bottom padding (overshoot only, no seeding since error comes from above)
    const TOTAL_BOTTOM: usize = Self::REACH;
    /// Offset from buffer edge to start of seeding area (= overshoot size)
    const SEED_OFFSET: usize = Self::REACH;
}

/// RGB (3-channel) error diffusion kernel trait.
///
/// Applies the same kernel pattern to three separate error buffers.
pub trait RgbKernel {
    /// Maximum reach of error diffusion in any direction.
    const REACH: usize;

    /// Apply kernel to RGB channels for left-to-right scanning.
    /// `orig_r/g/b` are the original pixel values (0-255 scale) before error adjustment,
    /// used by variable-coefficient kernels like Ostromoukhov.
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
    );

    /// Apply kernel to RGB channels for right-to-left scanning.
    /// `orig_r/g/b` are the original pixel values (0-255 scale) before error adjustment,
    /// used by variable-coefficient kernels like Ostromoukhov.
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
    );

    // Derived constants for buffer layout
    const TOTAL_LEFT: usize = Self::REACH * 2;
    const TOTAL_RIGHT: usize = Self::REACH * 2;
    const TOTAL_TOP: usize = Self::REACH;
    const TOTAL_BOTTOM: usize = Self::REACH;
    const SEED_OFFSET: usize = Self::REACH;
}

/// RGBA (4-channel) error diffusion kernel trait.
///
/// Applies the same kernel pattern to four separate error buffers.
pub trait RgbaKernel {
    /// Maximum reach of error diffusion in any direction.
    const REACH: usize;

    /// Apply kernel to RGBA channels for left-to-right scanning.
    /// `orig_r/g/b/a` are the original pixel values (0-255 scale) before error adjustment,
    /// used by variable-coefficient kernels like Ostromoukhov.
    fn apply_ltr(
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
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
        orig_a: f32,
    );

    /// Apply kernel to RGBA channels for right-to-left scanning.
    /// `orig_r/g/b/a` are the original pixel values (0-255 scale) before error adjustment,
    /// used by variable-coefficient kernels like Ostromoukhov.
    fn apply_rtl(
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
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
        orig_a: f32,
    );

    // Derived constants for buffer layout
    const TOTAL_LEFT: usize = Self::REACH * 2;
    const TOTAL_RIGHT: usize = Self::REACH * 2;
    const TOTAL_TOP: usize = Self::REACH;
    const TOTAL_BOTTOM: usize = Self::REACH;
    const SEED_OFFSET: usize = Self::REACH;
}

// ============================================================================
// Floyd-Steinberg kernel implementation
// ============================================================================

/// Floyd-Steinberg error diffusion kernel.
///
/// Compact 2-row kernel with good speed/quality trade-off.
/// Kernel weights (divided by 16):
/// ```text
///       * 7
///     3 5 1
/// ```
pub struct FloydSteinberg;

impl SingleChannelKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, _orig: f32) {
        apply_single_channel_kernel(buf, bx, y, err, false, false);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, _orig: f32) {
        apply_single_channel_kernel(buf, bx, y, err, false, true);
    }
}

impl RgbKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val, 0.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val, 0.0);
    }
}

impl RgbaKernel for FloydSteinberg {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
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
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
        _orig_a: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_a, bx, y, err_a_val, 0.0);
    }

    #[inline]
    fn apply_rtl(
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
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
        _orig_a: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_a, bx, y, err_a_val, 0.0);
    }
}

// ============================================================================
// Jarvis-Judice-Ninke kernel implementation
// ============================================================================

/// Jarvis-Judice-Ninke error diffusion kernel.
///
/// Larger 3-row kernel produces smoother gradients than Floyd-Steinberg.
/// Kernel weights (divided by 48):
/// ```text
///         * 7 5
///     3 5 7 5 3
///     1 3 5 3 1
/// ```
pub struct JarvisJudiceNinke;

impl SingleChannelKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, _orig: f32) {
        apply_single_channel_kernel(buf, bx, y, err, true, false);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, _orig: f32) {
        apply_single_channel_kernel(buf, bx, y, err, true, true);
    }
}

impl RgbKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val, 0.0);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val, 0.0);
    }
}

impl RgbaKernel for JarvisJudiceNinke {
    const REACH: usize = 2;

    #[inline]
    fn apply_ltr(
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
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
        _orig_a: f32,
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val, 0.0);
        <Self as SingleChannelKernel>::apply_ltr(err_a, bx, y, err_a_val, 0.0);
    }

    #[inline]
    fn apply_rtl(
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
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
        _orig_a: f32,
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val, 0.0);
        <Self as SingleChannelKernel>::apply_rtl(err_a, bx, y, err_a_val, 0.0);
    }
}

// ============================================================================
// Ostromoukhov kernel implementation
// ============================================================================

/// Ostromoukhov variable-coefficient error diffusion kernel.
///
/// Uses input intensity to select coefficients from a lookup table.
/// Produces visually pleasing results with reduced artifacts compared to
/// fixed-coefficient kernels. Same footprint as Floyd-Steinberg (REACH=1).
///
/// Reference: Victor Ostromoukhov, "A Simple and Efficient Error-Diffusion
/// Algorithm", SIGGRAPH 2001.
pub struct Ostromoukhov;

/// Coefficient table for Ostromoukhov dithering (128 entries).
/// Each entry is [A10, A11, A01] where:
/// - A10: coefficient for next pixel in scan direction
/// - A11: coefficient for diagonal (behind scan direction, next row)
/// - A01: coefficient for directly below
/// Actual weight = A / (A10 + A11 + A01)
/// For levels 128-255, use entry (255 - level) with mirrored directions.
const OSTRO_TABLE: [[i32; 3]; 128] = [
    [13, 0, 5],      // 0
    [13, 0, 5],      // 1
    [21, 0, 10],     // 2
    [7, 0, 4],       // 3
    [8, 0, 5],       // 4
    [47, 3, 28],     // 5
    [23, 3, 13],     // 6
    [15, 3, 8],      // 7
    [22, 6, 11],     // 8
    [43, 15, 20],    // 9
    [7, 3, 3],       // 10
    [501, 224, 211], // 11
    [249, 116, 103], // 12
    [165, 80, 67],   // 13
    [123, 62, 49],   // 14
    [489, 256, 191], // 15
    [81, 44, 31],    // 16
    [483, 272, 181], // 17
    [60, 35, 22],    // 18
    [53, 32, 19],    // 19
    [237, 148, 83],  // 20
    [471, 304, 161], // 21
    [3, 2, 1],       // 22
    [481, 314, 185], // 23
    [354, 226, 155], // 24
    [1389, 866, 685], // 25
    [227, 138, 125], // 26
    [267, 158, 163], // 27
    [327, 188, 220], // 28
    [61, 34, 45],    // 29
    [627, 338, 505], // 30
    [1227, 638, 1075], // 31
    [20, 10, 19],    // 32
    [1937, 1000, 1767], // 33
    [977, 520, 855], // 34
    [657, 360, 551], // 35
    [71, 40, 57],    // 36
    [2005, 1160, 1539], // 37
    [337, 200, 247], // 38
    [2039, 1240, 1425], // 39
    [257, 160, 171], // 40
    [691, 440, 437], // 41
    [1045, 680, 627], // 42
    [301, 200, 171], // 43
    [177, 120, 95],  // 44
    [2141, 1480, 1083], // 45
    [1079, 760, 513], // 46
    [725, 520, 323], // 47
    [137, 100, 57],  // 48
    [2209, 1640, 855], // 49
    [53, 40, 19],    // 50
    [2243, 1720, 741], // 51
    [565, 440, 171], // 52
    [759, 600, 209], // 53
    [1147, 920, 285], // 54
    [2311, 1880, 513], // 55
    [97, 80, 19],    // 56
    [335, 280, 57],  // 57
    [1181, 1000, 171], // 58
    [793, 680, 95],  // 59
    [599, 520, 57],  // 60
    [2413, 2120, 171], // 61
    [405, 360, 19],  // 62
    [2447, 2200, 57], // 63
    [11, 10, 0],     // 64
    [158, 151, 3],   // 65
    [178, 179, 7],   // 66
    [1030, 1091, 63], // 67
    [248, 277, 21],  // 68
    [318, 375, 35],  // 69
    [458, 571, 63],  // 70
    [878, 1159, 147], // 71
    [5, 7, 1],       // 72
    [172, 181, 37],  // 73
    [97, 76, 22],    // 74
    [72, 41, 17],    // 75
    [119, 47, 29],   // 76
    [4, 1, 1],       // 77
    [4, 1, 1],       // 78
    [4, 1, 1],       // 79
    [4, 1, 1],       // 80
    [4, 1, 1],       // 81
    [4, 1, 1],       // 82
    [4, 1, 1],       // 83
    [4, 1, 1],       // 84
    [4, 1, 1],       // 85
    [65, 18, 17],    // 86
    [95, 29, 26],    // 87
    [185, 62, 53],   // 88
    [30, 11, 9],     // 89
    [35, 14, 11],    // 90
    [85, 37, 28],    // 91
    [55, 26, 19],    // 92
    [80, 41, 29],    // 93
    [155, 86, 59],   // 94
    [5, 3, 2],       // 95
    [5, 3, 2],       // 96
    [5, 3, 2],       // 97
    [5, 3, 2],       // 98
    [5, 3, 2],       // 99
    [5, 3, 2],       // 100
    [5, 3, 2],       // 101
    [5, 3, 2],       // 102
    [5, 3, 2],       // 103
    [5, 3, 2],       // 104
    [5, 3, 2],       // 105
    [5, 3, 2],       // 106
    [5, 3, 2],       // 107
    [305, 176, 119], // 108
    [155, 86, 59],   // 109
    [105, 56, 39],   // 110
    [80, 41, 29],    // 111
    [65, 32, 23],    // 112
    [55, 26, 19],    // 113
    [335, 152, 113], // 114
    [85, 37, 28],    // 115
    [115, 48, 37],   // 116
    [35, 14, 11],    // 117
    [355, 136, 109], // 118
    [30, 11, 9],     // 119
    [365, 128, 107], // 120
    [185, 62, 53],   // 121
    [25, 8, 7],      // 122
    [95, 29, 26],    // 123
    [385, 112, 103], // 124
    [65, 18, 17],    // 125
    [395, 104, 101], // 126
    [4, 1, 1],       // 127
];

/// Get Ostromoukhov coefficients for a given input level (0-255 scale).
/// Returns (d10, d11, d01) normalized coefficients.
/// Values outside 0-255 are clamped.
#[inline]
fn ostro_coefficients(original_value: f32) -> (f32, f32, f32) {
    // Clamp to 0-255 range
    let level = (original_value + 0.5) as i32;
    let level = level.clamp(0, 255);

    // Mirror for levels > 127
    let idx = if level > 127 {
        255 - level
    } else {
        level
    };

    let coeffs = OSTRO_TABLE[idx as usize];
    let sum = (coeffs[0] + coeffs[1] + coeffs[2]) as f32;

    let d10 = coeffs[0] as f32 / sum;
    let d11 = coeffs[1] as f32 / sum;
    let d01 = coeffs[2] as f32 / sum;

    (d10, d11, d01)
}

/// Apply Ostromoukhov kernel to a single channel (left-to-right).
/// `original` is the original pixel value in 0-255 scale.
#[inline]
pub fn apply_ostromoukhov_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, original: f32) {
    let (d10, d11, d01) = ostro_coefficients(original);

    // LTR: next=bx+1, diagonal_behind=bx-1, below=bx
    buf[y][bx + 1] += err * d10;     // next pixel in scan direction
    buf[y + 1][bx - 1] += err * d11; // diagonal behind
    buf[y + 1][bx] += err * d01;     // directly below
}

/// Apply Ostromoukhov kernel to a single channel (right-to-left).
/// `original` is the original pixel value in 0-255 scale.
#[inline]
pub fn apply_ostromoukhov_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, original: f32) {
    let (d10, d11, d01) = ostro_coefficients(original);

    // RTL: next=bx-1, diagonal_behind=bx+1, below=bx
    buf[y][bx - 1] += err * d10;     // next pixel in scan direction
    buf[y + 1][bx + 1] += err * d11; // diagonal behind
    buf[y + 1][bx] += err * d01;     // directly below
}

/// Apply Ostromoukhov kernel to RGB channels (left-to-right).
#[inline]
pub fn apply_ostromoukhov_rgb_ltr(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    orig_r: f32,
    orig_g: f32,
    orig_b: f32,
) {
    apply_ostromoukhov_ltr(err_r, bx, y, err_r_val, orig_r);
    apply_ostromoukhov_ltr(err_g, bx, y, err_g_val, orig_g);
    apply_ostromoukhov_ltr(err_b, bx, y, err_b_val, orig_b);
}

/// Apply Ostromoukhov kernel to RGB channels (right-to-left).
#[inline]
pub fn apply_ostromoukhov_rgb_rtl(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    orig_r: f32,
    orig_g: f32,
    orig_b: f32,
) {
    apply_ostromoukhov_rtl(err_r, bx, y, err_r_val, orig_r);
    apply_ostromoukhov_rtl(err_g, bx, y, err_g_val, orig_g);
    apply_ostromoukhov_rtl(err_b, bx, y, err_b_val, orig_b);
}

/// Apply Ostromoukhov kernel to RGBA channels (left-to-right).
#[inline]
pub fn apply_ostromoukhov_rgba_ltr(
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
    orig_r: f32,
    orig_g: f32,
    orig_b: f32,
    orig_a: f32,
) {
    apply_ostromoukhov_ltr(err_r, bx, y, err_r_val, orig_r);
    apply_ostromoukhov_ltr(err_g, bx, y, err_g_val, orig_g);
    apply_ostromoukhov_ltr(err_b, bx, y, err_b_val, orig_b);
    apply_ostromoukhov_ltr(err_a, bx, y, err_a_val, orig_a);
}

/// Apply Ostromoukhov kernel to RGBA channels (right-to-left).
#[inline]
pub fn apply_ostromoukhov_rgba_rtl(
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
    orig_r: f32,
    orig_g: f32,
    orig_b: f32,
    orig_a: f32,
) {
    apply_ostromoukhov_rtl(err_r, bx, y, err_r_val, orig_r);
    apply_ostromoukhov_rtl(err_g, bx, y, err_g_val, orig_g);
    apply_ostromoukhov_rtl(err_b, bx, y, err_b_val, orig_b);
    apply_ostromoukhov_rtl(err_a, bx, y, err_a_val, orig_a);
}

impl SingleChannelKernel for Ostromoukhov {
    const REACH: usize = 1; // Same footprint as Floyd-Steinberg

    #[inline]
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, orig: f32) {
        apply_ostromoukhov_ltr(buf, bx, y, err, orig);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32, orig: f32) {
        apply_ostromoukhov_rtl(buf, bx, y, err, orig);
    }
}

impl RgbKernel for Ostromoukhov {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
    ) {
        apply_ostromoukhov_rgb_ltr(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, orig_r, orig_g, orig_b);
    }

    #[inline]
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
    ) {
        apply_ostromoukhov_rgb_rtl(err_r, err_g, err_b, bx, y, err_r_val, err_g_val, err_b_val, orig_r, orig_g, orig_b);
    }
}

impl RgbaKernel for Ostromoukhov {
    const REACH: usize = 1;

    #[inline]
    fn apply_ltr(
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
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
        orig_a: f32,
    ) {
        apply_ostromoukhov_rgba_ltr(
            err_r, err_g, err_b, err_a, bx, y,
            err_r_val, err_g_val, err_b_val, err_a_val,
            orig_r, orig_g, orig_b, orig_a,
        );
    }

    #[inline]
    fn apply_rtl(
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
        orig_r: f32,
        orig_g: f32,
        orig_b: f32,
        orig_a: f32,
    ) {
        apply_ostromoukhov_rgba_rtl(
            err_r, err_g, err_b, err_a, bx, y,
            err_r_val, err_g_val, err_b_val, err_a_val,
            orig_r, orig_g, orig_b, orig_a,
        );
    }
}

// ============================================================================
// No-op kernel implementation
// ============================================================================

/// No-op kernel that discards error (no diffusion).
///
/// Each pixel is independently quantized to nearest level.
/// Produces banding but useful as a baseline for comparison.
pub struct NoneKernel;

impl SingleChannelKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(_buf: &mut [Vec<f32>], _bx: usize, _y: usize, _err: f32, _orig: f32) {}

    #[inline]
    fn apply_rtl(_buf: &mut [Vec<f32>], _bx: usize, _y: usize, _err: f32, _orig: f32) {}
}

impl RgbKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
    ) {
    }

    #[inline]
    fn apply_rtl(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
    ) {
    }
}

impl RgbaKernel for NoneKernel {
    const REACH: usize = 0;

    #[inline]
    fn apply_ltr(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _err_a: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
        _err_a_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
        _orig_a: f32,
    ) {
    }

    #[inline]
    fn apply_rtl(
        _err_r: &mut [Vec<f32>],
        _err_g: &mut [Vec<f32>],
        _err_b: &mut [Vec<f32>],
        _err_a: &mut [Vec<f32>],
        _bx: usize,
        _y: usize,
        _err_r_val: f32,
        _err_g_val: f32,
        _err_b_val: f32,
        _err_a_val: f32,
        _orig_r: f32,
        _orig_g: f32,
        _orig_b: f32,
        _orig_a: f32,
    ) {
    }
}

// ============================================================================
// Zhou-Fang dithering tables and functions
// ============================================================================

/// Zhou-Fang variable-coefficient error diffusion with threshold modulation.
///
/// Key difference from Ostromoukhov: Zhou-Fang modulates the quantization
/// threshold based on position and input intensity, breaking up "worm" patterns.
///
/// Reference: Zhou & Fang, "Improving Mid-tone Quality of Variable-coefficient
/// Error Diffusion Using Threshold Modulation", SIGGRAPH 2003.

/// Key levels for coefficient interpolation
const ZF_COEFF_KEY_LEVELS: [i32; 18] = [0, 1, 2, 3, 4, 10, 22, 32, 44, 64, 72, 77, 85, 95, 102, 107, 112, 127];

/// Key coefficient values [A10, A11, A01] at each key level
const ZF_COEFF_KEY_VALUES: [[i32; 3]; 18] = [
    [13, 0, 5],              // 0
    [1300249, 0, 499250],    // 1
    [214114, 287, 99357],    // 2
    [351854, 0, 199965],     // 3
    [801100, 0, 490999],     // 4
    [704075, 297466, 303694], // 10
    [46613, 31917, 21469],   // 22
    [47482, 30617, 21900],   // 32
    [43024, 42131, 14826],   // 44
    [36411, 43219, 20369],   // 64
    [38477, 53843, 7678],    // 72
    [40503, 51547, 7948],    // 77
    [35865, 34108, 30026],   // 85
    [34117, 36899, 28983],   // 95
    [35464, 35049, 29485],   // 102
    [16477, 18810, 14712],   // 107
    [33360, 37954, 28685],   // 112
    [35269, 36066, 28664],   // 127
];

/// Key levels for modulation strength interpolation
const ZF_MOD_KEY_LEVELS: [i32; 9] = [0, 44, 64, 85, 95, 102, 107, 112, 127];

/// Key modulation strength values (stored as fixed-point, multiply by 1/1000)
const ZF_MOD_KEY_VALUES: [i32; 9] = [0, 340, 500, 1000, 170, 500, 700, 790, 1000];

/// Get Zhou-Fang coefficients for a given input level (0-255 scale).
/// Returns (d10, d11, d01) normalized coefficients.
#[inline]
pub fn zhou_fang_coefficients(level: i32) -> (f32, f32, f32) {
    let level = level.clamp(0, 255);

    // Mirror for levels > 127
    let idx = if level > 127 { 255 - level } else { level };

    // Find surrounding key levels
    let mut lo = 0usize;
    while lo + 1 < ZF_COEFF_KEY_LEVELS.len() && ZF_COEFF_KEY_LEVELS[lo + 1] <= idx {
        lo += 1;
    }
    let hi = (lo + 1).min(ZF_COEFF_KEY_LEVELS.len() - 1);

    let key_lo = ZF_COEFF_KEY_LEVELS[lo];
    let key_hi = ZF_COEFF_KEY_LEVELS[hi];

    // Interpolate
    let (a10, a11, a01) = if key_hi == key_lo {
        // Exact match
        (
            ZF_COEFF_KEY_VALUES[lo][0] as f32,
            ZF_COEFF_KEY_VALUES[lo][1] as f32,
            ZF_COEFF_KEY_VALUES[lo][2] as f32,
        )
    } else {
        let t = (idx - key_lo) as f32 / (key_hi - key_lo) as f32;
        (
            ZF_COEFF_KEY_VALUES[lo][0] as f32 + t * (ZF_COEFF_KEY_VALUES[hi][0] - ZF_COEFF_KEY_VALUES[lo][0]) as f32,
            ZF_COEFF_KEY_VALUES[lo][1] as f32 + t * (ZF_COEFF_KEY_VALUES[hi][1] - ZF_COEFF_KEY_VALUES[lo][1]) as f32,
            ZF_COEFF_KEY_VALUES[lo][2] as f32 + t * (ZF_COEFF_KEY_VALUES[hi][2] - ZF_COEFF_KEY_VALUES[lo][2]) as f32,
        )
    };

    let sum = a10 + a11 + a01;
    if sum > 0.0 {
        (a10 / sum, a11 / sum, a01 / sum)
    } else {
        (1.0, 0.0, 0.0) // Fallback
    }
}

/// Get Zhou-Fang modulation strength for a given input level (0-255 scale).
/// Returns modulation factor in range [0.0, 1.0].
#[inline]
pub fn zhou_fang_modulation(level: i32) -> f32 {
    let level = level.clamp(0, 255);

    // Mirror for levels > 127
    let idx = if level > 127 { 255 - level } else { level };

    // Find surrounding key levels
    let mut lo = 0usize;
    while lo + 1 < ZF_MOD_KEY_LEVELS.len() && ZF_MOD_KEY_LEVELS[lo + 1] <= idx {
        lo += 1;
    }
    let hi = (lo + 1).min(ZF_MOD_KEY_LEVELS.len() - 1);

    let key_lo = ZF_MOD_KEY_LEVELS[lo];
    let key_hi = ZF_MOD_KEY_LEVELS[hi];

    // Interpolate
    let m = if key_hi == key_lo {
        ZF_MOD_KEY_VALUES[lo] as f32
    } else {
        let t = (idx - key_lo) as f32 / (key_hi - key_lo) as f32;
        ZF_MOD_KEY_VALUES[lo] as f32 + t * (ZF_MOD_KEY_VALUES[hi] - ZF_MOD_KEY_VALUES[lo]) as f32
    };

    m / 1000.0 // Convert from fixed-point
}

/// Apply Zhou-Fang kernel to a single channel (left-to-right).
/// Uses position hash for threshold modulation.
#[inline]
pub fn apply_zhou_fang_ltr(
    buf: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err: f32,
    level: i32,
) {
    let (d10, d11, d01) = zhou_fang_coefficients(level);

    // LTR: next=bx+1, diagonal_behind=bx-1, below=bx
    buf[y][bx + 1] += err * d10;
    buf[y + 1][bx - 1] += err * d11;
    buf[y + 1][bx] += err * d01;
}

/// Apply Zhou-Fang kernel to a single channel (right-to-left).
#[inline]
pub fn apply_zhou_fang_rtl(
    buf: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err: f32,
    level: i32,
) {
    let (d10, d11, d01) = zhou_fang_coefficients(level);

    // RTL: next=bx-1, diagonal_behind=bx+1, below=bx
    buf[y][bx - 1] += err * d10;
    buf[y + 1][bx + 1] += err * d11;
    buf[y + 1][bx] += err * d01;
}

/// Compute modulated threshold for Zhou-Fang dithering.
/// Uses position-based noise to break up patterns in mid-tones.
///
/// Arguments:
///   - level: input intensity (0-255)
///   - x, y: pixel coordinates for noise generation
///   - seed: random seed for reproducibility
///
/// Returns: threshold in [0.0, 1.0] range (typically near 0.5)
#[inline]
pub fn zhou_fang_threshold(level: i32, x: usize, y: usize, seed: u32) -> f32 {
    use super::common::wang_hash;

    let m = zhou_fang_modulation(level);

    // Generate position-based noise in [0, 127]
    let pixel_hash = wang_hash((x as u32) ^ ((y as u32) << 16) ^ seed);
    let noise = (pixel_hash & 127) as f32 / 127.0; // [0.0, 1.0]

    // Modulate threshold: 0.5 + (noise - 0.5) * m
    // This gives threshold in range [0.5 - m/2, 0.5 + m/2]
    0.5 + (noise - 0.5) * m
}

/// Apply Zhou-Fang kernel to RGB channels (left-to-right).
#[inline]
pub fn apply_zhou_fang_rgb_ltr(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    level_r: i32,
    level_g: i32,
    level_b: i32,
) {
    apply_zhou_fang_ltr(err_r, bx, y, err_r_val, level_r);
    apply_zhou_fang_ltr(err_g, bx, y, err_g_val, level_g);
    apply_zhou_fang_ltr(err_b, bx, y, err_b_val, level_b);
}

/// Apply Zhou-Fang kernel to RGB channels (right-to-left).
#[inline]
pub fn apply_zhou_fang_rgb_rtl(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    level_r: i32,
    level_g: i32,
    level_b: i32,
) {
    apply_zhou_fang_rtl(err_r, bx, y, err_r_val, level_r);
    apply_zhou_fang_rtl(err_g, bx, y, err_g_val, level_g);
    apply_zhou_fang_rtl(err_b, bx, y, err_b_val, level_b);
}

/// Apply Zhou-Fang kernel to RGBA channels (left-to-right).
#[inline]
pub fn apply_zhou_fang_rgba_ltr(
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
    level_r: i32,
    level_g: i32,
    level_b: i32,
    level_a: i32,
) {
    apply_zhou_fang_ltr(err_r, bx, y, err_r_val, level_r);
    apply_zhou_fang_ltr(err_g, bx, y, err_g_val, level_g);
    apply_zhou_fang_ltr(err_b, bx, y, err_b_val, level_b);
    apply_zhou_fang_ltr(err_a, bx, y, err_a_val, level_a);
}

/// Apply Zhou-Fang kernel to RGBA channels (right-to-left).
#[inline]
pub fn apply_zhou_fang_rgba_rtl(
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
    level_r: i32,
    level_g: i32,
    level_b: i32,
    level_a: i32,
) {
    apply_zhou_fang_rtl(err_r, bx, y, err_r_val, level_r);
    apply_zhou_fang_rtl(err_g, bx, y, err_g_val, level_g);
    apply_zhou_fang_rtl(err_b, bx, y, err_b_val, level_b);
    apply_zhou_fang_rtl(err_a, bx, y, err_a_val, level_a);
}

// ============================================================================
// Second-order (H²) kernel implementations: FS² and JJN²
// ============================================================================

/// Maximum reach for H2 kernels (JJN² spans ±4 columns, 5 rows down).
pub const H2_REACH: usize = 4;

/// Seeding size for H2 buffers. Larger than standard to accommodate negative
/// weights that can propagate error backwards into the seeding region.
pub const H2_SEED: usize = 16;

/// Apply FS² (second-order Floyd-Steinberg) kernel, LTR only.
///
/// Kernel = 2·H_fs - H_fs², denominator 256, reach 2.
/// 11 taps across 3 rows. Contains negative weights.
///
/// ```text
///               *  224  -49
///          96  118  -38  -14
///     -9  -30  -31  -10   -1
/// ```
#[inline]
pub fn apply_fs2_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
    // Row 0 (dy=0)
    buf[y][bx + 1] += err * (224.0 / 256.0);
    buf[y][bx + 2] += err * (-49.0 / 256.0);
    // Row 1 (dy=1)
    buf[y + 1][bx - 1] += err * (96.0 / 256.0);
    buf[y + 1][bx] += err * (118.0 / 256.0);
    buf[y + 1][bx + 1] += err * (-38.0 / 256.0);
    buf[y + 1][bx + 2] += err * (-14.0 / 256.0);
    // Row 2 (dy=2)
    buf[y + 2][bx - 2] += err * (-9.0 / 256.0);
    buf[y + 2][bx - 1] += err * (-30.0 / 256.0);
    buf[y + 2][bx] += err * (-31.0 / 256.0);
    buf[y + 2][bx + 1] += err * (-10.0 / 256.0);
    buf[y + 2][bx + 2] += err * (-1.0 / 256.0);
}

/// Apply JJN² (second-order Jarvis-Judice-Ninke) kernel, LTR only.
///
/// Kernel = 2·H_jjn - H_jjn², denominator 2304, reach 4.
/// 38 taps across 5 rows. Contains negative weights.
/// Rows 3-4 are entirely negative (correction from -H² term).
#[inline]
pub fn apply_jjn2_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
    // Row 0 (dy=0): 4 taps
    buf[y][bx + 1] += err * (672.0 / 2304.0);
    buf[y][bx + 2] += err * (431.0 / 2304.0);
    buf[y][bx + 3] += err * (-70.0 / 2304.0);
    buf[y][bx + 4] += err * (-25.0 / 2304.0);
    // Row 1 (dy=1): 7 taps
    buf[y + 1][bx - 2] += err * (288.0 / 2304.0);
    buf[y + 1][bx - 1] += err * (438.0 / 2304.0);
    buf[y + 1][bx] += err * (572.0 / 2304.0);
    buf[y + 1][bx + 1] += err * (332.0 / 2304.0);
    buf[y + 1][bx + 2] += err * (148.0 / 2304.0);
    buf[y + 1][bx + 3] += err * (-92.0 / 2304.0);
    buf[y + 1][bx + 4] += err * (-30.0 / 2304.0);
    // Row 2 (dy=2): 9 taps
    buf[y + 2][bx - 4] += err * (-9.0 / 2304.0);
    buf[y + 2][bx - 3] += err * (-30.0 / 2304.0);
    buf[y + 2][bx - 2] += err * (29.0 / 2304.0);
    buf[y + 2][bx - 1] += err * (174.0 / 2304.0);
    buf[y + 2][bx] += err * (311.0 / 2304.0);
    buf[y + 2][bx + 1] += err * (88.0 / 2304.0);
    buf[y + 2][bx + 2] += err * (-63.0 / 2304.0);
    buf[y + 2][bx + 3] += err * (-74.0 / 2304.0);
    buf[y + 2][bx + 4] += err * (-19.0 / 2304.0);
    // Row 3 (dy=3): 9 taps (symmetric, all negative)
    buf[y + 3][bx - 4] += err * (-6.0 / 2304.0);
    buf[y + 3][bx - 3] += err * (-28.0 / 2304.0);
    buf[y + 3][bx - 2] += err * (-74.0 / 2304.0);
    buf[y + 3][bx - 1] += err * (-120.0 / 2304.0);
    buf[y + 3][bx] += err * (-142.0 / 2304.0);
    buf[y + 3][bx + 1] += err * (-120.0 / 2304.0);
    buf[y + 3][bx + 2] += err * (-74.0 / 2304.0);
    buf[y + 3][bx + 3] += err * (-28.0 / 2304.0);
    buf[y + 3][bx + 4] += err * (-6.0 / 2304.0);
    // Row 4 (dy=4): 9 taps (symmetric, all negative)
    buf[y + 4][bx - 4] += err * (-1.0 / 2304.0);
    buf[y + 4][bx - 3] += err * (-6.0 / 2304.0);
    buf[y + 4][bx - 2] += err * (-19.0 / 2304.0);
    buf[y + 4][bx - 1] += err * (-36.0 / 2304.0);
    buf[y + 4][bx] += err * (-45.0 / 2304.0);
    buf[y + 4][bx + 1] += err * (-36.0 / 2304.0);
    buf[y + 4][bx + 2] += err * (-19.0 / 2304.0);
    buf[y + 4][bx + 3] += err * (-6.0 / 2304.0);
    buf[y + 4][bx + 4] += err * (-1.0 / 2304.0);
}

/// Apply FS² (second-order Floyd-Steinberg) kernel, RTL (mirrored).
///
/// Mirror of `apply_fs2_ltr`: all horizontal offsets negated.
/// 11 taps across 3 rows. Contains negative weights.
///
/// ```text
///     -49  224   *
///     -14  -38  118   96
///      -1  -10  -31  -30   -9
/// ```
#[inline]
pub fn apply_fs2_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
    // Row 0 (dy=0)
    buf[y][bx - 1] += err * (224.0 / 256.0);
    buf[y][bx - 2] += err * (-49.0 / 256.0);
    // Row 1 (dy=1)
    buf[y + 1][bx + 1] += err * (96.0 / 256.0);
    buf[y + 1][bx] += err * (118.0 / 256.0);
    buf[y + 1][bx - 1] += err * (-38.0 / 256.0);
    buf[y + 1][bx - 2] += err * (-14.0 / 256.0);
    // Row 2 (dy=2)
    buf[y + 2][bx + 2] += err * (-9.0 / 256.0);
    buf[y + 2][bx + 1] += err * (-30.0 / 256.0);
    buf[y + 2][bx] += err * (-31.0 / 256.0);
    buf[y + 2][bx - 1] += err * (-10.0 / 256.0);
    buf[y + 2][bx - 2] += err * (-1.0 / 256.0);
}

/// Apply JJN² (second-order Jarvis-Judice-Ninke) kernel, RTL (mirrored).
///
/// Mirror of `apply_jjn2_ltr`: all horizontal offsets negated.
/// 38 taps across 5 rows. Contains negative weights.
#[inline]
pub fn apply_jjn2_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
    // Row 0 (dy=0): 4 taps
    buf[y][bx - 1] += err * (672.0 / 2304.0);
    buf[y][bx - 2] += err * (431.0 / 2304.0);
    buf[y][bx - 3] += err * (-70.0 / 2304.0);
    buf[y][bx - 4] += err * (-25.0 / 2304.0);
    // Row 1 (dy=1): 7 taps
    buf[y + 1][bx + 2] += err * (288.0 / 2304.0);
    buf[y + 1][bx + 1] += err * (438.0 / 2304.0);
    buf[y + 1][bx] += err * (572.0 / 2304.0);
    buf[y + 1][bx - 1] += err * (332.0 / 2304.0);
    buf[y + 1][bx - 2] += err * (148.0 / 2304.0);
    buf[y + 1][bx - 3] += err * (-92.0 / 2304.0);
    buf[y + 1][bx - 4] += err * (-30.0 / 2304.0);
    // Row 2 (dy=2): 9 taps
    buf[y + 2][bx + 4] += err * (-9.0 / 2304.0);
    buf[y + 2][bx + 3] += err * (-30.0 / 2304.0);
    buf[y + 2][bx + 2] += err * (29.0 / 2304.0);
    buf[y + 2][bx + 1] += err * (174.0 / 2304.0);
    buf[y + 2][bx] += err * (311.0 / 2304.0);
    buf[y + 2][bx - 1] += err * (88.0 / 2304.0);
    buf[y + 2][bx - 2] += err * (-63.0 / 2304.0);
    buf[y + 2][bx - 3] += err * (-74.0 / 2304.0);
    buf[y + 2][bx - 4] += err * (-19.0 / 2304.0);
    // Row 3 (dy=3): 9 taps (symmetric, all negative — same as LTR)
    buf[y + 3][bx - 4] += err * (-6.0 / 2304.0);
    buf[y + 3][bx - 3] += err * (-28.0 / 2304.0);
    buf[y + 3][bx - 2] += err * (-74.0 / 2304.0);
    buf[y + 3][bx - 1] += err * (-120.0 / 2304.0);
    buf[y + 3][bx] += err * (-142.0 / 2304.0);
    buf[y + 3][bx + 1] += err * (-120.0 / 2304.0);
    buf[y + 3][bx + 2] += err * (-74.0 / 2304.0);
    buf[y + 3][bx + 3] += err * (-28.0 / 2304.0);
    buf[y + 3][bx + 4] += err * (-6.0 / 2304.0);
    // Row 4 (dy=4): 9 taps (symmetric, all negative — same as LTR)
    buf[y + 4][bx - 4] += err * (-1.0 / 2304.0);
    buf[y + 4][bx - 3] += err * (-6.0 / 2304.0);
    buf[y + 4][bx - 2] += err * (-19.0 / 2304.0);
    buf[y + 4][bx - 1] += err * (-36.0 / 2304.0);
    buf[y + 4][bx] += err * (-45.0 / 2304.0);
    buf[y + 4][bx + 1] += err * (-36.0 / 2304.0);
    buf[y + 4][bx + 2] += err * (-19.0 / 2304.0);
    buf[y + 4][bx + 3] += err * (-6.0 / 2304.0);
    buf[y + 4][bx + 4] += err * (-1.0 / 2304.0);
}

/// Apply H2 kernel to a single channel.
/// Dispatches to FS² or JJN² based on the `use_jjn` flag,
/// and to LTR or RTL based on `is_rtl`.
#[inline]
pub fn apply_h2_single_channel_kernel(
    buf: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err: f32,
    use_jjn: bool,
    is_rtl: bool,
) {
    match (use_jjn, is_rtl) {
        (true, false) => apply_jjn2_ltr(buf, bx, y, err),
        (true, true) => apply_jjn2_rtl(buf, bx, y, err),
        (false, false) => apply_fs2_ltr(buf, bx, y, err),
        (false, true) => apply_fs2_rtl(buf, bx, y, err),
    }
}

/// Apply H2 kernel to RGB channels.
/// Each channel independently selects between FS² and JJN²
/// based on bits from the pixel_hash. Bit assignment: R=bit0, G=bit1, B=bit2.
#[inline]
pub fn apply_h2_kernel_rgb(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    pixel_hash: u32,
    is_rtl: bool,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;

    apply_h2_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_h2_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_h2_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
}

/// Apply H2 kernel to RGBA channels.
/// Each channel independently selects between FS² and JJN²
/// based on bits from the pixel_hash. Bit assignment: R=bit0, G=bit1, B=bit2, A=bit3.
#[inline]
pub fn apply_h2_kernel_rgba(
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

    apply_h2_single_channel_kernel(err_r, bx, y, err_r_val, use_jjn_r, is_rtl);
    apply_h2_single_channel_kernel(err_g, bx, y, err_g_val, use_jjn_g, is_rtl);
    apply_h2_single_channel_kernel(err_b, bx, y, err_b_val, use_jjn_b, is_rtl);
    apply_h2_single_channel_kernel(err_a, bx, y, err_a_val, use_jjn_a, is_rtl);
}

// ============================================================================
// Adaptive (gradient-blended 1st+2nd order) kernel implementations
// ============================================================================

/// Apply adaptive single-channel kernel: blend of 1st-order and 2nd-order.
///
/// Splits error between 1st-order (FS/JJN) and 2nd-order (FS²/JJN²) kernels
/// based on alpha (from gradient map). Both write to the same buffer.
/// Always LTR (is_rtl=false).
///
/// - alpha >= 1.0: pure 2nd-order
/// - alpha <= 0.0: pure 1st-order
/// - otherwise: (1-alpha)*err to H1, alpha*err to H2
#[inline]
pub fn apply_adaptive_single_channel_kernel(
    buf: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err: f32,
    alpha: f32,
    use_jjn: bool,
) {
    if alpha >= 1.0 {
        // Pure 2nd order (common case in smooth areas)
        apply_h2_single_channel_kernel(buf, bx, y, err, use_jjn, false);
    } else if alpha <= 0.0 {
        // Pure 1st order
        apply_single_channel_kernel(buf, bx, y, err, use_jjn, false);
    } else {
        // Split error between kernels
        let err_1st = err * (1.0 - alpha);
        let err_2nd = err * alpha;
        apply_single_channel_kernel(buf, bx, y, err_1st, use_jjn, false);
        apply_h2_single_channel_kernel(buf, bx, y, err_2nd, use_jjn, false);
    }
}

/// Apply adaptive kernel to RGB channels (3 separate buffers).
///
/// Each channel independently selects between FS and JJN based on hash bits,
/// then splits error between 1st-order and 2nd-order kernels based on alpha.
/// Bit assignment: R=bit0, G=bit1, B=bit2.
#[inline]
pub fn apply_adaptive_kernel_rgb(
    err_r: &mut [Vec<f32>],
    err_g: &mut [Vec<f32>],
    err_b: &mut [Vec<f32>],
    bx: usize,
    y: usize,
    err_r_val: f32,
    err_g_val: f32,
    err_b_val: f32,
    alpha: f32,
    pixel_hash: u32,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;

    apply_adaptive_single_channel_kernel(err_r, bx, y, err_r_val, alpha, use_jjn_r);
    apply_adaptive_single_channel_kernel(err_g, bx, y, err_g_val, alpha, use_jjn_g);
    apply_adaptive_single_channel_kernel(err_b, bx, y, err_b_val, alpha, use_jjn_b);
}

/// Apply adaptive kernel to RGBA channels (4 separate buffers).
///
/// Each channel independently selects between FS and JJN based on hash bits,
/// then splits error between 1st-order and 2nd-order kernels based on alpha.
/// Bit assignment: R=bit0, G=bit1, B=bit2, A=bit3.
#[inline]
pub fn apply_adaptive_kernel_rgba(
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
    alpha: f32,
    pixel_hash: u32,
) {
    let use_jjn_r = pixel_hash & 1 != 0;
    let use_jjn_g = pixel_hash & 2 != 0;
    let use_jjn_b = pixel_hash & 4 != 0;
    let use_jjn_a = pixel_hash & 8 != 0;

    apply_adaptive_single_channel_kernel(err_r, bx, y, err_r_val, alpha, use_jjn_r);
    apply_adaptive_single_channel_kernel(err_g, bx, y, err_g_val, alpha, use_jjn_g);
    apply_adaptive_single_channel_kernel(err_b, bx, y, err_b_val, alpha, use_jjn_b);
    apply_adaptive_single_channel_kernel(err_a, bx, y, err_a_val, alpha, use_jjn_a);
}
