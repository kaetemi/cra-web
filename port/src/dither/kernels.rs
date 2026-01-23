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
