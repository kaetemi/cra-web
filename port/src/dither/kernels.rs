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
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32);

    /// Apply kernel for right-to-left scanning (mirrored).
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32);

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
    fn apply_ltr(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
    );

    /// Apply kernel to RGB channels for right-to-left scanning.
    fn apply_rtl(
        err_r: &mut [Vec<f32>],
        err_g: &mut [Vec<f32>],
        err_b: &mut [Vec<f32>],
        bx: usize,
        y: usize,
        err_r_val: f32,
        err_g_val: f32,
        err_b_val: f32,
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
    );

    /// Apply kernel to RGBA channels for right-to-left scanning.
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
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
        apply_single_channel_kernel(buf, bx, y, err, false, false);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
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
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
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
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
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
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_ltr(err_a, bx, y, err_a_val);
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
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_rtl(err_a, bx, y, err_a_val);
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
    fn apply_ltr(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
        apply_single_channel_kernel(buf, bx, y, err, true, false);
    }

    #[inline]
    fn apply_rtl(buf: &mut [Vec<f32>], bx: usize, y: usize, err: f32) {
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
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
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
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
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
    ) {
        <Self as SingleChannelKernel>::apply_ltr(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_ltr(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_ltr(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_ltr(err_a, bx, y, err_a_val);
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
    ) {
        <Self as SingleChannelKernel>::apply_rtl(err_r, bx, y, err_r_val);
        <Self as SingleChannelKernel>::apply_rtl(err_g, bx, y, err_g_val);
        <Self as SingleChannelKernel>::apply_rtl(err_b, bx, y, err_b_val);
        <Self as SingleChannelKernel>::apply_rtl(err_a, bx, y, err_a_val);
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
    fn apply_ltr(_buf: &mut [Vec<f32>], _bx: usize, _y: usize, _err: f32) {}

    #[inline]
    fn apply_rtl(_buf: &mut [Vec<f32>], _bx: usize, _y: usize, _err: f32) {}
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
    ) {
    }
}
