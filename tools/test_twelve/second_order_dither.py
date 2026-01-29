#!/usr/bin/env python3
"""
Second-order error diffusion experiment.

Compares three approaches to noise shaping:
1. First-order: standard FS/JJN error diffusion → NTF = (1-H), ~6.8 dB/oct
2. 2H-H² kernel: precomputed kernel aiming for NTF = (1-H)², ~8.2 dB/oct
3. Dual integrator: two coupled error buffers aiming for NTF = (1-C)², ~7.1 dB/oct

Kite et al. insight: In a single feedback loop, the quantizer gain K varies
with the kernel (K≈2 for FS, K≈4.5 for JJN), and NTF = (1-H)/(1+(K-1)H)
which behaves as (1-H) regardless of K. One feedback loop = one order of
noise shaping.

Uses mixed FS/JJN kernel switching with hash-based selection.
The kernel switching is equivalent to threshold dithering but with less white noise
(the hash produces a white noise pattern, but the two kernels are similar enough
that the effective randomization is much smaller than full threshold dithering).
Implements edge seeding matching the Rust dither/basic.rs create_seeded_buffer.

Best results at 50% gray: 2H-H² kernel = 8.22 dB/oct (uniform H² assumption).

Approaches tried that gave worse results:
- Two-pass cascade (X→Y₁, residual X-Y₁→Y₂, threshold Y₁+Y₂):
  Nearly bit-identical to first-order. The residual is already shaped noise
  dancing around zero, so pass 2 contributes almost nothing (0.05% differ).
- MASH-style cascade (tap raw quantization error Q₁ from pass 1):
  Energy preservation broken (0.49→0.52), slope worse at 5.60 dB/oct.
  Without differentiating Y₂ before combining, just adds two +6dB sources.
- Storing shaped_err instead of raw err in error history (for reverse prediction):
  Slope dropped from ~6.3 to ~5.6 dB/oct. Worse, not better.
- FS-only kernels (no mixed FS/JJN switching):
  Collapses into limit cycles / checkerboard at 50% gray (0-5.58 dB/oct).
  Kernel mixing is essential for breaking limit cycles.
- Per-neighbor matched H² correction (look up each neighbor's actual kernel):
  Mathematically correct but slope dropped to 7.51 dB/oct. The cross-kernel
  H² terms are weaker than self-convolution terms.
- Cross-averaged kernels (K = 2H - H⊗H_avg, H_avg = (H_fs+H_jjn)/2):
  7.72 dB/oct. Statistically correct for 50/50 mixing but weaker than
  the uniform assumption which over-corrects beneficially.
- 4-way kernel selection (hash bit 0: FS/JJN, bit 1: FS²/JJN² correction):
  7.41 dB/oct. Adds variation to H² correction but averages to a weaker
  correction than uniform assumption.
- Cross-only selection (always assume OTHER kernel for H² correction):
  6.93 dB/oct. Cross-convolutions FS⊗JJN are weaker than self-convolutions.
- Same kernel for both dual integrator buffers (bit 0 for both instead of bit 0/bit 1):
  Slightly worse (7.08→7.01 at 50%, 7.34→6.91 at 33%). Different kernels per
  integrator give more decorrelation between the two error paths.
- Blue noise map for kernel selection (dither 50% gray as selection pattern):
  Helped 2H-H² slightly (8.22→8.33 at 50%) but hurt first-order (6.78→6.32)
  and dual integrator (7.08→6.52). White noise hash better for limit cycle
  breaking in first-order methods.
- MASH 1-1 digital cancellation with re-binarization (Y = threshold(Y₁ + (1-H)·Y₂)):
  Spatial-domain (1-H) filter: 1.26 dB/oct, re-dithering: 4.06 dB/oct, 1D
  (1-z⁻¹) differentiator: 4.08 dB/oct. The combined signal is continuous and
  re-thresholding introduces unshaped noise. Needs a different recombination
  strategy that preserves binary output.

Usage:
    python second_order_dither.py                    # Gray levels + gradient
    python second_order_dither.py --gray 0.5         # Single gray level
    python second_order_dither.py --gradient-only     # Gradient only
    python second_order_dither.py --image FILE        # Process an image
"""

import numpy as np
from numpy.fft import fft2, fftshift
from PIL import Image
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REACH = 2  # JJN kernel radius


def lowbias32(x: np.uint32) -> np.uint32:
    """Lowbias32 hash - improved version with bias 0.107."""
    x = np.uint32(x)
    x ^= x >> np.uint32(16)
    x = np.uint32(np.uint64(x) * np.uint64(0x21f0aaad) & 0xFFFFFFFF)
    x ^= x >> np.uint32(15)
    x = np.uint32(np.uint64(x) * np.uint64(0x735a2d97) & 0xFFFFFFFF)
    x ^= x >> np.uint32(15)
    return x


# ============================================================================
# Seeded buffer (matches Rust create_seeded_buffer)
# ============================================================================

def create_seeded_buffer(input_image):
    """Create padded buffer with edge seeding, matching Rust create_seeded_buffer.

    Buffer layout (reach=2):
        cols: [0..2) overshoot | [2..4) left seed | [4..4+W) image | [4+W..6+W) right seed | [6+W..8+W) overshoot
        rows: [0..2) top seed | [2..2+H) image | [2+H..4+H) bottom overshoot
    """
    height, width = input_image.shape
    total_left = REACH * 2
    total_right = REACH * 2
    total_top = REACH
    total_bottom = REACH

    buf_width = total_left + width + total_right
    buf_height = total_top + height + total_bottom
    buf = np.zeros((buf_height, buf_width), dtype=np.float64)

    # Copy real image data
    buf[total_top:total_top + height, total_left:total_left + width] = input_image

    seed_left_start = REACH  # columns [reach..reach*2]
    seed_right_start = total_left + width  # columns [total_left+width..]

    # Left seeding columns: duplicate first column of real image
    for sx in range(REACH):
        buf[total_top:total_top + height, seed_left_start + sx] = input_image[:, 0]

    # Right seeding columns: duplicate last column of real image
    for sx in range(REACH):
        buf[total_top:total_top + height, seed_right_start + sx] = input_image[:, -1]

    # Top seeding rows: duplicate first row (including seeding columns)
    for sy in range(REACH):
        # Left seeding area
        for sx in range(REACH):
            buf[sy, seed_left_start + sx] = input_image[0, 0]
        # Real image area (first row)
        buf[sy, total_left:total_left + width] = input_image[0, :]
        # Right seeding area
        for sx in range(REACH):
            buf[sy, seed_right_start + sx] = input_image[0, -1]

    return buf


def extract_result(buf, width, height):
    """Extract real pixels from seeded buffer."""
    total_left = REACH * 2
    total_top = REACH
    return buf[total_top:total_top + height, total_left:total_left + width].copy()


# ============================================================================
# Error diffusion kernels (no bounds checking - padded buffer handles edges)
# ============================================================================

def apply_fs_ltr(buf, bx, y, err):
    buf[y, bx + 1] += err * (7.0 / 16.0)
    buf[y + 1, bx - 1] += err * (3.0 / 16.0)
    buf[y + 1, bx] += err * (5.0 / 16.0)
    buf[y + 1, bx + 1] += err * (1.0 / 16.0)


def apply_fs_rtl(buf, bx, y, err):
    buf[y, bx - 1] += err * (7.0 / 16.0)
    buf[y + 1, bx + 1] += err * (3.0 / 16.0)
    buf[y + 1, bx] += err * (5.0 / 16.0)
    buf[y + 1, bx - 1] += err * (1.0 / 16.0)


def apply_jjn_ltr(buf, bx, y, err):
    buf[y, bx + 1] += err * (7.0 / 48.0)
    buf[y, bx + 2] += err * (5.0 / 48.0)
    buf[y + 1, bx - 2] += err * (3.0 / 48.0)
    buf[y + 1, bx - 1] += err * (5.0 / 48.0)
    buf[y + 1, bx] += err * (7.0 / 48.0)
    buf[y + 1, bx + 1] += err * (5.0 / 48.0)
    buf[y + 1, bx + 2] += err * (3.0 / 48.0)
    buf[y + 2, bx - 2] += err * (1.0 / 48.0)
    buf[y + 2, bx - 1] += err * (3.0 / 48.0)
    buf[y + 2, bx] += err * (5.0 / 48.0)
    buf[y + 2, bx + 1] += err * (3.0 / 48.0)
    buf[y + 2, bx + 2] += err * (1.0 / 48.0)


def apply_jjn_rtl(buf, bx, y, err):
    buf[y, bx - 1] += err * (7.0 / 48.0)
    buf[y, bx - 2] += err * (5.0 / 48.0)
    buf[y + 1, bx + 2] += err * (3.0 / 48.0)
    buf[y + 1, bx + 1] += err * (5.0 / 48.0)
    buf[y + 1, bx] += err * (7.0 / 48.0)
    buf[y + 1, bx - 1] += err * (5.0 / 48.0)
    buf[y + 1, bx - 2] += err * (3.0 / 48.0)
    buf[y + 2, bx + 2] += err * (1.0 / 48.0)
    buf[y + 2, bx + 1] += err * (3.0 / 48.0)
    buf[y + 2, bx] += err * (5.0 / 48.0)
    buf[y + 2, bx - 1] += err * (3.0 / 48.0)
    buf[y + 2, bx - 2] += err * (1.0 / 48.0)


def apply_error(buf, bx, y, err, use_jjn, is_rtl):
    if use_jjn:
        if is_rtl:
            apply_jjn_rtl(buf, bx, y, err)
        else:
            apply_jjn_ltr(buf, bx, y, err)
    else:
        if is_rtl:
            apply_fs_rtl(buf, bx, y, err)
        else:
            apply_fs_ltr(buf, bx, y, err)


# ============================================================================
# 2H-H² kernels (single-buffer approach)
# NTF = 1 - (2H - H²) = (1-H)² → +12 dB/oct target
# These have negative weights and wider reach than the originals.
# ============================================================================

def apply_fs2_ltr(buf, bx, y, err):
    """FS second-order kernel (2H_fs - H_fs²), LTR. Reach: dx -2..+2, dy 0..2."""
    buf[y, bx + 1] += err * (224.0 / 256.0)
    buf[y, bx + 2] += err * (-49.0 / 256.0)
    buf[y + 1, bx - 1] += err * (96.0 / 256.0)
    buf[y + 1, bx] += err * (118.0 / 256.0)
    buf[y + 1, bx + 1] += err * (-38.0 / 256.0)
    buf[y + 1, bx + 2] += err * (-14.0 / 256.0)
    buf[y + 2, bx - 2] += err * (-9.0 / 256.0)
    buf[y + 2, bx - 1] += err * (-30.0 / 256.0)
    buf[y + 2, bx] += err * (-31.0 / 256.0)
    buf[y + 2, bx + 1] += err * (-10.0 / 256.0)
    buf[y + 2, bx + 2] += err * (-1.0 / 256.0)


def apply_fs2_rtl(buf, bx, y, err):
    """FS second-order kernel, RTL (mirror x)."""
    buf[y, bx - 1] += err * (224.0 / 256.0)
    buf[y, bx - 2] += err * (-49.0 / 256.0)
    buf[y + 1, bx + 1] += err * (96.0 / 256.0)
    buf[y + 1, bx] += err * (118.0 / 256.0)
    buf[y + 1, bx - 1] += err * (-38.0 / 256.0)
    buf[y + 1, bx - 2] += err * (-14.0 / 256.0)
    buf[y + 2, bx + 2] += err * (-9.0 / 256.0)
    buf[y + 2, bx + 1] += err * (-30.0 / 256.0)
    buf[y + 2, bx] += err * (-31.0 / 256.0)
    buf[y + 2, bx - 1] += err * (-10.0 / 256.0)
    buf[y + 2, bx - 2] += err * (-1.0 / 256.0)


def apply_jjn2_ltr(buf, bx, y, err):
    """JJN second-order kernel (2H_jjn - H_jjn²), LTR. Reach: dx -4..+4, dy 0..4."""
    # Row 0
    buf[y, bx + 1] += err * (672.0 / 2304.0)
    buf[y, bx + 2] += err * (431.0 / 2304.0)
    buf[y, bx + 3] += err * (-70.0 / 2304.0)
    buf[y, bx + 4] += err * (-25.0 / 2304.0)
    # Row 1
    buf[y + 1, bx - 2] += err * (288.0 / 2304.0)
    buf[y + 1, bx - 1] += err * (438.0 / 2304.0)
    buf[y + 1, bx] += err * (572.0 / 2304.0)
    buf[y + 1, bx + 1] += err * (332.0 / 2304.0)
    buf[y + 1, bx + 2] += err * (148.0 / 2304.0)
    buf[y + 1, bx + 3] += err * (-92.0 / 2304.0)
    buf[y + 1, bx + 4] += err * (-30.0 / 2304.0)
    # Row 2
    buf[y + 2, bx - 4] += err * (-9.0 / 2304.0)
    buf[y + 2, bx - 3] += err * (-30.0 / 2304.0)
    buf[y + 2, bx - 2] += err * (29.0 / 2304.0)
    buf[y + 2, bx - 1] += err * (174.0 / 2304.0)
    buf[y + 2, bx] += err * (311.0 / 2304.0)
    buf[y + 2, bx + 1] += err * (88.0 / 2304.0)
    buf[y + 2, bx + 2] += err * (-63.0 / 2304.0)
    buf[y + 2, bx + 3] += err * (-74.0 / 2304.0)
    buf[y + 2, bx + 4] += err * (-19.0 / 2304.0)
    # Row 3
    buf[y + 3, bx - 4] += err * (-6.0 / 2304.0)
    buf[y + 3, bx - 3] += err * (-28.0 / 2304.0)
    buf[y + 3, bx - 2] += err * (-74.0 / 2304.0)
    buf[y + 3, bx - 1] += err * (-120.0 / 2304.0)
    buf[y + 3, bx] += err * (-142.0 / 2304.0)
    buf[y + 3, bx + 1] += err * (-120.0 / 2304.0)
    buf[y + 3, bx + 2] += err * (-74.0 / 2304.0)
    buf[y + 3, bx + 3] += err * (-28.0 / 2304.0)
    buf[y + 3, bx + 4] += err * (-6.0 / 2304.0)
    # Row 4
    buf[y + 4, bx - 4] += err * (-1.0 / 2304.0)
    buf[y + 4, bx - 3] += err * (-6.0 / 2304.0)
    buf[y + 4, bx - 2] += err * (-19.0 / 2304.0)
    buf[y + 4, bx - 1] += err * (-36.0 / 2304.0)
    buf[y + 4, bx] += err * (-45.0 / 2304.0)
    buf[y + 4, bx + 1] += err * (-36.0 / 2304.0)
    buf[y + 4, bx + 2] += err * (-19.0 / 2304.0)
    buf[y + 4, bx + 3] += err * (-6.0 / 2304.0)
    buf[y + 4, bx + 4] += err * (-1.0 / 2304.0)


def apply_jjn2_rtl(buf, bx, y, err):
    """JJN second-order kernel, RTL (mirror x)."""
    # Row 0
    buf[y, bx - 1] += err * (672.0 / 2304.0)
    buf[y, bx - 2] += err * (431.0 / 2304.0)
    buf[y, bx - 3] += err * (-70.0 / 2304.0)
    buf[y, bx - 4] += err * (-25.0 / 2304.0)
    # Row 1
    buf[y + 1, bx + 2] += err * (288.0 / 2304.0)
    buf[y + 1, bx + 1] += err * (438.0 / 2304.0)
    buf[y + 1, bx] += err * (572.0 / 2304.0)
    buf[y + 1, bx - 1] += err * (332.0 / 2304.0)
    buf[y + 1, bx - 2] += err * (148.0 / 2304.0)
    buf[y + 1, bx - 3] += err * (-92.0 / 2304.0)
    buf[y + 1, bx - 4] += err * (-30.0 / 2304.0)
    # Row 2
    buf[y + 2, bx + 4] += err * (-9.0 / 2304.0)
    buf[y + 2, bx + 3] += err * (-30.0 / 2304.0)
    buf[y + 2, bx + 2] += err * (29.0 / 2304.0)
    buf[y + 2, bx + 1] += err * (174.0 / 2304.0)
    buf[y + 2, bx] += err * (311.0 / 2304.0)
    buf[y + 2, bx - 1] += err * (88.0 / 2304.0)
    buf[y + 2, bx - 2] += err * (-63.0 / 2304.0)
    buf[y + 2, bx - 3] += err * (-74.0 / 2304.0)
    buf[y + 2, bx - 4] += err * (-19.0 / 2304.0)
    # Row 3
    buf[y + 3, bx + 4] += err * (-6.0 / 2304.0)
    buf[y + 3, bx + 3] += err * (-28.0 / 2304.0)
    buf[y + 3, bx + 2] += err * (-74.0 / 2304.0)
    buf[y + 3, bx + 1] += err * (-120.0 / 2304.0)
    buf[y + 3, bx] += err * (-142.0 / 2304.0)
    buf[y + 3, bx - 1] += err * (-120.0 / 2304.0)
    buf[y + 3, bx - 2] += err * (-74.0 / 2304.0)
    buf[y + 3, bx - 3] += err * (-28.0 / 2304.0)
    buf[y + 3, bx - 4] += err * (-6.0 / 2304.0)
    # Row 4
    buf[y + 4, bx + 4] += err * (-1.0 / 2304.0)
    buf[y + 4, bx + 3] += err * (-6.0 / 2304.0)
    buf[y + 4, bx + 2] += err * (-19.0 / 2304.0)
    buf[y + 4, bx + 1] += err * (-36.0 / 2304.0)
    buf[y + 4, bx] += err * (-45.0 / 2304.0)
    buf[y + 4, bx - 1] += err * (-36.0 / 2304.0)
    buf[y + 4, bx - 2] += err * (-19.0 / 2304.0)
    buf[y + 4, bx - 3] += err * (-6.0 / 2304.0)
    buf[y + 4, bx - 4] += err * (-1.0 / 2304.0)


def apply_error_2nd(buf, bx, y, err, use_jjn, is_rtl):
    """Apply second-order (2H - H²) kernel (precomputed, assumes uniform kernel)."""
    if use_jjn:
        if is_rtl:
            apply_jjn2_rtl(buf, bx, y, err)
        else:
            apply_jjn2_ltr(buf, bx, y, err)
    else:
        if is_rtl:
            apply_fs2_rtl(buf, bx, y, err)
        else:
            apply_fs2_ltr(buf, bx, y, err)


# ============================================================================
# 3H-3H²+H³ kernels (single-buffer approach)
# NTF = 1 - (3H-3H²+H³) = (1-H)³ → +18 dB/oct target
# Even wider reach and more negative weights than 2nd order.
# ============================================================================

def apply_fs3_ltr(buf, bx, y, err):
    """FS third-order kernel (3H - 3H² + H³), LTR. Reach: dx -3..+3, dy 0..3."""
    # Row 0
    buf[y, bx + 1] += err * (5376.0 / 4096.0)
    buf[y, bx + 2] += err * (-2352.0 / 4096.0)
    buf[y, bx + 3] += err * (343.0 / 4096.0)
    # Row 1
    buf[y + 1, bx - 1] += err * (2304.0 / 4096.0)
    buf[y + 1, bx] += err * (1824.0 / 4096.0)
    buf[y + 1, bx + 1] += err * (-2151.0 / 4096.0)
    buf[y + 1, bx + 2] += err * (63.0 / 4096.0)
    buf[y + 1, bx + 3] += err * (147.0 / 4096.0)
    # Row 2
    buf[y + 2, bx - 2] += err * (-432.0 / 4096.0)
    buf[y + 2, bx - 1] += err * (-1251.0 / 4096.0)
    buf[y + 2, bx] += err * (-858.0 / 4096.0)
    buf[y + 2, bx + 1] += err * (171.0 / 4096.0)
    buf[y + 2, bx + 2] += err * (162.0 / 4096.0)
    buf[y + 2, bx + 3] += err * (21.0 / 4096.0)
    # Row 3
    buf[y + 3, bx - 3] += err * (27.0 / 4096.0)
    buf[y + 3, bx - 2] += err * (135.0 / 4096.0)
    buf[y + 3, bx - 1] += err * (252.0 / 4096.0)
    buf[y + 3, bx] += err * (215.0 / 4096.0)
    buf[y + 3, bx + 1] += err * (84.0 / 4096.0)
    buf[y + 3, bx + 2] += err * (15.0 / 4096.0)
    buf[y + 3, bx + 3] += err * (1.0 / 4096.0)


def apply_jjn3_ltr(buf, bx, y, err):
    """JJN third-order kernel (3H - 3H² + H³), LTR. Reach: dx -6..+6, dy 0..6."""
    # Row 0 (dy=0)
    buf[y, bx + 1] += err * (48384.0 / 110592.0)
    buf[y, bx + 2] += err * (27504.0 / 110592.0)
    buf[y, bx + 3] += err * (-9737.0 / 110592.0)
    buf[y, bx + 4] += err * (-2865.0 / 110592.0)
    buf[y, bx + 5] += err * (525.0 / 110592.0)
    buf[y, bx + 6] += err * (125.0 / 110592.0)
    # Row 1 (dy=1)
    buf[y + 1, bx - 2] += err * (20736.0 / 110592.0)
    buf[y + 1, bx - 1] += err * (28512.0 / 110592.0)
    buf[y + 1, bx] += err * (34425.0 / 110592.0)
    buf[y + 1, bx + 1] += err * (14613.0 / 110592.0)
    buf[y + 1, bx + 2] += err * (2880.0 / 110592.0)
    buf[y + 1, bx + 3] += err * (-10668.0 / 110592.0)
    buf[y + 1, bx + 4] += err * (-2304.0 / 110592.0)
    buf[y + 1, bx + 5] += err * (1005.0 / 110592.0)
    buf[y + 1, bx + 6] += err * (225.0 / 110592.0)
    # Row 2 (dy=2)
    buf[y + 2, bx - 4] += err * (-1296.0 / 110592.0)
    buf[y + 2, bx - 3] += err * (-4131.0 / 110592.0)
    buf[y + 2, bx - 2] += err * (-1971.0 / 110592.0)
    buf[y + 2, bx - 1] += err * (6177.0 / 110592.0)
    buf[y + 2, bx] += err * (13476.0 / 110592.0)
    buf[y + 2, bx + 1] += err * (-3456.0 / 110592.0)
    buf[y + 2, bx + 2] += err * (-10689.0 / 110592.0)
    buf[y + 2, bx + 3] += err * (-6033.0 / 110592.0)
    buf[y + 2, bx + 4] += err * (51.0 / 110592.0)
    buf[y + 2, bx + 5] += err * (1074.0 / 110592.0)
    buf[y + 2, bx + 6] += err * (210.0 / 110592.0)
    # Row 3 (dy=3)
    buf[y + 3, bx - 6] += err * (27.0 / 110592.0)
    buf[y + 3, bx - 5] += err * (135.0 / 110592.0)
    buf[y + 3, bx - 4] += err * (-450.0 / 110592.0)
    buf[y + 3, bx - 3] += err * (-3016.0 / 110592.0)
    buf[y + 3, bx - 2] += err * (-8481.0 / 110592.0)
    buf[y + 3, bx - 1] += err * (-13296.0 / 110592.0)
    buf[y + 3, bx] += err * (-14597.0 / 110592.0)
    buf[y + 3, bx + 1] += err * (-10488.0 / 110592.0)
    buf[y + 3, bx + 2] += err * (-4509.0 / 110592.0)
    buf[y + 3, bx + 3] += err * (212.0 / 110592.0)
    buf[y + 3, bx + 4] += err * (1248.0 / 110592.0)
    buf[y + 3, bx + 5] += err * (681.0 / 110592.0)
    buf[y + 3, bx + 6] += err * (117.0 / 110592.0)
    # Row 4 (dy=4)
    buf[y + 4, bx - 6] += err * (27.0 / 110592.0)
    buf[y + 4, bx - 5] += err * (171.0 / 110592.0)
    buf[y + 4, bx - 4] += err * (462.0 / 110592.0)
    buf[y + 4, bx - 3] += err * (591.0 / 110592.0)
    buf[y + 4, bx - 2] += err * (-42.0 / 110592.0)
    buf[y + 4, bx - 1] += err * (-1149.0 / 110592.0)
    buf[y + 4, bx] += err * (-1482.0 / 110592.0)
    buf[y + 4, bx + 1] += err * (-153.0 / 110592.0)
    buf[y + 4, bx + 2] += err * (1248.0 / 110592.0)
    buf[y + 4, bx + 3] += err * (1509.0 / 110592.0)
    buf[y + 4, bx + 4] += err * (873.0 / 110592.0)
    buf[y + 4, bx + 5] += err * (282.0 / 110592.0)
    buf[y + 4, bx + 6] += err * (42.0 / 110592.0)
    # Row 5 (dy=5)
    buf[y + 5, bx - 6] += err * (9.0 / 110592.0)
    buf[y + 5, bx - 5] += err * (69.0 / 110592.0)
    buf[y + 5, bx - 4] += err * (282.0 / 110592.0)
    buf[y + 5, bx - 3] += err * (750.0 / 110592.0)
    buf[y + 5, bx - 2] += err * (1443.0 / 110592.0)
    buf[y + 5, bx - 1] += err * (2094.0 / 110592.0)
    buf[y + 5, bx] += err * (2367.0 / 110592.0)
    buf[y + 5, bx + 1] += err * (2094.0 / 110592.0)
    buf[y + 5, bx + 2] += err * (1443.0 / 110592.0)
    buf[y + 5, bx + 3] += err * (750.0 / 110592.0)
    buf[y + 5, bx + 4] += err * (282.0 / 110592.0)
    buf[y + 5, bx + 5] += err * (69.0 / 110592.0)
    buf[y + 5, bx + 6] += err * (9.0 / 110592.0)
    # Row 6 (dy=6)
    buf[y + 6, bx - 6] += err * (1.0 / 110592.0)
    buf[y + 6, bx - 5] += err * (9.0 / 110592.0)
    buf[y + 6, bx - 4] += err * (42.0 / 110592.0)
    buf[y + 6, bx - 3] += err * (126.0 / 110592.0)
    buf[y + 6, bx - 2] += err * (267.0 / 110592.0)
    buf[y + 6, bx - 1] += err * (414.0 / 110592.0)
    buf[y + 6, bx] += err * (479.0 / 110592.0)
    buf[y + 6, bx + 1] += err * (414.0 / 110592.0)
    buf[y + 6, bx + 2] += err * (267.0 / 110592.0)
    buf[y + 6, bx + 3] += err * (126.0 / 110592.0)
    buf[y + 6, bx + 4] += err * (42.0 / 110592.0)
    buf[y + 6, bx + 5] += err * (9.0 / 110592.0)
    buf[y + 6, bx + 6] += err * (1.0 / 110592.0)


def apply_error_3rd(buf, bx, y, err, use_jjn):
    """Apply third-order (3H - 3H² + H³) kernel (LTR only)."""
    if use_jjn:
        apply_jjn3_ltr(buf, bx, y, err)
    else:
        apply_fs3_ltr(buf, bx, y, err)


# ============================================================================
# Dithering functions
# ============================================================================

def dither_first_order(input_image, seed=0):
    """Standard mixed FS/JJN error diffusion with edge seeding (first-order, +6 dB/oct)."""
    height, width = input_image.shape
    buf = create_seeded_buffer(input_image)
    hashed_seed = lowbias32(np.uint32(seed))

    bx_start = REACH  # skip left overshoot
    process_height = REACH + height  # seeding rows + real rows
    process_width = REACH + width + REACH  # left seed + real + right seed

    for y in range(process_height):
        is_rtl = y % 2 == 1
        px_range = range(process_width - 1, -1, -1) if is_rtl else range(process_width)

        for px in px_range:
            bx = bx_start + px
            old_val = buf[y, bx]
            new_val = 1.0 if old_val > 0.5 else 0.0
            buf[y, bx] = new_val
            err = old_val - new_val

            # Wrap coordinates so seed pixels get varied hashes
            img_x = (px - REACH) & 0xFFFF
            img_y = (y - REACH) & 0xFFFF
            coord_hash = lowbias32(np.uint32(img_x) ^ (np.uint32(img_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1
            apply_error(buf, bx, y, err, use_jjn, is_rtl)

    return extract_result(buf, width, height)


def dither_dual_integrator(input_image, seed=0):
    """Dual-integrator error diffusion (~7.1 dB/oct at 50% gray).

    Two error buffers (integrators) run simultaneously:
        int1 = input + diffused err1 (standard first-order)
        int2 = int1 + diffused err2 (second integrator, feeds on int1)
        output = quantize(int2)
        err1 = int1 - output → diffuse to buf1
        err2 = int2 - output → diffuse to buf2

    Single feedback loop around one quantizer (Kite et al.: NTF behaves as
    (1-H) for single-loop architectures regardless of K).
    """
    height, width = input_image.shape

    # First integrator: seeded with input image
    buf1 = create_seeded_buffer(input_image)

    # Second integrator: starts at zero (pure error accumulation)
    buf2 = np.zeros_like(buf1)

    hashed_seed = lowbias32(np.uint32(seed))

    bx_start = REACH
    process_height = REACH + height
    process_width = REACH + width + REACH

    for y in range(process_height):
        is_rtl = y % 2 == 1
        px_range = range(process_width - 1, -1, -1) if is_rtl else range(process_width)

        for px in px_range:
            bx = bx_start + px

            # First integrator value at this pixel
            int1_val = buf1[y, bx]

            # Second integrator: int1 + accumulated second-order corrections
            int2_val = int1_val + buf2[y, bx]

            # Quantize based on second integrator
            new_val = 1.0 if int2_val > 0.5 else 0.0
            buf1[y, bx] = new_val  # store output for extraction

            # First integrator error
            err1 = int1_val - new_val

            # Second integrator error
            err2 = int2_val - new_val

            # Kernel selection — different bits for each integrator
            img_x = (px - REACH) & 0xFFFF
            img_y = (y - REACH) & 0xFFFF
            coord_hash = lowbias32(np.uint32(img_x) ^ (np.uint32(img_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn_1 = (coord_hash & 1) == 1
            use_jjn_2 = (coord_hash & 2) == 2

            # Diffuse errors to respective buffers
            apply_error(buf1, bx, y, err1, use_jjn_1, is_rtl)
            apply_error(buf2, bx, y, err2, use_jjn_2, is_rtl)

    return extract_result(buf1, width, height)


REACH_2ND = 4  # JJN² kernel radius (dx ±4, dy 0..4)
SEED_2ND = 16  # Seed area width (4x reach for warm-up with negative weights)

REACH_3RD = 6  # JJN³ kernel radius (dx ±6, dy 0..6)
SEED_3RD = 24  # Seed area width (4x reach)


def create_seeded_buffer_r4(input_image):
    """Create padded buffer for second-order kernels.

    Buffer layout (reach=4, seed=8):
        cols: [0..4) overshoot | [4..12) seed | [12..12+W) image | [12+W..20+W) seed | [20+W..24+W) overshoot
        rows: [0..8) seed | [8..8+H) image | [8+H..12+H) overshoot
    """
    height, width = input_image.shape
    r = REACH_2ND
    s = SEED_2ND
    total_left = r + s      # overshoot + seed
    total_right = s + r      # seed + overshoot
    total_top = s            # seed rows
    total_bottom = r         # overshoot rows

    buf_width = total_left + width + total_right
    buf_height = total_top + height + total_bottom
    buf = np.zeros((buf_height, buf_width), dtype=np.float64)

    buf[total_top:total_top + height, total_left:total_left + width] = input_image

    seed_left_start = r
    seed_right_start = total_left + width

    for sx in range(s):
        buf[total_top:total_top + height, seed_left_start + sx] = input_image[:, 0]
    for sx in range(s):
        buf[total_top:total_top + height, seed_right_start + sx] = input_image[:, -1]
    for sy in range(s):
        for sx in range(s):
            buf[sy, seed_left_start + sx] = input_image[0, 0]
        buf[sy, total_left:total_left + width] = input_image[0, :]
        for sx in range(s):
            buf[sy, seed_right_start + sx] = input_image[0, -1]

    return buf


def dither_kernel_2nd(input_image, seed=0):
    """Error diffusion with precomputed 2H-H² kernels (~8.2 dB/oct at 50% gray).

    Uses K = 2H - H² with uniform kernel assumption (H² = H_self⊗H_self).
    The uniform assumption over-corrects vs the true mixed-kernel H², but
    this gives the best measured slope. Single feedback loop around one
    quantizer (Kite et al.). Always LTR (no serpentine).
    """
    height, width = input_image.shape
    r = REACH_2ND
    s = SEED_2ND
    buf = create_seeded_buffer_r4(input_image)
    hashed_seed = lowbias32(np.uint32(seed))

    bx_start = r  # skip left overshoot
    process_height = s + height
    process_width = s + width + s

    for y in range(process_height):
        for px in range(process_width):
            bx = bx_start + px
            old_val = buf[y, bx]
            new_val = 1.0 if old_val > 0.5 else 0.0
            buf[y, bx] = new_val
            err = old_val - new_val

            img_x = (px - s) & 0xFFFF
            img_y = (y - s) & 0xFFFF
            coord_hash = lowbias32(np.uint32(img_x) ^ (np.uint32(img_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1
            apply_error_2nd(buf, bx, y, err, use_jjn, False)

    total_left = r + s
    total_top = s
    return buf[total_top:total_top + height, total_left:total_left + width].copy()


def dither_kernel_3rd(input_image, seed=0):
    """Error diffusion with precomputed 3H-3H²+H³ kernels.

    Uses K = 3H - 3H² + H³ with uniform kernel assumption.
    FS³ has 20 weights (reach 3), JJN³ has 78 weights (reach 6).
    Always LTR (no serpentine).
    """
    height, width = input_image.shape
    r = REACH_3RD
    s = SEED_3RD

    total_left = r + s
    total_right = s + r
    total_top = s
    total_bottom = r

    buf_width = total_left + width + total_right
    buf_height = total_top + height + total_bottom
    buf = np.zeros((buf_height, buf_width), dtype=np.float64)

    buf[total_top:total_top + height, total_left:total_left + width] = input_image

    seed_left_start = r
    seed_right_start = total_left + width

    for sx in range(s):
        buf[total_top:total_top + height, seed_left_start + sx] = input_image[:, 0]
    for sx in range(s):
        buf[total_top:total_top + height, seed_right_start + sx] = input_image[:, -1]
    for sy in range(s):
        for sx in range(s):
            buf[sy, seed_left_start + sx] = input_image[0, 0]
        buf[sy, total_left:total_left + width] = input_image[0, :]
        for sx in range(s):
            buf[sy, seed_right_start + sx] = input_image[0, -1]

    hashed_seed = lowbias32(np.uint32(seed))

    bx_start = r
    process_height = s + height
    process_width = s + width + s

    for y in range(process_height):
        for px in range(process_width):
            bx = bx_start + px
            old_val = buf[y, bx]
            new_val = 1.0 if old_val > 0.5 else 0.0
            buf[y, bx] = new_val
            err = old_val - new_val

            img_x = (px - s) & 0xFFFF
            img_y = (y - s) & 0xFFFF
            coord_hash = lowbias32(np.uint32(img_x) ^ (np.uint32(img_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1
            apply_error_3rd(buf, bx, y, err, use_jjn)

    return buf[total_top:total_top + height, total_left:total_left + width].copy()


def dither_dual_int_2nd(input_image, seed=0):
    """Dual integrator with 2H-H² kernel on first stage.

    First integrator uses 2nd-order (2H-H²) error diffusion kernels.
    Second integrator uses 1st-order kernels for accumulated correction.
    Always LTR (no serpentine, matching 2H-H² behavior).
    """
    height, width = input_image.shape
    r = REACH_2ND
    s = SEED_2ND

    # First integrator: seeded with input image (same layout as 2H-H²)
    buf1 = create_seeded_buffer_r4(input_image)

    # Second integrator: zeros, same size
    buf2 = np.zeros_like(buf1)

    hashed_seed = lowbias32(np.uint32(seed))

    bx_start = r
    process_height = s + height
    process_width = s + width + s

    for y in range(process_height):
        for px in range(process_width):
            bx = bx_start + px

            # First integrator value
            int1_val = buf1[y, bx]

            # Second integrator: int1 + accumulated corrections
            int2_val = int1_val + buf2[y, bx]

            # Quantize based on second integrator
            new_val = 1.0 if int2_val > 0.5 else 0.0
            buf1[y, bx] = new_val

            # Errors
            err1 = int1_val - new_val
            err2 = int2_val - new_val

            # Kernel selection — different bits for each integrator
            img_x = (px - s) & 0xFFFF
            img_y = (y - s) & 0xFFFF
            coord_hash = lowbias32(np.uint32(img_x) ^ (np.uint32(img_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn_1 = (coord_hash & 1) == 1
            use_jjn_2 = (coord_hash & 2) == 2

            # First integrator: 2nd-order kernel
            apply_error_2nd(buf1, bx, y, err1, use_jjn_1, False)

            # Second integrator: 1st-order kernel
            apply_error(buf2, bx, y, err2, use_jjn_2, False)

    total_left = r + s
    total_top = s
    return buf1[total_top:total_top + height, total_left:total_left + width].copy()


# ============================================================================
# Spectral analysis
# ============================================================================

def compute_power_spectrum(img):
    centered = img - img.mean()
    spectrum = fftshift(fft2(centered))
    power = np.abs(spectrum) ** 2
    return np.log1p(power)


def compute_segmented_radial_power(img):
    h, w = img.shape
    centered = img - img.mean()
    spectrum = np.abs(fftshift(fft2(centered))) ** 2

    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)
    img_size = min(h, w)

    y_coords, x_coords = np.ogrid[:h, :w]
    dy = y_coords - cy
    dx = x_coords - cx
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))

    horizontal, diagonal, vertical = [], [], []

    for r in range(1, max_radius):
        ring_mask = np.abs(distances - r) < 0.5
        if not ring_mask.any():
            horizontal.append(0)
            diagonal.append(0)
            vertical.append(0)
            continue

        ring_angles = angles[ring_mask]
        ring_power = spectrum[ring_mask]

        h_mask = (ring_angles < 22.5) | (ring_angles > 157.5)
        v_mask = (ring_angles > 67.5) & (ring_angles < 112.5)
        d_mask = ((ring_angles > 22.5) & (ring_angles < 67.5)) | \
                 ((ring_angles > 112.5) & (ring_angles < 157.5))

        horizontal.append(ring_power[h_mask].mean() if h_mask.any() else 0)
        diagonal.append(ring_power[d_mask].mean() if d_mask.any() else 0)
        vertical.append(ring_power[v_mask].mean() if v_mask.any() else 0)

    freqs = np.arange(1, max_radius) / img_size
    return freqs, np.array(horizontal), np.array(diagonal), np.array(vertical)


def measure_slope(freqs, h_pow, d_pow, v_pow):
    """Measure spectral slope in dB/octave."""
    avg_pow = (h_pow + d_pow + v_pow) / 3.0
    avg_db = 10 * np.log10(avg_pow + 1e-10)
    valid = freqs > 0.02
    if valid.sum() > 10:
        log2_f = np.log2(freqs[valid])
        coeffs = np.polyfit(log2_f, avg_db[valid], 1)
        return coeffs[0]
    return 0.0


# ============================================================================
# Visualization
# ============================================================================

def analyze_gray(gray_val, output_dir, size=256, seed=0):
    """Generate and analyze 1-bit dithering at a given gray level."""
    input_img = np.full((size, size), gray_val, dtype=np.float64)

    methods = [
        ('1st order (mixed FS/JJN)', dither_first_order),
        ('2H-H² kernel', dither_kernel_2nd),
        ('3H-3H²+H³ kernel', dither_kernel_3rd),
        ('dual integrator', dither_dual_integrator),
    ]

    results = []
    for label, fn in methods:
        halftone = fn(input_img, seed=seed) * 255.0
        freqs, h_pow, d_pow, v_pow = compute_segmented_radial_power(halftone)
        h_db = 10 * np.log10(h_pow + 1e-10)
        d_db = 10 * np.log10(d_pow + 1e-10)
        v_db = 10 * np.log10(v_pow + 1e-10)
        slope = measure_slope(freqs, h_pow, d_pow, v_pow)
        results.append((label, halftone, freqs, h_db, d_db, v_db, slope))

    # Shared y-axis limits
    all_db = np.concatenate([np.concatenate([h, d, v]) for _, _, _, h, d, v, _ in results])
    y_min = all_db.min() - 5
    y_max = all_db.max() + 5

    n_cols = len(results)
    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 14))

    for col, (label, ht, freqs, h_db, d_db, v_db, slope) in enumerate(results):
        density = ht.mean() / 255.0

        axes[0, col].imshow(ht, cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(f'{label}\n{density*100:.1f}% white, slope={slope:.1f} dB/oct',
                               fontsize=10, fontweight='bold')
        axes[0, col].axis('off')

        spectrum_2d = compute_power_spectrum(ht)
        axes[1, col].imshow(spectrum_2d, cmap='gray')
        axes[1, col].axis('off')

        freqs_ideal = np.linspace(0.004, 0.5, 500)
        P_ref = 10 ** (h_db.max() / 10)
        power_6db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 2 + 1e-10)
        power_12db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 4 + 1e-10)

        axes[2, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs_ideal, power_6db, 'k--', linewidth=1.5, alpha=0.6, label='f² (+6dB/oct)')
        axes[2, col].plot(freqs_ideal, power_12db, 'k-.', linewidth=1.5, alpha=0.6, label='f⁴ (+12dB/oct)')
        axes[2, col].set_xlim(0, 0.5)
        axes[2, col].set_ylim(y_min, y_max)
        axes[2, col].set_xlabel('Spatial Frequency (cycles/pixel)')
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].legend(loc='lower right', fontsize=8)
        if col == 0:
            axes[2, col].set_ylabel('Power (dB)')

    fig.text(0.02, 0.83, 'Halftone', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.5, 'Spectrum', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.17, 'Radial', va='center', rotation='vertical', fontsize=12)

    gray_pct = int(gray_val * 100)
    fig.suptitle(f'Error Diffusion Order Comparison — {gray_pct}% gray', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    out_path = output_dir / f"order_comparison_gray_{gray_pct:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    for label, _, _, _, _, _, slope in results:
        print(f"  {label}: {slope:.2f} dB/oct")


def analyze_gradient(output_dir, size=256, seed=0):
    """Generate gradient dithered with both orders."""
    gradient = np.tile(np.linspace(0, 1, size * 4), (size, 1))

    methods = [
        ('1st order', dither_first_order),
        ('2H-H² kernel', dither_kernel_2nd),
        ('3H-3H²+H³ kernel', dither_kernel_3rd),
        ('dual integrator', dither_dual_integrator),
    ]

    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 3 * len(methods)))

    for i, (label, fn) in enumerate(methods):
        result = fn(gradient, seed=seed)
        img = (result * 255).astype(np.uint8)

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
        axes[i].set_title(label, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('y')
        if i == len(methods) - 1:
            axes[i].set_xlabel('x (0% → 100% gray)')

    fig.suptitle('Gradient Dithering — Order Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = output_dir / "order_comparison_gradient.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Also save individual gradient PNGs
    for label, fn in methods:
        result = fn(gradient, seed=seed)
        img = (result * 255).astype(np.uint8)
        safe_label = label.replace(' ', '_').replace('²', '2').replace('³', '3').replace('⁴', '4')
        path = output_dir / f"gradient_{safe_label}.png"
        Image.fromarray(img, mode='L').save(path)
        print(f"Saved: {path}")


def process_image(image_path, output_dir, seed=0):
    """Process a real image with all dithering methods."""
    img = Image.open(image_path).convert('L')
    input_image = np.array(img, dtype=np.float64) / 255.0
    stem = Path(image_path).stem

    # Standard methods
    for label, fn in [('1st_order', dither_first_order), ('2H_H2_kernel', dither_kernel_2nd), ('3H_3H2_H3_kernel', dither_kernel_3rd), ('dual_int', dither_dual_integrator)]:
        result = fn(input_image, seed=seed)
        out_img = (result * 255).astype(np.uint8)
        path = output_dir / f"{stem}_{label}.png"
        Image.fromarray(out_img, mode='L').save(path)
        print(f"Saved: {path}")



def main():
    parser = argparse.ArgumentParser(
        description="Second-order error diffusion experiment"
    )
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--gray", type=float, nargs="+",
                        help="Gray level(s) to test (0.0-1.0)")
    parser.add_argument("--gradient-only", action="store_true",
                        help="Only generate gradient comparison")
    parser.add_argument("--image", type=str,
                        help="Process an image file (grayscale PNG)")

    args = parser.parse_args()
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        process_image(args.image, output_dir, seed=args.seed)
        return

    if args.gradient_only:
        analyze_gradient(output_dir, size=args.size, seed=args.seed)
        return

    gray_levels = args.gray if args.gray else [0.05, 0.125, 0.25, 0.333, 0.5, 0.667, 0.75, 0.875]

    for g in gray_levels:
        analyze_gray(g, output_dir, size=args.size, seed=args.seed)

    analyze_gradient(output_dir, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
