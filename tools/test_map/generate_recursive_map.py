#!/usr/bin/env python3
"""
Mixed FS/JJN error diffusion dithering in floating point.

Replicates our dithering method but uses 0.0-1.0 range instead of uint8.
Supports any bit depth quantization.

Usage:
    python generate_recursive_map.py --gradient 1 2 4 8  # Generate gradient at different bit depths
    python generate_recursive_map.py --bits 4 --gray 0.5  # Dither 50% gray at 4-bit
"""

import numpy as np
from numpy.fft import fft2, fftshift
from PIL import Image
from pathlib import Path
from collections import deque
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
# First-order kernels (padded buffer, no bounds checking)
# ============================================================================

REACH_1ST = 2   # JJN kernel radius


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


def apply_error_1st(buf, bx, y, err, use_jjn, is_rtl):
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


def create_seeded_buffer_1st(input_image):
    """Create padded buffer for first-order kernels (reach=2, seed=2)."""
    height, width = input_image.shape
    r = REACH_1ST
    total_left = r * 2
    total_right = r * 2
    total_top = r
    total_bottom = r

    buf_width = total_left + width + total_right
    buf_height = total_top + height + total_bottom
    buf = np.zeros((buf_height, buf_width), dtype=np.float64)

    buf[total_top:total_top + height, total_left:total_left + width] = input_image

    seed_left_start = r
    seed_right_start = total_left + width

    for sx in range(r):
        buf[total_top:total_top + height, seed_left_start + sx] = input_image[:, 0]
    for sx in range(r):
        buf[total_top:total_top + height, seed_right_start + sx] = input_image[:, -1]
    for sy in range(r):
        for sx in range(r):
            buf[sy, seed_left_start + sx] = input_image[0, 0]
        buf[sy, total_left:total_left + width] = input_image[0, :]
        for sx in range(r):
            buf[sy, seed_right_start + sx] = input_image[0, -1]

    return buf


# ============================================================================
# Second-order 2H-H² kernels
# ============================================================================

REACH_2ND = 4   # JJN² kernel radius (dx ±4, dy 0..4)
SEED_2ND = 16   # Seed area width (4x reach for warm-up with negative weights)


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


def apply_error_2nd(buf, bx, y, err, use_jjn):
    """Apply second-order (2H - H²) kernel (precomputed, assumes uniform kernel). LTR only."""
    if use_jjn:
        apply_jjn2_ltr(buf, bx, y, err)
    else:
        apply_fs2_ltr(buf, bx, y, err)


def create_seeded_buffer_2nd(input_image):
    """Create padded buffer for second-order kernels.

    Buffer layout (reach=4, seed=16):
        cols: [0..4) overshoot | [4..20) seed | [20..20+W) image | [20+W..36+W) seed | [36+W..40+W) overshoot
        rows: [0..16) seed | [16..16+H) image | [16+H..20+H) overshoot
    """
    height, width = input_image.shape
    r = REACH_2ND
    s = SEED_2ND
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

    return buf


def quantize(value: float, bits: int) -> float:
    """
    Quantize a value to N-bit levels.

    For N bits, we have 2^N levels: 0, 1/(2^N-1), 2/(2^N-1), ..., 1.0
    """
    num_levels = 2 ** bits
    max_level = num_levels - 1

    # Quantize: round to nearest level
    level = round(value * max_level)
    level = max(0, min(max_level, level))

    # Convert back to 0-1 range
    return level / max_level if max_level > 0 else 0.0


def transform_image(img, swap_xy, mirror_x, mirror_y):
    """Apply spatial transform: swap XY, mirror X, mirror Y."""
    result = img
    if swap_xy:
        result = result.T
    if mirror_x:
        result = result[:, ::-1]
    if mirror_y:
        result = result[::-1, :]
    return np.ascontiguousarray(result)


def inverse_transform_image(img, swap_xy, mirror_x, mirror_y):
    """Undo spatial transform (reverse order, each op is self-inverse)."""
    result = img
    if mirror_y:
        result = result[::-1, :]
    if mirror_x:
        result = result[:, ::-1]
    if swap_xy:
        result = result.T
    return np.ascontiguousarray(result)


def tpdf_threshold(x, y, hashed_seed, amplitude=64.0/255.0):
    """TPDF-perturbed threshold using two hashes for triangular distribution.

    Matches the Rust implementation: two independent hashes summed for
    triangular PDF in [-1, 1], scaled by amplitude around 0.5.
    """
    hash1 = lowbias32(np.uint32(np.uint32(x) * np.uint32(2)) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
    hash2 = lowbias32(np.uint32(np.uint32(x) * np.uint32(2) + np.uint32(1)) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
    r1 = float(hash1) / float(0xFFFFFFFF)
    r2 = float(hash2) / float(0xFFFFFFFF)
    tpdf = r1 + r2 - 1.0
    return 0.5 + tpdf * amplitude


def dither_transformed(input_image, bits=1, seed=0, delay=0, return_error=False, tpdf=False, method='1st'):
    """Dither with random spatial transform derived from seed."""
    swap_xy = bool(seed & 1)
    mirror_x = bool(seed & 2)
    mirror_y = bool(seed & 4)

    transformed = transform_image(input_image, swap_xy, mirror_x, mirror_y)

    if return_error:
        result, error_map = dither(transformed, bits=bits, seed=seed, delay=delay, return_error=True, tpdf=tpdf, method=method)
        return (
            inverse_transform_image(result, swap_xy, mirror_x, mirror_y),
            inverse_transform_image(error_map, swap_xy, mirror_x, mirror_y),
        )
    else:
        result = dither(transformed, bits=bits, seed=seed, delay=delay, tpdf=tpdf, method=method)
        return inverse_transform_image(result, swap_xy, mirror_x, mirror_y)


def compute_power_spectrum(img):
    """Compute 2D power spectrum with log scaling."""
    centered = img - img.mean()
    spectrum = fftshift(fft2(centered))
    power = np.abs(spectrum) ** 2
    return np.log1p(power)


def compute_segmented_radial_power(img):
    """Compute segmented radially averaged power spectrum.
    Returns (frequencies, horizontal, diagonal, vertical) in linear power."""
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


def analyze_ranked_output(rank, output_dir, gray_levels=None):
    """Analyze the ranked output at multiple gray levels, compared to void-and-cluster.
    Generates charts with: halftone, 2D spectrum, radial power + reference lines."""
    if gray_levels is None:
        gray_levels = [32, 64, 85, 127, 170, 191, 224]

    # Load void-and-cluster reference
    ref_path = Path(__file__).parent.parent / "test_images" / "blue_noise_256.png"
    ref_raw = np.array(Image.open(ref_path).convert('L')) if ref_path.exists() else None

    for gray in gray_levels:
        # Threshold both arrays
        halftone_ours = (rank < gray).astype(np.float64) * 255.0
        density = halftone_ours.mean() / 255.0

        panels = [('Recursive (ours)', halftone_ours)]
        if ref_raw is not None:
            halftone_ref = (ref_raw < gray).astype(np.float64) * 255.0
            panels.append(('Void-and-Cluster', halftone_ref))
        n_cols = len(panels)

        # Compute all radial spectra for shared y-axis
        all_radial = {}
        for label, ht in panels:
            freqs, h_pow, d_pow, v_pow = compute_segmented_radial_power(ht)
            h_db = 10 * np.log10(h_pow + 1e-10)
            d_db = 10 * np.log10(d_pow + 1e-10)
            v_db = 10 * np.log10(v_pow + 1e-10)
            all_radial[label] = (freqs, h_db, d_db, v_db)

        all_db = np.concatenate([np.concatenate([h, d, v]) for _, (_, h, d, v) in all_radial.items()])
        y_min = all_db.min() - 5
        y_max = all_db.max() + 5

        fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 14))
        if n_cols == 1:
            axes = axes.reshape(3, 1)

        for col, (label, ht) in enumerate(panels):
            axes[0, col].imshow(ht, cmap='gray', vmin=0, vmax=255)
            axes[0, col].set_title(f'{label}\n{density*100:.1f}% white', fontsize=11, fontweight='bold')
            axes[0, col].axis('off')

            spectrum_2d = compute_power_spectrum(ht)
            axes[1, col].imshow(spectrum_2d, cmap='gray')
            axes[1, col].axis('off')

            freqs, h_db, d_db, v_db = all_radial[label]
            freqs_ideal = np.linspace(0.004, 0.5, 500)
            P_ref = 10 ** (h_db.max() / 10)
            power_3db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) + 1e-10)
            power_6db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 2 + 1e-10)
            power_12db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 4 + 1e-10)

            axes[2, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
            axes[2, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
            axes[2, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
            axes[2, col].plot(freqs_ideal, power_6db, 'k--', linewidth=1.5, alpha=0.6, label='f² (+6dB/oct)')
            axes[2, col].plot(freqs_ideal, power_3db, 'k:', linewidth=1.5, alpha=0.6, label='f (+3dB/oct)')
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

        fig.suptitle(f'Dither Array Comparison — gray {gray}/255', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0.03, 0, 1, 0.96])
        out_path = output_dir / f"analysis_gray_{gray:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # Histogram of rank values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rank.ravel(), bins=256, range=(0, 256), color='steelblue', edgecolor='none')
    ax.set_xlabel('Rank value')
    ax.set_ylabel('Count')
    ax.set_title('Rank Value Histogram (should be uniform)')
    plt.tight_layout()
    hist_path = output_dir / "analysis_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {hist_path}")

    # Spectral slope at every threshold
    print("Computing spectral slopes across all thresholds...")
    thresholds = np.arange(1, 256)
    slopes = []
    for t in thresholds:
        halftone = (rank < t).astype(np.float64) * 255.0
        freqs_t, h_t, d_t, v_t = compute_segmented_radial_power(halftone)
        avg_pow = (h_t + d_t + v_t) / 3.0
        avg_db = 10 * np.log10(avg_pow + 1e-10)
        # Fit slope in log-log space (log2(freq) vs dB)
        # dB/octave = slope of dB vs log2(freq)
        valid = freqs_t > 0.02  # skip very low freqs
        if valid.sum() > 10:
            log2_f = np.log2(freqs_t[valid])
            db_vals = avg_db[valid]
            coeffs = np.polyfit(log2_f, db_vals, 1)
            slopes.append(coeffs[0])  # dB per octave
        else:
            slopes.append(0)
    slopes = np.array(slopes)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, slopes, 'b-', linewidth=1)
    ax.axhline(y=6, color='k', linestyle='--', alpha=0.5, label='+6 dB/oct (ideal)')
    ax.axhline(y=3, color='k', linestyle=':', alpha=0.5, label='+3 dB/oct')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Threshold (rank value)')
    ax.set_ylabel('Spectral Slope (dB/octave)')
    ax.set_title('Spectral Slope vs Threshold')
    ax.set_xlim(1, 255)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    slope_path = output_dir / "analysis_slopes.png"
    plt.savefig(slope_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {slope_path}")
    print(f"  Slope range: [{slopes.min():.2f}, {slopes.max():.2f}] dB/oct")
    print(f"  Mean slope: {slopes.mean():.2f} dB/oct")

    # Analysis of the ranked output vs blue_noise_256 reference
    rank_img = rank.astype(np.float64)
    ref_path = Path(__file__).parent.parent / "test_images" / "blue_noise_256.png"
    ref_img = np.array(Image.open(ref_path).convert('L')).astype(np.float64) if ref_path.exists() else None

    panels = [('Recursive (ours)', rank_img)]
    if ref_img is not None:
        panels.append(('Void-and-Cluster', ref_img))
    n_cols = len(panels)

    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 14))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    # Compute all spectra first for shared y-axis
    all_radial = {}
    for label, img in panels:
        freqs, h_pow, d_pow, v_pow = compute_segmented_radial_power(img)
        h_db = 10 * np.log10(h_pow + 1e-10)
        d_db = 10 * np.log10(d_pow + 1e-10)
        v_db = 10 * np.log10(v_pow + 1e-10)
        all_radial[label] = (freqs, h_db, d_db, v_db)

    all_db = np.concatenate([np.concatenate([h, d, v]) for _, (_, h, d, v) in all_radial.items()])
    y_min = all_db.min() - 5
    y_max = all_db.max() + 5

    for col, (label, img) in enumerate(panels):
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        spectrum_2d = compute_power_spectrum(img)
        axes[1, col].imshow(spectrum_2d, cmap='gray')
        axes[1, col].axis('off')

        freqs, h_db, d_db, v_db = all_radial[label]

        freqs_ideal = np.linspace(0.004, 0.5, 500)
        P_ref = 10 ** (h_db.max() / 10)
        power_3db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) + 1e-10)
        power_6db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 2 + 1e-10)
        power_12db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 4 + 1e-10)

        axes[2, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs_ideal, power_6db, 'k--', linewidth=1.5, alpha=0.6, label='f² (+6dB/oct)')
        axes[2, col].plot(freqs_ideal, power_3db, 'k:', linewidth=1.5, alpha=0.6, label='f (+3dB/oct)')
        axes[2, col].plot(freqs_ideal, power_12db, 'k-.', linewidth=1.5, alpha=0.6, label='f⁴ (+12dB/oct)')
        axes[2, col].set_xlim(0, 0.5)
        axes[2, col].set_ylim(y_min, y_max)
        axes[2, col].set_xlabel('Spatial Frequency (cycles/pixel)')
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].legend(loc='lower right', fontsize=8)
        if col == 0:
            axes[2, col].set_ylabel('Power (dB)')

    fig.text(0.02, 0.83, 'Array', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.5, 'Spectrum', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.17, 'Radial', va='center', rotation='vertical', fontsize=12)

    fig.suptitle('Dither Array Comparison — Raw Ranked Output', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    out_path = output_dir / "analysis_ranked.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _dither_1st_order(input_image, bits, seed, delay, return_error, tpdf):
    """First-order mixed FS/JJN with serpentine scanning."""
    height, width = input_image.shape
    r = REACH_1ST
    s = r  # seed = reach for first-order
    buf = create_seeded_buffer_1st(input_image)
    hashed_seed = lowbias32(np.uint32(seed))
    use_tpdf = tpdf and bits == 1
    error_map = np.zeros((height, width), dtype=np.float64) if return_error else None
    fifo = deque()

    total_left = r * 2
    total_top = r
    bx_start = r
    process_height = r + height
    process_width = r + width + r

    for y in range(process_height):
        is_rtl = y % 2 == 1
        px_range = range(process_width - 1, -1, -1) if is_rtl else range(process_width)

        for px in px_range:
            bx = bx_start + px
            old_val = buf[y, bx]

            img_x = px - r
            img_y = y - r
            in_image = (0 <= img_x < width) and (0 <= img_y < height)

            if return_error and in_image:
                error_map[img_y, img_x] = old_val

            if use_tpdf and in_image:
                thresh = tpdf_threshold(img_x, img_y, hashed_seed)
                new_val = 1.0 if old_val > thresh else 0.0
            else:
                new_val = quantize(old_val, bits)
            buf[y, bx] = new_val
            err = old_val - new_val

            coord_x = img_x & 0xFFFF
            coord_y = img_y & 0xFFFF
            coord_hash = lowbias32(np.uint32(coord_x) ^ (np.uint32(coord_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1

            fifo.append((bx, y, err, use_jjn, is_rtl))

            if len(fifo) > delay:
                dbx, dy, derr, d_jjn, d_rtl = fifo.popleft()
                apply_error_1st(buf, dbx, dy, derr, d_jjn, d_rtl)

    while fifo:
        dbx, dy, derr, d_jjn, d_rtl = fifo.popleft()
        apply_error_1st(buf, dbx, dy, derr, d_jjn, d_rtl)

    output = buf[total_top:total_top + height, total_left:total_left + width].copy()
    if return_error:
        return output, error_map
    return output


def _dither_2hh2(input_image, bits, seed, delay, return_error, tpdf):
    """2H-H² kernel with mixed FS²/JJN², always LTR."""
    height, width = input_image.shape
    r = REACH_2ND
    s = SEED_2ND
    buf = create_seeded_buffer_2nd(input_image)
    hashed_seed = lowbias32(np.uint32(seed))
    use_tpdf = tpdf and bits == 1
    error_map = np.zeros((height, width), dtype=np.float64) if return_error else None
    fifo = deque()

    total_left = r + s
    total_top = s
    bx_start = r
    process_height = s + height
    process_width = s + width + s

    for y in range(process_height):
        for px in range(process_width):
            bx = bx_start + px
            old_val = buf[y, bx]

            img_x = px - s
            img_y = y - s
            in_image = (0 <= img_x < width) and (0 <= img_y < height)

            if return_error and in_image:
                error_map[img_y, img_x] = old_val

            if use_tpdf and in_image:
                thresh = tpdf_threshold(img_x, img_y, hashed_seed)
                new_val = 1.0 if old_val > thresh else 0.0
            else:
                new_val = quantize(old_val, bits)
            buf[y, bx] = new_val
            err = old_val - new_val

            coord_x = img_x & 0xFFFF
            coord_y = img_y & 0xFFFF
            coord_hash = lowbias32(np.uint32(coord_x) ^ (np.uint32(coord_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1

            fifo.append((bx, y, err, use_jjn))

            if len(fifo) > delay:
                dbx, dy, derr, d_jjn = fifo.popleft()
                apply_error_2nd(buf, dbx, dy, derr, d_jjn)

    while fifo:
        dbx, dy, derr, d_jjn = fifo.popleft()
        apply_error_2nd(buf, dbx, dy, derr, d_jjn)

    output = buf[total_top:total_top + height, total_left:total_left + width].copy()
    if return_error:
        return output, error_map
    return output


def _dither_dual_integrator(input_image, bits, seed, delay, return_error, tpdf):
    """Dual-integrator: two coupled error buffers with first-order kernels, serpentine."""
    height, width = input_image.shape
    r = REACH_1ST
    s = r

    buf1 = create_seeded_buffer_1st(input_image)
    buf2 = np.zeros_like(buf1)

    hashed_seed = lowbias32(np.uint32(seed))
    use_tpdf = tpdf and bits == 1
    error_map = np.zeros((height, width), dtype=np.float64) if return_error else None

    total_left = r * 2
    total_top = r
    bx_start = r
    process_height = r + height
    process_width = r + width + r

    for y in range(process_height):
        is_rtl = y % 2 == 1
        px_range = range(process_width - 1, -1, -1) if is_rtl else range(process_width)

        for px in px_range:
            bx = bx_start + px

            int1_val = buf1[y, bx]
            int2_val = int1_val + buf2[y, bx]

            img_x = px - r
            img_y = y - r
            in_image = (0 <= img_x < width) and (0 <= img_y < height)

            if return_error and in_image:
                error_map[img_y, img_x] = int2_val

            if use_tpdf and in_image:
                thresh = tpdf_threshold(img_x, img_y, hashed_seed)
                new_val = 1.0 if int2_val > thresh else 0.0
            else:
                new_val = 1.0 if int2_val > 0.5 else 0.0 if bits == 1 else quantize(int2_val, bits)
            buf1[y, bx] = new_val

            err1 = int1_val - new_val
            err2 = int2_val - new_val

            coord_x = img_x & 0xFFFF
            coord_y = img_y & 0xFFFF
            coord_hash = lowbias32(np.uint32(coord_x) ^ (np.uint32(coord_y) << np.uint32(16)) ^ hashed_seed)
            use_jjn_1 = (coord_hash & 1) == 1
            use_jjn_2 = (coord_hash & 2) == 2

            apply_error_1st(buf1, bx, y, err1, use_jjn_1, is_rtl)
            apply_error_1st(buf2, bx, y, err2, use_jjn_2, is_rtl)

    output = buf1[total_top:total_top + height, total_left:total_left + width].copy()
    if return_error:
        return output, error_map
    return output


def dither(
    input_image: np.ndarray,
    bits: int = 1,
    seed: int = 0,
    delay: int = 0,
    return_error: bool = False,
    tpdf: bool = False,
    method: str = '1st'
):
    """
    Apply error diffusion dithering.

    Args:
        input_image: Input image with values in [0.0, 1.0]
        bits: Output bit depth (1 = binary, 2 = 4 levels, etc.)
        seed: Random seed for kernel selection
        delay: FIFO delay in pixels before error is diffused (0 = immediate)
        return_error: If True, also return the per-pixel error buffer value
        tpdf: If True, add TPDF threshold perturbation (1-bit only)
        method: '1st' (first-order FS/JJN), '2hh2' (2H-H² kernels), 'dual' (dual integrator)

    Returns:
        Dithered image, or (dithered, error_map) if return_error=True
    """
    if method == '2hh2':
        return _dither_2hh2(input_image, bits, seed, delay, return_error, tpdf)
    elif method == 'dual':
        return _dither_dual_integrator(input_image, bits, seed, delay, return_error, tpdf)
    else:
        return _dither_1st_order(input_image, bits, seed, delay, return_error, tpdf)


def generate_gradient(width: int, height: int) -> np.ndarray:
    """Generate a horizontal gradient from 0 to 1."""
    return np.tile(np.linspace(0, 1, width), (height, 1))


def main():
    parser = argparse.ArgumentParser(
        description="Mixed FS/JJN error diffusion in floating point"
    )
    parser.add_argument("--bits", type=int, default=1,
                        help="Output bit depth (default: 1)")
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--gray", type=float,
                        help="Dither uniform gray level (0.0-1.0)")
    parser.add_argument("--delay", type=int, default=0,
                        help="FIFO delay in pixels before error diffusion (default: 0)")
    parser.add_argument("--gradient", nargs="+", type=int,
                        help="Generate gradient at specified bit depths")
    parser.add_argument("--recursive-test", action="store_true",
                        help="Run recursive generation experiment")
    parser.add_argument("--tpdf", action="store_true",
                        help="Add TPDF threshold perturbation on top of mixed FS/JJN")
    parser.add_argument("--method", type=str, default="1st",
                        choices=["1st", "2hh2", "dual"],
                        help="Dither method: 1st (first-order), 2hh2 (2H-H²), dual (dual integrator)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path")

    args = parser.parse_args()
    output_dir = Path(__file__).parent

    if args.gradient:
        # Generate gradient visualizations at multiple bit depths
        for bits in args.gradient:
            print(f"\nGenerating {bits}-bit gradient...")

            # Create gradient
            gradient = generate_gradient(args.size, args.size)

            # Dither it
            dithered = dither(gradient, bits=bits, seed=args.seed, delay=args.delay, tpdf=args.tpdf, method=args.method)

            # Save as PNG (scale to 0-255)
            delay_suffix = f"_delay{args.delay}" if args.delay > 0 else ""
            out_path = output_dir / f"gradient_{bits}bit{delay_suffix}.png"
            img = (dithered * 255).astype(np.uint8)
            Image.fromarray(img, mode='L').save(out_path)
            print(f"Saved: {out_path}")

            # Save raw float data
            npy_path = output_dir / f"gradient_{bits}bit{delay_suffix}.npy"
            np.save(npy_path, dithered)
            print(f"Saved: {npy_path}")

            # Stats
            unique = np.unique(dithered)
            print(f"Unique levels: {len(unique)} (expected {2**bits})")
            print(f"Levels: {unique}")

    elif args.recursive_test:
        size = args.size
        next_seed = [args.seed]
        N = 8  # bits for final output

        def get_seed():
            s = next_seed[0]
            next_seed[0] += 1
            return s

        gray = np.full((size, size), 0.5, dtype=np.float64)
        rank = np.zeros((size, size), dtype=np.int32)

        def save_intermediate(rank_arr, level, n_bits):
            """Save the rank array snapped to the current number of resolved bits."""
            n_levels = 2 ** (level + 1)
            # Scale rank to 0-255 range for visualization
            # At this level, only the top (level+1) bits are set, so values are multiples of 2^(N-1-level)
            step = 2 ** (N - 1 - level)
            scaled = (rank_arr // step) * (255 // (n_levels - 1)) if n_levels > 1 else rank_arr
            path = output_dir / f"ranked_level{level}.png"
            Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), mode='L').save(path)
            print(f"  Saved: {path} ({n_levels} levels)")

        use_tpdf = args.tpdf
        if use_tpdf:
            print("TPDF threshold perturbation enabled")

        # Level 0: initial 1-bit split
        print("Level 0: 1-bit dither of 0.5 gray")
        method = args.method
        print(f"Method: {method}")
        result0 = dither_transformed(gray, bits=1, seed=get_seed(), tpdf=use_tpdf, method=method)
        rank |= (result0 > 0.5).astype(np.int32) << (N - 1)

        img0 = Image.fromarray((result0 * 255).astype(np.uint8), mode='L')
        img0.save(output_dir / "step1_1bit.png")
        w = (result0 > 0.5).sum()
        print(f"  white: {w} ({w/result0.size*100:.1f}%)")
        save_intermediate(rank, 0, N)

        # Tree: list of (result_image, mask)
        nodes = [(result0, np.ones((size, size), dtype=bool))]
        total_passes = 1

        for level in range(1, N):
            bit_pos = N - 1 - level
            num_nodes = len(nodes)
            print(f"\nLevel {level}: {num_nodes} nodes, determining bit {bit_pos}...")
            new_nodes = []

            for i, (parent_result, parent_mask) in enumerate(nodes):
                went_high = parent_mask & (parent_result > 0.5)
                went_low = parent_mask & (~(parent_result > 0.5))

                # Clean parent: force non-member pixels to boundary values
                # so they don't accidentally become 0.5 after scaling
                clean = parent_result.copy()
                # Non-member pixels that are 1 would become 0.5 in the hi scaling
                # Non-member pixels that are 0 would become 0.5 in the lo scaling
                # Set non-members to 0 for hi split, 1 for lo split

                # Split "high" group: scale to [0, 0.5]
                # Member 1→0.5 (split), Member 0→0 (boundary)
                # Non-member must be 0 so they stay at 0 after scaling
                input_hi = clean.copy()
                input_hi[~parent_mask] = 0.0
                input_hi = input_hi * 0.5
                result_hi = dither_transformed(input_hi, bits=1, seed=get_seed(), tpdf=use_tpdf, method=method)
                rank[went_high] |= (result_hi > 0.5).astype(np.int32)[went_high] << bit_pos
                new_nodes.append((result_hi, went_high))
                total_passes += 1

                # Split "low" group: scale to [0.5, 1]
                # Member 0→0.5 (split), Member 1→1 (boundary)
                # Non-member must be 1 so they stay at 1 after scaling
                input_lo = clean.copy()
                input_lo[~parent_mask] = 1.0
                input_lo = input_lo * 0.5 + 0.5
                result_lo = dither_transformed(input_lo, bits=1, seed=get_seed(), tpdf=use_tpdf, method=method)
                rank[went_low] |= (result_lo > 0.5).astype(np.int32)[went_low] << bit_pos
                new_nodes.append((result_lo, went_low))
                total_passes += 1

            nodes = new_nodes
            save_intermediate(rank, level, N)

        print(f"\nTotal dither passes: {total_passes}")

        # Output final ranked image
        unique_ranks = np.unique(rank)
        print(f"Unique rank values: {len(unique_ranks)} (expected {2**N})")
        print(f"Rank range: [{rank.min()}, {rank.max()}]")

        # Save as 8-bit grayscale
        out_path = output_dir / "ranked_output.png"
        Image.fromarray(rank.astype(np.uint8), mode='L').save(out_path)
        print(f"Saved: {out_path}")
        np.save(output_dir / "ranked_output.npy", rank)

        # Spectral analysis
        print("\n--- Spectral Analysis ---")
        analyze_ranked_output(rank, output_dir)

    elif args.gray is not None:
        # Dither uniform gray
        print(f"Dithering {args.gray:.3f} gray at {args.bits}-bit...")

        input_img = np.full((args.size, args.size), args.gray, dtype=np.float64)
        dithered, error_map = dither(input_img, bits=args.bits, seed=args.seed, delay=args.delay, return_error=True, tpdf=args.tpdf, method=args.method)

        out_path = args.output or str(output_dir / f"gray_{args.gray:.2f}_{args.bits}bit.png")
        img = (dithered * 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(out_path)
        print(f"Saved: {out_path}")

        # Stats
        unique, counts = np.unique(dithered, return_counts=True)
        total = args.size * args.size
        print(f"\nLevel distribution:")
        for level, count in zip(unique, counts):
            print(f"  {level:.4f}: {count:6d} ({count/total*100:.2f}%)")

        # Error (target) output
        emin, emax = error_map.min(), error_map.max()
        print(f"\nTarget range: [{emin:.4f}, {emax:.4f}]")
        error_norm = (error_map - emin) / (emax - emin) if emax > emin else error_map
        err_path = output_dir / f"gray_{args.gray:.2f}_{args.bits}bit_error.png"
        Image.fromarray((error_norm * 255).astype(np.uint8), mode='L').save(err_path)
        print(f"Saved: {err_path}")
        np.save(output_dir / f"gray_{args.gray:.2f}_{args.bits}bit_error.npy", error_map)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
