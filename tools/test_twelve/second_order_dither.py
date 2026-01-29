#!/usr/bin/env python3
"""
Second-order error diffusion experiment.

Standard error diffusion (first-order) gives +6 dB/octave noise shaping.
Second-order feeds back a shaped error: d = 2*err - C*err, giving NTF = (1-C)²
which should achieve +12 dB/octave.

Uses mixed FS/JJN kernel switching for forward diffusion (our method),
with a fixed FS reverse kernel for the second-order prediction.

Usage:
    python second_order_dither.py                    # Gray levels + gradient
    python second_order_dither.py --gray 0.5         # Single gray level
    python second_order_dither.py --gradient-only     # Gradient only
"""

import numpy as np
from numpy.fft import fft2, fftshift
from PIL import Image
from pathlib import Path
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
# Error diffusion kernels
# ============================================================================

def apply_fs_ltr(buf, x, y, err):
    h, w = buf.shape
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 16.0)
    if y + 1 < h:
        if x > 0:
            buf[y + 1, x - 1] += err * (3.0 / 16.0)
        buf[y + 1, x] += err * (5.0 / 16.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (1.0 / 16.0)


def apply_fs_rtl(buf, x, y, err):
    h, w = buf.shape
    if x > 0:
        buf[y, x - 1] += err * (7.0 / 16.0)
    if y + 1 < h:
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 16.0)
        buf[y + 1, x] += err * (5.0 / 16.0)
        if x > 0:
            buf[y + 1, x - 1] += err * (1.0 / 16.0)


def apply_jjn_ltr(buf, x, y, err):
    h, w = buf.shape
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 48.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (5.0 / 48.0)
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (3.0 / 48.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 48.0)
        buf[y + 1, x] += err * (7.0 / 48.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 48.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (3.0 / 48.0)
    if y + 2 < h:
        if x >= 2:
            buf[y + 2, x - 2] += err * (1.0 / 48.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (3.0 / 48.0)
        buf[y + 2, x] += err * (5.0 / 48.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (3.0 / 48.0)
        if x + 2 < w:
            buf[y + 2, x + 2] += err * (1.0 / 48.0)


def apply_jjn_rtl(buf, x, y, err):
    h, w = buf.shape
    if x >= 1:
        buf[y, x - 1] += err * (7.0 / 48.0)
    if x >= 2:
        buf[y, x - 2] += err * (5.0 / 48.0)
    if y + 1 < h:
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (3.0 / 48.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 48.0)
        buf[y + 1, x] += err * (7.0 / 48.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 48.0)
        if x >= 2:
            buf[y + 1, x - 2] += err * (3.0 / 48.0)
    if y + 2 < h:
        if x + 2 < w:
            buf[y + 2, x + 2] += err * (1.0 / 48.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (3.0 / 48.0)
        buf[y + 2, x] += err * (5.0 / 48.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (3.0 / 48.0)
        if x >= 2:
            buf[y + 2, x - 2] += err * (1.0 / 48.0)


def apply_error(buf, x, y, err, use_jjn, is_rtl):
    if use_jjn:
        if is_rtl:
            apply_jjn_rtl(buf, x, y, err)
        else:
            apply_jjn_ltr(buf, x, y, err)
    else:
        if is_rtl:
            apply_fs_rtl(buf, x, y, err)
        else:
            apply_fs_ltr(buf, x, y, err)


# ============================================================================
# Reverse kernel lookup (what previous errors contributed to current pixel)
# ============================================================================

def predicted_incoming_fs(err_history, x, y, is_rtl):
    """Estimate error that was diffused into (x,y) using FS reverse kernel.

    For LTR row (even y), previous row was RTL:
        7/16 from (x-1, y)   - same row, left
        3/16 from (x-1, y-1) - upper-left (from RTL row's forward-below)
        5/16 from (x, y-1)   - above
        1/16 from (x+1, y-1) - upper-right (from RTL row's back-below)

    For RTL row (odd y), previous row was LTR:
        7/16 from (x+1, y)   - same row, right
        3/16 from (x+1, y-1) - upper-right (from LTR row's back-below)
        5/16 from (x, y-1)   - above
        1/16 from (x-1, y-1) - upper-left (from LTR row's forward-below)
    """
    h, w = err_history.shape
    pred = 0.0

    if is_rtl:
        # Previous pixel is to the right
        if x + 1 < w:
            pred += (7.0 / 16.0) * err_history[y, x + 1]
        if y > 0:
            if x + 1 < w:
                pred += (3.0 / 16.0) * err_history[y - 1, x + 1]
            pred += (5.0 / 16.0) * err_history[y - 1, x]
            if x > 0:
                pred += (1.0 / 16.0) * err_history[y - 1, x - 1]
    else:
        # Previous pixel is to the left
        if x > 0:
            pred += (7.0 / 16.0) * err_history[y, x - 1]
        if y > 0:
            if x > 0:
                pred += (3.0 / 16.0) * err_history[y - 1, x - 1]
            pred += (5.0 / 16.0) * err_history[y - 1, x]
            if x + 1 < w:
                pred += (1.0 / 16.0) * err_history[y - 1, x + 1]

    return pred


# ============================================================================
# Dithering functions
# ============================================================================

def dither_first_order(input_image, seed=0):
    """Standard mixed FS/JJN error diffusion (first-order, +6 dB/oct)."""
    height, width = input_image.shape
    buf = input_image.copy().astype(np.float64)
    output = np.zeros((height, width), dtype=np.float64)
    hashed_seed = lowbias32(np.uint32(seed))

    for y in range(height):
        is_rtl = y % 2 == 1
        x_range = range(width - 1, -1, -1) if is_rtl else range(width)

        for x in x_range:
            old_val = buf[y, x]
            new_val = 1.0 if old_val > 0.5 else 0.0
            output[y, x] = new_val
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1
            apply_error(buf, x, y, err, use_jjn, is_rtl)

    return output


def dither_second_order(input_image, seed=0):
    """Second-order mixed FS/JJN error diffusion (+12 dB/oct target).

    Diffuses shaped error: d = 2*err - C*err (reverse FS kernel on raw errors).
    NTF = (1 - C(z))^2 for +12 dB/octave noise shaping.
    """
    height, width = input_image.shape
    buf = input_image.copy().astype(np.float64)
    output = np.zeros((height, width), dtype=np.float64)
    err_history = np.zeros((height, width), dtype=np.float64)
    hashed_seed = lowbias32(np.uint32(seed))

    for y in range(height):
        is_rtl = y % 2 == 1
        x_range = range(width - 1, -1, -1) if is_rtl else range(width)

        for x in x_range:
            old_val = buf[y, x]
            new_val = 1.0 if old_val > 0.5 else 0.0
            output[y, x] = new_val
            err = old_val - new_val

            # Second-order: shaped_err = 2*err - predicted_incoming
            predicted = predicted_incoming_fs(err_history, x, y, is_rtl)
            shaped_err = 2.0 * err - predicted

            # Store raw error for future reverse lookups
            err_history[y, x] = err

            # Diffuse shaped error using mixed FS/JJN
            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1
            apply_error(buf, x, y, shaped_err, use_jjn, is_rtl)

    return output


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
        ('2nd order (mixed FS/JJN)', dither_second_order),
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
        ('2nd order', dither_second_order),
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
        safe_label = label.replace(' ', '_')
        path = output_dir / f"gradient_{safe_label}.png"
        Image.fromarray(img, mode='L').save(path)
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

    args = parser.parse_args()
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gradient_only:
        analyze_gradient(output_dir, size=args.size, seed=args.seed)
        return

    gray_levels = args.gray if args.gray else [0.125, 0.25, 0.333, 0.5, 0.667, 0.75, 0.875]

    for g in gray_levels:
        analyze_gray(g, output_dir, size=args.size, seed=args.seed)

    analyze_gradient(output_dir, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
