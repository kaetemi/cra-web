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


def apply_fs_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Floyd-Steinberg kernel, left-to-right."""
    h, w = buf.shape
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 16.0)
    if y + 1 < h:
        if x > 0:
            buf[y + 1, x - 1] += err * (3.0 / 16.0)
        buf[y + 1, x] += err * (5.0 / 16.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (1.0 / 16.0)


def apply_fs_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Floyd-Steinberg kernel, right-to-left."""
    h, w = buf.shape
    if x > 0:
        buf[y, x - 1] += err * (7.0 / 16.0)
    if y + 1 < h:
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 16.0)
        buf[y + 1, x] += err * (5.0 / 16.0)
        if x > 0:
            buf[y + 1, x - 1] += err * (1.0 / 16.0)


def apply_jjn_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Jarvis-Judice-Ninke kernel, left-to-right."""
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


def apply_jjn_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Jarvis-Judice-Ninke kernel, right-to-left."""
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


def dither_transformed(input_image, bits=1, seed=0, delay=0, return_error=False):
    """Dither with random spatial transform derived from seed."""
    swap_xy = bool(seed & 1)
    mirror_x = bool(seed & 2)
    mirror_y = bool(seed & 4)

    transformed = transform_image(input_image, swap_xy, mirror_x, mirror_y)

    if return_error:
        result, error_map = dither(transformed, bits=bits, seed=seed, delay=delay, return_error=True)
        return (
            inverse_transform_image(result, swap_xy, mirror_x, mirror_y),
            inverse_transform_image(error_map, swap_xy, mirror_x, mirror_y),
        )
    else:
        result = dither(transformed, bits=bits, seed=seed, delay=delay)
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
    """Analyze the ranked output at multiple gray levels.
    Generates charts with: halftone, 2D spectrum, radial power + reference lines."""
    if gray_levels is None:
        gray_levels = [32, 64, 85, 127, 170, 191, 224]

    for gray in gray_levels:
        # Threshold the rank array at this gray level
        threshold = gray  # rank 0..255, threshold 0..255
        halftone = (rank < threshold).astype(np.float64) * 255.0
        density = halftone.mean() / 255.0

        # Compute spectra
        freqs, h_pow, d_pow, v_pow = compute_segmented_radial_power(halftone)
        h_db = 10 * np.log10(h_pow + 1e-10)
        d_db = 10 * np.log10(d_pow + 1e-10)
        v_db = 10 * np.log10(v_pow + 1e-10)

        # Reference lines
        freqs_ideal = np.linspace(0.004, 0.5, 500)
        P_ref = 10 ** (h_db.max() / 10)
        power_3db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) + 1e-10)
        power_6db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 2 + 1e-10)
        power_12db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 4 + 1e-10)

        # 2D spectrum
        spectrum_2d = compute_power_spectrum(halftone)

        # Plot: 3 rows (halftone, spectrum, radial), 1 column
        fig, axes = plt.subplots(3, 1, figsize=(8, 14))

        # Row 1: Halftone
        axes[0].imshow(halftone, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f'Threshold at {gray} ({density*100:.1f}% white)', fontsize=12)
        axes[0].axis('off')

        # Row 2: 2D FFT spectrum
        axes[1].imshow(spectrum_2d, cmap='gray')
        axes[1].set_title('2D Power Spectrum', fontsize=12)
        axes[1].axis('off')

        # Row 3: Radial power + reference lines
        axes[2].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[2].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[2].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[2].plot(freqs_ideal, power_6db, 'k--', linewidth=1.5, alpha=0.6, label='f² (+6dB/oct)')
        axes[2].plot(freqs_ideal, power_3db, 'k:', linewidth=1.5, alpha=0.6, label='f (+3dB/oct)')
        axes[2].plot(freqs_ideal, power_12db, 'k-.', linewidth=1.5, alpha=0.6, label='f⁴ (+12dB/oct)')
        axes[2].set_xlim(0, 0.5)
        axes[2].set_xlabel('Spatial Frequency (cycles/pixel)')
        axes[2].set_ylabel('Power (dB)')
        axes[2].set_title('Radial Power Spectrum', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='lower right', fontsize=8)

        fig.suptitle(f'Recursive Dither Array — gray {gray}/255', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
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

    # Analysis of the ranked output image itself (as grayscale)
    rank_img = rank.astype(np.float64)
    freqs, h_pow, d_pow, v_pow = compute_segmented_radial_power(rank_img)
    h_db = 10 * np.log10(h_pow + 1e-10)
    d_db = 10 * np.log10(d_pow + 1e-10)
    v_db = 10 * np.log10(v_pow + 1e-10)

    freqs_ideal = np.linspace(0.004, 0.5, 500)
    P_ref = 10 ** (h_db.max() / 10)
    power_3db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) + 1e-10)
    power_6db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 2 + 1e-10)
    power_12db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 4 + 1e-10)

    spectrum_2d = compute_power_spectrum(rank_img)

    fig, axes = plt.subplots(3, 1, figsize=(8, 14))
    axes[0].imshow(rank_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Ranked Output (8-bit dither array)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(spectrum_2d, cmap='gray')
    axes[1].set_title('2D Power Spectrum', fontsize=12)
    axes[1].axis('off')

    axes[2].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
    axes[2].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
    axes[2].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
    axes[2].plot(freqs_ideal, power_6db, 'k--', linewidth=1.5, alpha=0.6, label='f² (+6dB/oct)')
    axes[2].plot(freqs_ideal, power_3db, 'k:', linewidth=1.5, alpha=0.6, label='f (+3dB/oct)')
    axes[2].plot(freqs_ideal, power_12db, 'k-.', linewidth=1.5, alpha=0.6, label='f⁴ (+12dB/oct)')
    axes[2].set_xlim(0, 0.5)
    axes[2].set_xlabel('Spatial Frequency (cycles/pixel)')
    axes[2].set_ylabel('Power (dB)')
    axes[2].set_title('Radial Power Spectrum', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='lower right', fontsize=8)

    fig.suptitle('Recursive Dither Array — Raw Ranked Output', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "analysis_ranked.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def apply_error(buf, x, y, err, use_jjn, is_rtl):
    """Apply the appropriate error diffusion kernel."""
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


def dither(
    input_image: np.ndarray,
    bits: int = 1,
    seed: int = 0,
    delay: int = 0,
    return_error: bool = False
):
    """
    Apply mixed FS/JJN error diffusion dithering.

    Args:
        input_image: Input image with values in [0.0, 1.0]
        bits: Output bit depth (1 = binary, 2 = 4 levels, etc.)
        seed: Random seed for kernel selection
        delay: FIFO delay in pixels before error is diffused (0 = immediate)
        return_error: If True, also return the per-pixel error buffer value
                      (buf[y,x] at time of quantization), centered around 0.5

    Returns:
        Dithered image, or (dithered, error_map) if return_error=True
    """
    height, width = input_image.shape
    buf = input_image.copy().astype(np.float64)
    output = np.zeros((height, width), dtype=np.float64)
    error_map = np.zeros((height, width), dtype=np.float64) if return_error else None
    hashed_seed = lowbias32(np.uint32(seed))
    fifo = deque()

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            if return_error:
                error_map[y, x] = old_val
            new_val = quantize(old_val, bits)
            output[y, x] = new_val
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1

            fifo.append((x, y, err, use_jjn, is_rtl))

            if len(fifo) > delay:
                dx, dy, derr, d_jjn, d_rtl = fifo.popleft()
                apply_error(buf, dx, dy, derr, d_jjn, d_rtl)

    # Flush remaining
    while fifo:
        dx, dy, derr, d_jjn, d_rtl = fifo.popleft()
        apply_error(buf, dx, dy, derr, d_jjn, d_rtl)

    if return_error:
        return output, error_map
    return output


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
            dithered = dither(gradient, bits=bits, seed=args.seed, delay=args.delay)

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

        # Level 0: initial 1-bit split
        print("Level 0: 1-bit dither of 0.5 gray")
        result0 = dither_transformed(gray, bits=1, seed=get_seed())
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

                # Split "high" group: scale to [0, 0.5]
                # Parent 1→0.5 (split), Parent 0→0 (boundary)
                input_hi = parent_result * 0.5
                result_hi = dither_transformed(input_hi, bits=1, seed=get_seed())
                rank[went_high] |= (result_hi > 0.5).astype(np.int32)[went_high] << bit_pos
                new_nodes.append((result_hi, went_high))
                total_passes += 1

                # Split "low" group: scale to [0.5, 1]
                # Parent 0→0.5 (split), Parent 1→1 (boundary)
                input_lo = parent_result * 0.5 + 0.5
                result_lo = dither_transformed(input_lo, bits=1, seed=get_seed())
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
        dithered, error_map = dither(input_img, bits=args.bits, seed=args.seed, delay=args.delay, return_error=True)

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
