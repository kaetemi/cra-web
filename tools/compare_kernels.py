#!/usr/bin/env python3
"""
Compare two 1D dithering kernels across the full gray range.

Usage:
    python compare_kernels.py                    # Compare [38,10] vs [46,2]
    python compare_kernels.py --k1 38 10 --k2 46 2  # Custom kernels
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def lowbias32(x):
    """lowbias32 hash function."""
    x = np.uint32(x)
    x ^= x >> 16
    x = np.uint32(x * np.uint32(0x21f0aaad))
    x ^= x >> 15
    x = np.uint32(x * np.uint32(0x735a2d97))
    x ^= x >> 15
    return x


def blue_dither_1d_custom(brightness, count, k1, k2, kernel_sum=48, seed=12345):
    """1D blue noise dithering with custom kernel coefficients."""
    err0, err1 = 0, 0
    output = np.zeros(count, dtype=np.uint8)
    threshold = int(127.5 * kernel_sum)
    white_val = 255 * kernel_sum

    for i in range(count):
        pixel = brightness * kernel_sum + err0
        if pixel >= threshold:
            output[i] = 1
            quant_err = pixel - white_val
        else:
            output[i] = 0
            quant_err = pixel

        h = lowbias32(np.uint32(i) ^ np.uint32(seed))
        err0 = err1
        err1 = 0

        if h & 1:
            err0 += quant_err
        else:
            err0 += (quant_err * k1) // kernel_sum
            err1 += (quant_err * k2) // kernel_sum

    return output


def compute_spectrum(signal):
    """Compute power spectrum in dB."""
    signal = signal.astype(np.float64) - np.mean(signal)
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2 / len(signal)
    power_db = 10 * np.log10(power + 1e-20)
    freqs = np.fft.rfftfreq(len(signal))
    return freqs, power_db


def smooth_spectrum_log(freqs, power_db, bins_per_octave=12):
    """Smooth spectrum with log-spaced bins."""
    f_min = freqs[freqs > 0].min()
    f_max = freqs.max()
    n_bins = int(np.log2(f_max / f_min) * bins_per_octave)
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    bin_centers, bin_means = [], []
    for i in range(len(bin_edges) - 1):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i+1])
        if np.any(mask):
            bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i+1]))
            bin_means.append(np.mean(power_db[mask]))
    return np.array(bin_centers), np.array(bin_means)


def measure_slope(freqs, power_db, f_low=0.01, f_high=0.1):
    """Measure spectral slope in dB/octave."""
    f_smooth, p_smooth = smooth_spectrum_log(freqs, power_db)
    mask_low = (f_smooth >= f_low) & (f_smooth < f_low * 2)
    mask_high = (f_smooth >= f_high) & (f_smooth < f_high * 2)
    if np.any(mask_low) and np.any(mask_high):
        p_low = np.mean(p_smooth[mask_low])
        p_high = np.mean(p_smooth[mask_high])
        decades = np.log10(f_high / f_low)
        return (p_high - p_low) / decades / 3.32
    return None


def compare_kernels(k1_pair, k2_pair, count=131072, output_dir=None):
    """Compare two kernels across all gray levels."""
    if output_dir is None:
        output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    gray_levels = list(range(1, 255))  # 1 to 254

    k1_name = f'[{k1_pair[0]},{k1_pair[1]}] ({k1_pair[0]/k1_pair[1]:.1f}:1)'
    k2_name = f'[{k2_pair[0]},{k2_pair[1]}] ({k2_pair[0]/k2_pair[1]:.1f}:1)'

    kernels = {
        k1_name: k1_pair,
        k2_name: k2_pair,
    }

    print(f"Comparing {k1_name} vs {k2_name} across all gray levels (1-254)...")
    print()

    results = {}
    for name, (k1, k2) in kernels.items():
        slopes = []
        for gray in gray_levels:
            sig = blue_dither_1d_custom(gray, count, k1, k2)
            freqs, power = compute_spectrum(sig)
            slope = measure_slope(freqs, power)
            slopes.append(slope if slope else 0)
        results[name] = slopes
        print(f"{name}: avg={np.mean(slopes):+.2f}, min={np.min(slopes):+.2f}, max={np.max(slopes):+.2f}")

    # Calculate difference
    slopes_1 = np.array(results[k1_name])
    slopes_2 = np.array(results[k2_name])
    diff = slopes_2 - slopes_1

    print()
    print(f"Difference ({k2_name} - {k1_name}):")
    print(f"  Average: {np.mean(diff):+.3f} dB/octave")
    print(f"  {k2_name} better at {np.sum(diff > 0)} gray levels")
    print(f"  {k1_name} better at {np.sum(diff < 0)} gray levels")
    print(f"  Equal at {np.sum(diff == 0)} gray levels")

    # Find worst cases
    worst_1_idx = np.argmin(slopes_1)
    worst_2_idx = np.argmin(slopes_2)
    print()
    print(f"Worst case {k1_name}: gray {gray_levels[worst_1_idx]} = {slopes_1[worst_1_idx]:+.2f}")
    print(f"Worst case {k2_name}: gray {gray_levels[worst_2_idx]} = {slopes_2[worst_2_idx]:+.2f}")

    # Generate comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{k1_name} vs {k2_name} Kernel Comparison (Full Gray Range)', fontsize=14)

    # Plot 1: Slopes vs gray level
    ax = axes[0, 0]
    ax.plot(gray_levels, slopes_1, 'b-', linewidth=1.5, label=k1_name, alpha=0.8)
    ax.plot(gray_levels, slopes_2, 'r-', linewidth=1.5, label=k2_name, alpha=0.8)
    ax.axhline(y=6.0, color='k', linestyle='--', alpha=0.5, label='Ideal (+6.0)')
    ax.set_xlabel('Gray Level')
    ax.set_ylabel('Spectral Slope (dB/octave)')
    ax.set_title('Spectral Slope vs Gray Level')
    ax.set_xlim(0, 255)
    ax.set_ylim(3, 7)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Difference
    ax = axes[0, 1]
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax.bar(gray_levels, diff, color=colors, width=1.0, alpha=0.7)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xlabel('Gray Level')
    ax.set_ylabel('Difference (dB/octave)')
    ax.set_title(f'{k2_name} - {k1_name} (green = {k2_name} better)')
    ax.set_xlim(0, 255)
    ax.grid(True, alpha=0.3)

    # Plot 3: Spectrum at worst case for kernel 1
    ax = axes[1, 0]
    gray = gray_levels[worst_1_idx]
    for idx, (name, (k1, k2)) in enumerate(kernels.items()):
        sig = blue_dither_1d_custom(gray, count, k1, k2)
        freqs, power = compute_spectrum(sig)
        f_smooth, p_smooth = smooth_spectrum_log(freqs, power)
        color = 'blue' if idx == 0 else 'red'
        ax.semilogx(f_smooth, p_smooth, color=color, linewidth=2, label=name, alpha=0.8)
    f_ref = np.logspace(-3, np.log10(0.5), 100)
    ideal = -20 + 20 * np.log10(f_ref / 0.1)
    ax.semilogx(f_ref, ideal, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax.set_xlabel('Frequency (log)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'Spectrum at Gray {gray} (worst for {k1_name})')
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(-55, 5)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Spectrum at worst case for kernel 2
    ax = axes[1, 1]
    gray = gray_levels[worst_2_idx]
    for idx, (name, (k1, k2)) in enumerate(kernels.items()):
        sig = blue_dither_1d_custom(gray, count, k1, k2)
        freqs, power = compute_spectrum(sig)
        f_smooth, p_smooth = smooth_spectrum_log(freqs, power)
        color = 'blue' if idx == 0 else 'red'
        ax.semilogx(f_smooth, p_smooth, color=color, linewidth=2, label=name, alpha=0.8)
    ax.semilogx(f_ref, ideal, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax.set_xlabel('Frequency (log)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'Spectrum at Gray {gray} (worst for {k2_name})')
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(-55, 5)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate filename from kernel values
    filename = f'spectrum_1d_{k1_pair[0]}_{k1_pair[1]}_vs_{k2_pair[0]}_{k2_pair[1]}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare two 1D dithering kernels')
    parser.add_argument('--k1', type=int, nargs=2, default=[38, 10],
                        help='First kernel [k1, k2] (default: 38 10)')
    parser.add_argument('--k2', type=int, nargs=2, default=[46, 2],
                        help='Second kernel [k1, k2] (default: 46 2)')
    parser.add_argument('--count', type=int, default=131072,
                        help='Sample count (default: 131072)')
    args = parser.parse_args()

    compare_kernels(tuple(args.k1), tuple(args.k2), args.count)


if __name__ == '__main__':
    main()
