#!/usr/bin/env python3
"""
Analyze 1D temporal dithering kernels.

Compares different kernel pairs for blue noise quality.
All kernels normalized to sum=48 for integer math compatibility.

Usage:
    python analyze_1d_kernels.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Hash function (matches C implementation)
# =============================================================================

def lowbias32(x):
    """lowbias32 hash function."""
    x = np.uint32(x)
    x ^= x >> 16
    x = np.uint32(x * np.uint32(0x21f0aaad))
    x ^= x >> 15
    x = np.uint32(x * np.uint32(0x735a2d97))
    x ^= x >> 15
    return x

# =============================================================================
# 1D Dithering with configurable kernels
# =============================================================================

def blue_dither_1d_kernel(brightness, count, kernels, seed=12345):
    """
    1D blue noise dithering with configurable kernel set.
    All kernels must sum to 48 for integer math.
    """
    n_kernels = len(kernels)
    max_len = max(len(k) for k in kernels)

    err = [0] * (max_len + 1)
    output = np.zeros(count, dtype=np.uint8)

    threshold = 6120   # 127.5 * 48
    white_val = 255 * 48

    for i in range(count):
        pixel = brightness * 48 + err[0]

        if pixel >= threshold:
            output[i] = 1
            quant_err = pixel - white_val
        else:
            output[i] = 0
            quant_err = pixel

        h = lowbias32(np.uint32(i) ^ np.uint32(seed))
        kernel_idx = h % n_kernels
        kernel = kernels[kernel_idx]

        # Shift error buffer
        err = err[1:] + [0]

        # Apply kernel (all coefficients are /48)
        for j, coeff in enumerate(kernel):
            err[j] += (quant_err * coeff) // 48

    return output

# =============================================================================
# Spectrum Analysis
# =============================================================================

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

# =============================================================================
# Kernel Definitions
# =============================================================================

# All kernels sum to 48 for integer math compatibility
# Notation: [coeff_t+1, coeff_t+2, ...] where coefficients are /48

KERNEL_SETS = {
    # Current best
    '[48]+[38,10] BEST': [
        [48],           # 100% to t+1 (FS-like)
        [38, 10],       # ratio 3.8:1 = 2×[19,5] (both prime)
    ],

    # User requested: [1]+[3,1] scaled to sum=48
    '[48]+[36,12] (3:1)': [
        [48],           # [1] scaled
        [36, 12],       # [3,1] scaled, ratio 3:1
    ],

    # Original implementation
    '[48]+[28,20]': [
        [48],           # 100% to t+1 (FS-like)
        [28, 20],       # 7:5 ratio to t+1,t+2 (JJN-like)
    ],

    # Prime pairs (both coefficients are prime, sum to 48)
    '[48]+[43,5] prime': [
        [48],
        [43, 5],        # ratio 8.6:1, both prime - BEST prime pair
    ],

    '[48]+[41,7] prime': [
        [48],
        [41, 7],        # ratio 5.9:1, both prime
    ],

    '[48]+[37,11] prime': [
        [48],
        [37, 11],       # ratio 3.4:1, both prime
    ],

    '[48]+[31,17] prime': [
        [48],
        [31, 17],       # ratio 1.8:1, both prime
    ],

    # Length-3 experiments (both hurt low gray)
    '[48]+[34,10,4]': [
        [48],
        [34, 10, 4],    # Length-3
    ],

    '[48]+[28,12,8]': [
        [48],
        [28, 12, 8],    # Length-3, more balanced - even worse
    ],
}

# =============================================================================
# Main Analysis
# =============================================================================

def main():
    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 131072
    gray_levels = [1, 2, 5, 10, 20, 42, 64, 85, 127]

    print("1D Kernel Comparison - Spectral Slope (dB/octave)")
    print("=" * 85)
    print(f"{'Kernel Set':<28}", end="")
    for g in gray_levels:
        print(f" G{g:>3}", end="")
    print("    Avg")
    print("-" * 85)

    results = {}
    for name, kernels in KERNEL_SETS.items():
        slopes = []
        for gray in gray_levels:
            sig = blue_dither_1d_kernel(gray, count, kernels)
            freqs, power = compute_spectrum(sig)
            slope = measure_slope(freqs, power)
            slopes.append(slope if slope else 0)

        avg_slope = np.mean(slopes)
        results[name] = (slopes, avg_slope)

        print(f"{name:<28}", end="")
        for s in slopes:
            print(f" {s:>+4.1f}", end="")
        print(f"  {avg_slope:>+5.2f}")

    print("-" * 85)
    print(f"{'Ideal blue noise':<28}", end="")
    for _ in gray_levels:
        print(f" {6.0:>+4.1f}", end="")
    print(f"  {6.0:>+5.2f}")

    # Show kernel details
    print("\n\nKernel details (all normalized to /48):")
    print("-" * 60)
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    for name, kernels in KERNEL_SETS.items():
        print(f"\n{name}:")
        for i, k in enumerate(kernels):
            prime_count = sum(1 for c in k if c in primes)
            prime_pct = prime_count / len(k) * 100
            print(f"  K{i+1}: {k} sum={sum(k)} ({prime_pct:.0f}% prime)")

    # Generate comparison chart
    n_cols = 3
    n_rows = (len(gray_levels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle('1D Kernel Comparison - Log Frequency Spectrum', fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(KERNEL_SETS)))

    for idx, gray in enumerate(gray_levels):
        ax = axes[idx // n_cols, idx % n_cols]

        for kidx, (name, kernels) in enumerate(KERNEL_SETS.items()):
            sig = blue_dither_1d_kernel(gray, count, kernels)
            freqs, power = compute_spectrum(sig)
            f_smooth, p_smooth = smooth_spectrum_log(freqs, power)
            ax.semilogx(f_smooth, p_smooth, color=colors[kidx], linewidth=1.5,
                        label=name if idx == 0 else None, alpha=0.8)

        # Ideal reference
        f_ref = np.logspace(-3, np.log10(0.5), 100)
        ideal_blue = -20 + 20 * np.log10(f_ref / 0.1)
        ax.semilogx(f_ref, ideal_blue, 'k--', linewidth=2, alpha=0.5,
                    label='Ideal' if idx == 0 else None)

        ax.set_xlabel('Frequency (log)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(f'Gray {gray} ({gray*100/255:.1f}%)')
        ax.set_xlim(1e-3, 0.5)
        ax.set_ylim(-55, 5)
        ax.grid(True, alpha=0.3, which='both')

    # Hide empty subplots
    for idx in range(len(gray_levels), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    axes[0, 0].legend(loc='lower right', fontsize=6)
    plt.tight_layout()

    output_path = output_dir / 'spectrum_1d_kernel_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
