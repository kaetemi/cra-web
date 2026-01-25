#!/usr/bin/env python3
"""
Analyze prime pair kernels for 1D temporal dithering.

Tests all prime pairs [p, q] where p + q = 48.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def lowbias32(x):
    x = np.uint32(x)
    x ^= x >> 16
    x = np.uint32(x * np.uint32(0x21f0aaad))
    x ^= x >> 15
    x = np.uint32(x * np.uint32(0x735a2d97))
    x ^= x >> 15
    return x

def blue_dither_1d_kernel(brightness, count, kernels, seed=12345):
    n_kernels = len(kernels)
    max_len = max(len(k) for k in kernels)
    err = [0] * (max_len + 1)
    output = np.zeros(count, dtype=np.uint8)
    threshold, white_val = 6120, 255 * 48

    for i in range(count):
        pixel = brightness * 48 + err[0]
        if pixel >= threshold:
            output[i] = 1
            quant_err = pixel - white_val
        else:
            output[i] = 0
            quant_err = pixel

        h = lowbias32(np.uint32(i) ^ np.uint32(seed))
        kernel = kernels[h % n_kernels]
        err = err[1:] + [0]
        for j, coeff in enumerate(kernel):
            err[j] += (quant_err * coeff) // 48
    return output

def compute_spectrum(signal):
    signal = signal.astype(np.float64) - np.mean(signal)
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2 / len(signal)
    freqs = np.fft.rfftfreq(len(signal))
    return freqs, 10 * np.log10(power + 1e-20)

def smooth_spectrum_log(freqs, power_db, bins_per_octave=12):
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

def measure_slope(freqs, power_db):
    f_smooth, p_smooth = smooth_spectrum_log(freqs, power_db)
    mask_low = (f_smooth >= 0.01) & (f_smooth < 0.02)
    mask_high = (f_smooth >= 0.1) & (f_smooth < 0.2)
    if np.any(mask_low) and np.any(mask_high):
        p_low, p_high = np.mean(p_smooth[mask_low]), np.mean(p_smooth[mask_high])
        return (p_high - p_low) / np.log10(0.1 / 0.01) / 3.32
    return None

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def main():
    output_dir = Path(__file__).parent.parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # All prime pairs that sum to 48
    kernel_sets = {
        '[48]+[43,5]':  [[48], [43, 5]],   # ratio 8.6:1, both prime
        '[48]+[41,7]':  [[48], [41, 7]],   # ratio 5.9:1, both prime
        '[48]+[37,11]': [[48], [37, 11]],  # ratio 3.4:1, both prime
        '[48]+[31,17]': [[48], [31, 17]],  # ratio 1.8:1, both prime
        '[48]+[29,19]': [[48], [29, 19]],  # ratio 1.5:1, both prime
        # Non-prime comparisons
        '[48]+[38,10]': [[48], [38, 10]],  # ratio 3.8:1 (previous best)
        '[48]+[28,20]': [[48], [28, 20]],  # ratio 1.4:1 (current)
    }

    count = 131072
    gray_levels = [1, 2, 5, 10, 20, 42, 64, 85, 127]

    print("Prime Pair Kernels - All sum to 48")
    print("=" * 105)
    print(f"{'Kernel':<18} {'Ratio':<7} {'Prime':<6}", end="")
    for g in gray_levels:
        print(f" G{g:>3}", end="")
    print("    Avg")
    print("-" * 105)

    results = []
    for name, kernels in kernel_sets.items():
        k = kernels[1]
        ratio = k[0] / k[1]
        both_prime = is_prime(k[0]) and is_prime(k[1])

        slopes = []
        for gray in gray_levels:
            sig = blue_dither_1d_kernel(gray, count, kernels)
            freqs, power = compute_spectrum(sig)
            slope = measure_slope(freqs, power)
            slopes.append(slope if slope else 0)

        avg = np.mean(slopes)
        results.append((name, ratio, both_prime, slopes, avg))

        prime_str = "✓" if both_prime else ""
        print(f"{name:<18} {ratio:<7.1f} {prime_str:<6}", end="")
        for s in slopes:
            print(f" {s:>+4.1f}", end="")
        print(f"  {avg:>+5.2f}")

    print("-" * 105)
    print(f"{'Ideal':<18} {'':<7} {'':<6}", end="")
    for _ in gray_levels:
        print(f" +6.0", end="")
    print(f"  +6.00")

    # Sort by average and show ranking
    print("\n\nRanking by average slope:")
    results.sort(key=lambda x: -x[4])
    for i, (name, ratio, both_prime, slopes, avg) in enumerate(results, 1):
        prime_str = "both prime" if both_prime else ""
        print(f"  {i}. {name:<18} ratio {ratio:>4.1f}:1  avg {avg:>+.2f}  {prime_str}")

    # Generate chart
    fig, ax = plt.subplots(figsize=(10, 6))

    ratios = [r[1] for r in results]
    avgs = [r[4] for r in results]
    names = [r[0] for r in results]
    primes = [r[2] for r in results]

    colors = ['green' if p else 'blue' for p in primes]
    ax.scatter(ratios, avgs, c=colors, s=100, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name.split('+')[1], (ratios[i], avgs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(y=6.0, color='k', linestyle='--', alpha=0.5, label='Ideal (+6 dB/oct)')
    ax.set_xlabel('Kernel Ratio (first/second coefficient)')
    ax.set_ylabel('Average Spectral Slope (dB/octave)')
    ax.set_title('1D Kernel Performance vs Ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'spectrum_1d_prime_pairs.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
