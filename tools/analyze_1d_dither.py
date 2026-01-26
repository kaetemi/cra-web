#!/usr/bin/env python3
"""
Analyze 1D temporal dithering spectrum.

Compares our 1D blue noise dithering against:
- Ideal blue noise (power ∝ f², +6 dB/octave)
- White noise (flat spectrum)

Usage:
    python analyze_1d_dither.py              # Run all analyses
    python analyze_1d_dither.py --log        # Log-scale frequency axis
    python analyze_1d_dither.py --low-gray   # Focus on low gray levels
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# =============================================================================
# 1D Dithering Implementations (replicate C code for analysis)
# =============================================================================

def lowbias32(x):
    """lowbias32 hash function - matches C implementation."""
    x = np.uint32(x)
    x ^= x >> 16
    x = np.uint32(x * np.uint32(0x21f0aaad))
    x ^= x >> 15
    x = np.uint32(x * np.uint32(0x735a2d97))
    x ^= x >> 15
    return x

def blue_dither_1d(brightness, count, seed=12345, kernel='38_10'):
    """
    1D blue noise dithering - replicates C implementation.
    Mixes [48] with configurable second kernel.

    kernel options:
        '38_10' - ratio ~4:1 (current best)
        '36_12' - ratio 3:1 ([3,1] scaled)
        '28_20' - ratio 7:5 (original JJN-like)
    """
    err0, err1 = 0, 0
    output = np.zeros(count, dtype=np.uint8)
    threshold = 6120   # 127.5 * 48
    white_val = 255 * 48

    # Kernel coefficients
    if kernel == '38_10':
        k1_coeff, k2_coeff = 38, 10
    elif kernel == '36_12':
        k1_coeff, k2_coeff = 36, 12
    elif kernel == '28_20':
        k1_coeff, k2_coeff = 28, 20
    else:
        k1_coeff, k2_coeff = 38, 10

    for i in range(count):
        pixel = brightness * 48 + err0

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
            # FS-like: 100% to next
            err0 += quant_err
        else:
            # Second kernel
            err0 += (quant_err * k1_coeff) // 48
            err1 += (quant_err * k2_coeff) // 48

    return output

def pwm_dither(brightness, count):
    """Traditional PWM - regular on/off pattern at fixed frequency."""
    output = np.zeros(count, dtype=np.uint8)
    if brightness == 0:
        return output
    if brightness >= 255:
        return np.ones(count, dtype=np.uint8)

    # PWM with period 256 (8-bit resolution)
    period = 256
    on_time = brightness

    for i in range(count):
        phase = i % period
        output[i] = 1 if phase < on_time else 0

    return output

def white_noise_dither(brightness, count, seed=42):
    """White noise dithering (random threshold)."""
    rng = np.random.default_rng(seed)
    threshold = brightness / 255.0
    return (rng.random(count) < threshold).astype(np.uint8)


def sigma_delta_1st(brightness, count):
    """First-order sigma-delta modulation. Should produce +6 dB/octave (violet)."""
    vin = (brightness / 255.0) * 2 - 1  # scale to -1..+1 range
    vref = 1.0
    output = np.zeros(count, dtype=np.uint8)
    integrator = 0.0

    for i in range(count):
        # sum input with feedback
        if i == 0:
            feedback = 0
        else:
            feedback = vref if output[i-1] == 1 else -vref

        sum_val = vin - feedback

        # integrate
        integrator = integrator + sum_val

        # compare
        output[i] = 1 if integrator >= 0 else 0

    return output


def sigma_delta_2nd(brightness, count):
    """Second-order sigma-delta modulation. Should produce +12 dB/octave."""
    threshold = brightness / 255.0
    output = np.zeros(count, dtype=np.uint8)
    e1 = 0.0
    e2 = 0.0

    for i in range(count):
        value = threshold + 2*e1 - e2
        output[i] = 1 if value >= 0.5 else 0
        e2 = e1
        e1 = value - output[i]

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

def smooth_spectrum(freqs, power_db, window=50):
    """Smooth spectrum with uniform window."""
    from scipy.ndimage import uniform_filter1d
    return freqs, uniform_filter1d(power_db, window)

def smooth_spectrum_log(freqs, power_db, bins_per_octave=12):
    """Smooth spectrum with log-spaced bins."""
    f_min = freqs[freqs > 0].min()
    f_max = freqs.max()
    n_bins = int(np.log2(f_max / f_min) * bins_per_octave)
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)

    bin_centers = []
    bin_means = []
    for i in range(len(bin_edges) - 1):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i+1])
        if np.any(mask):
            bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i+1]))
            bin_means.append(np.mean(power_db[mask]))

    return np.array(bin_centers), np.array(bin_means)

def measure_slope(freqs, power_db, f_low=0.01, f_high=0.1):
    """Measure spectral slope in dB/octave between two frequencies."""
    f_smooth, p_smooth = smooth_spectrum_log(freqs, power_db)

    mask_low = (f_smooth >= f_low) & (f_smooth < f_low * 2)
    mask_high = (f_smooth >= f_high) & (f_smooth < f_high * 2)

    if np.any(mask_low) and np.any(mask_high):
        p_low = np.mean(p_smooth[mask_low])
        p_high = np.mean(p_smooth[mask_high])
        # dB per decade, convert to dB per octave
        decades = np.log10(f_high / f_low)
        slope_decade = (p_high - p_low) / decades
        slope_octave = slope_decade / 3.32
        return slope_octave
    return None

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_linear(gray_levels, count, output_dir):
    """Linear frequency axis analysis."""
    print(f"Analyzing {len(gray_levels)} gray levels (linear scale)...")

    for gray in gray_levels:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'1D Temporal Dithering Spectrum - Gray {gray} ({gray*100/255:.1f}%)', fontsize=14)

        signals = {
            'Our 1D Method': blue_dither_1d(gray, count),
            'ΣΔ 1st Order (+6dB/oct)': sigma_delta_1st(gray, count),
            'ΣΔ 2nd Order (+12dB/oct)': sigma_delta_2nd(gray, count),
            'PWM': pwm_dither(gray, count),
            'White Noise': white_noise_dither(gray, count),
        }

        colors = ['#2ecc71', '#9b59b6', '#8e44ad', '#3498db', '#e74c3c']

        for idx, (name, sig) in enumerate(signals.items()):
            if idx >= 5:
                break
            ax = axes[idx // 3, idx % 3]
            freqs, power_db = compute_spectrum(sig)
            f_log, p_log = smooth_spectrum_log(freqs, power_db)

            duty = np.mean(sig) * 100

            if name == 'PWM':
                # Show raw spectrum for PWM to reveal harmonic spikes
                ax.semilogx(freqs[1:], power_db[1:], color=colors[idx], linewidth=0.5,
                            label=f'{name} ({duty:.1f}%)', alpha=0.8)
            else:
                # Show raw spectrum as light envelope, smoothed as solid line
                ax.semilogx(freqs[1:], power_db[1:], color=colors[idx], linewidth=0.3, alpha=0.3)
                ax.semilogx(f_log, p_log, color=colors[idx], linewidth=2,
                            label=f'{name} ({duty:.1f}%)')

            # Reference lines
            f_ref = np.logspace(-3, np.log10(0.5), 100)
            anchor_idx = np.argmin(np.abs(f_log - 0.1)) if len(f_log) > 0 else 0
            anchor_db = p_log[anchor_idx] if len(p_log) > anchor_idx else -20
            ideal_6db = anchor_db + 20 * np.log10(f_ref / 0.1)
            ax.semilogx(f_ref, ideal_6db, 'k--', alpha=0.5, linewidth=1, label='Violet (+6dB/oct)')
            # Only show +12dB/oct line on 2nd order sigma-delta panel
            if '2nd Order' in name:
                ideal_12db = anchor_db + 40 * np.log10(f_ref / 0.1)
                ax.semilogx(f_ref, ideal_12db, 'k:', alpha=0.5, linewidth=1, label='+12dB/oct')

            ax.set_xlabel('Frequency (log scale)')
            ax.set_ylabel('Power (dB)')
            ax.set_title(name)
            ax.set_xlim(1e-3, 0.5)
            ax.set_ylim(-60, 50)
            ax.legend(loc='lower right', fontsize=7)
            ax.grid(True, alpha=0.3, which='both')

        # Summary in sixth panel
        ax = axes[1, 2]
        for idx, (name, sig) in enumerate(signals.items()):
            freqs, power_db = compute_spectrum(sig)
            if name == 'PWM':
                ax.semilogx(freqs[1:], power_db[1:], color=colors[idx], linewidth=0.5, label=name, alpha=0.8)
            else:
                f_log, p_log = smooth_spectrum_log(freqs, power_db)
                ax.semilogx(f_log, p_log, color=colors[idx], linewidth=2, label=name, alpha=0.8)

        # Reference line (only +6dB/oct for comparison panel)
        f_ref = np.logspace(-3, np.log10(0.5), 100)
        ideal_6db = -30 + 20 * np.log10(f_ref / 0.01)
        ax.semilogx(f_ref, ideal_6db, 'k--', alpha=0.5, linewidth=1, label='Violet (+6dB/oct)')

        ax.set_xlabel('Frequency (log scale)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Comparison')
        ax.set_xlim(1e-3, 0.5)
        ax.set_ylim(-60, 50)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        output_path = output_dir / f'spectrum_1d_gray_{gray:03d}.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  {output_path.name}")


def analyze_log_scale(gray_levels, count, output_dir):
    """Log frequency axis analysis - ideal for seeing blue noise slope."""
    print(f"Analyzing {len(gray_levels)} gray levels (log scale)...")

    n_cols = 3
    n_rows = (len(gray_levels) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('1D Blue Noise Dithering - Log Frequency Spectrum', fontsize=14)

    for idx, gray in enumerate(gray_levels):
        ax = axes[idx // n_cols, idx % n_cols]

        # Generate signals
        our_sig = blue_dither_1d(gray, count)
        white_sig = white_noise_dither(gray, count)

        duty_our = np.mean(our_sig) * 100
        duty_white = np.mean(white_sig) * 100

        # Compute spectra
        freqs, power_our = compute_spectrum(our_sig)
        _, power_white = compute_spectrum(white_sig)

        # Smooth with log bins
        f_our, p_our = smooth_spectrum_log(freqs, power_our)
        f_white, p_white = smooth_spectrum_log(freqs, power_white)

        # Plot
        ax.semilogx(f_our, p_our, 'g-', linewidth=2, label=f'Our 1D ({duty_our:.1f}%)')
        ax.semilogx(f_white, p_white, 'r-', linewidth=1.5, alpha=0.7, label=f'White ({duty_white:.1f}%)')

        # Ideal blue noise reference: +6 dB/octave
        f_ref = np.logspace(-3, np.log10(0.5), 100)
        anchor_idx = np.argmin(np.abs(f_our - 0.1))
        anchor_db = p_our[anchor_idx] if len(p_our) > anchor_idx else -20
        ideal_blue = anchor_db + 20 * np.log10(f_ref / 0.1)
        ax.semilogx(f_ref, ideal_blue, 'k--', linewidth=1, alpha=0.5, label='Ideal (+6dB/oct)')

        ax.set_xlabel('Frequency (log scale)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(f'Gray {gray} ({gray*100/255:.1f}%)')
        ax.set_xlim(1e-3, 0.5)
        ax.set_ylim(-60, 10)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    # Hide empty subplots
    for idx in range(len(gray_levels), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'spectrum_1d_logscale.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  {output_path.name}")

    # Print numerical analysis
    print("\n  Spectral slope analysis:")
    print(f"  {'Gray':>5} {'Duty':>8} {'Slope':>12} {'Quality':>10}")
    print(f"  {'-'*40}")

    for gray in gray_levels:
        sig = blue_dither_1d(gray, count)
        duty = np.mean(sig) * 100
        freqs, power = compute_spectrum(sig)
        slope = measure_slope(freqs, power)

        if slope is not None:
            if slope > 5:
                quality = "Excellent"
            elif slope > 4:
                quality = "Good"
            elif slope > 2:
                quality = "OK"
            else:
                quality = "Poor"
            print(f"  {gray:>5} {duty:>7.2f}% {slope:>+10.1f} dB/oct {quality:>8}")
        else:
            print(f"  {gray:>5} {duty:>7.2f}% {'N/A':>12} {'N/A':>10}")

    print(f"\n  Ideal blue noise: +6.0 dB/octave")


def analyze_comparison(gray_levels, count, output_dir):
    """Single comparison plot across gray levels."""
    print("Generating comparison plot...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gray_levels)))

    for idx, gray in enumerate(gray_levels):
        sig = blue_dither_1d(gray, count)
        freqs, power_db = compute_spectrum(sig)
        f_smooth, p_smooth = smooth_spectrum_log(freqs, power_db)

        ax.semilogx(f_smooth, p_smooth, color=colors[idx], linewidth=2,
                    label=f'Gray {gray} ({gray*100/255:.0f}%)')

    # Ideal reference
    f_ref = np.logspace(-3, np.log10(0.5), 100)
    ideal_blue = -30 + 20 * np.log10(f_ref / 0.01)
    ax.semilogx(f_ref, ideal_blue, 'k--', alpha=0.7, linewidth=2, label='Ideal (+6dB/oct)')

    ax.set_xlabel('Frequency (cycles/sample)', fontsize=12)
    ax.set_ylabel('Power (dB)', fontsize=12)
    ax.set_title('1D Blue Noise Dithering - Spectrum vs Gray Level', fontsize=14)
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(-60, 10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = output_dir / 'spectrum_1d_comparison.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  {output_path.name}")


def blue_dither_1d_custom(brightness, count, k1, k2, kernel_sum=48, seed=12345):
    """
    1D blue noise dithering with custom kernel coefficients.
    Mixes [kernel_sum] with [k1, k2] where k1 + k2 = kernel_sum.
    """
    err0, err1 = 0, 0
    output = np.zeros(count, dtype=np.uint8)
    threshold = int(127.5 * kernel_sum)  # 127.5 * kernel_sum
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
            # FS-like: 100% to next
            err0 += quant_err
        else:
            # Custom kernel
            err0 += (quant_err * k1) // kernel_sum
            err1 += (quant_err * k2) // kernel_sum

    return output


def analyze_all_kernels(count, output_dir, kernel_sum=48):
    """Test all possible [k1, k2] kernels where k1 + k2 = kernel_sum, output top 8."""
    print(f"Testing all [k1, k2] kernels (k1 + k2 = {kernel_sum})...")

    # Test gray levels - focus on challenging ones
    gray_levels = [1, 2, 5, 10, 20, 42, 64, 85, 127, 170, 212, 254]

    results = []
    n_unique = kernel_sum // 2  # Number of unique kernels (k1 > k2)

    # Test all kernels from [kernel_sum-1,1] to [kernel_sum/2+1, kernel_sum/2-1] (k1 > k2 to avoid duplicates)
    for k1 in range(kernel_sum - 1, kernel_sum // 2, -1):
        k2 = kernel_sum - k1
        slopes = []

        for gray in gray_levels:
            sig = blue_dither_1d_custom(gray, count, k1, k2, kernel_sum)
            freqs, power = compute_spectrum(sig)
            slope = measure_slope(freqs, power)
            slopes.append(slope if slope else 0)

        avg_slope = np.mean(slopes)
        min_slope = np.min(slopes)
        results.append({
            'k1': k1,
            'k2': k2,
            'ratio': k1 / k2,
            'slopes': slopes,
            'avg': avg_slope,
            'min': min_slope,
        })

        # Progress indicator
        if k1 % 5 == 0:
            print(f"  Tested [{kernel_sum-1},1] to [{k1},{k2}]...")

    # Sort by average slope (best first)
    results.sort(key=lambda x: x['avg'], reverse=True)

    # Print full ranking
    print(f"\n  Full ranking (all {len(results)} unique kernels for sum={kernel_sum}):")
    print(f"  {'Rank':<5} {'Kernel':<12} {'Ratio':<8} {'Avg':>8} {'Min':>8}")
    print(f"  {'-'*45}")
    for i, r in enumerate(results):
        print(f"  {i+1:<5} [{r['k1']},{r['k2']}]{'':<4} {r['ratio']:>6.2f}:1 {r['avg']:>+7.2f} {r['min']:>+7.2f}")

    # Top 8
    top8 = results[:8]

    print(f"\n  Top 8 kernels (sum={kernel_sum}):")
    print(f"  {'Kernel':<12}", end="")
    for g in gray_levels:
        print(f" G{g:>3}", end="")
    print("    Avg    Min")
    print(f"  {'-'*90}")

    for r in top8:
        print(f"  [{r['k1']},{r['k2']}]{'':<5}", end="")
        for s in r['slopes']:
            print(f" {s:>+4.1f}", end="")
        print(f"  {r['avg']:>+5.2f}  {r['min']:>+5.2f}")

    print(f"  {'-'*90}")
    print(f"  {'Ideal':<12}", end="")
    for _ in gray_levels:
        print(f" {6.0:>+4.1f}", end="")
    print(f"  {6.0:>+5.2f}  {6.0:>+5.2f}")

    # Generate comparison chart for top 8
    n_cols = 4
    n_rows = (len(gray_levels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Top 8 1D Kernels (sum={kernel_sum}, {len(results)} tested)', fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for idx, gray in enumerate(gray_levels):
        ax = axes[idx // n_cols, idx % n_cols]

        for kidx, r in enumerate(top8):
            sig = blue_dither_1d_custom(gray, count, r['k1'], r['k2'], kernel_sum)
            freqs, power = compute_spectrum(sig)
            f_smooth, p_smooth = smooth_spectrum_log(freqs, power)
            label = f"[{r['k1']},{r['k2']}] ({r['ratio']:.1f}:1)" if idx == 0 else None
            ax.semilogx(f_smooth, p_smooth, color=colors[kidx], linewidth=1.5,
                        label=label, alpha=0.8)

        # Ideal reference
        f_ref = np.logspace(-3, np.log10(0.5), 100)
        ideal_blue = -20 + 20 * np.log10(f_ref / 0.1)
        ax.semilogx(f_ref, ideal_blue, 'k--', linewidth=1.5, alpha=0.5,
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

    axes[0, 0].legend(loc='lower right', fontsize=7)
    plt.tight_layout()

    output_path = output_dir / f'spectrum_1d_top8_kernels_sum{kernel_sum}.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n  Saved: {output_path}")
    plt.close()

    return results


def analyze_all_48_kernels(count, output_dir):
    """Test all 48 possible [k1, k2] kernels where k1 + k2 = 48, output top 8."""
    return analyze_all_kernels(count, output_dir, kernel_sum=48)


def analyze_kernel_comparison(gray_levels, count, output_dir):
    """Compare different 1D kernels across all gray levels."""
    print("Comparing kernels across all gray levels...")

    kernels = {
        '[38,10] (3.8:1)': '38_10',
        '[36,12] (3:1)': '36_12',
        '[28,20] (1.4:1)': '28_20',
    }
    colors = {'38_10': '#2ecc71', '36_12': '#3498db', '28_20': '#e74c3c'}

    # Print table header
    print(f"\n  {'Kernel':<20}", end="")
    for g in gray_levels:
        print(f" G{g:>3}", end="")
    print("    Avg")
    print(f"  {'-'*70}")

    results = {}
    for name, kernel_id in kernels.items():
        slopes = []
        for gray in gray_levels:
            sig = blue_dither_1d(gray, count, kernel=kernel_id)
            freqs, power = compute_spectrum(sig)
            slope = measure_slope(freqs, power)
            slopes.append(slope if slope else 0)

        avg_slope = np.mean(slopes)
        results[name] = (slopes, avg_slope, kernel_id)

        print(f"  {name:<20}", end="")
        for s in slopes:
            print(f" {s:>+4.1f}", end="")
        print(f"  {avg_slope:>+5.2f}")

    print(f"  {'-'*70}")
    print(f"  {'Ideal blue noise':<20}", end="")
    for _ in gray_levels:
        print(f" {6.0:>+4.1f}", end="")
    print(f"  {6.0:>+5.2f}")

    # Generate comparison chart
    n_cols = min(4, len(gray_levels))
    n_rows = (len(gray_levels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('1D Kernel Comparison: [38,10] vs [36,12] vs [28,20]', fontsize=14)

    for idx, gray in enumerate(gray_levels):
        ax = axes[idx // n_cols, idx % n_cols]

        for name, (slopes, avg, kernel_id) in results.items():
            sig = blue_dither_1d(gray, count, kernel=kernel_id)
            freqs, power = compute_spectrum(sig)
            f_smooth, p_smooth = smooth_spectrum_log(freqs, power)
            ax.semilogx(f_smooth, p_smooth, color=colors[kernel_id], linewidth=2,
                        label=f'{name}' if idx == 0 else None, alpha=0.8)

        # Ideal reference
        f_ref = np.logspace(-3, np.log10(0.5), 100)
        ideal_blue = -20 + 20 * np.log10(f_ref / 0.1)
        ax.semilogx(f_ref, ideal_blue, 'k--', linewidth=1.5, alpha=0.5,
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

    axes[0, 0].legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    output_path = output_dir / 'spectrum_1d_kernel_full_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze 1D temporal dithering spectrum')
    parser.add_argument('--log', action='store_true', help='Use log frequency scale')
    parser.add_argument('--low-gray', action='store_true', help='Focus on low gray levels')
    parser.add_argument('--all', action='store_true', help='Run all analyses (default)')
    parser.add_argument('--kernel-compare', action='store_true', help='Compare kernels [38,10] vs [36,12] vs [28,20]')
    parser.add_argument('--find-best', action='store_true', help='Test all 24 kernels [47,1] to [25,23], show top 8')
    parser.add_argument('--kernel-sum', type=int, nargs='+', default=[48],
                        help='Kernel sum(s) to test with --find-best (default: 48)')
    parser.add_argument('--count', type=int, default=131072, help='Sample count (default: 131072)')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default gray levels
    if args.low_gray:
        gray_levels = [1, 2, 5, 10, 20, 42, 64, 85, 127]
    else:
        gray_levels = [42, 64, 85, 127, 170, 191, 213]

    count = args.count

    # Find best kernel mode
    if args.find_best:
        all_results = {}
        for kernel_sum in args.kernel_sum:
            print(f"\n{'='*80}")
            results = analyze_all_kernels(count, output_dir, kernel_sum=kernel_sum)
            all_results[kernel_sum] = results

        # If multiple sums tested, print cross-sum comparison
        if len(args.kernel_sum) > 1:
            print(f"\n{'='*80}")
            print("CROSS-SUM COMPARISON - Best kernel from each sum:")
            print(f"{'='*80}")
            print(f"  {'Sum':<6} {'Best Kernel':<14} {'Ratio':<10} {'Avg':>8} {'Min':>8}")
            print(f"  {'-'*50}")
            for kernel_sum in args.kernel_sum:
                best = all_results[kernel_sum][0]
                print(f"  {kernel_sum:<6} [{best['k1']},{best['k2']}]{'':<6} {best['ratio']:>6.2f}:1 {best['avg']:>+7.2f} {best['min']:>+7.2f}")

        print(f"\nDone! Results in {output_dir}")
        return

    # Kernel comparison mode
    if args.kernel_compare:
        # Use full range of gray levels for kernel comparison
        all_grays = [1, 2, 5, 10, 18, 20, 42, 43, 54, 64, 85, 91, 127, 128, 170, 191, 212, 213, 254]
        analyze_kernel_comparison(all_grays, count, output_dir)
        print(f"\nDone! Results in {output_dir}")
        return

    # Always run log-scale analysis (it's the most useful)
    analyze_log_scale(gray_levels, count, output_dir)

    # Linear analysis for detailed per-gray-level view
    if args.all or not args.log:
        analyze_linear(gray_levels, count, output_dir)

    analyze_comparison(gray_levels, count, output_dir)

    print(f"\nDone! Results in {output_dir}")


if __name__ == '__main__':
    main()
