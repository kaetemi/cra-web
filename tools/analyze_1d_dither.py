#!/usr/bin/env python3
"""
Analyze 1D temporal dithering spectrum.

Compares our 1D blue noise dithering against:
- Ideal blue noise (power ∝ f²)
- White noise (flat spectrum)
- Naive PWM (periodic)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Replicate the C implementation in Python for analysis
def lowbias32(x):
    """lowbias32 hash function"""
    x = np.uint32(x)
    x ^= x >> 16
    x = np.uint32(x * np.uint32(0x21f0aaad))
    x ^= x >> 15
    x = np.uint32(x * np.uint32(0x735a2d97))
    x ^= x >> 15
    return x

def blue_dither_1d(brightness, count, seed=12345):
    """
    1D blue noise dithering - replicates C implementation.
    Returns array of 0/1 values.
    """
    err0 = 0
    err1 = 0
    output = np.zeros(count, dtype=np.uint8)

    threshold = 6120  # 127.5 * 48
    white_val = 255 * 48

    for i in range(count):
        pixel = brightness * 48 + err0

        if pixel >= threshold:
            output[i] = 1
            quant_err = pixel - white_val
        else:
            output[i] = 0
            quant_err = pixel

        # Hash for kernel selection
        h = lowbias32(np.uint32(i) ^ np.uint32(seed))

        # Shift error buffer
        err0 = err1
        err1 = 0

        if h & 1:
            # FS-like: 100% to next
            err0 += quant_err
        else:
            # JJN-like: 7:5 split
            err0 += (quant_err * 7) // 12
            err1 += (quant_err * 5) // 12

    return output

def simple_error_diffusion_1d(brightness, count):
    """Simple 1D error diffusion (100% forward) for comparison."""
    err = 0
    output = np.zeros(count, dtype=np.uint8)
    threshold = 127.5

    for i in range(count):
        pixel = brightness + err
        if pixel >= threshold:
            output[i] = 1
            err = pixel - 255
        else:
            output[i] = 0
            err = pixel

    return output

def naive_pwm(brightness, count):
    """Naive threshold-based PWM (periodic pattern)."""
    period = 256
    threshold = brightness
    return np.array([(i % period) < threshold for i in range(count)], dtype=np.uint8)

def white_noise_dither(brightness, count, seed=42):
    """White noise dithering (random threshold)."""
    rng = np.random.default_rng(seed)
    threshold = brightness / 255.0
    return (rng.random(count) < threshold).astype(np.uint8)

def compute_spectrum(signal):
    """Compute power spectrum in dB."""
    # Remove DC component
    signal = signal.astype(np.float64) - np.mean(signal)

    # FFT
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2

    # Normalize
    power = power / len(signal)

    # Convert to dB (avoid log(0))
    power_db = 10 * np.log10(power + 1e-20)

    # Frequency axis (0 to 0.5 = Nyquist)
    freqs = np.fft.rfftfreq(len(signal))

    return freqs, power_db

def smooth_spectrum(freqs, power_db, window=50):
    """Smooth spectrum for visualization."""
    from scipy.ndimage import uniform_filter1d
    return freqs, uniform_filter1d(power_db, window)

def analyze_gray_level(gray, count=65536, output_dir=None):
    """Analyze spectrum for a single gray level."""
    print(f"\nAnalyzing gray level {gray} ({gray*100/255:.1f}%)...")

    # Generate signals
    signals = {
        'Our 1D Method': blue_dither_1d(gray, count),
        'Simple ED': simple_error_diffusion_1d(gray, count),
        'White Noise': white_noise_dither(gray, count),
        'Naive PWM': naive_pwm(gray, count),
    }

    # Print duty cycles
    for name, sig in signals.items():
        duty = np.mean(sig) * 100
        print(f"  {name}: {duty:.2f}% duty")

    # Compute spectra
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'1D Temporal Dithering Spectrum Analysis - Gray {gray} ({gray*100/255:.1f}%)', fontsize=14)

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # Plot each method
    for idx, (name, sig) in enumerate(signals.items()):
        ax = axes[idx // 2, idx % 2]
        freqs, power_db = compute_spectrum(sig)
        freqs_smooth, power_smooth = smooth_spectrum(freqs, power_db, window=100)

        # Raw spectrum (light)
        ax.plot(freqs[1:], power_db[1:], alpha=0.3, color=colors[idx], linewidth=0.5)
        # Smoothed spectrum (bold)
        ax.plot(freqs_smooth[1:], power_smooth[1:], color=colors[idx], linewidth=2, label=name)

        # Ideal blue noise reference (power ∝ f²)
        f_ref = freqs_smooth[1:]
        # Scale to match at f=0.25
        idx_quarter = len(f_ref) // 2
        blue_ref = 20 * np.log10(f_ref / f_ref[idx_quarter]) + power_smooth[1:][idx_quarter]
        ax.plot(f_ref, blue_ref, 'k--', alpha=0.5, linewidth=1, label='Ideal Blue (+6dB/oct)')

        # White noise reference (flat)
        white_ref = np.full_like(f_ref, np.mean(power_smooth[len(power_smooth)//4:]))
        ax.plot(f_ref, white_ref, 'k:', alpha=0.5, linewidth=1, label='White (flat)')

        ax.set_xlabel('Frequency (cycles/sample)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(name)
        ax.set_xlim(0, 0.5)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f'spectrum_1d_gray_{gray:03d}.png'
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path.name}")

    plt.close()

    return signals

def analyze_comparison(gray_levels, count=65536, output_dir=None):
    """Compare our method across gray levels."""
    print(f"\nComparing our 1D method across gray levels...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gray_levels)))

    for idx, gray in enumerate(gray_levels):
        sig = blue_dither_1d(gray, count)
        freqs, power_db = compute_spectrum(sig)
        freqs_smooth, power_smooth = smooth_spectrum(freqs, power_db, window=100)

        ax.plot(freqs_smooth[1:], power_smooth[1:], color=colors[idx],
                linewidth=2, label=f'Gray {gray} ({gray*100/255:.0f}%)')

    # Ideal blue noise reference
    f_ref = freqs_smooth[1:]
    idx_quarter = len(f_ref) // 2
    blue_ref = 20 * np.log10(f_ref / f_ref[idx_quarter]) + np.mean(power_smooth[1:][idx_quarter])
    ax.plot(f_ref, blue_ref, 'k--', alpha=0.7, linewidth=2, label='Ideal Blue (+6dB/oct)')

    ax.set_xlabel('Frequency (cycles/sample)', fontsize=12)
    ax.set_ylabel('Power (dB)', fontsize=12)
    ax.set_title('Our 1D Blue Noise Dithering - Spectrum vs Gray Level', fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / 'spectrum_1d_comparison.png'
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path.name}")

    plt.close()

def main():
    output_dir = Path('/root/cra-web/tools/test_images/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pathological gray levels
    gray_levels = [42, 64, 85, 127, 170, 191, 213]

    # Analyze each
    for gray in gray_levels:
        analyze_gray_level(gray, count=65536, output_dir=output_dir)

    # Comparison plot
    analyze_comparison(gray_levels, count=65536, output_dir=output_dir)

    print(f"\nDone! Results in {output_dir}")

if __name__ == '__main__':
    main()
