#!/usr/bin/env python3
"""
Analyze blue noise RNG spectrum across bit depths.

Generates spectral analysis comparing the blue noise RNG at bit depths 1-8,
showing how blue noise properties are maintained at all bit depths. The raw
multi-bit output is analyzed directly (no thresholding).

Usage:
    python tools/analyze_blue_noise_rng.py
    python tools/analyze_blue_noise_rng.py --count 262144
    python tools/analyze_blue_noise_rng.py --seed 42
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import subprocess
import sys

# =============================================================================
# Blue noise RNG - calls the C binary
# =============================================================================

# Path to compiled C binary (next to this script)
_BIN_PATH = Path(__file__).parent / 'blue_noise_rng'

def blue_noise_rng(count, bit_depth, seed):
    """
    Generate blue noise RNG output by calling the C binary in --raw mode.

    This ensures the analysis tests the actual C implementation rather than
    a Python reimplementation that could diverge.
    """
    if not _BIN_PATH.exists():
        print(f"Error: C binary not found at {_BIN_PATH}", file=sys.stderr)
        print(f"Build it with: gcc -O2 -o tools/blue_noise_rng tools/blue_noise_rng.c -lm", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [str(_BIN_PATH), '--raw', str(count), str(bit_depth), str(seed)],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"Error running blue_noise_rng: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    if bit_depth <= 8:
        return np.frombuffer(result.stdout, dtype=np.uint8).astype(np.int32)
    else:
        return np.frombuffer(result.stdout, dtype=np.uint16).astype(np.int32)

# =============================================================================
# White noise reference
# =============================================================================

def white_noise(count, bit_depth, seed=42):
    """Uniform random integers in [0, 2^bit_depth - 1]."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << bit_depth, size=count)

# =============================================================================
# Spectrum analysis (same approach as analyze_1d_dither.py)
# =============================================================================

def compute_spectrum(signal):
    """Compute power spectrum in dB (DC removed)."""
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

    bin_centers = []
    bin_means = []
    for i in range(len(bin_edges) - 1):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
        if np.any(mask):
            bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
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
        decades = np.log10(f_high / f_low)
        slope_decade = (p_high - p_low) / decades
        slope_octave = slope_decade / 3.32
        return slope_octave
    return None

# =============================================================================
# Analysis
# =============================================================================

def analyze_bit_depths(count, seed, output_dir):
    """Generate spectral comparison of blue noise RNG at bit depths 1-8."""
    bit_depths = list(range(1, 9))

    print(f"Analyzing blue noise RNG: {count} samples, seed {seed}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Blue Noise RNG - Spectrum vs Bit Depth ({count} samples, seed {seed})', fontsize=14)

    colors_bd = plt.cm.viridis(np.linspace(0.15, 0.9, 8))
    comparison_data = []

    for i, bd in enumerate(bit_depths):
        row, col = divmod(i, 3)
        ax = axes[row][col]
        max_val = (1 << bd) - 1

        print(f"  {bd}-bit (0-{max_val})...", end="", flush=True)

        # Generate blue noise RNG output
        sig_bn = blue_noise_rng(count, bd, seed)

        # Normalize to [0, 1] so spectra are comparable across bit depths
        if max_val > 0:
            sig_bn_norm = sig_bn.astype(np.float64) / max_val
        else:
            sig_bn_norm = sig_bn.astype(np.float64)

        # White noise reference (same normalization)
        sig_wn = white_noise(count, bd)
        if max_val > 0:
            sig_wn_norm = sig_wn.astype(np.float64) / max_val
        else:
            sig_wn_norm = sig_wn.astype(np.float64)

        # Compute spectra
        freqs_bn, power_bn = compute_spectrum(sig_bn_norm)
        freqs_wn, power_wn = compute_spectrum(sig_wn_norm)

        # Smooth
        f_bn, p_bn = smooth_spectrum_log(freqs_bn, power_bn)
        f_wn, p_wn = smooth_spectrum_log(freqs_wn, power_wn)

        # Plot: raw envelope + smoothed line for our method
        ax.semilogx(freqs_bn[1:], power_bn[1:], color='#2ecc71', linewidth=0.3, alpha=0.3)
        ax.semilogx(f_bn, p_bn, color='#2ecc71', linewidth=2, label='Blue Noise RNG')
        ax.semilogx(f_wn, p_wn, color='#e74c3c', linewidth=1.5, alpha=0.7, label='White Noise')

        # +6 dB/oct reference anchored at f=0.1
        f_ref = np.logspace(-3, np.log10(0.5), 100)
        anchor_idx = np.argmin(np.abs(f_bn - 0.1))
        anchor_db = p_bn[anchor_idx] if len(p_bn) > anchor_idx else -20
        ideal = anchor_db + 20 * np.log10(f_ref / 0.1)
        ax.semilogx(f_ref, ideal, 'k--', linewidth=1, alpha=0.5, label='+6dB/oct')

        # Measure slope
        slope = measure_slope(freqs_bn, power_bn)
        slope_str = f'{slope:+.1f} dB/oct' if slope is not None else 'N/A'

        ax.set_title(f'{bd}-bit (0\u2013{max_val})  [{slope_str}]')
        ax.set_xlabel('Frequency (log)')
        ax.set_ylabel('Power (dB)')
        ax.set_xlim(1e-3, 0.5)
        ax.set_ylim(-70, 5)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

        comparison_data.append((bd, f_bn, p_bn, colors_bd[i], slope))
        print(f" {slope_str}")

    # --- Comparison panel (anchored at f=0.1 for slope comparison) ---
    ax_comp = axes[2][2]

    for bd, f_smooth, p_smooth, color, slope in comparison_data:
        # Anchor each curve at f=0.1 -> 0 dB so slopes are directly comparable
        anchor_idx = np.argmin(np.abs(f_smooth - 0.1))
        anchor_val = p_smooth[anchor_idx] if len(p_smooth) > anchor_idx else 0
        p_shifted = p_smooth - anchor_val

        label = f'{bd}-bit'
        if slope is not None:
            label += f' ({slope:+.1f})'
        ax_comp.semilogx(f_smooth, p_shifted, color=color, linewidth=2,
                         label=label, alpha=0.8)

    # +6 dB/oct reference anchored at 0 dB at f=0.1
    f_ref = np.logspace(-3, np.log10(0.5), 100)
    ideal_ref = 20 * np.log10(f_ref / 0.1)
    ax_comp.semilogx(f_ref, ideal_ref, 'k--', linewidth=1.5, alpha=0.5, label='+6dB/oct')

    ax_comp.set_title('Comparison (anchored at f=0.1)')
    ax_comp.set_xlabel('Frequency (log)')
    ax_comp.set_ylabel('Power (dB, relative)')
    ax_comp.set_xlim(1e-3, 0.5)
    ax_comp.set_ylim(-30, 15)
    ax_comp.legend(loc='lower right', fontsize=6)
    ax_comp.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = output_dir / 'spectrum_blue_noise_rng.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")

    # --- Print slope summary ---
    print(f"\n  Spectral slope summary:")
    print(f"  {'Depth':>6} {'Range':>10} {'Slope':>12}")
    print(f"  {'-'*32}")
    for bd, _, _, _, slope in comparison_data:
        max_val = (1 << bd) - 1
        slope_str = f'{slope:+.1f} dB/oct' if slope is not None else 'N/A'
        print(f"  {bd:>4}-bit {'0-' + str(max_val):>8} {slope_str:>12}")
    print(f"  {'-'*32}")
    print(f"  {'Ideal':>6} {'':>10} {'+6.0 dB/oct':>12}")


def generate_reference_blue_noise(count, seed=42):
    """Generate ideal blue noise by filtering white noise with +6 dB/oct slope."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(count)

    # FFT, multiply amplitude by f for +6 dB/oct power spectrum
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(count)
    freqs[0] = 1e-10  # avoid DC
    fft *= freqs

    blue = np.fft.irfft(fft, n=count)

    # Normalize to 0-255
    blue = (blue - blue.min()) / (blue.max() - blue.min()) * 255
    return np.clip(np.round(blue), 0, 255).astype(np.uint8)


def generate_wav(seed, output_dir, bits=8):
    """Generate 30-second WAV files at 8kHz and 48kHz: our RNG, reference blue noise, white noise."""
    import wave

    sample_width = bits // 8
    bit_depth = bits
    max_val = (1 << bit_depth) - 1

    def write_wav(path, data, sr):
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(sample_width)
            w.setframerate(sr)
            w.writeframes(data)

    def to_pcm(unsigned_data):
        """Convert unsigned integer data to WAV PCM format."""
        if bits == 8:
            # 8-bit WAV is unsigned (0-255)
            return unsigned_data.astype(np.uint8).tobytes()
        else:
            # 16-bit WAV is signed: offset unsigned to signed
            pcm16 = (unsigned_data.astype(np.int32) - 32768).astype(np.int16)
            return pcm16.tobytes()

    suffix = f'_{bits}bit' if bits != 8 else ''

    for sr, name in [(8000, '8k'), (48000, '48k')]:
        count = sr * 30

        # Our blue noise RNG
        print(f"  blue_noise_rng_{name}{suffix}.wav ({sr}Hz, {bits}-bit)...", end="", flush=True)
        sig = blue_noise_rng(count, bit_depth, seed)
        write_wav(output_dir / f'blue_noise_rng_{name}{suffix}.wav',
                  to_pcm(sig), sr)
        print(" done")

        # Reference: filtered white noise with +6 dB/oct
        print(f"  ref_blue_noise_{name}{suffix}.wav ({sr}Hz, {bits}-bit)...", end="", flush=True)
        rng_ref = np.random.default_rng(42)
        white = rng_ref.standard_normal(count)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(count)
        freqs[0] = 1e-10
        fft *= freqs
        blue = np.fft.irfft(fft, n=count)
        blue = (blue - blue.min()) / (blue.max() - blue.min()) * max_val
        ref = np.clip(np.round(blue), 0, max_val).astype(np.int32)
        write_wav(output_dir / f'ref_blue_noise_{name}{suffix}.wav',
                  to_pcm(ref), sr)
        print(" done")

        # Reference: white noise
        print(f"  ref_white_noise_{name}{suffix}.wav ({sr}Hz, {bits}-bit)...", end="", flush=True)
        rng_wn = np.random.default_rng(42)
        wn = rng_wn.integers(0, max_val + 1, size=count)
        write_wav(output_dir / f'ref_white_noise_{name}{suffix}.wav',
                  to_pcm(wn), sr)
        print(" done")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze blue noise RNG spectrum across bit depths'
    )
    parser.add_argument('--count', type=int, default=131072,
                        help='Sample count (default: 131072)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='RNG seed (default: 12345)')
    parser.add_argument('--wav', action='store_true',
                        help='Generate 30s WAV files (8kHz and 48kHz, 8-bit)')
    parser.add_argument('--wav16', action='store_true',
                        help='Generate 30s WAV files (8kHz and 48kHz, 16-bit signed)')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_bit_depths(args.count, args.seed, output_dir)

    if args.wav:
        print("\nGenerating 8-bit WAV files:")
        generate_wav(args.seed, output_dir, bits=8)

    if args.wav16:
        print("\nGenerating 16-bit WAV files:")
        generate_wav(args.seed, output_dir, bits=16)

    print(f"\nDone! Results in {output_dir}")


if __name__ == '__main__':
    main()
