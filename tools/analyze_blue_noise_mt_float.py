#!/usr/bin/env python3
"""
Analyze blue noise RNG (MT19937 variant) floating-point output.

Same tests as analyze_blue_noise_float.py but using the mt19937 variant
for comparison against the lowbias32 version.

Usage:
    python tools/analyze_blue_noise_mt_float.py
    python tools/analyze_blue_noise_mt_float.py --count 10485760 --seed 42
    python tools/analyze_blue_noise_mt_float.py --no-wav
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import subprocess
import sys
import struct

_BIN_PATH = Path(__file__).parent / 'blue_noise_rng_mt_float_test'
_BIN_LB_PATH = Path(__file__).parent / 'blue_noise_rng_float_test'


def get_float_samples(count, bit_depth, seed):
    """Generate float64 samples by calling the C++ binary in --raw-f64 mode."""
    if not _BIN_PATH.exists():
        print(f"Error: C++ binary not found at {_BIN_PATH}", file=sys.stderr)
        print(f"Build: g++ -O2 -std=c++17 -o {_BIN_PATH} {_BIN_PATH}.cpp", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [str(_BIN_PATH), '--raw-f64', str(count), str(bit_depth), str(seed)],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    return np.frombuffer(result.stdout, dtype=np.float64)


def get_white_samples(bin_path, count, seed):
    """Generate white noise float64 samples from a C++ binary."""
    result = subprocess.run(
        [str(bin_path), '--raw-white-f64', str(count), str(seed)],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    return np.frombuffer(result.stdout, dtype=np.float64)


def spectral_slope(freqs, power, f_lo=0.01, f_hi=0.1):
    """Measure slope in dB/octave between f_lo and f_hi."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() < 2:
        return 0.0
    log_f = np.log2(freqs[mask])
    log_p = 10 * np.log10(power[mask] + 1e-30)
    coeffs = np.polyfit(log_f, log_p, 1)
    return coeffs[0]  # dB per octave


def smooth_spectrum(freqs, power, bins_per_octave=12):
    """Log-spaced smoothing of a power spectrum."""
    f_min = freqs[freqs > 0].min()
    f_max = freqs.max()
    n_octaves = np.log2(f_max / f_min)
    n_bins = max(1, int(n_octaves * bins_per_octave))
    edges = f_min * 2 ** np.linspace(0, n_octaves, n_bins + 1)

    sf, sp = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() > 0:
            sf.append(np.sqrt(lo * hi))  # geometric mean
            sp.append(power[mask].mean())
    return np.array(sf), np.array(sp)


def generate_spectrum(count, seed, output_dir):
    """Generate spectral analysis chart for nextf() at 16-bit depth."""
    bd = 16

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(
        f'Blue Noise RNG (MT19937) nextf() Spectral Analysis ({count:,} samples, seed {seed})',
        fontsize=14
    )

    freqs = np.fft.rfftfreq(count)
    mask = freqs > 0

    # White noise references from both RNG backends
    white_lb = get_white_samples(_BIN_LB_PATH, count, seed)
    fft_wlb = np.fft.rfft(white_lb - white_lb.mean())
    power_wlb = np.abs(fft_wlb) ** 2 / count
    sf_wlb, sp_wlb = smooth_spectrum(freqs[mask], power_wlb[mask])

    white_mt = get_white_samples(_BIN_PATH, count, seed)
    fft_wmt = np.fft.rfft(white_mt - white_mt.mean())
    power_wmt = np.abs(fft_wmt) ** 2 / count
    sf_wmt, sp_wmt = smooth_spectrum(freqs[mask], power_wmt[mask])

    # Blue noise spectrum
    data = get_float_samples(count, bd, seed)
    data_centered = data - data.mean()

    fft_vals = np.fft.rfft(data_centered)
    power = np.abs(fft_vals) ** 2 / count

    sf, sp = smooth_spectrum(freqs[mask], power[mask])
    slope = spectral_slope(sf, sp)

    # Raw spectrum (thin, light)
    ax.loglog(freqs[mask], power[mask], color='#a3d9a5', linewidth=0.3, alpha=0.5)
    # Smoothed blue noise
    ax.loglog(sf, sp, color='#27ae60', linewidth=2, label=f'nextf() bd={bd} (MT19937)')
    # White noise: lowbias32
    ax.loglog(sf_wlb, sp_wlb, color='red', linewidth=1, linestyle='--', alpha=0.6, label='White (lowbias32)')
    # White noise: mt19937
    ax.loglog(sf_wmt, sp_wmt, color='orange', linewidth=1, linestyle='--', alpha=0.6, label='White (mt19937)')
    # Ideal +6 dB/oct
    f_ref = np.array([sf[0], sf[-1]])
    p_ref = sp[len(sp) // 2] * (f_ref / sf[len(sf) // 2]) ** 2
    ax.loglog(f_ref, p_ref, color='black', linewidth=1, linestyle='--', alpha=0.5, label='+6 dB/oct')

    ax.set_title(f'bit_depth={bd}  ({slope:+.1f} dB/oct)')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'spectrum_blue_noise_mt_float.png'
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"  Saved spectrum: {path}")
    print(f"    bit_depth={bd}: {slope:+.1f} dB/oct")


def generate_distribution(count, bit_depth, seed, output_dir):
    """Generate distribution histogram binned at 16-bit resolution."""
    n_bins = 65536

    print(f"  Generating {count:,} float samples (bit_depth={bit_depth}, seed={seed})...")
    data = get_float_samples(count, bit_depth, seed)

    # Bin into 65536 bins: floor(value * 65536), clamp to [0, 65535]
    bins = np.clip((data * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(bins.astype(np.int64), minlength=n_bins)

    expected = count / n_bins
    deviation_pct = (counts - expected) / expected * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(
        f'Blue Noise RNG (MT19937) nextf() Distribution \u2014 65536 bins '
        f'({count:,} samples, bit_depth={bit_depth}, seed {seed})',
        fontsize=14
    )

    # Top: raw counts
    ax1.bar(range(n_bins), counts, width=1, color='#2ecc71', edgecolor='none')
    ax1.axhline(expected, color='red', linestyle='--', linewidth=1,
                label=f'Expected ({expected:,.1f})')
    ax1.set_xlabel('Bin (0-65535)')
    ax1.set_ylabel('Count')
    ax1.set_title('Bin counts (16-bit resolution)')
    ax1.legend()
    ax1.set_xlim(-100, n_bins + 100)

    # Bottom: deviation
    ax2.bar(range(n_bins), deviation_pct, width=1, color='#3498db', edgecolor='none')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Bin (0-65535)')
    ax2.set_ylabel('Deviation from expected (%)')
    ax2.set_title('Deviation from uniform')
    ax2.set_xlim(-100, n_bins + 100)

    plt.tight_layout()
    path = output_dir / f'distribution_mt_float_16bit_bd{bit_depth}.png'
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"  Saved distribution: {path}")
    print(f"    Total samples:    {count:,}")
    print(f"    Bins:             {n_bins}")
    print(f"    Expected per bin: {expected:,.2f}")
    print(f"    Min count:        {counts.min()} (bin {counts.argmin()})")
    print(f"    Max count:        {counts.max()} (bin {counts.argmax()})")
    print(f"    Std dev:          {counts.std():.2f}")
    print(f"    Max deviation:    {abs(deviation_pct).max():.2f}%")


def write_wav_float32(path, sample_rate, data):
    """Write a float32 WAV file (IEEE_FLOAT format, tag=3)."""
    n_samples = len(data)
    n_channels = 1
    bits_per_sample = 32
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = n_samples * block_align

    samples = data.astype(np.float32)

    with open(path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        # fmt chunk - IEEE float
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))           # chunk size
        f.write(struct.pack('<H', 3))            # format tag: IEEE_FLOAT
        f.write(struct.pack('<H', n_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(samples.tobytes())


def generate_wav(seed, output_dir):
    """Generate 30-second float32 WAV files at 8kHz and 48kHz."""
    bit_depth = 16
    duration = 30  # seconds

    for sample_rate in [8000, 48000]:
        n_samples = sample_rate * duration
        label = f'{sample_rate // 1000}k'

        print(f"  Generating {label} float32 WAV ({n_samples:,} samples)...")

        # Blue noise RNG output: [0, 1) -> centered to [-0.5, 0.5)
        data = get_float_samples(n_samples, bit_depth, seed)
        centered = (data - 0.5).astype(np.float32)

        path = output_dir / f'blue_noise_rng_mt_float_{label}.wav'
        write_wav_float32(path, sample_rate, centered)
        size_kb = path.stat().st_size / 1024
        print(f"    Saved: {path} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze blue noise RNG (MT19937) floating-point output'
    )
    parser.add_argument('--count', type=int, default=10_485_760,
                        help='Sample count for spectrum (default: 10485760)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='RNG seed (default: 12345)')
    parser.add_argument('--dist-count', type=int, default=134_217_728,
                        help='Sample count for distribution test (default: 134217728)')
    parser.add_argument('--dist-bits', type=int, default=16,
                        help='Bit depth for distribution test (default: 16)')
    parser.add_argument('--no-wav', action='store_true',
                        help='Skip WAV file generation')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Spectral analysis (nextf MT19937, bit_depth=16) ===")
    generate_spectrum(args.count, args.seed, output_dir)

    print(f"\n=== Distribution histogram (nextf MT19937, bit_depth={args.dist_bits}) ===")
    generate_distribution(args.dist_count, args.dist_bits, args.seed, output_dir)

    if not args.no_wav:
        print(f"\n=== Float32 WAV files (30s, centered, MT19937) ===")
        generate_wav(args.seed, output_dir)

    print(f"\nDone! Results in {output_dir}")


if __name__ == '__main__':
    main()
