#!/usr/bin/env python3
"""
Compare noise color spectra for graphics blue noise analysis.

Spectra shown:
- White:  0 dB/octave (flat)
- Blue +3 dB/octave (f) - claimed by some sources
- Blue +6 dB/octave (f²) - standard graphics blue noise
- +12 dB/octave (f⁴) - second-order noise shaping

Usage:
    python noise_color_comparison.py          # Both charts
    python noise_color_comparison.py --log    # Log scale only
    python noise_color_comparison.py --linear # Linear scale only
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_spectra(ax, f, f_ref, log_scale=True):
    """Plot noise spectra on given axes."""
    spectra = {
        'White (0 dB/oct)': 0 * np.log2(f / f_ref),
        'Blue +3 dB/oct (f)': 3 * np.log2(f / f_ref),
        'Blue +6 dB/oct (f²)': 6 * np.log2(f / f_ref),
        '+12 dB/oct (f⁴)': 12 * np.log2(f / f_ref),
    }

    colors = {
        'White (0 dB/oct)': 'gray',
        'Blue +3 dB/oct (f)': 'deepskyblue',
        'Blue +6 dB/oct (f²)': 'blue',
        '+12 dB/oct (f⁴)': 'darkblue',
    }

    for name, spectrum in spectra.items():
        if log_scale:
            ax.semilogx(f, spectrum, color=colors[name], linewidth=2.5, label=name)
        else:
            ax.plot(f, spectrum, color=colors[name], linewidth=2.5, label=name)

    ax.set_xlabel('Frequency (normalized)', fontsize=12)
    ax.set_ylabel('Power (dB)', fontsize=12)
    ax.set_ylim(-40, 20)
    ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')
    ax.legend(loc='lower right', fontsize=11)

    if log_scale:
        ax.set_xlim(1e-3, 0.5)
        ax.annotate('Graphics blue noise\ntypically +6 dB/octave',
                    xy=(0.05, 6 * np.log2(0.05 / f_ref)),
                    xytext=(0.008, -5),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    else:
        ax.set_xlim(0, 0.5)
        ax.annotate('Graphics blue noise\ntypically +6 dB/octave',
                    xy=(0.2, 6 * np.log2(0.2 / f_ref)),
                    xytext=(0.08, -25),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))


def main():
    parser = argparse.ArgumentParser(description='Compare noise color spectra')
    parser.add_argument('--log', action='store_true', help='Log scale only')
    parser.add_argument('--linear', action='store_true', help='Linear scale only')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    f_ref = 0.4  # Reference frequency for 0 dB

    do_log = args.log or (not args.log and not args.linear)
    do_linear = args.linear or (not args.log and not args.linear)

    if do_log:
        f = np.logspace(-3, np.log10(0.5), 500)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_spectra(ax, f, f_ref, log_scale=True)
        ax.set_title('Noise Color Spectra Comparison (Log Frequency)', fontsize=14)
        plt.tight_layout()
        output_path = output_dir / 'noise_color_comparison.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()

    if do_linear:
        f = np.linspace(0.001, 0.5, 500)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_spectra(ax, f, f_ref, log_scale=False)
        ax.set_title('Noise Color Spectra Comparison (Linear Frequency)', fontsize=14)
        plt.tight_layout()
        output_path = output_dir / 'noise_color_comparison_linear.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()

    # Print summary table
    print("\nNoise Color Reference:")
    print("=" * 50)
    print(f"{'Name':<25} {'Slope':<15} {'Power ∝':<10}")
    print("-" * 50)
    print(f"{'White':<25} {'0 dB/octave':<15} {'constant':<10}")
    print(f"{'Blue +3 dB (some claim)':<25} {'+3 dB/octave':<15} {'f':<10}")
    print(f"{'Blue +6 dB (graphics)':<25} {'+6 dB/octave':<15} {'f²':<10}")
    print(f"{'(2nd order shaping)':<25} {'+12 dB/octave':<15} {'f⁴':<10}")
    print("-" * 50)
    print("\nGraphics blue noise is typically +6 dB/octave (f²)")


if __name__ == '__main__':
    main()
