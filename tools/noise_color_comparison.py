#!/usr/bin/env python3
"""
Compare noise color spectra: audio blue noise vs violet noise vs higher orders.

Audio terminology:
- White:  0 dB/octave (flat)
- Pink:  -3 dB/octave (1/f)
- Red:   -6 dB/octave (1/f²)
- Blue:  +3 dB/octave (f)
- Violet: +6 dB/octave (f²)

Graphics "blue noise" typically means +6 dB/octave (violet in audio terms).

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
        'Blue - audio (+3 dB/oct)': 3 * np.log2(f / f_ref),
        'Violet (+6 dB/oct)': 6 * np.log2(f / f_ref),
        '+12 dB/oct': 12 * np.log2(f / f_ref),
    }

    colors = {
        'White (0 dB/oct)': 'gray',
        'Blue - audio (+3 dB/oct)': 'blue',
        'Violet (+6 dB/oct)': 'violet',
        '+12 dB/oct': 'darkviolet',
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
        ax.annotate('Graphics "blue noise"\nis actually violet\n(+6 dB/octave)',
                    xy=(0.05, 6 * np.log2(0.05 / f_ref)),
                    xytext=(0.008, -5),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='violet', lw=1.5))
    else:
        ax.set_xlim(0, 0.5)
        ax.annotate('Graphics "blue noise"\nis actually violet\n(+6 dB/octave)',
                    xy=(0.2, 6 * np.log2(0.2 / f_ref)),
                    xytext=(0.08, -25),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='violet', lw=1.5))


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
        ax.set_title('Noise Color Spectra Comparison (Log Frequency)\n(Audio terminology vs Graphics "blue noise")', fontsize=14)
        plt.tight_layout()
        output_path = output_dir / 'noise_color_comparison.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()

    if do_linear:
        f = np.linspace(0.001, 0.5, 500)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_spectra(ax, f, f_ref, log_scale=False)
        ax.set_title('Noise Color Spectra Comparison (Linear Frequency)\n(Audio terminology vs Graphics "blue noise")', fontsize=14)
        plt.tight_layout()
        output_path = output_dir / 'noise_color_comparison_linear.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()

    # Print summary table
    print("\nNoise Color Reference:")
    print("=" * 50)
    print(f"{'Color':<20} {'Slope':<15} {'Power ∝':<10}")
    print("-" * 50)
    print(f"{'Pink':<20} {'-3 dB/octave':<15} {'1/f':<10}")
    print(f"{'Red/Brown':<20} {'-6 dB/octave':<15} {'1/f²':<10}")
    print(f"{'White':<20} {'0 dB/octave':<15} {'constant':<10}")
    print(f"{'Blue (audio)':<20} {'+3 dB/octave':<15} {'f':<10}")
    print(f"{'Violet':<20} {'+6 dB/octave':<15} {'f²':<10}")
    print(f"{'(unnamed)':<20} {'+12 dB/octave':<15} {'f⁴':<10}")
    print("-" * 50)
    print("\nNote: Graphics 'blue noise' = Violet (+6 dB/octave)")


if __name__ == '__main__':
    main()
