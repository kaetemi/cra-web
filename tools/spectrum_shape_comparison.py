#!/usr/bin/env python3
"""
Compare the Mitsa-Parker blue noise model vs simple power law models.

The Mitsa-Parker model proposes a BANDPASS shape:
- Near zero below ~0.5*f_c (low frequency suppression)
- Steep rise to peak at principal frequency f_c
- Plateau after peak

This is fundamentally different from simple power laws:
- +6 dB/oct (f^2): keeps rising forever
- +3 dB/oct (f): keeps rising forever

Note: This is a theoretical model, not necessarily what empirical blue noise
looks like. The plateau essentially acts like a blur filter on the high end.

Usage:
    python spectrum_shape_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def mitsa_parker_model(f, f_c):
    """
    Mitsa-Parker theoretical blue noise model (bandpass shape).

    - Near zero below ~0.5*f_c
    - Steep rise to peak at f_c
    - Plateau after peak (acts like blur filter on high frequencies)
    """
    spectrum = np.zeros_like(f)
    # Rising edge (approximate as steep power law)
    rising = (f / f_c) ** 6  # steep rise
    # Plateau after peak
    plateau = np.ones_like(f)
    # Blend: use rising below f_c, plateau above
    spectrum = np.where(f < f_c, rising, plateau)
    # Add the peak bump
    peak = np.exp(-((f - f_c) / (0.15 * f_c))**2) * 0.5
    spectrum = spectrum + peak
    return spectrum


def main():
    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    f = np.logspace(-2, 0, 500)  # frequency axis (normalized, 0.01 to 1)

    # Principal frequency for g=0.25 with Mitsa-Parker: f_c = (1/sqrt(2)) * sqrt(0.25)
    f_c = 0.354

    # Mitsa-Parker bandpass model
    mp_model = mitsa_parker_model(f, f_c)

    # +6 dB/oct: power proportional to f^2
    power_6db = f ** 2

    # +3 dB/oct: power proportional to f
    power_3db = f

    # Normalize all to same value at f_c for comparison
    idx_fc = np.argmin(np.abs(f - f_c))
    power_6db = power_6db / power_6db[idx_fc]
    power_3db = power_3db / power_3db[idx_fc]
    mp_model = mp_model / mp_model[idx_fc]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax1.plot(f, mp_model, 'b-', linewidth=2.5, label='Mitsa-Parker Model (bandpass)')
    ax1.plot(f, power_6db, 'darkblue', linewidth=2, linestyle='--', label='+6 dB/oct ($f^2$)')
    ax1.plot(f, power_3db, 'deepskyblue', linewidth=2, linestyle=':', label='+3 dB/oct ($f$)')
    ax1.axvline(f_c, color='gray', linestyle='--', alpha=0.5, label=f'Principal freq $f_c$={f_c:.2f}')
    ax1.set_xlabel('Frequency (normalized)', fontsize=12)
    ax1.set_ylabel('Power (linear)', fontsize=12)
    ax1.set_title('Power Spectrum Comparison (Linear Scale)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3)

    # Annotate key difference
    ax1.annotate('Plateau\n(like blur)',
                 xy=(0.7, mp_model[np.argmin(np.abs(f - 0.7))]),
                 xytext=(0.8, 0.5),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax1.annotate('Keeps rising',
                 xy=(0.9, power_6db[np.argmin(np.abs(f - 0.9))]),
                 xytext=(0.75, 2.5),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))

    # Log-log scale (this is how spectra are usually shown)
    ax2.loglog(f, mp_model, 'b-', linewidth=2.5, label='Mitsa-Parker Model (bandpass)')
    ax2.loglog(f, power_6db, 'darkblue', linewidth=2, linestyle='--', label='+6 dB/oct ($f^2$)')
    ax2.loglog(f, power_3db, 'deepskyblue', linewidth=2, linestyle=':', label='+3 dB/oct ($f$)')
    ax2.axvline(f_c, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Frequency (normalized)', fontsize=12)
    ax2.set_ylabel('Power (log)', fontsize=12)
    ax2.set_title('Power Spectrum Comparison (Log-Log Scale)', fontsize=13)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim(0.01, 1)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = output_dir / 'spectrum_shape_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary
    print("\nKey Differences:")
    print("=" * 60)
    print("Mitsa-Parker: bandpass shape, plateaus after f_c (like blur)")
    print("+6 dB/oct:    keeps rising forever as f^2")
    print("+3 dB/oct:    keeps rising forever as f")
    print("-" * 60)
    print(f"\nAt f=1.0 (Nyquist), normalized to 1.0 at f_c={f_c:.2f}:")
    print(f"  Mitsa-Parker: {mp_model[-1]:.2f} (bounded)")
    print(f"  +6 dB/oct:    {power_6db[-1]:.2f} (unbounded)")
    print(f"  +3 dB/oct:    {power_3db[-1]:.2f} (unbounded)")
    print("-" * 60)
    print("\nNote: The Mitsa-Parker model is theoretical. The plateau")
    print("essentially acts like a blur filter on high frequencies.")


if __name__ == '__main__':
    main()
