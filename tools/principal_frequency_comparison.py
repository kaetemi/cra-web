#!/usr/bin/env python3
"""
Compare principal frequency scaling with gray level for blue noise methods.

The principal frequency f_c is where the blue noise power spectrum peaks.
For Poisson disk / blue noise patterns, spacing ~ 1/sqrt(density), so f_c ~ sqrt(density).

Formulation (Mitsa-Parker):
  f_c = K * sqrt(g/R)      for g < 0.5
  f_c = K * sqrt((1-g)/R)  for g >= 0.5

Where:
  - g is the gray level (0 to 1)
  - R is the aspect ratio (usually 1)
  - K is a constant:
    - Mitsa-Parker optimal: K = 1/sqrt(2) ~ 0.707
    - Error diffusion: K = 1

The sqrt(g) scaling is fundamental to blue noise: it comes from the relationship
between dot density and spacing in a Poisson disk distribution.

Usage:
    python principal_frequency_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    g = np.linspace(0.01, 0.99, 200)
    R = 1

    # Mitsa-Parker optimal (K = 1/sqrt(2))
    K_mp = 1 / np.sqrt(2)
    f_c_mp = np.where(g < 0.5, K_mp * np.sqrt(g / R), K_mp * np.sqrt((1 - g) / R))

    # Error diffusion (K = 1)
    K_ed = 1.0
    f_c_ed = np.where(g < 0.5, K_ed * np.sqrt(g / R), K_ed * np.sqrt((1 - g) / R))

    # Linear scaling (naive assumption) - normalized to compare
    f_c_linear = np.where(g < 0.5, g, (1 - g))
    f_c_linear = f_c_linear / 0.5 * 0.707  # normalize to match at g=0.5

    plt.figure(figsize=(10, 6))
    plt.plot(g, f_c_mp, 'b-', linewidth=2.5, label=r'Mitsa-Parker (K=1/$\sqrt{2}$) $\propto \sqrt{g}$')
    plt.plot(g, f_c_ed, 'g--', linewidth=2.5, label=r'Error Diffusion (K=1) $\propto \sqrt{g}$')
    plt.plot(g, f_c_linear, 'r:', linewidth=2, label=r'Linear scaling (naive) $\propto g$')

    plt.xlabel('Gray Level g', fontsize=12)
    plt.ylabel('Principal Frequency $f_c$', fontsize=12)
    plt.title('Blue Noise Principal Frequency vs Gray Level', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 0.8)

    # Add annotations for key points
    plt.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    # Annotate the sqrt relationship
    plt.annotate(r'$f_c = K\sqrt{g}$' + '\n(density = g)',
                 xy=(0.25, K_mp * np.sqrt(0.25)),
                 xytext=(0.12, 0.55),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    plt.tight_layout()
    output_path = output_dir / 'principal_frequency_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary table
    print("\nPrincipal Frequency at Key Gray Levels:")
    print("=" * 55)
    print(f"{'Gray Level':<15} {'Mitsa-Parker':<15} {'Error Diff':<15} {'Linear':<10}")
    print("-" * 55)

    for gray in [0.25, 0.5, 0.75]:
        if gray < 0.5:
            mp_val = K_mp * np.sqrt(gray)
            ed_val = K_ed * np.sqrt(gray)
            lin_val = gray / 0.5 * 0.707
        else:
            mp_val = K_mp * np.sqrt(1 - gray)
            ed_val = K_ed * np.sqrt(1 - gray)
            lin_val = (1 - gray) / 0.5 * 0.707

        print(f"{gray:<15.2f} {mp_val:<15.3f} {ed_val:<15.3f} {lin_val:<10.3f}")

    print("-" * 55)
    print("\nNote: sqrt(g) scaling is fundamental to blue noise.")
    print("It arises from Poisson disk spacing ~ 1/sqrt(density).")


if __name__ == '__main__':
    main()
