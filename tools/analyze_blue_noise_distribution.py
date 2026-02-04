#!/usr/bin/env python3
"""
Analyze blue noise RNG value distribution (uniformity test).

Generates a histogram showing how uniformly the RNG output values are
distributed across all bins, verifying that the population-splitting
tree produces a flat distribution.

Usage:
    python tools/analyze_blue_noise_distribution.py
    python tools/analyze_blue_noise_distribution.py --count 10000000
    python tools/analyze_blue_noise_distribution.py --bits 16 --count 1000000
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import subprocess
import sys

# Path to compiled C binary (next to this script)
_BIN_PATH = Path(__file__).parent / 'blue_noise_rng'


def blue_noise_rng(count, bit_depth, seed):
    """Generate blue noise RNG output by calling the C binary in --raw mode."""
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


def analyze_distribution(count, bit_depth, seed, output_dir):
    """Generate distribution histogram for the blue noise RNG."""
    n_bins = 1 << bit_depth
    expected = count / n_bins

    print(f"Generating {count:,} samples at {bit_depth}-bit (seed {seed})...")
    data = blue_noise_rng(count, bit_depth, seed)

    counts = np.bincount(data, minlength=n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f'Blue Noise RNG {bit_depth}-bit Distribution '
        f'({count:,} samples, seed {seed})',
        fontsize=14
    )

    # Top: raw counts
    ax1.bar(range(n_bins), counts, width=1, color='#2ecc71', edgecolor='none')
    ax1.axhline(expected, color='red', linestyle='--', linewidth=1,
                label=f'Expected ({expected:,.0f})')
    ax1.set_xlabel(f'Value (0-{n_bins - 1})')
    ax1.set_ylabel('Count')
    ax1.set_title('Bin counts')
    ax1.legend()
    ax1.set_xlim(-1, n_bins)

    # Bottom: deviation from expected (in %)
    deviation_pct = (counts - expected) / expected * 100
    ax2.bar(range(n_bins), deviation_pct, width=1, color='#3498db',
            edgecolor='none')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel(f'Value (0-{n_bins - 1})')
    ax2.set_ylabel('Deviation from expected (%)')
    ax2.set_title('Deviation from uniform')
    ax2.set_xlim(-1, n_bins)

    plt.tight_layout()
    path = output_dir / f'distribution_{bit_depth}bit.png'
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"\n  Saved: {path}")
    print(f"  Total samples: {count:,}")
    print(f"  Expected per bin: {expected:,.1f}")
    print(f"  Min count: {counts.min():,} (bin {counts.argmin()})")
    print(f"  Max count: {counts.max():,} (bin {counts.argmax()})")
    print(f"  Std dev of counts: {counts.std():,.1f}")
    print(f"  Max deviation: {abs(deviation_pct).max():.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze blue noise RNG value distribution'
    )
    parser.add_argument('--count', type=int, default=10_000_000,
                        help='Sample count (default: 10000000)')
    parser.add_argument('--bits', type=int, default=8,
                        help='Bit depth (default: 8)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='RNG seed (default: 12345)')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'test_images' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_distribution(args.count, args.bits, args.seed, output_dir)
    print(f"\nDone! Results in {output_dir}")


if __name__ == '__main__':
    main()
