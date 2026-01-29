#!/usr/bin/env python3
"""
Generate tileable blue noise texture using FFT filtering.

Method: Filter white noise in frequency domain with |H(f)| = f^(dB/6),
giving a configurable power spectrum slope. The result tiles seamlessly
because FFT assumes periodic boundary conditions.

    6 dB/oct → exponent 1 (f^1 amplitude, f² power)
   12 dB/oct → exponent 2 (f^2 amplitude, f⁴ power)

Output is an 8-bit grayscale PNG.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def tileable_blue_noise(size, db_per_octave=6, seed=None):
    """Generate tileable blue noise texture via FFT filtering.

    Args:
        size: Texture width/height in pixels (square).
        db_per_octave: Spectral slope in dB/octave (6=standard blue, 12=2nd order).
        seed: Random seed for reproducibility.

    Returns:
        2D float64 array in [0, 1], tileable.
    """
    if seed is not None:
        np.random.seed(seed)

    white = np.random.randn(size, size)

    fx = np.fft.fftfreq(size)
    fy = np.fft.fftfreq(size)
    Fx, Fy = np.meshgrid(fx, fy)
    F = np.sqrt(Fx**2 + Fy**2)

    # Exponent: 6 dB/oct → f^1, 12 dB/oct → f^2, etc.
    exponent = db_per_octave / 6
    H = F ** exponent
    H[0, 0] = 0  # zero DC

    blue = np.fft.ifft2(H * np.fft.fft2(white)).real
    blue = (blue - blue.min()) / (blue.max() - blue.min())

    return blue


def main():
    parser = argparse.ArgumentParser(
        description='Generate tileable blue noise texture via FFT filtering'
    )
    parser.add_argument('--size', type=int, default=256,
                        help='Texture size (default: 256)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--db', type=int, default=6,
                        help='Spectral slope in dB/octave (default: 6)')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output path (default: auto-named)')
    parser.add_argument('--tiled', action='store_true',
                        help='Also save a 2x2 tiled version')

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f'tools/test_images/fft_blue_noise_{args.db}db_{args.size}.png')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.size}x{args.size} FFT blue noise "
          f"({args.db} dB/oct, seed={args.seed})...")
    texture = tileable_blue_noise(args.size, db_per_octave=args.db, seed=args.seed)

    img = (texture * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(args.output)
    print(f"Saved: {args.output}")

    if args.tiled:
        tiled = np.tile(img, (2, 2))
        tiled_path = args.output.with_stem(args.output.stem + '_tiled')
        Image.fromarray(tiled, mode='L').save(tiled_path)
        print(f"Saved: {tiled_path}")


if __name__ == '__main__':
    main()
