#!/usr/bin/env python3
"""
Generate Bayer (ordered) dithered images using a recursive Bayer threshold matrix.

Each pixel is compared against the corresponding value in the Bayer matrix
(tiled to cover the image). No error diffusion is used.

The Bayer matrix is generated recursively from the 2x2 base pattern,
producing matrices of size 2^n x 2^n.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def bayer_matrix(n: int) -> np.ndarray:
    """
    Generate a 2^n x 2^n Bayer threshold matrix with values in [0, 255].

    Uses the recursive definition:
        M(1) = [[0, 2], [3, 1]]
        M(n) = [[4*M(n-1)+0, 4*M(n-1)+2], [4*M(n-1)+3, 4*M(n-1)+1]]
    """
    if n == 0:
        return np.array([[0]], dtype=np.float64)
    m = np.array([[0, 2], [3, 1]], dtype=np.float64)
    for _ in range(n - 1):
        m = np.block([
            [4 * m + 0, 4 * m + 2],
            [4 * m + 3, 4 * m + 1]
        ])
    size = m.shape[0]
    # Normalize to [0, 255]
    return (m / (size * size) * 255).astype(np.uint8)


def bayer_dither(image: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """
    Apply Bayer ordered dithering.

    Each pixel is compared against the corresponding threshold value,
    tiled to match the image dimensions.

    Args:
        image: Grayscale image, values 0-255
        threshold: Bayer threshold matrix, values 0-255

    Returns:
        Binary image (0 or 255)
    """
    h, w = image.shape
    th, tw = threshold.shape

    tiles_y = (h + th - 1) // th
    tiles_x = (w + tw - 1) // tw
    tiled = np.tile(threshold, (tiles_y, tiles_x))[:h, :w]

    return (image > tiled).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description='Generate Bayer ordered dithered images')
    parser.add_argument('--ref-dir', type=Path,
                        default=Path('tools/test_images/sources'),
                        help='Directory containing reference images')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('tools/test_images/dithered/bayer'),
                        help='Output directory')
    parser.add_argument('--order', type=int, default=3,
                        help='Bayer matrix order (size = 2^order, default: 3 = 8x8)')
    parser.add_argument('--suffix', type=str, default='bayer',
                        help='Output filename suffix: {basename}_{suffix}.png')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    size = 2 ** args.order
    threshold = bayer_matrix(args.order)
    print(f"Bayer matrix: {size}x{size} (order {args.order})")

    ref_images = list(args.ref_dir.glob("*.png"))
    if not ref_images:
        print(f"No reference images found in {args.ref_dir}")
        return

    print(f"Generating Bayer dithered images...")
    print(f"  Reference: {args.ref_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    for ref_path in sorted(ref_images):
        img = Image.open(ref_path).convert('L')
        img_array = np.array(img)

        dithered = bayer_dither(img_array, threshold)

        output_path = args.output_dir / f"{ref_path.stem}_{args.suffix}.png"
        Image.fromarray(dithered, mode='L').save(output_path)
        print(f"  {ref_path.stem} -> {output_path}")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
