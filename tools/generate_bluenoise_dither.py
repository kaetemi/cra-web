#!/usr/bin/env python3
"""
Generate blue noise dithered images using void-and-cluster threshold array.

Blue noise (ordered) dithering: threshold each pixel against a pre-computed
blue noise texture. This produces good spectral properties without error
diffusion, but may show tiling artifacts on large images.

The blue noise texture is generated using the void-and-cluster algorithm
(Ulichney 1993), which produces an optimal dither array with blue noise
spectral characteristics.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def bluenoise_dither(image: np.ndarray, blue_noise: np.ndarray) -> np.ndarray:
    """
    Apply blue noise threshold dithering.

    Each pixel is compared against the corresponding blue noise threshold,
    tiled to match the image dimensions.

    Args:
        image: Grayscale image, values 0-255
        blue_noise: Blue noise threshold array, values 0-255

    Returns:
        Binary image (0 or 255)
    """
    h, w = image.shape
    bn_h, bn_w = blue_noise.shape

    # Tile the blue noise to cover the entire image
    tiles_y = (h + bn_h - 1) // bn_h
    tiles_x = (w + bn_w - 1) // bn_w
    tiled = np.tile(blue_noise, (tiles_y, tiles_x))[:h, :w]

    return (image > tiled).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description='Generate blue noise dithered images')
    parser.add_argument('--ref-dir', type=Path,
                        default=Path('tools/test_wavelets/reference_images'),
                        help='Directory containing reference images')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('tools/test_wavelets/dithered/bluenoise'),
                        help='Output directory')
    parser.add_argument('--blue-noise', type=Path,
                        default=Path('tools/test_images/blue_noise_256.png'),
                        help='Blue noise threshold array image')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load blue noise texture
    if not args.blue_noise.exists():
        print(f"Error: Blue noise texture not found: {args.blue_noise}")
        print("Generate it using the instructions in tools/README.md")
        return

    blue_noise = np.array(Image.open(args.blue_noise).convert('L'))
    print(f"Loaded blue noise texture: {args.blue_noise} ({blue_noise.shape[0]}x{blue_noise.shape[1]})")

    # Find all reference images
    ref_images = list(args.ref_dir.glob("*.png"))
    if not ref_images:
        print(f"No reference images found in {args.ref_dir}")
        return

    print(f"Generating blue noise dithered images...")
    print(f"  Reference: {args.ref_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    for ref_path in sorted(ref_images):
        img = Image.open(ref_path).convert('L')
        img_array = np.array(img)

        dithered = bluenoise_dither(img_array, blue_noise)

        output_path = args.output_dir / ref_path.name
        Image.fromarray(dithered, mode='L').save(output_path)
        print(f"  {ref_path.stem} -> {output_path}")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
