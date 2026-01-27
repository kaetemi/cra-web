#!/usr/bin/env python3
"""
Generate white noise dithered images for comparison.

White noise dithering: threshold each pixel independently with random noise.
This produces perfect spectral flatness but terrible visual quality
(no error diffusion = no local tone preservation).
"""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def whitenoise_dither(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply white noise threshold dithering.

    Each pixel is compared against a random threshold [0, 255].
    No error diffusion - purely independent random decisions.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = image.shape
    thresholds = np.random.randint(0, 256, size=(h, w))
    return (image > thresholds).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description='Generate white noise dithered images')
    parser.add_argument('--ref-dir', type=Path,
                        default=Path('tools/test_wavelets/reference_images'),
                        help='Directory containing reference images')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('tools/test_wavelets/dithered/whitenoise'),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all reference images
    ref_images = list(args.ref_dir.glob("*.png"))
    if not ref_images:
        print(f"No reference images found in {args.ref_dir}")
        return

    print(f"Generating white noise dithered images...")
    print(f"  Reference: {args.ref_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    for ref_path in sorted(ref_images):
        img = Image.open(ref_path).convert('L')
        img_array = np.array(img)

        # Use consistent seed per image for reproducibility
        seed = args.seed + hash(ref_path.stem) % 10000
        dithered = whitenoise_dither(img_array, seed=seed)

        output_path = args.output_dir / ref_path.name
        Image.fromarray(dithered, mode='L').save(output_path)
        print(f"  {ref_path.stem} -> {output_path}")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
