#!/usr/bin/env python3
"""
Generate adaptive 1st/2nd order blend dithered images for wavelet analysis.

Uses gradient-adaptive blending from test_twelve/second_order_dither.py:
smooth areas get 2nd-order (2H-H²) kernels for steeper noise shaping,
sharp edges get 1st-order (FS/JJN) for stability.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

# Import from test_twelve
sys.path.insert(0, str(Path(__file__).parent / 'test_twelve'))
from second_order_dither import dither_adaptive_blend


def main():
    parser = argparse.ArgumentParser(description='Generate adaptive blend dithered images')
    parser.add_argument('--ref-dir', type=Path,
                        default=Path('tools/test_wavelets/reference_images'),
                        help='Directory containing reference images')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('tools/test_wavelets/dithered/adaptive-blend'),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_images = list(args.ref_dir.glob("*.png"))
    if not ref_images:
        print(f"No reference images found in {args.ref_dir}")
        return

    print(f"Generating adaptive 1st/2nd blend dithered images...")
    print(f"  Reference: {args.ref_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    for ref_path in sorted(ref_images):
        img = Image.open(ref_path).convert('L')
        input_image = np.array(img, dtype=np.float64) / 255.0

        result = dither_adaptive_blend(input_image, seed=args.seed)
        out_img = (result * 255).astype(np.uint8)

        output_path = args.output_dir / ref_path.name
        Image.fromarray(out_img, mode='L').save(output_path)
        print(f"  {ref_path.stem} -> {output_path}")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
