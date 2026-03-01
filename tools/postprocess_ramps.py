#!/usr/bin/env python3
"""
Post-process ramp images for blog display.
Splits wide ramp images at 512px intervals and stacks the strips vertically.

Input: tools/test_images/dithered/{mode}/ramp_continuous_*.png (4096x64)
       tools/test_images/dithered/{mode}/ramp_step_16_*.png (1024x64)
Output: tools/test_images/dithered/{mode}/ramp_continuous_*_blog.png (512x512)
        tools/test_images/dithered/{mode}/ramp_step_16_*_blog.png (512x128)
"""

import numpy as np
from pathlib import Path
from PIL import Image


STRIP_WIDTH = 512


def split_and_stack(img: Image.Image) -> Image.Image:
    """Split image at STRIP_WIDTH intervals and stack strips vertically."""
    # Convert palette images to grayscale to avoid losing palette data
    if img.mode == 'P':
        img = img.convert('L')

    w, h = img.size
    num_strips = w // STRIP_WIDTH
    if w % STRIP_WIDTH != 0:
        num_strips += 1

    result = Image.new(img.mode, (STRIP_WIDTH, h * num_strips))
    for i in range(num_strips):
        x0 = i * STRIP_WIDTH
        x1 = min(x0 + STRIP_WIDTH, w)
        strip = img.crop((x0, 0, x1, h))
        result.paste(strip, (0, i * h))

    return result


def main():
    dithered_dir = Path(__file__).parent / "test_images" / "dithered"

    prefixes = ["ramp_continuous_", "ramp_step_16_"]
    count = 0

    for mode_dir in sorted(dithered_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        for prefix in prefixes:
            for img_path in sorted(mode_dir.glob(f"{prefix}*.png")):
                if img_path.stem.endswith("_blog"):
                    continue
                img = Image.open(img_path)
                result = split_and_stack(img)
                out_path = img_path.with_stem(img_path.stem + "_blog")
                result.save(out_path)
                print(f"  {out_path.relative_to(dithered_dir)}")
                count += 1

    # Also process source ramps
    sources_dir = Path(__file__).parent / "test_images" / "sources"
    for prefix in ["ramp_continuous", "ramp_step_16"]:
        img_path = sources_dir / f"{prefix}.png"
        if img_path.exists():
            img = Image.open(img_path)
            result = split_and_stack(img)
            out_path = img_path.with_stem(img_path.stem + "_blog")
            result.save(out_path)
            print(f"  sources/{out_path.name}")
            count += 1

    print(f"\nDone! Processed {count} images")


if __name__ == "__main__":
    main()
