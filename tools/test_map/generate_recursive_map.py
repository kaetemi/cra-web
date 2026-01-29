#!/usr/bin/env python3
"""
Mixed FS/JJN error diffusion dithering in floating point.

Replicates our dithering method but uses 0.0-1.0 range instead of uint8.
Supports any bit depth quantization.

Usage:
    python generate_recursive_map.py --gradient 1 2 4 8  # Generate gradient at different bit depths
    python generate_recursive_map.py --bits 4 --gray 0.5  # Dither 50% gray at 4-bit
"""

import numpy as np
from PIL import Image
from pathlib import Path
from collections import deque
import argparse


def lowbias32(x: np.uint32) -> np.uint32:
    """Lowbias32 hash - improved version with bias 0.107."""
    x = np.uint32(x)
    x ^= x >> np.uint32(16)
    x = np.uint32(np.uint64(x) * np.uint64(0x21f0aaad) & 0xFFFFFFFF)
    x ^= x >> np.uint32(15)
    x = np.uint32(np.uint64(x) * np.uint64(0x735a2d97) & 0xFFFFFFFF)
    x ^= x >> np.uint32(15)
    return x


def apply_fs_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Floyd-Steinberg kernel, left-to-right."""
    h, w = buf.shape
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 16.0)
    if y + 1 < h:
        if x > 0:
            buf[y + 1, x - 1] += err * (3.0 / 16.0)
        buf[y + 1, x] += err * (5.0 / 16.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (1.0 / 16.0)


def apply_fs_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Floyd-Steinberg kernel, right-to-left."""
    h, w = buf.shape
    if x > 0:
        buf[y, x - 1] += err * (7.0 / 16.0)
    if y + 1 < h:
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 16.0)
        buf[y + 1, x] += err * (5.0 / 16.0)
        if x > 0:
            buf[y + 1, x - 1] += err * (1.0 / 16.0)


def apply_jjn_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Jarvis-Judice-Ninke kernel, left-to-right."""
    h, w = buf.shape
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 48.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (5.0 / 48.0)
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (3.0 / 48.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 48.0)
        buf[y + 1, x] += err * (7.0 / 48.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 48.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (3.0 / 48.0)
    if y + 2 < h:
        if x >= 2:
            buf[y + 2, x - 2] += err * (1.0 / 48.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (3.0 / 48.0)
        buf[y + 2, x] += err * (5.0 / 48.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (3.0 / 48.0)
        if x + 2 < w:
            buf[y + 2, x + 2] += err * (1.0 / 48.0)


def apply_jjn_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Jarvis-Judice-Ninke kernel, right-to-left."""
    h, w = buf.shape
    if x >= 1:
        buf[y, x - 1] += err * (7.0 / 48.0)
    if x >= 2:
        buf[y, x - 2] += err * (5.0 / 48.0)
    if y + 1 < h:
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (3.0 / 48.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 48.0)
        buf[y + 1, x] += err * (7.0 / 48.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 48.0)
        if x >= 2:
            buf[y + 1, x - 2] += err * (3.0 / 48.0)
    if y + 2 < h:
        if x + 2 < w:
            buf[y + 2, x + 2] += err * (1.0 / 48.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (3.0 / 48.0)
        buf[y + 2, x] += err * (5.0 / 48.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (3.0 / 48.0)
        if x >= 2:
            buf[y + 2, x - 2] += err * (1.0 / 48.0)


def quantize(value: float, bits: int) -> float:
    """
    Quantize a value to N-bit levels.

    For N bits, we have 2^N levels: 0, 1/(2^N-1), 2/(2^N-1), ..., 1.0
    """
    num_levels = 2 ** bits
    max_level = num_levels - 1

    # Quantize: round to nearest level
    level = round(value * max_level)
    level = max(0, min(max_level, level))

    # Convert back to 0-1 range
    return level / max_level if max_level > 0 else 0.0


def apply_error(buf, x, y, err, use_jjn, is_rtl):
    """Apply the appropriate error diffusion kernel."""
    if use_jjn:
        if is_rtl:
            apply_jjn_rtl(buf, x, y, err)
        else:
            apply_jjn_ltr(buf, x, y, err)
    else:
        if is_rtl:
            apply_fs_rtl(buf, x, y, err)
        else:
            apply_fs_ltr(buf, x, y, err)


def dither(
    input_image: np.ndarray,
    bits: int = 1,
    seed: int = 0,
    delay: int = 0
) -> np.ndarray:
    """
    Apply mixed FS/JJN error diffusion dithering.

    Args:
        input_image: Input image with values in [0.0, 1.0]
        bits: Output bit depth (1 = binary, 2 = 4 levels, etc.)
        seed: Random seed for kernel selection
        delay: FIFO delay in pixels before error is diffused (0 = immediate)

    Returns:
        Dithered image with values in [0.0, 1.0], quantized to 2^bits levels
    """
    height, width = input_image.shape
    buf = input_image.copy().astype(np.float64)
    output = np.zeros((height, width), dtype=np.float64)
    hashed_seed = lowbias32(np.uint32(seed))
    fifo = deque()

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = quantize(old_val, bits)
            output[y, x] = new_val
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1

            fifo.append((x, y, err, use_jjn, is_rtl))

            if len(fifo) > delay:
                dx, dy, derr, d_jjn, d_rtl = fifo.popleft()
                apply_error(buf, dx, dy, derr, d_jjn, d_rtl)

    # Flush remaining
    while fifo:
        dx, dy, derr, d_jjn, d_rtl = fifo.popleft()
        apply_error(buf, dx, dy, derr, d_jjn, d_rtl)

    return output


def generate_gradient(width: int, height: int) -> np.ndarray:
    """Generate a horizontal gradient from 0 to 1."""
    return np.tile(np.linspace(0, 1, width), (height, 1))


def main():
    parser = argparse.ArgumentParser(
        description="Mixed FS/JJN error diffusion in floating point"
    )
    parser.add_argument("--bits", type=int, default=1,
                        help="Output bit depth (default: 1)")
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--gray", type=float,
                        help="Dither uniform gray level (0.0-1.0)")
    parser.add_argument("--delay", type=int, default=0,
                        help="FIFO delay in pixels before error diffusion (default: 0)")
    parser.add_argument("--gradient", nargs="+", type=int,
                        help="Generate gradient at specified bit depths")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path")

    args = parser.parse_args()
    output_dir = Path(__file__).parent

    if args.gradient:
        # Generate gradient visualizations at multiple bit depths
        for bits in args.gradient:
            print(f"\nGenerating {bits}-bit gradient...")

            # Create gradient
            gradient = generate_gradient(args.size, args.size)

            # Dither it
            dithered = dither(gradient, bits=bits, seed=args.seed, delay=args.delay)

            # Save as PNG (scale to 0-255)
            delay_suffix = f"_delay{args.delay}" if args.delay > 0 else ""
            out_path = output_dir / f"gradient_{bits}bit{delay_suffix}.png"
            img = (dithered * 255).astype(np.uint8)
            Image.fromarray(img, mode='L').save(out_path)
            print(f"Saved: {out_path}")

            # Save raw float data
            npy_path = output_dir / f"gradient_{bits}bit{delay_suffix}.npy"
            np.save(npy_path, dithered)
            print(f"Saved: {npy_path}")

            # Stats
            unique = np.unique(dithered)
            print(f"Unique levels: {len(unique)} (expected {2**bits})")
            print(f"Levels: {unique}")

    elif args.gray is not None:
        # Dither uniform gray
        print(f"Dithering {args.gray:.3f} gray at {args.bits}-bit...")

        input_img = np.full((args.size, args.size), args.gray, dtype=np.float64)
        dithered = dither(input_img, bits=args.bits, seed=args.seed, delay=args.delay)

        out_path = args.output or str(output_dir / f"gray_{args.gray:.2f}_{args.bits}bit.png")
        img = (dithered * 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(out_path)
        print(f"Saved: {out_path}")

        # Stats
        unique, counts = np.unique(dithered, return_counts=True)
        total = args.size * args.size
        print(f"\nLevel distribution:")
        for level, count in zip(unique, counts):
            print(f"  {level:.4f}: {count:6d} ({count/total*100:.2f}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
