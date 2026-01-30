#!/usr/bin/env python3
"""
Recursive ranked dither map with wrapping error diffusion.

Instead of seeded buffers and edge padding, the error diffusion operates
directly on the image with horizontal wrapping. Each row is scanned 3 times
LTR so error from the right edge wraps back to the left and fully settles.

On passes 2+, pixels are already quantized — the re-quantization just
propagates leftover wrapped error until it's fully diffused.

Usage:
    python generate_repeat_map.py --size 256 --seed 42
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'test_map'))

import numpy as np
from PIL import Image
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(func):
            return func
        return wrapper

from generate_recursive_map import (
    triple32,
    quantize,
    tpdf_threshold,
    TPDF_AMPLITUDE,
    transform_image,
    inverse_transform_image,
    analyze_ranked_output,
)


@njit(cache=True)
def apply_error_wrap(buf, x, y, height, width, err, use_jjn):
    """Apply FS or JJN error with horizontal wrapping (LTR only)."""
    if use_jjn:
        buf[y, (x + 1) % width] += err * (7.0 / 48.0)
        buf[y, (x + 2) % width] += err * (5.0 / 48.0)
        if y + 1 < height:
            buf[y + 1, (x - 2) % width] += err * (3.0 / 48.0)
            buf[y + 1, (x - 1) % width] += err * (5.0 / 48.0)
            buf[y + 1, x] += err * (7.0 / 48.0)
            buf[y + 1, (x + 1) % width] += err * (5.0 / 48.0)
            buf[y + 1, (x + 2) % width] += err * (3.0 / 48.0)
        if y + 2 < height:
            buf[y + 2, (x - 2) % width] += err * (1.0 / 48.0)
            buf[y + 2, (x - 1) % width] += err * (3.0 / 48.0)
            buf[y + 2, x] += err * (5.0 / 48.0)
            buf[y + 2, (x + 1) % width] += err * (3.0 / 48.0)
            buf[y + 2, (x + 2) % width] += err * (1.0 / 48.0)
    else:
        buf[y, (x + 1) % width] += err * (7.0 / 16.0)
        if y + 1 < height:
            buf[y + 1, (x - 1) % width] += err * (3.0 / 16.0)
            buf[y + 1, x] += err * (5.0 / 16.0)
            buf[y + 1, (x + 1) % width] += err * (1.0 / 16.0)


@njit(cache=True)
def dither_wrap_loop(buf, height, width, hashed_seed, bits, use_tpdf, row_passes):
    """Dither with horizontal wrapping. Each row scanned row_passes times."""
    for y in range(height):
        for p in range(row_passes):
            for x in range(width):
                old_val = buf[y, x]

                # TPDF only on first pass — subsequent passes just settle error
                if use_tpdf and p == 0:
                    thresh = tpdf_threshold(x, y, hashed_seed, TPDF_AMPLITUDE)
                    new_val = 1.0 if old_val > thresh else 0.0
                else:
                    new_val = quantize(old_val, bits)

                buf[y, x] = new_val
                err = old_val - new_val

                coord_hash = triple32(
                    np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed
                )
                use_jjn = (coord_hash & 1) == 1

                apply_error_wrap(buf, x, y, height, width, err, use_jjn)


def dither_wrap(input_image, bits=1, seed=0, tpdf=False, row_passes=3):
    """Dither with horizontally-wrapping error diffusion."""
    height, width = input_image.shape
    buf = input_image.copy()
    hashed_seed = triple32(np.uint32(seed))
    use_tpdf = tpdf and bits == 1

    dither_wrap_loop(buf, height, width, hashed_seed, bits, use_tpdf, row_passes)
    return buf


def dither_wrap_transformed(input_image, seed=0, transform=True, **kwargs):
    """Dither with wrapping + optional random spatial transform."""
    if transform:
        swap_xy = bool(seed & 1)
        mirror_x = bool(seed & 2)
        mirror_y = bool(seed & 4)
        work = transform_image(input_image, swap_xy, mirror_x, mirror_y)
    else:
        work = input_image

    result = dither_wrap(work, seed=seed, **kwargs)

    if transform:
        result = inverse_transform_image(result, swap_xy, mirror_x, mirror_y)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate tileable recursive ranked dither map (wrapping error diffusion)'
    )
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting seed (default: 42)")
    parser.add_argument("--row-passes", type=int, default=3,
                        help="Number of LTR passes per row for wrapping (default: 3)")
    parser.add_argument("--tpdf", action="store_true",
                        help="Add TPDF threshold perturbation")
    parser.add_argument("--no-transform", action="store_true",
                        help="Disable random spatial transforms per pass")

    args = parser.parse_args()
    output_dir = Path(__file__).parent
    size = args.size
    N = 8

    next_seed = [args.seed]
    def get_seed():
        s = next_seed[0]
        next_seed[0] += 1
        return s

    gray = np.full((size, size), 0.5, dtype=np.float64)
    rank = np.zeros((size, size), dtype=np.int32)

    def save_intermediate(rank_arr, level):
        n_levels = 2 ** (level + 1)
        step = 2 ** (N - 1 - level)
        scaled = (rank_arr // step) * (255 // (n_levels - 1)) if n_levels > 1 else rank_arr
        path = output_dir / f"ranked_level{level}.png"
        Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), mode='L').save(path)
        print(f"  Saved: {path} ({n_levels} levels)")

    use_tpdf = args.tpdf
    use_transform = not args.no_transform
    row_passes = args.row_passes

    print(f"Wrapping error diffusion: {size}x{size}, {row_passes} passes/row")
    if use_tpdf:
        print("TPDF threshold perturbation enabled")
    if not use_transform:
        print("Spatial transforms disabled")

    # Level 0: initial 1-bit split
    print("Level 0: 1-bit dither of 0.5 gray")
    result0 = dither_wrap_transformed(gray, seed=get_seed(),
                                       transform=use_transform,
                                       bits=1, tpdf=use_tpdf,
                                       row_passes=row_passes)
    rank |= (result0 > 0.5).astype(np.int32) << (N - 1)

    img0 = Image.fromarray((result0 * 255).astype(np.uint8), mode='L')
    img0.save(output_dir / "step1_1bit.png")
    w = (result0 > 0.5).sum()
    print(f"  white: {w} ({w/result0.size*100:.1f}%)")
    save_intermediate(rank, 0)

    nodes = [(result0, np.ones((size, size), dtype=bool))]
    total_passes = 1

    for level in range(1, N):
        bit_pos = N - 1 - level
        num_nodes = len(nodes)
        print(f"\nLevel {level}: {num_nodes} nodes, determining bit {bit_pos}...")
        new_nodes = []

        for i, (parent_result, parent_mask) in enumerate(nodes):
            went_high = parent_mask & (parent_result > 0.5)
            went_low = parent_mask & (~(parent_result > 0.5))

            clean = parent_result.copy()

            # Split "high" group
            input_hi = clean.copy()
            input_hi[~parent_mask] = 0.0
            input_hi = input_hi * 0.5
            result_hi = dither_wrap_transformed(input_hi, seed=get_seed(),
                                                 transform=use_transform,
                                                 bits=1, tpdf=use_tpdf,
                                                 row_passes=row_passes)
            rank[went_high] |= (result_hi > 0.5).astype(np.int32)[went_high] << bit_pos
            new_nodes.append((result_hi, went_high))
            total_passes += 1

            # Split "low" group
            input_lo = clean.copy()
            input_lo[~parent_mask] = 1.0
            input_lo = input_lo * 0.5 + 0.5
            result_lo = dither_wrap_transformed(input_lo, seed=get_seed(),
                                                 transform=use_transform,
                                                 bits=1, tpdf=use_tpdf,
                                                 row_passes=row_passes)
            rank[went_low] |= (result_lo > 0.5).astype(np.int32)[went_low] << bit_pos
            new_nodes.append((result_lo, went_low))
            total_passes += 1

        nodes = new_nodes
        save_intermediate(rank, level)

    print(f"\nTotal dither passes: {total_passes}")

    unique_ranks = np.unique(rank)
    print(f"Unique rank values: {len(unique_ranks)} (expected {2**N})")
    print(f"Rank range: [{rank.min()}, {rank.max()}]")

    out_path = output_dir / "ranked_output.png"
    Image.fromarray(rank.astype(np.uint8), mode='L').save(out_path)
    print(f"Saved: {out_path}")
    np.save(output_dir / "ranked_output.npy", rank)

    print("\n--- Spectral Analysis ---")
    analyze_ranked_output(rank, output_dir)


if __name__ == '__main__':
    main()
