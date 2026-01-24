#!/usr/bin/env python3
"""
Standalone Python replication of "Our Method" (mixed FS/JJN dithering with lowbias32).

For testing/hacking purposes - generates 1-bit dithered images from arbitrary gray levels.

Usage:
    python our_method_dither.py 127.5                    # 50% gray, default output
    python our_method_dither.py 64 -o my_output.png      # 25% gray, custom output
    python our_method_dither.py 127.5 --size 512         # Larger image

Note: This is a simplified implementation that doesn't include edge seeding.
The CRA Rust implementation has edge seeding (duplicating edge pixels into buffer padding)
which normalizes edge behavior. This causes exact pixel-level differences but the
spectral characteristics and density are equivalent.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def lowbias32(x: np.uint32) -> np.uint32:
    """Lowbias32 hash - improved version with bias 0.107.

    Reference: https://github.com/skeeto/hash-prospector/issues/19
    Constants: [16 21f0aaad 15 735a2d97 15]
    """
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
    # Row 0: * 7 5
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 48.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (5.0 / 48.0)
    # Row 1: 3 5 7 5 3
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
    # Row 2: 1 3 5 3 1
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
    # Row 0: 5 7 *
    if x >= 1:
        buf[y, x - 1] += err * (7.0 / 48.0)
    if x >= 2:
        buf[y, x - 2] += err * (5.0 / 48.0)
    # Row 1: 3 5 7 5 3
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
    # Row 2: 1 3 5 3 1
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


def apply_wide20_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Wide 20 kernel, left-to-right.

    Kernel:         * 7
            1 3 5 3 1
    Positions: row 0: +1
               row 1: -2, -1, 0, +1, +2
    Divided by 20.
    """
    h, w = buf.shape
    # Row 0: * 7
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 20.0)
    # Row 1: 1 3 5 3 1 at positions -2, -1, 0, +1, +2
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (1.0 / 20.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (3.0 / 20.0)
        buf[y + 1, x] += err * (5.0 / 20.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 20.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (1.0 / 20.0)


def apply_wide20_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Wide 20 kernel, right-to-left.

    Kernel: 7 *
            1 3 5 3 1
    Positions: row 0: -1
               row 1: -2, -1, 0, +1, +2
    Divided by 20.
    """
    h, w = buf.shape
    # Row 0: 7 *
    if x >= 1:
        buf[y, x - 1] += err * (7.0 / 20.0)
    # Row 1: 1 3 5 3 1 at positions -2, -1, 0, +1, +2
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (1.0 / 20.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (3.0 / 20.0)
        buf[y + 1, x] += err * (5.0 / 20.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 20.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (1.0 / 20.0)


def apply_wide32_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Wide 32 kernel, left-to-right.

    Kernel:           * 7 3
            1 3 5 7 5 1
    Positions: row 0: +1, +2
               row 1: -3, -2, -1, 0, +1, +2
    Divided by 32.
    """
    h, w = buf.shape
    # Row 0: * 7 3
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 32.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (3.0 / 32.0)
    # Row 1: 1 3 5 7 5 1 at positions -3, -2, -1, 0, +1, +2
    if y + 1 < h:
        if x >= 3:
            buf[y + 1, x - 3] += err * (1.0 / 32.0)
        if x >= 2:
            buf[y + 1, x - 2] += err * (3.0 / 32.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 32.0)
        buf[y + 1, x] += err * (7.0 / 32.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 32.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (1.0 / 32.0)


def apply_wide32_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Wide 32 kernel, right-to-left.

    Kernel: 3 7 *
            1 5 7 5 3 1
    Positions: row 0: -2, -1
               row 1: -2, -1, 0, +1, +2, +3
    Divided by 32.
    """
    h, w = buf.shape
    # Row 0: 3 7 *
    if x >= 1:
        buf[y, x - 1] += err * (7.0 / 32.0)
    if x >= 2:
        buf[y, x - 2] += err * (3.0 / 32.0)
    # Row 1: 1 5 7 5 3 1 at positions -2, -1, 0, +1, +2, +3
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (1.0 / 32.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 32.0)
        buf[y + 1, x] += err * (7.0 / 32.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 32.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (3.0 / 32.0)
        if x + 3 < w:
            buf[y + 1, x + 3] += err * (1.0 / 32.0)


def apply_asym24_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Asymmetric 24 kernel, left-to-right.

    Kernel:       * 7 5
            1 3 5 3
    Positions: row 0: +1, +2
               row 1: -2, -1, 0, +1
    Divided by 24.
    """
    h, w = buf.shape
    # Row 0: * 7 5
    if x + 1 < w:
        buf[y, x + 1] += err * (7.0 / 24.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (5.0 / 24.0)
    # Row 1: 1 3 5 3 at positions -2, -1, 0, +1
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (1.0 / 24.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (3.0 / 24.0)
        buf[y + 1, x] += err * (5.0 / 24.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 24.0)


def apply_asym24_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Asymmetric 24 kernel, right-to-left.

    Kernel: 5 7 *
            3 5 3 1
    Positions: row 0: -1, -2
               row 1: -1, 0, +1, +2
    Divided by 24.
    """
    h, w = buf.shape
    # Row 0: 5 7 *
    if x >= 1:
        buf[y, x - 1] += err * (7.0 / 24.0)
    if x >= 2:
        buf[y, x - 2] += err * (5.0 / 24.0)
    # Row 1: 3 5 3 1 at positions -1, 0, +1, +2
    if y + 1 < h:
        if x >= 1:
            buf[y + 1, x - 1] += err * (3.0 / 24.0)
        buf[y + 1, x] += err * (5.0 / 24.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 24.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (1.0 / 24.0)


def apply_symmetric24_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Symmetric 24 kernel, left-to-right.

    Kernel:       * 5 3
            3 5 5 3
    Positions: row 0: +1, +2
               row 1: -2, -1, 0, +1
    Divided by 24.
    """
    h, w = buf.shape
    # Row 0: * 5 3
    if x + 1 < w:
        buf[y, x + 1] += err * (5.0 / 24.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (3.0 / 24.0)
    # Row 1: 3 5 5 3 at positions -2, -1, 0, +1
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (3.0 / 24.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (5.0 / 24.0)
        buf[y + 1, x] += err * (5.0 / 24.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (3.0 / 24.0)


def apply_symmetric24_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Symmetric 24 kernel, right-to-left.

    Kernel: 3 5 *
              3 5 5 3
    Positions: row 0: -1, -2
               row 1: -1, 0, +1, +2
    Divided by 24.
    """
    h, w = buf.shape
    # Row 0: 3 5 *
    if x >= 1:
        buf[y, x - 1] += err * (5.0 / 24.0)
    if x >= 2:
        buf[y, x - 2] += err * (3.0 / 24.0)
    # Row 1: 3 5 5 3 at positions -1, 0, +1, +2
    if y + 1 < h:
        if x >= 1:
            buf[y + 1, x - 1] += err * (3.0 / 24.0)
        buf[y + 1, x] += err * (5.0 / 24.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (5.0 / 24.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (3.0 / 24.0)


def apply_stucki_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Stucki kernel, left-to-right.

    Kernel:     * 8 4
            2 4 8 4 2
            1 2 4 2 1
    Divided by 42.
    """
    h, w = buf.shape
    # Row 0: * 8 4
    if x + 1 < w:
        buf[y, x + 1] += err * (8.0 / 42.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (4.0 / 42.0)
    # Row 1: 2 4 8 4 2
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (2.0 / 42.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (4.0 / 42.0)
        buf[y + 1, x] += err * (8.0 / 42.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (4.0 / 42.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (2.0 / 42.0)
    # Row 2: 1 2 4 2 1
    if y + 2 < h:
        if x >= 2:
            buf[y + 2, x - 2] += err * (1.0 / 42.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (2.0 / 42.0)
        buf[y + 2, x] += err * (4.0 / 42.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (2.0 / 42.0)
        if x + 2 < w:
            buf[y + 2, x + 2] += err * (1.0 / 42.0)


def apply_stucki_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Stucki kernel, right-to-left.

    Kernel: 4 8 *
            2 4 8 4 2
            1 2 4 2 1
    Divided by 42.
    """
    h, w = buf.shape
    # Row 0: 4 8 *
    if x >= 1:
        buf[y, x - 1] += err * (8.0 / 42.0)
    if x >= 2:
        buf[y, x - 2] += err * (4.0 / 42.0)
    # Row 1: 2 4 8 4 2
    if y + 1 < h:
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (2.0 / 42.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (4.0 / 42.0)
        buf[y + 1, x] += err * (8.0 / 42.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (4.0 / 42.0)
        if x >= 2:
            buf[y + 1, x - 2] += err * (2.0 / 42.0)
    # Row 2: 1 2 4 2 1
    if y + 2 < h:
        if x + 2 < w:
            buf[y + 2, x + 2] += err * (1.0 / 42.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (2.0 / 42.0)
        buf[y + 2, x] += err * (4.0 / 42.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (2.0 / 42.0)
        if x >= 2:
            buf[y + 2, x - 2] += err * (1.0 / 42.0)


def apply_sierra_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Sierra (full) kernel, left-to-right.

    Kernel:     * 5 3
            2 4 5 4 2
              2 3 2
    Divided by 32.
    """
    h, w = buf.shape
    # Row 0: * 5 3
    if x + 1 < w:
        buf[y, x + 1] += err * (5.0 / 32.0)
    if x + 2 < w:
        buf[y, x + 2] += err * (3.0 / 32.0)
    # Row 1: 2 4 5 4 2
    if y + 1 < h:
        if x >= 2:
            buf[y + 1, x - 2] += err * (2.0 / 32.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (4.0 / 32.0)
        buf[y + 1, x] += err * (5.0 / 32.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (4.0 / 32.0)
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (2.0 / 32.0)
    # Row 2: 2 3 2
    if y + 2 < h:
        if x >= 1:
            buf[y + 2, x - 1] += err * (2.0 / 32.0)
        buf[y + 2, x] += err * (3.0 / 32.0)
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (2.0 / 32.0)


def apply_sierra_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Sierra (full) kernel, right-to-left.

    Kernel: 3 5 *
            2 4 5 4 2
              2 3 2
    Divided by 32.
    """
    h, w = buf.shape
    # Row 0: 3 5 *
    if x >= 1:
        buf[y, x - 1] += err * (5.0 / 32.0)
    if x >= 2:
        buf[y, x - 2] += err * (3.0 / 32.0)
    # Row 1: 2 4 5 4 2
    if y + 1 < h:
        if x + 2 < w:
            buf[y + 1, x + 2] += err * (2.0 / 32.0)
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (4.0 / 32.0)
        buf[y + 1, x] += err * (5.0 / 32.0)
        if x >= 1:
            buf[y + 1, x - 1] += err * (4.0 / 32.0)
        if x >= 2:
            buf[y + 1, x - 2] += err * (2.0 / 32.0)
    # Row 2: 2 3 2
    if y + 2 < h:
        if x + 1 < w:
            buf[y + 2, x + 1] += err * (2.0 / 32.0)
        buf[y + 2, x] += err * (3.0 / 32.0)
        if x >= 1:
            buf[y + 2, x - 1] += err * (2.0 / 32.0)


def apply_sierra_lite_ltr(buf: np.ndarray, x: int, y: int, err: float):
    """Sierra Lite kernel, left-to-right.

    Kernel:   * 2
             1 1
    Divided by 4.
    """
    h, w = buf.shape
    if x + 1 < w:
        buf[y, x + 1] += err * (2.0 / 4.0)
    if y + 1 < h:
        if x > 0:
            buf[y + 1, x - 1] += err * (1.0 / 4.0)
        buf[y + 1, x] += err * (1.0 / 4.0)


def apply_sierra_lite_rtl(buf: np.ndarray, x: int, y: int, err: float):
    """Sierra Lite kernel, right-to-left.

    Kernel: 2 *
            1 1
    Divided by 4.
    """
    h, w = buf.shape
    if x > 0:
        buf[y, x - 1] += err * (2.0 / 4.0)
    if y + 1 < h:
        if x + 1 < w:
            buf[y + 1, x + 1] += err * (1.0 / 4.0)
        buf[y + 1, x] += err * (1.0 / 4.0)


def our_method_dither_fs_sierra(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Sierra (full) dithering.

    Like our standard method but alternates between Floyd-Steinberg and Sierra
    kernels instead of FS and JJN.

    Sierra has 10 coefficients (vs JJN's 12), similar complexity.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_sierra = (coord_hash & 1) == 1

            if use_sierra:
                if is_rtl:
                    apply_sierra_rtl(buf, x, y, err)
                else:
                    apply_sierra_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_fs_sierra_lite(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Sierra-Lite dithering.

    Like our standard method but alternates between Floyd-Steinberg and Sierra Lite
    kernels instead of FS and JJN.

    Sierra Lite is much simpler (only 3 coefficients vs JJN's 12).
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_sierra = (coord_hash & 1) == 1

            if use_sierra:
                if is_rtl:
                    apply_sierra_lite_rtl(buf, x, y, err)
                else:
                    apply_sierra_lite_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_fs_stucki(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Stucki dithering.

    Alternates between Floyd-Steinberg and Stucki kernels.
    Stucki is similar to JJN (12 coefficients, 3 rows) but with different weights.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_stucki = (coord_hash & 1) == 1

            if use_stucki:
                if is_rtl:
                    apply_stucki_rtl(buf, x, y, err)
                else:
                    apply_stucki_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_fs_asym24(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Asym24 dithering.

    Alternates between Floyd-Steinberg (4 coeff, 1 row) and
    Asym24 (6 coeff, 2 rows: * 7 5 / 1 3 5 3) kernels.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_asym24 = (coord_hash & 1) == 1

            if use_asym24:
                if is_rtl:
                    apply_asym24_rtl(buf, x, y, err)
                else:
                    apply_asym24_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_fs_symmetric24(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Symmetric24 dithering.

    Alternates between Floyd-Steinberg (4 coeff, 1 row) and
    Symmetric24 (6 coeff, 2 rows) kernels.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_sym24 = (coord_hash & 1) == 1

            if use_sym24:
                if is_rtl:
                    apply_symmetric24_rtl(buf, x, y, err)
                else:
                    apply_symmetric24_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_fs_wide20(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Wide20 dithering.

    Alternates between Floyd-Steinberg (4 coeff, 1 row) and
    Wide20 (6 coeff, 2 rows: * 7 / 1 3 5 3 1) kernels.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_wide20 = (coord_hash & 1) == 1

            if use_wide20:
                if is_rtl:
                    apply_wide20_rtl(buf, x, y, err)
                else:
                    apply_wide20_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_fs_wide32(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed FS/Wide32 dithering.

    Alternates between Floyd-Steinberg (4 coeff, 1 row) and
    Wide32 (8 coeff, 2 rows: * 7 3 / 1 3 5 7 5 1) kernels.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_wide32 = (coord_hash & 1) == 1

            if use_wide32:
                if is_rtl:
                    apply_wide32_rtl(buf, x, y, err)
                else:
                    apply_wide32_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_fs_rtl(buf, x, y, err)
                else:
                    apply_fs_ltr(buf, x, y, err)

    return output


def our_method_dither_jjn_sierra(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed JJN/Sierra dithering.

    Alternates between two large kernels (no Floyd-Steinberg).
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_sierra = (coord_hash & 1) == 1

            if use_sierra:
                if is_rtl:
                    apply_sierra_rtl(buf, x, y, err)
                else:
                    apply_sierra_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_jjn_rtl(buf, x, y, err)
                else:
                    apply_jjn_ltr(buf, x, y, err)

    return output


def our_method_dither_jjn_stucki(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed JJN/Stucki dithering.

    Alternates between two large kernels (no Floyd-Steinberg).
    Both have 12 coefficients across 3 rows.
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_stucki = (coord_hash & 1) == 1

            if use_stucki:
                if is_rtl:
                    apply_stucki_rtl(buf, x, y, err)
                else:
                    apply_stucki_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_jjn_rtl(buf, x, y, err)
                else:
                    apply_jjn_ltr(buf, x, y, err)

    return output


def our_method_dither_stucki_sierra(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0
) -> np.ndarray:
    """
    Experimental: Mixed Stucki/Sierra dithering.

    Alternates between two large kernels (no Floyd-Steinberg).
    Both have similar complexity (10-12 coefficients, 3 rows).
    """
    buf = np.full((height, width), gray_level, dtype=np.float32)
    hashed_seed = lowbias32(np.uint32(seed))
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)
            err = old_val - new_val

            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_sierra = (coord_hash & 1) == 1

            if use_sierra:
                if is_rtl:
                    apply_sierra_rtl(buf, x, y, err)
                else:
                    apply_sierra_ltr(buf, x, y, err)
            else:
                if is_rtl:
                    apply_stucki_rtl(buf, x, y, err)
                else:
                    apply_stucki_ltr(buf, x, y, err)

    return output


def our_method_dither_with_blue_noise_kernel(
    gray_level: float,
    width: int = 256,
    height: int = 256,
    seed: int = 0,
    kernel_pattern: np.ndarray = None,
    recursion_depth: int = 1
) -> np.ndarray:
    """
    Experimental: Use a pre-computed blue noise pattern for kernel selection.

    Instead of using hash(x,y) directly (white noise), this uses a dithered
    50% gray pattern (which has blue noise characteristics) to select kernels.

    Args:
        gray_level: Input gray value (0-255, can be fractional like 127.5)
        width: Output image width
        height: Output image height
        seed: Random seed for generating kernel pattern if not provided
        kernel_pattern: Pre-computed binary pattern for kernel selection.
                       If None, generates one recursively.
        recursion_depth: How many levels of blue noise kernel to use.
                        1 = use standard dither for kernel pattern
                        2 = use level-1 blue kernel dither for kernel pattern
                        etc.

    Returns:
        np.ndarray: 1-bit dithered image (values 0 or 255)
    """
    # Generate kernel selection pattern if not provided
    if kernel_pattern is None:
        # Use a different seed for kernel pattern to avoid correlation
        kernel_seed = seed ^ 0xDEADBEEF
        if recursion_depth <= 1:
            # Base case: use standard hash-based dither for kernel pattern
            kernel_pattern = our_method_dither(127.5, width, height, kernel_seed)
        else:
            # Recursive case: use blue-kernel dither at lower depth
            kernel_pattern = our_method_dither_with_blue_noise_kernel(
                127.5, width, height, kernel_seed,
                recursion_depth=recursion_depth - 1
            )

    # Initialize buffer with uniform gray level
    buf = np.full((height, width), gray_level, dtype=np.float32)

    # Output array
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        # Serpentine: alternate direction each row
        if y % 2 == 0:
            x_range = range(width)
            is_rtl = False
        else:
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]

            # 1-bit quantization: threshold at 127.5
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)

            # Compute error
            err = old_val - new_val

            # Use kernel pattern for selection (blue noise instead of white noise)
            use_jjn = kernel_pattern[y, x] > 0

            # Apply kernel
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

    return output


def our_method_dither(gray_level: float, width: int = 256, height: int = 256, seed: int = 0) -> np.ndarray:
    """
    Apply 'Our Method' dithering to a uniform gray level.

    Algorithm:
    - Mixed FS/JJN kernel selection per-pixel using lowbias32 hash
    - Serpentine scanning (alternating direction each row)
    - 1-bit quantization (0 or 255)

    Args:
        gray_level: Input gray value (0-255, can be fractional like 127.5)
        width: Output image width
        height: Output image height
        seed: Random seed for hash

    Returns:
        np.ndarray: 1-bit dithered image (values 0 or 255)
    """
    # Initialize buffer with uniform gray level
    buf = np.full((height, width), gray_level, dtype=np.float32)

    # Hash the seed
    hashed_seed = lowbias32(np.uint32(seed))

    # Output array
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        # Serpentine: alternate direction each row
        if y % 2 == 0:
            # Left-to-right
            x_range = range(width)
            is_rtl = False
        else:
            # Right-to-left
            x_range = range(width - 1, -1, -1)
            is_rtl = True

        for x in x_range:
            old_val = buf[y, x]

            # 1-bit quantization: threshold at 127.5
            new_val = 255.0 if old_val >= 127.5 else 0.0
            output[y, x] = int(new_val)

            # Compute error
            err = old_val - new_val

            # Hash coordinates to select kernel
            coord_hash = lowbias32(np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ hashed_seed)
            use_jjn = (coord_hash & 1) == 1

            # Apply kernel
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

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1-bit dithered image using Our Method (mixed FS/JJN with lowbias32)"
    )
    parser.add_argument("gray_level", type=float, help="Gray level (0-255, e.g., 127.5 for 50%%)")
    parser.add_argument("-o", "--output", type=str, help="Output path (default: our_method_{level}.png)")
    parser.add_argument("--size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--blue-kernel", action="store_true",
                        help="Experimental: use blue noise pattern for kernel selection instead of hash")
    parser.add_argument("--compare", action="store_true",
                        help="Generate both standard and blue-kernel versions for comparison")

    args = parser.parse_args()

    if args.compare:
        # Generate both versions for comparison
        print(f"Generating comparison at gray level {args.gray_level}...")

        standard = our_method_dither(args.gray_level, args.size, args.size, args.seed)
        blue_kernel = our_method_dither_with_blue_noise_kernel(args.gray_level, args.size, args.size, args.seed)

        # Save both
        standard_path = f"our_method_{args.gray_level:.1f}_standard.png"
        blue_path = f"our_method_{args.gray_level:.1f}_blue_kernel.png"

        Image.fromarray(standard, mode='L').save(standard_path)
        Image.fromarray(blue_kernel, mode='L').save(blue_path)

        print(f"Standard (hash-based kernel): {standard_path}")
        print(f"  White pixels: {np.mean(standard == 255) * 100:.2f}%")
        print(f"Blue noise kernel: {blue_path}")
        print(f"  White pixels: {np.mean(blue_kernel == 255) * 100:.2f}%")

        # Pixel difference
        diff = np.sum(standard != blue_kernel)
        print(f"Pixel difference: {diff} ({diff / (args.size * args.size) * 100:.2f}%)")

    else:
        output_path = args.output or f"our_method_{args.gray_level:.1f}.png"

        print(f"Generating {args.size}x{args.size} dithered image at gray level {args.gray_level}...")

        if args.blue_kernel:
            print("Using experimental blue noise kernel selection...")
            result = our_method_dither_with_blue_noise_kernel(args.gray_level, args.size, args.size, args.seed)
        else:
            result = our_method_dither(args.gray_level, args.size, args.size, args.seed)

        Image.fromarray(result, mode='L').save(output_path)
        print(f"Saved to {output_path}")

        # Print statistics
        white_pct = np.mean(result == 255) * 100
        print(f"White pixels: {white_pct:.2f}% (expected: {args.gray_level / 255 * 100:.2f}%)")


if __name__ == "__main__":
    main()
