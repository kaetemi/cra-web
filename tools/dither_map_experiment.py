#!/usr/bin/env python3
"""
Dither map experiments - generate and analyze threshold maps.

This file contains:
1. Clean implementation of "Our Method" (FS/JJN with lowbias32) for generating patterns
2. Tools for building threshold maps from error diffusion patterns
3. Spectral analysis to compare against blue noise and error diffusion

Usage:
    # Generate single dither pattern
    python dither_map_experiment.py --gray 127.5
    python dither_map_experiment.py --gray 64 -o my_output.png --size 512

    # Generate 8-bit threshold map (from 8 independent 50% patterns)
    python dither_map_experiment.py --generate-map -o threshold_map.png

    # Analyze a threshold map (spectral comparison charts)
    python dither_map_experiment.py --analyze-map threshold_map.png

    # Test a threshold map (text output of density accuracy)
    python dither_map_experiment.py --test-map threshold_map.png
"""

import numpy as np
from numpy.fft import fft2, fftshift
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


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


# =============================================================================
# Floyd-Steinberg kernel (4 coefficients, 1 row forward)
# =============================================================================
#
#       * 7
#     3 5 1    ÷16
#

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


# =============================================================================
# Jarvis-Judice-Ninke kernel (12 coefficients, 2 rows forward)
# =============================================================================
#
#         * 7 5
#     3 5 7 5 3    ÷48
#     1 3 5 3 1
#

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


# =============================================================================
# Main dithering function
# =============================================================================

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


# =============================================================================
# Example: Custom kernel (copy and modify for experiments)
# =============================================================================
#
# def apply_custom_ltr(buf: np.ndarray, x: int, y: int, err: float):
#     """Custom kernel, left-to-right.
#
#     Kernel:     * a b
#             c d e f g    ÷N
#     """
#     h, w = buf.shape
#     N = ???  # Sum of coefficients
#     # Row 0: * a b
#     if x + 1 < w:
#         buf[y, x + 1] += err * (a / N)
#     if x + 2 < w:
#         buf[y, x + 2] += err * (b / N)
#     # Row 1: c d e f g
#     if y + 1 < h:
#         if x >= 2:
#             buf[y + 1, x - 2] += err * (c / N)
#         if x >= 1:
#             buf[y + 1, x - 1] += err * (d / N)
#         buf[y + 1, x] += err * (e / N)
#         if x + 1 < w:
#             buf[y + 1, x + 1] += err * (f / N)
#         if x + 2 < w:
#             buf[y + 1, x + 2] += err * (g / N)


# =============================================================================
# Spectral analysis functions
# =============================================================================

def compute_segmented_radial_power(img: np.ndarray):
    """Compute radial power spectrum segmented by direction (H, D, V)."""
    h, w = img.shape
    centered = img - img.mean()
    spectrum = np.abs(fftshift(fft2(centered))) ** 2

    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)
    img_size = min(h, w)

    y_coords, x_coords = np.ogrid[:h, :w]
    dy = y_coords - cy
    dx = x_coords - cx
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))

    horizontal, diagonal, vertical = [], [], []

    for r in range(1, max_radius):
        ring_mask = np.abs(distances - r) < 0.5
        if not ring_mask.any():
            horizontal.append(0)
            diagonal.append(0)
            vertical.append(0)
            continue

        ring_angles = angles[ring_mask]
        ring_power = spectrum[ring_mask]

        h_mask = (ring_angles < 22.5) | (ring_angles > 157.5)
        v_mask = (ring_angles > 67.5) & (ring_angles < 112.5)
        d_mask = ((ring_angles > 22.5) & (ring_angles < 67.5)) | \
                 ((ring_angles > 112.5) & (ring_angles < 157.5))

        horizontal.append(ring_power[h_mask].mean() if h_mask.any() else 0)
        diagonal.append(ring_power[d_mask].mean() if d_mask.any() else 0)
        vertical.append(ring_power[v_mask].mean() if v_mask.any() else 0)

    freqs = np.arange(1, max_radius) / img_size
    return freqs, np.array(horizontal), np.array(diagonal), np.array(vertical)


def analyze_threshold_map(threshold_map: np.ndarray, output_dir: Path, gray_levels: list = None):
    """
    Analyze a threshold map by comparing it against blue noise and our method
    at various gray levels.
    """
    if gray_levels is None:
        gray_levels = [42, 85, 127, 170]

    # Try to load reference images
    base_dir = Path(__file__).parent / "test_images"
    blue_noise_path = base_dir / "blue_noise_256.png"

    if not blue_noise_path.exists():
        print(f"  Warning: Blue noise reference not found at {blue_noise_path}")
        blue_noise = None
    else:
        blue_noise = np.array(Image.open(blue_noise_path).convert('L'))

    to_db = lambda x: 10 * np.log10(x + 1e-10)

    for gray in gray_levels:
        print(f"  Analyzing gray level {gray}...")

        # Threshold our map
        our_map_result = ((gray >= threshold_map) * 255).astype(np.float32)

        # Load references
        images = {'8-Bit Map': our_map_result}

        if blue_noise is not None:
            images['Blue Noise'] = ((blue_noise < gray) * 255).astype(np.float32)

        # Try to load error diffusion result
        ed_path = base_dir / "dithered" / "boon-serpentine" / f"gray_{gray:03d}_boon-serpentine.png"
        if ed_path.exists():
            images['Our Method (ED)'] = np.array(Image.open(ed_path).convert('L')).astype(np.float32)

        # Create comparison figure
        n_images = len(images)
        fig, axes = plt.subplots(2, n_images, figsize=(5 * n_images, 10))

        if n_images == 1:
            axes = axes.reshape(2, 1)

        # Compute spectra and find global y-range
        all_spectra = {}
        y_min, y_max = float('inf'), float('-inf')

        for name, img in images.items():
            freqs, h, d, v = compute_segmented_radial_power(img)
            h_db, d_db, v_db = to_db(h), to_db(d), to_db(v)
            all_spectra[name] = (freqs, h_db, d_db, v_db)
            y_min = min(y_min, h_db.min(), d_db.min(), v_db.min())
            y_max = max(y_max, h_db.max(), d_db.max(), v_db.max())

        y_min -= 5
        y_max += 5

        for col, (name, img) in enumerate(images.items()):
            # Row 1: Image
            axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=255)
            white_pct = np.mean(img == 255) * 100
            axes[0, col].set_title(f'{name}\n({white_pct:.1f}% white)')
            axes[0, col].axis('off')

            # Row 2: Spectrum
            freqs, h_db, d_db, v_db = all_spectra[name]
            axes[1, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8)
            axes[1, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8)
            axes[1, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8)
            axes[1, col].set_xlim(0, 0.5)
            axes[1, col].set_ylim(y_min, y_max)
            axes[1, col].set_xlabel('cycles/px')
            if col == 0:
                axes[1, col].set_ylabel('Power (dB)')
            axes[1, col].legend(loc='lower right')
            axes[1, col].grid(True, alpha=0.3)

        expected_pct = gray / 255 * 100
        plt.suptitle(f'Threshold Map Analysis @ Gray {gray} ({expected_pct:.1f}%)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"threshold_map_analysis_gray{gray:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path}")


# =============================================================================
# Threshold map generation methods
# =============================================================================

def generate_xor_chain_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0) -> np.ndarray:
    """
    Generate threshold map using XOR chain degradation.

    Bit 7: pure blue noise (50% pattern)
    Bit 6: pattern XOR'd with bit 7 (degrades toward white noise)
    Bit 5: pattern XOR'd with bit 6
    ... and so on

    Each XOR step degrades blue noise toward white noise.
    """
    bits = [None] * 8

    # Generate 8 independent 50% blue noise patterns
    patterns = []
    for i in range(8):
        p = our_method_dither(127.5, width, height, seed=base_seed + i)
        patterns.append((p == 255).astype(np.uint8))

    # Bit 7: pure blue noise
    bits[7] = patterns[0]

    # Chain XOR down through the bits
    bits[6] = patterns[1] ^ bits[7]
    bits[5] = patterns[2] ^ bits[6]
    bits[4] = patterns[3] ^ bits[5]
    bits[3] = patterns[4] ^ bits[4]
    bits[2] = patterns[5] ^ bits[3]
    bits[1] = patterns[6] ^ bits[2]
    bits[0] = patterns[7] ^ bits[1]

    # Combine bits into threshold map
    threshold_map = np.zeros((height, width), dtype=np.uint8)
    for i in range(8):
        threshold_map += bits[i] * (1 << i)

    return threshold_map


def generate_weighted_stack_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0,
                                          pattern_density: float = 50.0) -> np.ndarray:
    """
    Generate threshold map by stacking weighted patterns.

    For i in 1..255:
        Generate pattern at pattern_density gray level
        Add i to pixels that are ON

    Then normalize the sum to 0-255.

    Args:
        pattern_density: Gray level for each pattern (default 50 = ~20% density)
    """
    total = np.zeros((height, width), dtype=np.float64)

    for i in range(1, 256):
        pattern = our_method_dither(pattern_density, width, height, seed=base_seed + i)
        total += (pattern == 255).astype(np.float64) * i

    # Normalize to 0-255
    threshold_map = ((total - total.min()) / (total.max() - total.min()) * 255).astype(np.uint8)
    return threshold_map


def generate_recursive_subdiv_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0) -> np.ndarray:
    """
    Generate threshold map using recursive subdivision.

    Algorithm:
    1. Generate 50% blue noise mask. ON pixels → "low" group (0-127), OFF → "high" group (128-255)
    2. Within each group, generate another 50% mask to subdivide further
    3. Recurse 8 times until every pixel has a unique threshold value

    This preserves perfect density at all gray levels and strong blue noise at 50%.
    """
    seed_counter = base_seed

    def get_pattern():
        nonlocal seed_counter
        p = our_method_dither(127.5, width, height, seed=seed_counter)
        seed_counter += 1
        return (p == 255)

    # Track the low bound for each pixel's threshold range
    low_bound = np.zeros((height, width), dtype=np.float32)
    range_size = 256.0

    for level in range(8):
        # Generate a 50% blue noise pattern
        split_mask = get_pattern()

        # ON pixels stay in lower half, OFF pixels move to upper half
        low_bound = np.where(split_mask, low_bound, low_bound + range_size / 2)
        range_size /= 2

    return low_bound.astype(np.uint8)


def generate_and_not_or_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0,
                                       base_pct: float = 50.0, or_pct: float = 25.0,
                                       xor_prev: bool = False, return_bits: bool = False):
    """
    Generate threshold map using AND NOT OR logic.

    For each bit (except MSB):
        base = (base_pct% pattern) AND NOT (previous bit)
        extra = or_pct% pattern
        bit = base OR extra
        if xor_prev: bit = bit XOR previous_bit

    Args:
        base_pct: Gray level percentage for the AND NOT part (default 50%)
        or_pct: Gray level percentage for the OR part (default 25%)
        xor_prev: Whether to XOR with the previous bit (default False)
    """
    bits = [None] * 8
    seed_counter = base_seed

    def get_pattern(gray_pct):
        nonlocal seed_counter
        gray_level = gray_pct * 255.0 / 100.0
        p = our_method_dither(gray_level, width, height, seed=seed_counter)
        seed_counter += 1
        return (p == 255).astype(np.uint8)

    # Bit 7: 50% pattern (always)
    bits[7] = get_pattern(50.0)

    # Build remaining bits
    for i in range(6, -1, -1):
        base = get_pattern(base_pct) & (~bits[i + 1] & 1)
        extra = get_pattern(or_pct)
        bits[i] = base | extra
        if xor_prev:
            bits[i] = bits[i] ^ bits[i + 1]

    # Combine bits into threshold map
    threshold_map = np.zeros((height, width), dtype=np.uint8)
    for i in range(8):
        threshold_map += bits[i] * (1 << i)

    if return_bits:
        return threshold_map, bits
    return threshold_map


def generate_boundary_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0) -> np.ndarray:
    """
    Generate threshold map using only boundary XOR (simplified approach).

    For each bit, only XOR the extreme boundary gray levels:
    - Bit 7: pattern at 128
    - Bit 6: pattern at 64 XOR pattern at 192
    - Bit 5: pattern at 32 XOR pattern at 224
    - Bit 4: pattern at 16 XOR pattern at 240
    - ... and so on

    No cascading - each bit is independent.
    """
    bits = [None] * 8
    seed_counter = base_seed

    def get_pattern(gray_level):
        nonlocal seed_counter
        pattern = our_method_dither(float(gray_level), width, height, seed=seed_counter)
        seed_counter += 1
        return (pattern == 255).astype(np.uint8)

    print("  Bit 7: gray 128")
    bits[7] = get_pattern(128)

    print("  Bit 6: gray 64 XOR 192")
    bits[6] = get_pattern(64) ^ get_pattern(192)

    print("  Bit 5: gray 32 XOR 224")
    bits[5] = get_pattern(32) ^ get_pattern(224)

    print("  Bit 4: gray 16 XOR 240")
    bits[4] = get_pattern(16) ^ get_pattern(240)

    print("  Bit 3: gray 8 XOR 248")
    bits[3] = get_pattern(8) ^ get_pattern(248)

    print("  Bit 2: gray 4 XOR 252")
    bits[2] = get_pattern(4) ^ get_pattern(252)

    print("  Bit 1: gray 2 XOR 254")
    bits[1] = get_pattern(2) ^ get_pattern(254)

    print("  Bit 0: gray 1 XOR 255")
    bits[0] = get_pattern(1) ^ get_pattern(255)

    # Combine bits into threshold map
    threshold_map = np.zeros((height, width), dtype=np.uint8)
    for i in range(8):
        threshold_map += bits[i] * (1 << i)

    return threshold_map


def generate_recursive_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0) -> np.ndarray:
    """
    Generate threshold map using iterative XOR cascade approach.

    Bit 7: generate at 128
    Bit 6: generate at 64 and 192, XOR them, then XOR with bit 7 to update bit 7
    Bit 5: generate at 32, 96, 160, 224, XOR all, cascade up
    ... and so on

    Returns:
        np.ndarray: 8-bit threshold map (values 0-255)
    """
    bits = [None] * 8  # bits[7] is MSB, bits[0] is LSB
    seed_counter = base_seed

    def get_pattern(gray_level):
        nonlocal seed_counter
        pattern = our_method_dither(float(gray_level), width, height, seed=seed_counter)
        seed_counter += 1
        return (pattern == 255).astype(np.uint8)

    # Bit 7 (weight 128): generate at 128
    print("  Bit 7: gray 128")
    bits[7] = get_pattern(128)

    # Bit 6 (weight 64): generate at 64 and 192, XOR them
    print("  Bit 6: gray 64 XOR 192, cascade to bit 7")
    p64 = get_pattern(64)
    p192 = get_pattern(192)
    bits[6] = p64 ^ p192
    bits[7] = bits[7] ^ bits[6]  # cascade up

    # Bit 5 (weight 32): generate at 32, 96, 160, 224
    print("  Bit 5: gray 32 XOR 96 XOR 160 XOR 224, cascade up")
    bits[5] = get_pattern(32) ^ get_pattern(96) ^ get_pattern(160) ^ get_pattern(224)
    bits[6] = bits[6] ^ bits[5]
    bits[7] = bits[7] ^ bits[6]

    # Bit 4 (weight 16): generate at 16, 48, 80, 112, 144, 176, 208, 240
    print("  Bit 4: 8 gray levels, cascade up")
    bits[4] = np.zeros((height, width), dtype=np.uint8)
    for g in [16, 48, 80, 112, 144, 176, 208, 240]:
        bits[4] ^= get_pattern(g)
    bits[5] = bits[5] ^ bits[4]
    bits[6] = bits[6] ^ bits[5]
    bits[7] = bits[7] ^ bits[6]

    # Bit 3 (weight 8): 16 gray levels
    print("  Bit 3: 16 gray levels, cascade up")
    bits[3] = np.zeros((height, width), dtype=np.uint8)
    for g in range(8, 256, 16):  # 8, 24, 40, ..., 248
        bits[3] ^= get_pattern(g)
    bits[4] = bits[4] ^ bits[3]
    bits[5] = bits[5] ^ bits[4]
    bits[6] = bits[6] ^ bits[5]
    bits[7] = bits[7] ^ bits[6]

    # Bit 2 (weight 4): 32 gray levels
    print("  Bit 2: 32 gray levels, cascade up")
    bits[2] = np.zeros((height, width), dtype=np.uint8)
    for g in range(4, 256, 8):  # 4, 12, 20, ..., 252
        bits[2] ^= get_pattern(g)
    bits[3] = bits[3] ^ bits[2]
    bits[4] = bits[4] ^ bits[3]
    bits[5] = bits[5] ^ bits[4]
    bits[6] = bits[6] ^ bits[5]
    bits[7] = bits[7] ^ bits[6]

    # Bit 1 (weight 2): 64 gray levels
    print("  Bit 1: 64 gray levels, cascade up")
    bits[1] = np.zeros((height, width), dtype=np.uint8)
    for g in range(2, 256, 4):  # 2, 6, 10, ..., 254
        bits[1] ^= get_pattern(g)
    bits[2] = bits[2] ^ bits[1]
    bits[3] = bits[3] ^ bits[2]
    bits[4] = bits[4] ^ bits[3]
    bits[5] = bits[5] ^ bits[4]
    bits[6] = bits[6] ^ bits[5]
    bits[7] = bits[7] ^ bits[6]

    # Bit 0 (weight 1): 128 gray levels
    print("  Bit 0: 128 gray levels, cascade up")
    bits[0] = np.zeros((height, width), dtype=np.uint8)
    for g in range(1, 256, 2):  # 1, 3, 5, ..., 255
        bits[0] ^= get_pattern(g)
    bits[1] = bits[1] ^ bits[0]
    bits[2] = bits[2] ^ bits[1]
    bits[3] = bits[3] ^ bits[2]
    bits[4] = bits[4] ^ bits[3]
    bits[5] = bits[5] ^ bits[4]
    bits[6] = bits[6] ^ bits[5]
    bits[7] = bits[7] ^ bits[6]

    # Combine bits into threshold map
    threshold_map = np.zeros((height, width), dtype=np.uint8)
    for i in range(8):
        threshold_map += bits[i] * (1 << i)

    return threshold_map


def generate_stacked_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0) -> np.ndarray:
    """
    Generate threshold map by stacking 255 independent patterns at each gray level.

    For each gray level from 1 to 255, generate an error-diffused pattern with
    a DIFFERENT seed. Count how many patterns have each pixel white.
    The count (0-255) is converted to threshold.

    Returns:
        np.ndarray: 8-bit threshold map (values 0-255)
    """
    # Count how many patterns have each pixel white
    white_count = np.zeros((height, width), dtype=np.int32)

    for gray in range(1, 256):
        if gray % 16 == 0:
            print(f"  Processing gray level {gray}/255...")
        # Generate pattern at this gray level with UNIQUE seed
        pattern = our_method_dither(float(gray), width, height, seed=base_seed + gray)
        # Count white pixels
        white_count += (pattern == 255).astype(np.int32)

    # white_count has limited variance (~6.5 std dev around 128)
    # To spread across full 0-255 range, RANK the pixels by count
    # Pixels with highest count get lowest threshold (easiest to turn white)
    flat_count = white_count.flatten()
    # argsort gives indices that would sort the array
    # We want highest count = lowest threshold, so negate
    ranks = np.argsort(np.argsort(-flat_count))  # double argsort gives ranks
    # Scale ranks to 0-255
    threshold_map = (ranks * 255 // (len(ranks) - 1)).reshape(height, width).astype(np.uint8)

    return threshold_map


def generate_8bit_threshold_map(width: int = 256, height: int = 256, base_seed: int = 0) -> np.ndarray:
    """
    Generate a 256-level threshold map by combining 8 independent 50% dither patterns.

    Each bit of the output threshold value comes from one dither pattern:
    - Bit 0 (value 1): from pattern with seed base_seed + 0
    - Bit 1 (value 2): from pattern with seed base_seed + 1
    - ...
    - Bit 7 (value 128): from pattern with seed base_seed + 7

    The resulting map can be used for ordered dithering: output = (input >= map[x,y]) ? 255 : 0

    Returns:
        np.ndarray: 8-bit threshold map (values 0-255)
    """
    threshold_map = np.zeros((height, width), dtype=np.uint8)

    for bit in range(8):
        print(f"  Generating bit {bit} (weight {1 << bit})...")
        # Generate 50% dither pattern with unique seed for this bit
        pattern = our_method_dither(127.5, width, height, seed=base_seed + bit)
        # Convert to binary (0 or 1)
        binary = (pattern == 255).astype(np.uint8)
        # Add this bit's contribution to threshold map
        threshold_map += binary * (1 << bit)

    return threshold_map


def test_threshold_map(threshold_map: np.ndarray, gray_level: int) -> np.ndarray:
    """Apply threshold map to dither a uniform gray level."""
    return ((gray_level >= threshold_map) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Dither map experiments - generate threshold maps from error diffusion patterns"
    )
    parser.add_argument("--generate-map", action="store_true",
                        help="Generate 8-bit threshold map from 8 independent 50%% patterns")
    parser.add_argument("--generate-stacked", action="store_true",
                        help="Generate threshold map by stacking 255 gray levels")
    parser.add_argument("--generate-recursive", action="store_true",
                        help="Generate threshold map using XOR cascade approach")
    parser.add_argument("--generate-boundary", action="store_true",
                        help="Generate threshold map using boundary XOR only")
    parser.add_argument("--generate-xor-chain", action="store_true",
                        help="Generate threshold map using XOR chain degradation")
    parser.add_argument("--generate-recursive-subdiv", action="store_true",
                        help="Generate threshold map using recursive subdivision")
    parser.add_argument("--generate-and-not-or", action="store_true",
                        help="Generate threshold map using AND NOT OR logic")
    parser.add_argument("--base-pct", type=float, default=50.0,
                        help="Base pattern percentage for AND NOT OR (default: 50)")
    parser.add_argument("--or-pct", type=float, default=25.0,
                        help="OR pattern percentage for AND NOT OR (default: 25)")
    parser.add_argument("--xor-prev", action="store_true",
                        help="XOR with previous bit in AND NOT OR method")
    parser.add_argument("--analyze-map", type=str, metavar="MAP_PATH",
                        help="Analyze a threshold map with spectral comparison charts")
    parser.add_argument("--test-map", type=str, metavar="MAP_PATH",
                        help="Test a threshold map by dithering various gray levels (text output)")
    parser.add_argument("--gray", type=float, default=127.5,
                        help="Gray level for single pattern generation (default: 127.5)")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    parser.add_argument("--size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    args = parser.parse_args()

    if args.generate_map:
        # Generate 8-bit threshold map
        print(f"Generating 8-bit threshold map ({args.size}x{args.size})...")
        threshold_map = generate_8bit_threshold_map(args.size, args.size, args.seed)

        output_path = args.output or f"threshold_map_8bit_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 127, 128, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%)")

    elif args.generate_stacked:
        # Generate stacked threshold map (255 gray levels)
        print(f"Generating stacked threshold map ({args.size}x{args.size}) from 255 gray levels...")
        threshold_map = generate_stacked_threshold_map(args.size, args.size, args.seed)

        output_path = args.output or f"threshold_map_stacked_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 127, 128, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%)")

    elif args.generate_recursive_subdiv:
        # Generate recursive subdivision threshold map
        print(f"Generating recursive subdivision threshold map ({args.size}x{args.size})...")
        threshold_map = generate_recursive_subdiv_threshold_map(args.size, args.size, args.seed)

        output_path = args.output or f"threshold_map_recursive_subdiv_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 85, 127, 128, 170, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            err = white_pct - expected_pct
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%, err {err:+.1f}%)")

    elif args.generate_and_not_or:
        # Generate AND NOT OR threshold map
        print(f"Generating AND NOT OR threshold map ({args.size}x{args.size})...")
        print(f"  base_pct={args.base_pct}, or_pct={args.or_pct}, xor_prev={args.xor_prev}")
        threshold_map, bits = generate_and_not_or_threshold_map(
            args.size, args.size, args.seed,
            base_pct=args.base_pct, or_pct=args.or_pct, xor_prev=args.xor_prev,
            return_bits=True
        )

        output_path = args.output or f"threshold_map_and_not_or_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Print and save bit statistics
        print(f"\nBit plane statistics:")
        for i in range(7, -1, -1):
            white_pct = np.mean(bits[i]) * 100
            print(f"  Bit {i} (weight {1<<i:3d}): {white_pct:.1f}% white")

        # Save bit plane visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for bit in range(8):
            ax = axes[bit // 4, bit % 4]
            ax.imshow(bits[bit] * 255, cmap='gray', vmin=0, vmax=255)
            white_pct = np.mean(bits[bit]) * 100
            ax.set_title(f'Bit {bit} (weight {1<<bit})\n{white_pct:.1f}% white')
            ax.axis('off')

        xor_str = " XOR prev" if args.xor_prev else ""
        plt.suptitle(f'({args.base_pct}% AND NOT prev) OR {args.or_pct}%{xor_str}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        bits_path = Path(output_path).with_name(Path(output_path).stem + '_bits.png')
        plt.savefig(bits_path, dpi=150)
        plt.close()
        print(f"  Bit planes saved: {bits_path}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 85, 127, 128, 170, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            err = white_pct - expected_pct
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%, err {err:+.1f}%)")

    elif args.generate_xor_chain:
        # Generate XOR chain threshold map
        print(f"Generating XOR chain threshold map ({args.size}x{args.size})...")
        threshold_map = generate_xor_chain_threshold_map(args.size, args.size, args.seed)

        output_path = args.output or f"threshold_map_xor_chain_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 127, 128, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%)")

    elif args.generate_boundary:
        # Generate boundary XOR threshold map
        print(f"Generating boundary XOR threshold map ({args.size}x{args.size})...")
        threshold_map = generate_boundary_threshold_map(args.size, args.size, args.seed)

        output_path = args.output or f"threshold_map_boundary_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 127, 128, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%)")

    elif args.generate_recursive:
        # Generate recursive XOR cascade threshold map
        print(f"Generating recursive XOR cascade threshold map ({args.size}x{args.size})...")
        threshold_map = generate_recursive_threshold_map(args.size, args.size, args.seed)

        output_path = args.output or f"threshold_map_recursive_{args.size}.png"
        Image.fromarray(threshold_map, mode='L').save(output_path)

        print(f"\nSaved: {output_path}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")
        print(f"  Mean: {threshold_map.mean():.1f} (ideal: 127.5)")
        print(f"  Unique values: {len(np.unique(threshold_map))}")

        # Test at a few gray levels
        print(f"\nTesting threshold map:")
        for g in [32, 64, 127, 128, 191, 224]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:.1f}%)")

    elif args.analyze_map:
        # Analyze threshold map with spectral comparison
        threshold_map = np.array(Image.open(args.analyze_map).convert('L'))
        print(f"Analyzing threshold map: {args.analyze_map}")
        print(f"  Size: {threshold_map.shape}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")

        output_dir = Path(args.output) if args.output else Path(__file__).parent / "test_images" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        analyze_threshold_map(threshold_map, output_dir)
        print(f"\nDone! Analysis saved to {output_dir}")

    elif args.test_map:
        # Test existing threshold map
        threshold_map = np.array(Image.open(args.test_map).convert('L'))
        print(f"Testing threshold map: {args.test_map}")
        print(f"  Size: {threshold_map.shape}")
        print(f"  Value range: {threshold_map.min()} - {threshold_map.max()}")

        for g in [16, 32, 64, 85, 127, 128, 170, 191, 224, 240]:
            result = test_threshold_map(threshold_map, g)
            white_pct = np.mean(result == 255) * 100
            expected_pct = g / 255 * 100
            error = white_pct - expected_pct
            print(f"  Gray {g:3d}: {white_pct:5.1f}% white (expected {expected_pct:5.1f}%, error {error:+.1f}%)")

            # Optionally save test output
            if args.output:
                test_path = f"{Path(args.output).stem}_gray{g:03d}.png"
                Image.fromarray(result, mode='L').save(test_path)

    else:
        # Default: generate single dither pattern
        print(f"Generating dither pattern at gray level {args.gray}...")
        output = our_method_dither(args.gray, args.size, args.size, args.seed)

        output_path = args.output or f"dither_pattern_{args.gray:.1f}.png"
        Image.fromarray(output, mode='L').save(output_path)

        white_pct = np.mean(output == 255) * 100
        expected_pct = args.gray / 255 * 100
        print(f"Saved: {output_path}")
        print(f"  Size: {args.size}x{args.size}")
        print(f"  Gray level: {args.gray} ({expected_pct:.1f}% expected)")
        print(f"  White pixels: {white_pct:.2f}%")


if __name__ == "__main__":
    main()
