#!/usr/bin/env python3
"""
Generate 1-bit noise images using various deterministic RNG methods.
Each pixel = (rng_output & 1) * 255
"""

import numpy as np
from pathlib import Path
from PIL import Image


def wang_hash(seed: np.uint32) -> np.uint32:
    """Wang hash - used in our mixed dithering."""
    seed = np.uint32(seed)
    seed = (seed ^ np.uint32(61)) ^ (seed >> np.uint32(16))
    seed = seed + (seed << np.uint32(3))
    seed = seed ^ (seed >> np.uint32(4))
    seed = seed * np.uint32(0x27d4eb2d)
    seed = seed ^ (seed >> np.uint32(15))
    return seed


def xorshift32(seed: np.uint32) -> np.uint32:
    """Xorshift32 - simple and fast."""
    seed = np.uint32(seed)
    seed ^= seed << np.uint32(13)
    seed ^= seed >> np.uint32(17)
    seed ^= seed << np.uint32(5)
    return seed


def lcg(seed: np.uint32) -> np.uint32:
    """Linear Congruential Generator (MINSTD)."""
    # Using MINSTD parameters
    return np.uint32((np.uint64(seed) * np.uint64(48271)) % np.uint64(0x7fffffff))


def murmur_finalizer(seed: np.uint32) -> np.uint32:
    """Murmur3 finalizer - good avalanche properties."""
    seed = np.uint32(seed)
    seed ^= seed >> np.uint32(16)
    seed = seed * np.uint32(0x85ebca6b)
    seed ^= seed >> np.uint32(13)
    seed = seed * np.uint32(0xc2b2ae35)
    seed ^= seed >> np.uint32(16)
    return seed


def pcg_hash(seed: np.uint32) -> np.uint32:
    """PCG-inspired hash."""
    seed = np.uint32(seed)
    state = seed * np.uint32(747796405) + np.uint32(2891336453)
    word = ((state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state) * np.uint32(277803737)
    return (word >> np.uint32(22)) ^ word


def splitmix32(seed: np.uint32) -> np.uint32:
    """SplitMix32 - derived from SplitMix64."""
    seed = np.uint32(seed)
    seed = seed + np.uint32(0x9e3779b9)
    seed = (seed ^ (seed >> np.uint32(15))) * np.uint32(0x85ebca6b)
    seed = (seed ^ (seed >> np.uint32(13))) * np.uint32(0xc2b2ae35)
    return seed ^ (seed >> np.uint32(16))


def double_wang_hash(seed: np.uint32) -> np.uint32:
    """Double Wang hash - apply Wang hash twice."""
    return wang_hash(wang_hash(seed))


def triple32(seed: np.uint32) -> np.uint32:
    """Triple32 - popular in GPU shaders (lowbias32 variant)."""
    seed = np.uint32(seed)
    seed ^= seed >> np.uint32(17)
    seed = seed * np.uint32(0xed5ad4bb)
    seed ^= seed >> np.uint32(11)
    seed = seed * np.uint32(0xac4c1b51)
    seed ^= seed >> np.uint32(15)
    seed = seed * np.uint32(0x31848bab)
    seed ^= seed >> np.uint32(14)
    return seed


def lowbias32_old(seed: np.uint32) -> np.uint32:
    """Original lowbias32 (bias 0.174) - kept for comparison."""
    seed = np.uint32(seed)
    seed ^= seed >> np.uint32(16)
    seed = seed * np.uint32(0x7feb352d)
    seed ^= seed >> np.uint32(15)
    seed = seed * np.uint32(0x846ca68b)
    seed ^= seed >> np.uint32(16)
    return seed


def lowbias32(seed: np.uint32) -> np.uint32:
    """Improved lowbias32 (bias 0.107) - better spectral properties."""
    seed = np.uint32(seed)
    seed ^= seed >> np.uint32(16)
    seed = seed * np.uint32(0x21f0aaad)
    seed ^= seed >> np.uint32(15)
    seed = seed * np.uint32(0x735a2d97)
    seed ^= seed >> np.uint32(15)
    return seed


def xxhash32_avalanche(seed: np.uint32) -> np.uint32:
    """xxHash32 avalanche/finalizer - very GPU friendly."""
    seed = np.uint32(seed)
    seed ^= seed >> np.uint32(15)
    seed = seed * np.uint32(0x85ebca77)
    seed ^= seed >> np.uint32(13)
    seed = seed * np.uint32(0xc2b2ae3d)
    seed ^= seed >> np.uint32(16)
    return seed


def iqint1(seed: np.uint32) -> np.uint32:
    """IQ's integer hash 1 - used in Shadertoy, very simple."""
    seed = np.uint32(seed)
    seed = np.uint32(1103515245) * seed + np.uint32(12345)
    seed = seed ^ (seed >> np.uint32(16))
    return seed


def iqint3(seed: np.uint32) -> np.uint32:
    """IQ's integer hash 3 - better quality, still GPU friendly."""
    seed = np.uint32(seed)
    seed = (seed << np.uint32(13)) ^ seed
    seed = seed * (seed * seed * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return seed


def generate_noise_image(width: int, height: int, hash_func, name: str, use_high_bit: bool = False) -> np.ndarray:
    """Generate 1-bit noise using hash(x ^ (y << 16) ^ seed)."""
    img = np.zeros((height, width), dtype=np.uint8)
    seed = np.uint32(12345)
    bit_mask = np.uint32(0x80000000) if use_high_bit else np.uint32(1)

    for y in range(height):
        for x in range(width):
            # Combine coordinates into seed
            coord_seed = np.uint32(x) ^ (np.uint32(y) << np.uint32(16)) ^ seed
            h = hash_func(coord_seed)
            img[y, x] = 255 if (h & bit_mask) else 0

    return img


def generate_noise_sequential(width: int, height: int, hash_func, name: str, use_high_bit: bool = False) -> np.ndarray:
    """Generate 1-bit noise using sequential hashing (hash of previous)."""
    img = np.zeros((height, width), dtype=np.uint8)
    state = np.uint32(12345)
    bit_mask = np.uint32(0x80000000) if use_high_bit else np.uint32(1)

    for y in range(height):
        for x in range(width):
            state = hash_func(state)
            img[y, x] = 255 if (state & bit_mask) else 0

    return img


def save_image(img: np.ndarray, path: Path):
    """Save as PNG."""
    Image.fromarray(img, mode='L').save(path)
    print(f"  {path.name}")


def main():
    output_dir = Path(__file__).parent / "test_images" / "rng_noise"
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 256, 256

    # Hash functions to test (coordinate-based)
    # GPU-friendly hashes (no division, simple ops)
    hash_funcs = [
        (wang_hash, "wang_hash"),
        (double_wang_hash, "double_wang"),
        (triple32, "triple32"),
        (lowbias32, "lowbias32"),
        (lowbias32_old, "lowbias32_old"),
        (xxhash32_avalanche, "xxhash32"),
        (iqint1, "iqint1"),
        (iqint3, "iqint3"),
        (murmur_finalizer, "murmur3"),
        (pcg_hash, "pcg"),
        (splitmix32, "splitmix32"),
        # Non-GPU-friendly for comparison
        (xorshift32, "xorshift32"),
        (lcg, "lcg"),
    ]

    print(f"Generating {width}x{height} 1-bit noise images...")
    print(f"Output: {output_dir}\n")

    print("Coordinate-based (hash(x ^ y<<16 ^ seed) & 1):")
    for func, name in hash_funcs:
        img = generate_noise_image(width, height, func, name)
        save_image(img, output_dir / f"{name}_coord.png")

    print("\nSequential (hash(prev_state) & 1):")
    for func, name in hash_funcs:
        img = generate_noise_sequential(width, height, func, name)
        save_image(img, output_dir / f"{name}_seq.png")

    print("\nCoordinate-based high bit (hash(x ^ y<<16 ^ seed) >> 31):")
    for func, name in hash_funcs:
        img = generate_noise_image(width, height, func, name, use_high_bit=True)
        save_image(img, output_dir / f"{name}_coord_highbit.png")

    print("\nSequential high bit (hash(prev_state) >> 31):")
    for func, name in hash_funcs:
        img = generate_noise_sequential(width, height, func, name, use_high_bit=True)
        save_image(img, output_dir / f"{name}_seq_highbit.png")

    # Also generate reference white noise using numpy
    print("\nReference:")
    np.random.seed(12345)
    white_noise = (np.random.randint(0, 2, (height, width)) * 255).astype(np.uint8)
    save_image(white_noise, output_dir / "numpy_random.png")

    # Blue noise reference (if we had one - for now skip)

    print(f"\nDone! Generated {len(hash_funcs) * 4 + 1} images")


if __name__ == "__main__":
    main()
