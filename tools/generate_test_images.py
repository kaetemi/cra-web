#!/usr/bin/env python3
"""Generate test images for dithering experiments."""

import numpy as np
from pathlib import Path
import urllib.request
import shutil

# PIL for image output
from PIL import Image


def save_gray(arr: np.ndarray, path: Path):
    """Save a 2D float array as grayscale PNG."""
    img = Image.fromarray(arr.astype(np.uint8), mode='L')
    img.save(path)
    print(f"  {path.name}")


def generate_pathological_grays(output_dir: Path):
    """Generate 256x256 solid gray images at pathological levels.

    These are values that stress dithering algorithms:
    - Endpoints (0, 255)
    - Mid-gray (127, 128)
    - Values that fall exactly between quantization levels
    - Values near but not at level boundaries
    """
    print("Generating pathological gray levels (256x256)...")

    size = (256, 256)

    # Key pathological values
    levels = [
        0,      # Black
        1,      # Near-black (tests error diffusion at boundary)
        127,    # Just below mid
        128,    # Just above mid
        254,    # Near-white
        255,    # White
        # Values that fall between 2-bit levels (0, 85, 170, 255)
        42,     # Midpoint between 0 and 85
        43,     # Just above midpoint
        127,    # Between 85 and 170
        212,    # Between 170 and 255
        213,    # Just above
        # Values that fall between 3-bit levels (0, 36, 73, 109, 146, 182, 219, 255)
        18,     # Between 0 and 36
        54,     # Between 36 and 73
        91,     # Between 73 and 109
        # 1-bit dithering stress tests (should produce 50/50 pattern)
        127,    # Mid
        85,     # 1/3
        170,    # 2/3
        64,     # 1/4
        191,    # 3/4
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_levels = []
    for l in levels:
        if l not in seen:
            seen.add(l)
            unique_levels.append(l)

    for level in unique_levels:
        arr = np.full(size, level, dtype=np.float32)
        save_gray(arr, output_dir / f"gray_{level:03d}.png")


def generate_continuous_ramp(output_dir: Path):
    """Generate continuous 0-255 gray ramp, 64px high, 4096px wide."""
    print("Generating continuous ramp (64x4096)...")

    height = 64
    width = 4096

    # Create horizontal gradient from 0 to 255
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    arr = np.tile(ramp, (height, 1))

    save_gray(arr, output_dir / "ramp_continuous.png")


def generate_step_ramp(output_dir: Path):
    """Generate step ramp jumping 32 values, 64px high, 1024px wide."""
    print("Generating step ramp (64x1024, 32-value steps)...")

    height = 64
    width = 1024

    # 1024px / 32 steps = 32px per step, values 0, 32, 64, 96, 128, 160, 192, 224
    # But we want 0-255 range... let's do 8 steps of 32 values each
    # Actually, jumping 32 each time: 0, 32, 64, 96, 128, 160, 192, 224 = 8 values
    # 1024 / 8 = 128px per step

    num_steps = 8
    step_width = width // num_steps

    arr = np.zeros((height, width), dtype=np.float32)
    for i in range(num_steps):
        value = i * 32
        x_start = i * step_width
        x_end = (i + 1) * step_width
        arr[:, x_start:x_end] = value

    save_gray(arr, output_dir / "ramp_step_32.png")

    # Also generate a finer step ramp with 16-value jumps
    print("Generating step ramp (64x1024, 16-value steps)...")
    num_steps = 16
    step_width = width // num_steps

    arr = np.zeros((height, width), dtype=np.float32)
    for i in range(num_steps):
        value = min(i * 16, 255)
        x_start = i * step_width
        x_end = (i + 1) * step_width
        arr[:, x_start:x_end] = value

    save_gray(arr, output_dir / "ramp_step_16.png")


def download_reference_images(output_dir: Path):
    """Download reference images from structure-aware-dithering repo."""
    print("Downloading reference images...")

    urls = [
        ("https://raw.githubusercontent.com/dalpil/structure-aware-dithering/main/examples/gradient-steps/original.png", "gradient_steps.png"),
        ("https://raw.githubusercontent.com/dalpil/structure-aware-dithering/main/examples/gradient/original.png", "gradient.png"),
        ("https://raw.githubusercontent.com/dalpil/structure-aware-dithering/main/examples/david-original.png", "david.png"),
    ]

    for url, filename in urls:
        output_path = output_dir / filename
        if output_path.exists():
            print(f"  {filename} (already exists)")
            continue
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with open(output_path, 'wb') as f:
                    shutil.copyfileobj(response, f)
            print(f"  {filename}")
        except Exception as e:
            print(f"  {filename} FAILED: {e}")


def main():
    output_dir = Path(__file__).parent / "test_images" / "sources"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    generate_pathological_grays(output_dir)
    print()
    generate_continuous_ramp(output_dir)
    print()
    generate_step_ramp(output_dir)
    print()
    download_reference_images(output_dir)

    print(f"\nDone! Generated images in {output_dir}")


if __name__ == "__main__":
    main()
