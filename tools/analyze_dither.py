#!/usr/bin/env python3
"""
Fourier analysis of dithered images.

Generates visualizations similar to Zhou-Fang paper Figure 5:
- 2D FFT power spectrum (shows isotropy/anisotropy)
- Segmented radially averaged power spectrum (H/D/V curves)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from numpy.fft import fft2, fftshift
import argparse


def load_image(path: Path) -> np.ndarray:
    """Load image as grayscale float array."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32)


def compute_power_spectrum(img: np.ndarray) -> np.ndarray:
    """Compute 2D power spectrum with log scaling."""
    # Center around zero
    centered = img - img.mean()
    # 2D FFT, shift zero-freq to center
    spectrum = fftshift(fft2(centered))
    # Power spectrum with log scaling
    power = np.abs(spectrum) ** 2
    return np.log1p(power)


def compute_segmented_radial_power(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute segmented radially averaged power spectrum.

    Returns (frequencies, horizontal, diagonal, vertical) power curves.
    - Horizontal: 0° ± 22.5° (red in plots)
    - Diagonal: 45° ± 22.5° (green in plots)
    - Vertical: 90° ± 22.5° (blue in plots)
    """
    h, w = img.shape
    centered = img - img.mean()
    spectrum = np.abs(fftshift(fft2(centered))) ** 2

    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)

    # Pre-compute coordinates
    y_coords, x_coords = np.ogrid[:h, :w]
    dy = y_coords - cy
    dx = x_coords - cx
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))  # 0 to 180

    # Accumulate power by radius and angle segment
    horizontal = []
    diagonal = []
    vertical = []

    for r in range(1, max_radius):
        # Pixels at this radius (within 0.5 pixel tolerance)
        ring_mask = np.abs(distances - r) < 0.5

        if not ring_mask.any():
            horizontal.append(0)
            diagonal.append(0)
            vertical.append(0)
            continue

        ring_angles = angles[ring_mask]
        ring_power = spectrum[ring_mask]

        # Horizontal: 0° ± 22.5° or 180° ± 22.5°
        h_mask = (ring_angles < 22.5) | (ring_angles > 157.5)
        # Vertical: 90° ± 22.5°
        v_mask = (ring_angles > 67.5) & (ring_angles < 112.5)
        # Diagonal: 45° ± 22.5° or 135° ± 22.5°
        d_mask = ((ring_angles > 22.5) & (ring_angles < 67.5)) | \
                 ((ring_angles > 112.5) & (ring_angles < 157.5))

        h_power = ring_power[h_mask].mean() if h_mask.any() else 0
        d_power = ring_power[d_mask].mean() if d_mask.any() else 0
        v_power = ring_power[v_mask].mean() if v_mask.any() else 0

        horizontal.append(h_power)
        diagonal.append(d_power)
        vertical.append(v_power)

    freqs = np.arange(1, max_radius)
    return freqs, np.array(horizontal), np.array(diagonal), np.array(vertical)


def plot_analysis(img: np.ndarray, title: str, output_path: Path):
    """Generate 3-panel analysis figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Dithered image
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Halftone Result')
    axes[0].axis('off')

    # Panel 2: FFT power spectrum
    spectrum = compute_power_spectrum(img)
    axes[1].imshow(spectrum, cmap='gray')
    axes[1].set_title('Frequency Spectrum')
    axes[1].axis('off')

    # Panel 3: Segmented radial power
    freqs, h, d, v = compute_segmented_radial_power(img)

    # Convert to log scale (dB) for better visualization
    h_db = 10 * np.log10(h + 1e-10)
    d_db = 10 * np.log10(d + 1e-10)
    v_db = 10 * np.log10(v + 1e-10)

    axes[2].plot(freqs, h_db, 'r-', label='Horizontal (0°)', alpha=0.8)
    axes[2].plot(freqs, d_db, 'g-', label='Diagonal (45°)', alpha=0.8)
    axes[2].plot(freqs, v_db, 'b-', label='Vertical (90°)', alpha=0.8)
    axes[2].set_xlabel('Frequency (pixels⁻¹)')
    axes[2].set_ylabel('Power (dB)')
    axes[2].set_title('Segmented Radial Power')
    axes[2].legend(loc='upper right')
    axes[2].set_xlim(0, len(freqs))
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(images: dict[str, np.ndarray], title: str, output_path: Path):
    """Generate comparison figure for multiple dither methods."""
    n_methods = len(images)
    fig, axes = plt.subplots(3, n_methods, figsize=(4 * n_methods, 12))

    if n_methods == 1:
        axes = axes.reshape(3, 1)

    for col, (method_name, img) in enumerate(images.items()):
        # Row 1: Dithered image
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(method_name, fontsize=10)
        axes[0, col].axis('off')

        # Row 2: FFT spectrum
        spectrum = compute_power_spectrum(img)
        axes[1, col].imshow(spectrum, cmap='gray')
        axes[1, col].axis('off')

        # Row 3: Radial power curves (log scale)
        freqs, h, d, v = compute_segmented_radial_power(img)

        # Convert to log scale (dB)
        h_db = 10 * np.log10(h + 1e-10)
        d_db = 10 * np.log10(d + 1e-10)
        v_db = 10 * np.log10(v + 1e-10)

        axes[2, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[2, col].set_xlim(0, len(freqs))
        axes[2, col].grid(True, alpha=0.3)
        if col == 0:
            axes[2, col].set_ylabel('Power (dB)')
        if col == n_methods - 1:
            axes[2, col].legend(loc='upper right', fontsize=8)

    # Row labels
    fig.text(0.02, 0.83, 'Halftone', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.5, 'Spectrum', va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.17, 'Radial', va='center', rotation='vertical', fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_single(input_path: Path, output_path: Path):
    """Analyze a single dithered image."""
    img = load_image(input_path)
    title = input_path.stem
    plot_analysis(img, title, output_path)
    print(f"  {output_path.name}")


def analyze_comparison(base_dir: Path, image_name: str, output_dir: Path):
    """Compare all dither methods for a single source image."""
    methods = [
        'fs-standard', 'fs-serpentine',
        'jjn-standard', 'jjn-serpentine',
        'boon-standard', 'boon-serpentine',
        'ostro-standard', 'ostro-serpentine',
        'zhou-fang-standard', 'zhou-fang-serpentine',
    ]

    images = {}

    # First, add the original source image
    source_file = base_dir.parent / f"{image_name}.png"
    if source_file.exists():
        images['Original'] = load_image(source_file)

    for method in methods:
        # Find the dithered file
        method_dir = base_dir / method
        # File naming: {image_name}_{method}.png
        dithered_file = method_dir / f"{image_name}_{method}.png"
        if dithered_file.exists():
            images[method] = load_image(dithered_file)

    if not images:
        print(f"  No dithered images found for {image_name}")
        return

    output_path = output_dir / f"{image_name}_comparison.png"
    plot_comparison(images, f"Dither Comparison: {image_name}", output_path)
    print(f"  {output_path.name}")


def analyze_serpentine_only(base_dir: Path, image_name: str, output_dir: Path):
    """Compare serpentine variants only (cleaner comparison)."""
    methods = [
        ('fs-serpentine', 'Floyd-Steinberg'),
        ('jjn-serpentine', 'Jarvis-Judice-Ninke'),
        ('boon-serpentine', 'Our Method'),
        ('ostro-serpentine', 'Ostromoukhov'),
        ('zhou-fang-serpentine', 'Zhou-Fang'),
    ]

    images = {}

    # First, add the original source image
    source_file = base_dir.parent / f"{image_name}.png"
    if source_file.exists():
        images['Original'] = load_image(source_file)

    for method_key, method_name in methods:
        method_dir = base_dir / method_key
        dithered_file = method_dir / f"{image_name}_{method_key}.png"
        if dithered_file.exists():
            images[method_name] = load_image(dithered_file)

    if not images:
        print(f"  No dithered images found for {image_name}")
        return

    output_path = output_dir / f"{image_name}_serpentine.png"
    plot_comparison(images, f"Serpentine Dither Comparison: {image_name}", output_path)
    print(f"  {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dithered images with Fourier methods')
    parser.add_argument('--input', '-i', type=Path, help='Single input image to analyze')
    parser.add_argument('--output', '-o', type=Path, help='Output path for single image analysis')
    parser.add_argument('--compare', '-c', action='store_true', help='Generate comparison charts for all test images')
    parser.add_argument('--serpentine', '-s', action='store_true', help='Compare serpentine variants only')
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "test_images"
    dithered_dir = base_dir / "dithered"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    if args.input:
        # Single image analysis
        output_path = args.output or output_dir / f"{args.input.stem}_analysis.png"
        analyze_single(args.input, output_path)
    elif args.compare or args.serpentine:
        # Find all source images
        source_images = [p.stem for p in base_dir.glob("*.png")]
        print(f"Analyzing {len(source_images)} images...")

        for image_name in sorted(source_images):
            if args.serpentine:
                analyze_serpentine_only(dithered_dir, image_name, output_dir)
            else:
                analyze_comparison(dithered_dir, image_name, output_dir)

        print(f"\nDone! Analysis saved to {output_dir}")
    else:
        # Default: serpentine comparison for key test images
        key_images = ['gray_064', 'gray_085', 'gray_127', 'david', 'gradient']
        print("Generating analysis for key test images...")
        for image_name in key_images:
            analyze_serpentine_only(dithered_dir, image_name, output_dir)
        print(f"\nDone! Analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
