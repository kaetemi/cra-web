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
    - Frequencies: in cycles/pixel (0 = DC, 0.5 = Nyquist)
    - Horizontal: 0° ± 22.5° (red in plots)
    - Diagonal: 45° ± 22.5° (green in plots)
    - Vertical: 90° ± 22.5° (blue in plots)
    """
    h, w = img.shape
    centered = img - img.mean()
    spectrum = np.abs(fftshift(fft2(centered))) ** 2

    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)
    img_size = min(h, w)  # for normalizing to cycles/pixel

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

    # Convert radius to cycles/pixel: radius / img_size
    # Range: 0 to 0.5 (Nyquist limit)
    freqs = np.arange(1, max_radius) / img_size
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

    # Convert to dB scale for power (linear frequency axis)
    h_db = 10 * np.log10(h + 1e-10)
    d_db = 10 * np.log10(d + 1e-10)
    v_db = 10 * np.log10(v + 1e-10)

    axes[2].plot(freqs, h_db, 'r-', label='Horizontal (0°)', alpha=0.8)
    axes[2].plot(freqs, d_db, 'g-', label='Diagonal (45°)', alpha=0.8)
    axes[2].plot(freqs, v_db, 'b-', label='Vertical (90°)', alpha=0.8)
    axes[2].set_xlabel('Spatial Frequency (cycles/pixel)')
    axes[2].set_ylabel('Power (dB)')
    axes[2].set_title('Segmented Radial Power')
    axes[2].legend(loc='upper right')
    axes[2].set_xlim(0, 0.5)
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

    # First pass: compute all power spectra and find global min/max for dithered images
    all_power_db = {}
    global_min_db = float('inf')
    global_max_db = float('-inf')

    for method_name, img in images.items():
        freqs, h, d, v = compute_segmented_radial_power(img)
        h_db = 10 * np.log10(h + 1e-10)
        d_db = 10 * np.log10(d + 1e-10)
        v_db = 10 * np.log10(v + 1e-10)
        all_power_db[method_name] = (freqs, h_db, d_db, v_db)

        # Only include non-Original in global scale
        if method_name != 'Original':
            global_min_db = min(global_min_db, h_db.min(), d_db.min(), v_db.min())
            global_max_db = max(global_max_db, h_db.max(), d_db.max(), v_db.max())

    # Add some padding to the range
    if global_min_db < float('inf') and global_max_db > float('-inf'):
        y_padding = (global_max_db - global_min_db) * 0.05
        global_min_db -= y_padding
        global_max_db += y_padding

    for col, (method_name, img) in enumerate(images.items()):
        # Row 1: Dithered image
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(method_name, fontsize=10)
        axes[0, col].axis('off')

        # Row 2: FFT spectrum
        spectrum = compute_power_spectrum(img)
        axes[1, col].imshow(spectrum, cmap='gray')
        axes[1, col].axis('off')

        # Row 3: Radial power curves (linear freq, dB power)
        freqs, h_db, d_db, v_db = all_power_db[method_name]

        axes[2, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[2, col].set_xlim(0, 0.5)
        axes[2, col].set_xlabel('cycles/px')
        axes[2, col].grid(True, alpha=0.3)

        # Use global scale for dithered images, auto scale for Original
        if method_name != 'Original' and global_min_db < float('inf'):
            axes[2, col].set_ylim(global_min_db, global_max_db)

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

    # First, add the original source image (from sources/ subfolder)
    source_file = base_dir.parent / "sources" / f"{image_name}.png"
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


def get_blue_noise_reference(base_dir: Path, image_name: str) -> np.ndarray | None:
    """Generate ideal blue noise pattern for gray_XXX images.

    Thresholds the blue noise dither array at the appropriate gray level.
    Returns None for non-gray images.
    """
    # Check if this is a gray_XXX image
    if not image_name.startswith('gray_'):
        return None

    try:
        gray_level = int(image_name.split('_')[1])
    except (IndexError, ValueError):
        return None

    # Load blue noise dither array (reference image in test_images/, not processed)
    # base_dir is dithered_dir (test_images/dithered), so go up one level
    blue_noise_path = base_dir.parent / 'blue_noise_256.png'
    if not blue_noise_path.exists():
        return None

    blue_noise = load_image(blue_noise_path)

    # Threshold at gray level to get ideal pattern at this density
    # blue_noise values are 0-255, representing dither thresholds
    # Pixels with threshold < gray_level become white (255)
    ideal_pattern = np.where(blue_noise < gray_level, 255.0, 0.0)

    return ideal_pattern


def get_coin_flip_reference(image_name: str, size: tuple = (256, 256)) -> np.ndarray | None:
    """Generate white noise (coin flip) pattern for gray_XXX images.

    Creates random noise at the appropriate density for comparison.
    Returns None for non-gray images.
    """
    # Check if this is a gray_XXX image
    if not image_name.startswith('gray_'):
        return None

    try:
        gray_level = int(image_name.split('_')[1])
    except (IndexError, ValueError):
        return None

    # Use fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # Probability of white pixel = gray_level / 255
    probability = gray_level / 255.0
    coin_flips = rng.random(size) < probability

    return np.where(coin_flips, 255.0, 0.0)


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

    # First, add the original source image (from sources/ subfolder)
    source_file = base_dir.parent / "sources" / f"{image_name}.png"
    if source_file.exists():
        images['Original'] = load_image(source_file)

    # Add reference patterns for gray images
    coin_flip_ref = get_coin_flip_reference(image_name)
    if coin_flip_ref is not None:
        images['Coin Flip (white noise)'] = coin_flip_ref

    blue_noise_ref = get_blue_noise_reference(base_dir, image_name)
    if blue_noise_ref is not None:
        images['Blue Noise Reference (Christoph Peters)'] = blue_noise_ref

    for method_key, method_name in methods:
        method_dir = base_dir / method_key
        dithered_file = method_dir / f"{image_name}_{method_key}.png"
        if dithered_file.exists():
            images[method_name] = load_image(dithered_file)

    if not images:
        print(f"  No dithered images found for {image_name}")
        return

    output_path = output_dir / f"{image_name}_serpentine.png"
    plot_comparison(images, f"Dither Comparison: {image_name}", output_path)
    print(f"  {output_path.name}")


def analyze_hash_comparison(base_dir: Path, image_name: str, output_dir: Path):
    """Compare boon hash function variants: lowbias32 (new) vs lowbias32_old vs wang_hash."""
    # Standard variants
    standard_methods = [
        ('boon-standard', 'Our Method (lowbias32)'),
        ('boon-lowbias', 'Our Method (lowbias32_old)'),
        ('boon-wanghash', 'Our Method (wang hash)'),
    ]
    # Serpentine variants
    serpentine_methods = [
        ('boon-serpentine', 'Our Method Serp. (lowbias32)'),
        ('boon-lowbias-serpentine', 'Our Method Serp. (lowbias32_old)'),
        ('boon-wanghash-serpentine', 'Our Method Serp. (wang hash)'),
    ]

    images = {}

    # First, add the original source image (from sources/ subfolder)
    source_file = base_dir.parent / "sources" / f"{image_name}.png"
    if source_file.exists():
        images['Original'] = load_image(source_file)

    # Add standard variants
    for method_key, method_name in standard_methods:
        method_dir = base_dir / method_key
        dithered_file = method_dir / f"{image_name}_{method_key}.png"
        if dithered_file.exists():
            images[method_name] = load_image(dithered_file)

    # Add serpentine variants
    for method_key, method_name in serpentine_methods:
        method_dir = base_dir / method_key
        dithered_file = method_dir / f"{image_name}_{method_key}.png"
        if dithered_file.exists():
            images[method_name] = load_image(dithered_file)

    if len(images) <= 1:  # Only original or nothing
        print(f"  No boon dithered images found for {image_name}")
        return

    output_path = output_dir / f"{image_name}_hash_comparison.png"
    plot_comparison(images, f"Hash Function Comparison: {image_name}", output_path)
    print(f"  {output_path.name}")


def plot_rng_comparison(images: dict, title: str, output_path: Path, excluded_from_scale: set):
    """Generate comparison figure for RNG methods with fixed scale for good hashes.

    Args:
        images: dict of method_name -> image array
        title: plot title
        output_path: where to save
        excluded_from_scale: set of method names that should use auto-scale (e.g., Wang, IQ)
    """
    n_methods = len(images)
    fig, axes = plt.subplots(3, n_methods, figsize=(4 * n_methods, 12))

    if n_methods == 1:
        axes = axes.reshape(3, 1)

    # Compute all power spectra
    all_power_db = {}
    global_min_db = float('inf')
    global_max_db = float('-inf')
    for method_name, img in images.items():
        freqs, h, d, v = compute_segmented_radial_power(img)
        h_db = 10 * np.log10(h + 1e-10)
        d_db = 10 * np.log10(d + 1e-10)
        v_db = 10 * np.log10(v + 1e-10)
        all_power_db[method_name] = (freqs, h_db, d_db, v_db)
        # Track global range for methods not excluded
        if method_name not in excluded_from_scale:
            global_min_db = min(global_min_db, h_db.min(), d_db.min(), v_db.min())
            global_max_db = max(global_max_db, h_db.max(), d_db.max(), v_db.max())

    # Add padding
    if global_min_db < float('inf') and global_max_db > float('-inf'):
        y_padding = (global_max_db - global_min_db) * 0.05
        global_min_db -= y_padding
        global_max_db += y_padding

    for col, (method_name, img) in enumerate(images.items()):
        # Row 1: Noise image
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(method_name, fontsize=10)
        axes[0, col].axis('off')

        # Row 2: FFT spectrum
        spectrum = compute_power_spectrum(img)
        axes[1, col].imshow(spectrum, cmap='gray')
        axes[1, col].axis('off')

        # Row 3: Radial power curves (linear freq, dB power)
        freqs, h_db, d_db, v_db = all_power_db[method_name]

        axes[2, col].plot(freqs, h_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, d_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[2, col].plot(freqs, v_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[2, col].set_xlim(0, 0.5)
        axes[2, col].set_xlabel('cycles/px')
        axes[2, col].grid(True, alpha=0.3)

        # Use global scale unless excluded
        if method_name not in excluded_from_scale and global_min_db < float('inf'):
            axes[2, col].set_ylim(global_min_db, global_max_db)

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


def analyze_rng_noise(base_dir: Path, output_dir: Path):
    """Compare RNG noise generators."""
    rng_dir = base_dir / "rng_noise"
    if not rng_dir.exists():
        print(f"RNG noise directory not found: {rng_dir}")
        return

    # Methods to exclude from fixed 75-100 scale (they have different characteristics)
    excluded_from_scale = {'Wang', 'IQ Int1', 'IQ Int3'}

    # GPU-friendly coordinate-based methods
    gpu_coord_methods = [
        ('wang_hash_coord.png', 'Wang'),
        ('double_wang_coord.png', 'Double Wang'),
        ('triple32_coord.png', 'Triple32'),
        ('lowbias32_coord.png', 'Lowbias32'),
        ('lowbias32_old_coord.png', 'Lowbias32_old'),
        ('xxhash32_coord.png', 'xxHash32'),
        ('iqint1_coord.png', 'IQ Int1'),
        ('iqint3_coord.png', 'IQ Int3'),
    ]

    images = {}
    for filename, label in gpu_coord_methods:
        path = rng_dir / filename
        if path.exists():
            images[label] = load_image(path)

    if images:
        output_path = output_dir / "rng_noise_gpu_coord.png"
        plot_rng_comparison(images, "GPU-Friendly RNG (Coordinate-based)", output_path, excluded_from_scale)
        print(f"  {output_path.name}")

    # Other coordinate-based methods
    other_coord_methods = [
        ('murmur3_coord.png', 'Murmur3'),
        ('pcg_coord.png', 'PCG'),
        ('splitmix32_coord.png', 'SplitMix32'),
        ('xorshift32_coord.png', 'Xorshift32'),
        ('lcg_coord.png', 'LCG'),
        ('numpy_random.png', 'NumPy'),
    ]

    images = {}
    for filename, label in other_coord_methods:
        path = rng_dir / filename
        if path.exists():
            images[label] = load_image(path)

    if images:
        output_path = output_dir / "rng_noise_other_coord.png"
        plot_comparison(images, "Other RNG (Coordinate-based)", output_path)
        print(f"  {output_path.name}")

    # Sequential methods (GPU-friendly)
    gpu_seq_methods = [
        ('wang_hash_seq.png', 'Wang'),
        ('double_wang_seq.png', 'Double Wang'),
        ('triple32_seq.png', 'Triple32'),
        ('lowbias32_seq.png', 'Lowbias32'),
        ('lowbias32_old_seq.png', 'Lowbias32_old'),
        ('xxhash32_seq.png', 'xxHash32'),
        ('iqint1_seq.png', 'IQ Int1'),
        ('iqint3_seq.png', 'IQ Int3'),
    ]

    images = {}
    for filename, label in gpu_seq_methods:
        path = rng_dir / filename
        if path.exists():
            images[label] = load_image(path)

    if images:
        output_path = output_dir / "rng_noise_gpu_seq.png"
        plot_rng_comparison(images, "GPU-Friendly RNG (Sequential)", output_path, excluded_from_scale)
        print(f"  {output_path.name}")


def analyze_blue_kernel_experiment(base_dir: Path, output_dir: Path):
    """
    Experimental: Compare standard dithering vs blue-noise kernel selection at various recursion depths.

    Standard: hash(x,y) selects FS vs JJN kernel (white noise selection)
    Blue-kernel d=1: Pre-dithered 50% pattern (using hash) selects kernel
    Blue-kernel d=2: Pre-dithered 50% pattern (using d=1) selects kernel
    Blue-kernel d=3: Pre-dithered 50% pattern (using d=2) selects kernel

    Tests at multiple gray levels to see if blue noise kernel selection improves results.
    """
    import sys
    sys.path.insert(0, str(base_dir.parent / 'tools'))
    from our_method_dither import (
        our_method_dither, our_method_dither_with_blue_noise_kernel,
        our_method_dither_fs_sierra, our_method_dither_fs_sierra_lite,
        our_method_dither_fs_stucki, our_method_dither_stucki_sierra,
        our_method_dither_jjn_stucki, our_method_dither_jjn_sierra
    )

    size = 256
    gray_levels = [64, 85, 127, 170, 191]  # Various densities

    for gray_level in gray_levels:
        images = {}

        # Standard method (hash-based kernel selection)
        standard = our_method_dither(float(gray_level), size, size, seed=0)
        images['FS/JJN'] = standard.astype(np.float32)

        # FS/Stucki alternation
        fs_stucki = our_method_dither_fs_stucki(float(gray_level), size, size, seed=0)
        images['FS/Stucki'] = fs_stucki.astype(np.float32)

        # FS/Sierra (full) alternation
        fs_sierra = our_method_dither_fs_sierra(float(gray_level), size, size, seed=0)
        images['FS/Sierra'] = fs_sierra.astype(np.float32)

        # FS/Sierra Lite alternation
        fs_sierra_lite = our_method_dither_fs_sierra_lite(float(gray_level), size, size, seed=0)
        images['FS/SierraLite'] = fs_sierra_lite.astype(np.float32)

        # Stucki/Sierra alternation (no FS)
        stucki_sierra = our_method_dither_stucki_sierra(float(gray_level), size, size, seed=0)
        images['Stucki/Sierra'] = stucki_sierra.astype(np.float32)

        # JJN/Stucki alternation (no FS)
        jjn_stucki = our_method_dither_jjn_stucki(float(gray_level), size, size, seed=0)
        images['JJN/Stucki'] = jjn_stucki.astype(np.float32)

        # JJN/Sierra alternation (no FS)
        jjn_sierra = our_method_dither_jjn_sierra(float(gray_level), size, size, seed=0)
        images['JJN/Sierra'] = jjn_sierra.astype(np.float32)

        # Blue noise reference (thresholded)
        blue_noise_path = base_dir / 'blue_noise_256.png'
        if blue_noise_path.exists():
            blue_noise = load_image(blue_noise_path)
            blue_ref = np.where(blue_noise < gray_level, 255.0, 0.0)
            images['Blue Noise Ref'] = blue_ref

        # Generate comparison
        output_path = output_dir / f"kernel_exp_gray_{gray_level:03d}.png"
        plot_comparison(images, f"Kernel Experiment: Gray {gray_level} ({gray_level/255*100:.0f}%)", output_path)
        print(f"  {output_path.name}")

        # Print white pixel stats (abbreviated)
        print(f"    All methods ~{np.mean(standard == 255) * 100:.1f}% white")

        # Compare variants vs FS/JJN
        diffs = {
            'FS/Stucki': np.sum(standard != fs_stucki) / (size * size) * 100,
            'FS/Sierra': np.sum(standard != fs_sierra) / (size * size) * 100,
            'FS/SierraLite': np.sum(standard != fs_sierra_lite) / (size * size) * 100,
            'Stucki/Sierra': np.sum(standard != stucki_sierra) / (size * size) * 100,
            'JJN/Stucki': np.sum(standard != jjn_stucki) / (size * size) * 100,
            'JJN/Sierra': np.sum(standard != jjn_sierra) / (size * size) * 100,
        }
        print(f"    vs FS/JJN: " + ", ".join(f"{k}={v:.1f}%" for k, v in diffs.items()))

    # Also generate blue kernel depth experiment
    print("\n  Blue kernel depth experiment:")
    for gray_level in gray_levels:
        images = {}
        images['FS/JJN'] = our_method_dither(float(gray_level), size, size, seed=0).astype(np.float32)
        for depth in [1, 2, 3]:
            blue_kernel = our_method_dither_with_blue_noise_kernel(
                float(gray_level), size, size, seed=0, recursion_depth=depth
            )
            images[f'Blue d={depth}'] = blue_kernel.astype(np.float32)
        blue_noise_path = base_dir / 'blue_noise_256.png'
        if blue_noise_path.exists():
            blue_noise = load_image(blue_noise_path)
            images['Blue Noise Ref'] = np.where(blue_noise < gray_level, 255.0, 0.0)
        output_path = output_dir / f"blue_kernel_depth_gray_{gray_level:03d}.png"
        plot_comparison(images, f"Blue Kernel Depth: Gray {gray_level} ({gray_level/255*100:.0f}%)", output_path)
        print(f"    {output_path.name}")


def analyze_sanity_check(base_dir: Path, output_dir: Path):
    """
    Sanity check: compare coin flip, blue noise, CRA tool, Python replication, and blue-kernel experiment at 50% gray.

    Verifies that our Python replication matches CRA output and shows the quality spectrum:
    - Coin flip (white noise) - worst
    - Blue noise reference - best
    - CRA tool output
    - Python replication (standard hash-based kernel)
    - Python blue-kernel (experimental: blue noise kernel selection)
    """
    import subprocess
    import sys

    gray_level = 127
    size = 256

    images = {}

    # 1. Coin flip at 50%
    rng = np.random.default_rng(seed=42)
    coin_flip = np.where(rng.random((size, size)) < 0.5, 255.0, 0.0)
    images['Coin Flip'] = coin_flip

    # 2. Blue noise reference at 50%
    blue_noise_path = base_dir / 'blue_noise_256.png'
    if blue_noise_path.exists():
        blue_noise = load_image(blue_noise_path)
        blue_ref = np.where(blue_noise < gray_level, 255.0, 0.0)
        images['Blue Noise Ref'] = blue_ref
    else:
        print(f"  Warning: {blue_noise_path} not found, skipping blue noise reference")

    # 3. CRA tool output for gray_127
    cra_path = base_dir / 'dithered' / 'boon-serpentine' / 'gray_127_boon-serpentine.png'
    if cra_path.exists():
        images['CRA Tool'] = load_image(cra_path)
    else:
        print(f"  Warning: {cra_path} not found, skipping CRA output")

    # 4. Python replication
    sys.path.insert(0, str(base_dir.parent / 'tools'))
    try:
        from our_method_dither import our_method_dither
        py_result = our_method_dither(float(gray_level), size, size, seed=0)
        images['Python Replication'] = py_result.astype(np.float32)
    except ImportError as e:
        print(f"  Warning: Could not import our_method_dither: {e}")

    # 5. C integer implementation
    c_tool = base_dir.parent / 'int_blue_dither'
    cra_tool = Path('/root/cra-web/port/target/release/cra')
    if c_tool.exists() and cra_tool.exists():
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / 'c_output.bin'
            png_path = Path(tmpdir) / 'c_output.png'
            # Run C tool
            result = subprocess.run(
                [str(c_tool), str(size), str(size), str(gray_level), str(bin_path)],
                capture_output=True, text=True
            )
            if result.returncode == 0 and bin_path.exists():
                # Convert to PNG using CRA
                metadata = f'{{"format":"L1","width":{size},"height":{size}}}'
                result = subprocess.run(
                    [str(cra_tool), '-i', str(bin_path), '--input-metadata', metadata, '-o', str(png_path)],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and png_path.exists():
                    images['C Integer'] = load_image(png_path)
                else:
                    print(f"  Warning: CRA conversion failed: {result.stderr}")
            else:
                print(f"  Warning: C tool failed: {result.stderr}")
    else:
        if not c_tool.exists():
            print(f"  Warning: {c_tool} not found, skipping C implementation")

    if len(images) < 2:
        print("  Not enough images for sanity check comparison")
        return

    # Generate comparison
    output_path = output_dir / "sanity_check_50pct.png"
    plot_comparison(images, "Sanity Check: 50% Gray Level Comparison", output_path)
    print(f"  {output_path.name}")

    # Print comparison stats
    # Note: Exact pixel match between CRA and Python isn't expected due to edge seeding
    # What matters is spectral similarity, not exact pixel match
    if 'CRA Tool' in images and 'Python Replication' in images:
        cra_img = images['CRA Tool']
        py_img = images['Python Replication']
        match_pct = np.mean(cra_img == py_img) * 100
        print(f"  CRA vs Python pixel match: {match_pct:.2f}% (edge seeding causes diff)")
        cra_white = np.mean(cra_img == 255) * 100
        py_white = np.mean(py_img == 255) * 100
        print(f"  CRA white: {cra_white:.2f}%, Python white: {py_white:.2f}%")

    if 'C Integer' in images:
        c_img = images['C Integer']
        c_white = np.mean(c_img == 255) * 100
        print(f"  C Integer white: {c_white:.2f}%")
        if 'CRA Tool' in images:
            match_pct = np.mean(images['CRA Tool'] == c_img) * 100
            print(f"  CRA vs C Integer pixel match: {match_pct:.2f}%")


def plot_ideal_blue_noise(output_dir: Path):
    """Plot ideal blue noise vs white noise reference curve."""
    # Frequency range (cycles/pixel): 0 to 0.5 (Nyquist)
    freqs = np.linspace(0.004, 0.5, 500)  # Start slightly above 0 to avoid log(0)

    # Ideal blue noise: power ∝ f^2
    # This gives +6 dB per octave (doubling frequency = 4x power = +6 dB)
    # Normalize so it matches typical dither power levels (~90 dB at f=0.5)
    P_ref = 1e9  # Reference power at f=0.5
    power = P_ref * (freqs / 0.5) ** 2
    power_db = 10 * np.log10(power + 1e-10)

    # White noise (flat) for comparison
    white_db = np.full_like(freqs, 10 * np.log10(P_ref))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freqs, power_db, 'b-', linewidth=2, label='Ideal Blue Noise (f²)')
    ax.plot(freqs, white_db, 'gray', linestyle='--', linewidth=1.5, label='White Noise (flat)')

    # Mark octaves
    octaves = [0.0625, 0.125, 0.25, 0.5]
    for oct in octaves:
        ax.axvline(x=oct, color='lightgray', linestyle=':', alpha=0.7)
        ax.text(oct, power_db.min() + 2, f'{oct}', ha='center', fontsize=8, color='gray')

    # Annotations
    ax.annotate('+6 dB/octave', xy=(0.2, 10*np.log10(P_ref * (0.2/0.5)**2)),
                xytext=(0.1, 75), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))

    ax.set_xlabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    ax.set_ylabel('Power (dB)', fontsize=12)
    ax.set_title('Ideal Blue Noise vs White Noise Power Spectrum', fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(50, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "ideal_blue_noise.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {output_path.name}")


def analyze_vs_blue_noise(base_dir: Path, output_dir: Path):
    """Compare our method vs void-and-cluster blue noise at various gray levels."""
    blue_noise_path = base_dir / "blue_noise_256.png"
    if not blue_noise_path.exists():
        print(f"  Blue noise reference not found: {blue_noise_path}")
        return

    blue_noise_raw = load_image(blue_noise_path)
    dithered_dir = base_dir / "dithered" / "boon-serpentine"

    # Gray levels to test
    gray_levels = [42, 85, 127, 170, 191]

    for gray in gray_levels:
        our_method_path = dithered_dir / f"gray_{gray:03d}_boon-serpentine.png"
        if not our_method_path.exists():
            print(f"  Skipping gray_{gray:03d}: not found")
            continue

        # Threshold blue noise at this gray level
        blue_noise_thresh = (blue_noise_raw < gray).astype(np.float32) * 255
        our_method = load_image(our_method_path)

        # Compute spectra
        freqs_bn, h_bn, d_bn, v_bn = compute_segmented_radial_power(blue_noise_thresh)
        freqs_om, h_om, d_om, v_om = compute_segmented_radial_power(our_method)

        # Convert to dB
        h_bn_db = 10 * np.log10(h_bn + 1e-10)
        d_bn_db = 10 * np.log10(d_bn + 1e-10)
        v_bn_db = 10 * np.log10(v_bn + 1e-10)

        h_om_db = 10 * np.log10(h_om + 1e-10)
        d_om_db = 10 * np.log10(d_om + 1e-10)
        v_om_db = 10 * np.log10(v_om + 1e-10)

        # Ideal f² curve for reference
        freqs_ideal = np.linspace(0.004, 0.5, 500)
        P_ref = 10 ** (h_bn_db.max() / 10)
        power_ideal_db = 10 * np.log10(P_ref * (freqs_ideal / 0.5) ** 2 + 1e-10)

        # Calculate stats
        bn_avg_db = (h_bn_db + d_bn_db + v_bn_db) / 3
        om_avg_db = (h_om_db + d_om_db + v_om_db) / 3
        low_freq_diff = om_avg_db[freqs_om < 0.1].mean() - bn_avg_db[freqs_bn < 0.1].mean()

        # Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        y_min = min(h_bn_db.min(), h_om_db.min()) - 5
        y_max = max(h_bn_db.max(), h_om_db.max()) + 5

        # Left: Blue Noise (void-and-cluster)
        axes[0].plot(freqs_bn, h_bn_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[0].plot(freqs_bn, d_bn_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[0].plot(freqs_bn, v_bn_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[0].plot(freqs_ideal, power_ideal_db, 'k--', linewidth=1, alpha=0.5, label='Ideal f²')
        axes[0].set_xlabel('Spatial Frequency (cycles/pixel)')
        axes[0].set_ylabel('Power (dB)')
        axes[0].set_title(f'Void-and-Cluster Blue Noise\n(thresholded at {gray})', fontsize=12, fontweight='bold')
        axes[0].set_xlim(0, 0.5)
        axes[0].set_ylim(y_min, y_max)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='lower right')

        # Right: Our Method
        axes[1].plot(freqs_om, h_om_db, 'r-', label='H', alpha=0.8, linewidth=1)
        axes[1].plot(freqs_om, d_om_db, 'g-', label='D', alpha=0.8, linewidth=1)
        axes[1].plot(freqs_om, v_om_db, 'b-', label='V', alpha=0.8, linewidth=1)
        axes[1].plot(freqs_ideal, power_ideal_db, 'k--', linewidth=1, alpha=0.5, label='Ideal f²')
        axes[1].set_xlabel('Spatial Frequency (cycles/pixel)')
        axes[1].set_ylabel('Power (dB)')
        axes[1].set_title(f'Our Method (FS/JJN/lowbias32)\ngray_{gray:03d}', fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, 0.5)
        axes[1].set_ylim(y_min, y_max)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='lower right')

        gray_pct = gray / 255 * 100
        plt.suptitle(f'Void-and-Cluster vs Our Method @ {gray_pct:.1f}% gray (low-freq diff: {low_freq_diff:.1f} dB)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"blue_noise_vs_our_method_{gray:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {output_path.name} (low-freq diff: {low_freq_diff:.1f} dB)")


def main():
    parser = argparse.ArgumentParser(description='Analyze dithered images with Fourier methods')
    parser.add_argument('--input', '-i', type=Path, help='Single input image to analyze')
    parser.add_argument('--output', '-o', type=Path, help='Output path for single image analysis')
    parser.add_argument('--compare', '-c', action='store_true', help='Generate comparison charts for all test images')
    parser.add_argument('--serpentine', '-s', action='store_true', help='Compare serpentine variants only')
    parser.add_argument('--rng', action='store_true', help='Analyze RNG noise images')
    parser.add_argument('--hash', action='store_true', help='Compare boon hash functions (lowbias32 vs wang)')
    parser.add_argument('--sanity', action='store_true', help='Sanity check: compare coin flip, blue noise, CRA, Python replication')
    parser.add_argument('--blue-kernel', action='store_true', help='Experimental: compare standard vs blue-noise kernel selection')
    parser.add_argument('--ideal', action='store_true', help='Plot ideal blue noise vs white noise reference curve')
    parser.add_argument('--vs-blue-noise', action='store_true', help='Compare our method vs void-and-cluster blue noise')
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "test_images"
    dithered_dir = base_dir / "dithered"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    if args.input:
        # Single image analysis
        output_path = args.output or output_dir / f"{args.input.stem}_analysis.png"
        analyze_single(args.input, output_path)
    elif args.blue_kernel:
        # Blue kernel experiment
        print("Running blue kernel experiment...")
        analyze_blue_kernel_experiment(base_dir, output_dir)
        print(f"\nDone! Blue kernel experiment saved to {output_dir}")
    elif args.sanity:
        # Sanity check comparison
        print("Running sanity check comparison...")
        analyze_sanity_check(base_dir, output_dir)
        print(f"\nDone! Sanity check saved to {output_dir}")
    elif args.ideal:
        # Plot ideal blue noise reference
        print("Plotting ideal blue noise reference...")
        plot_ideal_blue_noise(output_dir)
        print(f"\nDone! Ideal blue noise plot saved to {output_dir}")
    elif args.vs_blue_noise:
        # Compare our method vs void-and-cluster
        print("Comparing our method vs void-and-cluster blue noise...")
        analyze_vs_blue_noise(base_dir, output_dir)
        print(f"\nDone! Comparison saved to {output_dir}")
    elif args.hash:
        # Hash comparison (lowbias32 vs wang)
        source_images = [p.stem for p in (base_dir / "sources").glob("*.png")]
        print(f"Generating hash comparison for {len(source_images)} images...")
        for image_name in sorted(source_images):
            analyze_hash_comparison(dithered_dir, image_name, output_dir)
        print(f"\nDone! Hash comparison saved to {output_dir}")
    elif args.rng:
        # Analyze RNG noise images
        print("Analyzing RNG noise images...")
        analyze_rng_noise(base_dir, output_dir)
        print(f"\nDone! Analysis saved to {output_dir}")
    elif args.compare or args.serpentine:
        # Find all source images (from sources/ subfolder)
        source_images = [p.stem for p in (base_dir / "sources").glob("*.png")]
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
