#!/usr/bin/env python3
"""
Wavelet-based halftone quality analysis.

Compares dithered 1-bit images against originals using Haar wavelet decomposition
to detect artifacts (worms, checkerboards) and measure structure preservation.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json


def haar_decompose_2d(img: np.ndarray, levels: int = 4) -> dict:
    """
    Perform 2D Haar wavelet decomposition.

    Returns dict with:
        'LL': final low-frequency approximation
        'LH': list of horizontal detail subbands (levels 0 to levels-1)
        'HL': list of vertical detail subbands
        'HH': list of diagonal detail subbands

    Level 0 = finest (2px scale), level 3 = coarsest (16px scale)
    """
    # Work in float
    current = img.astype(np.float64)

    LH_list = []
    HL_list = []
    HH_list = []

    for level in range(levels):
        h, w = current.shape

        # Ensure even dimensions
        if h % 2 != 0:
            current = current[:-1, :]
            h -= 1
        if w % 2 != 0:
            current = current[:, :-1]
            w -= 1

        # Extract 2x2 blocks
        # a b
        # c d
        a = current[0::2, 0::2]
        b = current[0::2, 1::2]
        c = current[1::2, 0::2]
        d = current[1::2, 1::2]

        # Haar transform
        LL = (a + b + c + d) / 4.0  # Average
        LH = (a + b - c - d) / 4.0  # Horizontal detail (difference between top and bottom rows)
        HL = (a - b + c - d) / 4.0  # Vertical detail (difference between left and right cols)
        HH = (a - b - c + d) / 4.0  # Diagonal detail

        LH_list.append(LH)
        HL_list.append(HL)
        HH_list.append(HH)

        # Recurse on LL
        current = LL

    return {
        'LL': current,
        'LH': LH_list,
        'HL': HL_list,
        'HH': HH_list,
    }


def subband_energy(subband: np.ndarray) -> float:
    """Compute energy (sum of squares) of a subband."""
    return np.sum(subband ** 2)


def subband_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute correlation between two subbands."""
    energy_a = subband_energy(a)
    energy_b = subband_energy(b)

    if energy_a < 1e-10 or energy_b < 1e-10:
        return 0.0

    return np.sum(a * b) / np.sqrt(energy_a * energy_b)


def spectral_flatness(subband: np.ndarray) -> float:
    """
    Compute spectral flatness of a subband.

    Flatness = geometric_mean(power) / arithmetic_mean(power)

    - Flatness = 1.0 → flat spectrum = white noise = ideal
    - Flatness → 0.0 → peaked spectrum = periodic patterns = bad

    Returns flatness value in [0, 1].
    """
    h, w = subband.shape
    if h < 2 or w < 2:
        return 1.0  # Trivial case, assume flat

    # Compute 2D FFT and power spectrum
    fft = np.fft.fft2(subband)
    power = np.abs(fft) ** 2

    # Flatten and exclude DC component
    power_flat = power.flatten()
    power_flat = power_flat[1:]  # Remove DC

    if len(power_flat) == 0 or np.max(power_flat) < 1e-10:
        return 1.0  # No energy, consider flat

    # Avoid log(0) by adding tiny epsilon
    power_flat = power_flat + 1e-10

    # Spectral flatness = exp(mean(log(x))) / mean(x) = geomean / mean
    log_mean = np.mean(np.log(power_flat))
    geom_mean = np.exp(log_mean)
    arith_mean = np.mean(power_flat)

    flatness = geom_mean / arith_mean

    return float(np.clip(flatness, 0.0, 1.0))


def analyze_halftone(original: np.ndarray, halftone: np.ndarray, levels: int = 4) -> dict:
    """
    Analyze halftone quality using wavelet decomposition.

    Args:
        original: Grayscale image, values in [0, 1]
        halftone: Binary image, values in {0, 1}
        levels: Number of wavelet decomposition levels

    Returns dict with all metrics.
    """
    # Decompose both images
    orig_wav = haar_decompose_2d(original, levels)
    half_wav = haar_decompose_2d(halftone, levels)

    # Also decompose the error image
    error = halftone - original
    err_wav = haar_decompose_2d(error, levels)

    results = {
        'levels': levels,
        'subbands': {},
        'summary': {},
    }

    for level in range(levels):
        scale = 2 ** (level + 1)  # 2, 4, 8, 16 pixels
        perceptual_weight = 2 ** level  # Weight coarser scales more heavily

        level_results = {}

        for orientation, key in [('LH', 'horizontal'), ('HL', 'vertical'), ('HH', 'diagonal')]:
            orig_sub = orig_wav[orientation][level]
            half_sub = half_wav[orientation][level]
            err_sub = err_wav[orientation][level]

            # Raw energies
            energy_orig = subband_energy(orig_sub)
            energy_half = subband_energy(half_sub)
            energy_err = subband_energy(err_sub)

            # Normalize by subband area for cross-level comparison
            subband_pixels = orig_sub.shape[0] * orig_sub.shape[1]
            energy_orig_norm = energy_orig / subband_pixels
            energy_half_norm = energy_half / subband_pixels

            excess_raw = max(0, energy_half - energy_orig)
            missing_raw = max(0, energy_orig - energy_half)

            # Normalized excess: per-pixel, then weighted by scale
            excess_norm = max(0, energy_half_norm - energy_orig_norm)
            missing_norm = max(0, energy_orig_norm - energy_half_norm)
            excess_weighted = excess_norm * perceptual_weight
            missing_weighted = missing_norm * perceptual_weight

            correlation = subband_correlation(orig_sub, half_sub)

            # Spectral flatness of error subband (measures noise vs structure)
            flatness = spectral_flatness(err_sub)

            level_results[key] = {
                'energy_orig': energy_orig,
                'energy_half': energy_half,
                'energy_error': energy_err,
                'subband_pixels': subband_pixels,
                'excess_raw': excess_raw,
                'missing_raw': missing_raw,
                'excess_norm': excess_norm,
                'missing_norm': missing_norm,
                'excess_weighted': excess_weighted,
                'missing_weighted': missing_weighted,
                'perceptual_weight': perceptual_weight,
                'correlation': correlation,
                'flatness': flatness,
            }

        # Isotropy metrics for this level (based on error image)
        err_energies = [
            level_results['horizontal']['energy_error'],
            level_results['vertical']['energy_error'],
            level_results['diagonal']['energy_error'],
        ]
        total_err = sum(err_energies)

        if total_err > 1e-10:
            level_results['isotropy'] = {
                'h_ratio': err_energies[0] / total_err,
                'v_ratio': err_energies[1] / total_err,
                'd_ratio': err_energies[2] / total_err,
                'isotropy_score': min(err_energies) / max(err_energies) if max(err_energies) > 1e-10 else 1.0,
            }
        else:
            level_results['isotropy'] = {
                'h_ratio': 0.333,
                'v_ratio': 0.333,
                'd_ratio': 0.333,
                'isotropy_score': 1.0,
            }

        results['subbands'][f'level_{level}'] = level_results

    # Summary metrics
    # Raw totals (for backwards compatibility)
    total_excess_raw = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    total_missing_raw = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    # Weighted totals (normalized by area, weighted by perceptual scale)
    total_excess_weighted = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    total_missing_weighted = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    weighted_corr_sum = 0
    weight_sum = 0

    for level in range(levels):
        level_data = results['subbands'][f'level_{level}']
        for orientation in ['horizontal', 'vertical', 'diagonal']:
            total_excess_raw[orientation] += level_data[orientation]['excess_raw']
            total_missing_raw[orientation] += level_data[orientation]['missing_raw']
            total_excess_weighted[orientation] += level_data[orientation]['excess_weighted']
            total_missing_weighted[orientation] += level_data[orientation]['missing_weighted']

            # Weight correlation by original energy
            w = level_data[orientation]['energy_orig']
            weighted_corr_sum += w * level_data[orientation]['correlation']
            weight_sum += w

    results['summary'] = {
        # Raw scores (backwards compat)
        'total_excess': total_excess_raw,
        'total_missing': total_missing_raw,
        'artifact_score': sum(total_excess_raw.values()),
        # Weighted scores (new, perceptually meaningful)
        'total_excess_weighted': total_excess_weighted,
        'total_missing_weighted': total_missing_weighted,
        'artifact_score_weighted': sum(total_excess_weighted.values()),
        'worm_h_score': total_excess_weighted['horizontal'],
        'worm_v_score': total_excess_weighted['vertical'],
        'checkerboard_score': total_excess_weighted['diagonal'],
        'structure_score': weighted_corr_sum / weight_sum if weight_sum > 1e-10 else 0.0,
    }

    # Per-level breakdown
    per_level_flatness = []
    for level in range(levels):
        level_data = results['subbands'][f'level_{level}']
        # Average flatness across orientations for this level
        level_flatness = np.mean([level_data[o]['flatness'] for o in ['horizontal', 'vertical', 'diagonal']])
        per_level_flatness.append(level_flatness)
    results['summary']['per_level_flatness'] = per_level_flatness

    # Flatness score: average across all levels (higher = more noise-like = better)
    results['summary']['flatness_avg'] = np.mean(per_level_flatness)

    # Weighted flatness (weight coarser levels more, since patterns there are more visible)
    # Use inverse weighting: low flatness at coarse scale is worse
    weights = [2 ** l for l in range(levels)]
    total_weight = sum(weights)
    flatness_weighted = sum(per_level_flatness[l] * weights[l] for l in range(levels)) / total_weight
    results['summary']['flatness_weighted'] = flatness_weighted

    # Isotropy summary (geometric mean across levels)
    isotropy_scores = [results['subbands'][f'level_{l}']['isotropy']['isotropy_score']
                       for l in range(levels)]
    results['summary']['isotropy_score'] = np.exp(np.mean(np.log(np.array(isotropy_scores) + 1e-10)))

    # Normalized scores (for cross-image comparison)
    # artifact_score_weighted is already normalized per-pixel and scale-weighted
    results['summary']['artifact_score_norm'] = results['summary']['artifact_score_weighted']

    return results


def print_analysis(results: dict, name: str = ""):
    """Print analysis results in a readable format."""
    print(f"\n{'=' * 60}")
    if name:
        print(f"Analysis: {name}")
        print(f"{'=' * 60}")

    print("\nSubband Spectral Flatness (1.0 = noise, 0.0 = structured):")
    print("-" * 85)
    print(f"{'Level':<8} {'Scale':<6} {'LH flat':<10} {'HL flat':<10} {'HH flat':<10} {'Avg':<8} {'Iso':<8}")
    print("-" * 85)

    for level in range(results['levels']):
        scale = 2 ** (level + 1)
        data = results['subbands'][f'level_{level}']
        iso = data['isotropy']

        lh_f = data['horizontal']['flatness']
        hl_f = data['vertical']['flatness']
        hh_f = data['diagonal']['flatness']
        avg_f = (lh_f + hl_f + hh_f) / 3

        print(f"Level {level:<2} {scale:>3}px   {lh_f:>8.4f}  {hl_f:>8.4f}  {hh_f:>8.4f}  {avg_f:>6.4f}  {iso['isotropy_score']:>6.3f}")

    print("\nPer-level flatness (higher = more noise-like = better):")
    print("-" * 85)
    for level, flatness in enumerate(results['summary']['per_level_flatness']):
        scale = 2 ** (level + 1)
        print(f"  Level {level} ({scale:>2}px): flatness={flatness:.4f}")

    print("\nSummary:")
    print("-" * 70)
    s = results['summary']
    print(f"  Flatness (weighted):    {s['flatness_weighted']:.4f}  (higher = more noise-like = better)")
    print(f"  Flatness (average):     {s['flatness_avg']:.4f}")
    print(f"  Structure preservation: {s['structure_score']:.3f}")
    print(f"  Isotropy score:         {s['isotropy_score']:.3f}")


def visualize_analysis(original: np.ndarray, halftone: np.ndarray, results: dict,
                       output_path: Path = None, title: str = ""):
    """Create visualization of the wavelet analysis."""

    orig_wav = haar_decompose_2d(original, results['levels'])
    half_wav = haar_decompose_2d(halftone, results['levels'])
    error = halftone - original
    err_wav = haar_decompose_2d(error, results['levels'])

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)

    # Row 0: Original, Halftone, Error, and summary bar chart
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original, cmap='gray', vmin=0, vmax=1)
    ax_orig.set_title('Original')
    ax_orig.axis('off')

    ax_half = fig.add_subplot(gs[0, 1])
    ax_half.imshow(halftone, cmap='gray', vmin=0, vmax=1)
    ax_half.set_title('Halftone')
    ax_half.axis('off')

    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.imshow(error, cmap='RdBu', vmin=-1, vmax=1)
    ax_err.set_title('Error (H - O)')
    ax_err.axis('off')

    # Summary bar chart - show per-level spectral flatness
    ax_summary = fig.add_subplot(gs[0, 3:])
    s = results['summary']
    per_level_flat = s.get('per_level_flatness', [0, 0, 0, 0])
    level_labels = ['2px', '4px', '8px', '16px']
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22']
    bars = ax_summary.bar(level_labels, per_level_flat, color=colors)
    ax_summary.set_ylabel('Spectral Flatness')
    ax_summary.set_ylim(0, 1.0)
    ax_summary.axhline(y=1.0, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_summary.set_title(f'Flatness (Avg: {s["flatness_avg"]:.3f}, Struct: {s["structure_score"]:.3f}, Iso: {s["isotropy_score"]:.3f})')

    # Rows 1-3: Wavelet subbands for each level
    orientations = ['LH', 'HL', 'HH']
    orientation_names = ['LH (Horiz)', 'HL (Vert)', 'HH (Diag)']

    for row, (orient, orient_name) in enumerate(zip(orientations, orientation_names), start=1):
        for level in range(min(4, results['levels'])):
            ax = fig.add_subplot(gs[row, level])

            # Show error subband
            subband = err_wav[orient][level]
            vmax = max(0.1, np.abs(subband).max())
            ax.imshow(subband, cmap='RdBu', vmin=-vmax, vmax=vmax)

            # Add metrics as title
            orient_key = {'LH': 'horizontal', 'HL': 'vertical', 'HH': 'diagonal'}[orient]
            data = results['subbands'][f'level_{level}'][orient_key]
            scale = 2 ** (level + 1)
            ax.set_title(f'{orient_name}\n{scale}px, flat={data["flatness"]:.3f}', fontsize=8)
            ax.axis('off')

        # Isotropy chart in last column
        ax_iso = fig.add_subplot(gs[row, 4])
        ratios = []
        for level in range(results['levels']):
            iso = results['subbands'][f'level_{level}']['isotropy']
            key = {'LH': 'h_ratio', 'HL': 'v_ratio', 'HH': 'd_ratio'}[orient]
            ratios.append(iso[key])

        ax_iso.bar(range(results['levels']), ratios, color=colors[row-1], alpha=0.7)
        ax_iso.axhline(y=0.333, color='black', linestyle='--', alpha=0.5)
        ax_iso.set_ylim(0, 0.6)
        ax_iso.set_xlabel('Level')
        ax_iso.set_ylabel('Ratio')
        ax_iso.set_title(f'{orient_name} Ratio', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_chart(all_results: dict, image_name: str, output_dir: Path):
    """Create a bar chart comparing all methods."""
    methods = list(all_results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    # Flatness weighted - higher is better
    ax = axes[0, 0]
    values = [all_results[m]['summary']['flatness_weighted'] for m in methods]
    ax.bar(range(n_methods), values, color=colors)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title('Flatness Weighted (higher=better)')

    # Flatness average - higher is better
    ax = axes[0, 1]
    values = [all_results[m]['summary']['flatness_avg'] for m in methods]
    ax.bar(range(n_methods), values, color=colors)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title('Flatness Average')

    # Per-level flatness breakdown
    ax = axes[0, 2]
    x = np.arange(4)
    width = 0.8 / n_methods
    for i, method in enumerate(methods):
        per_level = all_results[method]['summary'].get('per_level_flatness', [0, 0, 0, 0])
        ax.bar(x + i * width - 0.4 + width/2, per_level, width, label=method, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(['2px', '4px', '8px', '16px'])
    ax.set_ylim(0, 1.0)
    ax.set_title('Per-Level Flatness')
    ax.legend(fontsize=5, ncol=2)

    # Isotropy - higher is better
    ax = axes[1, 0]
    values = [all_results[m]['summary']['isotropy_score'] for m in methods]
    ax.bar(range(n_methods), values, color=colors)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_title('Isotropy (higher=better)')
    ax.set_ylim(0, 1.1)

    # Structure preservation - higher is better
    ax = axes[1, 1]
    values = [all_results[m]['summary']['structure_score'] for m in methods]
    ax.bar(range(n_methods), values, color=colors)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_title('Structure Preservation')
    ax.set_ylim(0, 1.1)

    # Hide unused subplot
    axes[1, 2].axis('off')

    fig.suptitle(f'Wavelet Quality Comparison: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f"wavelet_comparison_{image_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {output_path}")


def create_summary_table(all_data: dict, output_dir: Path):
    """Create a summary table across all images and methods."""

    # Collect all methods and images
    all_methods = set()
    all_images = list(all_data.keys())
    for img_data in all_data.values():
        all_methods.update(img_data.keys())
    all_methods = sorted(all_methods)

    # Create summary CSV
    csv_path = output_dir / "wavelet_summary.csv"
    with open(csv_path, 'w') as f:
        # Header
        f.write("Image,Method,FlatnessWeighted,FlatnessAvg,Structure,Isotropy,Flat_2px,Flat_4px,Flat_8px,Flat_16px\n")

        for img in all_images:
            for method in all_methods:
                if method in all_data[img]:
                    s = all_data[img][method]['summary']
                    per_level = s.get('per_level_flatness', [0, 0, 0, 0])
                    f.write(f"{img},{method},{s['flatness_weighted']:.6f},"
                           f"{s['flatness_avg']:.6f},"
                           f"{s['structure_score']:.4f},{s['isotropy_score']:.4f},"
                           f"{per_level[0]:.6f},{per_level[1]:.6f},{per_level[2]:.6f},{per_level[3]:.6f}\n")

    print(f"  Saved summary CSV: {csv_path}")

    # Create aggregated comparison chart (average across all images)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    metrics = [
        ('flatness_weighted', 'Flatness Weighted (higher=better)', axes[0, 0], True),
        ('flatness_avg', 'Flatness Average', axes[0, 1], True),
        ('structure_score', 'Structure Preservation', axes[0, 2], True),
        ('isotropy_score', 'Isotropy', axes[0, 3], True),
    ]

    # Calculate averages per method
    method_avgs = {}
    for method in all_methods:
        method_avgs[method] = {}
        for metric_key, _, _, _ in metrics:
            values = []
            for img in all_images:
                if method in all_data[img]:
                    values.append(all_data[img][method]['summary'][metric_key])
            if values:
                method_avgs[method][metric_key] = np.mean(values)

    n_methods = len(all_methods)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for metric_key, metric_name, ax, higher_better in metrics:
        values = [method_avgs[m].get(metric_key, 0) for m in all_methods]
        bars = ax.bar(range(n_methods), values, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric_name)

        if higher_better:
            ax.set_ylim(0, 1.1)

    # Per-level flatness chart (key metric)
    ax_levels = axes[1, 0]
    level_avgs = {m: [] for m in all_methods}
    for method in all_methods:
        for level in range(4):
            values = []
            for img in all_images:
                if method in all_data[img]:
                    per_level = all_data[img][method]['summary'].get('per_level_flatness', [0, 0, 0, 0])
                    if level < len(per_level):
                        values.append(per_level[level])
            level_avgs[method].append(np.mean(values) if values else 0)

    x = np.arange(4)
    width = 0.8 / n_methods
    for i, method in enumerate(all_methods):
        ax_levels.bar(x + i * width - 0.4 + width/2, level_avgs[method], width, label=method, color=colors[i])
    ax_levels.set_xticks(x)
    ax_levels.set_xticklabels(['2px', '4px', '8px', '16px'])
    ax_levels.set_ylim(0, 1.0)
    ax_levels.set_title('Per-Level Flatness (higher=better)')
    ax_levels.legend(fontsize=5, ncol=2)

    # Hide unused subplots
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    fig.suptitle('Wavelet Quality Summary (averaged across all images)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "wavelet_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary chart: {output_path}")

    # Also save as JSON for further analysis
    json_path = output_dir / "wavelet_results.json"

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj

    with open(json_path, 'w') as f:
        json.dump(convert_to_native(all_data), f, indent=2)
    print(f"  Saved results JSON: {json_path}")


def run_comparison(ref_dir: Path, dithered_dir: Path, output_dir: Path, levels: int = 4):
    """
    Run comparison across all images and dithering methods.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all reference images
    ref_images = list(ref_dir.glob("*.png"))
    if not ref_images:
        print(f"No reference images found in {ref_dir}")
        return

    # Find all dithering methods (subdirectories of dithered_dir)
    methods = [d.name for d in dithered_dir.iterdir() if d.is_dir()]
    if not methods:
        print(f"No dithered method directories found in {dithered_dir}")
        return

    print(f"Found {len(ref_images)} images and {len(methods)} methods")
    print(f"Methods: {', '.join(sorted(methods))}")
    print()

    all_data = {}

    for ref_path in sorted(ref_images):
        img_name = ref_path.stem
        print(f"Analyzing: {img_name}")

        # Load original
        orig_img = Image.open(ref_path).convert('L')
        original = np.array(orig_img).astype(np.float64) / 255.0

        all_data[img_name] = {}

        for method in sorted(methods):
            halftone_path = dithered_dir / method / f"{img_name}.png"
            if not halftone_path.exists():
                print(f"  [skip] {method} - not found")
                continue

            # Load halftone
            half_img = Image.open(halftone_path).convert('L')
            halftone = np.array(half_img).astype(np.float64) / 255.0

            # Ensure same size
            if halftone.shape != original.shape:
                print(f"  [skip] {method} - size mismatch")
                continue

            # Analyze
            results = analyze_halftone(original, halftone, levels)
            all_data[img_name][method] = results

            # Print summary
            s = results['summary']
            print(f"  {method:30s} flat={s['flatness_avg']:.4f} struct={s['structure_score']:.3f} iso={s['isotropy_score']:.3f}")

            # Generate individual visualization
            vis_path = output_dir / f"wavelet_{img_name}_{method}.png"
            visualize_analysis(original, halftone, results, vis_path,
                              f"{img_name} - {method}")

        # Generate per-image comparison
        if all_data[img_name]:
            create_comparison_chart(all_data[img_name], img_name, output_dir)

        print()

    # Generate overall summary
    print("Generating summary...")
    create_summary_table(all_data, output_dir)

    return all_data


def main():
    parser = argparse.ArgumentParser(description='Wavelet-based halftone quality analysis')
    parser.add_argument('--original', '-o', type=Path, help='Original grayscale image')
    parser.add_argument('--halftone', type=Path, help='Halftone (dithered) image')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare all dithering methods for test images')
    parser.add_argument('--ref-dir', type=Path, default=Path('tools/test_wavelets/reference_images'),
                        help='Directory containing reference images')
    parser.add_argument('--dithered-dir', type=Path, default=Path('tools/test_wavelets/dithered'),
                        help='Directory containing dithered subdirectories')
    parser.add_argument('--output-dir', type=Path, default=Path('tools/test_wavelets/analysis'),
                        help='Output directory for charts')
    parser.add_argument('--levels', '-l', type=int, default=4,
                        help='Wavelet decomposition levels (default: 4)')
    parser.add_argument('--image', '-i', type=str, default=None,
                        help='Specific test image name (e.g., "lena_gray_512")')

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.original and args.halftone:
        # Single image analysis
        orig_img = Image.open(args.original).convert('L')
        original = np.array(orig_img).astype(np.float64) / 255.0

        half_img = Image.open(args.halftone).convert('L')
        halftone = np.array(half_img).astype(np.float64) / 255.0

        results = analyze_halftone(original, halftone, args.levels)
        print_analysis(results, f"{args.original.stem} vs {args.halftone.stem}")

        output_path = args.output_dir / f"wavelet_{args.original.stem}_{args.halftone.stem}.png"
        visualize_analysis(original, halftone, results, output_path)
        print(f"Saved: {output_path}")

    elif args.compare:
        # Compare all dithering methods
        run_comparison(args.ref_dir, args.dithered_dir, args.output_dir, args.levels)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
