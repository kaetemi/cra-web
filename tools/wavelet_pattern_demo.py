#!/usr/bin/env python3
"""
Wavelet Pattern Encoding Demonstration

Generates a visual sheet showing how Haar wavelets encode different patterns:
- Horizontal lines → LH subband (horizontal detail)
- Vertical lines → HL subband (vertical detail)
- Diagonal lines → HH subband (diagonal detail)
- Checkerboard → HH subband
- Noise → all subbands (flat distribution)

This demonstrates why wavelet analysis is effective for detecting
dithering artifacts like "worms" (directional patterns).
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def haar_decompose_2d(img: np.ndarray, levels: int = 3) -> dict:
    """
    Perform 2D Haar wavelet decomposition.

    Returns dict with:
        'LL': final low-frequency approximation
        'LH': list of horizontal detail subbands (levels 0 to levels-1)
        'HL': list of vertical detail subbands
        'HH': list of diagonal detail subbands

    Level 0 = finest (2px scale), higher levels = coarser scales
    """
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

        # Extract 2x2 blocks: a b / c d
        a = current[0::2, 0::2]
        b = current[0::2, 1::2]
        c = current[1::2, 0::2]
        d = current[1::2, 1::2]

        # Haar transform
        LL = (a + b + c + d) / 4.0  # Average
        LH = (a + b - c - d) / 4.0  # Horizontal detail (top-bottom difference)
        HL = (a - b + c - d) / 4.0  # Vertical detail (left-right difference)
        HH = (a - b - c + d) / 4.0  # Diagonal detail

        LH_list.append(LH)
        HL_list.append(HL)
        HH_list.append(HH)

        current = LL

    return {
        'LL': current,
        'LH': LH_list,
        'HL': HL_list,
        'HH': HH_list,
    }


def generate_patterns(size: int = 128) -> dict:
    """Generate test patterns that demonstrate wavelet selectivity."""
    patterns = {}

    # Horizontal lines (2px period)
    h_lines_2 = np.zeros((size, size))
    h_lines_2[0::2, :] = 1.0
    patterns['Horizontal Lines (2px)'] = h_lines_2

    # Horizontal lines (4px period)
    h_lines_4 = np.zeros((size, size))
    h_lines_4[0::4, :] = 1.0
    h_lines_4[1::4, :] = 1.0
    patterns['Horizontal Lines (4px)'] = h_lines_4

    # Horizontal lines (8px period)
    h_lines_8 = np.zeros((size, size))
    for i in range(0, size, 8):
        h_lines_8[i:i+4, :] = 1.0
    patterns['Horizontal Lines (8px)'] = h_lines_8

    # Vertical lines (2px period)
    v_lines_2 = np.zeros((size, size))
    v_lines_2[:, 0::2] = 1.0
    patterns['Vertical Lines (2px)'] = v_lines_2

    # Vertical lines (4px period)
    v_lines_4 = np.zeros((size, size))
    v_lines_4[:, 0::4] = 1.0
    v_lines_4[:, 1::4] = 1.0
    patterns['Vertical Lines (4px)'] = v_lines_4

    # Vertical lines (8px period)
    v_lines_8 = np.zeros((size, size))
    for i in range(0, size, 8):
        v_lines_8[:, i:i+4] = 1.0
    patterns['Vertical Lines (8px)'] = v_lines_8

    # Checkerboard (2px)
    checker_2 = np.zeros((size, size))
    checker_2[0::2, 0::2] = 1.0
    checker_2[1::2, 1::2] = 1.0
    patterns['Checkerboard (2px)'] = checker_2

    # Checkerboard (4px)
    checker_4 = np.zeros((size, size))
    for i in range(0, size, 4):
        for j in range(0, size, 4):
            if (i // 4 + j // 4) % 2 == 0:
                checker_4[i:i+4, j:j+4] = 1.0
    patterns['Checkerboard (4px)'] = checker_4

    # Diagonal lines (45 deg, going down-right)
    diag_dr = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 4 < 2:
                diag_dr[i, j] = 1.0
    patterns['Diagonal \\ (4px)'] = diag_dr

    # Diagonal lines (135 deg, going down-left)
    diag_dl = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i - j) % 4 < 2:
                diag_dl[i, j] = 1.0
    patterns['Diagonal / (4px)'] = diag_dl

    # White noise
    np.random.seed(42)
    noise = (np.random.random((size, size)) > 0.5).astype(np.float64)
    patterns['White Noise'] = noise

    # Blue noise (approximated using ordered dither pattern)
    # Create a simple approximation using Bayer-like pattern
    bayer_8 = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ]) / 64.0

    blue = np.tile(bayer_8, (size // 8, size // 8))
    blue = (blue < 0.5).astype(np.float64)
    patterns['Ordered Dither (Bayer)'] = blue

    # "Worm" pattern - simulated FS artifact
    worm = np.zeros((size, size))
    for i in range(size):
        offset = (i * 3) % 8  # Slanted lines
        for j in range(size):
            if (j + offset) % 4 < 2:
                worm[i, j] = 1.0
    patterns['"Worm" Pattern'] = worm

    # Flat gray (50%)
    flat = np.ones((size, size)) * 0.5
    patterns['Flat 50% Gray'] = flat

    return patterns


def compute_subband_energy(subband: np.ndarray) -> float:
    """Compute mean squared value (signal power) of a subband."""
    return np.mean(subband ** 2)


def create_demo_sheet(output_path: Path, size: int = 128):
    """Create a demonstration sheet showing wavelet pattern encoding."""

    patterns = generate_patterns(size)
    n_patterns = len(patterns)

    # Create figure with gridspec
    # Columns: Pattern | LH (2px) | HL (2px) | HH (2px) | LH (4px) | HL (4px) | HH (4px)
    fig = plt.figure(figsize=(20, n_patterns * 1.8 + 2))

    # Header row + pattern rows
    gs = gridspec.GridSpec(n_patterns + 1, 8, figure=fig,
                           hspace=0.15, wspace=0.08,
                           height_ratios=[0.3] + [1.0] * n_patterns)

    # Column headers
    headers = ['Pattern', 'LH 2px\n(Horiz)', 'HL 2px\n(Vert)', 'HH 2px\n(Diag)',
               'LH 4px', 'HL 4px', 'HH 4px', 'Energy\nProfile']
    for col, header in enumerate(headers):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, header, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.axis('off')

    # Process each pattern
    for row, (name, pattern) in enumerate(patterns.items(), start=1):
        # Decompose
        wav = haar_decompose_2d(pattern, levels=3)

        # Pattern image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(pattern, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name, fontsize=8, pad=2)
        ax.axis('off')

        # Level 0 (2px scale) subbands
        for col, (key, subband_name) in enumerate([('LH', 'LH'), ('HL', 'HL'), ('HH', 'HH')]):
            ax = fig.add_subplot(gs[row, col + 1])
            subband = wav[key][0]
            vmax = max(0.1, np.abs(subband).max())
            ax.imshow(subband, cmap='RdBu', vmin=-vmax, vmax=vmax)
            energy = compute_subband_energy(subband)
            ax.set_title(f'E={energy:.4f}', fontsize=7, pad=1)
            ax.axis('off')

        # Level 1 (4px scale) subbands
        for col, (key, subband_name) in enumerate([('LH', 'LH'), ('HL', 'HL'), ('HH', 'HH')]):
            ax = fig.add_subplot(gs[row, col + 4])
            subband = wav[key][1]
            vmax = max(0.1, np.abs(subband).max())
            ax.imshow(subband, cmap='RdBu', vmin=-vmax, vmax=vmax)
            energy = compute_subband_energy(subband)
            ax.set_title(f'E={energy:.4f}', fontsize=7, pad=1)
            ax.axis('off')

        # Energy profile bar chart
        ax = fig.add_subplot(gs[row, 7])
        energies = []
        labels = []
        for level in range(2):
            for key in ['LH', 'HL', 'HH']:
                energies.append(compute_subband_energy(wav[key][level]))
                labels.append(f'{key[0]}{level}')

        colors = ['#e74c3c', '#3498db', '#9b59b6'] * 2
        ax.barh(range(6), energies, color=colors)
        ax.set_yticks(range(6))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlim(0, max(energies) * 1.2 if max(energies) > 0.01 else 0.1)
        ax.tick_params(axis='x', labelsize=5)
        ax.invert_yaxis()

    # Title
    fig.suptitle('Wavelet Pattern Encoding Demonstration\n'
                 'LH = Horizontal detail (detects horizontal edges/lines)\n'
                 'HL = Vertical detail (detects vertical edges/lines)\n'
                 'HH = Diagonal detail (detects diagonal/checkerboard patterns)',
                 fontsize=12, fontweight='bold', y=0.995)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_selectivity_demo(output_path: Path, size: int = 128):
    """
    Create a focused demo showing subband selectivity with clear examples.
    """
    fig, axes = plt.subplots(4, 7, figsize=(16, 10))

    # Row labels
    row_labels = ['Horizontal\nPattern', 'Vertical\nPattern', 'Diagonal\nPattern', 'White\nNoise']
    col_labels = ['Pattern', 'LH 2px', 'HL 2px', 'HH 2px', 'LH 4px', 'HL 4px', 'HH 4px']

    # Generate specific patterns
    patterns = []

    # Horizontal lines (2px period) - should light up LH
    h_lines = np.zeros((size, size))
    h_lines[0::2, :] = 1.0
    patterns.append(('Horizontal Lines\n(2px period)', h_lines))

    # Vertical lines (2px period) - should light up HL
    v_lines = np.zeros((size, size))
    v_lines[:, 0::2] = 1.0
    patterns.append(('Vertical Lines\n(2px period)', v_lines))

    # Checkerboard (2px) - should light up HH
    checker = np.zeros((size, size))
    checker[0::2, 0::2] = 1.0
    checker[1::2, 1::2] = 1.0
    patterns.append(('Checkerboard\n(2px period)', checker))

    # White noise - should be uniform across all
    np.random.seed(42)
    noise = (np.random.random((size, size)) > 0.5).astype(np.float64)
    patterns.append(('White Noise', noise))

    # Column headers
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=10, fontweight='bold', pad=10)

    for row, (name, pattern) in enumerate(patterns):
        # Decompose
        wav = haar_decompose_2d(pattern, levels=2)

        # Pattern
        axes[row, 0].imshow(pattern, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center', labelpad=50)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Compute all energies for normalization
        all_energies = []
        for level in range(2):
            for key in ['LH', 'HL', 'HH']:
                all_energies.append(compute_subband_energy(wav[key][level]))
        max_energy = max(all_energies) if max(all_energies) > 0.001 else 0.1

        # Show subbands
        col = 1
        for level in range(2):
            for key in ['LH', 'HL', 'HH']:
                subband = wav[key][level]
                vmax = max(0.1, np.abs(subband).max())

                ax = axes[row, col]
                ax.imshow(subband, cmap='RdBu', vmin=-vmax, vmax=vmax)

                energy = compute_subband_energy(subband)
                rel_energy = energy / max_energy * 100 if max_energy > 0 else 0

                # Highlight high-energy subbands
                if rel_energy > 50:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)

                ax.set_title(f'{rel_energy:.0f}%', fontsize=8, color='green' if rel_energy > 50 else 'gray')
                ax.set_xticks([])
                ax.set_yticks([])
                col += 1

    fig.suptitle('Wavelet Subband Selectivity\n'
                 'Each pattern type activates specific subbands (highlighted in green)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_scale_demo(output_path: Path, size: int = 128):
    """
    Demonstrate how different scales appear at different wavelet levels.
    """
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))

    # Generate patterns at different scales
    patterns = []

    # Horizontal lines at 2px, 4px, 8px periods
    for period, label in [(2, '2px'), (4, '4px'), (8, '8px')]:
        p = np.zeros((size, size))
        for i in range(0, size, period):
            p[i:i+period//2, :] = 1.0
        patterns.append((f'Horizontal {label}', p))

    # Vertical lines at 2px, 4px, 8px periods
    for period, label in [(2, '2px'), (4, '4px'), (8, '8px')]:
        p = np.zeros((size, size))
        for i in range(0, size, period):
            p[:, i:i+period//2] = 1.0
        patterns.append((f'Vertical {label}', p))

    # Checkerboard at 2px, 4px, 8px
    for period, label in [(2, '2px'), (4, '4px'), (8, '8px')]:
        p = np.zeros((size, size))
        for i in range(0, size, period):
            for j in range(0, size, period):
                if ((i // period) + (j // period)) % 2 == 0:
                    p[i:i+period, j:j+period] = 1.0
        patterns.append((f'Checker {label}', p))

    # White noise as reference
    np.random.seed(42)
    noise = (np.random.random((size, size)) > 0.5).astype(np.float64)
    patterns.append(('White Noise', noise))

    col_labels = ['Pattern', 'Level 0 (2px)', 'Level 1 (4px)', 'Level 2 (8px)', 'Level 3 (16px)', 'Energy/Level']

    # Column headers
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=10, fontweight='bold', pad=10)

    # Layout: 4 rows, first 3 are H/V/Checker, last is noise
    row_configs = [
        (0, [('Horizontal 2px', patterns[0][1]),
             ('Horizontal 4px', patterns[1][1]),
             ('Horizontal 8px', patterns[2][1])]),
        (1, [('Vertical 2px', patterns[3][1]),
             ('Vertical 4px', patterns[4][1]),
             ('Vertical 8px', patterns[5][1])]),
        (2, [('Checker 2px', patterns[6][1]),
             ('Checker 4px', patterns[7][1]),
             ('Checker 8px', patterns[8][1])]),
        (3, [('White Noise', patterns[9][1])]),
    ]

    for row, (row_idx, row_patterns) in enumerate(row_configs):
        # Pick representative pattern for this row
        if len(row_patterns) > 1:
            name, pattern = row_patterns[1]  # Use middle scale
        else:
            name, pattern = row_patterns[0]

        # Decompose with 4 levels
        wav = haar_decompose_2d(pattern, levels=4)

        # Determine which subband to show based on pattern type
        if 'Horizontal' in name:
            key = 'LH'
            subband_label = 'LH'
        elif 'Vertical' in name:
            key = 'HL'
            subband_label = 'HL'
        else:
            key = 'HH'
            subband_label = 'HH'

        # Pattern column
        axes[row, 0].imshow(pattern, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(name.replace(' ', '\n'), fontsize=9,
                                 rotation=0, ha='right', va='center', labelpad=40)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Show relevant subband at each level
        energies = []
        for level in range(4):
            ax = axes[row, level + 1]
            subband = wav[key][level]
            vmax = max(0.05, np.abs(subband).max())
            ax.imshow(subband, cmap='RdBu', vmin=-vmax, vmax=vmax)

            energy = compute_subband_energy(subband)
            energies.append(energy)

            if level == 0:
                ax.set_ylabel(f'{subband_label}', fontsize=8)
            ax.set_title(f'E={energy:.4f}', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        # Energy bar chart
        ax = axes[row, 5]
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
        ax.bar(range(4), energies, color=colors)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'], fontsize=7)
        ax.set_ylabel('Energy', fontsize=7)
        ax.tick_params(axis='y', labelsize=6)

    fig.suptitle('Wavelet Multi-Scale Analysis\n'
                 'Pattern period determines which decomposition level has highest energy',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate wavelet pattern encoding demonstration')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('tools/test_wavelets/analysis'),
                        help='Output directory')
    parser.add_argument('--size', '-s', type=int, default=128,
                        help='Pattern size in pixels (default: 128)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate all demo sheets')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating wavelet pattern encoding demonstrations...")
    print()

    # Main comprehensive demo
    create_demo_sheet(args.output_dir / 'wavelet_pattern_demo.png', args.size)

    # Selectivity demo (cleaner, focused)
    create_selectivity_demo(args.output_dir / 'wavelet_selectivity_demo.png', args.size)

    # Multi-scale demo
    create_scale_demo(args.output_dir / 'wavelet_scale_demo.png', args.size)

    print()
    print("Done! Generated 3 demonstration sheets showing:")
    print("  1. wavelet_pattern_demo.png - Comprehensive pattern catalog")
    print("  2. wavelet_selectivity_demo.png - Subband selectivity (H/V/D)")
    print("  3. wavelet_scale_demo.png - Multi-scale analysis")


if __name__ == '__main__':
    main()
