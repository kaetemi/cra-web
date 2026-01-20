#!/usr/bin/env python3
"""
Tent-Space Kernel Derivation Tool

Derives effective direct kernels for arbitrary downsampling ratios by composing:
1. Box → Tent expansion (with volume-preserving sharpening)
2. Resampling with a 1D kernel in tent space
3. Tent → Box contraction

The key insight is that the tent-space pipeline involves THREE operations:
- EXPANSION: Each input pixel generates contributions to 3 tent positions
  (corner-center-corner), with centers adjusted for volume preservation
- RESAMPLING: The tent surface is sampled at new positions using a kernel
- CONTRACTION: Each output pixel integrates 3 tent positions (1/4, 1/2, 1/4)

For a ratio-R downsample:
- Output tent position j maps to input tent position j * R
- At each position, apply the sampling kernel with specified width (default 2)
- Apply contraction weights to get the final box-space value

Usage:
    python tent_kernel.py [--ratio R] [--offset O] [--kernel K] [--width W]

Examples:
    # 2× downsample with box kernel (matches TENT-SPACE.md)
    python tent_kernel.py --ratio 2 --kernel box --width 2

    # 3× downsample with Lanczos-2 kernel
    python tent_kernel.py --ratio 3 --kernel lanczos2 --width 4
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable
import argparse
import math


# =============================================================================
# Symbolic Coefficient Representation
# =============================================================================

@dataclass
class SymbolicCoeffs:
    """
    Represents a value as a weighted sum of source pixel values.
    coeffs[i] = weight of source pixel V_i
    """
    coeffs: dict[int, Fraction]

    @staticmethod
    def zero() -> 'SymbolicCoeffs':
        return SymbolicCoeffs({})

    @staticmethod
    def unit(idx: int) -> 'SymbolicCoeffs':
        return SymbolicCoeffs({idx: Fraction(1)})

    def __add__(self, other: 'SymbolicCoeffs') -> 'SymbolicCoeffs':
        result = dict(self.coeffs)
        for idx, coeff in other.coeffs.items():
            result[idx] = result.get(idx, Fraction(0)) + coeff
        return SymbolicCoeffs({k: v for k, v in result.items() if v != 0})

    def __sub__(self, other: 'SymbolicCoeffs') -> 'SymbolicCoeffs':
        return self + (other * Fraction(-1))

    def __mul__(self, scalar: Fraction | int | float) -> 'SymbolicCoeffs':
        if isinstance(scalar, float):
            scalar = Fraction(scalar).limit_denominator(1000000)
        elif isinstance(scalar, int):
            scalar = Fraction(scalar)
        return SymbolicCoeffs({k: v * scalar for k, v in self.coeffs.items() if v * scalar != 0})

    def __rmul__(self, scalar) -> 'SymbolicCoeffs':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Fraction | int | float) -> 'SymbolicCoeffs':
        if isinstance(scalar, float):
            scalar = Fraction(scalar).limit_denominator(1000000)
        elif isinstance(scalar, int):
            scalar = Fraction(scalar)
        return SymbolicCoeffs({k: v / scalar for k, v in self.coeffs.items()})

    def min_index(self) -> int:
        return min(self.coeffs.keys()) if self.coeffs else 0

    def max_index(self) -> int:
        return max(self.coeffs.keys()) if self.coeffs else 0

    def sum_coeffs(self) -> Fraction:
        return sum(self.coeffs.values(), Fraction(0))

    def normalize(self) -> 'SymbolicCoeffs':
        s = self.sum_coeffs()
        if s != 0:
            return self / s
        return self


# =============================================================================
# 1D Tent-Space Expansion (Box → Tent)
# =============================================================================
#
# Tent space layout for source pixels V_0, V_1, V_2, ...:
#
#   Tent position:  0    1    2    3    4    5    6   ...
#                   C_0  M_0  C_1  M_1  C_2  M_2  C_3 ...
#
# Where:
#   C_i = corner at position 2i = (V_{i-1} + V_i) / 2
#   M_i = center at position 2i+1 = 3/2*V_i - 1/4*V_{i-1} - 1/4*V_{i+1}

def tent_corner_coeffs_from(idx: int, source_func: Callable[[int], SymbolicCoeffs]) -> SymbolicCoeffs:
    """Corner C_idx at tent position 2*idx, using source_func for input values."""
    return (source_func(idx - 1) + source_func(idx)) * Fraction(1, 2)


def tent_center_coeffs_from(idx: int, source_func: Callable[[int], SymbolicCoeffs]) -> SymbolicCoeffs:
    """Center M_idx at tent position 2*idx + 1 (volume-preserving), using source_func for input values."""
    return (
        Fraction(3, 2) * source_func(idx)
        - Fraction(1, 4) * source_func(idx - 1)
        - Fraction(1, 4) * source_func(idx + 1)
    )


def tent_value_coeffs_from(tent_pos: int, source_func: Callable[[int], SymbolicCoeffs]) -> SymbolicCoeffs:
    """Get symbolic coefficients for tent value at integer position, using source_func for input values."""
    if tent_pos % 2 == 0:
        return tent_corner_coeffs_from(tent_pos // 2, source_func)
    else:
        return tent_center_coeffs_from(tent_pos // 2, source_func)


def tent_corner_coeffs(idx: int) -> SymbolicCoeffs:
    """Corner C_idx at tent position 2*idx."""
    return tent_corner_coeffs_from(idx, SymbolicCoeffs.unit)


def tent_center_coeffs(idx: int) -> SymbolicCoeffs:
    """Center M_idx at tent position 2*idx + 1 (volume-preserving)."""
    return tent_center_coeffs_from(idx, SymbolicCoeffs.unit)


def tent_value_coeffs(tent_pos: int) -> SymbolicCoeffs:
    """Get symbolic coefficients for tent value at integer position."""
    return tent_value_coeffs_from(tent_pos, SymbolicCoeffs.unit)


def make_recursive_tent_func(levels: int) -> Callable[[int], SymbolicCoeffs]:
    """
    Create a function that returns coefficients for a recursively expanded tent space.

    Level 0: Original box pixels (SymbolicCoeffs.unit)
    Level 1: Single tent expansion (tent_value_coeffs)
    Level 2: Tent of tent (tent expansion applied to level 1)
    ...

    Each level doubles the resolution: position p at level N maps to position p*2^N in box space.
    """
    if levels <= 0:
        return SymbolicCoeffs.unit

    # Build up the function recursively
    # Level 1 uses SymbolicCoeffs.unit as source
    # Level 2 uses level 1's function as source, etc.

    # Cache to avoid recomputation
    cache: dict[tuple[int, int], SymbolicCoeffs] = {}

    def get_level_value(level: int, pos: int) -> SymbolicCoeffs:
        """Get the value at position `pos` at recursion `level`."""
        if level <= 0:
            return SymbolicCoeffs.unit(pos)

        key = (level, pos)
        if key in cache:
            return cache[key]

        # This level's value is a tent expansion of the previous level
        prev_level_func = lambda p: get_level_value(level - 1, p)
        result = tent_value_coeffs_from(pos, prev_level_func)
        cache[key] = result
        return result

    return lambda pos: get_level_value(levels, pos)


# =============================================================================
# 1D Sampling Kernels
# =============================================================================

def box_kernel(x: float) -> float:
    """Box kernel: constant 1 within [-0.5, 0.5]."""
    return 1.0 if abs(x) <= 0.5 else 0.0


def triangle_kernel(x: float) -> float:
    """Triangle (bilinear) kernel: linear falloff."""
    x = abs(x)
    return max(0.0, 1.0 - x)


def lanczos_kernel(x: float, a: int = 2) -> float:
    """Lanczos kernel with parameter a."""
    if abs(x) < 1e-10:
        return 1.0
    if abs(x) >= a:
        return 0.0
    pi_x = math.pi * x
    return (math.sin(pi_x) / pi_x) * (math.sin(pi_x / a) / (pi_x / a))


def sinc_kernel(x: float) -> float:
    """Pure sinc kernel."""
    if abs(x) < 1e-10:
        return 1.0
    pi_x = math.pi * x
    return math.sin(pi_x) / pi_x


def mitchell_kernel(x: float) -> float:
    """Mitchell-Netravali (B=C=1/3)."""
    x = abs(x)
    if x >= 2.0:
        return 0.0
    if x >= 1.0:
        return ((-7/18) * x**3 + 2 * x**2 - (10/3) * x + 16/9)
    return ((7/6) * x**3 - 2 * x**2 + 8/9)


# KERNELS: (function, support_half_width, description)
# support_half_width is the half-width of the kernel's natural support
# e.g., box returns 1 for |x| <= 0.5, so support_half_width = 0.5
KERNELS: dict[str, tuple[Callable[[float], float], float, str]] = {
    'box': (box_kernel, 0.5, "Box filter"),
    'triangle': (triangle_kernel, 1.0, "Triangle/bilinear"),
    'lanczos2': (lambda x: lanczos_kernel(x, 2), 2.0, "Lanczos a=2"),
    'lanczos3': (lambda x: lanczos_kernel(x, 3), 3.0, "Lanczos a=3"),
    'sinc': (sinc_kernel, 8.0, "Pure sinc (truncated)"),
    'mitchell': (mitchell_kernel, 2.0, "Mitchell-Netravali"),
}


# =============================================================================
# Resampling in Tent Space
# =============================================================================

def resample_tent_trapezoidal(
    center_pos: float,
    width: float,
    kernel_func: Callable[[float], float],
    support_half_width: float,
    tent_value_func: Callable[[int], SymbolicCoeffs] | None = None,
) -> SymbolicCoeffs:
    """
    Resample the tent surface at center_pos using a kernel of given width.

    Uses proper area-weighted integration: each tent position contributes
    proportionally to its overlap with the sampling interval.

    Args:
        center_pos: Center position in tent space (can be fractional)
        width: Kernel width in tent units (total extent of sampling)
        kernel_func: The sampling kernel function
        support_half_width: Half-width of the kernel's natural support domain
        tent_value_func: Function to get tent value coefficients (default: tent_value_coeffs)

    Returns:
        SymbolicCoeffs representing the resampled value
    """
    if tent_value_func is None:
        tent_value_func = tent_value_coeffs

    half_width = width / 2
    interval_start = center_pos - half_width
    interval_end = center_pos + half_width

    # Sample all integer positions that could overlap with the interval
    # Each tent position i "owns" the range [i - 0.5, i + 0.5] for integration
    # Add padding of 3 to catch any edge cases and verify they're zero
    PADDING = 3
    start = int(math.floor(interval_start + 0.5)) - PADDING
    end = int(math.floor(interval_end + 0.5)) + PADDING

    result = SymbolicCoeffs.zero()
    weight_sum = 0.0

    for pos in range(start, end + 1):
        # Compute overlap between [interval_start, interval_end] and [pos - 0.5, pos + 0.5]
        overlap_start = max(interval_start, pos - 0.5)
        overlap_end = min(interval_end, pos + 0.5)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap < 1e-10:
            continue

        # For box filter, the weight is just the overlap (box is constant 1.0 within its support)
        # For other kernels, weight is overlap × kernel value at position center
        if kernel_func == box_kernel:
            combined_weight = overlap
        else:
            # Map position to kernel's natural domain for kernel weighting
            kernel_arg = (pos - center_pos) / half_width * support_half_width if half_width > 1e-10 else 0.0
            kernel_weight = kernel_func(kernel_arg)

            if abs(kernel_weight) < 1e-10:
                continue

            # Combined weight: overlap area × kernel weight
            combined_weight = overlap * kernel_weight

        result = result + tent_value_func(pos) * combined_weight
        weight_sum += combined_weight

    # Normalize
    if weight_sum > 1e-10:
        result = result / weight_sum

    return result


def resample_tent_exact(
    center_pos: int,
    half_width: int,
    tent_value_func: Callable[[int], SymbolicCoeffs] | None = None,
) -> SymbolicCoeffs:
    """
    Exact resampling using trapezoidal rule for integer positions and widths.

    This matches the derivation in TENT-SPACE.md for the 2× downsample kernel.
    For half_width=0, returns point sampling at center_pos.
    """
    if tent_value_func is None:
        tent_value_func = tent_value_coeffs

    # Point sampling when width=0
    if half_width == 0:
        return tent_value_func(center_pos)

    result = SymbolicCoeffs.zero()

    for offset in range(-half_width, half_width + 1):
        pos = center_pos + offset
        # Trapezoidal weights: 0.5 at endpoints, 1.0 in middle
        weight = Fraction(1, 2) if abs(offset) == half_width else Fraction(1)
        result = result + tent_value_func(pos) * weight

    # Normalize by total weight (2 * half_width)
    result = result / (2 * half_width)

    return result


# =============================================================================
# Full Pipeline: Expand → Resample → Contract
# =============================================================================

def contract_tent_values(corner_left: SymbolicCoeffs, center: SymbolicCoeffs, corner_right: SymbolicCoeffs) -> SymbolicCoeffs:
    """Apply tent contraction: 1/4 * corner_left + 1/2 * center + 1/4 * corner_right."""
    return (
        corner_left * Fraction(1, 4) +
        center * Fraction(1, 2) +
        corner_right * Fraction(1, 4)
    )


def derive_direct_kernel(
    ratio: float,
    offset: float = 0.0,
    kernel_name: str = 'box',
    kernel_width: float = 2.0,
    recurse: int = 1,
) -> SymbolicCoeffs:
    """
    Derive the effective direct kernel for tent-space downsampling.

    Full pipeline for recurse=N:
    1. Expand box → tent N times (2^N resolution increase)
    2. Resample at the finest tent level (level N) with the kernel
    3. Contract N times back to box space

    Each contraction needs 3 inputs (corner, center, corner).
    Recurse=1: 3 samples, recurse=2: 3×3=9 samples, recurse=N: 3^N samples.

    Args:
        ratio: Downsampling ratio (e.g., 2 for 2× downsample)
        offset: Offset of output pixel center from input position 0.
        kernel_name: Resampling kernel name ('box', 'triangle', etc.)
        kernel_width: Kernel width in tent units (default 2)
        recurse: Number of tent expansion/contraction levels (default 1).

    Returns:
        SymbolicCoeffs representing the direct kernel
    """
    kernel_func, kernel_radius, _ = KERNELS.get(kernel_name, KERNELS['box'])

    # We'll sample from the deepest tent level and contract back
    deepest_tent_func = make_recursive_tent_func(recurse)

    def resample_at_deepest(pos: float, width: float) -> SymbolicCoeffs:
        """Resample the deepest tent surface at a given position."""
        # Use exact (trapezoidal) only when:
        # - Position is integer
        # - Width is even integer (trapezoidal matches box overlap for even widths)
        # - Kernel is box
        # For odd widths, trapezoidal rule gives wrong weights, so use overlap method
        use_exact = (
            pos == int(pos) and
            width == int(width) and
            int(width) % 2 == 0 and  # Only for even widths
            kernel_name == 'box'
        )

        if use_exact:
            hw = int(width) // 2
            return resample_tent_exact(int(pos), hw, deepest_tent_func)
        else:
            return resample_tent_trapezoidal(pos, width, kernel_func, kernel_radius, deepest_tent_func)

    def derive_recursive(levels_remaining: int, center_pos: float, spacing: float, width: float) -> SymbolicCoeffs:
        """
        Recursively derive kernel by sampling and contracting.

        At each level we need 3 positions (corner, center, corner) spaced by `spacing`.
        - If levels_remaining == 1: sample from deepest tent, contract once
        - If levels_remaining > 1: recurse to get 3 values, then contract

        Position and spacing are in the CURRENT level's coordinates.
        """
        # The 3 positions for contraction at this level
        left_pos = center_pos - spacing
        right_pos = center_pos + spacing

        if levels_remaining == 1:
            # Base case: sample from deepest tent level at these positions
            # Positions are already in deepest level coordinates
            samples = [
                resample_at_deepest(left_pos, width),
                resample_at_deepest(center_pos, width),
                resample_at_deepest(right_pos, width),
            ]
            return contract_tent_values(samples[0], samples[1], samples[2])
        else:
            # Recursive case: each position needs to be computed from finer level
            # When we go one level deeper, positions double and spacing doubles
            # (because the deeper level has 2× resolution)

            corner_left = derive_recursive(
                levels_remaining - 1,
                left_pos * 2 + 1,    # Map to deeper level: pos → 2*pos + 1
                spacing * 2,          # Spacing doubles at deeper level
                width                 # Width stays constant (same physical extent)
            )
            center = derive_recursive(
                levels_remaining - 1,
                center_pos * 2 + 1,
                spacing * 2,
                width
            )
            corner_right = derive_recursive(
                levels_remaining - 1,
                right_pos * 2 + 1,
                spacing * 2,
                width
            )

            return contract_tent_values(corner_left, center, corner_right)

    # Starting position: center of output pixel in L1 tent space
    # Output pixel centered at input position `offset`
    # In L1 tent space, pixel i's center is at position 2*i + 1
    center_L1 = 2 * offset + 1

    # Spacing in L1: for ratio R downsampling, corner positions are at center ± R
    spacing_L1 = ratio

    # Start the recursion from L1, going deeper
    result = derive_recursive(recurse, center_L1, spacing_L1, kernel_width)

    return result


# =============================================================================
# Output Formatting
# =============================================================================

def pretty_print_kernel(coeffs: SymbolicCoeffs, name: str = "Kernel"):
    """Pretty print a kernel in various formats."""
    if not coeffs.coeffs:
        print(f"{name}: 0")
        return

    min_idx = coeffs.min_index()
    max_idx = coeffs.max_index()

    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)

    # Find common denominator
    denoms = [c.denominator for c in coeffs.coeffs.values() if c != 0]
    if denoms:
        from math import lcm
        from functools import reduce
        common_denom = reduce(lcm, denoms)

        print(f"\nInteger coefficients (÷{common_denom}):")
        int_coeffs = []
        for i in range(min_idx, max_idx + 1):
            c = coeffs.coeffs.get(i, Fraction(0))
            int_c = int(c * common_denom)
            int_coeffs.append(int_c)

        print(f"  [{', '.join(map(str, int_coeffs))}]")
        print(f"  Sum: {sum(int_coeffs)} (should be {common_denom})")

    # Floating point
    print("\nFloating point coefficients:")
    float_coeffs = []
    for i in range(min_idx, max_idx + 1):
        c = coeffs.coeffs.get(i, Fraction(0))
        float_coeffs.append(float(c))
    print(f"  [{', '.join(f'{x:.6f}' for x in float_coeffs)}]")

    # Sum check
    total = coeffs.sum_coeffs()
    print(f"\n  Sum: {float(total):.10f}")

    # Index mapping
    print(f"\nIndex mapping (relative to center of output pixel):")
    for i in range(min_idx, max_idx + 1):
        c = coeffs.coeffs.get(i, Fraction(0))
        if c != 0:
            print(f"  V[{i:+d}]: {c} = {float(c):.6f}")


def verify_against_known():
    """Verify against the known 2× kernel from TENT-SPACE.md."""
    print("="*70)
    print("VERIFICATION AGAINST TENT-SPACE.md AND BRUTEFORCE")
    print("="*70)

    print("\nExpected 2× downsample kernel from docs:")
    print("  1D: [-1, 7, 26, 26, 7, -1] / 64")

    # The standard centering: output pixel 0 covers input [0, ratio), centered at ratio/2
    # offset = (ratio - 1) / 2, width = ratio
    coeffs = derive_direct_kernel(ratio=2, offset=0.5, kernel_name='box', kernel_width=2)
    pretty_print_kernel(coeffs, "Derived 2× kernel (box, width=2, offset=0.5)")

    # Check if matches
    min_i = coeffs.min_index()
    max_i = coeffs.max_index()
    expected = [-1, 7, 26, 26, 7, -1]
    actual = [int(coeffs.coeffs.get(i, 0) * 64) for i in range(min_i, max_i + 1)]

    print("\n" + "-"*70)
    if actual == expected:
        print("✓ MATCH! Derived 2× kernel matches TENT-SPACE.md exactly.")
    else:
        print(f"✗ MISMATCH")
        print(f"  Expected: {expected}")
        print(f"  Got:      {actual}")

    print("\n" + "="*70)
    print("Additional Examples (matching bruteforce)")
    print("="*70)

    # 3× downsample - offset=(3-1)/2=1, width=3
    print("\n3× downsample (offset=1, width=3):")
    print("  Expected from bruteforce: [-1, 7, 9, 18, 9, 7, -1] / 48")
    coeffs3 = derive_direct_kernel(ratio=3, offset=1, kernel_name='box', kernel_width=3)
    pretty_print_kernel(coeffs3, "3× kernel")

    expected3 = [-1, 7, 9, 18, 9, 7, -1]
    min_i3 = coeffs3.min_index()
    max_i3 = coeffs3.max_index()
    actual3 = [int(coeffs3.coeffs.get(i, 0) * 48) for i in range(min_i3, max_i3 + 1)]
    if actual3 == expected3:
        print("✓ MATCH! 3× kernel matches bruteforce.")
    else:
        print(f"✗ MISMATCH: expected {expected3}, got {actual3}")

    # 4× downsample - offset=(4-1)/2=1.5, width=4
    print("\n4× downsample (offset=1.5, width=4):")
    print("  Expected from bruteforce: [1, 1, 2, 2, 1, 1] / 8")
    coeffs4 = derive_direct_kernel(ratio=4, offset=1.5, kernel_name='box', kernel_width=4)
    pretty_print_kernel(coeffs4, "4× kernel")

    expected4 = [1, 1, 2, 2, 1, 1]
    min_i4 = coeffs4.min_index()
    max_i4 = coeffs4.max_index()
    actual4 = [int(coeffs4.coeffs.get(i, 0) * 8) for i in range(min_i4, max_i4 + 1)]
    if actual4 == expected4:
        print("✓ MATCH! 4× kernel matches bruteforce.")
    else:
        print(f"✗ MISMATCH: expected {expected4}, got {actual4}")

    # 8× downsample - offset=(8-1)/2=3.5, width=8
    print("\n8× downsample (offset=3.5, width=8):")
    print("  Expected from bruteforce: [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1] / 16")
    coeffs8 = derive_direct_kernel(ratio=8, offset=3.5, kernel_name='box', kernel_width=8)
    pretty_print_kernel(coeffs8, "8× kernel")

    expected8 = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]
    min_i8 = coeffs8.min_index()
    max_i8 = coeffs8.max_index()
    actual8 = [int(coeffs8.coeffs.get(i, 0) * 16) for i in range(min_i8, max_i8 + 1)]
    if actual8 == expected8:
        print("✓ MATCH! 8× kernel matches bruteforce.")
    else:
        print(f"✗ MISMATCH: expected {expected8}, got {actual8}")


def main():
    parser = argparse.ArgumentParser(
        description="Derive tent-space sampling kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Derive 2× downsample kernel (uses default offset and width)
  python tent_kernel.py --ratio 2

  # 3× downsample (uses default offset=1, width=3)
  python tent_kernel.py --ratio 3

  # 4× with custom offset
  python tent_kernel.py --ratio 4 --offset 2.0

  # 2× with triangle (bilinear) kernel
  python tent_kernel.py --ratio 2 --kernel triangle

  # Verify against documented kernel
  python tent_kernel.py --verify
"""
    )
    parser.add_argument('--ratio', '-r', type=float, default=2.0,
                       help="Downsampling ratio (default: 2.0)")
    parser.add_argument('--offset', '-o', type=float, default=None,
                       help="Pixel offset. Default: (ratio-1)/2 to center output pixel 0 "
                            "over input pixels [0, ratio).")
    parser.add_argument('--kernel', '-k', type=str, default='box',
                       choices=list(KERNELS.keys()),
                       help="Sampling kernel (default: box)")
    parser.add_argument('--width', '-w', type=float, default=None,
                       help="Kernel width in tent units. Default: ratio (matches filter_scale "
                            "for proper coverage of output pixel footprint).")
    parser.add_argument('--recurse', '-R', type=int, default=1,
                       help="Tent expansion recursion levels (default: 1). "
                            "Higher values create finer grids (2^N resolution).")
    parser.add_argument('--verify', '-v', action='store_true',
                       help="Verify against known 2× kernel from TENT-SPACE.md")
    parser.add_argument('--list-kernels', '-l', action='store_true',
                       help="List available kernels")

    args = parser.parse_args()

    if args.list_kernels:
        print("Available kernels:")
        for name, (_, radius, desc) in KERNELS.items():
            print(f"  {name:12s} (radius={radius:.1f}): {desc}")
        return

    if args.verify:
        verify_against_known()
        return

    # Apply defaults based on ratio
    offset = args.offset if args.offset is not None else (args.ratio - 1) / 2
    width = args.width if args.width is not None else args.ratio

    print(f"Parameters: ratio={args.ratio}, offset={offset}, "
          f"kernel={args.kernel}, width={width}, recurse={args.recurse}")

    coeffs = derive_direct_kernel(
        ratio=args.ratio,
        offset=offset,
        kernel_name=args.kernel,
        kernel_width=width,
        recurse=args.recurse,
    )

    recurse_str = f", {args.recurse}× tent recursion" if args.recurse > 1 else ""
    name = f"{args.ratio}× downsample with {args.kernel} kernel (width={width}{recurse_str})"
    pretty_print_kernel(coeffs, name)


if __name__ == "__main__":
    main()
