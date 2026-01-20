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

def derive_direct_kernel(
    ratio: float,
    offset: float = 0.0,
    kernel_name: str = 'box',
    kernel_width: float = 2.0,
    recurse: int = 1,
) -> SymbolicCoeffs:
    """
    Derive the effective direct kernel for tent-space downsampling.

    Full pipeline:
    1. Expand box → tent (potentially multiple times with recurse > 1)
    2. Output tent positions are at corner_left, center, corner_right
    3. At each position, resample with the kernel
    4. Apply contraction weights (1/4, 1/2, 1/4) to get final value

    Args:
        ratio: Downsampling ratio (e.g., 2 for 2× downsample)
        offset: Offset of output pixel center from input position 0.
                Default 0 means centered on position 0.
        kernel_name: Resampling kernel name ('box', 'triangle', etc.)
        kernel_width: Kernel width in tent units (default 2)
        recurse: Number of tent expansion levels (default 1).
                 Higher values create finer grids before resampling.
                 Level N has 2^N times the resolution of box space.

    Returns:
        SymbolicCoeffs representing the direct kernel
    """
    kernel_func, kernel_radius, _ = KERNELS.get(kernel_name, KERNELS['box'])

    # Create the tent value function for the specified recursion level
    tent_value_func = make_recursive_tent_func(recurse)

    # Scale factor for positions: each recursion level doubles the resolution
    # At level N, tent positions are at 2^N times the box-space resolution
    scale = 2 ** recurse

    # Output pixel centered at input position `offset`
    # In box space, pixel i is centered at position i (integer coordinates)
    # At recursion level N, the center of pixel i is at position:
    #   scale * i + (scale - 1) / 2 for odd scale (which is never the case)
    #   Actually: position = scale * (2*i + 1) / 2 = scale*i + scale/2
    # For level 1 (scale=2): center of pixel 0 is at position 1
    # For level 2 (scale=4): center of pixel 0 is at position 2
    # General: center_tent = scale * offset + scale / 2 = scale * (offset + 0.5)
    # But we want integer positions, so: center_tent = scale * offset + (scale // 2)
    # Wait, let me think again...
    #
    # Level 1: positions 0,1,2,3,4... where odd positions are centers
    #   Pixel 0 center is at position 1
    # Level 2: positions 0,1,2,3,4,5,6,7,8...
    #   Level 1's position 0 (corner) expands to positions 0,1 with corner at 0, center at 1
    #   Level 1's position 1 (center) expands to positions 2,3 with corner at 2, center at 3
    #   So pixel 0's center (level 1 pos 1) becomes level 2 pos 3
    # General pattern: level N center of pixel i is at position (2i+1) * 2^(N-1)
    #   = (2*0+1) * 2^0 = 1 for level 1
    #   = (2*0+1) * 2^1 = 2 for level 2? No, we said it's 3...
    #
    # Let me trace more carefully:
    # Level 1: tent_pos maps to box_idx via: corner at 2i, center at 2i+1
    #   Box pixel 0: center at tent_pos = 1
    # Level 2: tent_pos at level 2 maps to tent_pos at level 1 via same rule
    #   Level 1 tent_pos 0: corner at level 2 tent_pos 0, center at 1
    #   Level 1 tent_pos 1: corner at level 2 tent_pos 2, center at 3
    #   So box pixel 0's center (L1 pos 1) → L2 pos 3 (the center of L1 pos 1)
    #
    # Pattern: for level N, pixel i's center is at: (2i+1) * 2^(N-1) + 2^(N-1) - 1
    #   = (2i+1) * 2^(N-1) + 2^(N-1) - 1
    # For i=0, N=1: 1 * 1 + 1 - 1 = 1 ✓
    # For i=0, N=2: 1 * 2 + 2 - 1 = 3 ✓
    # Simplify: 2^(N-1) * (2i + 2) - 1 = 2^N * (i + 1) - 1
    # For i=0: 2^N - 1
    # Hmm, let's verify: N=1: 2-1=1 ✓, N=2: 4-1=3 ✓
    #
    # So center_tent for pixel at offset is: 2^recurse * (offset + 1) - 1
    #   = scale * (offset + 1) - 1
    #   = scale * offset + scale - 1

    center_tent = scale * offset + scale - 1

    # Ratio in tent space is also scaled
    tent_ratio = ratio * scale / 2  # Divide by 2 because each tent level doubles resolution

    # Output tent positions: corner_left, center, corner_right
    # These are spaced by `tent_ratio` in the final tent space
    out_corner_left_pos = center_tent - tent_ratio
    out_center_pos = center_tent
    out_corner_right_pos = center_tent + tent_ratio

    # Kernel width also scales with recursion level
    tent_kernel_width = kernel_width * scale / 2

    # Use exact computation for integer positions and widths
    use_exact = (
        tent_ratio == int(tent_ratio) and
        (center_tent - tent_ratio) == int(center_tent - tent_ratio) and
        tent_kernel_width == int(tent_kernel_width) and
        kernel_name == 'box'
    )

    if use_exact:
        w = int(tent_kernel_width)
        hw = w // 2

        out_corner_left = resample_tent_exact(int(out_corner_left_pos), hw, tent_value_func)
        out_center = resample_tent_exact(int(out_center_pos), hw, tent_value_func)
        out_corner_right = resample_tent_exact(int(out_corner_right_pos), hw, tent_value_func)
    else:
        out_corner_left = resample_tent_trapezoidal(
            out_corner_left_pos, tent_kernel_width, kernel_func, kernel_radius, tent_value_func
        )
        out_center = resample_tent_trapezoidal(
            out_center_pos, tent_kernel_width, kernel_func, kernel_radius, tent_value_func
        )
        out_corner_right = resample_tent_trapezoidal(
            out_corner_right_pos, tent_kernel_width, kernel_func, kernel_radius, tent_value_func
        )

    # Contraction: 1/4 * corner_left + 1/2 * center + 1/4 * corner_right
    result = (
        out_corner_left * Fraction(1, 4) +
        out_center * Fraction(1, 2) +
        out_corner_right * Fraction(1, 4)
    )

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
    print("VERIFICATION AGAINST TENT-SPACE.md")
    print("="*70)

    print("\nExpected 2× downsample kernel from docs:")
    print("  1D: [-1, 7, 26, 26, 7, -1] / 64")
    print("  (centered between pixels, i.e., offset=-0.5)")

    # The documented kernel is centered between two pixels (offset=-0.5)
    # This is the natural position for 2× downsampling where output pixel
    # covers input pixels -1 and 0 symmetrically
    coeffs = derive_direct_kernel(ratio=2, offset=-0.5, kernel_name='box', kernel_width=2)
    pretty_print_kernel(coeffs, "Derived 2× kernel (box, width=2)")

    # Check if matches
    min_i = coeffs.min_index()
    max_i = coeffs.max_index()
    expected = [-1, 7, 26, 26, 7, -1]
    actual = [int(coeffs.coeffs.get(i, 0) * 64) for i in range(min_i, max_i + 1)]

    print("\n" + "-"*70)
    if actual == expected:
        print("✓ MATCH! Derived kernel matches TENT-SPACE.md exactly.")
    else:
        print(f"✗ MISMATCH")
        print(f"  Expected: {expected}")
        print(f"  Got:      {actual}")

    print("\n" + "="*70)
    print("Additional Examples")
    print("="*70)

    # 3× downsample
    print("\n3× downsample with box kernel (width=2):")
    coeffs3 = derive_direct_kernel(ratio=3, offset=0, kernel_name='box', kernel_width=2)
    pretty_print_kernel(coeffs3, "3× kernel")

    # 4× downsample
    print("\n4× downsample with box kernel (width=2):")
    coeffs4 = derive_direct_kernel(ratio=4, offset=0, kernel_name='box', kernel_width=2)
    pretty_print_kernel(coeffs4, "4× kernel")


def main():
    parser = argparse.ArgumentParser(
        description="Derive tent-space sampling kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Derive 2× downsample kernel (matches TENT-SPACE.md)
  python tent_kernel.py --ratio 2

  # 3× downsample with wider kernel
  python tent_kernel.py --ratio 3 --width 4

  # 2× with triangle (bilinear) kernel
  python tent_kernel.py --ratio 2 --kernel triangle --width 2

  # Verify against documented kernel
  python tent_kernel.py --verify
"""
    )
    parser.add_argument('--ratio', '-r', type=float, default=2.0,
                       help="Downsampling ratio (default: 2.0)")
    parser.add_argument('--offset', '-o', type=float, default=0.0,
                       help="Pixel offset (default: 0.0)")
    parser.add_argument('--kernel', '-k', type=str, default='box',
                       choices=list(KERNELS.keys()),
                       help="Sampling kernel (default: box)")
    parser.add_argument('--width', '-w', type=float, default=2.0,
                       help="Kernel width in tent units (default: 2.0)")
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

    print(f"Parameters: ratio={args.ratio}, offset={args.offset}, "
          f"kernel={args.kernel}, width={args.width}, recurse={args.recurse}")

    coeffs = derive_direct_kernel(
        ratio=args.ratio,
        offset=args.offset,
        kernel_name=args.kernel,
        kernel_width=args.width,
        recurse=args.recurse,
    )

    recurse_str = f", {args.recurse}× tent recursion" if args.recurse > 1 else ""
    name = f"{args.ratio}× downsample with {args.kernel} kernel (width={args.width}{recurse_str})"
    pretty_print_kernel(coeffs, name)


if __name__ == "__main__":
    main()
