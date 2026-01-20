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

def tent_corner_coeffs(idx: int) -> SymbolicCoeffs:
    """Corner C_idx at tent position 2*idx."""
    return (SymbolicCoeffs.unit(idx - 1) + SymbolicCoeffs.unit(idx)) * Fraction(1, 2)


def tent_center_coeffs(idx: int) -> SymbolicCoeffs:
    """Center M_idx at tent position 2*idx + 1 (volume-preserving)."""
    return (
        Fraction(3, 2) * SymbolicCoeffs.unit(idx)
        - Fraction(1, 4) * SymbolicCoeffs.unit(idx - 1)
        - Fraction(1, 4) * SymbolicCoeffs.unit(idx + 1)
    )


def tent_value_coeffs(tent_pos: int) -> SymbolicCoeffs:
    """Get symbolic coefficients for tent value at integer position."""
    if tent_pos % 2 == 0:
        return tent_corner_coeffs(tent_pos // 2)
    else:
        return tent_center_coeffs(tent_pos // 2)


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


KERNELS: dict[str, tuple[Callable[[float], float], float, str]] = {
    'box': (box_kernel, 1.0, "Box filter"),
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
    kernel_radius: float,
) -> SymbolicCoeffs:
    """
    Resample the tent surface at center_pos using a kernel of given width.

    Uses trapezoidal integration over integer tent positions within the kernel support.
    The kernel is evaluated at normalized distances (position / (width/2)).

    Args:
        center_pos: Center position in tent space (can be fractional)
        width: Kernel width in tent units
        kernel_func: The sampling kernel function
        kernel_radius: Base radius of the kernel (in normalized units)

    Returns:
        SymbolicCoeffs representing the resampled value
    """
    half_width = width / 2
    effective_radius = half_width * kernel_radius

    # Determine range of tent positions to sample
    start = int(math.floor(center_pos - effective_radius))
    end = int(math.ceil(center_pos + effective_radius))

    result = SymbolicCoeffs.zero()
    weight_sum = 0.0

    for pos in range(start, end + 1):
        # Normalized distance for kernel evaluation
        d = (pos - center_pos) / half_width
        w = kernel_func(d / kernel_radius) if kernel_radius > 0 else (1.0 if abs(d) < 0.5 else 0.0)

        if abs(w) > 1e-10:
            # Trapezoidal weighting: half weight at boundaries
            trap_weight = 0.5 if pos == start or pos == end else 1.0
            combined_weight = w * trap_weight

            result = result + tent_value_coeffs(pos) * combined_weight
            weight_sum += combined_weight

    # Normalize
    if weight_sum > 1e-10:
        result = result / weight_sum

    return result


def resample_tent_exact(center_pos: int, half_width: int) -> SymbolicCoeffs:
    """
    Exact resampling using trapezoidal rule for integer positions and widths.

    This matches the derivation in TENT-SPACE.md for the 2× downsample kernel.
    """
    result = SymbolicCoeffs.zero()

    for offset in range(-half_width, half_width + 1):
        pos = center_pos + offset
        # Trapezoidal weights: 0.5 at endpoints, 1.0 in middle
        weight = Fraction(1, 2) if abs(offset) == half_width else Fraction(1)
        result = result + tent_value_coeffs(pos) * weight

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
) -> SymbolicCoeffs:
    """
    Derive the effective direct kernel for tent-space downsampling.

    Full pipeline:
    1. Output tent positions are at 0, 1, 2 (for output pixel 0)
    2. These map to input tent positions: 0*ratio, 1*ratio, 2*ratio
    3. At each position, resample with the kernel
    4. Apply contraction weights (1/4, 1/2, 1/4) to get final value

    Args:
        ratio: Downsampling ratio (e.g., 2 for 2× downsample)
        offset: Offset in input pixels for the output pixel center
        kernel_name: Resampling kernel name ('box', 'triangle', etc.)
        kernel_width: Kernel width in tent units (default 2)

    Returns:
        SymbolicCoeffs representing the direct kernel
    """
    kernel_func, kernel_radius, _ = KERNELS.get(kernel_name, KERNELS['box'])

    # Output pixel 0 has tent positions 0, 1, 2
    # With offset, these shift by 2*offset in tent space
    # Mapping to input tent: multiply by ratio

    base_tent = 2 * offset  # Starting position in input tent space

    # Output tent positions mapped to input tent positions
    out_corner_left_pos = base_tent + 0 * ratio  # Output tent 0
    out_center_pos = base_tent + 1 * ratio       # Output tent 1
    out_corner_right_pos = base_tent + 2 * ratio # Output tent 2

    # Use exact computation for integer positions and widths
    use_exact = (
        ratio == int(ratio) and
        offset == int(offset) and
        kernel_width == int(kernel_width) and
        kernel_name == 'box'
    )

    if use_exact:
        r = int(ratio)
        o = int(offset)
        w = int(kernel_width)
        hw = w // 2

        out_corner_left = resample_tent_exact(2 * o + 0 * r, hw)
        out_center = resample_tent_exact(2 * o + 1 * r, hw)
        out_corner_right = resample_tent_exact(2 * o + 2 * r, hw)
    else:
        out_corner_left = resample_tent_trapezoidal(
            out_corner_left_pos, kernel_width, kernel_func, kernel_radius
        )
        out_center = resample_tent_trapezoidal(
            out_center_pos, kernel_width, kernel_func, kernel_radius
        )
        out_corner_right = resample_tent_trapezoidal(
            out_corner_right_pos, kernel_width, kernel_func, kernel_radius
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

    coeffs = derive_direct_kernel(ratio=2, offset=0, kernel_name='box', kernel_width=2)
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
          f"kernel={args.kernel}, width={args.width}")

    coeffs = derive_direct_kernel(
        ratio=args.ratio,
        offset=args.offset,
        kernel_name=args.kernel,
        kernel_width=args.width,
    )

    name = f"{args.ratio}× downsample with {args.kernel} kernel (width={args.width})"
    pretty_print_kernel(coeffs, name)


if __name__ == "__main__":
    main()
