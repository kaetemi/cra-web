#!/usr/bin/env python3
"""
2D Tent-Space Kernel Derivation Tool

Derives effective direct kernels for 2D downsampling by composing:
1. Box → Tent expansion (with volume-preserving sharpening)
2. Resampling with a 2D kernel in tent space
3. Tent → Box contraction

For 2D, the tent-space layout is:
- Positions where both indices are even: corners (4-way average)
- Positions where one index is even, one odd: edges (2-way average)
- Positions where both indices are odd: centers (volume-preserving)

Integration weights for contraction (3×3 neighborhood):
    1/16  1/8  1/16
    1/8   1/4  1/8
    1/16  1/8  1/16

Usage:
    python tent_kernel_2d.py [--ratio R] [--offset O] [--kernel K] [--width W]

Examples:
    # 2× downsample with box kernel
    python tent_kernel_2d.py --ratio 2 --kernel box --width 2

    # 3× downsample with Lanczos-3 kernel
    python tent_kernel_2d.py --ratio 3 --kernel lanczos3

    # 2× downsample with EWA lanczos3-jinc (non-separable)
    python tent_kernel_2d.py --ratio 2 --kernel ewa-lanczos3-jinc
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable
import argparse
import math

# Try to import scipy for Bessel functions, fall back to approximation
try:
    from scipy.special import j1 as bessel_j1
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

    def bessel_j1(x: float) -> float:
        """Approximation of Bessel function J1(x) for when scipy is not available."""
        x = abs(x)
        if x < 8.0:
            y = x * x
            j1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
                + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
            j1 /= (144725228442.0 + y * (2300535178.0 + y * (18583304.74
                + y * (99447.43394 + y * (376.9991397 + y)))))
            return j1
        else:
            z = 8.0 / x
            y = z * z
            xx = x - 2.356194491
            f1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
            f2 = 0.04687499995 + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
            return math.sqrt(0.636619772 / x) * (math.cos(xx) * f1 - z * math.sin(xx) * f2)


# =============================================================================
# Symbolic Coefficient Representation (2D)
# =============================================================================

@dataclass
class SymbolicCoeffs2D:
    """
    Represents a value as a weighted sum of 2D source pixel values.
    coeffs[(y, x)] = weight of source pixel V_{y,x}
    """
    coeffs: dict[tuple[int, int], Fraction]

    @staticmethod
    def zero() -> 'SymbolicCoeffs2D':
        return SymbolicCoeffs2D({})

    @staticmethod
    def unit(y: int, x: int) -> 'SymbolicCoeffs2D':
        return SymbolicCoeffs2D({(y, x): Fraction(1)})

    def __add__(self, other: 'SymbolicCoeffs2D') -> 'SymbolicCoeffs2D':
        result = dict(self.coeffs)
        for idx, coeff in other.coeffs.items():
            result[idx] = result.get(idx, Fraction(0)) + coeff
        return SymbolicCoeffs2D({k: v for k, v in result.items() if v != 0})

    def __sub__(self, other: 'SymbolicCoeffs2D') -> 'SymbolicCoeffs2D':
        return self + (other * Fraction(-1))

    def __mul__(self, scalar: Fraction | int | float) -> 'SymbolicCoeffs2D':
        if isinstance(scalar, float):
            scalar = Fraction(scalar).limit_denominator(1000000)
        elif isinstance(scalar, int):
            scalar = Fraction(scalar)
        return SymbolicCoeffs2D({k: v * scalar for k, v in self.coeffs.items() if v * scalar != 0})

    def __rmul__(self, scalar) -> 'SymbolicCoeffs2D':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Fraction | int | float) -> 'SymbolicCoeffs2D':
        if isinstance(scalar, float):
            scalar = Fraction(scalar).limit_denominator(1000000)
        elif isinstance(scalar, int):
            scalar = Fraction(scalar)
        return SymbolicCoeffs2D({k: v / scalar for k, v in self.coeffs.items()})

    def min_y(self) -> int:
        return min(k[0] for k in self.coeffs.keys()) if self.coeffs else 0

    def max_y(self) -> int:
        return max(k[0] for k in self.coeffs.keys()) if self.coeffs else 0

    def min_x(self) -> int:
        return min(k[1] for k in self.coeffs.keys()) if self.coeffs else 0

    def max_x(self) -> int:
        return max(k[1] for k in self.coeffs.keys()) if self.coeffs else 0

    def sum_coeffs(self) -> Fraction:
        return sum(self.coeffs.values(), Fraction(0))

    def normalize(self) -> 'SymbolicCoeffs2D':
        s = self.sum_coeffs()
        if s != 0:
            return self / s
        return self

    def to_2d_array(self) -> list[list[Fraction]]:
        """Convert to a 2D array with indices relative to min_y, min_x."""
        if not self.coeffs:
            return [[Fraction(0)]]

        min_y, max_y = self.min_y(), self.max_y()
        min_x, max_x = self.min_x(), self.max_x()

        rows = []
        for y in range(min_y, max_y + 1):
            row = []
            for x in range(min_x, max_x + 1):
                row.append(self.coeffs.get((y, x), Fraction(0)))
            rows.append(row)
        return rows


# =============================================================================
# 2D Sampling Kernels (1D, used separably)
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
KERNELS: dict[str, tuple[Callable[[float], float], float, str]] = {
    'box': (box_kernel, 0.5, "Box filter"),
    'triangle': (triangle_kernel, 1.0, "Triangle/bilinear"),
    'lanczos2': (lambda x: lanczos_kernel(x, 2), 2.0, "Lanczos a=2"),
    'lanczos3': (lambda x: lanczos_kernel(x, 3), 3.0, "Lanczos a=3"),
    'sinc': (sinc_kernel, 8.0, "Pure sinc (truncated)"),
    'mitchell': (mitchell_kernel, 2.0, "Mitchell-Netravali"),
}


# =============================================================================
# EWA (Elliptical Weighted Average) Kernels - Radial 2D
# =============================================================================

def jinc(r: float) -> float:
    """
    Jinc function: 2D analog of sinc.
    jinc(r) = 2 * J1(π*r) / (π*r)
    """
    if abs(r) < 1e-10:
        return 1.0
    pi_r = math.pi * r
    return 2.0 * bessel_j1(pi_r) / pi_r


def ewa_lanczos_jinc(r: float, a: int = 3) -> float:
    """
    EWA Lanczos kernel using jinc.
    kernel(r) = jinc(r) * jinc(r/a) for r < a, else 0
    """
    if abs(r) < 1e-10:
        return 1.0
    if abs(r) >= a:
        return 0.0
    return jinc(r) * jinc(r / a)


def ewa_lanczos_sinc(r: float, a: int = 3) -> float:
    """EWA using sinc instead of jinc (radial application of sinc)."""
    if abs(r) < 1e-10:
        return 1.0
    if abs(r) >= a:
        return 0.0
    pi_r = math.pi * r
    return (math.sin(pi_r) / pi_r) * (math.sin(pi_r / a) / (pi_r / a))


# EWA_KERNELS: (function, support_radius, description)
EWA_KERNELS: dict[str, tuple] = {
    'ewa-lanczos2-jinc': (lambda r: ewa_lanczos_jinc(r, 2), 2.0, "EWA Lanczos2 with jinc"),
    'ewa-lanczos3-jinc': (lambda r: ewa_lanczos_jinc(r, 3), 3.0, "EWA Lanczos3 with jinc"),
    'ewa-lanczos4-jinc': (lambda r: ewa_lanczos_jinc(r, 4), 4.0, "EWA Lanczos4 with jinc"),
    'ewa-lanczos2-sinc': (lambda r: ewa_lanczos_sinc(r, 2), 2.0, "EWA Lanczos2 with sinc"),
    'ewa-lanczos3-sinc': (lambda r: ewa_lanczos_sinc(r, 3), 3.0, "EWA Lanczos3 with sinc"),
    'ewa-jinc': (jinc, 8.0, "Pure jinc (truncated)"),
}

ALL_KERNELS = {**KERNELS, **EWA_KERNELS}


# =============================================================================
# 2D Tent-Space Expansion (Box → Tent)
# =============================================================================
#
# Tent space layout for 2D:
# - Position (2y, 2x) = corner (4-way average of surrounding pixels)
# - Position (2y+1, 2x) = vertical edge (2-way average along x)
# - Position (2y, 2x+1) = horizontal edge (2-way average along y)
# - Position (2y+1, 2x+1) = center (volume-preserving)

def tent_corner_coeffs_2d(
    y_idx: int, x_idx: int,
    source_func: Callable[[int, int], SymbolicCoeffs2D]
) -> SymbolicCoeffs2D:
    """
    Corner at tent position (2*y_idx, 2*x_idx).
    4-way average of the 4 surrounding pixels.
    """
    return (
        source_func(y_idx - 1, x_idx - 1) +
        source_func(y_idx - 1, x_idx) +
        source_func(y_idx, x_idx - 1) +
        source_func(y_idx, x_idx)
    ) * Fraction(1, 4)


def tent_vedge_coeffs_2d(
    y_idx: int, x_idx: int,
    source_func: Callable[[int, int], SymbolicCoeffs2D]
) -> SymbolicCoeffs2D:
    """
    Vertical edge at tent position (2*y_idx+1, 2*x_idx).
    2-way average of left and right pixels.
    """
    return (source_func(y_idx, x_idx - 1) + source_func(y_idx, x_idx)) * Fraction(1, 2)


def tent_hedge_coeffs_2d(
    y_idx: int, x_idx: int,
    source_func: Callable[[int, int], SymbolicCoeffs2D]
) -> SymbolicCoeffs2D:
    """
    Horizontal edge at tent position (2*y_idx, 2*x_idx+1).
    2-way average of top and bottom pixels.
    """
    return (source_func(y_idx - 1, x_idx) + source_func(y_idx, x_idx)) * Fraction(1, 2)


def tent_center_coeffs_2d(
    y_idx: int, x_idx: int,
    source_func: Callable[[int, int], SymbolicCoeffs2D]
) -> SymbolicCoeffs2D:
    """
    Center at tent position (2*y_idx+1, 2*x_idx+1).
    Volume-preserving adjustment.

    The integration weights are:
        1/16 * corners + 1/8 * edges + 1/4 * center = V

    So: M = 4*V - 1/4*(corners) - 1/2*(edges)

    The corners of the 3×3 neighborhood around center (2y+1, 2x+1):
    - (2y, 2x), (2y, 2x+2), (2y+2, 2x), (2y+2, 2x+2)

    The edges:
    - (2y, 2x+1), (2y+2, 2x+1), (2y+1, 2x), (2y+1, 2x+2)

    But we compute this from the source values directly.
    """
    # Direct formula from the source pixel and its neighbors
    # Center M_ij = 4*V_ij - corners/4 - edges/2

    # The pixel V_ij contributes to tent positions:
    # - Corners: (2i, 2j), (2i, 2j+2), (2i+2, 2j), (2i+2, 2j+2) with weight 1/4 each
    # - Edges: (2i+1, 2j), (2i+1, 2j+2), (2i, 2j+1), (2i+2, 2j+1) with weight 1/2 each
    # - Center: (2i+1, 2j+1) directly

    # For volume preservation:
    # The 4 corners of the 3×3 tent neighborhood (relative to center 2i+1, 2j+1) are:
    # (2i, 2j), (2i, 2j+2), (2i+2, 2j), (2i+2, 2j+2)
    # Each involves 4 source pixels with weight 1/4

    # Corner (2i, 2j) = 1/4*(V[i-1,j-1] + V[i-1,j] + V[i,j-1] + V[i,j])
    # Corner (2i, 2j+2) = 1/4*(V[i-1,j] + V[i-1,j+1] + V[i,j] + V[i,j+1])
    # etc.

    # Let's compute this properly:
    # First get the 4 corners (interpolated tent values)
    corner_TL = tent_corner_coeffs_2d(y_idx, x_idx, source_func)      # (2y, 2x)
    corner_TR = tent_corner_coeffs_2d(y_idx, x_idx + 1, source_func)  # (2y, 2x+2)
    corner_BL = tent_corner_coeffs_2d(y_idx + 1, x_idx, source_func)  # (2y+2, 2x)
    corner_BR = tent_corner_coeffs_2d(y_idx + 1, x_idx + 1, source_func)  # (2y+2, 2x+2)

    # Get the 4 edges
    edge_T = tent_hedge_coeffs_2d(y_idx, x_idx, source_func)      # (2y, 2x+1)
    edge_B = tent_hedge_coeffs_2d(y_idx + 1, x_idx, source_func)  # (2y+2, 2x+1)
    edge_L = tent_vedge_coeffs_2d(y_idx, x_idx, source_func)      # (2y+1, 2x)
    edge_R = tent_vedge_coeffs_2d(y_idx, x_idx + 1, source_func)  # (2y+1, 2x+2)

    # Volume preservation formula
    corner_sum = corner_TL + corner_TR + corner_BL + corner_BR
    edge_sum = edge_T + edge_B + edge_L + edge_R

    original = source_func(y_idx, x_idx)
    center = original * 4 - corner_sum * Fraction(1, 4) - edge_sum * Fraction(1, 2)

    return center


def tent_value_coeffs_2d(
    tent_y: int, tent_x: int,
    source_func: Callable[[int, int], SymbolicCoeffs2D]
) -> SymbolicCoeffs2D:
    """Get symbolic coefficients for 2D tent value at integer position."""
    y_even = (tent_y % 2 == 0)
    x_even = (tent_x % 2 == 0)

    if y_even and x_even:
        # Corner
        return tent_corner_coeffs_2d(tent_y // 2, tent_x // 2, source_func)
    elif not y_even and x_even:
        # Vertical edge
        return tent_vedge_coeffs_2d((tent_y - 1) // 2, tent_x // 2, source_func)
    elif y_even and not x_even:
        # Horizontal edge
        return tent_hedge_coeffs_2d(tent_y // 2, (tent_x - 1) // 2, source_func)
    else:
        # Center
        return tent_center_coeffs_2d((tent_y - 1) // 2, (tent_x - 1) // 2, source_func)


def make_recursive_tent_func_2d(levels: int) -> Callable[[int, int], SymbolicCoeffs2D]:
    """
    Create a function that returns coefficients for a recursively expanded 2D tent space.
    """
    if levels <= 0:
        return SymbolicCoeffs2D.unit

    cache: dict[tuple[int, int, int], SymbolicCoeffs2D] = {}

    def get_level_value(level: int, y: int, x: int) -> SymbolicCoeffs2D:
        if level <= 0:
            return SymbolicCoeffs2D.unit(y, x)

        key = (level, y, x)
        if key in cache:
            return cache[key]

        prev_level_func = lambda py, px: get_level_value(level - 1, py, px)
        result = tent_value_coeffs_2d(y, x, prev_level_func)
        cache[key] = result
        return result

    return lambda y, x: get_level_value(levels, y, x)


# =============================================================================
# 2D Resampling in Tent Space
# =============================================================================

def resample_tent_trapezoidal_2d(
    center_y: float,
    center_x: float,
    width: float,
    kernel_func: Callable[[float], float],
    support_half_width: float,
    tent_value_func: Callable[[int, int], SymbolicCoeffs2D],
    is_ewa: bool = False,
    ewa_kernel_func: Callable[[float], float] | None = None,
) -> SymbolicCoeffs2D:
    """
    Resample the 2D tent surface using a kernel.

    For separable kernels, weight = kernel(y) * kernel(x).
    For EWA kernels, weight = ewa_kernel(sqrt(y² + x²)).
    """
    half_width = width / 2
    interval_y_start = center_y - half_width
    interval_y_end = center_y + half_width
    interval_x_start = center_x - half_width
    interval_x_end = center_x + half_width

    PADDING = 3
    y_start = int(math.floor(interval_y_start + 0.5)) - PADDING
    y_end = int(math.floor(interval_y_end + 0.5)) + PADDING
    x_start = int(math.floor(interval_x_start + 0.5)) - PADDING
    x_end = int(math.floor(interval_x_end + 0.5)) + PADDING

    result = SymbolicCoeffs2D.zero()
    weight_sum = 0.0

    for py in range(y_start, y_end + 1):
        # Compute Y overlap
        overlap_y_start = max(interval_y_start, py - 0.5)
        overlap_y_end = min(interval_y_end, py + 0.5)
        overlap_y = max(0.0, overlap_y_end - overlap_y_start)
        if overlap_y < 1e-10:
            continue

        for px in range(x_start, x_end + 1):
            # Compute X overlap
            overlap_x_start = max(interval_x_start, px - 0.5)
            overlap_x_end = min(interval_x_end, px + 0.5)
            overlap_x = max(0.0, overlap_x_end - overlap_x_start)
            if overlap_x < 1e-10:
                continue

            overlap_area = overlap_y * overlap_x

            if kernel_func == box_kernel and not is_ewa:
                # Box filter: weight is just the overlap area
                combined_weight = overlap_area
            elif is_ewa and ewa_kernel_func is not None:
                # EWA: use radial distance
                ky = (py - center_y) / half_width * support_half_width if half_width > 1e-10 else 0.0
                kx = (px - center_x) / half_width * support_half_width if half_width > 1e-10 else 0.0
                r = math.sqrt(ky * ky + kx * kx)
                kernel_weight = ewa_kernel_func(r)
                if abs(kernel_weight) < 1e-10:
                    continue
                combined_weight = overlap_area * kernel_weight
            else:
                # Separable: kernel(y) * kernel(x)
                ky = (py - center_y) / half_width * support_half_width if half_width > 1e-10 else 0.0
                kx = (px - center_x) / half_width * support_half_width if half_width > 1e-10 else 0.0
                wy = kernel_func(ky)
                wx = kernel_func(kx)
                if abs(wy) < 1e-10 or abs(wx) < 1e-10:
                    continue
                combined_weight = overlap_area * wy * wx

            result = result + tent_value_func(py, px) * combined_weight
            weight_sum += combined_weight

    if weight_sum > 1e-10:
        result = result / weight_sum

    return result


# =============================================================================
# 2D Contraction
# =============================================================================

def contract_tent_values_2d(
    corner_TL: SymbolicCoeffs2D, edge_T: SymbolicCoeffs2D, corner_TR: SymbolicCoeffs2D,
    edge_L: SymbolicCoeffs2D, center: SymbolicCoeffs2D, edge_R: SymbolicCoeffs2D,
    corner_BL: SymbolicCoeffs2D, edge_B: SymbolicCoeffs2D, corner_BR: SymbolicCoeffs2D
) -> SymbolicCoeffs2D:
    """
    Apply 2D tent contraction.
    Weights:
        1/16  1/8  1/16
        1/8   1/4  1/8
        1/16  1/8  1/16
    """
    return (
        corner_TL * Fraction(1, 16) + edge_T * Fraction(1, 8) + corner_TR * Fraction(1, 16) +
        edge_L * Fraction(1, 8) + center * Fraction(1, 4) + edge_R * Fraction(1, 8) +
        corner_BL * Fraction(1, 16) + edge_B * Fraction(1, 8) + corner_BR * Fraction(1, 16)
    )


# =============================================================================
# Full 2D Pipeline: Expand → Resample → Contract
# =============================================================================

def derive_direct_kernel_2d(
    ratio: float,
    offset_y: float = 0.0,
    offset_x: float = 0.0,
    kernel_name: str = 'box',
    kernel_width: float = 2.0,
    recurse: int = 1,
) -> SymbolicCoeffs2D:
    """
    Derive the effective 2D direct kernel for tent-space downsampling.

    Full pipeline for recurse=N:
    1. Expand box → tent N times (2^N resolution increase in each dimension)
    2. Resample at the finest tent level with the kernel
    3. Contract N times back to box space
    """
    is_ewa = kernel_name in EWA_KERNELS

    if is_ewa:
        ewa_kernel_func, kernel_radius, _ = EWA_KERNELS.get(kernel_name, EWA_KERNELS['ewa-lanczos3-jinc'])
        kernel_func = box_kernel  # Not used for EWA
    else:
        kernel_func, kernel_radius, _ = KERNELS.get(kernel_name, KERNELS['box'])
        ewa_kernel_func = None

    deepest_tent_func = make_recursive_tent_func_2d(recurse)

    def resample_at_deepest(pos_y: float, pos_x: float, width: float) -> SymbolicCoeffs2D:
        """Resample the deepest 2D tent surface at a given position."""
        return resample_tent_trapezoidal_2d(
            pos_y, pos_x, width,
            kernel_func, kernel_radius,
            deepest_tent_func,
            is_ewa=is_ewa,
            ewa_kernel_func=ewa_kernel_func
        )

    def derive_recursive_2d(
        levels_remaining: int,
        center_y: float, center_x: float,
        spacing: float, width: float
    ) -> SymbolicCoeffs2D:
        """
        Recursively derive 2D kernel by sampling and contracting.

        At each level we need 3×3 positions for contraction.
        """
        # The 9 positions for contraction (3×3 grid)
        positions = []
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                row.append((center_y + dy * spacing, center_x + dx * spacing))
            positions.append(row)

        if levels_remaining == 1:
            # Base case: sample from deepest tent level
            samples = []
            for row in positions:
                sample_row = []
                for (py, px) in row:
                    sample_row.append(resample_at_deepest(py, px, width))
                samples.append(sample_row)
            return contract_tent_values_2d(
                samples[0][0], samples[0][1], samples[0][2],
                samples[1][0], samples[1][1], samples[1][2],
                samples[2][0], samples[2][1], samples[2][2]
            )
        else:
            # Recursive case: each position needs computation from finer level
            samples = []
            for row in positions:
                sample_row = []
                for (py, px) in row:
                    # Map to deeper level: pos → 2*pos + 1
                    sample = derive_recursive_2d(
                        levels_remaining - 1,
                        py * 2 + 1, px * 2 + 1,
                        spacing * 2, width
                    )
                    sample_row.append(sample)
                samples.append(sample_row)
            return contract_tent_values_2d(
                samples[0][0], samples[0][1], samples[0][2],
                samples[1][0], samples[1][1], samples[1][2],
                samples[2][0], samples[2][1], samples[2][2]
            )

    # Starting position: center of output pixel in L1 tent space
    center_y_L1 = 2 * offset_y + 1
    center_x_L1 = 2 * offset_x + 1

    # Spacing in L1
    spacing_L1 = ratio

    result = derive_recursive_2d(recurse, center_y_L1, center_x_L1, spacing_L1, kernel_width)

    return result


# =============================================================================
# Output Formatting
# =============================================================================

def pretty_print_kernel_2d(coeffs: SymbolicCoeffs2D, name: str = "Kernel"):
    """Pretty print a 2D kernel in various formats."""
    if not coeffs.coeffs:
        print(f"{name}: 0")
        return

    min_y, max_y = coeffs.min_y(), coeffs.max_y()
    min_x, max_x = coeffs.min_x(), coeffs.max_x()

    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)
    print(f"Non-zero region: y=[{min_y},{max_y}], x=[{min_x},{max_x}]")
    print(f"Kernel size: {max_y - min_y + 1} × {max_x - min_x + 1}")

    # Get 2D array
    kernel_2d = coeffs.to_2d_array()

    # Find common denominator
    all_denoms = []
    for row in kernel_2d:
        for c in row:
            if c != 0:
                all_denoms.append(c.denominator)

    if all_denoms:
        from math import lcm
        from functools import reduce
        common_denom = reduce(lcm, all_denoms)

        print(f"\nInteger coefficients (÷{common_denom}):")
        int_kernel = []
        for row in kernel_2d:
            int_row = [int(c * common_denom) for c in row]
            int_kernel.append(int_row)
            print(f"  {int_row}")

        int_sum = sum(sum(row) for row in int_kernel)
        print(f"Sum: {int_sum} (should be {common_denom})")

    # Floating point
    print("\nFloating point coefficients:")
    for row in kernel_2d:
        float_row = [float(c) for c in row]
        print(f"  [{', '.join(f'{x:.6f}' for x in float_row)}]")

    # Sum check
    total = coeffs.sum_coeffs()
    print(f"\nSum: {float(total):.10f}")


def verify_against_bruteforce():
    """Verify against the bruteforce implementation."""
    print("="*70)
    print("VERIFICATION AGAINST BRUTEFORCE")
    print("="*70)

    # 2× downsample with box
    print("\n2× downsample with box kernel:")
    coeffs = derive_direct_kernel_2d(ratio=2, offset_y=0.5, offset_x=0.5, kernel_name='box', kernel_width=2)
    pretty_print_kernel_2d(coeffs, "2× box kernel")

    # Expected from bruteforce:
    # [-1, -9, -22, -22, -9, -1]
    # [-9, 47, 186, 186, 47, -9]
    # [-22, 186, 668, 668, 186, -22]
    # [-22, 186, 668, 668, 186, -22]
    # [-9, 47, 186, 186, 47, -9]
    # [-1, -9, -22, -22, -9, -1]
    # / 4096


def main():
    parser = argparse.ArgumentParser(
        description="Derive 2D tent-space sampling kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Derive 2× downsample kernel
  python tent_kernel_2d.py --ratio 2

  # 3× downsample with Lanczos-3
  python tent_kernel_2d.py --ratio 3 --kernel lanczos3

  # 2× with EWA lanczos3-jinc
  python tent_kernel_2d.py --ratio 2 --kernel ewa-lanczos3-jinc

  # Verify against bruteforce
  python tent_kernel_2d.py --verify
"""
    )
    parser.add_argument('--ratio', '-r', type=float, default=2.0,
                       help="Downsampling ratio (default: 2.0)")
    parser.add_argument('--offset', '-o', type=float, default=None,
                       help="Pixel offset (applied to both Y and X). Default: (ratio-1)/2")
    parser.add_argument('--offset-y', type=float, default=None,
                       help="Y pixel offset (overrides --offset for Y)")
    parser.add_argument('--offset-x', type=float, default=None,
                       help="X pixel offset (overrides --offset for X)")
    parser.add_argument('--kernel', '-k', type=str, default='box',
                       choices=list(ALL_KERNELS.keys()),
                       help="Sampling kernel (default: box)")
    parser.add_argument('--width', '-w', type=float, default=None,
                       help="Kernel width in source (box) space. Default: ratio/2.")
    parser.add_argument('--recurse', '-R', type=int, default=1,
                       help="Tent expansion recursion levels (default: 1)")
    parser.add_argument('--verify', '-v', action='store_true',
                       help="Verify against bruteforce")
    parser.add_argument('--list-kernels', '-l', action='store_true',
                       help="List available kernels")

    args = parser.parse_args()

    if args.list_kernels:
        print("Separable 1D kernels:")
        for name, (_, radius, desc) in KERNELS.items():
            print(f"  {name:18s} (radius={radius:.1f}): {desc}")
        print("\nEWA (radial 2D) kernels:")
        for name, (_, radius, desc) in EWA_KERNELS.items():
            print(f"  {name:18s} (radius={radius:.1f}): {desc}")
        return

    if args.verify:
        verify_against_bruteforce()
        return

    # Apply defaults based on ratio
    default_offset = (args.ratio - 1) / 2
    if args.offset is not None:
        offset_y = args.offset
        offset_x = args.offset
    else:
        offset_y = default_offset
        offset_x = default_offset

    if args.offset_y is not None:
        offset_y = args.offset_y
    if args.offset_x is not None:
        offset_x = args.offset_x

    width_box = args.width if args.width is not None else args.ratio / 2
    width_tent = width_box * 2

    print(f"Parameters: ratio={args.ratio}, offset_y={offset_y}, offset_x={offset_x}, "
          f"kernel={args.kernel}, width={width_box} (box space), recurse={args.recurse}")

    coeffs = derive_direct_kernel_2d(
        ratio=args.ratio,
        offset_y=offset_y,
        offset_x=offset_x,
        kernel_name=args.kernel,
        kernel_width=width_tent,
        recurse=args.recurse,
    )

    recurse_str = f", {args.recurse}× tent recursion" if args.recurse > 1 else ""
    name = f"{args.ratio}× 2D downsample with {args.kernel} kernel (width={width_box} box px{recurse_str})"
    pretty_print_kernel_2d(coeffs, name)


if __name__ == "__main__":
    main()
