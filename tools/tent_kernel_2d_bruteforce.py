#!/usr/bin/env python3
"""
Brute-force 2D tent-space kernel derivation via impulse response.

For each input pixel position, set it to 1.0 (all others 0.0),
run through the full 2D pipeline, and measure the output at a reference pixel.
The 2D array of outputs IS the effective kernel.

Pipeline: box → tent_expand_2d (×recurse) → resample_2d → tent_contract_2d (×recurse) → box
"""

from __future__ import annotations
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
        # Use polynomial approximation for small x, asymptotic for large x
        x = abs(x)
        if x < 8.0:
            # Polynomial approximation (Abramowitz & Stegun)
            y = x * x
            j1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
                + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
            j1 /= (144725228442.0 + y * (2300535178.0 + y * (18583304.74
                + y * (99447.43394 + y * (376.9991397 + y)))))
            return j1
        else:
            # Asymptotic approximation
            z = 8.0 / x
            y = z * z
            xx = x - 2.356194491
            f1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
            f2 = 0.04687499995 + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
            return math.sqrt(0.636619772 / x) * (math.cos(xx) * f1 - z * math.sin(xx) * f2)


# =============================================================================
# 2D Sampling Kernels (1D kernels used separably)
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
    """Pure sinc kernel (truncated at radius 8)."""
    if abs(x) < 1e-10:
        return 1.0
    if abs(x) >= 8.0:
        return 0.0
    pi_x = math.pi * x
    return math.sin(pi_x) / pi_x


def sinc_full_kernel(x: float) -> float:
    """Pure sinc kernel - no truncation (infinite support)."""
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
# These are 1D kernels used for separable resampling
KERNELS: dict[str, tuple] = {
    'box': (box_kernel, 0.5, "Box filter"),
    'triangle': (triangle_kernel, 1.0, "Triangle/bilinear"),
    'lanczos2': (lambda x: lanczos_kernel(x, 2), 2.0, "Lanczos a=2"),
    'lanczos3': (lambda x: lanczos_kernel(x, 3), 3.0, "Lanczos a=3"),
    'sinc': (sinc_kernel, 8.0, "Pure sinc (truncated r=8)"),
    'sinc-full': (sinc_full_kernel, 10000.0, "Pure sinc (full width)"),
    'mitchell': (mitchell_kernel, 2.0, "Mitchell-Netravali"),
}


# =============================================================================
# 2D EWA (Elliptical Weighted Average) Kernels
# =============================================================================

def jinc(x: float) -> float:
    """
    Jinc function: 2D analog of sinc.
    jinc(r) = 2 * J1(π*r) / (π*r)

    This is the Fourier transform of a circular aperture,
    just as sinc is the FT of a rectangular aperture.
    """
    if abs(x) < 1e-10:
        return 1.0
    pi_x = math.pi * x
    return 2.0 * bessel_j1(pi_x) / pi_x


def ewa_lanczos_jinc(r: float, a: int = 3) -> float:
    """
    EWA Lanczos kernel using jinc instead of sinc.
    kernel(r) = jinc(r) * jinc(r/a) for r < a, else 0

    This is the proper radial generalization of Lanczos.
    """
    if abs(r) < 1e-10:
        return 1.0
    if abs(r) >= a:
        return 0.0
    return jinc(r) * jinc(r / a)


def ewa_lanczos_sinc(r: float, a: int = 3) -> float:
    """
    EWA using sinc instead of jinc (radial application of sinc).
    Less correct mathematically but sometimes used.
    """
    if abs(r) < 1e-10:
        return 1.0
    if abs(r) >= a:
        return 0.0
    pi_r = math.pi * r
    return (math.sin(pi_r) / pi_r) * (math.sin(pi_r / a) / (pi_r / a))


def ewa_box(r: float, radius: float = 0.5) -> float:
    """Circular box (cylinder) kernel."""
    return 1.0 if abs(r) <= radius else 0.0


def ewa_gaussian(r: float, sigma: float = 0.5) -> float:
    """Gaussian EWA kernel."""
    return math.exp(-0.5 * (r / sigma) ** 2)


# EWA_KERNELS: (function, support_radius, description)
# These are radial 2D kernels for non-separable EWA resampling
EWA_KERNELS: dict[str, tuple] = {
    'ewa-lanczos2-jinc': (lambda r: ewa_lanczos_jinc(r, 2), 2.0, "EWA Lanczos2 with jinc"),
    'ewa-lanczos3-jinc': (lambda r: ewa_lanczos_jinc(r, 3), 3.0, "EWA Lanczos3 with jinc"),
    'ewa-lanczos4-jinc': (lambda r: ewa_lanczos_jinc(r, 4), 4.0, "EWA Lanczos4 with jinc"),
    'ewa-lanczos2-sinc': (lambda r: ewa_lanczos_sinc(r, 2), 2.0, "EWA Lanczos2 with sinc"),
    'ewa-lanczos3-sinc': (lambda r: ewa_lanczos_sinc(r, 3), 3.0, "EWA Lanczos3 with sinc"),
    'ewa-jinc': (jinc, 8.0, "Pure jinc (truncated)"),
    'ewa-box': (ewa_box, 0.5, "Circular box"),
    'ewa-gaussian': (ewa_gaussian, 3.0, "Gaussian (sigma=0.5)"),
}

# Combined kernels dict for CLI
ALL_KERNELS = {**KERNELS, **EWA_KERNELS}


# =============================================================================
# 2D Array Helpers
# =============================================================================

def create_2d(h: int, w: int, val: float = 0.0) -> list[list[float]]:
    """Create a 2D array initialized to val."""
    return [[val for _ in range(w)] for _ in range(h)]


def copy_2d(src: list[list[float]]) -> list[list[float]]:
    """Deep copy a 2D array."""
    return [row[:] for row in src]


def get_2d(arr: list[list[float]], y: int, x: int, clamp: bool = True) -> float:
    """Get value with optional clamp-to-edge."""
    h = len(arr)
    w = len(arr[0]) if h > 0 else 0
    if clamp:
        y = max(0, min(h - 1, y))
        x = max(0, min(w - 1, x))
    if 0 <= y < h and 0 <= x < w:
        return arr[y][x]
    return 0.0


# =============================================================================
# 2D Tent-Space Expansion (Box → Tent)
# =============================================================================

def tent_expand_2d(src: list[list[float]], debug: bool = False) -> list[list[float]]:
    """
    Expand 2D box-space array to tent-space.

    (H, W) → (2H+1, 2W+1)

    Tent-space layout:
    - Positions where both indices are even: corners (4-way average)
    - Positions where one index is even, one odd: edges (2-way average)
    - Positions where both indices are odd: centers (volume-preserving)

    For a 2×2 region of source pixels:
        A  B
        C  D

    The tent expansion creates:
        c  e  c  e  c
        e  M  e  M  e
        c  e  c  e  c
        e  M  e  M  e
        c  e  c  e  c

    Where:
    - c = corner (4-way average)
    - e = edge (2-way average along the edge direction)
    - M = center (volume-preserving)
    """
    h = len(src)
    w = len(src[0]) if h > 0 else 0

    dst_h = 2 * h + 1
    dst_w = 2 * w + 1
    dst = create_2d(dst_h, dst_w)

    # Helper with clamp-to-edge
    def get_src(y: int, x: int) -> float:
        return src[max(0, min(h - 1, y))][max(0, min(w - 1, x))]

    # Pass 1: Interpolate all positions
    for dy in range(dst_h):
        for dx in range(dst_w):
            y_even = (dy % 2 == 0)
            x_even = (dx % 2 == 0)

            if not y_even and not x_even:
                # Center: directly from source pixel
                sy = (dy - 1) // 2
                sx = (dx - 1) // 2
                dst[dy][dx] = get_src(sy, sx)

            elif y_even and x_even:
                # Corner: 4-way average
                sy_top = (dy // 2) - 1
                sy_bot = dy // 2
                sx_left = (dx // 2) - 1
                sx_right = dx // 2
                dst[dy][dx] = (get_src(sy_top, sx_left) + get_src(sy_top, sx_right) +
                               get_src(sy_bot, sx_left) + get_src(sy_bot, sx_right)) / 4

            elif y_even and not x_even:
                # Horizontal edge: 2-way vertical average
                sy_top = (dy // 2) - 1
                sy_bot = dy // 2
                sx = (dx - 1) // 2
                dst[dy][dx] = (get_src(sy_top, sx) + get_src(sy_bot, sx)) / 2

            else:  # not y_even and x_even
                # Vertical edge: 2-way horizontal average
                sy = (dy - 1) // 2
                sx_left = (dx // 2) - 1
                sx_right = dx // 2
                dst[dy][dx] = (get_src(sy, sx_left) + get_src(sy, sx_right)) / 2

    if debug:
        print(f"    After pass 1 (interpolation): {dst_h}×{dst_w}")

    # Pass 2: Adjust centers for volume preservation
    # In 2D, integration weights are:
    #   1/16 * corners + 1/8 * edges + 1/4 * center = V (original pixel value)
    #
    # For the center at (2sy+1, 2sx+1), the 3×3 tent neighborhood is:
    #   [dy-1, dx-1] [dy-1, dx]   [dy-1, dx+1]    (corners and edges)
    #   [dy,   dx-1] [dy,   dx]   [dy,   dx+1]
    #   [dy+1, dx-1] [dy+1, dx]   [dy+1, dx+1]
    #
    # Corners: (dy-1,dx-1), (dy-1,dx+1), (dy+1,dx-1), (dy+1,dx+1) → weight 1/16 each
    # Edges: (dy-1,dx), (dy+1,dx), (dy,dx-1), (dy,dx+1) → weight 1/8 each
    # Center: (dy,dx) → weight 1/4
    #
    # V = 1/16*(c1+c2+c3+c4) + 1/8*(e1+e2+e3+e4) + 1/4*M
    # M = 4*V - 1/4*(c1+c2+c3+c4) - 1/2*(e1+e2+e3+e4)

    for sy in range(h):
        for sx in range(w):
            dy = sy * 2 + 1  # Center position in tent space
            dx = sx * 2 + 1
            original = src[sy][sx]

            # Corners of the 3×3 neighborhood
            corner_sum = (dst[dy-1][dx-1] + dst[dy-1][dx+1] +
                          dst[dy+1][dx-1] + dst[dy+1][dx+1])

            # Edges of the 3×3 neighborhood
            edge_sum = (dst[dy-1][dx] + dst[dy+1][dx] +
                        dst[dy][dx-1] + dst[dy][dx+1])

            # Volume-preserving adjustment
            adjusted = 4 * original - 0.25 * corner_sum - 0.5 * edge_sum
            dst[dy][dx] = adjusted

    return dst


def tent_contract_2d(src: list[list[float]]) -> list[list[float]]:
    """
    Contract 2D tent-space array to box-space.

    (2H+1, 2W+1) → (H, W)

    Integration weights for 3×3 neighborhood:
        1/16  1/8  1/16
        1/8   1/4  1/8
        1/16  1/8  1/16
    """
    src_h = len(src)
    src_w = len(src[0]) if src_h > 0 else 0

    assert src_h % 2 == 1, "Source height must be odd"
    assert src_w % 2 == 1, "Source width must be odd"

    dst_h = (src_h - 1) // 2
    dst_w = (src_w - 1) // 2
    dst = create_2d(dst_h, dst_w)

    for dy in range(dst_h):
        for dx in range(dst_w):
            # Map to tent-space center
            sy = dy * 2 + 1
            sx = dx * 2 + 1

            # 3×3 integration
            # Corners: 1/16
            corner_sum = (src[sy-1][sx-1] + src[sy-1][sx+1] +
                          src[sy+1][sx-1] + src[sy+1][sx+1])
            # Edges: 1/8
            edge_sum = (src[sy-1][sx] + src[sy+1][sx] +
                        src[sy][sx-1] + src[sy][sx+1])
            # Center: 1/4
            center = src[sy][sx]

            dst[dy][dx] = (1/16) * corner_sum + (1/8) * edge_sum + (1/4) * center

    return dst


# =============================================================================
# 2D Resampling
# =============================================================================

def box_integrated_1d(src_pos: float, si: int, filter_scale: float) -> float:
    """
    Compute 1D overlap between destination pixel footprint and source pixel cell.
    """
    half_width = 0.5 * filter_scale
    dst_start = src_pos - half_width
    dst_end = src_pos + half_width

    src_start = si - 0.5
    src_end = si + 0.5

    overlap_start = max(dst_start, src_start)
    overlap_end = min(dst_end, src_end)

    return max(0.0, overlap_end - overlap_start)


def resample_2d_box(
    src: list[list[float]],
    dst_h: int,
    dst_w: int,
    ratio: float,
    depth: int = 1,
    debug: bool = False
) -> list[list[float]]:
    """
    Resample 2D array using separable box filter with exact overlap computation.
    """
    src_h = len(src)
    src_w = len(src[0]) if src_h > 0 else 0
    dst = create_2d(dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return copy_2d(src)

    scale = ratio
    filter_scale = ratio
    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1)))

    box_radius = int(filter_scale / 2 + 1) + 1

    if debug:
        print(f"    Resample 2D: {src_h}×{src_w} → {dst_h}×{dst_w}, scale={scale:.4f}, offset={offset:.4f}")

    for dy in range(dst_h):
        src_y = dy * scale + offset
        cy = int(src_y)
        start_y = max(0, cy - box_radius)
        end_y = min(src_h - 1, cy + box_radius)

        for dx in range(dst_w):
            src_x = dx * scale + offset
            cx = int(src_x)
            start_x = max(0, cx - box_radius)
            end_x = min(src_w - 1, cx + box_radius)

            weight_sum = 0.0
            value_sum = 0.0

            for sy in range(start_y, end_y + 1):
                wy = box_integrated_1d(src_y, sy, filter_scale)
                if wy < 1e-10:
                    continue

                for sx in range(start_x, end_x + 1):
                    wx = box_integrated_1d(src_x, sx, filter_scale)
                    if wx < 1e-10:
                        continue

                    w = wx * wy  # Separable: 2D weight = product of 1D weights
                    weight_sum += w
                    value_sum += src[sy][sx] * w

            if weight_sum > 1e-8:
                dst[dy][dx] = value_sum / weight_sum
            else:
                fallback_y = max(0, min(src_h - 1, int(round(src_y))))
                fallback_x = max(0, min(src_w - 1, int(round(src_x))))
                dst[dy][dx] = src[fallback_y][fallback_x]

    return dst


def resample_2d_ewa(
    src: list[list[float]],
    dst_h: int,
    dst_w: int,
    ratio: float,
    depth: int = 1,
    kernel_name: str = 'ewa-lanczos3-jinc',
    filter_width: float | None = None,
    debug: bool = False
) -> list[list[float]]:
    """
    Resample 2D array using EWA (non-separable radial) kernel.

    Uses radial distance for proper circular kernel support,
    which is the mathematically correct approach for isotropic resampling.
    """
    kernel_func, kernel_radius, _ = EWA_KERNELS.get(kernel_name, EWA_KERNELS['ewa-lanczos3-jinc'])

    src_h = len(src)
    src_w = len(src[0]) if src_h > 0 else 0
    dst = create_2d(dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return copy_2d(src)

    scale = ratio
    filter_scale = filter_width if filter_width is not None else ratio
    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1)))

    # Radius in samples: kernel support scaled by filter
    sample_radius = int(kernel_radius * filter_scale / 2 + 2)

    if debug:
        print(f"    Resample 2D EWA ({kernel_name}): {src_h}×{src_w} → {dst_h}×{dst_w}, scale={scale:.4f}, radius={sample_radius}")

    for dy in range(dst_h):
        src_y = dy * scale + offset
        cy = int(src_y)
        start_y = max(0, cy - sample_radius)
        end_y = min(src_h - 1, cy + sample_radius)

        for dx in range(dst_w):
            src_x = dx * scale + offset
            cx = int(src_x)
            start_x = max(0, cx - sample_radius)
            end_x = min(src_w - 1, cx + sample_radius)

            weight_sum = 0.0
            value_sum = 0.0
            half_width = filter_scale / 2

            for sy in range(start_y, end_y + 1):
                rel_y = (sy - src_y) / half_width * kernel_radius if half_width > 1e-10 else 0.0

                for sx in range(start_x, end_x + 1):
                    rel_x = (sx - src_x) / half_width * kernel_radius if half_width > 1e-10 else 0.0

                    # Radial distance for EWA
                    r = math.sqrt(rel_x * rel_x + rel_y * rel_y)

                    w = kernel_func(r)
                    if abs(w) > 1e-10:
                        weight_sum += w
                        value_sum += src[sy][sx] * w

            if abs(weight_sum) > 1e-8:
                dst[dy][dx] = value_sum / weight_sum
            else:
                fallback_y = max(0, min(src_h - 1, int(round(src_y))))
                fallback_x = max(0, min(src_w - 1, int(round(src_x))))
                dst[dy][dx] = src[fallback_y][fallback_x]

    return dst


def resample_2d_kernel(
    src: list[list[float]],
    dst_h: int,
    dst_w: int,
    ratio: float,
    depth: int = 1,
    kernel_name: str = 'box',
    filter_width: float | None = None,
    debug: bool = False
) -> list[list[float]]:
    """
    Resample 2D array using specified kernel.

    For box kernel, uses exact overlap computation.
    For EWA kernels, uses non-separable radial weighting.
    For other kernels, uses separable weighted point sampling.
    """
    if kernel_name == 'box':
        return resample_2d_box(src, dst_h, dst_w, ratio, depth, debug)

    # Check if this is an EWA kernel (non-separable, radial)
    if kernel_name in EWA_KERNELS:
        return resample_2d_ewa(src, dst_h, dst_w, ratio, depth, kernel_name, filter_width, debug)

    kernel_func, kernel_radius, _ = KERNELS.get(kernel_name, KERNELS['box'])

    src_h = len(src)
    src_w = len(src[0]) if src_h > 0 else 0
    dst = create_2d(dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return copy_2d(src)

    scale = ratio
    filter_scale = filter_width if filter_width is not None else ratio
    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1)))

    sample_radius = int(kernel_radius * filter_scale / 2 + 2)

    if debug:
        print(f"    Resample 2D ({kernel_name}): {src_h}×{src_w} → {dst_h}×{dst_w}, scale={scale:.4f}")

    for dy in range(dst_h):
        src_y = dy * scale + offset
        cy = int(src_y)
        start_y = max(0, cy - sample_radius)
        end_y = min(src_h - 1, cy + sample_radius)

        for dx in range(dst_w):
            src_x = dx * scale + offset
            cx = int(src_x)
            start_x = max(0, cx - sample_radius)
            end_x = min(src_w - 1, cx + sample_radius)

            weight_sum = 0.0
            value_sum = 0.0
            half_width = filter_scale / 2

            for sy in range(start_y, end_y + 1):
                if half_width > 1e-10:
                    ky = (sy - src_y) / half_width * kernel_radius
                else:
                    ky = 0.0
                wy = kernel_func(ky)
                if abs(wy) < 1e-10:
                    continue

                for sx in range(start_x, end_x + 1):
                    if half_width > 1e-10:
                        kx = (sx - src_x) / half_width * kernel_radius
                    else:
                        kx = 0.0
                    wx = kernel_func(kx)
                    if abs(wx) < 1e-10:
                        continue

                    w = wx * wy  # Separable
                    weight_sum += w
                    value_sum += src[sy][sx] * w

            if abs(weight_sum) > 1e-8:
                dst[dy][dx] = value_sum / weight_sum
            else:
                fallback_y = max(0, min(src_h - 1, int(round(src_y))))
                fallback_x = max(0, min(src_w - 1, int(round(src_x))))
                dst[dy][dx] = src[fallback_y][fallback_x]

    return dst


# =============================================================================
# Full 2D Pipeline
# =============================================================================

def full_pipeline_2d(
    src: list[list[float]],
    output_h: int,
    output_w: int,
    ratio: float,
    recurse: int = 1,
    kernel_name: str = 'box',
    filter_width: float | None = None,
    debug: bool = False
) -> list[list[float]]:
    """
    Full 2D tent-space downscaling pipeline.

    box → tent_expand_2d (×recurse) → resample_2d → tent_contract_2d (×recurse) → box
    """
    data = copy_2d(src)

    if debug:
        print(f"  Input box: {len(data)}×{len(data[0])}")

    # Expand to tent space (recurse times)
    for i in range(recurse):
        prev_h, prev_w = len(data), len(data[0])
        data = tent_expand_2d(data, debug=debug)
        if debug:
            print(f"  Expand {i+1}: {prev_h}×{prev_w} → {len(data)}×{len(data[0])}")

    # Calculate target tent-space size
    scale = 2 ** recurse
    tent_target_h = scale * output_h + (scale - 1)
    tent_target_w = scale * output_w + (scale - 1)

    if debug:
        print(f"  Resample target: {len(data)}×{len(data[0])} → {tent_target_h}×{tent_target_w}")

    # Resample in tent space
    data = resample_2d_kernel(
        data, tent_target_h, tent_target_w,
        ratio=ratio, depth=recurse,
        kernel_name=kernel_name, filter_width=filter_width,
        debug=debug
    )

    if debug:
        print(f"  After resample: {len(data)}×{len(data[0])}")

    # Contract back to box space (recurse times)
    for i in range(recurse):
        prev_h, prev_w = len(data), len(data[0])
        data = tent_contract_2d(data)
        if debug:
            print(f"  Contract {i+1}: {prev_h}×{prev_w} → {len(data)}×{len(data[0])}")

    return data


def derive_kernel_bruteforce_2d(
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    output_y: int,
    output_x: int,
    ratio: float,
    recurse: int = 1,
    kernel_name: str = 'box',
    filter_width: float | None = None
) -> list[list[float]]:
    """
    Derive the effective 2D kernel by computing impulse responses.

    For each input position (y, x), create an impulse (1.0 at (y,x), 0.0 elsewhere),
    run through the pipeline, and record the output at (output_y, output_x).
    """
    kernel = create_2d(input_h, input_w)

    for iy in range(input_h):
        for ix in range(input_w):
            # Create impulse at position (iy, ix)
            impulse = create_2d(input_h, input_w)
            impulse[iy][ix] = 1.0

            # Run through pipeline
            output = full_pipeline_2d(
                impulse, output_h, output_w,
                ratio=ratio, recurse=recurse,
                kernel_name=kernel_name, filter_width=filter_width
            )

            # Record contribution to reference output pixel
            if 0 <= output_y < len(output) and 0 <= output_x < len(output[0]):
                kernel[iy][ix] = output[output_y][output_x]

    return kernel


# =============================================================================
# Output Formatting
# =============================================================================

def format_2d_kernel(kernel: list[list[float]], threshold: float = 1e-10) -> str:
    """Format 2D kernel for display."""
    h = len(kernel)
    w = len(kernel[0]) if h > 0 else 0

    # Find bounding box of non-zero values
    min_y, max_y, min_x, max_x = h, -1, w, -1
    for y in range(h):
        for x in range(w):
            if abs(kernel[y][x]) > threshold:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)

    if max_y < 0:
        return "(all zeros)"

    lines = []
    lines.append(f"Non-zero region: y=[{min_y},{max_y}], x=[{min_x},{max_x}]")
    lines.append(f"Kernel size: {max_y - min_y + 1} × {max_x - min_x + 1}")

    # Extract the non-zero region
    subkernel = []
    for y in range(min_y, max_y + 1):
        row = [kernel[y][x] for x in range(min_x, max_x + 1)]
        subkernel.append(row)

    # Calculate sum
    total = sum(sum(row) for row in subkernel)
    lines.append(f"Sum: {total:.10f}")

    # Try to find integer representation
    for denom in [4, 16, 64, 256, 1024, 4096, 16384]:
        int_kernel = [[round(v * denom) for v in row] for row in subkernel]
        reconstructed = [[c / denom for c in row] for row in int_kernel]
        error = sum(
            abs(subkernel[y][x] - reconstructed[y][x])
            for y in range(len(subkernel))
            for x in range(len(subkernel[0]))
        )
        if error < 1e-8:
            int_sum = sum(sum(row) for row in int_kernel)
            lines.append(f"\nInteger coefficients (÷{denom}):")
            for row in int_kernel:
                lines.append(f"  {row}")
            lines.append(f"Sum: {int_sum} (should be {denom})")
            break
    else:
        lines.append("\nFloating point:")
        for row in subkernel:
            lines.append(f"  [{', '.join(f'{v:.6f}' for v in row)}]")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Brute-force 2D tent-space kernel derivation via impulse response"
    )
    parser.add_argument('--input-size', '-i', type=int, default=16,
                       help="Input array size (H=W, default: 16)")
    parser.add_argument('--ratio', '-r', type=float, default=2.0,
                       help="Downscaling ratio (default: 2.0)")
    parser.add_argument('--output-y', '-y', type=int, default=None,
                       help="Output Y index to measure (default: center)")
    parser.add_argument('--output-x', '-x', type=int, default=None,
                       help="Output X index to measure (default: center)")
    parser.add_argument('--recurse', '-R', type=int, default=1,
                       help="Number of tent expansion/contraction cycles (default: 1)")
    parser.add_argument('--debug', '-d', action='store_true',
                       help="Show intermediate stages")
    parser.add_argument('--kernel', '-k', type=str, default='box',
                       choices=list(ALL_KERNELS.keys()),
                       help="Sampling kernel (default: box)")
    parser.add_argument('--width', '-w', type=float, default=None,
                       help="Filter width in box space (default: ratio/2)")
    parser.add_argument('--compare', '-c', action='store_true',
                       help="Compare all kernels")

    args = parser.parse_args()

    input_size = args.input_size
    output_size = int(input_size / args.ratio)
    output_y = args.output_y if args.output_y is not None else output_size // 2
    output_x = args.output_x if args.output_x is not None else output_size // 2

    if output_y >= output_size or output_x >= output_size:
        print(f"Error: output position ({output_y}, {output_x}) out of bounds (size: {output_size})")
        return

    # Width in box space, convert to tent space
    width_box = args.width if args.width is not None else args.ratio / 2
    filter_width = width_box * 2

    print(f"Input size: {input_size}×{input_size}")
    print(f"Output size: {output_size}×{output_size} (ratio {args.ratio}×)")
    print(f"Output position: ({output_y}, {output_x})")
    print(f"Recurse levels: {args.recurse}")
    print(f"Kernel: {args.kernel}")
    print(f"Width: {width_box} box px ({filter_width} tent units)")
    print()

    # Show dimension progression
    print("Dimension progression:")
    sz = input_size
    for i in range(args.recurse):
        tent_sz = 2 * sz + 1
        fringe = (2 ** (i+1) - 1) / 2.0
        print(f"  Depth {i+1}: box {sz}×{sz} → tent {tent_sz}×{tent_sz} (fringe = {fringe:.1f})")
        sz = tent_sz

    scale = 2 ** args.recurse
    tent_target = scale * output_size + (scale - 1)
    print(f"  Resample: tent {sz}×{sz} → tent {tent_target}×{tent_target}")

    sz = tent_target
    for i in range(args.recurse):
        box_sz = (sz - 1) // 2
        print(f"  Contract {i+1}: tent {sz}×{sz} → box {box_sz}×{box_sz}")
        sz = box_sz
    print()

    if args.compare:
        print("=" * 60)
        print("Kernel comparison (2D)")
        print("=" * 60)
        print()
        for kname in ALL_KERNELS.keys():
            kernel = derive_kernel_bruteforce_2d(
                input_size, input_size,
                output_size, output_size,
                output_y, output_x,
                ratio=args.ratio, recurse=args.recurse,
                kernel_name=kname, filter_width=filter_width
            )
            print(f"--- {kname} ---")
            print(format_2d_kernel(kernel))
            print()
        return

    # Debug mode
    if args.debug:
        print("=" * 60)
        print(f"Debug: impulse at input position ({output_y * int(args.ratio)}, {output_x * int(args.ratio)})")
        print("=" * 60)
        impulse = create_2d(input_size, input_size)
        iy = min(output_y * int(args.ratio), input_size - 1)
        ix = min(output_x * int(args.ratio), input_size - 1)
        impulse[iy][ix] = 1.0
        output = full_pipeline_2d(
            impulse, output_size, output_size,
            ratio=args.ratio, recurse=args.recurse,
            kernel_name=args.kernel, filter_width=filter_width,
            debug=True
        )
        print()
        print(f"Final output[{output_y}][{output_x}]: {output[output_y][output_x]:.10f}")
        print()

    # Derive kernel
    print("=" * 60)
    print("Kernel derivation")
    print("=" * 60)
    kernel = derive_kernel_bruteforce_2d(
        input_size, input_size,
        output_size, output_size,
        output_y, output_x,
        ratio=args.ratio, recurse=args.recurse,
        kernel_name=args.kernel, filter_width=filter_width
    )

    print(format_2d_kernel(kernel))
    print()

    # Verify with constant input
    print("=" * 60)
    print("Verification: constant input should give constant output")
    test_input = [[0.5 for _ in range(input_size)] for _ in range(input_size)]
    test_output = full_pipeline_2d(
        test_input, output_size, output_size,
        ratio=args.ratio, recurse=args.recurse,
        kernel_name=args.kernel, filter_width=filter_width
    )
    print(f"  Input:  all 0.5")
    print(f"  Output[{output_y}][{output_x}]: {test_output[output_y][output_x]:.10f}")
    print(f"  (Should be 0.5)")


if __name__ == "__main__":
    main()
