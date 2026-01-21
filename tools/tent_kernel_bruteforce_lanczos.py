#!/usr/bin/env python3
"""
Lanczos3-constrained tent-space kernel derivation via impulse response.

Same as tent_kernel_bruteforce.py but uses Lanczos3 interpolation to constrain
peaks instead of volume preservation.

For each input pixel position, set it to 1.0 (all others 0.0),
run through the full pipeline, and measure the output at a reference pixel.
The sequence of outputs IS the effective kernel.

Pipeline: box → tent_expand_lanczos (×recurse) → resample → tent_contract (×recurse) → box
"""

from __future__ import annotations
import argparse
import math


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
# Lanczos3 constraint function
# =============================================================================

def lanczos3_constraint(x: float) -> float:
    """
    Lanczos3 kernel for constraining peaks: sinc(x) * sinc(x/3) for |x| < 3.

    Key property: zeros at x = ±1, ±2, ±3, value 1 at x = 0.
    This means peaks at integer spacing don't interfere with each other.
    """
    if abs(x) < 1e-10:
        return 1.0
    if abs(x) >= 3.0:
        return 0.0
    pi_x = math.pi * x
    return (math.sin(pi_x) / pi_x) * (math.sin(pi_x / 3) / (pi_x / 3))


# =============================================================================
# Tent Expand/Contract with Lanczos3-constrained peaks
# =============================================================================

def tent_expand_1d(src: list[float], debug: bool = False) -> list[float]:
    """
    Expand 1D box-space array to tent-space using Lanczos3-constrained peaks.

    (W) → (2W+1)

    Structure:
    - Corners (even positions): averages of neighboring box values
    - Peaks (odd positions): Lanczos3 interpolation of corners

    Since Lanczos3 has zeros at integer positions, each corner only affects
    nearby peaks without interference from distant corners.
    """
    w = len(src)
    dst_w = 2 * w + 1
    dst = [0.0] * dst_w

    def get_box(i: int) -> float:
        return src[max(0, min(w - 1, i))]

    # Step 1: Compute corners (even positions) - same as volume-based
    for j in range(w + 1):
        tent_pos = 2 * j
        if j == 0:
            dst[tent_pos] = get_box(0)
        elif j == w:
            dst[tent_pos] = get_box(w - 1)
        else:
            dst[tent_pos] = (get_box(j - 1) + get_box(j)) / 2

    # Step 2: Compute peaks (odd positions) via Lanczos3 interpolation of corners
    num_corners = w + 1
    for i in range(w):
        tent_pos = 2 * i + 1
        corner_coord = tent_pos / 2.0  # = i + 0.5

        value = 0.0
        weight_sum = 0.0

        for j in range(num_corners):
            dist = corner_coord - j
            weight = lanczos3_constraint(dist)
            if abs(weight) > 1e-12:
                value += dst[2 * j] * weight
                weight_sum += weight

        if abs(weight_sum) > 1e-10:
            dst[tent_pos] = value / weight_sum
        else:
            nearest = int(round(corner_coord))
            nearest = max(0, min(num_corners - 1, nearest))
            dst[tent_pos] = dst[2 * nearest]

    if debug:
        print(f"    Expand (Lanczos3): corners={[dst[2*j] for j in range(min(w+1, 8))]}")

    return dst


def tent_contract_1d(src: list[float]) -> list[float]:
    """
    Contract 1D tent-space array to box-space.

    (2W+1) → (W)
    Integration weights: 1/4 * corner_left + 1/2 * center + 1/4 * corner_right
    """
    src_w = len(src)
    assert src_w % 2 == 1, "Source width must be odd"

    dst_w = (src_w - 1) // 2
    dst = [0.0] * dst_w

    for dx in range(dst_w):
        # Map to tent-space center
        sx = dx * 2 + 1

        corner_left = src[sx - 1]
        center = src[sx]
        corner_right = src[sx + 1]

        # Integration: 1/4 + 1/2 + 1/4 = 1
        dst[dx] = 0.25 * corner_left + 0.5 * center + 0.25 * corner_right

    return dst


# =============================================================================
# Resampling
# =============================================================================

def box_integrated(src_pos: float, si: int, filter_scale: float) -> float:
    """Compute overlap between destination pixel footprint and source pixel cell."""
    half_width = 0.5 * filter_scale
    dst_start = src_pos - half_width
    dst_end = src_pos + half_width

    src_start = si - 0.5
    src_end = si + 0.5

    overlap_start = max(dst_start, src_start)
    overlap_end = min(dst_end, src_end)

    return max(0.0, overlap_end - overlap_start)


def resample_1d_box(src: list[float], dst_len: int, ratio: float, depth: int = 1, extra_offset: float = 0.0, debug: bool = False) -> list[float]:
    """Resample 1D array using box filter with exact overlap computation."""
    src_len = len(src)
    dst = [0.0] * dst_len

    if src_len == dst_len and abs(extra_offset) < 1e-10:
        return list(src)

    if dst_len == 1:
        return [sum(src) / src_len]

    scale = ratio
    filter_scale = ratio

    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1))) + extra_offset

    if debug:
        print(f"    Resample: {src_len} → {dst_len}, scale={scale:.6f}, filter_scale={filter_scale:.6f}, offset={offset:.6f} (extra={extra_offset:.6f})")

    box_radius = int(filter_scale / 2 + 1) + 1

    for di in range(dst_len):
        src_pos = di * scale + offset
        center = int(src_pos)

        start = max(0, center - box_radius)
        end = min(src_len - 1, center + box_radius)

        weights = []
        weight_sum = 0.0

        for si in range(start, end + 1):
            w = box_integrated(src_pos, si, filter_scale)
            weights.append(w)
            weight_sum += w

        if weight_sum > 1e-8:
            result = 0.0
            for i, w in enumerate(weights):
                result += src[start + i] * w / weight_sum
            dst[di] = result
        else:
            fallback = int(round(src_pos))
            fallback = max(0, min(src_len - 1, fallback))
            dst[di] = src[fallback]

    return dst


def resample_1d_kernel(
    src: list[float],
    dst_len: int,
    ratio: float,
    depth: int = 1,
    kernel_name: str = 'box',
    filter_width: float | None = None,
    extra_offset: float = 0.0,
    debug: bool = False
) -> list[float]:
    """Resample 1D array using specified kernel."""
    if kernel_name == 'box':
        return resample_1d_box(src, dst_len, ratio, depth, extra_offset, debug)

    kernel_func, kernel_radius, _ = KERNELS.get(kernel_name, KERNELS['box'])

    src_len = len(src)
    dst = [0.0] * dst_len

    if src_len == dst_len and abs(extra_offset) < 1e-10:
        return list(src)

    if dst_len == 1:
        return [sum(src) / src_len]

    scale = ratio
    filter_scale = filter_width if filter_width is not None else ratio

    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1))) + extra_offset

    sample_radius = int(kernel_radius * filter_scale / 2 + 2)

    if debug:
        print(f"    Resample ({kernel_name}): {src_len} → {dst_len}, scale={scale:.4f}, radius={sample_radius}")

    for di in range(dst_len):
        src_pos = di * scale + offset
        center = int(src_pos)

        start = max(0, center - sample_radius)
        end = min(src_len - 1, center + sample_radius)

        weight_sum = 0.0
        value_sum = 0.0

        for si in range(start, end + 1):
            half_width = filter_scale / 2
            if half_width > 1e-10:
                kernel_arg = (si - src_pos) / half_width * kernel_radius
            else:
                kernel_arg = 0.0

            w = kernel_func(kernel_arg)
            if abs(w) > 1e-10:
                weight_sum += w
                value_sum += src[si] * w

        if abs(weight_sum) > 1e-8:
            dst[di] = value_sum / weight_sum
        else:
            fallback = int(round(src_pos))
            fallback = max(0, min(src_len - 1, fallback))
            dst[di] = src[fallback]

    return dst


def format_array(arr: list[float], max_show: int = 20) -> str:
    """Format array for display, showing values as fractions where clean."""
    if len(arr) <= max_show:
        return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"
    else:
        first = ", ".join(f"{v:.4f}" for v in arr[:5])
        last = ", ".join(f"{v:.4f}" for v in arr[-5:])
        return f"[{first}, ... ({len(arr)} total) ..., {last}]"


def full_pipeline(src: list[float], output_len: int, ratio: float, recurse: int = 1, kernel_name: str = 'box', filter_width: float | None = None, extra_offset: float = 0.0, debug: bool = False) -> list[float]:
    """
    Full Lanczos3-constrained tent-space downscaling pipeline.

    box → tent_expand (×recurse) → resample → tent_contract (×recurse) → box
    """
    data = src

    if debug:
        print(f"  Input box ({len(data)}): {format_array(data)}")

    # Expand to tent space (recurse times)
    for i in range(recurse):
        prev_len = len(data)
        data = tent_expand_1d(data, debug=debug)
        if debug:
            scale_factor = 2 ** (i + 1)
            fringe_box = (scale_factor - 1) / 2.0
            print(f"  Expand {i+1}: {prev_len} → {len(data)} (fringe = {fringe_box:.1f} box px)")
            print(f"    {format_array(data)}")

    # Calculate target tent-space size
    scale = 2 ** recurse
    tent_target = scale * output_len + (scale - 1)

    if debug:
        print(f"  Resample target: {len(data)} → {tent_target}")

    # Resample in tent space
    data = resample_1d_kernel(data, tent_target, ratio=ratio, depth=recurse, kernel_name=kernel_name, filter_width=filter_width, extra_offset=extra_offset, debug=debug)

    if debug:
        print(f"  After resample ({len(data)}): {format_array(data)}")

    # Contract back to box space
    for i in range(recurse):
        prev_len = len(data)
        data = tent_contract_1d(data)
        if debug:
            print(f"  Contract {i+1}: {prev_len} → {len(data)}")
            print(f"    {format_array(data)}")

    return data


def derive_kernel_bruteforce(input_len: int, output_len: int, output_idx: int, ratio: float, recurse: int = 1, kernel_name: str = 'box', filter_width: float | None = None, extra_offset: float = 0.0) -> list[float]:
    """Derive the effective kernel by computing impulse responses."""
    kernel = []

    for i in range(input_len):
        impulse = [0.0] * input_len
        impulse[i] = 1.0

        output = full_pipeline(impulse, output_len, ratio=ratio, recurse=recurse, kernel_name=kernel_name, filter_width=filter_width, extra_offset=extra_offset)

        if 0 <= output_idx < len(output):
            kernel.append(output[output_idx])
        else:
            kernel.append(0.0)

    return kernel


def main():
    parser = argparse.ArgumentParser(
        description="Lanczos3-constrained tent-space kernel derivation via impulse response"
    )
    parser.add_argument('--input-len', '-i', type=int, default=16,
                       help="Input array length (default: 16)")
    parser.add_argument('--ratio', '-r', type=float, default=2.0,
                       help="Downscaling ratio (default: 2.0)")
    parser.add_argument('--output-idx', '-o', type=int, default=None,
                       help="Output index to measure (default: center)")
    parser.add_argument('--recurse', '-R', type=int, default=1,
                       help="Number of tent expansion/contraction cycles (default: 1)")
    parser.add_argument('--debug', '-d', action='store_true',
                       help="Show all intermediate stages")
    parser.add_argument('--sequence', '-s', action='store_true',
                       help="Use sequential input [0,1,2,3,...] instead of impulse")
    parser.add_argument('--kernel', '-k', type=str, default='box',
                       choices=list(KERNELS.keys()),
                       help="Sampling kernel (default: box)")
    parser.add_argument('--width', '-w', type=float, default=None,
                       help="Filter width in box space (default: ratio/2)")
    parser.add_argument('--offset', type=float, default=0.0,
                       help="Fractional offset for phase shift testing (in tent-space units, default: 0.0)")
    parser.add_argument('--compare', '-c', action='store_true',
                       help="Compare all kernels")

    args = parser.parse_args()

    input_len = args.input_len
    output_len = int(input_len / args.ratio)
    output_idx = args.output_idx if args.output_idx is not None else output_len // 2

    if output_idx >= output_len:
        print(f"Error: output_idx {output_idx} is out of bounds (output length is {output_len}, valid indices 0-{output_len-1})")
        return

    width_box = args.width if args.width is not None else args.ratio / 2
    filter_width = width_box * 2

    print(f"Input length: {input_len}")
    print(f"Output length: {output_len} (ratio {args.ratio}×)")
    print(f"Output index: {output_idx}")
    print(f"Recurse levels: {args.recurse}")
    print(f"Kernel: {args.kernel}")
    print(f"Width: {width_box} box px ({filter_width} tent units)")
    print(f"Extra offset: {args.offset} tent units")
    print(f"Tent mode: lanczos3 (Lanczos3-constrained peaks)")
    print()

    # Show dimension progression
    print("Dimension progression:")
    w = input_len
    for i in range(args.recurse):
        tent_w = 2 * w + 1
        fringe = (2 ** (i+1) - 1) / 2.0
        print(f"  Depth {i+1}: box {w} → tent {tent_w} (fringe = {fringe:.1f} original box px)")
        w = tent_w

    scale = 2 ** args.recurse
    tent_target = scale * output_len + (scale - 1)
    print(f"  Resample: tent {w} → tent {tent_target}")

    w = tent_target
    for i in range(args.recurse):
        box_w = (w - 1) // 2
        print(f"  Contract {i+1}: tent {w} → box {box_w}")
        w = box_w
    print()

    if args.sequence:
        print("=" * 60)
        print("Sequential input mode: [0, 1, 2, 3, ...]")
        print("=" * 60)
        seq_input = list(range(input_len))
        output = full_pipeline(seq_input, output_len, ratio=args.ratio, recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width, extra_offset=args.offset, debug=args.debug)

        print()
        print(f"Input:  {format_array(list(map(float, seq_input)))}")
        print(f"Output: {format_array(output)}")

        print()
        print("Output linearity check:")
        expected_scale = args.ratio
        for i, v in enumerate(output):
            expected = i * expected_scale + (expected_scale - 1) / 2
            diff = v - expected
            marker = " " if abs(diff) < 0.001 else " ← DEVIATION"
            print(f"  output[{i}] = {v:.4f}, expected ≈ {expected:.4f}, diff = {diff:+.4f}{marker}")
        return

    if args.debug:
        print("=" * 60)
        print(f"Debug: impulse at input position {output_idx * int(args.ratio)}")
        print("=" * 60)
        impulse = [0.0] * input_len
        impulse_pos = min(output_idx * int(args.ratio), input_len - 1)
        impulse[impulse_pos] = 1.0
        output = full_pipeline(impulse, output_len, ratio=args.ratio, recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width, extra_offset=args.offset, debug=True)
        print()
        print(f"Final output: {format_array(output)}")
        print()

    if args.compare:
        print("=" * 60)
        print("Kernel comparison")
        print("=" * 60)
        print()
        for kname in KERNELS.keys():
            kernel = derive_kernel_bruteforce(input_len, output_len, output_idx, ratio=args.ratio, recurse=args.recurse, kernel_name=kname, filter_width=filter_width, extra_offset=args.offset)
            nonzero = [(i, k) for i, k in enumerate(kernel) if abs(k) > 1e-10]
            if nonzero:
                first_idx = nonzero[0][0]
                last_idx = nonzero[-1][0]
                kernel_slice = kernel[first_idx:last_idx + 1]
                for denom in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    int_coeffs = [round(k * denom) for k in kernel_slice]
                    reconstructed = [c / denom for c in int_coeffs]
                    error = sum(abs(a - b) for a, b in zip(kernel_slice, reconstructed))
                    if error < 1e-8:
                        print(f"{kname:12s}: {int_coeffs} / {denom}")
                        break
                else:
                    print(f"{kname:12s}: {[f'{k:.4f}' for k in kernel_slice]}")
            else:
                print(f"{kname:12s}: (all zeros)")
        return

    # Derive kernel
    print("=" * 60)
    print("Kernel derivation (Lanczos3-constrained)")
    print("=" * 60)
    kernel = derive_kernel_bruteforce(input_len, output_len, output_idx, ratio=args.ratio, recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width, extra_offset=args.offset)

    nonzero = [(i, k) for i, k in enumerate(kernel) if abs(k) > 1e-10]

    if nonzero:
        first_idx = nonzero[0][0]
        last_idx = nonzero[-1][0]

        print(f"Non-zero kernel region: input indices {first_idx} to {last_idx}")
        print(f"Kernel width: {last_idx - first_idx + 1} samples")
        print()

        kernel_slice = kernel[first_idx:last_idx + 1]

        total = sum(kernel_slice)
        print(f"  Sum: {total:.10f}")
        print()

        n = len(kernel_slice)
        is_symmetric = all(abs(kernel_slice[i] - kernel_slice[n-1-i]) < 1e-8 for i in range(n//2 + 1))
        print(f"  Symmetric: {is_symmetric}")
        if not is_symmetric:
            print("  Symmetry differences:")
            for i in range(n//2 + 1):
                j = n - 1 - i
                diff = kernel_slice[i] - kernel_slice[j]
                if abs(diff) > 1e-10:
                    print(f"    [{i}] vs [{j}]: {kernel_slice[i]:.8f} vs {kernel_slice[j]:.8f} (diff={diff:+.8f})")
        print()

        for denom in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            int_coeffs = [round(k * denom) for k in kernel_slice]
            reconstructed = [c / denom for c in int_coeffs]
            error = sum(abs(a - b) for a, b in zip(kernel_slice, reconstructed))
            if error < 1e-8:
                print(f"  Integer coefficients (÷{denom}):")
                print(f"    {int_coeffs}")
                print(f"    Sum: {sum(int_coeffs)} (should be {denom})")
                break
        else:
            print("  Floating point:")
            print(f"    {[f'{k:.6f}' for k in kernel_slice]}")

        print()
        print("  Per-position breakdown:")
        center = (output_idx + 0.5) * args.ratio
        for i, k in enumerate(kernel_slice):
            input_pos = first_idx + i
            rel_pos = input_pos - center
            print(f"    Input[{input_pos}] (rel {rel_pos:+.1f}): {k:.6f}")
    else:
        print("Kernel is all zeros!")

    print()
    print("=" * 60)
    print("Verification: constant input should give constant output")
    test_input = [0.5] * input_len
    test_output = full_pipeline(test_input, output_len, ratio=args.ratio, recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width, extra_offset=args.offset)
    print(f"  Input:  all 0.5")
    print(f"  Output[{output_idx}]: {test_output[output_idx]:.10f}")
    print(f"  (Should be 0.5)")


if __name__ == "__main__":
    main()
