#!/usr/bin/env python3
"""
Lanczos3-constrained tent-space kernel derivation.

Alternative to volume-based constraint: uses Lanczos3 interpolation to define
tent-space values from box-space peaks. Since Lanczos3 has zeros at integer
positions (±1, ±2, ±3), and tent peaks are integer-spaced in the normalized
coordinate system, each peak only affects itself - no interference.

This is fully reversible: the peak values ARE the box pixel values,
and all other tent positions are Lanczos3 interpolations of those peaks.

Pipeline: box → tent_expand_lanczos (×recurse) → resample → tent_contract_lanczos (×recurse) → box
"""

from __future__ import annotations
import argparse
import math


# =============================================================================
# Lanczos3 Kernel
# =============================================================================

def lanczos3(x: float) -> float:
    """
    Lanczos3 kernel: sinc(x) * sinc(x/3) for |x| < 3, else 0.

    Key property: zeros at x = ±1, ±2, ±3, value 1 at x = 0.
    """
    if abs(x) < 1e-10:
        return 1.0
    if abs(x) >= 3.0:
        return 0.0
    pi_x = math.pi * x
    return (math.sin(pi_x) / pi_x) * (math.sin(pi_x / 3) / (pi_x / 3))


# =============================================================================
# 1D Tent Expand/Contract using Lanczos3 Constraint
# =============================================================================

def tent_expand_lanczos_1d(src: list[float], debug: bool = False) -> list[float]:
    """
    Expand 1D box-space array to tent-space using Lanczos3-constrained peaks.

    (W) → (2W+1)

    Structure:
    - Corners (even positions 0, 2, 4, ..., 2W): averages of neighboring box values
    - Peaks (odd positions 1, 3, 5, ..., 2W-1): Lanczos3 interpolation of corners

    Corner computation (same as volume-based):
    - corner[0] = box[0]  (left edge, clamped)
    - corner[2j] = (box[j-1] + box[j]) / 2  for j in 1..W-1 (interior)
    - corner[2W] = box[W-1]  (right edge, clamped)

    Peak computation (Lanczos3 constraint):
    - peak[2i+1] = Lanczos3 interpolation of corners
    - Since corners are at even positions (0, 2, 4, ...), in corner-index space
      they're at 0, 1, 2, ... and the peak at tent position 2i+1 maps to
      corner-index (2i+1)/2 = i + 0.5
    - Lanczos3 zeros at integers mean only nearby corners contribute

    This is fully reversible: corners encode box values via averaging,
    and the averaging relation can be inverted.
    """
    w = len(src)
    dst_w = 2 * w + 1
    dst = [0.0] * dst_w

    # Helper to get box value with clamping
    def get_box(i: int) -> float:
        return src[max(0, min(w - 1, i))]

    # Step 1: Compute corners (even positions)
    # Corner at tent position 2j corresponds to the boundary between box[j-1] and box[j]
    for j in range(w + 1):
        tent_pos = 2 * j
        if j == 0:
            # Left edge: clamp to box[0]
            dst[tent_pos] = get_box(0)
        elif j == w:
            # Right edge: clamp to box[w-1]
            dst[tent_pos] = get_box(w - 1)
        else:
            # Interior: average of neighbors
            dst[tent_pos] = (get_box(j - 1) + get_box(j)) / 2

    # Step 2: Compute peaks (odd positions) via Lanczos3 interpolation of corners
    # Corners are at tent positions 0, 2, 4, ..., 2W
    # In "corner index" space, corner j is at position j (for j = 0, 1, ..., W)
    # Peak at tent position 2i+1 maps to corner-index (2i+1)/2 = i + 0.5
    num_corners = w + 1

    for i in range(w):
        tent_pos = 2 * i + 1
        corner_coord = tent_pos / 2.0  # = i + 0.5

        value = 0.0
        weight_sum = 0.0

        for j in range(num_corners):
            dist = corner_coord - j
            weight = lanczos3(dist)
            if abs(weight) > 1e-12:
                corner_val = dst[2 * j]  # Corner at tent position 2j
                value += corner_val * weight
                weight_sum += weight

        if abs(weight_sum) > 1e-10:
            dst[tent_pos] = value / weight_sum
        else:
            # Fallback: nearest corner
            nearest = int(round(corner_coord))
            nearest = max(0, min(num_corners - 1, nearest))
            dst[tent_pos] = dst[2 * nearest]

    if debug:
        print(f"    Expand Lanczos: {len(src)} → {len(dst)}")
        print(f"    Corners: {[dst[2*j] for j in range(w+1)]}")
        print(f"    Peaks: {[dst[2*i+1] for i in range(w)]}")

    return dst


def tent_contract_lanczos_1d(src: list[float], debug: bool = False) -> list[float]:
    """
    Contract 1D tent-space array to box-space using integration.

    (2W+1) → (W)

    Uses the same integration formula as volume-based contraction:
    box[i] = 1/4 * corner_left + 1/2 * peak + 1/4 * corner_right

    This works for any tent-space data (including resampled data where
    corners no longer satisfy the original averaging relationship).

    Note: For unmodified expand output, use tent_contract_exact_1d instead
    for perfect reversibility.
    """
    src_w = len(src)
    assert src_w % 2 == 1, "Source width must be odd"

    dst_w = (src_w - 1) // 2
    dst = [0.0] * dst_w

    for dx in range(dst_w):
        # Peak (center) is at tent position 2*dx + 1
        sx = dx * 2 + 1

        corner_left = src[sx - 1]
        peak = src[sx]
        corner_right = src[sx + 1]

        # Integration: 1/4 + 1/2 + 1/4 = 1
        dst[dx] = 0.25 * corner_left + 0.5 * peak + 0.25 * corner_right

    if debug:
        print(f"    Contract (integration): {src_w} → {dst_w}")

    return dst


def tent_contract_exact_1d(src: list[float], debug: bool = False) -> list[float]:
    """
    Contract 1D tent-space array to box-space by inverting corner averaging.

    (2W+1) → (W)

    This is the exact inverse of tent_expand_lanczos_1d for UNMODIFIED data.
    It inverts: corner[j] = (box[j-1] + box[j]) / 2

    WARNING: This is numerically unstable for long arrays or after resampling.
    Use tent_contract_lanczos_1d (integration) for resampled data.
    """
    src_w = len(src)
    assert src_w % 2 == 1, "Source width must be odd"

    dst_w = (src_w - 1) // 2
    dst = [0.0] * dst_w

    # Extract corners from tent space
    corners = [src[2 * j] for j in range(dst_w + 1)]

    # Invert the averaging relationship
    dst[0] = corners[0]
    for j in range(1, dst_w):
        dst[j] = 2 * corners[j] - dst[j - 1]

    if debug:
        print(f"    Contract (exact inversion): {src_w} → {dst_w}")
        print(f"    Corners used: {corners}")
        print(f"    Result: {dst}")

    return dst


# =============================================================================
# Resampling (shared with original bruteforce)
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


def resample_1d_box(src: list[float], dst_len: int, ratio: float, depth: int = 1,
                    extra_offset: float = 0.0, debug: bool = False) -> list[float]:
    """Resample 1D array using box filter with exact overlap computation."""
    src_len = len(src)
    dst = [0.0] * dst_len

    if src_len == dst_len and abs(extra_offset) < 1e-10:
        return list(src)

    if dst_len == 1:
        return [sum(src) / src_len]

    scale = ratio
    filter_scale = ratio

    # Offset to align content centers
    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1))) + extra_offset

    if debug:
        print(f"    Resample: {src_len} → {dst_len}, scale={scale:.6f}, offset={offset:.6f}")

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


# =============================================================================
# Additional sampling kernels for resampling step
# =============================================================================

def lanczos_kernel(x: float, a: int = 3) -> float:
    """Lanczos kernel with parameter a."""
    if abs(x) < 1e-10:
        return 1.0
    if abs(x) >= a:
        return 0.0
    pi_x = math.pi * x
    return (math.sin(pi_x) / pi_x) * (math.sin(pi_x / a) / (pi_x / a))


def triangle_kernel(x: float) -> float:
    """Triangle (bilinear) kernel."""
    x = abs(x)
    return max(0.0, 1.0 - x)


def box_kernel(x: float) -> float:
    """Box kernel."""
    return 1.0 if abs(x) <= 0.5 else 0.0


KERNELS = {
    'box': (box_kernel, 0.5),
    'triangle': (triangle_kernel, 1.0),
    'lanczos2': (lambda x: lanczos_kernel(x, 2), 2.0),
    'lanczos3': (lambda x: lanczos_kernel(x, 3), 3.0),
}


def resample_1d_kernel(src: list[float], dst_len: int, ratio: float, depth: int = 1,
                       kernel_name: str = 'box', filter_width: float | None = None,
                       extra_offset: float = 0.0, debug: bool = False) -> list[float]:
    """Resample 1D array using specified kernel."""
    if kernel_name == 'box':
        return resample_1d_box(src, dst_len, ratio, depth, extra_offset, debug)

    kernel_func, kernel_radius = KERNELS.get(kernel_name, KERNELS['box'])

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


# =============================================================================
# Full Pipeline
# =============================================================================

def format_array(arr: list[float], max_show: int = 20) -> str:
    """Format array for display."""
    if len(arr) <= max_show:
        return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"
    else:
        first = ", ".join(f"{v:.4f}" for v in arr[:5])
        last = ", ".join(f"{v:.4f}" for v in arr[-5:])
        return f"[{first}, ... ({len(arr)} total) ..., {last}]"


def full_pipeline(src: list[float], output_len: int, ratio: float, recurse: int = 1,
                  kernel_name: str = 'box', filter_width: float | None = None,
                  extra_offset: float = 0.0, debug: bool = False) -> list[float]:
    """
    Full Lanczos3-constrained tent-space downscaling pipeline.

    box → tent_expand_lanczos (×recurse) → resample → tent_contract_lanczos (×recurse) → box
    """
    data = src

    if debug:
        print(f"  Input box ({len(data)}): {format_array(data)}")

    # Expand to tent space using Lanczos3-constrained peaks
    for i in range(recurse):
        prev_len = len(data)
        data = tent_expand_lanczos_1d(data, debug=debug)
        if debug:
            print(f"  Expand {i+1}: {prev_len} → {len(data)}")
            print(f"    {format_array(data)}")

    # Calculate target tent-space size
    scale = 2 ** recurse
    tent_target = scale * output_len + (scale - 1)

    if debug:
        print(f"  Resample target: {len(data)} → {tent_target}")

    # Resample in tent space
    data = resample_1d_kernel(data, tent_target, ratio=ratio, depth=recurse,
                              kernel_name=kernel_name, filter_width=filter_width,
                              extra_offset=extra_offset, debug=debug)

    if debug:
        print(f"  After resample ({len(data)}): {format_array(data)}")

    # Contract back to box space (invert corner averaging)
    for i in range(recurse):
        prev_len = len(data)
        data = tent_contract_lanczos_1d(data, debug=debug)
        if debug:
            print(f"  Contract {i+1}: {prev_len} → {len(data)}")
            print(f"    {format_array(data)}")

    return data


def derive_kernel_bruteforce(input_len: int, output_len: int, output_idx: int, ratio: float,
                              recurse: int = 1, kernel_name: str = 'box',
                              filter_width: float | None = None,
                              extra_offset: float = 0.0) -> list[float]:
    """Derive the effective kernel by computing impulse responses."""
    kernel = []

    for i in range(input_len):
        impulse = [0.0] * input_len
        impulse[i] = 1.0

        output = full_pipeline(impulse, output_len, ratio=ratio, recurse=recurse,
                               kernel_name=kernel_name, filter_width=filter_width,
                               extra_offset=extra_offset)

        if 0 <= output_idx < len(output):
            kernel.append(output[output_idx])
        else:
            kernel.append(0.0)

    return kernel


# =============================================================================
# Verification: Lanczos3 zeros property
# =============================================================================

def verify_lanczos_zeros():
    """Verify that Lanczos3 has zeros at integer positions."""
    print("Lanczos3 values at integer positions:")
    for x in range(-6, 7):
        val = lanczos3(float(x))
        print(f"  Lanczos3({x}) = {val:.10f}")
    print()

    print("Lanczos3 values at half-integer positions:")
    for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
        val = lanczos3(x)
        print(f"  Lanczos3({x}) = {val:.6f}")


def verify_reversibility():
    """Verify that expand → contract is reversible for unmodified data."""
    print("Reversibility test (no resampling):")

    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"  Original: {test_data}")

    # Expand
    expanded = tent_expand_lanczos_1d(test_data, debug=True)
    print(f"  Expanded ({len(expanded)}): {format_array(expanded)}")

    # Show corners and peaks
    w = len(test_data)
    corners = [expanded[2*j] for j in range(w + 1)]
    peaks = [expanded[2*i + 1] for i in range(w)]
    print(f"  Corners (even positions): {corners}")
    print(f"  Peaks (odd positions, Lanczos3 of corners): {[f'{p:.4f}' for p in peaks]}")

    # Contract using exact inversion (for unmodified data)
    contracted_exact = tent_contract_exact_1d(expanded, debug=True)
    print(f"  Contracted (exact): {contracted_exact}")
    error_exact = sum(abs(a - b) for a, b in zip(test_data, contracted_exact))
    print(f"  Round-trip error (exact): {error_exact:.2e}")

    # Contract using integration (general purpose)
    contracted_int = tent_contract_lanczos_1d(expanded)
    print(f"  Contracted (integration): {[f'{v:.4f}' for v in contracted_int]}")
    error_int = sum(abs(a - b) for a, b in zip(test_data, contracted_int))
    print(f"  Round-trip error (integration): {error_int:.2e}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lanczos3-constrained tent-space kernel derivation"
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
    parser.add_argument('--kernel', '-k', type=str, default='box',
                        choices=list(KERNELS.keys()),
                        help="Sampling kernel (default: box)")
    parser.add_argument('--width', '-w', type=float, default=None,
                        help="Filter width in box space (default: ratio/2)")
    parser.add_argument('--offset', type=float, default=0.0,
                        help="Fractional offset for phase shift testing")
    parser.add_argument('--verify', '-v', action='store_true',
                        help="Run verification tests")
    parser.add_argument('--compare', '-c', action='store_true',
                        help="Compare with volume-based approach")

    args = parser.parse_args()

    if args.verify:
        verify_lanczos_zeros()
        print()
        verify_reversibility()
        return

    input_len = args.input_len
    output_len = int(input_len / args.ratio)
    output_idx = args.output_idx if args.output_idx is not None else output_len // 2

    if output_idx >= output_len:
        print(f"Error: output_idx {output_idx} out of bounds (output length is {output_len})")
        return

    width_box = args.width if args.width is not None else args.ratio / 2
    filter_width = width_box * 2  # Convert to tent space

    print(f"Input length: {input_len}")
    print(f"Output length: {output_len} (ratio {args.ratio}×)")
    print(f"Output index: {output_idx}")
    print(f"Recurse levels: {args.recurse}")
    print(f"Kernel: {args.kernel}")
    print(f"Width: {width_box} box px ({filter_width} tent units)")
    print()

    if args.compare:
        # Compare Lanczos3-constrained vs volume-based
        print("=" * 60)
        print("Comparison: Lanczos3-constrained vs Volume-based")
        print("=" * 60)
        print()

        # Import the volume-based version
        try:
            import tent_kernel_bruteforce as volume_based

            kernel_lanczos = derive_kernel_bruteforce(
                input_len, output_len, output_idx, ratio=args.ratio,
                recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width,
                extra_offset=args.offset
            )

            kernel_volume = volume_based.derive_kernel_bruteforce(
                input_len, output_len, output_idx, ratio=args.ratio,
                recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width,
                extra_offset=args.offset
            )

            # Find non-zero regions
            nonzero_l = [(i, k) for i, k in enumerate(kernel_lanczos) if abs(k) > 1e-10]
            nonzero_v = [(i, k) for i, k in enumerate(kernel_volume) if abs(k) > 1e-10]

            if nonzero_l:
                first_l, last_l = nonzero_l[0][0], nonzero_l[-1][0]
                kernel_l_slice = kernel_lanczos[first_l:last_l + 1]
                print("Lanczos3-constrained kernel:")
                for denom in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    int_coeffs = [round(k * denom) for k in kernel_l_slice]
                    reconstructed = [c / denom for c in int_coeffs]
                    error = sum(abs(a - b) for a, b in zip(kernel_l_slice, reconstructed))
                    if error < 1e-8:
                        print(f"  {int_coeffs} / {denom}")
                        break
                else:
                    print(f"  {[f'{k:.6f}' for k in kernel_l_slice]}")

            if nonzero_v:
                first_v, last_v = nonzero_v[0][0], nonzero_v[-1][0]
                kernel_v_slice = kernel_volume[first_v:last_v + 1]
                print("Volume-based kernel:")
                for denom in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    int_coeffs = [round(k * denom) for k in kernel_v_slice]
                    reconstructed = [c / denom for c in int_coeffs]
                    error = sum(abs(a - b) for a, b in zip(kernel_v_slice, reconstructed))
                    if error < 1e-8:
                        print(f"  {int_coeffs} / {denom}")
                        break
                else:
                    print(f"  {[f'{k:.6f}' for k in kernel_v_slice]}")

            # Difference
            if nonzero_l and nonzero_v:
                print()
                print("Difference (Lanczos - Volume):")
                min_idx = min(first_l, first_v)
                max_idx = max(last_l, last_v)
                for i in range(min_idx, max_idx + 1):
                    kl = kernel_lanczos[i] if i < len(kernel_lanczos) else 0.0
                    kv = kernel_volume[i] if i < len(kernel_volume) else 0.0
                    diff = kl - kv
                    if abs(kl) > 1e-10 or abs(kv) > 1e-10:
                        print(f"  [{i}]: {kl:.6f} - {kv:.6f} = {diff:+.6f}")
        except ImportError:
            print("Cannot import tent_kernel_bruteforce.py for comparison")
        return

    # Derive kernel
    print("=" * 60)
    print("Kernel derivation (Lanczos3-constrained)")
    print("=" * 60)
    kernel = derive_kernel_bruteforce(
        input_len, output_len, output_idx, ratio=args.ratio,
        recurse=args.recurse, kernel_name=args.kernel, filter_width=filter_width,
        extra_offset=args.offset
    )

    # Find non-zero region
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

        # Check symmetry
        n = len(kernel_slice)
        is_symmetric = all(abs(kernel_slice[i] - kernel_slice[n-1-i]) < 1e-8 for i in range(n//2 + 1))
        print(f"  Symmetric: {is_symmetric}")
        print()

        # Print as fractions of a power of 2
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

    # Verification
    print()
    print("=" * 60)
    print("Verification: constant input should give constant output")
    test_input = [0.5] * input_len
    test_output = full_pipeline(test_input, output_len, ratio=args.ratio,
                                recurse=args.recurse, kernel_name=args.kernel,
                                filter_width=filter_width, extra_offset=args.offset)
    print(f"  Input:  all 0.5")
    print(f"  Output[{output_idx}]: {test_output[output_idx]:.10f}")
    print(f"  (Should be 0.5)")


if __name__ == "__main__":
    main()
