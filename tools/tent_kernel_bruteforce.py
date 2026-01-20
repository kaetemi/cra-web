#!/usr/bin/env python3
"""
Brute-force tent-space kernel derivation via impulse response.

For each input pixel position, set it to 1.0 (all others 0.0),
run through the full pipeline, and measure the output at a reference pixel.
The sequence of outputs IS the effective kernel.

Pipeline: box → tent_expand → resample → tent_contract → box
"""

from __future__ import annotations
import argparse


def tent_expand_1d(src: list[float]) -> list[float]:
    """
    Expand 1D box-space array to tent-space.

    (W) → (2W+1)
    - Even positions: corners (average of neighbors)
    - Odd positions: centers (volume-preserving adjustment)
    """
    w = len(src)
    dst_w = 2 * w + 1
    dst = [0.0] * dst_w

    # Helper with clamp-to-edge
    def get_src(i: int) -> float:
        return src[max(0, min(w - 1, i))]

    # Pass 1: Interpolate all positions
    for dx in range(dst_w):
        if dx % 2 == 1:
            # Center: directly from source pixel
            sx = (dx - 1) // 2
            dst[dx] = get_src(sx)
        else:
            # Corner: average of two neighbors
            sx_left = (dx // 2) - 1
            sx_right = dx // 2
            dst[dx] = (get_src(sx_left) + get_src(sx_right)) / 2

    # Pass 2: Adjust centers for volume preservation
    # M' = 4V - 0.5*sum(edges) - 0.25*sum(corners)
    for sx in range(w):
        dx = sx * 2 + 1  # Center position in tent space
        original = src[sx]

        # Edges (horizontal neighbors in tent space)
        edge_left = dst[dx - 1]   # corner to the left
        edge_right = dst[dx + 1]  # corner to the right
        edge_sum = edge_left + edge_right

        # In 1D, "corners" of the integration stencil are the same as edges
        # Actually in 1D, the integration is just: 1/4 * left + 1/2 * center + 1/4 * right
        # So we solve: V = 1/4*left + 1/2*M + 1/4*right
        # M = 2V - 0.5*left - 0.5*right = 2V - 0.5*(left + right)
        adjusted = 2 * original - 0.5 * edge_sum
        dst[dx] = adjusted

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


def box_integrated(src_pos: float, si: int, filter_scale: float) -> float:
    """
    Compute overlap between destination pixel footprint and source pixel cell.

    This matches the Rust implementation in kernels.rs exactly:
    - Destination pixel at src_pos has footprint [src_pos - half_width, src_pos + half_width]
    - Source pixel si owns the cell [si - 0.5, si + 0.5]
    - Returns the length of the overlap
    """
    half_width = 0.5 * filter_scale
    dst_start = src_pos - half_width
    dst_end = src_pos + half_width

    src_start = si - 0.5
    src_end = si + 0.5

    overlap_start = max(dst_start, src_start)
    overlap_end = min(dst_end, src_end)

    return max(0.0, overlap_end - overlap_start)


def resample_1d_box(src: list[float], dst_len: int) -> list[float]:
    """
    Resample 1D array using box filter with exact overlap computation.

    This matches the Rust implementation in kernels.rs / separable.rs:
    - Uses sample-to-sample mapping for tent space
    - Each source sample "owns" a unit cell [si - 0.5, si + 0.5]
    - Weight = overlap between filter footprint and cell
    """
    src_len = len(src)
    dst = [0.0] * dst_len

    if src_len == dst_len:
        return list(src)

    if dst_len == 1:
        # Average all samples
        return [sum(src) / src_len]

    # Sample-to-sample scale (tent mode)
    scale = (src_len - 1) / (dst_len - 1)

    # Center offset to align centers (matches Rust)
    center_offset = (src_len - dst_len * scale) / 2.0

    # Filter scale for box integration (use scale directly, clamped to >= 1)
    filter_scale = max(scale, 1.0)

    # Box radius: how far to search for overlapping source samples
    box_radius = int((0.5 * filter_scale) + 1) + 1

    for di in range(dst_len):
        # Map destination position to source position (matches Rust formula exactly)
        src_pos = (di + 0.5) * scale - 0.5 + center_offset
        center = int(src_pos)

        # Find valid source index range
        start = max(0, center - box_radius)
        end = min(src_len - 1, center + box_radius)

        # Collect weights
        weights = []
        weight_sum = 0.0

        for si in range(start, end + 1):
            w = box_integrated(src_pos, si, filter_scale)
            weights.append(w)
            weight_sum += w

        # Compute weighted sum (normalize)
        if weight_sum > 1e-8:
            result = 0.0
            for i, w in enumerate(weights):
                result += src[start + i] * w / weight_sum
            dst[di] = result
        else:
            # Fallback: nearest sample
            fallback = int(round(src_pos))
            fallback = max(0, min(src_len - 1, fallback))
            dst[di] = src[fallback]

    return dst


def full_pipeline(src: list[float], output_len: int, recurse: int = 1) -> list[float]:
    """
    Full tent-space downscaling pipeline.

    box → tent_expand (×recurse) → resample → tent_contract (×recurse) → box
    """
    # Expand to tent space (recurse times)
    data = src
    for _ in range(recurse):
        data = tent_expand_1d(data)

    # Calculate target tent-space size for desired output
    # After recurse expansions: len = 2^recurse * W + (2^recurse - 1)
    # After recurse contractions of M samples: (M - (2^recurse - 1)) / 2^recurse
    # We want final output to be output_len, so:
    # tent_target = 2^recurse * output_len + (2^recurse - 1)
    scale = 2 ** recurse
    tent_target = scale * output_len + (scale - 1)

    # Resample in tent space using box filter (matches Rust implementation)
    data = resample_1d_box(data, tent_target)

    # Contract back to box space (recurse times)
    for _ in range(recurse):
        data = tent_contract_1d(data)

    return data


def derive_kernel_bruteforce(input_len: int, output_len: int, output_idx: int, recurse: int = 1) -> list[float]:
    """
    Derive the effective kernel by computing impulse responses.

    For each input position i, create an impulse (1.0 at i, 0.0 elsewhere),
    run through the pipeline, and record the output at output_idx.
    """
    kernel = []

    for i in range(input_len):
        # Create impulse at position i
        impulse = [0.0] * input_len
        impulse[i] = 1.0

        # Run through pipeline
        output = full_pipeline(impulse, output_len, recurse)

        # Record contribution to reference output pixel
        if 0 <= output_idx < len(output):
            kernel.append(output[output_idx])
        else:
            kernel.append(0.0)

    return kernel


def main():
    parser = argparse.ArgumentParser(
        description="Brute-force tent-space kernel derivation via impulse response"
    )
    parser.add_argument('--input-len', '-i', type=int, default=16,
                       help="Input array length (default: 16)")
    parser.add_argument('--ratio', '-r', type=float, default=2.0,
                       help="Downscaling ratio (default: 2.0)")
    parser.add_argument('--output-idx', '-o', type=int, default=None,
                       help="Output index to measure (default: center)")
    parser.add_argument('--recurse', '-R', type=int, default=1,
                       help="Number of tent expansion/contraction cycles (default: 1)")

    args = parser.parse_args()

    input_len = args.input_len
    output_len = int(input_len / args.ratio)
    output_idx = args.output_idx if args.output_idx is not None else output_len // 2

    print(f"Input length: {input_len}")
    print(f"Output length: {output_len} (ratio {args.ratio}×)")
    print(f"Output index: {output_idx}")
    print(f"Recurse levels: {args.recurse}")
    print()

    # Derive kernel
    kernel = derive_kernel_bruteforce(input_len, output_len, output_idx, args.recurse)

    # Find non-zero region
    nonzero = [(i, k) for i, k in enumerate(kernel) if abs(k) > 1e-10]

    if nonzero:
        first_idx = nonzero[0][0]
        last_idx = nonzero[-1][0]

        print(f"Non-zero kernel region: input indices {first_idx} to {last_idx}")
        print(f"Kernel width: {last_idx - first_idx + 1} samples")
        print()

        # Print kernel values
        print("Kernel coefficients:")
        kernel_slice = kernel[first_idx:last_idx + 1]

        # Try to find a common denominator
        total = sum(kernel_slice)
        print(f"  Sum: {total:.10f}")
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
        for i, k in enumerate(kernel_slice):
            input_pos = first_idx + i
            # Position relative to output pixel center
            # Output pixel output_idx covers input range [output_idx * ratio, (output_idx+1) * ratio]
            # Center is at (output_idx + 0.5) * ratio
            center = (output_idx + 0.5) * args.ratio
            rel_pos = input_pos - center
            print(f"    Input[{input_pos}] (rel {rel_pos:+.1f}): {k:.6f}")
    else:
        print("Kernel is all zeros!")

    # Verify by running a simple test
    print()
    print("=" * 60)
    print("Verification: constant input should give constant output")
    test_input = [0.5] * input_len
    test_output = full_pipeline(test_input, output_len, args.recurse)
    print(f"  Input:  all 0.5")
    print(f"  Output[{output_idx}]: {test_output[output_idx]:.10f}")
    print(f"  (Should be 0.5)")


if __name__ == "__main__":
    main()
