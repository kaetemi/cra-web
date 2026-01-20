#!/usr/bin/env python3
"""
Brute-force tent-space kernel derivation via impulse response.

For each input pixel position, set it to 1.0 (all others 0.0),
run through the full pipeline, and measure the output at a reference pixel.
The sequence of outputs IS the effective kernel.

Pipeline: box → tent_expand (×recurse) → resample → tent_contract (×recurse) → box
"""

from __future__ import annotations
import argparse


def tent_expand_1d(src: list[float], debug: bool = False) -> list[float]:
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

    if debug:
        print(f"    After pass 1 (interpolation): {format_array(dst)}")

    # Pass 2: Adjust centers for volume preservation
    # In 1D: V = 1/4*left + 1/2*center + 1/4*right
    # So: M = 2V - 0.5*left - 0.5*right = 2V - 0.5*(left + right)
    for sx in range(w):
        dx = sx * 2 + 1  # Center position in tent space
        original = src[sx]

        # Edges (horizontal neighbors in tent space)
        edge_left = dst[dx - 1]   # corner to the left
        edge_right = dst[dx + 1]  # corner to the right
        edge_sum = edge_left + edge_right

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


def resample_1d_box(src: list[float], dst_len: int, ratio: float, depth: int = 1, debug: bool = False) -> list[float]:
    """
    Resample 1D array using box filter with exact overlap computation.

    For R× downscale in tent space, scale = R and filter_scale = 2.0 (1x native).
    The offset accounts for the fringe growth at each depth level.

    Key insight: The fringe grows as (fringe * 2) + 0.5 per depth:
    - Depth 1: fringe = 0.5 box pixels = 1 sample
    - Depth 2: fringe = 1.5 box pixels = 3 samples
    - Depth 3: fringe = 3.5 box pixels = 7 samples
    Formula: fringe_samples = 2^depth - 1

    The mapping must align content centers, not sample endpoints.
    """
    src_len = len(src)
    dst = [0.0] * dst_len

    if src_len == dst_len:
        return list(src)

    if dst_len == 1:
        return [sum(src) / src_len]

    # For R× downscale in tent space:
    # - scale = R (coordinate mapping: output sample i → input position R*i + offset)
    # - filter_scale = R (filter spans R units to cover the full output pixel footprint)
    #
    # The filter must span R tent-space units to properly integrate over all R input
    # box pixels that contribute to each output pixel. With scale=R, consecutive output
    # samples are R apart in input space, so the filter width must match.
    #
    # Note: "1x native" in TENT-SPACE.md refers to the 2× case specifically.
    # The general rule is filter_scale = ratio (which equals 2 for 2× downscale).
    scale = ratio
    filter_scale = ratio  # Filter spans ratio units to cover full footprint

    # Offset to align content centers
    # At depth d, output tent sample 'fringe' (= 2^d - 1) corresponds to output box pixel 0 center.
    # For R× downscale, output box 0 covers input box [0, R), centered at input box R/2.
    #
    # At depth d, the continuous mapping is: tent = 2^d × box + (2^(d-1) - 1)
    # Input tent position for center R/2 = 2^d × (R/2) + 2^(d-1) - 1 = (R+1) × 2^(d-1) - 1
    #
    # Mapping: src_pos = dst_i × scale + offset
    # (2^d - 1) × R + offset = (R+1) × 2^(d-1) - 1
    # offset = (R+1) × 2^(d-1) - 1 - R × (2^d - 1)
    #        = (R+1) × 2^(d-1) - 1 - R × 2^d + R
    #        = (R-1) × (1 - 2^(d-1))
    offset = (ratio - 1.0) * (1.0 - (2 ** (depth - 1)))

    if debug:
        print(f"    Resample: {src_len} → {dst_len}, scale={scale:.6f}, filter_scale={filter_scale:.6f}, offset={offset:.6f}")

    box_radius = int(filter_scale / 2 + 1) + 1

    # For debug: track sample counts
    sample_counts = []

    for di in range(dst_len):
        # Content-aligned mapping: dst sample di maps to src position di * scale + offset
        src_pos = di * scale + offset
        center = int(src_pos)

        # Find valid source index range
        start = max(0, center - box_radius)
        end = min(src_len - 1, center + box_radius)

        # Collect weights using the filter half-width
        weights = []
        weight_sum = 0.0

        for si in range(start, end + 1):
            w = box_integrated(src_pos, si, filter_scale)
            weights.append(w)
            weight_sum += w

        # Count non-zero weights
        nonzero_weights = [(si, w) for si, w in enumerate(weights, start) if w > 1e-10]
        sample_counts.append(len(nonzero_weights))

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

    if debug:
        # Show sample count statistics
        from collections import Counter
        count_hist = Counter(sample_counts)
        print(f"    Box filter sample counts: {dict(sorted(count_hist.items()))}")
        # Show a few examples
        print(f"    Examples (dst_idx → src_pos, samples used):")
        examples = [0, 1, dst_len // 2, dst_len - 2, dst_len - 1]
        for di in examples:
            if di < dst_len:
                src_pos = di * scale + offset
                center = int(src_pos)
                start = max(0, center - box_radius)
                end = min(src_len - 1, center + box_radius)
                weights_debug = []
                for si in range(start, end + 1):
                    w = box_integrated(src_pos, si, filter_scale)
                    if w > 1e-10:
                        weights_debug.append((si, w))
                weight_str = ", ".join(f"{si}:{w:.3f}" for si, w in weights_debug)
                print(f"      dst[{di}] → src_pos={src_pos:.2f}, {len(weights_debug)} samples: [{weight_str}]")

    return dst


def format_array(arr: list[float], max_show: int = 20) -> str:
    """Format array for display, showing values as fractions where clean."""
    if len(arr) <= max_show:
        return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"
    else:
        first = ", ".join(f"{v:.4f}" for v in arr[:5])
        last = ", ".join(f"{v:.4f}" for v in arr[-5:])
        return f"[{first}, ... ({len(arr)} total) ..., {last}]"


def full_pipeline(src: list[float], output_len: int, ratio: float, recurse: int = 1, debug: bool = False) -> list[float]:
    """
    Full tent-space downscaling pipeline.

    box → tent_expand (×recurse) → resample → tent_contract (×recurse) → box
    """
    data = src

    if debug:
        print(f"  Input box ({len(data)}): {format_array(data)}")

    # Track dimensions through the pipeline
    box_w = len(src)

    # Expand to tent space (recurse times)
    for i in range(recurse):
        prev_len = len(data)
        data = tent_expand_1d(data, debug=debug)
        if debug:
            # Compute fringe in original box units
            # After i+1 expansions: tent_len = 2^(i+1) * box_w + (2^(i+1) - 1)
            # The "content" centers span from sample 2^(i+1)-1 to tent_len - 2^(i+1)
            # Actually simpler: fringe = (2^n - 1) / 2 box pixels on each side
            scale_factor = 2 ** (i + 1)
            fringe_box = (scale_factor - 1) / 2.0
            print(f"  Expand {i+1}: {prev_len} → {len(data)} (fringe = {fringe_box:.1f} box px)")
            print(f"    {format_array(data)}")

    # Calculate target tent-space size for desired output
    # After recurse expansions: len = 2^recurse * W + (2^recurse - 1)
    # We want final output to be output_len, so:
    # tent_target = 2^recurse * output_len + (2^recurse - 1)
    scale = 2 ** recurse
    tent_target = scale * output_len + (scale - 1)

    if debug:
        print(f"  Resample target: {len(data)} → {tent_target}")

    # Resample in tent space using box filter
    data = resample_1d_box(data, tent_target, ratio=ratio, depth=recurse, debug=debug)

    if debug:
        print(f"  After resample ({len(data)}): {format_array(data)}")

    # Contract back to box space (recurse times)
    for i in range(recurse):
        prev_len = len(data)
        data = tent_contract_1d(data)
        if debug:
            print(f"  Contract {i+1}: {prev_len} → {len(data)}")
            print(f"    {format_array(data)}")

    return data


def derive_kernel_bruteforce(input_len: int, output_len: int, output_idx: int, ratio: float, recurse: int = 1) -> list[float]:
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
        output = full_pipeline(impulse, output_len, ratio=ratio, recurse=recurse)

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
    parser.add_argument('--debug', '-d', action='store_true',
                       help="Show all intermediate stages")
    parser.add_argument('--sequence', '-s', action='store_true',
                       help="Use sequential input [0,1,2,3,...] instead of impulse")

    args = parser.parse_args()

    input_len = args.input_len
    output_len = int(input_len / args.ratio)
    output_idx = args.output_idx if args.output_idx is not None else output_len // 2

    # Bounds check
    if output_idx >= output_len:
        print(f"Error: output_idx {output_idx} is out of bounds (output length is {output_len}, valid indices 0-{output_len-1})")
        return

    print(f"Input length: {input_len}")
    print(f"Output length: {output_len} (ratio {args.ratio}×)")
    print(f"Output index: {output_idx}")
    print(f"Recurse levels: {args.recurse}")
    print()

    # Show dimension progression
    print("Dimension progression:")
    w = input_len
    for i in range(args.recurse):
        tent_w = 2 * w + 1
        fringe = (2 ** (i+1) - 1) / 2.0
        print(f"  Depth {i+1}: box {w} → tent {tent_w} (fringe = {fringe:.1f} original box px)")
        w = tent_w

    # Target tent size
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
        # Run with sequential input to see patterns
        print("=" * 60)
        print("Sequential input mode: [0, 1, 2, 3, ...]")
        print("=" * 60)
        seq_input = list(range(input_len))
        output = full_pipeline(seq_input, output_len, ratio=args.ratio, recurse=args.recurse, debug=args.debug)

        print()
        print(f"Input:  {format_array(list(map(float, seq_input)))}")
        print(f"Output: {format_array(output)}")

        # Check if output is linear (which it should be for correct resampling)
        print()
        print("Output linearity check:")
        expected_scale = args.ratio
        for i, v in enumerate(output):
            expected = i * expected_scale + (expected_scale - 1) / 2  # Midpoint of input range
            diff = v - expected
            marker = " " if abs(diff) < 0.001 else " ← DEVIATION"
            print(f"  output[{i}] = {v:.4f}, expected ≈ {expected:.4f}, diff = {diff:+.4f}{marker}")
        return

    if args.debug:
        # Debug mode: show stages for a single impulse
        print("=" * 60)
        print(f"Debug: impulse at input position {output_idx * int(args.ratio)}")
        print("=" * 60)
        impulse = [0.0] * input_len
        impulse_pos = min(output_idx * int(args.ratio), input_len - 1)
        impulse[impulse_pos] = 1.0
        output = full_pipeline(impulse, output_len, ratio=args.ratio, recurse=args.recurse, debug=True)
        print()
        print(f"Final output: {format_array(output)}")
        print()

    # Derive kernel
    print("=" * 60)
    print("Kernel derivation")
    print("=" * 60)
    kernel = derive_kernel_bruteforce(input_len, output_len, output_idx, ratio=args.ratio, recurse=args.recurse)

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

        # Check symmetry
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

    # Verify by running a simple test
    print()
    print("=" * 60)
    print("Verification: constant input should give constant output")
    test_input = [0.5] * input_len
    test_output = full_pipeline(test_input, output_len, ratio=args.ratio, recurse=args.recurse)
    print(f"  Input:  all 0.5")
    print(f"  Output[{output_idx}]: {test_output[output_idx]:.10f}")
    print(f"  (Should be 0.5)")


if __name__ == "__main__":
    main()
