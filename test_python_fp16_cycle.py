#!/usr/bin/env python3
"""
Test FP16 roundtrip cycling using standard Python ML libraries.

This simulates what happens when ML pipelines load images, convert to FP16
for inference, and save back. Compares degradation vs CRA's dithered approach.

Pipeline per cycle:
1. Load PNG with PIL
2. Convert to numpy float32, normalize to 0-1
3. Cast to float16 (standard ML approach - no dithering)
4. Cast back to float32
5. Denormalize to 0-255, convert to uint8
6. Save PNG

Saves output at cycles 1, 10, and 100.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL and numpy are required")
    print("Install with: pip install pillow numpy")
    sys.exit(1)


def png_to_fp16_to_png(img_array: np.ndarray, use_rounding: bool = True) -> np.ndarray:
    """
    Simulate ML FP16 roundtrip:
    uint8 -> float32 (0-1) -> float16 -> float32 -> uint8

    Args:
        img_array: Input uint8 image
        use_rounding: If True, use proper rounding (lossless for uint8).
                      If False, use truncation (common bug in ML code).

    With rounding: All 256 uint8 values roundtrip perfectly through FP16.
    With truncation: 131/256 values lose 1 each cycle (systematic drift to 0).
    """
    # Normalize to 0-1 float32 (standard preprocessing)
    float32 = img_array.astype(np.float32) / 255.0

    # Cast to FP16 (what happens when model uses half precision)
    float16 = float32.astype(np.float16)

    # Cast back to float32 (for post-processing)
    float32_back = float16.astype(np.float32)

    # Denormalize and convert to uint8
    if use_rounding:
        # Correct: round before truncating to uint8
        uint8_back = np.clip(np.round(float32_back * 255.0), 0, 255).astype(np.uint8)
    else:
        # Common bug: astype(uint8) truncates, causing systematic drift
        uint8_back = np.clip(float32_back * 255.0, 0, 255).astype(np.uint8)

    return uint8_back


def run_cycle_test(input_path: Path, output_dir: Path, cycles: int = 100,
                   save_at: list[int] = [1, 10, 100], verbose: bool = True,
                   use_rounding: bool = True):
    """
    Run the FP16 cycle test.

    Args:
        input_path: Input PNG file
        output_dir: Output directory
        cycles: Number of cycles to run
        save_at: Cycle numbers at which to save output
        verbose: Print progress
        use_rounding: Use proper rounding (True) or truncation (False)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get basename for output files
    basename = input_path.stem
    mode = "rounded" if use_rounding else "truncated"

    if verbose:
        print("=" * 60)
        print(f"Python FP16 Cycle Test ({mode})")
        print("=" * 60)
        print(f"\nInput:  {input_path}")
        print(f"Output: {output_dir}")
        print(f"Cycles: {cycles}")
        print(f"Save at: {save_at}")
        print(f"Mode: {mode} ({'lossless' if use_rounding else 'lossy - 131/256 values drift down'})")
        print()

    # Load original image
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Save original for comparison
    original_path = output_dir / f"{basename}_original.png"
    img.save(original_path)
    if verbose:
        print(f"Saved original: {original_path}")

    # Convert to numpy array
    current = np.array(img)

    if verbose:
        print(f"\nRunning {cycles} cycles...")

    for i in range(1, cycles + 1):
        # Run one FP16 roundtrip cycle
        current = png_to_fp16_to_png(current, use_rounding=use_rounding)

        # Progress
        if verbose and i % 10 == 0:
            print(f"  Cycle {i} complete")

        # Save at specified cycles
        if i in save_at:
            cycle_img = Image.fromarray(current)
            cycle_path = output_dir / f"{basename}_python_fp16_{mode}_cycle_{i}.png"
            cycle_img.save(cycle_path)
            if verbose:
                print(f"  -> Saved cycle {i}: {cycle_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print("\nOutput files:")
        for f in sorted(output_dir.glob(f"{basename}*.png")):
            size = f.stat().st_size
            print(f"  {f.name}: {size:,} bytes")
        print()
        print("Compare with CRA's dithered FP16 output (test-fp16-cycle.sh)")
        print("to see the difference between naive truncation and dithering.")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test FP16 roundtrip cycling using standard Python ML approach"
    )
    parser.add_argument("--input", "-i", type=Path,
                       help="Input PNG file (default: scripts/assets/forest_plain.png)")
    parser.add_argument("--output-dir", "-o", type=Path,
                       default=Path("test_output/python_fp16_cycle"),
                       help="Output directory")
    parser.add_argument("--cycles", "-c", type=int, default=100,
                       help="Number of cycles (default: 100)")
    parser.add_argument("--truncate", action="store_true",
                       help="Use truncation instead of rounding (demonstrates common ML bug)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode")

    args = parser.parse_args()

    # Find input file (same default as test-fp16-cycle.sh)
    script_dir = Path(__file__).parent
    default_input = script_dir / "scripts" / "assets" / "forest_plain.png"

    if args.input:
        input_path = args.input
    elif default_input.exists():
        input_path = default_input
    else:
        print(f"Error: Default input file not found: {default_input}")
        print("Use --input to specify an input PNG file")
        sys.exit(1)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    run_cycle_test(
        input_path=input_path,
        output_dir=args.output_dir,
        cycles=args.cycles,
        save_at=[1, 10, 100],
        verbose=not args.quiet,
        use_rounding=not args.truncate
    )


if __name__ == "__main__":
    main()
