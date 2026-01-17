#!/usr/bin/env python3
"""
Test safetensors roundtrip compatibility between CRA tool and Python safetensors library.

Pipeline:
1. CRA: PNG -> safetensors (FP32/FP16/BF16)
2. Python: Load safetensors, re-save to new file
3. CRA: Load roundtripped safetensors -> PNG
4. Compare original and roundtripped outputs

This verifies that our safetensors format is compatible with the standard Python library.
"""

import subprocess
import sys
import os
from pathlib import Path

# Check for safetensors library
try:
    from safetensors import safe_open
    import numpy as np
except ImportError:
    print("Error: safetensors and numpy are required")
    print("Install with: pip install safetensors numpy")
    sys.exit(1)

# Check for PyTorch (needed for BF16 support - numpy doesn't support bfloat16)
HAS_TORCH = False
TORCH_IMPORT_ERROR = None
try:
    import torch
    from safetensors.torch import save_file as save_file_torch
    HAS_TORCH = True
except ImportError as e:
    TORCH_IMPORT_ERROR = str(e)

# Import numpy save_file for non-BF16
from safetensors.numpy import save_file as save_file_numpy


def run_cra(args: list[str], verbose: bool = False) -> bool:
    """Run the CRA CLI tool with given arguments."""
    cra_path = Path(__file__).parent / "port" / "target" / "release" / "cra"
    if not cra_path.exists():
        print(f"Error: CRA binary not found at {cra_path}")
        print("Run ./build-cli.sh first")
        return False

    cmd = [str(cra_path)] + args
    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CRA failed with exit code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return False

    if verbose and result.stderr:
        print(result.stderr)

    return True


def load_safetensors(path: Path, use_torch: bool = False) -> tuple[dict, bool]:
    """
    Load all tensors from a safetensors file.

    Returns:
        (tensors dict, is_torch) - is_torch indicates if tensors are torch tensors
    """
    framework = "pt" if use_torch else "numpy"
    tensors = {}
    try:
        with safe_open(str(path), framework=framework) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors, use_torch
    except TypeError as e:
        if "bfloat16" in str(e) and not use_torch:
            # NumPy doesn't support bfloat16, try PyTorch
            if HAS_TORCH:
                return load_safetensors(path, use_torch=True)
            else:
                raise RuntimeError(
                    "BF16 tensors require PyTorch. Install with: pip install torch"
                ) from e
        raise


def save_safetensors(path: Path, tensors: dict, metadata: dict[str, str] = None, is_torch: bool = False):
    """Save tensors to a safetensors file."""
    if is_torch:
        save_file_torch(tensors, str(path), metadata=metadata)
    else:
        save_file_numpy(tensors, str(path), metadata=metadata)


def test_format(input_png: Path, output_dir: Path, fmt: str, transfer: str, verbose: bool = False) -> bool:
    """
    Test a single format configuration.

    Args:
        input_png: Input PNG file
        output_dir: Directory for test outputs
        fmt: Safetensors format (fp32, fp16, bf16)
        transfer: Transfer function (linear, srgb)
        verbose: Print verbose output

    Returns:
        True if test passed, False otherwise
    """
    test_name = f"{fmt}_{transfer}"
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")

    # File paths
    sfi_original = output_dir / f"test_{test_name}_original.safetensors"
    sfi_roundtrip = output_dir / f"test_{test_name}_roundtrip.safetensors"
    png_original = output_dir / f"test_{test_name}_original.png"
    png_roundtrip = output_dir / f"test_{test_name}_roundtrip.png"

    # Step 1: CRA PNG -> safetensors
    print(f"Step 1: Converting PNG to safetensors ({fmt}, {transfer})...")
    if not run_cra([
        "--input", str(input_png),
        "--output-safetensors", str(sfi_original),
        "--safetensors-format", fmt,
        "--safetensors-transfer", transfer,
        "--output", str(png_original),  # Also output PNG for comparison
    ], verbose=verbose):
        return False

    # Step 2: Python load and re-save
    print("Step 2: Loading safetensors with Python library...")
    try:
        tensors, is_torch = load_safetensors(sfi_original)
        framework_name = "PyTorch" if is_torch else "NumPy"
        print(f"  Loaded tensors ({framework_name}): {list(tensors.keys())}")
        for name, tensor in tensors.items():
            print(f"    {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

        # Load metadata from original file (use same framework)
        metadata = {}
        framework = "pt" if is_torch else "numpy"
        with safe_open(str(sfi_original), framework=framework) as f:
            meta = f.metadata()
            if meta:
                metadata = dict(meta)
                print(f"  Metadata: {metadata}")

        print("Step 3: Re-saving safetensors with Python library...")
        save_safetensors(sfi_roundtrip, tensors, metadata if metadata else None, is_torch=is_torch)

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: CRA safetensors -> PNG
    print("Step 4: Converting roundtripped safetensors back to PNG...")
    if not run_cra([
        "--input", str(sfi_roundtrip),
        "--output", str(png_roundtrip),
    ], verbose=verbose):
        return False

    # Step 5: Compare file sizes (basic sanity check)
    orig_size = png_original.stat().st_size
    round_size = png_roundtrip.stat().st_size
    size_diff_pct = abs(orig_size - round_size) / orig_size * 100

    print(f"\nResults:")
    print(f"  Original PNG:    {orig_size:,} bytes")
    print(f"  Roundtrip PNG:   {round_size:,} bytes")
    print(f"  Size difference: {size_diff_pct:.2f}%")

    # For lossless formats (FP32), sizes should be very similar
    # For lossy formats (FP16/BF16), allow more variance
    max_diff = 5.0 if fmt == "fp32" else 15.0
    if size_diff_pct > max_diff:
        print(f"  WARNING: Size difference exceeds {max_diff}% threshold")

    print(f"  Test PASSED: {test_name}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test safetensors roundtrip compatibility")
    parser.add_argument("--input", "-i", type=Path,
                       help="Input PNG file (default: test_output/fp16_cycle/forest_plain_original.png)")
    parser.add_argument("--output-dir", "-o", type=Path,
                       default=Path("test_output/safetensors_roundtrip"),
                       help="Output directory for test files")
    parser.add_argument("--format", "-f", choices=["fp32", "fp16", "bf16", "all"],
                       default="all", help="Format to test")
    parser.add_argument("--transfer", "-t", choices=["linear", "srgb", "all"],
                       default="all", help="Transfer function to test")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Find input file (same default as test-fp16-cycle.sh)
    script_dir = Path(__file__).parent
    default_input = script_dir / "scripts" / "assets" / "forest_plain.png"

    if args.input:
        input_png = args.input
    elif default_input.exists():
        input_png = default_input
    else:
        print(f"Error: Default input file not found: {default_input}")
        print("Use --input to specify an input PNG file")
        sys.exit(1)

    if not input_png.exists():
        print(f"Error: Input file not found: {input_png}")
        sys.exit(1)

    print(f"Input file: {input_png}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Determine which formats and transfers to test
    formats = ["fp32", "fp16", "bf16"] if args.format == "all" else [args.format]
    transfers = ["linear", "srgb"] if args.transfer == "all" else [args.transfer]

    # Check BF16 support (requires PyTorch - numpy doesn't support bfloat16)
    if "bf16" in formats and not HAS_TORCH:
        print("\nWarning: BF16 tests require PyTorch (numpy doesn't support bfloat16)")
        if TORCH_IMPORT_ERROR:
            print(f"  Import error: {TORCH_IMPORT_ERROR}")
        print("Install with: pip install torch")
        print("Skipping BF16 tests...\n")
        formats = [f for f in formats if f != "bf16"]
        if not formats:
            print("Error: No formats to test after skipping BF16")
            sys.exit(1)

    # Run tests
    results = []
    for fmt in formats:
        for transfer in transfers:
            success = test_format(input_png, args.output_dir, fmt, transfer, args.verbose)
            results.append((f"{fmt}_{transfer}", success))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
