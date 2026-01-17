#!/bin/bash
# Test script for alternating RGB565 <-> RGB332 dithering cycles
# This tests the accumulation of quantization error when cycling between
# two different reduced bit-depth formats

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRA="${SCRIPT_DIR}/port/target/release/cra"

# Default input image
INPUT_IMAGE="${1:-${SCRIPT_DIR}/scripts/assets/forest_plain.png}"
OUTPUT_DIR="${SCRIPT_DIR}/test_output/rgb565_rgb332_cycle"

# Cycles to save
SAVE_CYCLES="1 2 3 5 10 20 50 100"
MAX_CYCLE=100

# Check CLI exists
if [ ! -f "$CRA" ]; then
    echo "Error: CRA CLI not found at $CRA"
    echo "Run ./build-cli.sh first"
    exit 1
fi

# Check input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: Input image not found at $INPUT_IMAGE"
    echo "Usage: $0 [input_image.png]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Get basename for output files
INPUT_BASENAME=$(basename "$INPUT_IMAGE" .png)

echo "=========================================="
echo "RGB565 <-> RGB332 Alternating Cycle Test"
echo "=========================================="
echo ""
echo "Input:    $INPUT_IMAGE"
echo "Output:   $OUTPUT_DIR"
echo "Cycles:   $SAVE_CYCLES"
echo ""
echo "Each cycle alternates between:"
echo "  Odd cycles:  Apply RGB565 dithering (16-bit: 5-6-5)"
echo "  Even cycles: Apply RGB332 dithering (8-bit: 3-3-2)"
echo ""

# Copy original to compare
cp "$INPUT_IMAGE" "$OUTPUT_DIR/${INPUT_BASENAME}_original.png"

# Initialize: Start with the input PNG
cp "$INPUT_IMAGE" "$TEMP_DIR/current.png"

echo "Running $MAX_CYCLE cycles..."
echo ""

for i in $(seq 1 $MAX_CYCLE); do
    # Alternate between RGB565 (odd) and RGB332 (even)
    if [ $((i % 2)) -eq 1 ]; then
        FORMAT="RGB565"
    else
        FORMAT="RGB332"
    fi

    # Apply dithering to the reduced format, output as PNG (internally dithered)
    "$CRA" \
        -i "$TEMP_DIR/current.png" \
        --format "$FORMAT" \
        -o "$TEMP_DIR/current.png"

    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Cycle $i complete ($FORMAT)"
    fi

    # Save at specified cycles
    for save_cycle in $SAVE_CYCLES; do
        if [ "$i" -eq "$save_cycle" ]; then
            cp "$TEMP_DIR/current.png" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${i}.png"
            echo "  -> Saved cycle $i ($FORMAT)"
            break
        fi
    done
done

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/${INPUT_BASENAME}*
echo ""
echo "Compare visually:"
echo "  Original: ${OUTPUT_DIR}/${INPUT_BASENAME}_original.png"
for save_cycle in $SAVE_CYCLES; do
    if [ $((save_cycle % 2)) -eq 1 ]; then
        FORMAT="RGB565"
    else
        FORMAT="RGB332"
    fi
    echo "  Cycle $save_cycle:  ${OUTPUT_DIR}/${INPUT_BASENAME}_cycle_${save_cycle}.png ($FORMAT)"
done
echo ""
echo "Done!"
