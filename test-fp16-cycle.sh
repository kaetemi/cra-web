#!/bin/bash
# Test script for PNG <-> FP16 safetensors cycling
# This tests the accumulation of quantization error over multiple round-trips

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRA="${SCRIPT_DIR}/port/target/release/cra"

# Default input image
INPUT_IMAGE="${1:-${SCRIPT_DIR}/scripts/assets/forest_plain.png}"
OUTPUT_DIR="${SCRIPT_DIR}/test_output/fp16_cycle"

# Cycles to save
CYCLE_1=1
CYCLE_10=10
CYCLE_100=100

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
echo "FP16 Safetensors Cycle Test"
echo "=========================================="
echo ""
echo "Input:    $INPUT_IMAGE"
echo "Output:   $OUTPUT_DIR"
echo "Cycles:   $CYCLE_1, $CYCLE_10, and $CYCLE_100"
echo ""

# Copy original to compare
cp "$INPUT_IMAGE" "$OUTPUT_DIR/${INPUT_BASENAME}_original.png"

# Initialize: Start with the input PNG
cp "$INPUT_IMAGE" "$TEMP_DIR/current.png"

echo "Running $CYCLE_100 cycles..."
echo ""

for i in $(seq 1 $CYCLE_100); do
    # PNG -> FP16 safetensors
    "$CRA" \
        -i "$TEMP_DIR/current.png" \
        --output-safetensors "$TEMP_DIR/current.safetensors" \
        --safetensors-format fp16 \
        --safetensors-transfer srgb

    # FP16 safetensors -> PNG (RGB888)
    "$CRA" \
        -i "$TEMP_DIR/current.safetensors" \
        --format RGB888 \
        -o "$TEMP_DIR/current.png"

    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Cycle $i complete"
    fi

    # Save at specified cycles
    if [ "$i" -eq "$CYCLE_1" ]; then
        cp "$TEMP_DIR/current.png" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${CYCLE_1}.png"
        cp "$TEMP_DIR/current.safetensors" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${CYCLE_1}.safetensors"
        echo "  -> Saved cycle $CYCLE_1 outputs"
    fi

    if [ "$i" -eq "$CYCLE_10" ]; then
        cp "$TEMP_DIR/current.png" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${CYCLE_10}.png"
        cp "$TEMP_DIR/current.safetensors" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${CYCLE_10}.safetensors"
        echo "  -> Saved cycle $CYCLE_10 outputs"
    fi

    if [ "$i" -eq "$CYCLE_100" ]; then
        cp "$TEMP_DIR/current.png" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${CYCLE_100}.png"
        cp "$TEMP_DIR/current.safetensors" "$OUTPUT_DIR/${INPUT_BASENAME}_cycle_${CYCLE_100}.safetensors"
        echo "  -> Saved cycle $CYCLE_100 outputs"
    fi
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
echo "  Original:  ${OUTPUT_DIR}/${INPUT_BASENAME}_original.png"
echo "  Cycle 1:   ${OUTPUT_DIR}/${INPUT_BASENAME}_cycle_${CYCLE_1}.png"
echo "  Cycle 10:  ${OUTPUT_DIR}/${INPUT_BASENAME}_cycle_${CYCLE_10}.png"
echo "  Cycle 100: ${OUTPUT_DIR}/${INPUT_BASENAME}_cycle_${CYCLE_100}.png"
echo ""
echo "Done!"
