#!/bin/bash
# Run dithering tests on extra test images (for blog visuals)
# Generates L1 (1-bit) dithered output using selected dither modes

set -e

CRA="/home/kaetemi/cra-web/port/target/release/cra"
INPUT_DIR="/home/kaetemi/cra-web/tools/test_extra/sources"
OUTPUT_BASE="/home/kaetemi/cra-web/tools/test_extra/dithered"

# Dither modes for blog comparison
DITHER_MODES=(
    "fs-standard"
    "fs-serpentine"
    "jjn-standard"
    "jjn-serpentine"
    "boon-standard"
    "boon-serpentine"
    "boon-wanghash"
    "boon-wanghash-lowbit"
    "boon-lowbias"
    "boon-h2"
    "ostro-standard"
    "ostro-serpentine"
    "zhou-fang-standard"
    "zhou-fang-serpentine"
    "fs-tpdf-standard"
    "fs-tpdf-serpentine"
)

# Create output directories
for mode in "${DITHER_MODES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/${mode}"
done

# Get all input images
INPUT_FILES=$(find "${INPUT_DIR}" -maxdepth 1 -name "*.png" -type f | sort)

echo "CRA Extra Dithering Tests (blog visuals)"
echo "========================================="
echo "Format: L1 (1-bit grayscale)"
echo "Modes: ${DITHER_MODES[*]}"
echo ""

total_files=$(echo "$INPUT_FILES" | wc -l)
total_modes=${#DITHER_MODES[@]}
total_ops=$((total_files * total_modes))
current=0

for input_file in $INPUT_FILES; do
    filename=$(basename "$input_file")
    basename="${filename%.png}"

    echo "Processing: ${filename}"

    for mode in "${DITHER_MODES[@]}"; do
        output_file="${OUTPUT_BASE}/${mode}/${basename}_${mode}.png"

        "$CRA" \
            -i "$input_file" \
            -o "$output_file" \
            --format L1 \
            --output-dither "$mode" \
            --no-colorspace-aware-output \
            2>/dev/null

        current=$((current + 1))
    done

    echo "  -> ${total_modes} modes completed"
done

echo ""
echo "Done! Output in ${OUTPUT_BASE}/"
echo ""

# Summary
echo "Output structure:"
for mode in "${DITHER_MODES[@]}"; do
    count=$(ls -1 "${OUTPUT_BASE}/${mode}"/*.png 2>/dev/null | wc -l)
    echo "  ${mode}/: ${count} files"
done
