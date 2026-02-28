#!/bin/bash
# Run dithering tests on all test images
# Generates L1 (1-bit) dithered output using all serpentine dither modes

set -e

CRA="/home/kaetemi/cra-web/port/target/release/cra"
INPUT_DIR="/home/kaetemi/cra-web/tools/test_images/sources"
OUTPUT_BASE="/home/kaetemi/cra-web/tools/test_images/dithered"

# All dither modes: standard and serpentine variants
DITHER_MODES=(
    "fs-standard"
    "fs-serpentine"
    "jjn-standard"
    "jjn-serpentine"
    "boon-standard"
    "boon-serpentine"
    "boon-wanghash"
    "boon-wanghash-serpentine"
    "boon-lowbias"
    "boon-lowbias-serpentine"
    "ostro-standard"
    "ostro-serpentine"
    "zhou-fang-standard"
    "zhou-fang-serpentine"
    "ulichney-standard"
    "ulichney-serpentine"
    "ulichney-weight-standard"
    "ulichney-weight-serpentine"
    "fs-tpdf-standard"
    "fs-tpdf-serpentine"
)

# Create output directories
for mode in "${DITHER_MODES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/${mode}"
done

# Get all input images
INPUT_FILES=$(find "${INPUT_DIR}" -maxdepth 1 -name "*.png" -type f | sort)

echo "CRA Dithering Test Suite"
echo "========================"
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

# Generate void-and-cluster blue noise dithered images (Python, not CRA)
echo ""
echo "Generating void-and-cluster dithered images..."
VENV="/home/kaetemi/venv/bin/activate"
if [ -f "$VENV" ]; then
    source "$VENV"
    mkdir -p "${OUTPUT_BASE}/void-and-cluster"
    python tools/generate_bluenoise_dither.py \
        --ref-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_BASE}/void-and-cluster" \
        --suffix "void-and-cluster"
else
    echo "  Warning: Python venv not found at $VENV, skipping void-and-cluster"
fi

echo ""
echo "Done! Output in ${OUTPUT_BASE}/"
echo ""

# Summary
echo "Output structure:"
for mode in "${DITHER_MODES[@]}"; do
    count=$(ls -1 "${OUTPUT_BASE}/${mode}"/*.png 2>/dev/null | wc -l)
    echo "  ${mode}/: ${count} files"
done
count=$(ls -1 "${OUTPUT_BASE}/void-and-cluster"/*.png 2>/dev/null | wc -l)
echo "  void-and-cluster/: ${count} files"
