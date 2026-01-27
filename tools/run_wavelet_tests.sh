#!/bin/bash
#
# Run comprehensive wavelet-based dithering quality analysis
#
# Generates dithered versions of all reference images with all methods,
# then runs wavelet analysis to detect artifacts.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CRA="${PROJECT_DIR}/port/target/release/cra"

# Directories
REF_DIR="${SCRIPT_DIR}/test_wavelets/reference_images"
DITHERED_DIR="${SCRIPT_DIR}/test_wavelets/dithered"
ANALYSIS_DIR="${SCRIPT_DIR}/test_wavelets/analysis"

# Create output directories
mkdir -p "$DITHERED_DIR"
mkdir -p "$ANALYSIS_DIR"

# Check CRA binary
if [ ! -f "$CRA" ]; then
    echo "Building CRA..."
    cd "${PROJECT_DIR}/port"
    cargo build --release --bin cra --features cli
    cd "$SCRIPT_DIR"
fi

# Dithering methods to test (serpentine variants for quality)
# Note: ulichney and fs-tpdf are 1-bit only, which is what we're testing
METHODS=(
    "fs-serpentine"
    "jjn-serpentine"
    "boon-serpentine"
    "ostro-serpentine"
    "zhou-fang-serpentine"
    "ulichney-serpentine"
    "ulichney-weight-serpentine"
    "fs-tpdf-serpentine"
    "none"
)

# Also test standard (non-serpentine) for comparison
METHODS_STANDARD=(
    "fs-standard"
    "jjn-standard"
    "boon-standard"
)

echo "=============================================="
echo "Wavelet Dithering Quality Analysis"
echo "=============================================="
echo ""
echo "Reference images: $REF_DIR"
echo "Output: $DITHERED_DIR"
echo "Analysis: $ANALYSIS_DIR"
echo ""

# Get list of reference images
IMAGES=$(ls "$REF_DIR"/*.png 2>/dev/null | xargs -n1 basename | sed 's/.png$//')

if [ -z "$IMAGES" ]; then
    echo "ERROR: No reference images found in $REF_DIR"
    exit 1
fi

echo "Found images:"
for img in $IMAGES; do
    echo "  - $img"
done
echo ""

# Generate dithered images
echo "=============================================="
echo "Generating dithered images (L1 format)"
echo "=============================================="

for method in "${METHODS[@]}" "${METHODS_STANDARD[@]}"; do
    method_dir="${DITHERED_DIR}/${method}"
    mkdir -p "$method_dir"

    for img in $IMAGES; do
        input="${REF_DIR}/${img}.png"
        output="${method_dir}/${img}.png"

        if [ -f "$output" ]; then
            echo "  [skip] $method/$img.png (exists)"
            continue
        fi

        echo "  [gen]  $method/$img.png"
        # Use --no-colorspace-aware-output for L1 (grayscale) since colorspace-aware only applies to RGB
        "$CRA" -i "$input" -o "$output" -f L1 --output-dither "$method" --no-colorspace-aware-output
    done
done

echo ""
echo "=============================================="
echo "Generating Python-based dithered images"
echo "=============================================="

# Activate Python environment
source /root/venv/bin/activate

# Generate white noise dithered images (random threshold, no error diffusion)
echo "  Generating whitenoise dithered images..."
python "${SCRIPT_DIR}/generate_whitenoise_dither.py" \
    --ref-dir "$REF_DIR" \
    --output-dir "${DITHERED_DIR}/whitenoise"

# Generate blue noise dithered images (void-and-cluster threshold array)
echo "  Generating bluenoise dithered images..."
python "${SCRIPT_DIR}/generate_bluenoise_dither.py" \
    --ref-dir "$REF_DIR" \
    --output-dir "${DITHERED_DIR}/bluenoise"

echo ""
echo "=============================================="
echo "Running wavelet analysis"
echo "=============================================="

# Run the wavelet analysis
python "${SCRIPT_DIR}/analyze_wavelet.py" \
    --compare \
    --ref-dir "$REF_DIR" \
    --dithered-dir "$DITHERED_DIR" \
    --output-dir "$ANALYSIS_DIR"

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "Results saved to: $ANALYSIS_DIR"
echo ""
