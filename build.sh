#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building CRA Web Application"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"

# Source cargo environment if exists
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Step 1: Build WASM
echo ""
echo "[1/4] Building WASM dither module..."
cd "$SCRIPT_DIR/dither"
wasm-pack build --target web --release
cd "$SCRIPT_DIR"
echo "WASM build complete."

# Step 2: Create dist directory
echo ""
echo "[2/4] Creating dist directory..."
rm -rf dist
mkdir -p dist/wasm dist/scripts dist/assets

# Step 3: Copy files
echo ""
echo "[3/4] Copying files..."

# Copy WASM files
cp dither/pkg/dither.js dist/wasm/
cp dither/pkg/dither_bg.wasm dist/wasm/

# Copy Python scripts
cp scripts/color_correction_basic.py dist/scripts/
cp scripts/color_correction_basic_rgb.py dist/scripts/
cp scripts/color_correction_cra.py dist/scripts/
cp scripts/color_correction_cra_rgb.py dist/scripts/
cp scripts/color_correction_tiled.py dist/scripts/

# Copy web files
cp index.html dist/
cp app.js dist/
cp worker.js dist/
cp sw.js dist/

# Copy default images
cp scripts/assets/retarget_input.jpg dist/assets/
cp scripts/assets/retarget_ref.jpg dist/assets/

echo "Files copied."

# Step 4: Print summary
echo ""
echo "[4/4] Build complete!"
echo ""
echo "=========================================="
echo "Build Summary"
echo "=========================================="
echo ""
echo "Output directory: dist/"
echo ""
echo "Files:"
ls -la dist/
echo ""
echo "WASM files:"
ls -la dist/wasm/
echo ""
echo "Python scripts:"
ls -la dist/scripts/
echo ""
echo "=========================================="
echo "To test locally:"
echo "  cd dist && python -m http.server 8000"
echo "  Then open http://localhost:8000"
echo "=========================================="
