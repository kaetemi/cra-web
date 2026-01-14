#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/port"

echo "=========================================="
echo "Building CRA CLI Binary"
echo "=========================================="

# Source cargo environment if exists
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Ensure build targets are installed
echo ""
echo "[1/5] Checking build targets..."
if ! rustup target list --installed | grep -q x86_64-unknown-linux-musl; then
    echo "Installing musl target..."
    rustup target add x86_64-unknown-linux-musl
fi
if ! rustup target list --installed | grep -q x86_64-pc-windows-gnu; then
    echo "Installing Windows GNU target..."
    rustup target add x86_64-pc-windows-gnu
fi
echo "Build targets ready."

# Build release binary (dynamic)
echo ""
echo "[2/5] Building release binary..."
cargo build --release --features cli
echo "Release build complete."

# Build static binary (musl)
echo ""
echo "[3/5] Building static binary (musl)..."
cargo build --release --features cli --target x86_64-unknown-linux-musl
echo "Static build complete."

# Build Windows binary
echo ""
echo "[4/5] Building Windows binary..."
cargo build --release --features cli --target x86_64-pc-windows-gnu
echo "Windows build complete."

# Create output directory and copy binaries
echo ""
echo "[5/5] Copying and stripping binaries..."
mkdir -p "$SCRIPT_DIR/dist/bin"

# Copy cra binaries
cp target/release/cra "$SCRIPT_DIR/dist/bin/cra-dynamic"
cp target/x86_64-unknown-linux-musl/release/cra "$SCRIPT_DIR/dist/bin/cra-static"
cp target/x86_64-pc-windows-gnu/release/cra.exe "$SCRIPT_DIR/dist/bin/cra.exe"

# Strip binaries to reduce size
strip "$SCRIPT_DIR/dist/bin/cra-dynamic"
strip "$SCRIPT_DIR/dist/bin/cra-static"
x86_64-w64-mingw32-strip "$SCRIPT_DIR/dist/bin/cra.exe"

# Also create a convenience symlink/copy for the static one as the default
cp "$SCRIPT_DIR/dist/bin/cra-static" "$SCRIPT_DIR/dist/bin/cra"

echo ""
echo "=========================================="
echo "Build Summary"
echo "=========================================="
echo ""
echo "Binaries in dist/bin/:"
ls -lh "$SCRIPT_DIR/dist/bin/"
echo ""
echo "Usage:"
echo "  ./dist/bin/cra --help"
echo ""
echo "Color correction with dithering:"
echo "  ./dist/bin/cra -i input.jpg -r ref.jpg -o output.png"
echo ""
echo "Dither only (no color correction):"
echo "  ./dist/bin/cra -i input.png -o output.png --format RGB565"
echo ""
echo "Raw binary output for embedded systems:"
echo "  ./dist/bin/cra -i input.png -o output.png --format L4 --output-raw output.raw"
echo ""
echo "The 'cra' and 'cra-static' binaries are fully portable"
echo "(no dependencies, runs on any x86_64 Linux system)."
echo ""
echo "The '.exe' binary runs on Windows x86_64."
echo "=========================================="
