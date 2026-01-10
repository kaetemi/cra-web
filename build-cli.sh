#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/port"

echo "=========================================="
echo "Building CRA CLI Binaries"
echo "=========================================="

# Source cargo environment if exists
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Ensure musl target is installed
echo ""
echo "[1/4] Checking build targets..."
if ! rustup target list --installed | grep -q x86_64-unknown-linux-musl; then
    echo "Installing musl target..."
    rustup target add x86_64-unknown-linux-musl
fi
echo "Build targets ready."

# Build release binary (dynamic)
echo ""
echo "[2/4] Building release binary..."
cargo build --release --features cli
echo "Release build complete."

# Build static binary (musl)
echo ""
echo "[3/4] Building static binary (musl)..."
cargo build --release --features cli --target x86_64-unknown-linux-musl
echo "Static build complete."

# Create output directory and copy binaries
echo ""
echo "[4/4] Copying and stripping binaries..."
mkdir -p "$SCRIPT_DIR/dist/bin"

cp target/release/cra "$SCRIPT_DIR/dist/bin/cra-dynamic"
cp target/x86_64-unknown-linux-musl/release/cra "$SCRIPT_DIR/dist/bin/cra-static"

# Strip binaries to reduce size
strip "$SCRIPT_DIR/dist/bin/cra-dynamic"
strip "$SCRIPT_DIR/dist/bin/cra-static"

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
echo "  ./dist/bin/cra -i input.jpg -r ref.jpg -o output.jpg"
echo ""
echo "The 'cra' and 'cra-static' binaries are fully portable"
echo "(no dependencies, runs on any x86_64 Linux system)."
echo "=========================================="
