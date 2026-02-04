#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "CRA Development Environment Setup"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"

# =============================================================================
# Step 1: System packages (requires sudo)
# =============================================================================
echo ""
echo "[1/4] Checking system packages..."

PACKAGES_NEEDED=()

if ! dpkg -s python3-venv >/dev/null 2>&1; then
    PACKAGES_NEEDED+=(python3-venv)
fi

if ! dpkg -s musl-tools >/dev/null 2>&1; then
    PACKAGES_NEEDED+=(musl-tools)
fi

if ! dpkg -s mingw-w64 >/dev/null 2>&1; then
    PACKAGES_NEEDED+=(mingw-w64)
fi

if [ ${#PACKAGES_NEEDED[@]} -gt 0 ]; then
    echo "Installing: ${PACKAGES_NEEDED[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${PACKAGES_NEEDED[@]}"
else
    echo "All system packages already installed."
fi

# =============================================================================
# Step 2: Python virtual environment
# =============================================================================
echo ""
echo "[2/4] Setting up Python virtual environment..."

VENV_DIR="$HOME/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR."
fi

source "$VENV_DIR/bin/activate"

echo "Installing Python dependencies..."
pip install --quiet numpy pillow matplotlib numba scipy pypng

echo "Python environment ready."

# =============================================================================
# Step 3: Rust toolchain
# =============================================================================
echo ""
echo "[3/4] Setting up Rust toolchain..."

if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if command -v rustup >/dev/null 2>&1; then
    echo "Rust already installed: $(rustc --version)"
else
    echo "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "Installed: $(rustc --version)"
fi

# Add required targets
for TARGET in wasm32-unknown-unknown x86_64-unknown-linux-musl x86_64-pc-windows-gnu; do
    if ! rustup target list --installed | grep -q "^${TARGET}$"; then
        echo "Adding target: $TARGET"
        rustup target add "$TARGET"
    fi
done

echo "Rust targets ready."

# =============================================================================
# Step 4: wasm-pack
# =============================================================================
echo ""
echo "[4/4] Checking wasm-pack..."

if command -v wasm-pack >/dev/null 2>&1; then
    echo "wasm-pack already installed: $(wasm-pack --version)"
else
    echo "Installing wasm-pack..."
    cargo install wasm-pack
    echo "Installed: $(wasm-pack --version)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
echo ""
echo "Rust:      $(rustc --version)"
echo "Cargo:     $(cargo --version)"
echo "wasm-pack: $(wasm-pack --version)"
echo "Python:    $(python3 --version)"
echo "Venv:      $VENV_DIR"
echo ""
echo "Targets:"
rustup target list --installed | sed 's/^/  /'
echo ""
echo "Python packages:"
pip list --format=columns 2>/dev/null | grep -E '(numpy|Pillow|matplotlib|numba|scipy|pypng)' | sed 's/^/  /'
echo ""
echo "Next steps:"
echo "  ./build.sh          # Build web demo (WASM)"
echo "  ./build-cli.sh      # Build CLI binary"
echo "=========================================="
