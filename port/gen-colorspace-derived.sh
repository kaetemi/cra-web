#!/bin/bash
# Regenerate colorspace_derived.rs from primary constants
#
# Run from the port/ directory:
#   ./gen-colorspace-derived.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building generator..."
cargo build --manifest-path tools/gen_colorspace_derived/Cargo.toml --release 2>/dev/null

echo "Generating src/colorspace_derived.rs..."
cargo run --manifest-path tools/gen_colorspace_derived/Cargo.toml --release 2>/dev/null > src/colorspace_derived.rs

echo "Running tests..."
cargo test colorspace --no-default-features 2>/dev/null

echo "Done."
