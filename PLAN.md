# Client-Side Python Image Processing with WebAssembly

## Project Overview

### Current State

We have three Python scripts for color correction that rely on the following dependencies:

```
opencv-python
Pillow
scikit-image
blendmodes
numba
```

The scripts are:
- `color_correction_basic.py`
- `color_correction_cra.py`
- `color_correction_tiled.py`

Each script contains a performance-critical `floyd_steinberg_dither` function decorated with `@numba.jit(nopython=True)`. Without Numba's JIT compilation, the dithering is unacceptably slow. The output must be bit-perfect.

### Goal

Run these scripts entirely client-side in the browser with no server backend, while maintaining:
- Bit-perfect output matching the native Python+Numba version
- Acceptable performance
- Ability to run the same code natively (for development/testing)

### Problem

Numba cannot run in WebAssembly. It works by JIT-compiling Python to native machine code via LLVM at runtime—something the browser sandbox doesn't allow.

### Solution

1. Run Python in the browser via **Pyodide** (CPython compiled to WebAssembly)
2. Replace the Numba-accelerated dither function with a **Rust → WebAssembly** implementation
3. Use conditional imports so the same Python code runs natively with Numba or in-browser with WASM

---

## Dependency Analysis

| Package | Pyodide Support | Notes |
|---------|-----------------|-------|
| numpy | ✅ Built-in | |
| opencv-python | ✅ Built-in | v4.10.0.84 |
| Pillow | ✅ Built-in | Supports WEBP, TIFF |
| scikit-image | ✅ Built-in | Pulls in scipy, large |
| blendmodes | ✅ Pure Python | Install via micropip |
| numba | ❌ Not supported | Replace with Rust WASM |

---

## Implementation

### Part 1: Rust WASM Dither Function

#### Algorithm

The original Numba implementation uses 2D indexing with boundary issues. We use a cleaner approach: treat the image as a linear buffer and pad the end to absorb overflow writes from error diffusion.

Floyd-Steinberg distributes quantization error to neighboring pixels:
```
        current  → +7/16
           ↓
  +3/16 ← +5/16 → +1/16
```

For a pixel at linear index `i` in an image of width `w`, the neighbors are:
- Right: `i + 1`
- Bottom-left: `i + w - 1`
- Bottom: `i + w`
- Bottom-right: `i + w + 1`

At the right edge and bottom row, some of these indices overflow. Rather than branching on every pixel, we allocate `w + 2` extra floats at the end of the buffer. Overflow writes land there harmlessly and get discarded.

#### Project Setup

```bash
cargo new --lib dither
cd dither
```

#### Cargo.toml

```toml
[package]
name = "dither"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
```

#### src/lib.rs

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn floyd_steinberg_dither(img: Vec<f32>, w: usize, h: usize) -> Vec<u8> {
    let len = w * h;
    
    // Allocate buffer with overflow padding
    let mut buf = vec![0.0f32; len + w + 2];
    buf[..len].copy_from_slice(&img);

    for i in 0..len {
        let old = buf[i];
        let new = old.round();
        buf[i] = new;
        let err = old - new;

        // Distribute error to neighbors
        // Overflow writes hit padding, which we discard
        buf[i + 1] += err * (7.0 / 16.0);
        buf[i + w - 1] += err * (3.0 / 16.0);
        buf[i + w] += err * (5.0 / 16.0);
        buf[i + w + 1] += err * (1.0 / 16.0);
    }

    // Clamp and convert to u8, discard padding
    buf[..len]
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect()
}
```

#### Build

```bash
# Install wasm-pack if needed
cargo install wasm-pack

# Build for browser
wasm-pack build --target web --release
```

Output in `pkg/`:
- `dither.js` — JS bindings
- `dither_bg.wasm` — Compiled WebAssembly (~20KB)

---

### Part 2: Python Implementation

The Python code must work in two environments:
1. **Native:** Use Numba for JIT compilation
2. **Browser (Pyodide):** Call the Rust WASM function via JS interop

In each of the three scripts, replace the existing numba import and dither function with this block:

```python
import numpy as np

try:
    import numba

    @numba.jit(nopython=True)
    def floyd_steinberg_dither(img):
        """
        Floyd-Steinberg dithering with linear buffer and overflow padding.
        Matches the Rust WASM implementation exactly for bit-perfect output.
        
        Args:
            img: 2D numpy array of float32, values in range [0, 255]
        
        Returns:
            2D numpy array of uint8, same shape as input
        """
        h, w = img.shape
        len_ = h * w
        
        # Linear buffer with overflow padding
        buf = np.zeros(len_ + w + 2, dtype=np.float32)
        buf[:len_] = img.ravel()

        for i in range(len_):
            old = buf[i]
            new = np.round(old)
            buf[i] = new
            err = old - new

            buf[i + 1] += err * (7.0 / 16.0)
            buf[i + w - 1] += err * (3.0 / 16.0)
            buf[i + w] += err * (5.0 / 16.0)
            buf[i + w + 1] += err * (1.0 / 16.0)

        return np.clip(buf[:len_], 0, 255).astype(np.uint8).reshape(h, w)

except ImportError:
    # Running in Pyodide — use WASM implementation
    from js import floyd_steinberg_dither_wasm
    from pyodide.ffi import to_js

    def floyd_steinberg_dither(img):
        """
        Floyd-Steinberg dithering via Rust WASM.
        Called when running in Pyodide where Numba is unavailable.
        """
        h, w = img.shape
        flat = img.ravel().astype(np.float32)
        
        # Call Rust WASM function via JS interop
        result = floyd_steinberg_dither_wasm(to_js(flat.tolist()), w, h)
        
        return np.array(result.to_py(), dtype=np.uint8).reshape(h, w)
```

The rest of each script remains unchanged. This block goes near the top, after the other imports.

---

### Part 3: JS ↔ Python ↔ WASM Bridge

#### Data Flow

```
Python (Pyodide)
    │
    │  img.ravel().astype(np.float32)
    │  to_js(flat.tolist())
    ▼
JavaScript
    │
    │  Float32Array passed to WASM
    ▼
Rust WASM
    │
    │  Vec<f32> → processing → Vec<u8>
    ▼
JavaScript
    │
    │  Uint8Array returned
    ▼
Python (Pyodide)
    │  np.array(result.to_py(), dtype=np.uint8)
    │  .reshape(h, w)
    ▼
numpy array
```

#### Exposing WASM to Python

Before running any Python code, the WASM function must be attached to the global JS scope so Pyodide can access it via `from js import ...`:

```javascript
import init, { floyd_steinberg_dither } from './wasm/dither.js';

// Initialize WASM module
await init();

// Expose to global scope for Pyodide
window.floyd_steinberg_dither_wasm = floyd_steinberg_dither;
```

After this, Python code running in Pyodide can do:

```python
from js import floyd_steinberg_dither_wasm
```

---

### Part 4: Pyodide Setup

#### Loading Packages

```javascript
const pyodide = await loadPyodide({
    indexURL: "/static/pyodide/"  // Self-hosted
});

// Load built-in packages
await pyodide.loadPackage([
    "numpy",
    "opencv-python",
    "pillow",
    "scikit-image",
    "micropip"
]);

// Load pure Python package from wheel
await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("/static/wheels/blendmodes-2025-py3-none-any.whl")
`);
```

#### Loading Python Scripts

```javascript
// Fetch and execute a script
const script = await fetch("/static/scripts/color_correction_basic.py").then(r => r.text());
await pyodide.runPythonAsync(script);

// Call the main function
await pyodide.runPythonAsync(`
    main("/input.png", "/ref.png", "/output.png")
`);
```

#### File System

Pyodide provides a virtual filesystem. Load images into it before processing:

```javascript
const imageBytes = new Uint8Array(await file.arrayBuffer());
pyodide.FS.writeFile("/input.png", imageBytes);
```

Read results out:

```javascript
const outputBytes = pyodide.FS.readFile("/output.png");
```

---

### Part 5: Self-Hosting Pyodide

Download a Pyodide release and host the files yourself for offline capability and faster loads.

#### Required Files

```
pyodide-bundle/
├── pyodide.js
├── pyodide.asm.js
├── pyodide.asm.wasm
├── python_stdlib.zip
├── packages.json
├── numpy-*.whl
├── opencv_python-*.whl
├── pillow-*.whl
├── scikit_image-*.whl
├── scipy-*.whl
├── micropip-*.whl
└── ... (transitive dependencies)
```

#### Tree Shaking with pyodide-pack

Reduce bundle size by removing unused modules:

```bash
pip install pyodide-pack

# Create test file that exercises all imports
cat > test_imports.py << 'EOF'
import numpy as np
from PIL import Image
import cv2
from skimage import exposure
img = np.zeros((10, 10, 3), dtype=np.float32)
cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
exposure.match_histograms(img, img, channel_axis=2)
EOF

# Generate trimmed bundle
pyodide-pack test_imports.py \
    --packages numpy opencv-python pillow scikit-image micropip \
    --output dist/pyodide-bundle/
```

---

### Part 6: Final Project Structure

```
project/
├── dither/                          # Rust crate
│   ├── Cargo.toml
│   ├── src/
│   │   └── lib.rs
│   └── pkg/                         # wasm-pack output
│       ├── dither.js
│       └── dither_bg.wasm
│
├── scripts/
│   ├── color_correction_basic.py    # Self-contained with dither
│   ├── color_correction_cra.py      # Self-contained with dither
│   └── color_correction_tiled.py    # Self-contained with dither
│
├── dist/                            # Deployment build
│   ├── pyodide-bundle/              # Tree-shaken Pyodide
│   ├── wasm/
│   │   ├── dither.js
│   │   └── dither_bg.wasm
│   ├── wheels/
│   │   └── blendmodes-2025-py3-none-any.whl
│   └── scripts/
│       ├── color_correction_basic.py
│       ├── color_correction_cra.py
│       └── color_correction_tiled.py
│
├── test_imports.py                  # For pyodide-pack
└── build.sh                         # Build script
```

---

### Part 7: Build Script

```bash
#!/bin/bash
set -e

echo "Building WASM..."
cd dither
wasm-pack build --target web --release
cd ..

echo "Creating dist..."
rm -rf dist
mkdir -p dist/wasm dist/wheels dist/scripts

echo "Copying WASM..."
cp dither/pkg/dither.js dist/wasm/
cp dither/pkg/dither_bg.wasm dist/wasm/

echo "Running pyodide-pack..."
pyodide-pack test_imports.py \
    --packages numpy opencv-python pillow scikit-image micropip \
    --output dist/pyodide-bundle/

echo "Copying wheels..."
cp wheels/blendmodes-*.whl dist/wheels/

echo "Copying scripts..."
cp scripts/*.py dist/scripts/

echo "Done. Serve dist/ to test."
```

---

### Part 8: Testing

#### Native (with Numba)

```bash
pip install numpy opencv-python pillow scikit-image blendmodes numba

python scripts/color_correction_basic.py \
    --input test_input.png \
    --ref test_ref.png \
    --output output_native.png
```

#### Browser (with Pyodide + WASM)

1. Build: `./build.sh`
2. Serve: `cd dist && python -m http.server 8000`
3. Open `http://localhost:8000` in browser
4. Compare `output_browser.png` to `output_native.png` — should be identical

---

### Estimated Bundle Sizes

| Component | Size |
|-----------|------|
| Pyodide core | ~6 MB |
| NumPy | ~4 MB |
| OpenCV | ~5 MB |
| scikit-image + scipy | ~8 MB |
| Pillow | ~500 KB |
| blendmodes | ~50 KB |
| Rust WASM dither | ~20 KB |
| **Total (before tree-shaking)** | **~24 MB** |
| **Total (after tree-shaking)** | **~15-20 MB** |

First load downloads everything; subsequent loads use browser cache.