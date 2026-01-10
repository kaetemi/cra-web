# CRA Web Development Progress

## Completed

### 1. Rust WASM Dither Module
- Created `dither/` Rust crate with `floyd_steinberg_dither` function
- Configured for WebAssembly compilation with `wasm-pack`
- Builds to ~13KB WASM binary
- Located at: `dither/src/lib.rs`

### 2. Python Script Modifications
All 5 Python scripts have been updated with conditional numba/WASM imports:
- `color_correction_basic.py` - LAB histogram matching
- `color_correction_basic_rgb.py` - RGB histogram matching
- `color_correction_cra.py` - CRA in LAB space
- `color_correction_cra_rgb.py` - CRA in RGB space
- `color_correction_tiled.py` - Tiled CRA processing

Each script now:
- Uses numba JIT when running natively
- Falls back to WASM via Pyodide's JS interop when numba is unavailable
- Maintains bit-perfect output between both implementations

### 3. Web Frontend
- `index.html` - Modern responsive UI with:
  - Drag-and-drop image upload
  - Method selection dropdown
  - Method descriptions
  - Side-by-side comparison view
  - Download button for results
  - Loading overlay during Pyodide initialization

- `app.js` - JavaScript application with:
  - Pyodide initialization with progress indicator
  - WASM module loading and global exposure
  - File handling and image preview
  - Python script execution
  - Error handling

### 4. Build System
- `build.sh` - Shell script that:
  - Builds WASM with `wasm-pack`
  - Creates `dist/` directory structure
  - Copies all required files

## Available Methods
1. **Lab** - Basic LAB histogram matching
2. **RGB** - Basic RGB histogram matching
3. **CRA Lab** - Chroma Rotation Averaging in LAB (recommended)
4. **CRA Lab Tiled** - Tiled processing with luminosity
5. **CRA Lab Tiled AB** - Tiled processing, original luminosity preserved
6. **CRA RGB** - CRA in RGB space
7. **CRA RGB Perceptual** - CRA RGB with perceptual weighting

## Project Structure
```
cra-web/
├── index.html              # Main webpage
├── app.js                  # JavaScript application
├── build.sh                # Build script
├── dither/                 # Rust WASM crate
│   ├── Cargo.toml
│   ├── src/lib.rs
│   └── pkg/                # wasm-pack output
├── scripts/                # Python color correction scripts
│   ├── color_correction_basic.py
│   ├── color_correction_basic_rgb.py
│   ├── color_correction_cra.py
│   ├── color_correction_cra_rgb.py
│   └── color_correction_tiled.py
└── dist/                   # Built files for deployment
    ├── index.html
    ├── app.js
    ├── wasm/
    │   ├── dither.js
    │   └── dither_bg.wasm
    └── scripts/
        └── *.py
```

## Running Locally
```bash
# Build the project
./build.sh

# Serve locally
cd dist && python -m http.server 8000

# Open in browser
# http://localhost:8000
```

## Technical Notes

### Pyodide Package Loading
The app uses CDN-hosted Pyodide (v0.27.5) and loads these packages:
- numpy
- opencv-python
- pillow
- scikit-image
- micropip

First load downloads ~20MB of packages (cached by browser afterward).

### WASM Integration
The Rust dither function is exposed globally as `window.floyd_steinberg_dither_wasm` before Pyodide loads, allowing Python to import it via `from js import floyd_steinberg_dither_wasm`.

### Data Flow
```
User uploads image
    ↓
JavaScript reads as Uint8Array
    ↓
Written to Pyodide virtual filesystem
    ↓
Python script processes (using WASM for dithering)
    ↓
Output read from virtual filesystem
    ↓
Displayed as blob URL in browser
```

## Next Steps (Optional Enhancements)
- [ ] Self-host Pyodide for offline capability <- DO THIS
- [ ] Add pyodide-pack tree shaking to reduce bundle size <- DO THIS
- [ ] Add Web Worker for non-blocking processing <- DO THIS
- [ ] Add more options (keep-luminosity toggle)
- [ ] Add batch processing support
- [ ] Add image size/format validation
