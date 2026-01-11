# CRA Web Demo

An interactive web demo for **Chroma Rotation Averaging (CRA)** color correction techniques, featuring a complete Rust port of the original Python implementation with additional features.

Based on the original [chroma-rotation-averaging](https://github.com/kaetemi/chroma-rotation-averaging) repository. See the [scripts/README.md](scripts/README.md) for a detailed explanation of the CRA technique and the problem it solves.

## Overview

This project provides:

1. **Web Demo**: A client-side web application for experimenting with CRA color correction in the browser
2. **Rust/WASM Port**: Complete port of the CRA algorithms to Rust, compiled to WebAssembly
3. **CLI Tool**: A standalone command-line tool for batch processing

## Rust Port Features

The Rust port in `port/` includes all original Python functionality plus several enhancements:

### Color Spaces
- **LAB**: Original CIE LAB implementation (basic, CRA, tiled)
- **RGB**: Linear RGB with optional perceptual weighting
- **Oklab**: Perceptually uniform color space with better hue linearity than LAB (basic, CRA, tiled)

### F32 Histogram Matching
Optional sort-based histogram matching using full 32-bit floating-point precision. Eliminates quantization artifacts from the traditional 256-bin histogram approach.

### Mixed Error Diffusion Dithering
A novel per-pixel kernel switching technique that randomly selects between Floyd-Steinberg and Jarvis-Judice-Ninke dithering kernels for each pixel using Wang hash. This breaks up the regular patterns that can appear with a single dithering kernel.

Available dither modes:
- **Floyd-Steinberg Standard/Serpentine**: Classic 4-pixel error diffusion
- **Jarvis-Judice-Ninke Standard/Serpentine**: Larger 12-pixel kernel for smoother gradients
- **Mixed Standard/Serpentine/Random**: Per-pixel kernel selection with configurable scan direction

### Configurable Bit Depth
Support for 1-8 bit output with proper bit replication (e.g., 3-bit value ABC becomes ABCABCAB in 8-bit output).

## Building the Web Demo

### Prerequisites

- Rust toolchain with `wasm32-unknown-unknown` target
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

```bash
# Install wasm-pack if needed
cargo install wasm-pack

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Build

```bash
./build.sh
```

This will:
1. Build the WASM modules from `dither/` and `port/`
2. Create the `dist/` directory with all required files
3. Copy the web application files

### Run Locally

```bash
cd dist && python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Deployment

The `dist/` directory contains all files needed for deployment. Upload to any static hosting service (GitHub Pages, Netlify, Vercel, etc.).

## Building the CLI Tool

### Prerequisites

- Rust toolchain
- For static builds: musl target (`x86_64-unknown-linux-musl`)

### Build

```bash
./build-cli.sh
```

This produces:
- `dist/bin/cra` - Fully static binary (portable to any x86_64 Linux)
- `dist/bin/cra-dynamic` - Dynamically linked binary
- `dist/bin/cra-static` - Same as `cra`, explicit name

### Usage

```bash
./dist/bin/cra -i input.jpg -r reference.jpg -o output.jpg
```

### CLI Options

```
Options:
  -i, --input <INPUT>              Input image path
  -r, --ref <REF>                  Reference image path
  -o, --output <OUTPUT>            Output image path
  -m, --method <METHOD>            Color correction method [default: cra-lab]
                                   [values: basic-lab, basic-rgb, basic-oklab,
                                    cra-lab, cra-rgb, cra-oklab, tiled-lab, tiled-oklab]
      --keep-luminosity            Preserve original L channel
      --tiled-luminosity           Process L channel per-tile before global match
  -p, --perceptual                 Use perceptual weighting (cra-rgb only)
      --f32-histogram              Use f32 sort-based histogram matching
      --output-dither <DITHER>     Output dithering method [default: jjn-standard]
      --histogram-dither <DITHER>  Histogram processing dither [default: mixed-standard]
  -v, --verbose                    Enable verbose output
  -h, --help                       Print help
  -V, --version                    Print version
```

### Examples

```bash
# CRA LAB (default, recommended)
./dist/bin/cra -i input.jpg -r reference.jpg -o output.jpg

# CRA Oklab with f32 histogram (highest quality)
./dist/bin/cra -i input.jpg -r reference.jpg -o output.jpg -m cra-oklab --f32-histogram

# Tiled LAB for images with varying color casts
./dist/bin/cra -i input.jpg -r reference.jpg -o output.jpg -m tiled-lab

# Preserve original luminosity
./dist/bin/cra -i input.jpg -r reference.jpg -o output.jpg --keep-luminosity
```

## Project Structure

```
cra-web/
├── dither/              # Floyd-Steinberg dither WASM module
├── port/                # Main Rust port of CRA algorithms
│   ├── src/
│   │   ├── lib.rs       # WASM exports
│   │   ├── bin/cra.rs   # CLI tool
│   │   ├── basic_*.rs   # Basic histogram matching
│   │   ├── cra_*.rs     # CRA implementations
│   │   ├── tiled_*.rs   # Tiled processing
│   │   ├── dither.rs    # Error diffusion dithering
│   │   ├── histogram.rs # Histogram matching algorithms
│   │   └── color.rs     # Color space conversions
│   └── Cargo.toml
├── scripts/             # Original Python scripts (submodule)
├── index.html           # Web demo
├── build.sh             # Web build script
├── build-cli.sh         # CLI build script
└── dist/                # Build output
```

## Acknowledgements

Web demo and Rust port built with Claude Opus 4.5 using [Claude Code](https://claude.ai/claude-code).

## License

See [scripts/LICENSE.md](scripts/LICENSE.md) for the original Python implementation license.
