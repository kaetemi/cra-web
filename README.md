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

### Color Profile Support
- **ICC Profiles**: Conversion of embedded ICC color profiles via moxcms
- **CICP Metadata**: Support for CICP (Coding-Independent Code Points) color metadata
- **Auto-detection**: CICP checked first for sRGB/linear detection (authoritative), then ICC for conversion, CICP conversion as fallback if no ICC present

### F32 Histogram Matching
Optional sort-based histogram matching using full 32-bit floating-point precision. Eliminates quantization artifacts from the traditional 256-bin histogram approach. Two alignment modes: endpoint-aligned (preserves extremes) and midpoint-aligned (statistically correct).

### Error Diffusion Dithering
Multiple dithering algorithms with standard and serpentine scanning variants:

- **Floyd-Steinberg**: Classic 4-pixel error diffusion kernel
- **Jarvis-Judice-Ninke**: Larger 12-pixel kernel for smoother gradients
- **Our Method**: Novel per-pixel kernel switching between FS and JJN using lowbias32 hash — breaks up regular patterns, produces blue noise characteristics
- **Our Method (2nd Order Kernels)**: Precomputed second-order convolution kernels (FS² and JJN²) with wider reach and negative weights for steeper noise shaping (~8.2 dB/oct vs ~6.8 dB/oct)
- **Floyd-Steinberg TPDF**: Floyd-Steinberg with triangular PDF threshold dither (1-bit only)

### Colorspace-Aware Dithering
Joint RGB quantization using perceptual distance metrics to select the best candidate color, with error diffusion in linear RGB space. Unlike per-channel dithering, this processes all channels together to minimize perceived color error:
- **Distance metric**: OKLab Lr (default), OKLab, CIELAB (CIE76/CIE94/CIEDE2000), Linear RGB, Y'CbCr (BT.709/BT.601), or sRGB
- **Error accumulation**: Linear RGB (physically correct blending)
- **Overshoot penalty**: Penalizes quantization choices whose error would push neighboring pixels outside [0,1] RGB (reduces color fringing)
- **Candidate search**: Evaluates nearby quantization levels jointly to find perceptually closest match

### Configurable Bit Depth
Support for 1-8 bit output with proper bit replication (e.g., 3-bit value ABC becomes ABCABCAB in 8-bit output).

### Palette Dithering
Dither to arbitrary palettes loaded from PNG files:
- **Hull tracing**: Projects colors onto the palette's convex hull for better color relationships
- **Hull error decay**: Configurable error decay when palette colors are sparse near the gamut boundary
- Supports paletted PNG output and GIF output

### Output Formats
- **PNG**: Standard lossless image output (grayscale, RGB, or RGBA); palettized for ≤8bpp formats
- **GIF**: Paletted output with 1-bit transparency support
- **RGB**: RGB332, RGB565, RGB8, etc. (4-24 bits per pixel)
- **ARGB**: ARGB1555, ARGB4, ARGB8, etc. (with alpha channel)
- **Grayscale**: L1, L2, L4, L8 (1-8 bits per pixel)
- **Grayscale+Alpha**: LA1, LA2, LA4, LA8, LA48, etc. (luminosity with alpha)
- **Safetensors**: FP32, FP16, BF16 floating-point tensor format
- **Raw Binary**: Packed or row-aligned with configurable stride (1-128 bytes)

### Input Formats
- Standard image formats: PNG, JPEG, WebP, TIFF, BMP, GIF
- HDR formats: OpenEXR (with premultiplied alpha handling)
- Tensor formats: Safetensors (.safetensors, .safetensors.gz)
- Raw binary: With JSON metadata for format specification

### Alpha Channel Support
- Transparent images preserved through the pipeline (when using ARGB output formats)
- Alpha-aware rescaling prevents color bleeding from transparent pixels
- Automatic premultiplied alpha detection and un-premultiplication for EXR files
- Separate alpha channel output for embedded systems

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
Input/Output:
  -i, --input <PATH>               Input image path (PNG, JPEG, WebP, TIFF, EXR, safetensors)
      --input-metadata <JSON>      Metadata for raw binary input (format, width, height, stride)
      --input-profile <MODE>       Input color profile [default: auto]
                                   [values: srgb, linear, auto, icc, cicp]
      --input-premultiplied-alpha  Premultiplied alpha handling [default: auto]
                                   [values: auto, yes, no]
  -r, --ref <PATH>                 Reference image path (for histogram matching)
      --ref-profile <MODE>         Reference color profile [default: auto]
  -o, --output <PATH>              Output PNG image path
      --output-gif <PATH>          Output GIF image path (paletted formats only, 1-bit transparency)
      --no-palettized-output       Disable palettized PNG for ≤8bpp formats
      --output-raw <PATH>          Output raw binary file
      --output-raw-r <PATH>        Output red channel only (raw binary)
      --output-raw-g <PATH>        Output green channel only (raw binary)
      --output-raw-b <PATH>        Output blue channel only (raw binary)
      --output-raw-a <PATH>        Output alpha channel only (requires ARGB format)
      --output-raw-palette <PATH>  Output raw palette file (ARGB8888 binary, up to 256 colors)
      --output-meta <PATH>         Output metadata JSON file
      --output-safetensors <PATH>  Output safetensors file (.safetensors)

Format:
  -f, --format <FORMAT>            Output format [default: ARGB8888 if input has alpha, RGB888 otherwise]
                                   RGB: RGB8, RGB332, RGB565, RGB888
                                   ARGB: ARGB8, ARGB4, ARGB1555, ARGB4444, ARGB8888
                                   Grayscale: L1, L2, L4, L8
                                   Grayscale+Alpha: LA1, LA2, LA4, LA8, LA48
      --stride <N>                 Row stride alignment in bytes (power of 2, 1-128) [default: 1]
      --stride-fill <MODE>         How to fill stride padding [default: black]
                                   [values: black, repeat]

Histogram Matching:
      --histogram <METHOD>         Histogram matching method [default: cra-oklab if --ref provided]
                                   [values: none, basic-lab, basic-rgb, basic-oklab,
                                    cra-lab, cra-rgb, cra-oklab, tiled-lab, tiled-oklab]
      --histogram-mode <MODE>      Histogram matching mode [default: binned]
                                   [values: binned, f32-endpoint, f32-midpoint]
      --keep-luminosity            Preserve original luminosity (basic-lab, basic-oklab,
                                   cra-lab, cra-oklab)
      --tiled-luminosity           Process L channel per-tile (tiled-lab, tiled-oklab)
      --perceptual                 Use perceptual weighting (cra-rgb)

Dithering:
      --output-dither <MODE>       Output dithering method [default: boon-standard]
                                   [values: fs-standard, fs-serpentine, jjn-standard,
                                    jjn-serpentine, boon-standard, boon-serpentine,
                                    fs-tpdf-standard, fs-tpdf-serpentine, boon-h2, none]
      --output-alpha-dither <MODE> Alpha channel dithering method [default: same as --output-dither]
      --output-distance-space      Perceptual space for output dithering
                                   [default: oklab-lr for RGB, lab-cie94 for grayscale]
                                   [values: oklab, oklab-lr, lab-cie76, lab-cie94, lab-ciede2000,
                                    linear-rgb, y-cb-cr, srgb, y-cb-cr-bt601]
      --no-colorspace-aware-output Disable colorspace-aware dithering (use per-channel)
      --no-overshoot-penalty       Disable overshoot penalty for colorspace-aware dithering
      --histogram-dither <MODE>    Dithering for histogram quantization [default: boon-standard]
      --histogram-distance-space   Perceptual space for histogram dithering [default: oklab-lr]
      --no-colorspace-aware-histogram  Disable colorspace-aware histogram dithering

Palette Dithering:
      --input-palette <PNG_FILE>   Use palette from a PNG file (extracts PLTE or unique colors)
      --no-hull-tracing            Disable hull tracing for palette dithering
      --hull-error-decay <FACTOR>  Error decay when palette color is farther than hull boundary
                                   [default: 1.0, typical: 0.5-0.9]

Safetensors Output:
      --safetensors-format <FMT>   Data format [default: fp32] [values: fp32, fp16, bf16]
      --safetensors-transfer <T>   Transfer function [default: srgb]
                                   [values: auto, linear, srgb]
      --safetensors-no-alpha       Strip alpha channel from output
      --safetensors-dither <MODE>  Dithering for FP16/BF16 quantization [default: boon-standard]
      --safetensors-distance-space Perceptual space for safetensors dithering [default: oklab-lr]

Resize:
      --width <W>                  Resize to width (preserves aspect ratio)
      --height <H>                 Resize to height (preserves aspect ratio)
      --scale-method <METHOD>      Resize method [default: ewa-lanczos3]
                                   Separable: bilinear, lanczos2, lanczos3, mitchell, catmull-rom, box
                                   EWA (2D): ewa-lanczos2, ewa-lanczos3, ewa-lanczos3-sharp,
                                             ewa-lanczos4-sharpest, ewa-sinc-lanczos2,
                                             ewa-sinc-lanczos3, ewa-mitchell, ewa-catmull-rom
                                   Iterative: bilinear-iterative, hybrid-lanczos3
                                   Research: sinc, jinc, lanczos3-scatter, sinc-scatter,
                                             stochastic-jinc, stochastic-jinc-scatter,
                                             stochastic-jinc-scatter-normalized
      --non-uniform                Disable automatic uniform scaling detection

Tonemapping:
      --input-tonemapping <MODE>   Apply before histogram matching [values: aces, aces-inverse]
      --tonemapping <MODE>         Apply after processing [values: aces, aces-inverse]
      --exposure <FACTOR>          Exposure adjustment (linear multiplier, before tonemapping)
                                   Values > 1.0 brighten, < 1.0 darken (2.0 = +1 stop)

General:
  -s, --seed <SEED>                Random seed for mixed dithering [default: 12345]
  -v, --verbose                    Enable verbose output
      --progress                   Show progress bar during processing
  -h, --help                       Print help
  -V, --version                    Print version
```

### Examples

```bash
# CRA Oklab color correction (default when reference provided)
./dist/bin/cra -i input.jpg -r reference.jpg -o output.png

# CRA Oklab with f32 histogram matching (highest quality, no quantization)
./dist/bin/cra -i input.jpg -r reference.jpg -o output.png --histogram-mode f32-endpoint

# Tiled Oklab for images with varying color casts across regions
./dist/bin/cra -i input.jpg -r reference.jpg -o output.png --histogram tiled-oklab

# Preserve original luminosity during color correction
./dist/bin/cra -i input.jpg -r reference.jpg -o output.png --keep-luminosity

# Dither-only mode (no color correction) with RGB565 output
./dist/bin/cra -i input.png -o output.png -f RGB565

# Output raw binary with 4-byte row alignment for embedded systems
./dist/bin/cra -i input.png --output-raw output.bin -f RGB565 --stride 4

# Export to safetensors format (FP16, linear color space)
./dist/bin/cra -i input.png --output-safetensors output.safetensors \
    --safetensors-format fp16 --safetensors-transfer linear

# Process image with ICC color profile handling
./dist/bin/cra -i wide-gamut.png -r srgb-ref.jpg -o output.png --input-profile icc

# Output separate RGB channels as raw binary files
./dist/bin/cra -i input.png --output-raw-r r.bin --output-raw-g g.bin --output-raw-b b.bin -f RGB888

# ARGB output with alpha channel for sprites/UI
./dist/bin/cra -i sprite.png -o output.png -f ARGB8888

# Dither to a custom palette from a PNG file
./dist/bin/cra -i input.png -o output.png --input-palette palette.png

# GIF output with palette dithering
./dist/bin/cra -i input.png --output-gif output.gif --input-palette palette.png
```

## Project Structure

```
cra-web/
├── dither/              # Floyd-Steinberg dither WASM module
├── port/                # Main Rust port of CRA algorithms
│   ├── src/
│   │   ├── lib.rs       # WASM exports
│   │   ├── bin/cra/     # CLI tool (main.rs, args.rs)
│   │   ├── basic_*.rs   # Basic histogram matching
│   │   ├── cra_*.rs     # CRA implementations
│   │   ├── tiled_*.rs   # Tiled processing
│   │   ├── dither/      # Error diffusion dithering
│   │   ├── histogram.rs # Histogram matching algorithms
│   │   └── color.rs     # Color space conversions
│   └── Cargo.toml
├── scripts/             # Original Python scripts (submodule)
├── tools/               # Testing and analysis tools (see tools/README.md)
├── index.html           # Color correction web demo
├── dither.html          # Dithering web demo
├── resize.html          # Resize web demo
├── app.js               # Color correction demo logic
├── build.sh             # Web build script
├── build-cli.sh         # CLI build script
└── dist/                # Build output
```

## Acknowledgements

Web demo and Rust port built with Claude Opus 4.5 using [Claude Code](https://claude.com/product/claude-code). All markdown documents have been manually reviewed.

## License

See [scripts/LICENSE.md](scripts/LICENSE.md) for the original Python implementation license.
