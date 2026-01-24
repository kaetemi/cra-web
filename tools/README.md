# Dither Testing Tools

Tools for generating test images and analyzing dithering algorithm quality.

## Environment

- Python virtual environment: `/root/venv/bin/activate`
- CRA binary: `/root/cra-web/port/target/release/cra`

### Python Dependencies

```bash
source /root/venv/bin/activate
pip install numpy pillow matplotlib
```

## Scripts

### 1. `generate_test_images.py`

Generates synthetic test images for dithering experiments.

**Outputs** (`tools/test_images/`):
- `gray_XXX.png` - Pathological gray levels (256x256) at values like 0, 1, 42, 64, 85, 127, 128, 170, 191, 212, 213, 254, 255
- `ramp_continuous.png` - Smooth 0-255 gradient (64x4096)
- `ramp_step_32.png` - 8-step ramp, 32-value jumps (64x1024)
- `ramp_step_16.png` - 16-step ramp, 16-value jumps (64x1024)

```bash
source /root/venv/bin/activate
python tools/generate_test_images.py
```

### 2. `run_dither_tests.sh`

Runs all test images through CRA with every dithering mode.

**Modes tested:**
- `fs-standard`, `fs-serpentine` - Floyd-Steinberg
- `jjn-standard`, `jjn-serpentine` - Jarvis-Judice-Ninke
- `boon-standard`, `boon-serpentine` - Boon (Our method): FS/JJN per-pixel
- `ostro-standard`, `ostro-serpentine` - Ostromoukhov variable coefficients
- `zhou-fang-standard`, `zhou-fang-serpentine` - Zhou-Fang threshold modulation

**Outputs** (`tools/test_images/dithered/{mode}/`):
- L1 (1-bit) dithered versions of all test images

```bash
./tools/run_dither_tests.sh
```

### 3. `analyze_dither.py`

Generates Fourier analysis charts (like Zhou-Fang paper Figure 5):
- 2D FFT power spectrum (shows isotropy/anisotropy)
- Segmented radially averaged power spectrum (H/D/V curves)

**Outputs** (`tools/test_images/analysis/`):
- `{image}_serpentine.png` - 3-row comparison chart for all serpentine modes

```bash
source /root/venv/bin/activate
python tools/analyze_dither.py --serpentine   # All images, serpentine modes
python tools/analyze_dither.py --compare      # All images, all modes
python tools/analyze_dither.py -i path/to/image.png  # Single image analysis
```

## Full Regeneration Sequence

```bash
# 1. Build CRA (if needed)
cd /root/cra-web/port
cargo build --release --bin cra --features cli

# 2. Generate test images
source /root/venv/bin/activate
python tools/generate_test_images.py

# 3. Download reference images (optional)
cd tools/test_images
curl -sL "https://raw.githubusercontent.com/dalpil/structure-aware-dithering/main/examples/gradient-steps/original.png" -o gradient_steps.png
curl -sL "https://raw.githubusercontent.com/dalpil/structure-aware-dithering/main/examples/gradient/original.png" -o gradient.png
curl -sL "https://raw.githubusercontent.com/dalpil/structure-aware-dithering/main/examples/david-original.png" -o david.png
cd ../..

# 4. Run dither tests
./tools/run_dither_tests.sh

# 5. Generate analysis charts
python tools/analyze_dither.py --serpentine
```

## Output Structure

```
tools/
├── test_images/
│   ├── *.png                    # Source test images
│   ├── dithered/
│   │   ├── fs-standard/         # Floyd-Steinberg standard
│   │   ├── fs-serpentine/       # Floyd-Steinberg serpentine
│   │   ├── jjn-standard/        # Jarvis-Judice-Ninke standard
│   │   ├── jjn-serpentine/      # Jarvis-Judice-Ninke serpentine
│   │   ├── boon-standard/      # Boon (Our method) standard
│   │   ├── boon-serpentine/    # Boon (Our method) serpentine
│   │   ├── ostro-standard/      # Ostromoukhov standard
│   │   ├── ostro-serpentine/    # Ostromoukhov serpentine
│   │   ├── zhou-fang-standard/  # Zhou-Fang standard
│   │   └── zhou-fang-serpentine/# Zhou-Fang serpentine
│   └── analysis/
│       └── *_serpentine.png     # Fourier analysis charts
```

## Interpreting Analysis Charts

Each chart has 3 rows:
1. **Halftone**: Visual dither pattern
2. **Spectrum**: 2D FFT - symmetric dark center + bright ring = good blue noise
3. **Radial**: RGB power curves - overlapping = isotropic (good), diverging = directional artifacts (bad)

Key test cases:
- `gray_064` (25%) - Low density torture test
- `gray_085` (33%) - 1/3 density
- `gray_127` (50%) - Maximum pattern complexity
- `david` - Real-world image with gradients and detail
