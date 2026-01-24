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

### 2. `generate_rng_noise.py`

Generates 1-bit noise images using various hash functions for spectral comparison.

**Outputs** (`tools/test_images/rng_noise/`):
- `*_coord.png` - Coordinate-based hashing (GPU-friendly)
- `*_seq.png` - Sequential hashing

Hash functions tested: Wang, Double Wang, Triple32, Lowbias32, Lowbias32_old, xxHash32, IQ Int1, IQ Int3, Murmur3, PCG, SplitMix32, Xorshift32, LCG, NumPy random.

```bash
source /root/venv/bin/activate
python tools/generate_rng_noise.py
```

### 3. `run_dither_tests.sh`

Runs all test images through CRA with every dithering mode.

**Modes tested:**
- `fs-standard`, `fs-serpentine` - Floyd-Steinberg
- `jjn-standard`, `jjn-serpentine` - Jarvis-Judice-Ninke
- `boon-standard`, `boon-serpentine` - Boon (Our method): FS/JJN per-pixel with lowbias32 hash
- `boon-wanghash`, `boon-wanghash-serpentine` - Boon with legacy wang_hash (for comparison)
- `boon-lowbias`, `boon-lowbias-serpentine` - Boon with original lowbias32 (bias 0.174, for comparison)
- `ostro-standard`, `ostro-serpentine` - Ostromoukhov variable coefficients
- `zhou-fang-standard`, `zhou-fang-serpentine` - Zhou-Fang threshold modulation

**Outputs** (`tools/test_images/dithered/{mode}/`):
- L1 (1-bit) dithered versions of all test images

```bash
./tools/run_dither_tests.sh
```

### 4. `analyze_dither.py`

Generates Fourier analysis charts (like Zhou-Fang paper Figure 5):
- 2D FFT power spectrum (shows isotropy/anisotropy)
- Segmented radially averaged power spectrum (H/D/V curves)

**Outputs** (`tools/test_images/analysis/`):
- `{image}_serpentine.png` - Comparison of serpentine dither methods
- `{image}_hash_comparison.png` - Comparison of boon hash functions (lowbias32 vs lowbias32_old vs wang)
- `rng_noise_*.png` - RNG spectral analysis

```bash
source /root/venv/bin/activate
python tools/analyze_dither.py --serpentine   # All images, serpentine modes
python tools/analyze_dither.py --compare      # All images, all modes
python tools/analyze_dither.py --hash         # Hash function comparison (lowbias32 vs wang)
python tools/analyze_dither.py --rng          # RNG noise spectral analysis
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

# 4. Generate RNG noise images
python tools/generate_rng_noise.py

# 5. Run dither tests
./tools/run_dither_tests.sh

# 6. Generate analysis charts
python tools/analyze_dither.py --serpentine
python tools/analyze_dither.py --hash
python tools/analyze_dither.py --rng
```

## Output Structure

```
tools/
├── test_images/
│   ├── *.png                    # Source test images
│   ├── rng_noise/               # RNG noise test images
│   │   ├── *_coord.png          # Coordinate-based hashing
│   │   └── *_seq.png            # Sequential hashing
│   ├── dithered/
│   │   ├── fs-standard/         # Floyd-Steinberg standard
│   │   ├── fs-serpentine/       # Floyd-Steinberg serpentine
│   │   ├── jjn-standard/        # Jarvis-Judice-Ninke standard
│   │   ├── jjn-serpentine/      # Jarvis-Judice-Ninke serpentine
│   │   ├── boon-standard/       # Boon (lowbias32) standard
│   │   ├── boon-serpentine/     # Boon (lowbias32) serpentine
│   │   ├── boon-wanghash/       # Boon (wang_hash) standard
│   │   ├── boon-wanghash-serpentine/  # Boon (wang_hash) serpentine
│   │   ├── boon-lowbias/        # Boon (lowbias32_old) standard
│   │   ├── boon-lowbias-serpentine/  # Boon (lowbias32_old) serpentine
│   │   ├── ostro-standard/      # Ostromoukhov standard
│   │   ├── ostro-serpentine/    # Ostromoukhov serpentine
│   │   ├── zhou-fang-standard/  # Zhou-Fang standard
│   │   └── zhou-fang-serpentine/# Zhou-Fang serpentine
│   └── analysis/
│       ├── *_serpentine.png     # Dither method comparison
│       ├── *_hash_comparison.png # Hash function comparison
│       └── rng_noise_*.png      # RNG spectral analysis
```

## Interpreting Analysis Charts

Each chart has 3 rows:
1. **Halftone**: Visual dither pattern
2. **Spectrum**: 2D FFT - symmetric dark center + bright ring = good blue noise
3. **Radial**: H/D/V power curves - overlapping = isotropic (good), diverging = directional artifacts (bad)

Key test cases:
- `gray_064` (25%) - Low density torture test
- `gray_085` (33%) - 1/3 density
- `gray_127` (50%) - Maximum pattern complexity
- `david` - Real-world image with gradients and detail

## Hash Function Notes

The Boon dithering method uses a hash function for per-pixel kernel selection:
- **lowbias32** (default): Improved version with bias 0.107, best spectral properties
- **lowbias32_old** (comparison): Original version with bias 0.174
- **wang_hash** (legacy): Slight diagonal bias visible in spectral analysis

Reference: https://github.com/skeeto/hash-prospector/issues/19

Use `--hash` analysis to compare their spectral characteristics.

## Blue Noise Reference

The file `test_images/blue_noise_256.png` is a 256x256 blue noise dither array generated using the void-and-cluster algorithm. It serves as a "gold standard" reference for spectral analysis comparisons.

**Generation method:**
- Algorithm: Void-and-cluster (Ulichney 1993)
- Implementation: [MomentsInGraphics/BlueNoise](https://github.com/MomentsInGraphics/BlueNoise) by Christoph Peters (CC0 Public Domain)
- Parameters: StandardDeviation=1.5, seed=42
- Output: 8-bit grayscale PNG with pixel values representing dither thresholds (0-255)

**To regenerate:**
```bash
# Download generator (requires scipy, pypng)
curl -sL "https://raw.githubusercontent.com/MomentsInGraphics/BlueNoise/master/BlueNoise.py" -o /tmp/BlueNoise.py
# Fix for modern numpy
sed -i 's/np\.int\b/np.int64/g; s/np\.bool\b/np.bool_/g' /tmp/BlueNoise.py

source /root/venv/bin/activate
pip install scipy pypng
python3 << 'EOF'
import sys; sys.path.insert(0, '/tmp')
from BlueNoise import GetVoidAndClusterBlueNoise
import numpy as np
from PIL import Image

np.random.seed(42)
dither_array = GetVoidAndClusterBlueNoise((256, 256), StandardDeviation=1.5)
dither_array = (dither_array * 255.0 / (256*256 - 1)).astype(np.uint8)
Image.fromarray(dither_array, mode='L').save('tools/test_images/blue_noise_256.png')
EOF
```

**Usage for comparison:** Threshold at gray level G to get ideal blue noise pattern at that density:
```python
blue_noise = load_image('blue_noise_256.png')
threshold = G  # e.g., 127 for 50% density
ideal_pattern = (blue_noise < threshold) * 255
```
