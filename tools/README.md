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

Generates synthetic test images and downloads reference images for dithering experiments.

**Outputs** (`tools/test_images/sources/`):
- `gray_XXX.png` - Pathological gray levels (256x256) at values like 0, 1, 42, 64, 85, 127, 128, 170, 191, 212, 213, 254, 255
- `ramp_continuous.png` - Smooth 0-255 gradient (64x4096)
- `ramp_step_32.png` - 8-step ramp, 32-value jumps (64x1024)
- `ramp_step_16.png` - 16-step ramp, 16-value jumps (64x1024)
- `david.png`, `gradient.png`, `gradient_steps.png` - Reference images (downloaded)

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
- `ulichney-standard`, `ulichney-serpentine` - Ulichney threshold perturbation: FS with ±30% threshold noise (1-bit only)
- `ulichney-weight-standard`, `ulichney-weight-serpentine` - Ulichney weight perturbation: FS with ±50% paired weight noise (1-bit only)
- `fs-tpdf-standard`, `fs-tpdf-serpentine` - Floyd-Steinberg with TPDF threshold dither: FS with triangular PDF noise on threshold (1-bit only)

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
python tools/analyze_dither.py --sanity       # Sanity check: coin flip vs blue noise vs CRA vs Python
python tools/analyze_dither.py --blue-kernel  # Kernel mixing experiments (see below)
python tools/analyze_dither.py --ideal        # Plot ideal blue noise vs white noise reference
python tools/analyze_dither.py --vs-blue-noise # Compare our method vs void-and-cluster blue noise
python tools/analyze_dither.py -i path/to/image.png  # Single image analysis
```

### 5. `our_method_dither.py`

Standalone Python replication of "Our Method" for testing/hacking purposes.

Generates 1-bit dithered images from arbitrary gray levels (including fractional values like 127.5).

**Features:**
- Mixed FS/JJN kernel selection using lowbias32 hash (standard method)
- Serpentine scanning
- Accepts any gray level 0-255

**Experimental kernel combinations:**
- `our_method_dither()` - FS/JJN (standard)
- `our_method_dither_fs_stucki()` - FS/Stucki
- `our_method_dither_fs_sierra()` - FS/Sierra
- `our_method_dither_fs_sierra_lite()` - FS/Sierra Lite
- `our_method_dither_jjn_stucki()` - JJN/Stucki (no FS)
- `our_method_dither_jjn_sierra()` - JJN/Sierra (no FS)
- `our_method_dither_stucki_sierra()` - Stucki/Sierra (no FS)
- `our_method_dither_with_blue_noise_kernel()` - Uses pre-dithered blue noise pattern for kernel selection

**Note:** Simplified implementation without edge seeding. Produces equivalent spectral characteristics but not exact pixel match with CRA tool.

```bash
source /root/venv/bin/activate
python tools/our_method_dither.py 127.5                    # 50% gray
python tools/our_method_dither.py 64 -o my_output.png      # 25% gray, custom output
python tools/our_method_dither.py 127.5 --size 512         # Larger image
```

### 6. `blue_dither.h` - Header-Only C Library

Single-header library for blue noise dithering using integer-only arithmetic. Suitable for embedded systems, FPGAs, or anywhere floating point is expensive.

**The Technique:**
Traditional error diffusion (Floyd-Steinberg, Jarvis-Judice-Ninke) produces structured patterns because each kernel has characteristic periodicities. Our method randomly selects between two kernels (FS and JJN) at each pixel using a hash function. This breaks up the periodic structures, producing blue noise characteristics: energy concentrated at high frequencies with suppressed low-frequency content.

**Key design:**
- Uses 48 as common denominator (LCM of FS=16 and JJN=48)
- FS coefficients scaled: 21/48, 9/48, 15/48, 3/48
- JJN coefficients native: 7/48, 5/48, etc.
- All arithmetic uses 32-bit integers, no floating point required
- Three-row circular buffer for 2D error diffusion
- Serpentine scanning for optimal quality

**APIs:**

| Mode | Struct | Functions | Use Case |
|------|--------|-----------|----------|
| 1D Temporal | `BlueDither1D` | `_init()`, `_next()`, `_reset()` | LED PWM, audio DAC |
| 2D Spatial | `BlueDither2D` | `_init()`, `_row()`, `_free()`, `_reset()` | Image dithering |

**1D Kernel (Optimized):**
The 1D mode mixes two kernels for blue noise temporal dithering:
- K1: `[48]` - 100% error to t+1 (tight response)
- K2: `[38,10]` - 79% to t+1, 21% to t+2 (spread response) = 2×[19,5]

The `[38,10]` ratio (~4:1) achieves +5.62 dB/octave average (ideal is +6.0). Note: 38 = 2×19 and 10 = 2×5, where both 19 and 5 are prime.

**Note:** Exhaustive testing shows `[46,2]` (23:1 ratio) achieves slightly better results (+5.67 avg, +4.84 min) compared to `[38,10]` (+5.62 avg, +4.69 min). See `analyze_1d_dither.py --find-best` for details.

**Usage:**
```c
#define BLUE_DITHER_IMPLEMENTATION
#include "blue_dither.h"

// 1D LED PWM example
BlueDither1D bd;
blue_dither_1d_init(&bd, seed);
while (1) {
    int on = blue_dither_1d_next(&bd, brightness);  // brightness 0-255
    set_led(on);
    delay_us(100);  // 10kHz update rate
}

// 2D image example
BlueDither2D bd;
blue_dither_2d_init(&bd, width, seed);
for (int y = 0; y < height; y++) {
    blue_dither_2d_row(&bd, input_row, output_row, y);
}
blue_dither_2d_free(&bd);
```

### 7. `int_blue_dither.c`

Command-line tool demonstrating both 1D and 2D dithering modes using `blue_dither.h`.

**Build:**
```bash
gcc -O2 -o tools/int_blue_dither tools/int_blue_dither.c
```

**Usage:**
```bash
# 2D mode: Generate 256x256 at 50% gray
./tools/int_blue_dither 256 256 127 output.bin

# Convert to PNG using CRA
cra -i output.bin --input-metadata '{"format":"L1","width":256,"height":256}' -o output.png

# 1D mode: Demo temporal dithering
./tools/int_blue_dither --1d 1000 127
```

**Validation:** Included in `--sanity` check, producing identical white pixel ratio to CRA and Python implementations.

### 8. `analyze_1d_dither.py`

Spectral analysis of 1D temporal dithering. Compares our method against:
- **ΣΔ 2nd + TPDF** - Second-order sigma-delta with triangular dither (+12 dB/octave noise shaping)
- **ΣΔ 1st + TPDF** - First-order sigma-delta with triangular dither (+6 dB/octave)
- **PWM** - Traditional pulse width modulation (shows harmonic spikes that cause flicker)
- **White noise** - Random threshold dithering (flat spectrum)

Reference line for +6 dB/octave shown on all charts.

All charts use log frequency scale to clearly show noise shaping characteristics.

**Outputs** (`tools/test_images/analysis/`):
- `spectrum_1d_logscale.png` - Log-frequency spectrum across gray levels
- `spectrum_1d_gray_*.png` - Detailed per-gray-level analysis
- `spectrum_1d_comparison.png` - All gray levels overlaid
- `spectrum_1d_top8_kernels.png` - Top 8 kernels from exhaustive search (default sum=48)
- `spectrum_1d_top8_kernels_sum*.png` - Top 8 kernels for specific kernel sums

```bash
source /root/venv/bin/activate
python tools/analyze_1d_dither.py              # All analyses
python tools/analyze_1d_dither.py --log        # Log-scale only
python tools/analyze_1d_dither.py --low-gray   # Focus on low gray levels (1,2,5,10,20,42,64,85,127)
python tools/analyze_1d_dither.py --count 262144  # Custom sample count
python tools/analyze_1d_dither.py --find-best  # Test all kernels, show top 8 (sum=48)
python tools/analyze_1d_dither.py --find-best --kernel-sum 28 36 48 60  # Test multiple kernel sums
python tools/analyze_1d_dither.py --kernel-compare  # Compare [38,10] vs [36,12] vs [28,20]
```

### 9. `analyze_1d_kernels.py`

Compares different 1D kernel configurations for spectral quality.

**Kernels tested:**
- `[48]+[46,2]` - **Best** (23:1 ratio), avg +5.67 dB/octave, min +4.84
- `[48]+[38,10]` - Runner-up (~4:1 ratio), avg +5.62 dB/octave
- `[48]+[43,5]` - Prime pair (8.6:1 ratio)
- `[48]+[41,7]` - Prime pair (5.9:1 ratio)
- `[48]+[37,11]` - Prime pair (3.4:1 ratio)
- `[48]+[31,17]` - Prime pair (1.8:1 ratio)
- `[48]+[28,20]` - Original (1.4:1 ratio)
- Length-3 variants (generally worse at low gray levels)

**Cross-sum comparison** (best kernel for each sum):
| Sum | Best Kernel | Ratio | Avg | Min |
|-----|-------------|-------|-----|-----|
| 28 | [21,7] | 3.0:1 | +5.51 | +4.36 |
| 36 | [32,4] | 8.0:1 | +5.58 | +4.54 |
| **48** | **[46,2]** | **23.0:1** | **+5.67** | **+4.84** |
| 60 | [54,6] | 9.0:1 | +5.50 | +4.29 |

Sum=48 with kernel [46,2] achieves the highest average and minimum spectral slopes.

**Output:** `spectrum_1d_kernel_comparison.png`

```bash
source /root/venv/bin/activate
python tools/analyze_1d_kernels.py
```

### 10. `experiments/analyze_1d_prime_pairs.py`

Tests all prime pairs [p, q] where p + q = 48 for spectral quality.

**Output:** `spectrum_1d_prime_pairs.png`

```bash
source /root/venv/bin/activate
python tools/experiments/analyze_1d_prime_pairs.py
```

### 11. `noise_color_comparison.py`

Generates reference charts comparing noise color spectra:
- **White**: 0 dB/octave (flat)
- **+3 dB/octave**: (f, claimed by some sources as "blue")
- **+6 dB/octave**: (f², graphics "blue noise")
- **+12 dB/octave**: (f⁴, second-order noise shaping)

In graphics, "blue noise" refers to noise with steep low-frequency suppression, typically around +6 dB/octave. The +3 dB/octave line is included for reference since some sources claim that value, though it's unclear where this originates.

**Outputs** (`tools/test_images/analysis/`):
- `noise_color_comparison.png` - Log frequency scale
- `noise_color_comparison_linear.png` - Linear frequency scale

```bash
source /root/venv/bin/activate
python tools/noise_color_comparison.py          # Both charts
python tools/noise_color_comparison.py --log    # Log scale only
python tools/noise_color_comparison.py --linear # Linear scale only
```

## Full Regeneration Sequence

```bash
# 1. Build CRA (if needed)
cd /root/cra-web/port
cargo build --release --bin cra --features cli

# 2. Generate test images (includes downloading reference images)
source /root/venv/bin/activate
python tools/generate_test_images.py

# 3. Generate RNG noise images
python tools/generate_rng_noise.py

# 4. Run dither tests
./tools/run_dither_tests.sh

# 5. Generate 2D analysis charts
python tools/analyze_dither.py --serpentine
python tools/analyze_dither.py --hash
python tools/analyze_dither.py --rng
python tools/analyze_dither.py --sanity
python tools/analyze_dither.py --blue-kernel

# 6. Generate 1D temporal analysis charts
python tools/analyze_1d_dither.py --all              # Default gray levels (42-213)
python tools/analyze_1d_dither.py --low-gray --all   # Low gray levels (1-127)
python tools/analyze_1d_kernels.py
```

## Output Structure

```
tools/
├── test_images/
│   ├── blue_noise_256.png       # Reference blue noise (not processed)
│   ├── sources/                 # Source test images (processed by run_dither_tests.sh)
│   │   ├── gray_*.png           # Pathological gray levels
│   │   ├── ramp_*.png           # Gradient ramps
│   │   ├── david.png            # Reference image
│   │   ├── gradient.png         # Reference image
│   │   └── gradient_steps.png   # Reference image
│   ├── rng_noise/               # RNG noise test images (not processed)
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
│   │   ├── zhou-fang-serpentine/# Zhou-Fang serpentine
│   │   ├── ulichney-standard/   # Ulichney threshold perturbation standard
│   │   ├── ulichney-serpentine/ # Ulichney threshold perturbation serpentine
│   │   ├── ulichney-weight-standard/   # Ulichney weight perturbation standard
│   │   ├── ulichney-weight-serpentine/ # Ulichney weight perturbation serpentine
│   │   ├── fs-tpdf-standard/           # FS-TPDF threshold dither standard
│   │   └── fs-tpdf-serpentine/         # FS-TPDF threshold dither serpentine
│   └── analysis/
│       ├── *_serpentine.png           # Dither method comparison
│       ├── *_hash_comparison.png      # Hash function comparison
│       ├── rng_noise_*.png            # RNG spectral analysis
│       ├── sanity_check_50pct.png     # Sanity check comparison
│       ├── kernel_exp_gray_*.png      # Kernel mixing experiments
│       ├── blue_kernel_depth_gray_*.png # Blue kernel recursion depth
│       ├── blue_noise_vs_our_method_*.png # Comparison vs void-and-cluster
│       ├── ideal_blue_noise.png       # Ideal blue noise reference
│       ├── spectrum_1d_logscale.png   # 1D temporal: log-frequency analysis
│       ├── spectrum_1d_gray_*.png     # 1D temporal: per-gray-level analysis
│       ├── spectrum_1d_comparison.png # 1D temporal: all gray levels overlaid
│       ├── spectrum_1d_kernel_comparison.png # 1D kernel comparison
│       ├── spectrum_1d_kernel_full_comparison.png # 1D kernel full gray range
│       ├── spectrum_1d_top8_kernels.png # Top 8 kernels (sum=48)
│       ├── spectrum_1d_top8_kernels_sum*.png # Top 8 kernels for various sums
│       ├── spectrum_1d_prime_pairs.png # 1D prime pair kernel analysis
│       ├── noise_color_comparison.png # Noise color spectra (log scale)
│       └── noise_color_comparison_linear.png # Noise color spectra (linear scale)
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

### 1D Temporal Spectrum Charts

The 1D analysis charts (`spectrum_1d_gray_*.png`) show six panels:
1. **Our 1D Method** - Blue noise dithering with smoothed spectrum + raw envelope
2. **ΣΔ 2nd + TPDF** - Second-order sigma-delta with triangular dither (+12 dB/octave)
3. **ΣΔ 1st + TPDF** - First-order sigma-delta with triangular dither (+6 dB/octave)
4. **PWM** - Traditional PWM showing harmonic spikes (comb pattern)
5. **White Noise** - Random threshold with flat spectrum
6. **Comparison** - All methods overlaid

Key features:
- **Log-frequency scale**: Makes dB/octave slopes appear as straight diagonal lines
- **Reference line**: Dashed line at +6 dB/octave (ideal blue noise)
- **PWM harmonics**: Vertical spikes at f = 1/256, 3/256, 5/256... (the cause of visible flicker)
- **Sigma-delta noise shaping**: 1st order = +6 dB/oct, 2nd order = +12 dB/oct; TPDF dither cleans up tonal artifacts

**Quality ratings** based on spectral slope:
- **Excellent**: >5 dB/octave (close to ideal +6)
- **Good**: 4-5 dB/octave
- **OK**: 2-4 dB/octave
- **Poor**: <2 dB/octave

Blue noise temporal dithering produces less perceptible flicker than PWM or white noise because it spreads energy across high frequencies where the eye is less sensitive, rather than concentrating it at specific harmonics.

## Hash Function Notes

The Boon dithering method uses a hash function for per-pixel kernel selection:
- **lowbias32** (default): Improved version with bias 0.107, best spectral properties
- **lowbias32_old** (comparison): Original version with bias 0.174
- **wang_hash** (legacy): Slight diagonal bias visible in spectral analysis

Reference: https://github.com/skeeto/hash-prospector/issues/19

Use `--hash` analysis to compare their spectral characteristics.

## Kernel Mixing Experiments

The `--blue-kernel` analysis compares different error diffusion kernel combinations:

**Kernel sizes:**
- **Floyd-Steinberg (FS)**: 4 coefficients, 1 row forward
- **Jarvis-Judice-Ninke (JJN)**: 12 coefficients, 3 rows
- **Stucki**: 12 coefficients, 3 rows
- **Sierra**: 10 coefficients, 3 rows
- **Sierra Lite**: 3 coefficients, 1 row forward

**Outputs:**
- `kernel_exp_gray_*.png` - Compares FS/JJN, FS/Stucki, FS/Sierra, FS/SierraLite, Stucki/Sierra, JJN/Stucki, JJN/Sierra
- `blue_kernel_depth_gray_*.png` - Compares recursion depths for blue noise kernel selection

**Key findings:**
- Mixing a small kernel (FS) with a large kernel (JJN/Stucki/Sierra) produces good blue noise
- Mixing two large kernels (e.g., Stucki/Sierra, JJN/Stucki) produces harsh linear patterns in radial spectrum
- The asymmetry between kernel sizes appears important for pattern disruption

## Blue Noise Reference

The file `test_images/blue_noise_256.png` is a 256x256 blue noise dither array generated using the void-and-cluster algorithm. It serves as a "gold standard" reference for spectral analysis comparisons. This file is kept in `test_images/` (not `sources/`) so it is not processed by `run_dither_tests.sh`.

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
