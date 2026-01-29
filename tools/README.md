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
- **ΣΔ 1st order** - First-order sigma-delta modulation (+6 dB/octave noise shaping, tonal spikes)
- **ΣΔ 1st + TPDF dither** - First-order sigma-delta with triangular dither (cleaner spectrum)
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

### 11. `principal_frequency_comparison.py`

Shows how blue noise principal frequency (f_c) scales with gray level.

**Key insight:** For Poisson disk / blue noise patterns, dot spacing ~ 1/sqrt(density), so f_c ~ sqrt(g). This is fundamental to understanding why blue noise dithering works.

**Formulation (Mitsa-Parker):**
```
f_c = K * sqrt(g)      for g < 0.5
f_c = K * sqrt(1-g)    for g >= 0.5
```

Where K is a constant:
- Mitsa-Parker optimal: K = 1/sqrt(2) ~ 0.707
- Error diffusion: K = 1

**Output:** `principal_frequency_comparison.png`

```bash
source /root/venv/bin/activate
python tools/principal_frequency_comparison.py
```

### 12. `spectrum_shape_comparison.py`

Compares the Mitsa-Parker blue noise model vs simple power law models.

**Mitsa-Parker model proposes a BANDPASS shape:**
- Near zero below ~0.5*f_c (low frequency suppression)
- Steep rise to peak at principal frequency f_c
- Plateau after peak (does NOT keep rising)

This is fundamentally different from simple power laws (+3 or +6 dB/oct) which keep rising toward Nyquist.

Note: This is a theoretical model, not necessarily what empirical blue noise looks like. The plateau essentially acts like a blur filter on high frequencies.

**Output:** `spectrum_shape_comparison.png`

```bash
source /root/venv/bin/activate
python tools/spectrum_shape_comparison.py
```

### 13. `noise_color_comparison.py`

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

### 14. `generate_whitenoise_dither.py`

Generates white noise dithered images for comparison/validation.

**Method:** Each pixel is compared against an independent random threshold [0, 255]. No error diffusion.

**Properties:**
- Perfect spectral flatness (error is literally white noise)
- Highest isotropy (no directional bias)
- Terrible visual quality (no local tone preservation)
- Blueness = 0 by definition (baseline for calibration)

```bash
source /root/venv/bin/activate
python tools/generate_whitenoise_dither.py
python tools/generate_whitenoise_dither.py --ref-dir path/to/images --output-dir path/to/output
```

### 15. `generate_bluenoise_dither.py`

Generates blue noise (ordered) dithered images using the void-and-cluster threshold array.

**Method:** Each pixel is compared against the corresponding value in the blue noise texture (tiled to cover the image).

**Properties:**
- Good spectral properties (blue noise characteristics)
- No error diffusion (purely threshold-based)
- May show tiling artifacts on large images (texture is 256x256)
- Useful as "gold standard" blue noise reference

```bash
source /root/venv/bin/activate
python tools/generate_bluenoise_dither.py
python tools/generate_bluenoise_dither.py --blue-noise path/to/texture.png
```

### 16. `run_wavelet_tests.sh`

Comprehensive test script that generates dithered images from reference photographs and runs wavelet-based quality analysis.

**Reference images** (`tools/test_wavelets/reference_images/`):
- Classic test images: cameraman, lena, mandril, pirate, etc.

**Dithering methods tested:**
- `fs-standard`, `fs-serpentine` - Floyd-Steinberg
- `jjn-standard`, `jjn-serpentine` - Jarvis-Judice-Ninke
- `boon-standard`, `boon-serpentine` - Boon (our method)
- `ostro-serpentine` - Ostromoukhov
- `zhou-fang-serpentine` - Zhou-Fang
- `ulichney-serpentine`, `ulichney-weight-serpentine` - Ulichney perturbation variants
- `fs-tpdf-serpentine` - FS with TPDF threshold dither
- `none` - No dithering (banding baseline)

```bash
./tools/run_wavelet_tests.sh
```

### 17. `analyze_wavelet.py`

Wavelet-based halftone quality analysis using Haar wavelet decomposition.

**Method:**
Compares dithered 1-bit images against their grayscale originals by decomposing both into wavelet subbands (LH=horizontal, HL=vertical, HH=diagonal) at multiple scales (2px, 4px, 8px, 16px).

**Metrics computed:**
- **Spectral flatness**: Measures whether the error subband looks like noise (flat spectrum) or has structure (peaked spectrum). Computed as geometric_mean(power) / arithmetic_mean(power).
  - Flatness = 1.0 → flat spectrum = white noise = ideal dithering
  - Flatness → 0.0 → peaked spectrum = periodic patterns = worms/checkerboards
- **Correlation**: Structure preservation between original and halftone subbands
- **Isotropy ratios**: H/V/D energy distribution (ideal = 0.333 each)
- **Isotropy score**: min/max ratio across orientations (1.0 = perfect)

**Summary scores:**
- `blueness` - Rate of energy decay across scales, normalized to white noise (0=white, +=blue, -=red)
- `flatness_avg` - Average spectral flatness across all levels (higher = more noise-like = better)
- `structure_score` - Energy-weighted correlation (higher = better edge/detail preservation)
- `isotropy_score` - Geometric mean of per-level isotropy (higher = more uniform, less directional bias)

**Outputs** (`tools/test_wavelets/analysis/`):
- `wavelet_{image}_{method}.png` - Individual analysis visualization
- `wavelet_comparison_{image}.png` - Per-image method comparison
- `wavelet_summary.png` - Aggregated comparison across all images
- `wavelet_summary.csv` - Raw metrics data
- `wavelet_results.json` - Full results for further analysis

```bash
source /root/venv/bin/activate

# Compare all methods on all reference images
python tools/analyze_wavelet.py --compare

# Single image pair analysis
python tools/analyze_wavelet.py -o original.png --halftone dithered.png

# Custom directories
python tools/analyze_wavelet.py --compare \
    --ref-dir tools/test_wavelets/reference_images \
    --dithered-dir tools/test_wavelets/dithered \
    --output-dir tools/test_wavelets/analysis
```

**Interpreting results:**
- **Positive blueness = blue noise** (low-frequency suppression); FS has highest (+0.32), JJN lower (+0.24)
- **Blueness = 0 = white noise** (baseline); **Negative = red/pink** (banding is -0.42)
- **Higher flatness = better** (error looks more like noise at each scale)
- Higher isotropy = more uniform directional distribution (less worm-like)
- Higher structure score = better edge/detail preservation
- Boon balances blueness (+0.29) with highest flatness (0.54) among error diffusion methods
- Zhou-Fang has best isotropy (0.67) due to threshold modulation

### 18. `test_map/generate_recursive_map.py`

Floating-point mixed FS/JJN error diffusion with recursive bit decomposition for generating multi-bit ranked dither arrays.

**Features:**
- Generates gradient dithering at multiple bit depths (1-8 bits)
- Recursive ranked output: builds 8-bit threshold maps by decomposing each bit level
- Spectral analysis comparing recursive dither arrays against void-and-cluster blue noise
- Spatial transform support (XY swap, mirror X/Y) for randomized dithering
- Delay parameter for FIFO-based error diffusion timing experiments

**Outputs** (`tools/test_map/`):
- `gradient_*bit.png/.npy` - Gradient dithering at various bit depths
- `gradient_*_delay*.png/.npy` - Delayed error diffusion experiments
- `ranked_output.png/.npy` - Final 8-bit ranked threshold map
- `ranked_level*.png` - Intermediate ranked maps (levels 0-7)
- `analysis_gray_*.png` - Spectral analysis per gray level vs void-and-cluster
- `analysis_histogram.png` - Rank value distribution check
- `analysis_slopes.png` - Spectral slope across all thresholds
- `analysis_ranked.png` - Ranked array vs void-and-cluster comparison

```bash
source /root/venv/bin/activate
python tools/test_map/generate_recursive_map.py --gradient 1 2 4 8   # Gradients at bit depths 1-8
python tools/test_map/generate_recursive_map.py --bits 4 --gray 0.5  # 50% gray at 4-bit
python tools/test_map/generate_recursive_map.py --recursive-test     # Full ranked output + analysis
```

### 19. `dither_map_experiment.py`

Tools for building and analyzing threshold maps from error diffusion patterns.

**Features:**
- Clean implementation of "Our Method" (FS/JJN with lowbias32) for generating patterns
- Generate 8-bit threshold maps from 8 independent 50% patterns
- Spectral analysis comparing threshold maps against blue noise and error diffusion
- Density accuracy testing

```bash
source /root/venv/bin/activate
python tools/dither_map_experiment.py --gray 127.5                    # Single dither pattern
python tools/dither_map_experiment.py --generate-map -o threshold.png # 8-bit threshold map
python tools/dither_map_experiment.py --analyze-map threshold.png     # Spectral analysis
python tools/dither_map_experiment.py --test-map threshold.png        # Density accuracy test
```

### 20. `compare_kernels.py`

Compares two 1D dithering kernels across the full gray range for spectral quality.

```bash
source /root/venv/bin/activate
python tools/compare_kernels.py                        # Compare [38,10] vs [46,2]
python tools/compare_kernels.py --k1 38 10 --k2 46 2  # Custom kernels
```

### 21. `wavelet_pattern_demo.py`

Visual demonstration of how Haar wavelets encode different patterns (horizontal/vertical/diagonal lines, checkerboard, noise). Shows why wavelet analysis is effective for detecting dithering artifacts like directional "worms".

```bash
source /root/venv/bin/activate
python tools/wavelet_pattern_demo.py
```

### 22. `tent_kernel.py` (and variants)

Tent-space kernel derivation tools for computing effective direct kernels for arbitrary downsampling ratios. The key insight is composing box-to-tent expansion, kernel resampling, and tent-to-box contraction.

**Variants:**
- `tent_kernel.py` - 1D tent-space kernel derivation
- `tent_kernel_2d.py` - 2D tent-space kernel derivation
- `tent_kernel_bruteforce.py` - 1D brute-force kernel exploration
- `tent_kernel_2d_bruteforce.py` - 2D brute-force kernel exploration
- `tent_kernel_bruteforce_lanczos.py` - Brute-force Lanczos kernel optimization
- `tent_kernel_lanczos_constraint.py` - Constrained Lanczos kernel derivation

```bash
source /root/venv/bin/activate
python tools/tent_kernel.py --ratio 2 --kernel box --width 2      # 2x downsample
python tools/tent_kernel.py --ratio 3 --kernel lanczos2 --width 4 # 3x with Lanczos-2
```

### 23. Color Science Tools

Tools for deriving and validating color science constants.

- `derive_d65.py` - Derives D65 chromaticity coordinates (0.31272, 0.32903) from CIE source SPD data. Proves chromaticity is derived from spectral power distribution, not defined independently.
- `derive_d65_1nm.py` - 1nm-resolution version of D65 derivation
- `d65_constants.py` - Pre-computed D65 constants for fast lookup
- `derive_k_from_matrix.py` - Derives correlated color temperature from an RGB-to-XYZ matrix white point. Shows how the sRGB matrix implicitly defines a D65 white point.

```bash
source /root/venv/bin/activate
python tools/derive_d65.py
python tools/derive_k_from_matrix.py
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

# 4. Run dither tests (flat grays, gradients)
./tools/run_dither_tests.sh

# 5. Generate 2D spectral analysis charts
python tools/analyze_dither.py --serpentine
python tools/analyze_dither.py --hash
python tools/analyze_dither.py --rng
python tools/analyze_dither.py --sanity
python tools/analyze_dither.py --blue-kernel

# 6. Generate 1D temporal analysis charts
python tools/analyze_1d_dither.py --all              # Default gray levels (42-213)
python tools/analyze_1d_dither.py --low-gray --all   # Low gray levels (1-127)
python tools/analyze_1d_kernels.py

# 7. Generate reference charts
python tools/principal_frequency_comparison.py       # f_c vs gray level
python tools/spectrum_shape_comparison.py            # Bandpass vs power law
python tools/noise_color_comparison.py               # Noise color spectra

# 8. Run wavelet-based quality analysis (real images)
./tools/run_wavelet_tests.sh                         # Generates dithered + analysis

# 9. Generate recursive dither map + analysis
python tools/test_map/generate_recursive_map.py --recursive-test
```

## Output Structure

```
tools/
├── test_images/                 # Flat gray and gradient tests (spectral analysis)
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
│       ├── principal_frequency_comparison.png # f_c vs gray level
│       ├── spectrum_shape_comparison.png # Blue noise vs power law spectra
│       ├── noise_color_comparison.png # Noise color spectra (log scale)
│       └── noise_color_comparison_linear.png # Noise color spectra (linear scale)
│
├── test_map/                    # Recursive dither map experiments
│   ├── generate_recursive_map.py    # Generator script
│   ├── gradient_*bit.png/.npy       # Gradient dithering outputs
│   ├── ranked_output.png/.npy       # Final 8-bit ranked threshold map
│   ├── ranked_level*.png            # Intermediate ranked maps
│   ├── analysis_gray_*.png          # Per-gray spectral analysis vs blue noise
│   ├── analysis_histogram.png       # Rank value distribution
│   ├── analysis_slopes.png          # Spectral slopes across thresholds
│   └── analysis_ranked.png          # Ranked array vs void-and-cluster
│
├── test_wavelets/               # Real image tests (wavelet analysis)
│   ├── reference_images/        # Source photographs (cameraman, lena, etc.)
│   ├── dithered/                # Dithered versions by method
│   │   ├── fs-serpentine/
│   │   ├── jjn-serpentine/
│   │   ├── boon-serpentine/
│   │   ├── ostro-serpentine/
│   │   ├── zhou-fang-serpentine/
│   │   ├── ulichney-serpentine/
│   │   ├── ulichney-weight-serpentine/
│   │   ├── fs-tpdf-serpentine/
│   │   ├── none/
│   │   ├── whitenoise/              # Python: random threshold (validation baseline)
│   │   ├── bluenoise/               # Python: void-and-cluster threshold array
│   │   └── ...
│   └── analysis/
│       ├── wavelet_{image}_{method}.png  # Individual visualizations
│       ├── wavelet_comparison_{image}.png # Per-image method comparison
│       ├── wavelet_summary.png            # Aggregated comparison chart
│       ├── wavelet_summary.csv            # Raw metrics data
│       └── wavelet_results.json           # Full results JSON
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
2. **ΣΔ 1st Order** - First-order sigma-delta (+6 dB/octave, shows tonal spikes)
3. **ΣΔ 1st + TPDF Dither** - First-order sigma-delta with triangular dither (cleaner)
4. **PWM** - Traditional PWM showing harmonic spikes (comb pattern)
5. **White Noise** - Random threshold with flat spectrum
6. **Comparison** - All methods overlaid

Key features:
- **Log-frequency scale**: Makes dB/octave slopes appear as straight diagonal lines
- **Reference line**: Dashed line at +6 dB/octave (ideal blue noise)
- **PWM harmonics**: Vertical spikes at f = 1/256, 3/256, 5/256... (the cause of visible flicker)
- **Sigma-delta tonal spikes**: Standard ΣΔ shows periodic artifacts; TPDF dither eliminates them

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
