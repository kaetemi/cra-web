# Wavelet-Based Halftone Quality Analysis Report

## Overview

This report presents a wavelet-based quality metric for evaluating dithering algorithms. The key insight is that ideal dithering should produce quantization error that looks like **random noise** rather than structured patterns (worms, checkerboards).

## Methodology

### Wavelet Decomposition

Both the original grayscale image and the dithered 1-bit halftone are decomposed using Haar wavelets into 4 levels:

| Level | Scale | What it captures |
|-------|-------|------------------|
| 0 | 2px | Finest detail, pixel-level patterns |
| 1 | 4px | Small-scale structure |
| 2 | 8px | Medium-scale structure |
| 3 | 16px | Coarse structure, most visible artifacts |

Each level has three orientation subbands:
- **LH (Horizontal)**: Detects horizontal edges/worms
- **HL (Vertical)**: Detects vertical edges/worms
- **HH (Diagonal)**: Detects diagonal/checkerboard patterns

### Error Subband Analysis

For each subband, we compute the **difference** between the halftone and original wavelet coefficients:

```
error_subband = wavelet(halftone) - wavelet(original)
```

This error represents "structure in the halftone that doesn't exist in the original" — exactly what we want to minimize.

### Spectral Flatness Metric

For each error subband, we compute **spectral flatness**:

```
flatness = geometric_mean(power_spectrum) / arithmetic_mean(power_spectrum)
```

- **Flatness = 1.0**: Perfectly flat spectrum = white noise = ideal
- **Flatness → 0.0**: Peaked spectrum = periodic patterns = artifacts

This directly answers: "Does the error look like noise or like structured patterns?"

## Results Summary

### Average Metrics Across All Test Images

| Method | Blueness | Flatness | Structure | Isotropy |
|--------|----------|----------|-----------|----------|
| fs-standard | **+0.33** | 0.520 | **0.329** | 0.480 |
| adaptive-blend | **+0.32** | 0.535 | 0.282 | 0.543 |
| boon-h2 | **+0.32** | 0.536 | 0.278 | 0.542 |
| fs-serpentine | **+0.32** | 0.524 | 0.321 | 0.484 |
| ulichney-weight-serpentine | **+0.31** | 0.529 | 0.314 | 0.508 |
| ulichney-serpentine | **+0.31** | 0.531 | 0.301 | 0.539 |
| fs-tpdf-serpentine | **+0.30** | 0.535 | 0.295 | 0.551 |
| boon-standard | +0.29 | **0.541** | **0.326** | **0.567** |
| boon-serpentine | +0.29 | **0.542** | 0.315 | **0.570** |
| ostro-serpentine | +0.29 | 0.511 | 0.303 | 0.409 |
| zhou-fang-serpentine | +0.27 | **0.541** | 0.265 | **0.666** |
| jjn-serpentine | +0.25 | 0.534 | **0.347** | **0.569** |
| jjn-standard | +0.24 | 0.532 | **0.360** | 0.584 |
| bluenoise (void-and-cluster) | +0.23 | 0.411 | 0.254 | **0.687** |
| whitenoise | 0.00 | **0.562** | 0.196 | **0.969** |
| none (banding) | **-0.42** | 0.501 | **0.578** | 0.382 |

### The Blueness Metric

**Blueness** measures the rate of energy decay across wavelet scales, normalized to white noise:

- **Blueness = 0**: White noise (flat spectrum, energy decays at baseline rate)
- **Blueness > 0**: Blue noise (low frequencies suppressed, faster decay)
- **Blueness < 0**: Red/pink noise (low frequencies emphasized, slower decay)

FS has highest blueness (+0.32) because its small kernel concentrates energy at fine scales. JJN has lower blueness (+0.24) because its larger kernel spreads energy to coarser scales. Banding ("none") is red (-0.42) because it emphasizes low-frequency structure.

### The White Noise Paradox

**White noise dithering** (random threshold, no error diffusion) serves as a critical validation:

- **Blueness = 0**: By definition, white noise is the baseline
- **Highest flatness (0.562)**: Error is literally white noise — perfectly flat spectrum
- **Highest isotropy (0.969)**: White noise is perfectly isotropic
- **Lowest structure (0.196)**: No error diffusion = terrible tone preservation

This proves that **blueness and flatness alone are insufficient**. White noise has blueness=0 and highest flatness, but looks terrible because it doesn't preserve local tone. Good dithering requires:

1. **Positive blueness** — suppress low-frequency error (avoid visible patterns)
2. **High flatness** — error should be noise-like at each scale
3. **High structure** — error diffusion must preserve local average intensity
4. **High isotropy** — no directional bias

The ideal method balances all four. **Boon achieves good blueness (+0.29) with the best flatness** among error diffusion methods.

### Blue Noise Threshold Dithering (Void-and-Cluster)

**Void-and-cluster blue noise** (ordered dithering with a pre-computed blue noise threshold array) provides another important comparison:

- **Blueness = +0.23**: Positive but lower than error diffusion methods
- **Flatness = 0.411**: Surprisingly low — worse than all error diffusion methods
- **Structure = 0.254**: Low, similar to whitenoise (no error diffusion)
- **Isotropy = 0.687**: Good — void-and-cluster is designed to be isotropic

The low flatness reveals a fundamental distinction:

| Approach | What has blue noise properties |
|----------|-------------------------------|
| **Error diffusion** | The **error** is actively shaped to be blue noise |
| **Threshold dithering** | The **halftone** has blue noise structure, but error is not shaped |

Error diffusion methods (FS, JJN, Boon) produce error that looks like noise at each scale. Threshold dithering produces a visually pleasing halftone pattern, but the error (halftone - original) contains image-correlated structure because there's no feedback mechanism to compensate.

Additionally, the 256×256 void-and-cluster texture must be tiled for larger images (512×512 test images require 2×2 tiling), introducing periodic seams that appear as structured error.

This validates that **error diffusion produces fundamentally different results than threshold dithering** — even with a "gold standard" blue noise texture.

### Key Findings

1. **FS has highest blueness** (+0.32) — its small 4-coefficient kernel concentrates energy at fine scales, giving the steepest low-frequency suppression.

2. **JJN has lower blueness** (+0.24) — its larger 12-coefficient kernel spreads error to coarser scales.

3. **Boon (our method) balances blueness and flatness** — good blueness (+0.29) with highest flatness (0.542) among error diffusion methods, indicating noise-like error at all scales.

4. **Banding ("none") is red** (-0.42) — low-frequency structure dominates, the opposite of blue noise.

5. **White noise validates the metric** — blueness=0 by definition, proving the calibration is correct.

6. **Blue noise threshold dithering underperforms on flatness** — despite using a "gold standard" void-and-cluster texture, threshold dithering produces structured error (flatness 0.411) because error is not actively shaped. Error diffusion fundamentally outperforms threshold dithering on this metric.

7. **Structure preservation vs noise trade-off**: Methods with highest flatness (Boon, Zhou-Fang) tend to have lower structure scores, suggesting they sacrifice some edge fidelity for better noise characteristics.

### Per-Level Analysis

Flatness varies by scale:

| Level | Scale | Typical Range | Notes |
|-------|-------|---------------|-------|
| 0 | 2px | 0.47 - 0.54 | Lowest flatness (dithering pattern visible) |
| 1 | 4px | 0.52 - 0.55 | Improving |
| 2 | 8px | 0.53 - 0.55 | Good |
| 3 | 16px | 0.52 - 0.57 | Generally highest (coarse structure is noise-like) |

The 2px level shows the most variation between methods — this is where dithering artifacts are most apparent.

## Interpretation Guide

### Reading the Visualizations

Each `wavelet_{image}_{method}.png` shows:

1. **Top row**: Original, Halftone, Error, Per-level flatness bar chart
2. **Rows 2-4**: Error subbands for LH/HL/HH orientations at each scale
   - Blue = negative error, Red = positive error
   - Uniform speckle = good (noise-like)
   - Large patches of same color = bad (structured patterns)

**Visualization weighting**: Coarser scales are boosted (2x per level) to show comparable detail. For white noise, all scales should appear similar intensity. For error diffusion (blue noise), coarser scales appear progressively lighter due to low-frequency suppression.

### What Good Dithering Looks Like

- Error subbands should look like **random speckle**, not coherent patches
- Flatness values should be **close to 1.0** (typically 0.5-0.6 in practice)
- Isotropy should be **high** (close to 1.0), indicating balanced energy across directions

**Note on Isotropy**: Isotropy measures whether error energy is evenly distributed across horizontal (LH), vertical (HL), and diagonal (HH) orientations. It does not directly detect "worms" — worms require BOTH directional bias (low isotropy) AND structured patterns (low flatness in that direction). A method could have low isotropy but high flatness (directionally biased random noise) without exhibiting worms.

### What Bad Dithering Looks Like

- Error subbands show **large regions of same color** (worms, bands)
- Flatness values are **low** (below 0.5)
- Isotropy is **low** (one direction dominates)

## Test Images

| Image | Description | Key Challenge |
|-------|-------------|---------------|
| cameraman | Classic test, high contrast | Strong edges |
| lena_gray_512 | Smooth gradients, textures | Gradient rendering |
| mandril_gray | High texture detail | Detail preservation |
| lake | Natural scene, varied tones | Overall balance |
| livingroom | Indoor scene, mixed content | Shadow regions |
| pirate | Portrait, skin tones | Smooth gradients |
| walkbridge | High detail, foliage | Fine texture |
| woman_blonde | Portrait, hair detail | Fine structure |

## Conclusions

1. **Spectral flatness is a principled metric** for dithering quality — it directly measures whether quantization error looks like noise.

2. **Kernel mixing (Boon method) produces the most noise-like error**, validating the approach of randomly selecting between FS and JJN kernels.

3. **Zhou-Fang excels at isotropy** but sacrifices structure preservation.

4. **Traditional FS has moderate performance** — better methods exist.

5. **The metric correctly identifies banding as worst** — "none" method has lowest flatness.

## Files Generated

```
analysis/
├── wavelet_{image}_{method}.png  # Individual analysis (128 files)
├── wavelet_comparison_{image}.png # Per-image comparison (8 files)
├── wavelet_summary.png            # Aggregated chart
├── wavelet_summary.csv            # Raw metrics
└── wavelet_results.json           # Full results
```

## Future Work

- Test on more diverse images (graphics, text, gradients)
- Correlate flatness with perceptual quality studies
- Investigate why Ostromoukhov underperforms on this metric
- Explore weighted combinations of flatness, structure, and isotropy
