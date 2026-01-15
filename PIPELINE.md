# Processing Pipeline

This document describes the image processing pipeline used by CRA for color correction, resizing, and dithering operations.

## Overview

The pipeline processes images through several stages with careful attention to color space correctness:

```
Input → Load → [Linear Path or sRGB Direct] → Dither → Quantize → Output
```

All interpolation and color correction happens in linear RGB space for physically correct results. The final quantization uses perceptually-aware error diffusion dithering.

---

## 1. Image Loading

### Bit Depth Handling

Images are loaded preserving their native bit depth (8-bit or 16-bit). Conversion to floating point uses canonical divisors:

| Input | Conversion | Range |
|-------|------------|-------|
| uint8 | ÷ 255 | 0.0 – 1.0 |
| uint16 | ÷ 65535 | 0.0 – 1.0 |

These divisors are coherent: a uint8 value bit-replicated to uint16 produces the same float.

For non-divisor bit depths (3, 5, 6 bits used in formats like RGB332, RGB565), values are first bit-replicated to uint8, then divided by 255. See [BITDEPTH.md](BITDEPTH.md) for details.

### ICC Profile Detection

The pipeline examines embedded ICC profiles to determine color space handling:

| Profile Mode | Behavior |
|--------------|----------|
| `sRGB` | Use builtin sRGB gamma functions |
| `Linear` | Skip gamma decode (for normal maps, data textures) |
| `Auto` (default) | Check ICC profile; use moxcms if non-sRGB |
| `ICC` | Always use embedded ICC profile via moxcms |

In `Auto` mode, the ICC profile is compared against sRGB characteristics. If it differs significantly, the image is transformed through XYZ to linear sRGB using moxcms.

---

## 2. Path Selection

The pipeline chooses between two processing paths based on the operations required:

### Linear RGB Path

Used when any of these conditions apply:
- Reference image provided (color correction needed)
- Resize operation requested
- Grayscale output selected
- Non-sRGB ICC profile detected

This path converts to linear RGB (0–1 float), performs all processing, then converts back to sRGB.

### sRGB Direct Path

Used when all of these are true:
- No color correction (no reference image)
- No resize needed
- RGB output (not grayscale)
- Standard sRGB input (no special ICC profile)

This path converts directly to sRGB float (0–255 range) without gamma decode, avoiding unnecessary round-trips through linear space.

```
sRGB Direct Path:
  Input (u8/u16) → f32 [0-255] sRGB → Dither → Output

Linear Path:
  Input → f32 [0-1] linear → Process → sRGB encode → f32 [0-255] → Dither → Output
```

The sRGB direct path preserves full precision for dither-only operations. Even though the range is 0–255, the f32 representation maintains sub-integer precision that participates in error diffusion.

---

## 3. Linear Processing Stages

When the linear path is taken, operations occur in this order:

```
Linear RGB → Resize → Color Correct → Grayscale → sRGB Encode → Dither
```

### 3.1 Resize

Resizing is performed in linear RGB space for physically correct color blending.

**Why linear?** The sRGB gamma curve breaks additive light mixing. Interpolating in sRGB causes darkening and color shifts because the math assumes values are proportional to light intensity—but sRGB values are not.

Available methods:
- **Lanczos3** (default): High quality, good for significant scaling
- **Bilinear**: Faster, adequate for moderate scaling

Aspect ratio preservation uses a primary dimension (width or height) with the other calculated to maintain the ratio.

### 3.2 Color Correction

Color correction matches the input image's color distribution to a reference image using histogram matching. All correction happens in linear RGB space.

See [Section 4: Color Correction Methods](#4-color-correction-methods) for algorithm details.

### 3.3 Grayscale Conversion

If grayscale output is requested, luminance is extracted using Rec.709 coefficients:

```
Y = 0.2126 R + 0.7152 G + 0.0722 B
```

This is done in linear space where the coefficients are physically meaningful (they represent the eye's sensitivity to each primary).

### 3.4 sRGB Encode

After linear processing, values are converted back to sRGB using the standard transfer function:

```
if linear ≤ 0.0031308:
    srgb = 12.92 × linear
else:
    srgb = 1.055 × linear^(1/2.4) − 0.055
```

Values are then scaled to the 0–255 range for dithering.

---

## 4. Color Correction Methods

### 4.1 Basic Histogram Matching

Per-channel histogram matching in LAB, RGB, or OKLab space. Each channel's cumulative distribution is matched independently to the reference.

**Limitation**: Treating channels independently can break color relationships—a pixel might receive L from one source region, a* from another, and b* from a third.

### 4.2 Chroma Rotation Averaging (CRA)

CRA mitigates color flips from per-channel matching by considering multiple rotations of the color plane:

1. Rotate the chroma plane (a*b* in LAB, or around the gray axis in RGB)
2. Perform histogram matching at each rotation
3. Rotate results back to the original orientation
4. Average the results

**Rotations used**:
- LAB CRA: 0°, 30°, 60° (rotates AB plane only)
- RGB CRA: 0°, 40°, 80° (rotates around [1,1,1] axis using Rodrigues' formula)

**Iterative application**: Results are blended with increasing strength (25% → 50% → 100%) for stability.

**Why it works**: Color relationships that are consistent across rotations reinforce each other, while per-channel artifacts (which differ at each rotation) average out. This is analogous to tomographic reconstruction—multiple projections recover structure better than any single projection.

### 4.3 Tiled CRA

For images with spatially varying color casts (mixed lighting, gradients), tiled processing provides localized correction:

1. **Block generation**: Create a 9×9 grid of tiles, form 64 overlapping 2×2-tile blocks (50% overlap)
2. **Per-block CRA**: Each block undergoes CRA correction matched against the corresponding reference region
3. **Hamming window blending**: Results are accumulated using Hamming window weights for smooth transitions
4. **Global histogram match**: Final global match ensures overall distribution matches reference

| Channel | Default | With `--tiled-luminosity` |
|---------|---------|---------------------------|
| A, B (chroma) | Per-block CRA → global | Per-block CRA → global |
| L (luminosity) | Original → global | Per-block → global |

### 4.4 Histogram Modes

Three modes for the histogram matching step itself:

| Mode | Description |
|------|-------------|
| Binned (256 bins) | Classic approach; requires dithering to reduce quantization color shift |
| F32 Endpoint-aligned | No quantization; preserves extremes |
| F32 Midpoint-aligned | No quantization; statistically correct |

When using binned mode, a separate dithering pass distributes the binning error spatially rather than allowing systematic color shift.

---

## 5. Dithering and Quantization

### 5.1 Error Diffusion Methods

| Method | Kernel Size | Characteristics |
|--------|-------------|-----------------|
| Floyd-Steinberg | 4 pixels | Classic, fast |
| Jarvis-Judice-Ninke | 12 pixels | Smoother gradients, slower |
| Mixed | Varies | Randomly selects FS or JJN per-pixel |

**Scan patterns**:
- Standard: Left-to-right, top-to-bottom
- Serpentine: Alternating direction each row
- Random: Random direction per row (Mixed mode only)

The **Mixed** mode breaks up periodic patterns that can appear with fixed kernels.

### 5.2 Colorspace-Aware Dithering

Instead of dithering each channel independently in sRGB (the common approach), colorspace-aware dithering:

1. Generates candidate quantized values for the current pixel
2. Measures perceptual distance to the original color (in OKLab, CIELAB, etc.)
3. Selects the candidate closest perceptually
4. Diffuses error in linear RGB space (physically correct error propagation)

The per-channel approach simply quantizes each channel to its nearest value and diffuses error in sRGB space. This is faster but can produce visible color shifts since sRGB is neither perceptually uniform nor physically linear—it doesn't model how the eye perceives color differences or how light actually mixes.

**Perceptual spaces** (for distance measurement):
- **OKLab** (default, recommended): Modern, perceptually uniform—outperforms all CIELAB variants in practice
- **CIELAB + CIE76**: Simple Euclidean in Lab
- **CIELAB + CIE94**: Weighted distance
- **CIELAB + CIEDE2000**: Complex weighted formula, but still underperforms OKLab
- Linear RGB, Y'CbCr: Available but not recommended

See [COLORSPACES.md](COLORSPACES.md) for color space definitions and distance formulas.

### 5.3 Bit Depth Quantization

**Truncation** (higher → lower bit depth):
```
uint8 → uint3: value >> 5
uint8 → uint4: value >> 4
```

**Extension** (lower → higher bit depth):
```
uint3 ABC → uint8 ABCABCAB  (bit replication)
```

Truncation is the exact inverse of extension—no rounding is used. See [BITDEPTH.md](BITDEPTH.md) for the mathematical foundation.

---

## 6. Output Encoding

### 6.1 Supported Formats

**RGB formats**: Configurable bits per channel
- RGB332 (8 bpp), RGB565 (16 bpp), RGB888 (24 bpp), etc.

**Grayscale formats**: Configurable bits
- L1, L2, L4, L8

### 6.2 Binary Output Options

| Option | Description |
|--------|-------------|
| Packed | Bits packed contiguously across rows |
| Row-aligned stride | Each row padded to power-of-2 boundary (2–128 bytes) |
| Per-channel | Separate files for R, G, B channels |

Stride padding can be filled with black (zeros) or by repeating the last pixel.

---

## 7. Pipeline Diagram

```
                              Input Image
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              [Has ICC?]                    [sRGB assumed]
                    │                             │
                    ▼                             ▼
             ICC Transform                  sRGB Decode
             (via moxcms)                   (gamma 2.4)
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                         Linear RGB [0-1 float]
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
         [Resize?]          [Color Correct?]     [Grayscale?]
              │                    │                    │
              ▼                    ▼                    ▼
        Lanczos/Bilinear    CRA Histogram          Luminance
        (linear space)       Matching           (Y = Rec.709)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                         sRGB Encode (gamma)
                                   │
                                   ▼
                         Denormalize (×255)
                                   │
                                   ▼
                      Error Diffusion Dithering
                       (FS / JJN / Mixed kernel)
                                   │
                      ┌────────────┴────────────┐
                      │                         │
               [Colorspace-Aware]          [Per-Channel]
               Perceptual candidate       Nearest per channel
               Error diffused in linear   Error diffused in sRGB
                      │                         │
                      └────────────┬────────────┘
                                   │
                                   ▼
                         Quantized Output
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
                  PNG         Raw Binary    Metadata JSON
```

---

## 8. Known Limitations

- **Reference image profile**: Currently assumes sRGB. Should use the same ICC detection as input images. (Bug to fix)
