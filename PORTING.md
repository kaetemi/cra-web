# Histogram Matching and Lab Conversion Reference

## Overview

This document covers two core algorithms needed for the color correction pipeline:

1. **Histogram Matching** — transforms an image's intensity distribution to match a reference image
2. **Linear RGB ↔ Lab Conversion** — converts between linear RGB and perceptually uniform Lab color space

Both operate on individual channels independently.

---

## 1. Histogram Matching (uint8)

### Purpose

Histogram matching adjusts pixel values so that the cumulative distribution function (CDF) of the source image matches the CDF of a reference image. This transfers the tonal characteristics (contrast, brightness distribution) from reference to source.

### How It Works

The algorithm builds a lookup table by:
1. Computing the CDF of both images (cumulative histogram, normalized to [0,1])
2. For each source intensity level, finding the reference intensity with the closest CDF value
3. Applying this mapping to every pixel

For uint8 images, this is efficient because there are only 256 possible values, so we can use direct array indexing rather than sorting.

### Pseudocode

```
function match_histogram(source: [u8], reference: [u8]) -> [u8]:
    // Step 1: Count occurrences of each value (0-255)
    src_counts = [0u64; 256]
    ref_counts = [0u64; 256]
    
    for pixel in source:
        src_counts[pixel] += 1
    for pixel in reference:
        ref_counts[pixel] += 1
    
    // Step 2: Compute normalized CDFs
    src_cdf = [0.0f32; 256]
    ref_cdf = [0.0f32; 256]
    
    src_total = source.len() as f32
    ref_total = reference.len() as f32
    
    cumsum = 0
    for i in 0..256:
        cumsum += src_counts[i]
        src_cdf[i] = cumsum as f32 / src_total
    
    cumsum = 0
    for i in 0..256:
        cumsum += ref_counts[i]
        ref_cdf[i] = cumsum as f32 / ref_total
    
    // Step 3: Build lookup table via interpolation
    // For each source CDF value, find corresponding reference intensity
    lookup = [0u8; 256]
    
    // Collect reference values that actually appear (non-zero counts)
    ref_values = []
    ref_quantiles = []
    for i in 0..256:
        if ref_counts[i] > 0:
            ref_values.push(i)
            ref_quantiles.push(ref_cdf[i])
    
    // Interpolate: map src_cdf[i] -> reference intensity
    for i in 0..256:
        q = src_cdf[i]
        // Linear interpolation in ref_quantiles -> ref_values
        lookup[i] = interpolate(q, ref_quantiles, ref_values)
    
    // Step 4: Apply lookup table
    output = []
    for pixel in source:
        output.push(lookup[pixel])
    
    return output


function interpolate(x: f32, xp: [f32], fp: [u8]) -> u8:
    // Linear interpolation: find where x falls in xp, interpolate fp
    // xp must be monotonically increasing
    
    if x <= xp[0]:
        return fp[0]
    if x >= xp[xp.len()-1]:
        return fp[fp.len()-1]
    
    // Binary search for interval
    i = binary_search(xp, x)  // largest index where xp[i] <= x
    
    // Linear interpolation between fp[i] and fp[i+1]
    t = (x - xp[i]) / (xp[i+1] - xp[i])
    result = fp[i] as f32 + t * (fp[i+1] - fp[i]) as f32
    
    return round(result) as u8
```

### Multi-Channel Images

For multi-channel images (e.g., Lab with 3 channels), apply histogram matching independently to each channel:

```
function match_histogram_multichannel(source: [[[u8; C]; W]; H], 
                                       reference: [[[u8; C]; W]; H]) -> [[[u8; C]; W]; H]:
    output = allocate same shape as source
    
    for channel in 0..C:
        src_channel = extract_channel(source, channel)
        ref_channel = extract_channel(reference, channel)
        matched = match_histogram(src_channel, ref_channel)
        set_channel(output, channel, matched)
    
    return output
```

---

## 2. Linear RGB ↔ Lab Conversion

### Purpose

Lab (CIE L\*a\*b\*) is a perceptually uniform color space where:
- **L** (lightness): 0 = black, 100 = white
- **a**: green (−) to magenta (+), roughly ±127
- **b**: blue (−) to yellow (+), roughly ±127

Perceptual uniformity means equal numerical distances correspond to roughly equal perceived color differences. This makes Lab ideal for color manipulation—histogram matching in Lab produces more natural results than in RGB.

### Why Linear RGB?

Standard images (JPEG, PNG) use sRGB, which has gamma encoding for display. Lab conversion requires *linear* RGB (light intensity proportional to pixel value). Our script handles sRGB↔linear conversion separately, so the Lab routines assume linear RGB input/output in [0, 1] range.

### Constants

```
// RGB to XYZ matrix (sRGB/Rec.709 primaries, D65 illuminant)
RGB_TO_XYZ = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227]
]

// XYZ to RGB matrix (inverse of above)
XYZ_TO_RGB = [
    [ 3.240479, -1.537150, -0.498535],
    [-0.969256,  1.875991,  0.041556],
    [ 0.055648, -0.204043,  1.057311]
]

// D65 white point
X_N = 0.950456
Y_N = 1.0
Z_N = 1.088754

// Lab threshold: (6/29)^3
EPSILON = 0.008856

// Lab linear segment slope: (29/6)^2 / 3
KAPPA = 903.3  // Used for L calculation
KAPPA_INV = 7.787  // Used for f(t) linear segment
```

### Linear RGB → Lab

```
function linear_rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32):
    // Input: linear RGB, each channel in [0, 1]
    
    // Step 1: RGB -> XYZ, normalized by white point
    x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / X_N
    y = (0.212671 * r + 0.715160 * g + 0.072169 * b) / Y_N  // Y_N = 1.0
    z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / Z_N
    
    // Step 2: Apply f(t) - attempt to linearize perceptual response
    fx = lab_f(x)
    fy = lab_f(y)
    fz = lab_f(z)
    
    // Step 3: XYZ -> Lab
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return (L, a, b)


function lab_f(t: f32) -> f32:
    // Attempt to linearize cube root near zero
    if t > EPSILON:
        return cbrt(t)  // t^(1/3)
    else:
        return KAPPA_INV * t + 16.0/116.0  // 7.787*t + 0.137931
```

### Lab → Linear RGB

```
function lab_to_linear_rgb(L: f32, a: f32, b: f32) -> (f32, f32, f32):
    // Step 1: Lab -> f values
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    // Step 2: Invert f(t) to get XYZ (normalized)
    x = lab_f_inv(fx)
    y = lab_f_inv(fy)
    z = lab_f_inv(fz)
    
    // Step 3: Denormalize by white point
    x = x * X_N
    y = y * Y_N  // Y_N = 1.0
    z = z * Z_N
    
    // Step 4: XYZ -> linear RGB
    r =  3.240479 * x - 1.537150 * y - 0.498535 * z
    g = -0.969256 * x + 1.875991 * y + 0.041556 * z
    b =  0.055648 * x - 0.204043 * y + 1.057311 * z
    
    return (r, g, b)


function lab_f_inv(t: f32) -> f32:
    // Inverse of lab_f
    // Threshold in f-space: f(EPSILON) ≈ 0.206893
    if t > 0.206893:
        return t * t * t  // t^3
    else:
        return (t - 16.0/116.0) / KAPPA_INV  // (t - 0.137931) / 7.787
```

### Image Conversion

```
function convert_image_rgb_to_lab(rgb: [[[f32; 3]; W]; H]) -> [[[f32; 3]; W]; H]:
    lab = allocate [[[f32; 3]; W]; H]
    
    for y in 0..H:
        for x in 0..W:
            (r, g, b) = rgb[y][x]
            (L, a, b_ch) = linear_rgb_to_lab(r, g, b)
            lab[y][x] = [L, a, b_ch]
    
    return lab
```
