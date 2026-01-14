# Lanczos Rescaling Implementation

This document describes the Lanczos3 image rescaling implementation in `port/src/rescale.rs`.

## Lanczos Kernel

The Lanczos kernel is a windowed sinc function. For Lanczos3 (a=3):

```
L(x) = sinc(x) * sinc(x/3)    for |x| < 3
L(x) = 0                       for |x| >= 3
```

Where `sinc(x) = sin(πx) / (πx)` and `sinc(0) = 1`.

This kernel provides high-quality resampling with good sharpness and minimal ringing compared to other interpolation methods.

## Separable Filtering

2D Lanczos filtering is separable, meaning we can decompose it into two 1D passes:

1. **Horizontal pass**: Resample each row independently (width transformation)
2. **Vertical pass**: Resample each column of the intermediate result (height transformation)

This reduces complexity from O(r²) per pixel to O(2r) per pixel, where r is the kernel radius.

```
Source (W×H) → Horizontal → Intermediate (W'×H) → Vertical → Output (W'×H')
```

## Adaptive Filter Radius

For downscaling, the filter radius must increase to properly integrate all source pixels and avoid aliasing:

```
filter_scale = max(src_size / dst_size, 1.0)
radius = ceil(3.0 * filter_scale)
```

| Scale Factor | Filter Scale | Radius | Kernel Width |
|--------------|--------------|--------|--------------|
| 0.5× (upscale) | 1.0 | 3 | 7 |
| 1× (identity) | 1.0 | 3 | 7 |
| 2× downscale | 2.0 | 6 | 13 |
| 4× downscale | 4.0 | 12 | 25 |
| 15× downscale | 15.0 | 45 | 91 |

## Kernel Weight Precomputation

For each destination position, the kernel weights depend only on:
- The destination index
- The scale factor
- The filter radius

They do NOT depend on actual pixel values. This means we can precompute weights once per destination coordinate and reuse them across all rows (horizontal) or columns (vertical).

### Data Structure

```rust
struct KernelWeights {
    start_idx: usize,    // First source index to sample
    weights: Vec<f32>,   // Normalized weights (sum to 1.0)
    fallback_idx: usize, // Fallback for empty weights (edge case)
}
```

### Precomputation

For destination index `dst_i`:

```
src_pos = (dst_i + 0.5) * scale - 0.5
center = floor(src_pos)
start = max(center - radius, 0)
end = min(center + radius, src_len - 1)

For each si in start..=end:
    weight[si - start] = lanczos3((src_pos - si) / filter_scale)

Normalize: weights /= sum(weights)
```

### Application

```rust
for (dst_i, kernel) in kernels.iter().enumerate() {
    let mut sum = 0.0;
    for (i, &weight) in kernel.weights.iter().enumerate() {
        sum += src[kernel.start_idx + i] * weight;
    }
    dst[dst_i] = sum;
}
```

## Memory and Computation Trade-off

For a W×H → W'×H' resize:

**Without precomputation:**
- Kernel evaluations: W' × H' × (kernel_width_h + kernel_width_v)
- For 4K→256: 36,864 × ~182 = ~6.7M `lanczos3()` calls

**With precomputation:**
- Kernel evaluations: W' + H'
- For 4K→256: 256 + 144 = 400 `lanczos3()` calls
- Memory: (W' + H') × avg_kernel_width × 4 bytes

The precomputation reduces `lanczos3()` evaluations by ~16,000× for this example.

## Coordinate Mapping

Pixel centers are at half-integer coordinates. The mapping from destination to source:

```
mapped_src_len = dst_len * scale
offset = (src_len - mapped_src_len) / 2
src_pos = (dst_i + 0.5) * scale - 0.5 + offset
```

This ensures:
- Destination pixel 0's center (0.5) maps to source position `0.5 * scale - 0.5 + offset`
- Edge pixels are handled correctly without shift artifacts

### Centering for Uniform Scaling

When using uniform scale modes (UniformWidth/UniformHeight), one dimension may not match its natural scale. The offset centers the mapping:

- **Independent scaling**: `scale = src_len/dst_len`, so `offset = 0`
- **Uniform scaling**: `scale` may differ from `src_len/dst_len`, so `offset ≠ 0`

Example: 100×50 source → 200×200 target with UniformWidth (scale=0.5):
- Horizontal: `offset = (100 - 200*0.5) / 2 = 0` (exact fit)
- Vertical: `offset = (50 - 200*0.5) / 2 = -25` (centered, extends 25 pixels past each edge)

When `src_pos` falls outside `[0, src_len-1]`, the kernel is clipped to valid indices and weights are renormalized. This effectively clamps to edge pixels.

## Weight Normalization

Weights are normalized to sum to 1.0 to ensure:
- Constant regions remain constant (no brightness shift)
- Energy preservation across the image

At image boundaries where the kernel is clipped, normalization compensates for the missing samples.

## Out-of-Gamut Values

The implementation does NOT clamp intermediate values. Lanczos ringing can produce values outside [0,1], but these are valid intermediate results in a linear RGB pipeline. Final quantization/dithering handles the conversion to integer output.

## Color Space

Rescaling should be performed in **linear RGB** space, not gamma-encoded sRGB. Interpolating in sRGB produces incorrect results (typically darker midtones and color shifts). The typical pipeline is:

```
sRGB input → linearize → rescale → gamma encode → dither → output
```

See COLORSPACES.md for details on linear vs sRGB processing.
