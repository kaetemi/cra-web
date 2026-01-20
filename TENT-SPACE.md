# Tent-Space Resampling

This module implements a volume-preserving transformation between two image representations:

- **Box space**: Traditional pixels. Each value represents the integral of light over a unit square. This is what sensors capture and displays emit.

- **Tent space**: A piecewise bilinear surface. The image is represented as a continuous function that can be sampled or integrated anywhere.

## Why Tent Space?

Traditional resampling conflates two operations:

1. Reconstructing a continuous signal from discrete samples
2. Resampling that signal at new locations

Different kernels (Lanczos, Mitchell, etc.) make different tradeoffs in how they guess what exists "between" pixels. This leads to artifacts: ringing, blurring, aliasing.

Tent space separates these concerns. The continuous surface is *defined explicitly*, not guessed. Resampling becomes pure geometry—integrating a known surface rather than reconstructing a hypothetical signal.

## The Core Insight: Eliminating Infinite Slopes

Traditional "box space" pixels are implicitly step functions—each pixel is a constant value with infinitely sharp edges at the boundaries. These discontinuities contain infinite frequency content, which is the root cause of ringing artifacts.

When you apply a sinc or Lanczos filter to step discontinuities, the infinite frequencies interact with the filter's oscillating lobes, producing Gibbs phenomenon (ringing). The entire history of resampling filter design—windowing, Mitchell-Netravali tuning, EWA—is essentially damage control for this fundamental mismatch.

Tent space eliminates the problem at the source. The bilinear surface is C0 continuous: finite slopes everywhere, no discontinuities. The highest frequencies simply don't exist in the representation.

```
Box space:     ████████        Step discontinuity
               ████████        Infinite slope at edge
               ████████        Infinite frequency content
                               → Ringing with any oscillating kernel

Tent space:      /\            Piecewise linear
                /  \           Finite slopes everywhere
               /    \          Bandlimited by construction
                               → No ringing, even with pure sinc
```

The sharpening kernel in the expansion step doesn't remove edge information—it *relocates* it. Sharp transitions are encoded as amplitude adjustments at known grid locations (the centers), not as actual discontinuities in the surface. The information is preserved but the infinite slopes are gone.

This is why kernel choice becomes nearly irrelevant for downscaling: you're integrating a well-behaved surface, not fighting discontinuities.

## The Transform: Box → Tent

Expansion maps (W, H) → (2W+1, 2H+1) using a **polyphase filter bank**—four different kernels applied based on output coordinate parity:

```
    dx even    dx odd
   ┌─────────┬─────────┐
   │ Corner  │  Edge   │  dy even
   │ (0,0)   │  (1,0)  │
   ├─────────┼─────────┤
   │  Edge   │ Center  │  dy odd
   │ (0,1)   │  (1,1)  │
   └─────────┴─────────┘
```

### Phase (even, even) — Corners

Pure 2×2 box filter (bilinear interpolation of 4 source pixels):

```
    0.25  0.25
    0.25  0.25
```

### Phase (odd, even) — Horizontal Edges

Vertical 2-tap average:

```
    0.5
    0.5
```

### Phase (even, odd) — Vertical Edges

Horizontal 2-tap average:

```
    0.5  0.5
```

### Phase (odd, odd) — Centers

**Sharpening kernel** (this is where the magic happens):

```
    -0.0625  -0.375  -0.0625
    -0.375    2.75   -0.375
    -0.0625  -0.375  -0.0625
```

The center kernel has negative lobes—it's an edge-enhancing filter! This encodes high-frequency information at known locations, ensuring that integration over any unit square recovers the original pixel value exactly.

## Volume Preservation

The center kernel is derived from the constraint that integrating the bilinear surface over each pixel's unit square must equal the original pixel value.

Integration weights for a bilinear surface over a unit square:

```
    1/16 ──(1/8)── 1/16
      │      │      │
    (1/8)──(1/4)──(1/8)     Sum = 1.0
      │      │      │
    1/16 ──(1/8)── 1/16
```

Given original value V, edge values E (interpolated), and corner values C (interpolated), we solve for center value M:

```
    V = (1/4)M + (1/8)ΣE + (1/16)ΣC
    M = 4V - (1/2)ΣE - (1/4)ΣC
```

Expanding E and C in terms of source pixels yields the 3×3 sharpening kernel.

## Properties

- **Exact roundtrip**: `tent_contract(tent_expand(img)) = img` (within float precision)

- **Localized ringing**: High-frequency "overshoot" is confined to center points, never spreads. Traditional Lanczos ringing propagates; this doesn't.

- **Reversible scaling**: Upscale then downscale ≈ identity, because you're sampling a defined surface, not inventing information.

## Resampling Behavior

### Downscaling (kernel-agnostic)

When downscaling, the output pixel integrates over a region larger than one tent-space cell. The sharpening encoded in center points gets averaged together with surrounding values, naturally producing the correct result. Kernel choice barely matters—even box filtering produces high-quality output because the surface is doing the work. Filter width stays at 1x (native), taking advantage of the higher-resolution surface.

### Upscaling (kernel-sensitive)

When upscaling, you're sampling the surface more densely than the original grid. A 1x kernel will "see" the raw center adjustments as visible ringing artifacts. Solution: scale the kernel width proportionally, up to 2x width at 2x upscale. This smooths over the encoded sharpness while preserving edge definition.

| Scale factor | Kernel width |
|--------------|--------------|
| 0.5× (down)  | 1.0×         |
| 1.0× (none)  | 1.0×         |
| 1.5× (up)    | 1.5×         |
| 2.0× (up)    | 2.0×         |

The asymmetry exists because downscaling integrates (averages away the encoded sharpness) while upscaling samples (exposes it directly). Wider kernels for upscaling smooth the traversal of the surface without destroying edge information.

### Why sinc works here

Pure sinc filtering normally causes severe ringing (Gibbs phenomenon) due to its infinite lobes interacting with discontinuities. Windowed approximations (Lanczos, etc.) exist specifically to mitigate this.

In tent space, the bilinear surface is C0 continuous—no discontinuities exist for sinc to ring against. You get sinc's ideal frequency response without its spatial artifacts. The representation has pre-smoothed the signal.

## Implementation

The polyphase structure allows two clean implementation strategies:

### Strategy 1: Two-Pass

**Pass 1 — Interpolation**: Compute all (2W+1, 2H+1) grid points using simple linear interpolation. Centers get the source value directly, edges average 2 neighbors, corners average 4 neighbors.

**Pass 2 — Adjustment**: For each center point only, apply the volume-preservation correction: `M' = 4V - 0.5*ΣE - 0.25*ΣC`. This reads from the interpolated edges and corners computed in pass 1.

```
    ┌─────────────────┐      ┌─────────────────┐
    │ Interpolate all │ ───► │ Adjust centers  │
    │ grid points     │      │ only            │
    └─────────────────┘      └─────────────────┘
```

### Strategy 2: Four Parallel Filters

Apply four separate convolutions to the source image, interleaving results by output coordinate parity:

```
    Source ──┬── Corner kernel (2×2 box) ────────► even,even outputs
             ├── H-edge kernel (2×1 average) ────► odd,even outputs
             ├── V-edge kernel (1×2 average) ────► even,odd outputs
             └── Center kernel (3×3 sharpen) ────► odd,odd outputs
```

Both strategies produce identical results. Strategy 1 is simpler and has better cache locality. Strategy 2 may parallelize better on GPU.

### Contraction (Tent → Box)

Single pass: for each output pixel, gather the 3×3 stencil from tent space and apply integration weights:

```
    output = 0.25*center + 0.125*Σedges + 0.0625*Σcorners
```

This is a standard weighted average—no special structure needed.

## Usage

**Downscaling** (the primary use case):

```
    source (box) → tent_expand → resample in tent space → tent_contract → output (box)
```

**Upscaling** (works, with wider kernel):

Scale kernel width up to 2x to smooth over the encoded sharpness. Results are sharper and smoother than traditional upscaling because edge information is explicitly encoded rather than guessed.
