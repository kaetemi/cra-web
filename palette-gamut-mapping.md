# Palette Gamut Mapping with Ghost Entries

This document explains the ghost entry technique used in palette-based dithering to prevent error diffusion from escaping the representable color gamut.

## The Problem

When dithering an image to a limited palette, error diffusion algorithms (Floyd-Steinberg, Jarvis-Judice-Ninke, etc.) propagate quantization error to neighboring pixels. This works well when the target color space is a regular grid (like RGB565 or RGB332), but causes problems with arbitrary palettes.

### Why Arbitrary Palettes Are Difficult

Consider a 16-color CGA palette. The representable colors form an irregular polyhedron in RGB space - the **convex hull** of the palette colors. Any color that can be approximated through dithering must lie within this hull.

The problem occurs at the hull boundary:

```
           Hull boundary
               │
    Interior   │   Outside (unreachable)
    color ●────┼────→ Error vector
               │
```

When a pixel color lies on or near the hull boundary, the nearest palette color is often an **interior** color. The quantization error then points **outward** - away from the chosen color, through the hull surface, into unreachable color space.

This error gets diffused to neighboring pixels, pushing them outside the hull. They in turn get quantized to interior colors with even larger outward errors. The result is a cascade of errors that:

1. Causes visible color banding at gamut boundaries
2. Produces incorrect average colors in smooth gradients
3. Creates artifacts where the image "wants" to be a color the palette can't represent

## The Solution: Ghost Entries

The solution is to ensure that colors near hull boundaries get matched to other boundary colors, keeping the error vector **tangent** to the hull surface rather than pointing through it.

### Convex Hull Computation

First, we compute the convex hull of the palette in linear RGB space. The hull is represented as a set of planes, each defined by a point and an outward-facing normal vector.

```
Hull plane: { point: [r, g, b], normal: [nx, ny, nz] }
```

A color is inside the hull if its signed distance to every plane is ≤ 0 (or within epsilon).

### Ghost Entry Generation

For each real palette entry, we project it onto every hull face:

```
Real entry ●
            \
             \  projection
              \
               ◐ Ghost entry (on hull surface)
              ╱
    ─────────╱─────── Hull face
```

If the projection point doesn't already have a palette entry (within epsilon), we create a **ghost entry** at that location. Each ghost knows which hull surface it belongs to.

### Surface Membership Tracking

We also track which **real** palette entries lie on each hull surface. A real entry is "on" a surface if its signed distance to that plane is within epsilon.

This gives us two things for each hull face:
- Ghost entries (projected points)
- Real entries (original palette colors on that face)

### The Search Algorithm

When finding the nearest palette color for a pixel:

1. **Initial search**: Find the nearest entry (real or ghost) using perceptual distance in the full extended palette.

2. **Ghost redirection**: If the nearest entry is a ghost:
   - Note which hull surface it belongs to
   - Redo the search, but only among **real** entries on that same surface
   - Return the nearest real entry from that surface

3. **Result**: Always a real palette entry, but one that lies on the same hull face as the original match.

### Why This Works

```
Before (without ghosts):
                    Hull boundary
                         │
    Palette color A ●────┼────→ Large outward error
    (interior)           │
                         │
                    ○ Input color (on boundary)


After (with ghosts):
                    Hull boundary
                         │
    Palette color B ●────│
    (on boundary)        │← Small tangent error
                         │
                    ◐ Ghost → redirects to B
                         │
                    ○ Input color
```

The ghost entry "catches" boundary colors and redirects them to real palette entries that are also on the boundary. The quantization error stays parallel to the hull surface instead of pointing through it.

### Input Color Clamping

As a second defense, input colors outside the hull are clamped before processing:

1. Check if the color is inside the hull
2. If outside, iteratively project onto the most-violated plane
3. Repeat until inside (typically 1-3 iterations)
4. Final clamp to valid RGB range [0, 1]

This ensures the dithering algorithm never tries to represent colors that are fundamentally outside the palette's gamut.

## Implementation Details

### Epsilon Value

We use `epsilon = 1.0 / 65535.0` (one unit in 16-bit color space) for all comparisons. This provides:
- Robustness against floating-point errors
- Sub-perceptual precision (invisible differences)
- Consistent behavior across different palette sizes

### Perceptual Distance

All distance calculations use perceptual color space (OkLab by default) rather than linear RGB. This ensures the "nearest" color is perceptually nearest, not just numerically nearest.

### Memory Overhead

Ghost entries add some memory overhead:
- Each ghost stores: linear RGB (12 bytes) + plane index (8 bytes) + perceptual coords (12 bytes) = 32 bytes
- Worst case: O(palette_size × hull_faces) ghosts
- Typical case: Much fewer, as many projections land on existing entries

For a 256-color palette with ~100 hull faces, worst case is ~25K ghosts × 32 bytes = 800KB. In practice, it's usually much less.

## Results

With ghost entries enabled:
- Smooth gradients remain smooth at gamut boundaries
- No color banding from error accumulation
- Correct average colors in dithered regions
- Stable behavior regardless of palette geometry

The technique is especially important for:
- Small palettes (CGA, EGA, NES)
- Palettes with irregular shapes (artist-selected colors)
- Images with smooth gradients near gamut boundaries
- High-quality dithering where artifacts are unacceptable

## References

- Convex hull: Quickhull algorithm (Barber et al., 1996)
- Error diffusion: Floyd-Steinberg (1976), Jarvis-Judice-Ninke (1976)
- Perceptual color: OkLab (Ottosson, 2020)
