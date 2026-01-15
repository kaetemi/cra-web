# Floyd-Steinberg Dithering (Perceptual Distance, Linear Error)

- **Distance metric:** OKLab (default), CIELAB (CIE76/CIE94/CIEDE2000)
- **Error accumulation:** Linear RGB
- **Quantization:** sRGB uint8

---

## Data Structures

```
QuantParams (one per channel, supports different bit depths):
  bits: u8                          // e.g., 5 for R, 6 for G, 5 for B (RGB565)
  num_levels: usize                 // 2^bits, e.g., 32
  level_values: [u8; num_levels]    // level index → extended sRGB value
                                    // e.g., for 2-bit: [0, 85, 170, 255]
  lut_floor_level: [u8; 256]        // sRGB value → floor level index
  lut_ceil_level: [u8; 256]         // sRGB value → ceil level index

linear_lut: [f32; 256]              // sRGB component → Linear RGB component

lab_lut: Vec<Lab>                   // Flat array: r_level * n² + g_level * n + b_level → Lab
                                    // Only built when all channels have same bit depth
                                    // Otherwise compute on-the-fly during dithering
```

---

## Precomputation

```
// QuantParams (one per channel)
for level in 0..num_levels:
  level_values[level] = bit_replicate(level, bits)

for v in 0..256:
  // Bit truncation to find nearby level
  trunc_idx = v >> (8 - bits)
  trunc_val = level_values[trunc_idx]

  if trunc_val == v:
    floor_idx = ceil_idx = trunc_idx
  elif trunc_val < v:
    floor_idx = trunc_idx
    ceil_idx = min(trunc_idx + 1, num_levels - 1)
  else:
    floor_idx = max(trunc_idx - 1, 0)
    ceil_idx = trunc_idx

  lut_floor_level[v] = floor_idx
  lut_ceil_level[v] = ceil_idx

// Linear LUT
for v in 0..256:
  linear_lut[v] = srgb_component_to_linear(v / 255.0)

// Lab LUT - only if all channels have same bit depth
if bits_r == bits_g == bits_b:
  for r_level in 0..num_levels:
    for g_level in 0..num_levels:
      for b_level in 0..num_levels:
        r_ext = level_values[r_level]
        g_ext = level_values[g_level]
        b_ext = level_values[b_level]

        linear = (linear_lut[r_ext], linear_lut[g_ext], linear_lut[b_ext])
        lab_lut[r_level * n² + g_level * n + b_level] = linear_rgb_to_lab(linear)
```

---

## Main Algorithm

**Input:** `(r_channel, g_channel, b_channel)` - f32 arrays, sRGB 0-255

**Output:** `(r_out, g_out, b_out)` - u8 arrays

```
error_buf = zeros(width, height, 3)  // Linear RGB, ~0-1 range

for y in 0..height:
  for x in 0..width:

    // 1. Read input, convert to Linear RGB
    srgb_in = (r_channel[x,y], g_channel[x,y], b_channel[x,y])
    linear_original = srgb_to_linear(srgb_in)

    // 2. Add accumulated error
    linear_adjusted = linear_original + error_buf[x, y]

    // 3. Convert back to sRGB for quantization bounds
    linear_clamped = clamp(linear_adjusted, 0, 1)
    srgb_adjusted = linear_to_srgb(linear_clamped)  // float, 0-255

    // 4. Get level index bounds
    r_min = lut_floor_level[floor(srgb_adjusted.r)]
    r_max = lut_ceil_level[ceil(srgb_adjusted.r)]

    g_min = lut_floor_level[floor(srgb_adjusted.g)]
    g_max = lut_ceil_level[ceil(srgb_adjusted.g)]

    b_min = lut_floor_level[floor(srgb_adjusted.b)]
    b_max = lut_ceil_level[ceil(srgb_adjusted.b)]

    // 5. Convert target to Lab (use unclamped for true distance)
    lab_target = linear_rgb_to_lab(linear_adjusted)

    // 6. Search candidates
    best_r_level = 0
    best_g_level = 0
    best_b_level = 0
    best_dist = infinity

    for r_level in r_min..=r_max:
      for g_level in g_min..=g_max:
        for b_level in b_min..=b_max:
          // Use LUT if available, otherwise compute on-the-fly
          lab_candidate = lookup_or_compute(r_level, g_level, b_level)
          dist = perceptual_distance_sq(lab_target, lab_candidate, space)
          if dist < best_dist:
            best_dist = dist
            best_r_level = r_level
            best_g_level = g_level
            best_b_level = b_level

    // 7. Get extended values for output and error calculation
    best_r = level_values[best_r_level]
    best_g = level_values[best_g_level]
    best_b = level_values[best_b_level]

    // 8. Write output
    r_out[x, y] = best_r
    g_out[x, y] = best_g
    b_out[x, y] = best_b

    // 9. Compute error in Linear RGB
    best_linear = (linear_lut[best_r], linear_lut[best_g], linear_lut[best_b])
    error = linear_adjusted - best_linear

    // 10. Diffuse error
    error_buf[x+1, y  ] += error * (7/16)
    error_buf[x-1, y+1] += error * (3/16)
    error_buf[x  , y+1] += error * (5/16)
    error_buf[x+1, y+1] += error * (1/16)
```

---

## Implementation Notes

1. **Index spaces:**
   - level index: 0..num_levels (e.g., 0..32 for 5-bit)
   - extended value: 0..255, but only num_levels distinct values are valid
   - level_values[] maps level index → extended value
   - lab_lut is indexed by level indices, NOT extended values

2. **LUT sizing** (when same bit depth for all channels):
   - lab_lut memory: num_levels³ × 12 bytes (3 × f32)
     - 5-bit: 32³ × 12 = 393 KB
     - 6-bit: 64³ × 12 = 3.1 MB
     - 8-bit: 256³ × 12 = 201 MB (falls back to on-the-fly computation)

3. **Different bit depths per channel:**
   - Supports asymmetric formats like RGB565 (5-bit R, 6-bit G, 5-bit B)
   - LUT cannot be pre-built; candidate colors computed on-the-fly
   - Each channel has its own QuantParams

4. **Candidate count:**
   - Each range r_min..=r_max typically spans 1-2 levels
   - Total candidates: typically 1-8, worst case ~27 (3×3×3)
   - No heap allocation needed, just nested loops

5. **Gamut handling:**
   - Clamp before sRGB conversion (step 3) to get valid LUT indices
   - Use unclamped linear_adjusted for Lab conversion (step 5) to compute true perceptual distance even to out-of-gamut targets
   - Error can be negative or push values out of gamut; this is correct and will be compensated in subsequent pixels

6. **Color space conversions:**
   - `srgb_to_linear`: sRGB 0-1 → Linear RGB 0-1 (decode transfer function)
   - `linear_to_srgb`: Linear RGB 0-1 → sRGB 0-1 (encode transfer function)
   - `linear_rgb_to_oklab`: Linear RGB → OKLab (direct matrix + cube root)
   - `linear_rgb_to_lab`: Linear RGB → XYZ → CIELAB (D65 reference white)

7. **Perceptual distance functions:**
   - OKLab: Euclidean distance (L² + a² + b²) - default, recommended
   - CIE76: Simple Euclidean in CIELAB (fast but over-weights chromaticity)
   - CIE94: Weighted distance with chroma-dependent weights
   - CIEDE2000: Most accurate, with hue rotation and lightness correction

8. **Precision:**
   - error_buf should be f32, can accumulate values outside 0-1
   - lab_lut can be f32, precision is sufficient for perceptual comparison
   - floor/ceil on sRGB need to handle edge cases (0.0, 255.0)

9. **Edge handling:**
   - error_buf needs padding or bounds checks for diffusion
   - Same padding strategy as existing implementation

10. **API change from existing code:**
    - Old: `dither(channel)` per channel independently
    - New: `dither(r, g, b)` jointly - channels cannot be processed separately when using perceptual distance metric
    - Both modes still available via OutputTechnique enum
