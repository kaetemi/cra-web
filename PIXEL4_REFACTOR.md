# Pixel4 ([f32; 4]) Refactoring Checklist

This document tracks the refactoring of the codebase to use SIMD-friendly `[f32; 4]` pixel format throughout the processing pipeline.

## Goals
- Use `Pixel4` ([f32; 4]) for all internal RGB/color processing
- Eliminate unnecessary format conversions (interleaved ↔ separate channels)
- Enable in-place operations where possible
- Improve cache locality and SIMD vectorization potential
- Keep single-channel (`f32`) API for grayscale operations

## Architecture

```
WASM Input (u8 RGB/RGBA)
    ↓
Convert once to Vec<Pixel4>
    ↓
[All processing in Pixel4 space - in-place where possible]
    ↓
Convert once to output format
    ↓
WASM Output (u8 RGB/RGBA or f32)
```

---

## Phase 1: Core Infrastructure ✅

- [x] Create `pixel.rs` module with `Pixel4` type
- [x] Add conversion utilities (channels ↔ pixels, interleaved ↔ pixels)
- [x] Add arithmetic operations (lerp, add, sub, mul, scale, clamp)
- [x] Add map functions (map_pixels, map_rgb)
- [x] Add u8 conversion utilities (srgb_u8_to_pixels, pixels_to_srgb_u8)

---

## Phase 2: Rescaling ✅

- [x] Add `rescale_pixels()` for Pixel4 arrays
- [x] Add `rescale_bilinear_pixels()` internal function
- [x] Add `rescale_lanczos3_pixels()` internal function
- [x] Keep `rescale_channel()` for grayscale (single channel)
- [x] Add WASM exports: `rescale_srgb_pixel4_wasm`, `rescale_linear_pixel4_wasm`, `rescale_channel_wasm`

---

## Phase 3: Color Conversion Functions ✅

### color.rs - Core Conversions

- [x] Add `srgb_to_linear_pixel4(p: Pixel4) -> Pixel4`
- [x] Add `linear_to_srgb_pixel4(p: Pixel4) -> Pixel4`
- [x] Add `srgb_to_linear_pixels_inplace(pixels: &mut [Pixel4])`
- [x] Add `linear_to_srgb_pixels_inplace(pixels: &mut [Pixel4])`
- [x] Add `linear_rgb_to_lab_pixel4(p: Pixel4) -> Pixel4` (L,a,b in channels 0,1,2)
- [x] Add `lab_to_linear_rgb_pixel4(p: Pixel4) -> Pixel4`
- [x] Add `linear_rgb_to_oklab_pixel4(p: Pixel4) -> Pixel4`
- [x] Add `oklab_to_linear_rgb_pixel4(p: Pixel4) -> Pixel4`
- [x] Add `linear_rgb_to_lab_pixels_inplace()` and `lab_to_linear_rgb_pixels_inplace()`
- [x] Add `linear_rgb_to_oklab_pixels_inplace()` and `oklab_to_linear_rgb_pixels_inplace()`
- [ ] Deprecate `*_channels()` functions (keep for backward compat temporarily)

### Scale Functions (in-place)

- [x] Add `scale_to_255_pixels_inplace(pixels: &mut [Pixel4])`
- [x] Add `scale_from_255_pixels_inplace(pixels: &mut [Pixel4])`
- [x] Add `linear_to_srgb_255_pixel4()` / `srgb_255_to_linear_pixel4()` combined functions
- [x] Add `linear_to_srgb_255_pixels_inplace()` / `srgb_255_to_linear_pixels_inplace()`

---

## Phase 4: Output/Finalization ✅

### output.rs

- [x] Add `finalize_pixels_to_srgb_u8(pixels: &mut [Pixel4]) -> Vec<u8>`
- [x] Add `finalize_pixels_to_srgb_u8_dithered(pixels: &mut [Pixel4], ...) -> Vec<u8>`
- [x] Add `finalize_pixels_to_srgb_u8_color_aware(pixels: &mut [Pixel4], ...) -> Vec<u8>`
- [x] Add `finalize_pixels_to_srgb_u8_with_options(pixels: &mut [Pixel4], ...) -> Vec<u8>`
- [x] Integrate with colorspace-aware dithering (already uses joint processing!)

---

## Phase 5: Basic Color Correction ✅

Refactored in-place to use Pixel4 at API boundaries (no duplicate functions).

### basic_lab.rs

- [x] `color_correct_basic_lab_linear(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`
- [x] Internal: convert to Lab channels for histogram matching, back to Pixel4 on return
- [x] Uses `linear_rgb_to_lab_pixel4()` / `lab_to_linear_rgb_pixel4()` for conversion

### basic_rgb.rs

- [x] `color_correct_basic_rgb_linear(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`

### basic_oklab.rs

- [x] `color_correct_basic_oklab_linear(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`
- [x] Uses `linear_rgb_to_oklab_pixel4()` / `oklab_to_linear_rgb_pixel4()` for conversion

---

## Phase 6: CRA Color Correction ✅

Refactored in-place to use Pixel4 at API boundaries.

### cra_lab.rs

- [x] `color_correct_cra_linear(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`
- [x] `color_correct_cra_lab_linear(...)` - convenience wrapper for CIELAB
- [x] `color_correct_cra_oklab_linear(...)` - convenience wrapper for OkLab
- [x] Added `pixels_to_lab_channels()` / `lab_channels_to_pixels()` helpers
- [x] Internal rotation processing still uses separate channels (required for histogram ops)

### cra_rgb.rs

- [x] `color_correct_cra_rgb_linear(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`
- [x] Converts Pixel4 to interleaved RGB for rotation, back to Pixel4 on return

---

## Phase 7: Tiled Processing ✅

Refactored in-place to use Pixel4 at API boundaries.

### tiled_lab.rs

- [x] `color_correct_tiled_linear(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`
- [x] `color_correct_tiled_lab_linear(...)` - convenience wrapper for CIELAB
- [x] `color_correct_tiled_oklab_linear(...)` - convenience wrapper for OkLab
- [x] Added `pixels_to_lab_channels()` / `lab_channels_to_pixels()` helpers
- [x] Per-tile format conversions still occur (histogram matching requires channels)

---

## Phase 8: Histogram Matching (Deferred)

### histogram.rs

- [ ] Consider joint-channel histogram operations
- [x] Current approach: extract channels from Pixel4 only when needed for histogram LUT building
- Note: Histogram matching inherently operates per-channel, so separate channels are required

---

## Phase 9: Dithering (No Changes Needed)

### dither.rs

- [x] `dither_with_mode` works with separate channels (used internally by histogram prep)
- [x] `colorspace_aware_dither_rgb_with_mode` handles joint RGB processing

### dither_colorspace_aware.rs

- [x] Already does joint RGB processing with perceptual color distance
- [x] Used by `output::finalize_pixels_to_srgb_u8_color_aware()` for Pixel4 output

---

## Phase 10: WASM Exports (lib.rs) ✅

Updated existing WASM exports to use Pixel4 internally (no new function names needed).

- [x] `color_correct_basic_lab()` - uses `pixel::srgb_u8_to_pixels()` + Pixel4 processing
- [x] `color_correct_basic_rgb()` - uses Pixel4 internally
- [x] `color_correct_basic_oklab()` - uses Pixel4 internally
- [x] `color_correct_cra_lab()` - uses Pixel4 internally
- [x] `color_correct_cra_rgb()` - uses Pixel4 internally
- [x] `color_correct_cra_oklab()` - uses Pixel4 internally
- [x] `color_correct_tiled_lab()` - uses Pixel4 internally
- [x] `color_correct_tiled_oklab()` - uses Pixel4 internally
- [x] All use `output::finalize_pixels_to_srgb_u8_*()` for final conversion

---

## Phase 11: CLI Updates (bin/cra.rs) ✅

- [x] `load_image_linear()` returns `(Vec<Pixel4>, u32, u32)`
- [x] `resize_linear()` uses Pixel4 throughout
- [x] All color correction calls use Pixel4 APIs directly
- [x] `linear_pixels_to_grayscale()` takes `&[Pixel4]`
- [x] Output path uses `finalize_pixels_to_srgb_u8_with_options()`
- [x] Removed temporary `pixels_to_channels` / `channels_to_pixels` workaround

---

## Phase 12: Web Demo Updates

- [ ] Update dither.html to use new Pixel4 WASM APIs (if beneficial)
- [ ] Test performance improvements
- Note: WASM API signatures unchanged, so no JS changes required

---

## Cleanup

- [x] Refactored functions in-place (no deprecated functions to remove)
- [ ] Remove unused `*_channels()` conversion utilities from `color.rs` (optional)
- [x] Updated documentation (this file)

---

## Current Status: Core Refactoring Complete ✅

All color correction algorithms now use Pixel4 at their API boundaries:
- **Input**: `&[Pixel4]` (linear RGB)
- **Output**: `Vec<Pixel4>` (linear RGB)
- **Internal**: May use separate channels for histogram operations (inherent requirement)

The WASM layer handles u8 ↔ Pixel4 conversion, so JavaScript API unchanged.

---

## Performance Targets

For 1000×1000 image:
- **Before**: ~30M unnecessary element copies, 10-15 intermediate vectors
- **After**: ~3M element copies (input/output only), 2-3 working buffers

---

## Notes

### Data Format Convention
- `Pixel4` = `[f32; 4]` = `[R, G, B, _]` where `_` is padding (typically 0.0)
- For Lab/OkLab: `[L, a, b, _]`
- Range: 0.0-1.0 for linear, 0.0-255.0 for sRGB/scaled

### Backward Compatibility
- WASM exports unchanged (internal refactoring only)
- Rust library API changed from separate channels to Pixel4
- CLI updated to use new Pixel4 APIs

### Testing
- All 142 existing tests pass
- Library and CLI build successfully
- Output behavior unchanged (same algorithms, different internal format)

### Refactoring Approach
Instead of adding duplicate `*_pixel4_*` functions, we refactored existing functions in-place:
- Changed function signatures from `(r: &[f32], g: &[f32], b: &[f32])` to `(input: &[Pixel4])`
- Changed return types from `(Vec<f32>, Vec<f32>, Vec<f32>)` to `Vec<Pixel4>`
- Added helper functions like `pixels_to_lab_channels()` for internal conversion
- This keeps the API surface small while achieving the Pixel4 benefits
