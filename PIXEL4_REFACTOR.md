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

## Phase 5: Basic Color Correction

### basic_lab.rs

- [ ] Add `color_correct_basic_lab_pixel4(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`
- [ ] Internal: process histogram matching on Pixel4 arrays
- [ ] Avoid intermediate separate-channel vectors

### basic_rgb.rs

- [ ] Add `color_correct_basic_rgb_pixel4(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`

### basic_oklab.rs

- [ ] Add `color_correct_basic_oklab_pixel4(input: &[Pixel4], reference: &[Pixel4], ...) -> Vec<Pixel4>`

---

## Phase 6: CRA Color Correction

### cra_lab.rs

- [ ] Add `color_correct_cra_lab_pixel4(...)`
- [ ] Add `color_correct_cra_oklab_pixel4(...)`
- [ ] Refactor rotation processing to use Pixel4
- [ ] Avoid 3× channel separation per rotation

### cra_rgb.rs

- [ ] Add `color_correct_cra_rgb_pixel4(...)`
- [ ] Refactor internal processing to Pixel4

---

## Phase 7: Tiled Processing

### tiled_lab.rs

- [ ] Add `color_correct_tiled_lab_pixel4(...)`
- [ ] Add `color_correct_tiled_oklab_pixel4(...)`
- [ ] Process tiles as Pixel4 blocks
- [ ] Reduce per-tile format conversions

---

## Phase 8: Histogram Matching

### histogram.rs

- [ ] Consider joint-channel histogram operations
- [ ] Or: extract channels from Pixel4 only when needed for histogram LUT building

---

## Phase 9: Dithering

### dither.rs

- [ ] Ensure `dither_with_mode` works efficiently with Pixel4 output
- [ ] Already has `colorspace_aware_dither_rgb_with_mode` - verify it uses Pixel4 efficiently

### dither_colorspace_aware.rs

- [ ] Already does joint RGB processing - ensure it accepts Pixel4 input
- [ ] Add `colorspace_aware_dither_pixel4(...)` if needed

---

## Phase 10: WASM Exports (lib.rs)

- [ ] Add `color_correct_basic_lab_pixel4_wasm(...)`
- [ ] Add `color_correct_basic_rgb_pixel4_wasm(...)`
- [ ] Add `color_correct_basic_oklab_pixel4_wasm(...)`
- [ ] Add `color_correct_cra_lab_pixel4_wasm(...)`
- [ ] Add `color_correct_cra_rgb_pixel4_wasm(...)`
- [ ] Add `color_correct_cra_oklab_pixel4_wasm(...)`
- [ ] Add `color_correct_tiled_lab_pixel4_wasm(...)`
- [ ] Add `color_correct_tiled_oklab_pixel4_wasm(...)`
- [ ] Replace manual interleave loops with `pixel::*` utilities

---

## Phase 11: CLI Updates (bin/cra.rs)

- [ ] Refactor `load_image_linear` to return `Vec<Pixel4>`
- [ ] Refactor `resize_linear` to use `rescale_pixels`
- [ ] Update all color correction calls to use Pixel4 variants
- [ ] Update output path to use Pixel4 finalization

---

## Phase 12: Web Demo Updates

- [ ] Update dither.html to use new Pixel4 WASM APIs (if beneficial)
- [ ] Test performance improvements

---

## Cleanup

- [ ] Remove deprecated separate-channel functions (after migration complete)
- [ ] Remove unused conversion utilities
- [ ] Update documentation

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
- Keep old WASM exports working during transition
- Add new `*_pixel4_*` variants alongside existing functions
- Deprecate old functions after full migration

### Testing
- Each phase should have corresponding tests
- Verify output matches between old and new implementations
- Benchmark memory usage and performance
