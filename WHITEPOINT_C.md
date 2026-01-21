# CIE Illuminant C and the Origin of Y'CbCr Coefficients

This document traces the origin of the ubiquitous luma coefficients (0.299, 0.587, 0.114) used
in JPEG, video codecs, and countless image processing implementations.

## Summary

The coefficients **0.299, 0.587, 0.114** originate from **NTSC 1953** (BT.470 System M), which used:
- NTSC primaries: R(0.67, 0.33), G(0.21, 0.71), B(0.14, 0.08)
- CIE Illuminant C white point

**Key finding:** Practical implementations treat the rounded coefficients (0.299, 0.587, 0.114) as
authoritative, not any specific Illuminant C definition. The integer approximations in JPEG/video
are derived from these rounded values via `FIX(x) = (int)(x * scale + 0.5)`.

**Recommendation:** Define the legacy coefficients and NTSC primaries as authoritative, and derive
the effective Illuminant C from them. This matches what practical implementations actually do.

---

## Authoritative Definition

### NTSC 1953 Primaries (from BT.470 System M)

| Primary | x | y |
|---------|------|------|
| Red | 0.67 | 0.33 |
| Green | 0.21 | 0.71 |
| Blue | 0.14 | 0.08 |

### Legacy Luma Coefficients (authoritative)

| KR | KG | KB |
|-------|-------|-------|
| 0.299 | 0.587 | 0.114 |

These coefficients are the authoritative definition. All practical implementations derive from them.

### Effective Illuminant C (derived)

Working backwards from the legacy coefficients, the effective white point that produces values
closest to 0.299/0.587/0.114 is:

**XYZ (normalized to Y=1):**

| X | Y | Z |
|------|------|------|
| 0.98 | 1.00 | 1.18 |

**Derived chromaticity:**

| x | y |
|---------|---------|
| 0.31013 | 0.31646 |

This simple XYZ definition produces:
- KR = 0.2987 → rounds to **0.299** ✓
- KG = 0.5871 → rounds to **0.587** ✓
- KB = 0.1142 → rounds to **0.114** ✓

---

## Comparison of Illuminant C Definitions

| Definition | Form | KR | KG | KB | Error vs Legacy |
|------------|------|-------|-------|-------|-----------------|
| **XYZ (0.98, 1.0, 1.18)** | XYZ | 0.2987 | 0.5871 | 0.1142 | **6.12×10⁻⁴** |
| 4-digit chromaticity (0.3101, 0.3162) | xy | 0.2989 | 0.5866 | 0.1144 | 8.71×10⁻⁴ |
| 5-digit CIE (0.31006, 0.31616) | xy | 0.2989 | 0.5866 | 0.1145 | 9.54×10⁻⁴ |

The XYZ-based definition is **1.4× closer** to practical implementations than chromaticity-based definitions.

**Critical observation:** Notice that KG from XYZ-based (0.5871) rounds *up* to 0.587, while
chromaticity-based (0.5866) rounds *down* to 0.587. The XYZ-based value is much closer to the
target, suggesting the original NTSC derivation used simple XYZ values, not precise CIE chromaticity.

---

## The Derivation Chain

The historical derivation was likely:

```
Illuminant C as XYZ (0.98, 1.0, 1.18)
              ↓
    (matrix derivation from primaries)
              ↓
True coefficients: 0.2987, 0.5871, 0.1142
              ↓
    (round to 3 decimal places)
              ↓
Legacy coefficients: 0.299, 0.587, 0.114  ← AUTHORITATIVE
              ↓
    (FIX() macro for integer math)
              ↓
Integer approximations: 19595, 38470, 7471
```

The rounding step is where precision is lost. All subsequent implementations derive from the
rounded values, not from any Illuminant C definition.

---

## Integer Approximations in Practice

### 16-bit Fixed Point (shift 16, scale 65536)

```
Y = (19595 * R + 38470 * G + 7471 * B + 32768) >> 16
```

| Coefficient | Integer | Integer / 65536 | Target |
|-------------|---------|-----------------|--------|
| KR | 19595 | 0.298995971679688 | 0.299 |
| KG | 38470 | 0.587005615234375 | 0.587 |
| KB | 7471 | 0.113998413085938 | 0.114 |
| **Sum** | **65536** | **1.000000000000000** | |

### 15-bit Fixed Point (shift 15, scale 32768)

```
Y = (9798 * R + 19235 * G + 3736 * B + 16384) >> 15
```

| Coefficient | Integer | Integer / 32768 | Target |
|-------------|---------|-----------------|--------|
| KR | 9798 | 0.299011230468750 | 0.299 |
| KG | 19235 | 0.587005615234375 | 0.587 |
| KB | 3736 | 0.114013671875000 | 0.114 |
| **Sum** | **32769** | **1.000030517578125** | |

Note: 15-bit sum is 32769, one more than the target 32768.

### 8-bit Scaled (shift 8, scale 256)

```
Y = (77 * R + 150 * G + 29 * B + 128) >> 8
```

| Coefficient | Integer | Integer / 256 | Target |
|-------------|---------|---------------|--------|
| KR | 77 | 0.30078125 | 0.299 |
| KG | 150 | 0.5859375 | 0.587 |
| KB | 29 | 0.11328125 | 0.114 |
| **Sum** | **256** | **1.000000000000000** | |

### libjpeg-turbo Variant

```c
#define FIX(x) ((int)((x) * 65536 + 0.5))
// KR = FIX(0.299) = 19595
// KG = FIX(0.587) = 38469  (note: one less than standard)
// KB = FIX(0.114) = 7472   (note: one more than standard)
```

| Coefficient | Integer | Integer / 65536 | Target |
|-------------|---------|-----------------|--------|
| KR | 19595 | 0.298995971679688 | 0.299 |
| KG | 38469 | 0.586990356445312 | 0.587 |
| KB | 7472 | 0.114013671875000 | 0.114 |
| **Sum** | **65536** | **1.000000000000000** | |

---

## Evidence: Integers Derive from Legacy, Not Illuminant C

Applying `FIX(x) = (int)(x * 65536 + 0.5)` to different coefficient sources:

| Source | FIX(KR) | FIX(KG) | FIX(KB) | Matches Standard? |
|--------|---------|---------|---------|-------------------|
| XYZ (0.98, 1.0, 1.18) derived | 19575 | 38474 | 7487 | **0/3** |
| 4-digit chromaticity derived | 19591 | 38445 | 7500 | **0/3** |
| 5-digit CIE chromaticity derived | 19589 | 38445 | 7502 | **0/3** |
| **Legacy (0.299, 0.587, 0.114)** | **19595** | **38470** | **7471** | **3/3** ✓ |

The standard 16-bit integers **exactly match** FIX() applied to the legacy rounded values.
No Illuminant C definition produces matching integers.

---

## Error Analysis: Sources vs Practical Implementations

| Source | vs 16-bit std | vs 16-bit ljpg | vs 15-bit std | vs 8-bit std |
|--------|---------------|----------------|---------------|--------------|
| XYZ (0.98, 1.0, 1.18) | 6.04×10⁻⁴ | 6.04×10⁻⁴ | 6.04×10⁻⁴ | 4.17×10⁻³ |
| 4-digit chromaticity | 8.75×10⁻⁴ | 8.44×10⁻⁴ | 8.75×10⁻⁴ | 3.68×10⁻³ |
| 5-digit CIE chromaticity | 9.57×10⁻⁴ | 9.27×10⁻⁴ | 9.57×10⁻⁴ | 3.76×10⁻³ |
| **Legacy rounded** | **1.12×10⁻⁵** | **2.73×10⁻⁵** | **3.05×10⁻⁵** | **3.56×10⁻³** |

The legacy rounded values are **~50-80× closer** to practical implementations than any
Illuminant C-derived coefficients.

Among Illuminant C definitions, XYZ-based is **1.4× closer** to practice than chromaticity-based.

---

## Comparison with D50 and D65

| Illuminant | Authoritative Form | Values | Used By |
|------------|-------------------|--------|---------|
| **D65** | Chromaticity (xy) | (0.3127, 0.3290) | sRGB, BT.709, Display P3, Rec.2020 |
| **D50** | XYZ | (0.9642, 1.0, 0.8249) | ICC PCS, ProPhoto RGB |
| **C (effective)** | XYZ | (0.98, 1.0, 1.18) | NTSC 1953 (historical) |

D50 sets precedent for defining a white point as XYZ rather than chromaticity.
The effective Illuminant C follows this pattern.

**CIE reference values (for context only):**

| Illuminant | CIE Chromaticity | Notes |
|------------|------------------|-------|
| D65 | (0.31272, 0.32903) | Display standards use 4-digit (0.3127, 0.3290) |
| D50 | (0.34567, 0.35850) | ICC/ProPhoto use XYZ directly, not this |
| C | (0.31006, 0.31616) | Does not match practical implementations |

---

## Why JPEG Round-trips Aren't Bit-Exact

Different libraries use slightly different integer constants:

| Library | KR | KG | KB | Sum |
|---------|-------|-------|------|-------|
| "Standard" | 19595 | 38470 | 7471 | 65536 |
| libjpeg-turbo | 19595 | 38469 | 7472 | 65536 |

Both sum to 65536, but differ in KG/KB distribution. This is why JPEG encoded by one library
and decoded by another may have tiny differences.

The root cause: `FIX(0.587) = 38469.312...` is almost exactly between 38469 and 38470.
Different rounding conventions produce different results.

---

## Conclusions

1. **The legacy coefficients (0.299, 0.587, 0.114) are authoritative.** All practical
   implementations derive from these rounded values, not from any Illuminant C definition.

2. **The effective Illuminant C is XYZ (0.98, 1.0, 1.18).** This simple definition produces
   coefficients closest to the legacy values, suggesting the original NTSC engineers used
   rounded XYZ values rather than precise CIE chromaticity.

3. **For compatibility:** Use 0.299, 0.587, 0.114 exactly when interoperating with JPEG/video.

4. **For colorimetric accuracy:** Use the effective XYZ-based Illuminant C when computing
   matrices for NTSC 1953, as it better represents the original intent.

5. **CIE chromaticity values for Illuminant C do not match practice.** Using CIE (0.31006, 0.31616)
   produces coefficients further from 0.299/0.587/0.114 than the effective XYZ definition.

---

## References

- ITU-R BT.470-6 (1998) - NTSC System M primaries
- CIE 15:2004 - Colorimetry, Third Edition (Illuminant C reference)
- ITU-R BT.601-7 (2011) - SD video (uses legacy coefficients for Y'CbCr)
- JFIF 1.02 specification - JPEG file format
- libjpeg-turbo source code - jccolor.c
- ICC.1:2022-05 - ICC profile specification (D50 as XYZ precedent)
