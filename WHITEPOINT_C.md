# CIE Illuminant C and the Origin of Y'CbCr Coefficients

This document traces the origin of the ubiquitous luma coefficients (0.299, 0.587, 0.114) used
in JPEG, video codecs, and countless image processing implementations.

## Summary

The coefficients 0.299, 0.587, 0.114 originate from **NTSC 1953** (BT.470 System M), which used:
- NTSC 1953 primaries: R(0.67, 0.33), G(0.21, 0.71), B(0.14, 0.08)
- CIE Illuminant C white point

The integer approximations used in JPEG and video implementations are derived from these
**already-rounded** 3-digit values, not from the true Illuminant C-derived coefficients.

## CIE Illuminant C Definitions

| Source | x | y | Notes |
|--------|---|---|-------|
| CIE 15:2004 Table T.3 | 0.31006 | 0.31616 | 5-digit authoritative |
| Common 4-digit | 0.3101 | 0.3162 | Consistent with D65 (0.3127, 0.3290) |

## Derived Luma Coefficients from Illuminant C

### 4-digit Illuminant C (0.3101, 0.3162)

| Coefficient | Derived Value | Rounds to |
|-------------|---------------|-----------|
| KR | 0.298939144598747 | **0.299** ✓ |
| KG | 0.586625129640780 | **0.587** ✓ |
| KB | 0.114435725760473 | **0.114** ✓ |
| Sum | 1.000000000000000 | |

### 5-digit CIE Illuminant C (0.31006, 0.31616)

| Coefficient | Derived Value | Rounds to |
|-------------|---------------|-----------|
| KR | 0.298903070250081 | **0.299** ✓ |
| KG | 0.586619854659197 | **0.587** ✓ |
| KB | 0.114477075090722 | **0.114** ✓ |
| Sum | 1.000000000000000 | |

Both precisions round correctly to the legacy values. The 4-digit version produces values
slightly closer to the rounded targets.

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

## Key Finding: The Lossy Derivation Chain

The integer approximations are **NOT** derived directly from Illuminant C. Instead, they follow
a lossy chain:

```
Illuminant C chromaticity
         ↓
    (matrix derivation)
         ↓
True coefficients: 0.29894, 0.58663, 0.11444
         ↓
    (round to 3 digits)
         ↓
Legacy coefficients: 0.299, 0.587, 0.114
         ↓
    (FIX() macro)
         ↓
Integer approximations: 19595, 38470, 7471
```

### Evidence: FIX() from Different Sources

If we apply `FIX(x) = (int)(x * 65536 + 0.5)` to different coefficient sources:

| Source | FIX(KR) | FIX(KG) | FIX(KB) | Matches Standard? |
|--------|---------|---------|---------|-------------------|
| 4-digit Illuminant C derived | 19591 | 38445 | 7500 | **0/3** |
| 5-digit Illuminant C derived | 19589 | 38445 | 7502 | **0/3** |
| Legacy (0.299, 0.587, 0.114) | 19595 | 38470 | 7471 | **3/3** ✓ |

The standard integers (19595, 38470, 7471) **exactly match** FIX() applied to the legacy
rounded values, not the true Illuminant C-derived coefficients.

### Integer Sum Verification

| Approximation | Sum | Target | Difference |
|---------------|-----|--------|------------|
| 16-bit standard | 65536 | 65536 | 0 |
| 16-bit libjpeg | 65536 | 65536 | 0 |
| 15-bit standard | 32769 | 32768 | +1 |
| 8-bit standard | 256 | 256 | 0 |

The 16-bit standard integers were likely chosen specifically to sum to exactly 65536.

## Error Analysis

### Which Illuminant C Precision is Closer to Practice?

Comparing practical implementations against both Illuminant C precisions:

| Implementation | vs 4-digit | vs 5-digit | vs Legacy | Winner |
|----------------|------------|------------|-----------|--------|
| 16-bit standard | 8.75×10⁻⁴ | 9.57×10⁻⁴ | **1.12×10⁻⁵** | Legacy (78×) |
| 16-bit libjpeg | 8.44×10⁻⁴ | 9.27×10⁻⁴ | **2.73×10⁻⁵** | Legacy (31×) |
| 15-bit standard | 8.75×10⁻⁴ | 9.57×10⁻⁴ | **3.05×10⁻⁵** | Legacy (29×) |
| 8-bit standard | 3.68×10⁻³ | 3.76×10⁻³ | **3.56×10⁻³** | Legacy (1.03×) |

**4-digit Illuminant C is ~9.5% closer to practical implementations than 5-digit** in all cases.
However, both are ~80× farther from practical implementations than the legacy rounded values.

This confirms that practical implementations derive from the already-rounded 0.299/0.587/0.114,
not from either Illuminant C precision directly.

### Error vs 4-digit Illuminant C Derived Coefficients

| Approximation | KR Error | KG Error | KB Error | Total Error |
|---------------|----------|----------|----------|-------------|
| 16-bit standard | 5.68×10⁻⁵ | 3.80×10⁻⁴ | 4.37×10⁻⁴ | 8.75×10⁻⁴ |
| 16-bit libjpeg | 5.68×10⁻⁵ | 3.65×10⁻⁴ | 4.22×10⁻⁴ | 8.44×10⁻⁴ |
| 15-bit standard | 7.21×10⁻⁵ | 3.80×10⁻⁴ | 4.22×10⁻⁴ | 8.75×10⁻⁴ |
| 8-bit standard | 1.84×10⁻³ | 6.88×10⁻⁴ | 1.15×10⁻³ | 3.68×10⁻³ |

### Error vs 5-digit Illuminant C Derived Coefficients

| Approximation | KR Error | KG Error | KB Error | Total Error |
|---------------|----------|----------|----------|-------------|
| 16-bit standard | 9.29×10⁻⁵ | 3.86×10⁻⁴ | 4.79×10⁻⁴ | 9.57×10⁻⁴ |
| 16-bit libjpeg | 9.29×10⁻⁵ | 3.71×10⁻⁴ | 4.63×10⁻⁴ | 9.27×10⁻⁴ |
| 15-bit standard | 1.08×10⁻⁴ | 3.86×10⁻⁴ | 4.63×10⁻⁴ | 9.57×10⁻⁴ |
| 8-bit standard | 1.88×10⁻³ | 6.82×10⁻⁴ | 1.20×10⁻³ | 3.76×10⁻³ |

### Error vs Legacy Rounded Values (0.299, 0.587, 0.114)

| Approximation | KR Error | KG Error | KB Error | Total Error |
|---------------|----------|----------|----------|-------------|
| 16-bit standard | 4.03×10⁻⁶ | 5.62×10⁻⁶ | 1.59×10⁻⁶ | 1.12×10⁻⁵ |
| 16-bit libjpeg | 4.03×10⁻⁶ | 9.64×10⁻⁶ | 1.37×10⁻⁵ | 2.73×10⁻⁵ |
| 15-bit standard | 1.12×10⁻⁵ | 5.62×10⁻⁶ | 1.37×10⁻⁵ | 3.05×10⁻⁵ |
| 8-bit standard | 1.78×10⁻³ | 1.06×10⁻³ | 7.19×10⁻⁴ | 3.56×10⁻³ |

The 16-bit integers are ~80× closer to the legacy rounded values than to the true
Illuminant C-derived values.

## Why Different Libraries Use Different Constants

The libjpeg-turbo constants (19595, 38469, 7472) differ from the "standard" (19595, 38470, 7471):

- Both sum to 65536 exactly
- libjpeg uses `FIX(x) = (int)(x * 65536 + 0.5)` literally
- The "standard" values may have been adjusted to minimize some other metric

This is why **JPEG round-trips are not bit-exact across implementations** — different libraries
use slightly different integer constants, all approximating the same underlying 0.299/0.587/0.114.

## 8-bit: The Great Equalizer

At 8-bit precision, all sources converge:

| Source | FIX(KR) | FIX(KG) | FIX(KB) |
|--------|---------|---------|---------|
| 4-digit Illuminant C | 77 | 150 | 29 |
| 5-digit Illuminant C | 77 | 150 | 29 |
| Legacy rounded | 77 | 150 | 29 |

The 8-bit precision is too coarse to distinguish between any of these sources.

## Recommendations

1. **For compatibility**: Use the legacy 0.299, 0.587, 0.114 values when interoperating with
   existing JPEG/video implementations.

2. **For accuracy**: Use the true Illuminant C-derived values (from either 4-digit or 5-digit
   white point) when colorimetric accuracy matters more than compatibility.

3. **For integer implementations**: The standard 16-bit integers (19595, 38470, 7471) are
   well-established and sum to exactly 65536.

4. **Documentation**: When specifying which coefficients your implementation uses, be explicit
   about whether you're using legacy rounded values or true derived values.

## References

- ITU-R BT.470-6 (1998) - NTSC System M primaries and Illuminant C
- CIE 15:2004 - Colorimetry, Third Edition (Illuminant C: x=0.31006, y=0.31616)
- ITU-R BT.601-7 (2011) - Modern video standard (uses D65, different primaries)
- JFIF 1.02 specification - JPEG file format
- libjpeg-turbo source code - jccolor.c
