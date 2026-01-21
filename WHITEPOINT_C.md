# CIE Illuminant C and NTSC 1953

This document describes CIE Illuminant C and its use in the NTSC 1953 color space,
and traces the origin of the ubiquitous luma coefficients (0.299, 0.587, 0.114).

## Summary

**NTSC 1953** (BT.470 System M) is defined by:
- NTSC primaries: R(0.67, 0.33), G(0.21, 0.71), B(0.14, 0.08)
- CIE Illuminant C white point: (0.3101, 0.3162)

The RGB→XYZ matrix Y row gives luma coefficients that round to 0.299/0.587/0.114.

**Key decision:** We use **4-digit chromaticity (0.3101, 0.3162)** for Illuminant C because:
1. It rounds to the CIE specification (0.310, 0.316)
2. The resulting matrix coefficients round to 0.299/0.587/0.114
3. It maintains colorimetric coherence with international standards

---

## Authoritative Definitions

### NTSC 1953 Primaries (from BT.470 System M)

| Primary | x | y |
|---------|------|------|
| Red | 0.67 | 0.33 |
| Green | 0.21 | 0.71 |
| Blue | 0.14 | 0.08 |

### CIE Illuminant C (4-digit chromaticity)

| x | y |
|--------|--------|
| 0.3101 | 0.3162 |

This 4-digit chromaticity rounds to the CIE specification (0.310, 0.316).

**Derived XYZ (Y=1):**

| X | Y | Z |
|--------|------|--------|
| 0.9807 | 1.00 | 1.1818 |

---

## Matrix-Derived Luma Coefficients

Using NTSC primaries with Illuminant C chromaticity (0.3101, 0.3162):

| Coefficient | Matrix Value | Rounds To |
|-------------|--------------|-----------|
| KR | 0.2989... | 0.299 ✓ |
| KG | 0.5866... | 0.587 ✓ |
| KB | 0.1144... | 0.114 ✓ |

---

## Comparison of Illuminant C Definitions

| Definition | Chromaticity | Rounds To | KR | KG | KB |
|------------|--------------|-----------|-------|-------|-------|
| **4-digit (0.3101, 0.3162)** | (0.3101, 0.3162) | **(0.310, 0.316) ✓** | 0.2989 | 0.5866 | 0.1144 |
| 5-digit CIE (0.31006, 0.31616) | (0.31006, 0.31616) | (0.310, 0.316) ✓ | 0.2989 | 0.5866 | 0.1145 |
| XYZ-derived (0.98, 1.0, 1.18) | (0.3105, 0.3168) | **(0.311, 0.317) ✗** | 0.2987 | 0.5871 | 0.1142 |

**The 4-digit chromaticity is preferred** because:
1. It rounds to the CIE specification (0.310, 0.316)
2. The derived luma coefficients all round correctly to 0.299/0.587/0.114
3. It maintains colorimetric coherence with international standards

The XYZ-derived approach (deriving white point from legacy coefficients) produces coefficients
*closer* to 0.299/0.587/0.114, but the resulting chromaticity (0.311, 0.317) does not match the
CIE specification—it's out of spec.

---

## The Trade-off

There is an inherent inconsistency in the NTSC specification:

| Approach | Illuminant C Chromaticity | Luma Coefficients |
|----------|--------------------------|-------------------|
| **Use CIE-spec chromaticity** | ✓ In spec (0.310, 0.316) | Approximate (rounds correctly) |
| Derive from legacy coefficients | ✗ Out of spec (0.311, 0.317) | Exact (0.299, 0.587, 0.114) |

**We choose the chromaticity-based approach** because:
1. Illuminant C should match its CIE definition
2. The luma coefficients round correctly to the legacy values
3. Practical implementations use the rounded values anyway

---

## Historical Note: The "30/59/11" Formula

The coefficients 0.299, 0.587, 0.114 (often approximated as 30/59/11) originated from
NTSC 1953 but have been widely misused:

1. **Original context**: NTSC 1953 with Illuminant C white point
2. **Common misuse**: Applied to sRGB/BT.709 data (D65 white point, different primaries)

This color space mismatch is pervasive in JPEG and video implementations. The formula
produces reasonable-looking results because human vision is more sensitive to luminance
errors than chrominance errors, but it's technically incorrect for modern color spaces.

For proper Y'CbCr conversion:
- **BT.709/sRGB**: Use coefficients derived from the sRGB matrix (0.2126, 0.7152, 0.0722)
- **BT.601**: Use coefficients derived from BT.601 primaries with D65

---

## Integer Approximations in Practice

The historical derivation was likely:

```
CIE Illuminant C chromaticity
              ↓
    (matrix derivation from primaries)
              ↓
True coefficients: ~0.2989, ~0.5866, ~0.1144
              ↓
    (round to 3 decimal places)
              ↓
Rounded coefficients: 0.299, 0.587, 0.114
              ↓
    (FIX() macro for integer math)
              ↓
Integer approximations: 19595, 38470, 7471
```

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

## Evidence: Integers Derive from Rounded Values

Applying `FIX(x) = (int)(x * 65536 + 0.5)` to different coefficient sources:

| Source | FIX(KR) | FIX(KG) | FIX(KB) | Matches Standard? |
|--------|---------|---------|---------|-------------------|
| 4-digit chromaticity derived | 19591 | 38445 | 7500 | **0/3** |
| 5-digit CIE chromaticity derived | 19589 | 38445 | 7502 | **0/3** |
| XYZ (0.98, 1.0, 1.18) derived | 19575 | 38474 | 7487 | **0/3** |
| **Rounded (0.299, 0.587, 0.114)** | **19595** | **38470** | **7471** | **3/3** ✓ |

The standard 16-bit integers **exactly match** FIX() applied to the rounded values.
No Illuminant C definition produces matching integers directly.

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

## Comparison with D50 and D65

| Illuminant | Authoritative Form | Values | Used By |
|------------|-------------------|--------|---------|
| **D65** | Chromaticity (xy) | (0.3127, 0.3290) | sRGB, BT.709, Display P3, Rec.2020 |
| **D50** | XYZ | (0.9642, 1.0, 0.8249) | ICC PCS, ProPhoto RGB |
| **C** | Chromaticity (xy) | (0.3101, 0.3162) | NTSC 1953 |

**CIE reference values:**

| Illuminant | CIE Chromaticity | Notes |
|------------|------------------|-------|
| D65 | (0.31272, 0.32903) | Display standards use 4-digit (0.3127, 0.3290) |
| D50 | (0.34567, 0.35850) | ICC/ProPhoto use XYZ directly |
| C | (0.31006, 0.31616) | We use 4-digit (0.3101, 0.3162) |

---

## References

- ITU-R BT.470-6 (1998) - NTSC System M primaries
- CIE 15:2004 - Colorimetry, Third Edition (Illuminant C reference)
- ITU-R BT.601-7 (2011) - SD video
- ITU-R BT.709-6 (2015) - HD video
- JFIF 1.02 specification - JPEG file format
- libjpeg-turbo source code - jccolor.c
