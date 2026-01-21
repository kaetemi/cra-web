# Color Space Reference: Extended

These color spaces are not foundationally distinct but may appear in practical workflows. Each is defined in terms of spaces from the primary specification.

---

## D50 Illuminant

The standard white point for print-oriented color spaces and ICC workflows.

Unlike D65 color spaces (where chromaticity is authoritative), D50-based specifications define the white point directly as XYZ values.

**For ICC PCS and ProPhoto RGB (ISO 22028-2):**

| X | Y | Z |
|--------|--------|--------|
| 0.9642 | 1.0000 | 0.8249 |

This is the authoritative definition shared by both specifications. Matrices and conversions must use these XYZ values directly.

**Derived chromaticity:**

| x | y |
|--------|--------|
| 0.3457 | 0.3585 |

**CIE D50 reference (for context):**

CIE 15:2004 Table T.3 defines D50 via chromaticity (0.34567, 0.35850), which yields slightly different XYZ (0.9643, 1.0, 0.8251). This difference is irrelevant for practical workflows since no common color space uses CIE D50 chromaticity as its authoritative definition.

**Note:** Converting between D50 and D65 color spaces requires chromatic adaptation (Bradford transform). See the main specification for details.

---

## Display P3

Apple's wide-gamut color space, used on modern iPhones, iPads, and Macs since ~2016. Common in photos from iOS devices.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6800 | 0.3200 |
| Green | 0.2650 | 0.6900 |
| Blue | 0.1500 | 0.0600 |
| White (D65) | 0.3127 | 0.3290 |

**Transfer function:** Identical to sRGB (piecewise).

**Derived XYZ matrices:**

```
[linearized] Display P3 → XYZ:

| X |   | 0.4865709  0.2656677  0.1982173 |   | R |
| Y | = | 0.2289746  0.6917385  0.0792869 | × | G |
| Z |   | 0.0000000  0.0451134  1.0439444 |   | B |
```

```
XYZ → [linearized] Display P3:

| R |   |  2.4934969 -0.9313836 -0.4027108 |   | X |
| G | = | -0.8294890  1.7626641  0.0236247 | × | Y |
| B |   |  0.0358458 -0.0761724  0.9568845 |   | Z |
```

**Derived white point XYZ (normalized to Y=1):**

| X | Y | Z |
|---------|---------|---------|
| 0.9504559 | 1.0000000 | 1.0890578 |

**When you encounter it:** Images from Apple devices, web content using CSS `color(display-p3 ...)`.

**Relation to sRGB:** Wider gamut (~25% larger). sRGB colors are a subset. Converting P3→sRGB may require gamut clipping.

---

## Adobe RGB (1998)

A wider-gamut space designed for photography and print. Common in professional photo workflows.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6400 | 0.3300 |
| Green | 0.2100 | 0.7100 |
| Blue | 0.1500 | 0.0600 |
| White (D65) | 0.3127 | 0.3290 |

Red and blue primaries are identical to sRGB. Only green differs (more saturated).

**Transfer function:**

```
encoded = linear^(256/563)
linear = encoded^(563/256)
```

Note: The gamma is often approximated as 2.2, but the precise value specified by Adobe is 563/256 = 2.19921875. The difference is small but measurable.

**Derived XYZ matrices:**

```
[linearized] Adobe RGB → XYZ:

| X |   | 0.5766690  0.1855582  0.1882286 |   | R |
| Y | = | 0.2973450  0.6273636  0.0752915 | × | G |
| Z |   | 0.0270314  0.0706889  0.9913375 |   | B |
```

```
XYZ → [linearized] Adobe RGB:

| R |   |  2.0415879 -0.5650070 -0.3447314 |   | X |
| G | = | -0.9692436  1.8759675  0.0415551 | × | Y |
| B |   |  0.0134443 -0.1183624  1.0151750 |   | Z |
```

**Derived white point XYZ (normalized to Y=1):**

| X | Y | Z |
|---------|---------|---------|
| 0.9504559 | 1.0000000 | 1.0890578 |

**When you encounter it:** DSLR cameras, Photoshop workflows, print-oriented photography.

**Trivia:** The green primary reportedly resulted from an accidental use of NTSC primaries adapted to the wrong white point. Adobe kept it because users liked the results.

---

## ProPhoto RGB

An extremely wide-gamut space used in high-end photography. Encompasses nearly all visible colors but includes imaginary colors that cannot exist physically.

Also known as ROMM RGB (Reference Output Medium Metric). Specified in ISO 22028-2:2013.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.7347 | 0.2653 |
| Green | 0.1596 | 0.8404 |
| Blue | 0.0366 | 0.0001 |

**White point (D50) — defined as XYZ:**

| X | Y | Z |
|--------|--------|--------|
| 0.9642 | 1.0000 | 0.8249 |

Note: Unlike D65 color spaces where chromaticity is authoritative, ProPhoto RGB specifies the white point directly as XYZ values (shared with ICC PCS).

**Transfer function (encode: linear → ProPhoto):**

```
Et = 1/512  (≈0.001953)

if linear < Et:
    encoded = 16 × linear
else:
    encoded = linear^(1/1.8)
```

**Transfer function (decode: ProPhoto → linear):**

```
if encoded < 16 × Et:
    linear = encoded / 16
else:
    linear = encoded^1.8
```

Note: The linear segment exists (similar to sRGB) to avoid numerical issues near black. Many implementations approximate this as a pure γ=1.8 power function, which is adequate for most purposes but technically incorrect per ISO 22028-2:2013.

**Derived XYZ matrix (from ISO 22028-2):**

```
XYZ (D50) → [linearized] ProPhoto RGB:

| R |   |  1.3460  -0.2556  -0.0511 |   | X |
| G | = | -0.5446   1.5082   0.0205 | × | Y |
| B |   |  0.0000   0.0000   1.2123 |   | Z |
```

**When you encounter it:** Lightroom internal processing, archival photography workflows.

**Caution:** Contains imaginary colors. Some RGB triplets represent colors outside human vision.

---

## BT.601 (Rec.601)

Standard-definition video color spaces. Defined in ITU-R BT.601-7. Two variants exist for different broadcast systems.

### BT.601 625-line (PAL/SECAM)

Used for European and other PAL/SECAM broadcast systems.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.640 | 0.330 |
| Green | 0.290 | 0.600 |
| Blue | 0.150 | 0.060 |
| White (D65) | 0.3127 | 0.3290 |

**Derived XYZ matrices:**

```
[linearized] BT.601-625 → XYZ:

| X |   | 0.4305538  0.3415498  0.1783523 |   | R |
| Y | = | 0.2220043  0.7066548  0.0713409 | × | G |
| Z |   | 0.0201822  0.1295534  0.9393222 |   | B |
```

**True luminance coefficients (Y row of matrix):**

| KR | KG | KB |
|--------|--------|--------|
| 0.2220 | 0.7067 | 0.0713 |

### BT.601 525-line (NTSC)

Used for North American and Japanese NTSC broadcast systems.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.630 | 0.340 |
| Green | 0.310 | 0.595 |
| Blue | 0.155 | 0.070 |
| White (D65) | 0.3127 | 0.3290 |

**Derived XYZ matrices:**

```
[linearized] BT.601-525 → XYZ:

| X |   | 0.3935209  0.3652581  0.1916769 |   | R |
| Y | = | 0.2123764  0.7010599  0.0865638 | × | G |
| Z |   | 0.0187391  0.1119339  0.9583847 |   | B |
```

**True luminance coefficients (Y row of matrix):**

| KR | KG | KB |
|--------|--------|--------|
| 0.2124 | 0.7011 | 0.0866 |

### Original NTSC 1953 (BT.470 M/NTSC)

The **source** of the 0.299/0.587/0.114 luma coefficients is the original 1953 NTSC color space, defined in ITU-R BT.470 System M.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.67 | 0.33 |
| Green | 0.21 | 0.71 |
| Blue | 0.14 | 0.08 |
| White (Illuminant C) | 0.3101 | 0.3162 |

**Derived RGB→XYZ matrix:**

```
[linearized] NTSC 1953 → XYZ:

| X |   | 0.6067225  0.1736120  0.2006781 |   | R |
| Y | = | 0.2989391  0.5866251  0.1144357 | × | G |
| Z |   | 0.0000000  0.0660948  1.1172925 |   | B |
```

**True luminance coefficients (Y row of matrix):**

| KR | KG | KB |
|--------|--------|--------|
| 0.2989 | 0.5866 | 0.1144 |

These round exactly to **0.299, 0.587, 0.114** — the legacy Y'CbCr coefficients.

### Legacy Y'CbCr Coefficients

The traditional BT.601 Y'CbCr formula uses:

```
Y' = 0.299 R' + 0.587 G' + 0.114 B'
```

These coefficients originate from **NTSC 1953 with Illuminant C** (above), not from the modern BT.601 color spaces which use D65.

**Comparison of luminance coefficients:**

| Source | KR | KG | KB | White Point |
|--------|--------|--------|--------|-------------|
| **NTSC 1953** | 0.2989 | 0.5866 | 0.1144 | Illuminant C |
| Legacy Y'CbCr | 0.299 | 0.587 | 0.114 | (rounded from above) |
| BT.601 625-line (PAL) | 0.2220 | 0.7067 | 0.0713 | D65 |
| BT.601 525-line (NTSC) | 0.2124 | 0.7011 | 0.0866 | D65 |

The legacy coefficients were carried forward into BT.601 for backward compatibility with existing NTSC equipment, even though BT.601 redefined the primaries and white point to D65.

**When you encounter it:** SD video (DVD, analog broadcast), JPEG compression (which uses BT.601 Y'CbCr regardless of the RGB color space).

**Note:** JPEG always uses the legacy 0.299/0.587/0.114 coefficients for Y'CbCr, even when the underlying RGB is sRGB. This creates a mathematical inconsistency—the Y' channel doesn't represent true luminance in any color space.

---

## Rec.2020

Ultra-wide gamut space for HDR and UHD television. Defined in ITU-R BT.2020.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.7080 | 0.2920 |
| Green | 0.1700 | 0.7970 |
| Blue | 0.1310 | 0.0460 |
| White (D65) | 0.3127 | 0.3290 |

**Transfer function:** Same as Rec.709/sRGB for SDR content. HDR uses PQ or HLG.

**Derived XYZ matrices:**

```
[linearized] Rec.2020 → XYZ:

| X |   | 0.6370102  0.1446150  0.1688448 |   | R |
| Y | = | 0.2627098  0.6779735  0.0593168 | × | G |
| Z |   | 0.0000000  0.0280834  1.0608196 |   | B |
```

```
XYZ → [linearized] Rec.2020:

| R |   |  1.7166512 -0.3556708 -0.2533663 |   | X |
| G | = | -0.6666844  1.6164812  0.0157685 | × | Y |
| B |   |  0.0176399 -0.0427706  0.9421031 |   | Z |
```

**Derived white point XYZ (normalized to Y=1):**

| X | Y | Z |
|---------|---------|---------|
| 0.9504559 | 1.0000000 | 1.0890578 |

**When you encounter it:** 4K/8K UHD content, HDR video.

**Caution:** Gamut is so wide that no current display can fully reproduce it. Content is mastered with the expectation of partial coverage.

---

## OKLch

Cylindrical (polar) coordinates for OKLab. Not a different color space—just a coordinate transform.

**Definition:** Given OKLab (L, a, b):

```
L = L
C = √(a² + b²)
h = atan2(b, a)
```

**Inverse:**

```
L = L
a = C × cos(h)
b = C × sin(h)
```

**When you use it:** Hue-preserving operations, chroma adjustments, analyzing perceptual uniformity along hue.

---

## HSL / HSV

Cylindrical transforms of sRGB. Not perceptually uniform. Included because they're common in UI code.

These operate on gamma-encoded sRGB values.

**HSV (Hue, Saturation, Value):**

```
V = max(R', G', B')
S = (V - min(R', G', B')) / V    [if V ≠ 0, else 0]
H = [derived from which channel is max]
```

**HSL (Hue, Saturation, Lightness):**

```
L = (max(R', G', B') + min(R', G', B')) / 2
S = (max - min) / (1 - |2L - 1|)  [if L ≠ 0 or 1, else 0]
H = [same as HSV]
```

**When you encounter it:** CSS, color pickers, UI code.

**For dithering:** Not useful. These are conveniences for human color selection, not principled spaces for color math.

---

## ICtCp

A modern perceptual luma-chroma space designed for HDR. Defined in ITU-R BT.2100.

**Definition:** Operates on linear Rec.2020, then applies PQ (perceptual quantizer) or HLG transfer function before a matrix transform to ICtCp.

The structure is:
1. Linear Rec.2020 → LMS (matrix)
2. LMS → L'M'S' (PQ or HLG)
3. L'M'S' → ICtCp (matrix)

**When you encounter it:** HDR video processing, HDR color grading.

**Advantage over Y'CbCr:** Better perceptual uniformity, hue linearity. Designed with actual perceptual data rather than legacy CRT constraints.

**For dithering:** Only relevant if doing HDR. Uses PQ/HLG transfer functions which are outside typical SDR workflows.

---

## ICC Profiles

ICC profiles are not a color space but a container format for describing color spaces and device characteristics. Defined by the International Color Consortium in ICC.1 (currently ICC.1:2022-05).

### Profile Connection Space (PCS)

All ICC profiles convert to/from a common intermediate called the Profile Connection Space. The PCS is either CIEXYZ or CIELAB, referenced to the D50 illuminant.

**PCS White Point (D50) — defined as XYZ:**

The ICC specification defines the PCS illuminant directly as XYZ values, not chromaticity:

| X | Y | Z |
|--------|--------|--------|
| 0.9642 | 1.0000 | 0.8249 |

This is the normative definition from ICC.1:2022-05 Section 7.2.16. Per Section 6.3.1, the PCS adopted white chromaticity is "the chromaticity of the D50 illuminant defined in ISO 3664," but the XYZ values above are authoritative for ICC interoperability.

**Derived chromaticity:**

| x | y |
|--------|--------|
| 0.3457 | 0.3585 |

**Comparison with CIE D50:**

| Source | X | Y | Z | Status |
|--------|--------|--------|--------|--------|
| ICC PCS / ProPhoto RGB | 0.9642 | 1.0000 | 0.8249 | Authoritative for these specs |
| CIE D50 (from chromaticity) | 0.9643 | 1.0000 | 0.8251 | Not used by any common color space |

The difference arises because ICC defines XYZ at 4 decimal places independently, rather than deriving from CIE chromaticity. For ICC and ProPhoto RGB workflows, use the ICC values exactly as specified.

**Binary Representation:**

ICC profiles store XYZ values as s15Fixed16Number (signed 16-bit fixed-point with 16 fractional bits). The D50 white point encodes as:

| Component | Hex | Decimal (exact) |
|-----------|--------|------------------------|
| X | 0xF6D6 | 0.964202880859375 |
| Y | 0x10000 | 1.0 |
| Z | 0xD32D | 0.8249053955078125 |

Note: The hex values are the authoritative binary representation. When divided by 65536, they yield values that round to the 4-decimal specification.

### Conversion Flow

When converting between two color spaces via ICC profiles:

```
Source RGB → [Source Profile] → PCS (XYZ D50) → [Dest Profile] → Dest RGB
```

Each profile contains:
- **Colorant matrix** (rXYZ, gXYZ, bXYZ tags): RGB→XYZ transform, already adapted to D50
- **TRC curves**: Transfer functions for linearization
- **CHAD tag** (optional): Chromatic adaptation matrix used to adapt the original colorants to D50

### Computational Model

Per ICC.1:2022-05 Section F.3, the matrix-based conversion is:

```
XYZ_D50 = colorantMatrix × linear_RGB
```

The colorant tags are used directly as matrix columns. No additional white point scaling is applied—the adaptation to D50 is already baked into the colorants.

---

## Summary

### Authoritative Definitions

| Space | White Point Definition | Primaries | Transfer |
|-------|------------------------|-----------|----------|
| Display P3 | D65 chromaticity (0.3127, 0.3290) | Chromaticity | sRGB piecewise |
| Adobe RGB | D65 chromaticity (0.3127, 0.3290) | Chromaticity | γ = 563/256 |
| ProPhoto RGB | D50 XYZ (0.9642, 1.0, 0.8249) | Chromaticity | Piecewise 1.8 |
| NTSC 1953 | Illuminant C (0.3101, 0.3162) | Chromaticity | γ = 2.2 |
| BT.601-625 (PAL) | D65 chromaticity (0.3127, 0.3290) | Chromaticity | γ = 2.2 (approx) |
| BT.601-525 (NTSC) | D65 chromaticity (0.3127, 0.3290) | Chromaticity | γ = 2.2 (approx) |
| Rec.2020 | D65 chromaticity (0.3127, 0.3290) | Chromaticity | sRGB (SDR) / PQ,HLG (HDR) |
| ICC PCS | D50 XYZ (0.9642, 1.0, 0.8249) | N/A | N/A |

### Gamut Comparison

| Space | Gamut vs sRGB | When you encounter it |
|-------|---------------|----------------------|
| Display P3 | ~25% larger | Apple devices, modern web |
| Adobe RGB | ~40% larger | Photography, print |
| ProPhoto RGB | ~90% of visible | Lightroom, archival |
| NTSC 1953 | Similar to sRGB | Historical (source of Y'CbCr coefficients) |
| BT.601 (PAL/NTSC) | Similar to sRGB | SD video, DVD, JPEG |
| Rec.2020 | ~75% of visible | UHD/HDR video |
| OKLch | Same as sRGB | Hue-preserving ops |
| HSL/HSV | Same as sRGB | UI, color pickers |
| ICtCp | Same as Rec.2020 | HDR video |

---

## Dependency Graph

Includes all spaces from both the base specification and this extended reference.

```
CIE XYZ (empirical root)
    │
    ├── CIELAB
    │
    ├── Linear RGB (D65)
    │       │
    │       ├── sRGB
    │       │     │
    │       │     ├── Y'CbCr (BT.709)
    │       │     ├── HSL
    │       │     └── HSV
    │       │
    │       ├── Gamma 2.2 RGB
    │       │
    │       └── OKLab
    │             │
    │             └── OKLch
    │
    ├── Display P3 (D65)
    │
    ├── Adobe RGB (D65)
    │
    ├── NTSC 1953 (Illuminant C)
    │       │
    │       └── Y'CbCr (0.299/0.587/0.114 originates here)
    │
    ├── BT.601-625 PAL (D65)
    │
    ├── BT.601-525 NTSC (D65)
    │
    ├── Rec.2020 (D65)
    │       │
    │       └── ICtCp
    │
    ├── Apple RGB (D65)
    │
    ├── ProPhoto RGB (D50)
    │
    └── ICC PCS (D50)
          │
          └── [all ICC-profiled spaces]
```
