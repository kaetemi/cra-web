# Color Space Specification

## CIE 1931 XYZ

XYZ is the root of colorimetry. It is defined empirically from the CIE 1931 color matching experiments, where human observers matched spectral colors to mixtures of three primaries. The resulting color matching functions x̄(λ), ȳ(λ), z̄(λ) define how to convert any spectral power distribution to a tristimulus value:

```
X = ∫ S(λ) × x̄(λ) dλ
Y = ∫ S(λ) × ȳ(λ) dλ
Z = ∫ S(λ) × z̄(λ) dλ
```

Y corresponds directly to luminance. Chromaticity coordinates project out the brightness:

```
x = X / (X + Y + Z)
y = Y / (X + Y + Z)
```

Everything else in this document ultimately derives from XYZ.

---

## sRGB / Linear RGB

IEC 61966-2-1:1999. The standard color space for the web and consumer displays.

**Authoritative definition:**

The sRGB specification defines the color space in terms of chromaticity coordinates, not matrices. These are the normative values from which all else is derived:

| Primary | x | y |
|---------|-------|-------|
| Red | 0.640 | 0.330 |
| Green | 0.300 | 0.600 |
| Blue | 0.150 | 0.060 |

| White Point (D65) | x | y |
|-------------------|--------|--------|
| | 0.3127 | 0.3290 |

**Linear RGB** is the intermediate state before the transfer function is applied. It shares these primaries and white point with a linear (identity) transfer function.

**Derived XYZ matrices:**

The RGB↔XYZ matrices are derived from the chromaticity coordinates above through standard colorimetric calculation. The 4-decimal versions in IEC 61966-2-1 are truncated for presentation:

```
Linear RGB → XYZ (4 decimal, per specification):

| X |   | 0.4124  0.3576  0.1805 |   | R |
| Y | = | 0.2126  0.7152  0.0722 | × | G |
| Z |   | 0.0193  0.1192  0.9505 |   | B |
```

The specification explicitly notes these truncated matrices are insufficient for higher bit depths. For N-bit encoding where N > 8, IEC 61966-2-1 Amendment 1 recommends deriving matrices with additional precision from the defining chromaticities.

**Higher precision matrices (7 decimal places):**

Derived from the authoritative chromaticity coordinates:

```
Linear RGB → XYZ:

| X |   | 0.4123908  0.3575843  0.1804808 |   | R |
| Y | = | 0.2126390  0.7151687  0.0721923 | × | G |
| Z |   | 0.0193308  0.1191948  0.9505322 |   | B |
```

```
XYZ → Linear RGB:

| R |   |  3.2409699 -1.5373832 -0.4986108 |   | X |
| G | = | -0.9692436  1.8759675  0.0415551 | × | Y |
| B |   |  0.0556301 -0.2039770  1.0569715 |   | Z |
```

For maximum precision, derive the matrices directly from the chromaticity coordinates, carrying full floating-point precision throughout the calculation.

**Derived white point XYZ (normalized to Y=1):**

| X | Y | Z |
|---------|---------|---------|
| 0.9504559 | 1.0000000 | 1.0890578 |

**Transfer function (encode: linear → sRGB):**

```
if linear ≤ 0.0031308:
    srgb = 12.92 × linear
else:
    srgb = 1.055 × linear^(1/2.4) − 0.055
```

**Transfer function (decode: sRGB → linear):**

```
if srgb ≤ 0.04045:
    linear = srgb / 12.92
else:
    linear = ((srgb + 0.055) / 1.055)^2.4
```

The linear segment exists because a pure power curve has infinite slope at zero, which would amplify noise in darks. The piecewise function is continuous with continuous first derivative at the junction.

**Luminance coefficients:**

The second row of the forward matrix gives the luminance coefficients: Y = 0.2126R + 0.7152G + 0.0722B (or with higher precision as derived).

---

## D65 Illuminant

The standard daylight white point used by most display-oriented color spaces. Multiple representations exist due to rounding at different stages of standardization.

### D65 (CIE)

Derived from the D65 spectral power distribution and CIE color matching functions. Extended-precision calculations yield values like x = 0.31272, y = 0.32903 (or further: 0.3127205252, 0.3290306850), but these are computed from source data with only 4-5 significant digits of precision. The extra decimal places are computational artifacts, not genuine accuracy.

### D65 (sRGB / Rec.709)

The white point defined by IEC 61966-2-1 and ITU-R BT.709:

| x | y |
|--------|--------|
| 0.3127 | 0.3290 |

These are the authoritative values for sRGB. Any implementation claiming sRGB conformance must use exactly these coordinates when deriving transformation matrices.

**Practical impact:** Using "more accurate" D65 values from CIE sources (e.g., 0.31272, 0.32903) produces matrices for a subtly different color space—not sRGB. The sRGB standard's declared values take precedence over external references.

---

## Gamma 2.2 RGB

No formal standard. Often conflated with sRGB but technically distinct—the same encoded value decodes to slightly different linear values between the two, primarily in darks.

**Definition:** Linear RGB (sRGB primaries and white point) with γ=2.2.

**Transfer function:**

```
encoded = linear^(1/2.2)
linear = encoded^2.2
```

---

## Y'CbCr (Rec.709)

A luma-chroma color space used for video transmission and compression. Separates a brightness-like channel (Y') from two color difference channels (Cb, Cr).

**Definition:** Operates on gamma-encoded sRGB values, not linear light.

The prime notation (Y') indicates gamma-encoded. This is not a linear luminance measurement.

**Forward transform (sRGB → Y'CbCr):**

```
Y'  =  0.2126 R' + 0.7152 G' + 0.0722 B'
Cb  = (B' - Y') / 1.8556
Cr  = (R' - Y') / 1.5748
```

Or as a matrix:

```
| Y' |   |  0.2126    0.7152    0.0722  |   | R' |
| Cb | = | -0.1146   -0.3854    0.5000  | × | G' |
| Cr |   |  0.5000   -0.4542   -0.0458  |   | B' |
```

Y' ranges [0, 1], Cb and Cr range [−0.5, 0.5].

**Inverse transform (Y'CbCr → sRGB):**

```
R'  = Y' + 1.5748 Cr
G'  = Y' - 0.1873 Cb - 0.4681 Cr
B'  = Y' + 1.8556 Cb
```

**On the coefficients:**

The coefficients (0.2126, 0.7152, 0.0722) are derived from the second row of the Linear RGB → XYZ matrix. When applied to Linear RGB, they yield true CIE luminance—the Y in XYZ:

```
Y = 0.2126 R + 0.7152 G + 0.0722 B
```

This is correct for linear light. Y'CbCr takes these coefficients and applies them to gamma-encoded values instead, which is physically incorrect but was standardized for video.

**Caveats:**

CRT gamma is a physical property of electron guns, not a perceptual encoding. The fact that γ≈2.2 vaguely resembles perceptual response is coincidence. Actual perceptual lightness uses curves like CIELAB's L* or OKLab's cube root, derived from psychophysical experiments.

Y'CbCr exists because video signals were always gamma-encoded for CRT displays, and the standards were built around that. There is no linear-light variant of YCbCr. If you need actual luminance, linearize first or use XYZ's Y channel.

**Note on Rec.601 coefficients:**

Older video (NTSC/PAL) uses different primaries than sRGB/BT.709. The classic "30/59/11" formula:

```
Y' = 0.299 R' + 0.587 G' + 0.114 B'
```

These coefficients are **legacy values** that do not match the true luminance from either BT.601 color space (625-line PAL or 525-line NTSC). The true luminance coefficients derived from the respective RGB→XYZ matrices are quite different:

| Source | KR | KG | KB |
|--------|--------|--------|--------|
| BT.601-625 (PAL) | 0.222 | 0.707 | 0.071 |
| BT.601-525 (NTSC) | 0.212 | 0.701 | 0.087 |
| Legacy Y'CbCr | 0.299 | 0.587 | 0.114 |

See COLORSPACESEXT.md for full BT.601 color space definitions and matrices.

**Y'CbCr and JPEG:**

JPEG (JFIF) always uses BT.601 Y'CbCr for compression—same transform, same coefficients—but the input RGB can come from different color spaces (sRGB, ProPhoto, Display P3, etc.). The ICC profile declares which RGB space to interpret the decoded values as, while the Y'CbCr compression stage remains BT.601 throughout. This effectively creates non-standard Y'CbCr color spaces: the BT.601 transform applied to a different RGB base than it was designed for.

---

## Apple RGB

A legacy color space from classic Macintosh systems (pre-OS X era). The primaries were based on the phosphors in Macintosh CRT monitors. The γ=1.8 transfer function was chosen to approximate the dot gain of Apple LaserWriter printers.

**Authoritative definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6250 | 0.3400 |
| Green | 0.2800 | 0.5950 |
| Blue | 0.1550 | 0.0700 |
| White | 0.3127 | 0.3290 |

| γ | 1.8 |
|---|-----|

**Transfer function:**

```
encoded = linear^(1/1.8)
linear = encoded^1.8
```

**Derived XYZ matrices:**

```
[linearized] Apple RGB → XYZ:

| X |   | 0.4496616  0.3162561  0.1845382 |   | R |
| Y | = | 0.2446159  0.6720443  0.0833398 | × | G |
| Z |   | 0.0251811  0.1411858  0.9226909 |   | B |
```

```
XYZ → [linearized] Apple RGB:

| R |   |  2.9519785 -1.2896043 -0.4739153 |   | X |
| G | = | -1.0850836  1.9908093  0.0372017 | × | Y |
| B |   |  0.0854722 -0.2694297  1.0910277 |   | Z |
```

**Full conversion (Apple RGB → XYZ):**
1. Decode: linear = encoded^1.8
2. Matrix multiply

**Converting Apple RGB to sRGB:**

Because the primaries differ, gamma adjustment alone is incorrect. The correct path:

1. Decode Apple RGB (γ=1.8)
2. Matrix: linearized Apple RGB → XYZ
3. Matrix: XYZ → Linear RGB
4. Encode sRGB (piecewise function)

---

## CIELAB

CIE 1976 (L\*a\*b\*). A perceptual color space defined in terms of XYZ.

CIELAB requires a reference white (Xn, Yn, Zn) but does not specify which illuminant to use—that is application-dependent. Common choices:
- **D50** is standard for ICC profiles and print workflows
- **D65** is used for display-oriented workflows

Mixing reference whites produces incorrect results.

**Implementation note:** For display-oriented workflows using sRGB, use the white point XYZ derived from the sRGB chromaticity coordinates (approximately 0.9504559, 1.0, 1.0890578). This ensures sRGB neutrals map exactly to a\*=0, b\*=0.

**Definition (XYZ → Lab):**

```
L* = 116 × f(Y/Yn) − 16
a* = 500 × (f(X/Xn) − f(Y/Yn))
b* = 200 × (f(Y/Yn) − f(Z/Zn))
```

where:

```
f(t) = t^(1/3)                      if t > (6/29)³
f(t) = (1/3)(29/6)²t + 4/29         otherwise
```

The constants:
- Threshold: (6/29)³ ≈ 0.008856
- Linear slope: (1/3)(29/6)² ≈ 7.787
- Linear offset: 4/29 ≈ 0.137931

**Inverse (Lab → XYZ):**

```
Y = Yn × f⁻¹((L* + 16) / 116)
X = Xn × f⁻¹((L* + 16) / 116 + a*/500)
Z = Zn × f⁻¹((L* + 16) / 116 − b*/200)
```

where:

```
f⁻¹(t) = t³                         if t > 6/29
f⁻¹(t) = 3(6/29)²(t − 4/29)         otherwise
```

---

## CIELAB Color Difference Formulas

CIELAB defines a color space, but measuring the "distance" between two colors requires a separate formula. Three standard formulas exist, each improving on the previous:

### CIE76 (ΔE\*ab)

The original CIELAB color difference formula (CIE 1976). Simple Euclidean distance in L\*a\*b\* space.

**Definition:**

```
ΔE*ab = √((L₂* - L₁*)² + (a₂* - a₁*)² + (b₂* - b₁*)²)
```

**Constants:** None. This is pure Euclidean distance.

**Limitations:** CIELAB is not perfectly perceptually uniform. CIE76 over-weights chromatic differences relative to lightness differences, especially for saturated colors. Equal ΔE values do not represent equal perceptual differences across the color space.

---

### CIE94 (ΔE\*94)

CIE 1994 color difference formula. Introduces weighting factors that scale chromatic differences based on chroma magnitude, improving perceptual uniformity.

**Definition:**

```
ΔE*94 = √((ΔL*/kL·SL)² + (ΔC*/kC·SC)² + (ΔH*/kH·SH)²)
```

Where:
- ΔL\* = L₂\* - L₁\*
- C₁\* = √(a₁\*² + b₁\*²)
- C₂\* = √(a₂\*² + b₂\*²)
- ΔC\* = C₂\* - C₁\*
- ΔH\* = √((a₂\* - a₁\*)² + (b₂\* - b₁\*)² - ΔC\*²)

**Primary constants (graphic arts application):**

| Constant | Value | Description |
|----------|-------|-------------|
| kL | 1 | Lightness weighting (2 for textiles) |
| kC | 1 | Chroma weighting |
| kH | 1 | Hue weighting |
| K₁ | 0.045 | Chroma scaling coefficient |
| K₂ | 0.015 | Hue scaling coefficient |

**Derived weighting functions:**

```
SL = 1
SC = 1 + K₁ × C₁* = 1 + 0.045 × C₁*
SH = 1 + K₂ × C₁* = 1 + 0.015 × C₁*
```

**Note:** CIE94 uses the first color's chroma (C₁\*) as the reference for weighting. This makes the formula asymmetric—swapping the two colors may yield a different result. For consistent results, use the reference/standard color as color 1.

---

### CIEDE2000 (ΔE₀₀)

CIE 2000 color difference formula. The most accurate perceptual color difference metric, incorporating corrections for:
- Lightness dependence
- Chroma dependence
- Hue dependence
- Interaction between chroma and hue differences
- Hue rotation term for the blue region

**Definition:**

```
ΔE₀₀ = √((ΔL'/(kL·SL))² + (ΔC'/(kC·SC))² + (ΔH'/(kH·SH))² + RT·(ΔC'/(kC·SC))·(ΔH'/(kH·SH)))
```

**Primary constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| kL | 1 | Lightness parametric factor |
| kC | 1 | Chroma parametric factor |
| kH | 1 | Hue parametric factor |
| 25⁷ | 6103515625 | Chroma correction threshold (exact) |

**Step 1 — Calculate CIELAB chroma and mean chroma:**

```
C₁* = √(a₁*² + b₁*²)
C₂* = √(a₂*² + b₂*²)
C̄* = (C₁* + C₂*) / 2
```

**Step 2 — Calculate G factor (a\* axis adjustment):**

The G factor stretches the a\* axis to improve uniformity for low-chroma colors.

```
G = 0.5 × (1 - √(C̄*⁷ / (C̄*⁷ + 25⁷)))
```

**Step 3 — Adjusted a' values and chroma:**

```
a₁' = a₁* × (1 + G)
a₂' = a₂* × (1 + G)

C₁' = √(a₁'² + b₁*²)
C₂' = √(a₂'² + b₂*²)
```

**Step 4 — Hue angles (in degrees, 0-360):**

```
h₁' = atan2(b₁*, a₁') mod 360°    (undefined if C₁' = 0)
h₂' = atan2(b₂*, a₂') mod 360°    (undefined if C₂' = 0)
```

**Step 5 — Differences:**

```
ΔL' = L₂* - L₁*
ΔC' = C₂' - C₁'
```

For Δh' (hue difference):
```
If C₁'×C₂' = 0:     Δh' = 0
Else if |h₂'-h₁'| ≤ 180°:  Δh' = h₂' - h₁'
Else if h₂'-h₁' > 180°:    Δh' = h₂' - h₁' - 360°
Else:                       Δh' = h₂' - h₁' + 360°
```

```
ΔH' = 2 × √(C₁'×C₂') × sin(Δh'/2)
```

**Step 6 — Mean values:**

```
L̄' = (L₁* + L₂*) / 2
C̄' = (C₁' + C₂') / 2
```

For h̄' (mean hue):
```
If C₁'×C₂' = 0:            h̄' = h₁' + h₂'
Else if |h₁'-h₂'| ≤ 180°:  h̄' = (h₁' + h₂') / 2
Else if h₁'+h₂' < 360°:    h̄' = (h₁' + h₂' + 360°) / 2
Else:                       h̄' = (h₁' + h₂' - 360°) / 2
```

**Step 7 — T factor (hue-dependent weighting):**

```
T = 1 - 0.17×cos(h̄' - 30°) + 0.24×cos(2h̄') + 0.32×cos(3h̄' + 6°) - 0.20×cos(4h̄' - 63°)
```

**Derived constants in T factor:**

| Angle (degrees) | Radians (derived) |
|-----------------|-------------------|
| 30° | π/6 = 0.52359877559829887307... |
| 6° | π/30 = 0.10471975511965977461... |
| 63° | 7π/20 = 1.09955742875642894663... |

**Step 8 — SL, SC, SH weighting functions:**

```
SL = 1 + (0.015 × (L̄' - 50)²) / √(20 + (L̄' - 50)²)
SC = 1 + 0.045 × C̄'
SH = 1 + 0.015 × C̄' × T
```

**Primary constants in weighting functions:**

| Constant | Value | Appears in | Notes |
|----------|-------|------------|-------|
| K₁ | 0.045 | SC | Shared with CIE94 |
| K₂ | 0.015 | SL, SH | Shared with CIE94 |
| L* midpoint | 50 | SL | Perceptual midpoint |
| SL denom offset | 20 | SL | Empirical constant |

**Step 9 — RT rotation term (blue region correction):**

```
Δθ = 30° × exp(-((h̄' - 275°)/25°)²)
RC = 2 × √(C̄'⁷ / (C̄'⁷ + 25⁷))
RT = -sin(2Δθ) × RC
```

**Derived constants:**

| Expression | Value |
|------------|-------|
| 25⁷ | 6103515625 (exact) |
| 275° in formula | 275 (exact, degrees) |
| 25° in formula | 25 (exact, degrees) |
| 30° in Δθ | 30 (exact, degrees) |

**Step 10 — Final calculation:**

With kL = kC = kH = 1:

```
ΔE₀₀ = √((ΔL'/SL)² + (ΔC'/SC)² + (ΔH'/SH)² + RT×(ΔC'/SC)×(ΔH'/SH))
```

**Implementation note:** The RT term can be negative, and the expression under the square root can theoretically become slightly negative due to floating-point error. Implementations should clamp to zero before taking the square root.

---

### Comparison of CIELAB Distance Formulas

| Formula | Year | Complexity | Perceptual Accuracy | Use Case |
|---------|------|------------|---------------------|----------|
| CIE76 | 1976 | O(1), trivial | Poor for saturated colors | Fast approximation |
| CIE94 | 1994 | O(1), simple | Good | General purpose |
| CIEDE2000 | 2000 | O(1), complex | Best available | Color-critical applications |

For dithering applications, CIE94 offers a good balance of accuracy and computational cost. CIEDE2000 is more expensive due to trigonometric functions but provides the most accurate perceptual distance. OKLab with Euclidean distance is an alternative that achieves similar perceptual uniformity with simpler computation.

---

## OKLab

Designed by Björn Ottosson (2020). A perceptual color space optimized for uniformity using modern perceptual data.

OKLab is defined directly in terms of Linear RGB, bypassing XYZ. The matrices were numerically optimized for perceptual uniformity rather than derived from colorimetric principles. All three matrices and the cube root are the definition—none are derived.

**Definition (Linear RGB → OKLab):**

Step 1 — Approximate cone response:
```
| l |   | 0.4122214708  0.5363325363  0.0514459929 |   | R |
| m | = | 0.2119034982  0.6806995451  0.1073969566 | × | G |
| s |   | 0.0883024619  0.2817188376  0.6299787005 |   | B |
```

Step 2 — Cube root:
```
l' = ∛l
m' = ∛m
s' = ∛s
```

Step 3 — Opponent channels:
```
| L |   | 0.2104542553  0.7936177850 -0.0040720468 |   | l' |
| a | = | 1.9779984951 -2.4285922050  0.4505937099 | × | m' |
| b |   | 0.0259040371  0.7827717662 -0.8086757660 |   | s' |
```

**Inverse (OKLab → Linear RGB):**

Step 1:
```
| l' |   | 1.0000000000  0.3963377774  0.2158037573 |   | L |
| m' | = | 1.0000000000 -0.1055613458 -0.0638541728 | × | a |
| s' |   | 1.0000000000 -0.0894841775 -1.2914855480 |   | b |
```

Step 2:
```
l = l'³
m = m'³
s = s'³
```

Step 3:
```
| R |   |  4.0767416621 -3.3077115913  0.2309699292 |   | l |
| G | = | -1.2684380046  2.6097574011 -0.3413193965 | × | m |
| B |   | -0.0041960863 -0.7034186147  1.7076147010 |   | s |
```

---

## Dependency Graph

```
CIE XYZ (empirical root)
    │
    ├── CIELAB ─────────┬── CIE76 (ΔE*ab)
    │                   ├── CIE94 (ΔE*94)
    │                   └── CIEDE2000 (ΔE₀₀)
    │
    ├── Linear RGB
    │       │
    │       ├── sRGB
    │       │     │
    │       │     └── Y'CbCr
    │       │
    │       ├── Gamma 2.2 RGB
    │       └── OKLab
    │
    └── Apple RGB
```

---

## Summary

| Space/Formula | Authoritative Definition | Derived |
|---------------|-------------------------|---------|
| CIE XYZ | Color matching functions (empirical) | — |
| Linear RGB / sRGB | Chromaticity coordinates (primaries + D65 white point) | XYZ matrices |
| sRGB | Linear RGB + piecewise transfer function | — |
| Gamma 2.2 RGB | Linear RGB + γ=2.2 | — |
| Y'CbCr | sRGB + matrix transform | — |
| Apple RGB | Chromaticity coordinates + γ=1.8 | XYZ matrices |
| CIELAB | XYZ + transform equations + reference white | — |
| CIE76 | CIELAB + Euclidean distance | — |
| CIE94 | CIELAB + kL, kC, kH, K₁=0.045, K₂=0.015 | SC, SH |
| CIEDE2000 | CIELAB + kL, kC, kH, 25⁷, K₁, K₂, etc. | G, T, SL, SC, SH, RT |
| OKLab | Linear RGB + defined matrices + cube root | — |

---

## Appendix: Bradford Chromatic Adaptation

When converting between color spaces with different white points (e.g., ProPhoto RGB at D50 to sRGB at D65), a chromatic adaptation transform is required. The Bradford transform is the industry standard, used by ICC color profiles.

**Bradford matrix (XYZ → LMS):**

```
| L |   |  0.8951   0.2664  -0.1614 |   | X |
| M | = | -0.7502   1.7135   0.0367 | × | Y |
| S |   |  0.0389  -0.0685   1.0296 |   | Z |
```

From Lam (1985) and Hunt (1994). This transforms XYZ to a "sharpened" cone-like response space optimized for chromatic adaptation.

**Adaptation formula:**

To adapt from source white point (Ws) to destination white point (Wd):

```
M_adapt = BRADFORD⁻¹ × diag(LMS_Wd / LMS_Ws) × BRADFORD
```

Where `LMS_W = BRADFORD × XYZ_W` for each white point.

The resulting 3×3 matrix transforms XYZ coordinates directly from one illuminant to another. This can be precomposed with RGB↔XYZ matrices to create direct RGB-to-RGB conversions between spaces with different white points.
