# Color Space Reference: Extended

These color spaces are not foundationally distinct but may appear in practical workflows. Each is defined in terms of spaces from the primary specification.

---

## Display P3

Apple's wide-gamut color space, used on modern iPhones, iPads, and Macs since ~2016. Common in photos from iOS devices.

**Definition:** Same structure as sRGB, but with different primaries derived from DCI-P3 cinema standard.

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6800 | 0.3200 |
| Green | 0.2650 | 0.6900 |
| Blue | 0.1500 | 0.0600 |
| White | D65 | |

**Transfer function:** Identical to sRGB (piecewise).

**Derived XYZ conversion:**

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

**When you encounter it:** Images from Apple devices, web content using CSS `color(display-p3 ...)`.

**Relation to sRGB:** Wider gamut. sRGB colors are a subset. Converting P3→sRGB may require gamut clipping.

---

## Adobe RGB (1998)

A wider-gamut space designed for photography and print. Common in professional photo workflows.

**Definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6400 | 0.3300 |
| Green | 0.2100 | 0.7100 |
| Blue | 0.1500 | 0.0600 |
| White | D65 | |
| γ | 2.2 | |

Red and blue primaries are identical to sRGB. Only green differs (more saturated).

**Transfer function:**

```
encoded = linear^(1/2.2)
linear = encoded^2.2
```

**Derived XYZ conversion:**

```
[linearized] Adobe RGB → XYZ:

| X |   | 0.5767309  0.1855540  0.1881852 |   | R |
| Y | = | 0.2973769  0.6273491  0.0752741 | × | G |
| Z |   | 0.0270343  0.0706872  0.9911085 |   | B |
```

```
XYZ → [linearized] Adobe RGB:

| R |   |  2.0415879 -0.5650070 -0.3447314 |   | X |
| G | = | -0.9692436  1.8759675  0.0415551 | × | Y |
| B |   |  0.0134442 -0.1183624  1.0151750 |   | Z |
```

**When you encounter it:** DSLR cameras, Photoshop workflows, print-oriented photography.

**Trivia:** The green primary reportedly resulted from an accidental use of NTSC primaries adapted to the wrong white point. Adobe kept it because users liked the results.

---

## ProPhoto RGB

An extremely wide-gamut space used in high-end photography. Encompasses nearly all visible colors but includes imaginary colors that cannot exist physically.

**Definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.7347 | 0.2653 |
| Green | 0.1596 | 0.8404 |
| Blue | 0.0366 | 0.0001 |
| White | D50 | |
| γ | 1.8 | |

Note the D50 white point—different from the D65 spaces above.

**Transfer function:**

```
encoded = linear^(1/1.8)
linear = encoded^1.8
```

**When you encounter it:** Lightroom internal processing, archival photography workflows.

**Caution:** Contains imaginary colors. Some RGB triplets represent colors outside human vision.

---

## Rec.2020

Ultra-wide gamut space for HDR and UHD television. Defined in ITU-R BT.2020.

**Definition:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.7080 | 0.2920 |
| Green | 0.1700 | 0.7970 |
| Blue | 0.1310 | 0.0460 |
| White | D65 | |

**Transfer function:** Same as Rec.709/sRGB for SDR content. HDR uses PQ or HLG.

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

## Summary

| Space | Based on | Gamut vs sRGB | When you encounter it |
|-------|----------|---------------|----------------------|
| Display P3 | Linear P3 + sRGB transfer | ~25% larger | Apple devices, modern web |
| Adobe RGB | XYZ, γ=2.2 | ~40% larger | Photography, print |
| ProPhoto RGB | XYZ (D50), γ=1.8 | ~90% of visible | Lightroom, archival |
| Rec.2020 | XYZ, sRGB transfer (SDR) | ~75% of visible | UHD/HDR video |
| OKLch | OKLab (polar coords) | Same as sRGB | Hue-preserving ops |
| HSL/HSV | sRGB (coordinate transform) | Same as sRGB | UI, color pickers |
| ICtCp | Rec.2020 + PQ/HLG | Same as Rec.2020 | HDR video |

---

## Dependency Graph

Includes all spaces from both the base specification and this extended reference.

```
CIE XYZ (empirical root)
    │
    ├── CIELAB
    │
    ├── Linear RGB
    │       │
    │       ├── sRGB
    │       │     │
    │       │     ├── Y'CbCr
    │       │     ├── HSL
    │       │     └── HSV
    │       │
    │       ├── Gamma 2.2 RGB
    │       │
    │       └── OKLab
    │             │
    │             └── OKLch
    │
    ├── Display P3
    │
    ├── Adobe RGB
    │
    ├── ProPhoto RGB (D50)
    │
    ├── Rec.2020
    │       │
    │       └── ICtCp
    │
    └── Apple RGB
```
