# The D65 Whitepoint: Precision, History, and Derivation

## Executive Summary

D65 is the CIE standard daylight illuminant representing average noon daylight. While commonly associated with "6500K," the relationship between D65 and color temperature involves two different temperature scales: the 1931 scale on which D65 was originally defined, and the modern ITS-90 scale used in current CIE formulas. This document explains the precision issues, historical changes, and the derivation relationships between them.

| Variant | x | y | Origin |
|---------|---|---|--------|
| **CIE Official (5 dp)** | 0.31272 | 0.32903 | CIE 15:2004 Table T.3 |
| **380-780nm @ 5nm** | 0.3127205252 | 0.3290306850 | Integration per Table T.3 method (81 points) |
| **380-780nm @ 1nm** | 0.3127385128 | 0.3290520326 | 1nm interpolated SPD (401 points) |
| **360-830nm @ 1nm** | 0.3127268710 | 0.3290232066 | Full range 1nm data (471 points) |
| **sRGB matrix-implied** | 0.3127159072 | 0.3290014805 | Row sums of IEC 61966-2-1 matrix |
| **4-digit (specs)** | 0.3127 | 0.3290 | ITU-R BT.709, Adobe RGB (1998) |
| **Polynomial at 6503.62K (ITS-90)** | 0.312720273260 | 0.329125276333 | CIE daylight polynomial with temperature conversion |
| **Reconstructed via M₁,M₂** | 0.3127089233 | 0.3289234905 | Basis function reconstruction |

---

## 1. The Authoritative Definition

D65 is defined by the CIE as a **tabulated spectral power distribution (SPD)**—a table of relative spectral radiance values from 300–830nm, published with 6 significant figures in CIE S 005-1998 and ISO 11664-2:2007. A commonly-referenced 5nm version is provided in Appendix A for reference; the authoritative 1nm tables from the ISO standards should be used for high-precision work.

From this SPD, the chromaticity coordinates are derived by integration against the CIE 1931 2° standard observer color matching functions:
```
x = 0.31272
y = 0.32903
```

These coordinates are published to 5 decimal places. **The SPD is the primary definition; the chromaticity is derived from it.**

### 1.1 Deriving Chromaticity from SPD

The tristimulus values are computed by integrating the SPD against the color matching functions:

```
X = k · Σ S(λ)x̄(λ)Δλ
Y = k · Σ S(λ)ȳ(λ)Δλ
Z = k · Σ S(λ)z̄(λ)Δλ
```

Where k is chosen such that Y = 100 for the perfect reflecting diffuser:

```
k = 100 / Σ S(λ)ȳ(λ)Δλ
```

The chromaticity coordinates are then:

```
x = X / (X + Y + Z)
y = Y / (X + Y + Z)
```

### 1.2 Verification Against Official Values

CIE 15:2004 Table T.3 specifies that the official chromaticity was computed using "5 nm intervals over the range 380 nm to 780 nm." We can verify this by deriving chromaticity from the authoritative 1nm source data using different sampling configurations.

#### Source Data Used

| Source | Data | Notes |
|--------|------|-------|
| CIE 15:2004 Table T.1 | D65 SPD (5nm, 300-830nm) | Public domain, with historical rounding |
| ISO 11664-2:2007 / CIE S 014-2:2006 Table 1 | D65 SPD (1nm, 300-830nm) | **Authoritative SPD** (used for derivations) |
| CIE 018:2019 Table 6 | CMF (1nm, 360-830nm) | **Authoritative CMFs** (used for derivations) |

#### Derivation Results Comparison

Using the authoritative 1nm source data (ISO 11664-2:2007 SPD + CIE 018:2019 CMF):

| Configuration | X | Y | Z | x | y |
|---------------|---|---|---|---|---|
| **Official CIE 15:2004** | 95.04 | 100.00 | 108.88 | 0.31272 | 0.32903 |
| 380-780nm @ 5nm | 95.0430 | 100.0000 | 108.8801 | 0.3127205252 | 0.3290306850 |
| 380-780nm @ 1nm | 95.0423 | 100.0000 | 108.8610 | 0.3127385128 | 0.3290520326 |
| 360-830nm @ 1nm (full) | 95.0471 | 100.0000 | 108.8829 | 0.3127268710 | 0.3290232066 |

#### Error Analysis (×10⁻⁵ from official)

| Configuration | Δx | Δy | Notes |
|---------------|----|----|-------|
| 380-780nm @ 5nm | +0.05 | +0.07 | **Matches official method** |
| 380-780nm @ 1nm | +1.85 | +2.20 | Finer sampling introduces systematic shift |
| 360-830nm @ 1nm | +0.69 | −0.68 | Extended range partially compensates |

**Key Finding:** The 5nm sampling at 380-780nm reproduces the official chromaticity to within 0.07×10⁻⁵, confirming CIE 15:2004's statement about its derivation method. Using 1nm data introduces small systematic differences due to interpolation and integration effects.

Rounded to 5 decimal places: **(0.31272, 0.32903)** ✓

---

## 2. The Three Values of c₂

The second radiation constant (c₂) in Planck's law has three relevant values:

| Constant | Value | Source | Used For |
|----------|-------|--------|----------|
| Old c₂ | 0.01438 m·K | 1931 CIE definition | D65 tabulated values |
| ITS-90 c₂ | 0.014388 m·K | International Temperature Scale 1990 | CIE 15:2004 polynomial |
| CODATA c₂ | 0.01438776877 m·K | Physical measurement | **Not used by CIE** |

The CODATA value represents the best physical measurement of c₂, but CIE colorimetry deliberately uses the ITS-90 defined constant for temperature scale interoperability—not for physical accuracy.

### 2.1 Temperature Scale Effects

This change affects how **Correlated Color Temperature (CCT)** is calculated. When c₂ changed, the Planckian locus shifted slightly, changing all CCT values.

**The D65 chromaticity did not change**—it is defined by its SPD. But the CCT assigned to that chromaticity changed:

| Description | Temperature |
|-------------|-------------|
| D65's old CCT (1931 scale) | 6500K |
| D65's new CCT (ITS-90 scale) | ~6504K |

The name "D65" is a historical artifact referring to the old CCT.

---

## 3. The Root Data: Two Independent Measurement Campaigns

The entire CIE colorimetric system for daylight rests on two independent sets of empirical measurements:

### 3.1 Color Matching Experiments (1920s–1930s)

The CIE 1931 2° standard observer color matching functions (CMFs) originate from experiments by W.D. Wright (1928–29) and J. Guild (1931). In these experiments, human observers adjusted mixtures of three primary lights to match monochromatic test lights across the visible spectrum.

The original data was:
- Measured at 5–10nm intervals
- Recorded with ~3 significant figures
- Based on different primary wavelengths than the final CIE system

The published CMFs underwent extensive processing:
1. **Transformation** from experimental primaries to theoretical CIE RGB, then to XYZ
2. **Smoothing** to remove measurement noise
3. **Interpolation** to generate 1nm tables from sparser measurements

The 5nm CMF values (CIE 15:2004 Table T.2) are closer to "authoritative" than 1nm tables, which contain interpolated precision beyond the original measurements.

### 3.2 Daylight Spectral Measurements (1960s)

By 1964, three research groups had independently measured the spectral power distribution of natural daylight:

| Researcher(s) | Location | Samples |
|---------------|----------|---------|
| H.W. Budde | National Research Council of Canada, Ottawa | 99 |
| H.R. Condit & F. Grum | Eastman Kodak Company, Rochester, NY | 249 |
| S.T. Henderson & D. Hodgkiss | Thorn Electrical Industries, Enfield, UK | 274 |
| **Total** | | **622** |

These measurements were:
- Spectral power distributions of skylight and sunlight-plus-skylight
- Recorded at **10nm intervals** from **330–700nm**
- The raw empirical data underlying all D-series illuminants

### 3.3 The Two Roots

| Root | What Was Measured | When | Leads To |
|------|-------------------|------|----------|
| Color matching experiments | Human perception | 1920s–30s | CMF (x̄, ȳ, z̄) |
| Daylight spectral measurements | Physical light (spectroradiometer) | 1960s | Basis functions (S₀, S₁, S₂) → D-illuminants |

Everything else in the CIE daylight system is **derived** from these two independent measurement campaigns—one characterizing human vision, the other characterizing physical daylight.

---

## 4. How D65 Was Constructed

D65 is not a direct measurement of any particular sky. It is a **synthetic illuminant**—a mathematical idealization of "average daylight" constructed through the following process.

### 4.1 The 1964 Analysis

Judd, MacAdam, Wyszecki, and colleagues analyzed the 622 daylight samples and made two key discoveries:

**Discovery 1: The Daylight Locus**

The chromaticity coordinates of the 622 samples clustered around a simple quadratic curve:

```
y = -3.000x² + 2.870x - 0.275
```

This curve, slightly offset from (greener than) the Planckian blackbody locus, became known as the **daylight locus**.

**Discovery 2: Principal Component Analysis**

Characteristic vector analysis (PCA) revealed that the 622 SPDs could be approximated using only three basis functions:

```
S(λ) = S₀(λ) + M₁·S₁(λ) + M₂·S₂(λ)
```

Where:
- **S₀(λ)** = mean of all 622 SPD samples
- **S₁(λ)** = first principal component (yellow-blue variation)
- **S₂(λ)** = second principal component (pink-green variation)

These basis functions capture nearly all the variance in natural daylight spectra using just two free parameters (M₁, M₂).

### 4.2 The D65 Derivation Chain

With the analysis complete, D65 was constructed as follows:

```
Step 1: Choose temperature
        T = 6500K (1931 scale) — representing "average daylight"
              ↓
Step 2: Compute chromaticity on daylight locus
        x from temperature (tabulated by Judd et al.)
        y from quadratic: y = -3.000x² + 2.870x - 0.275
              ↓
Step 3: Compute M coefficients from chromaticity
        M₁, M₂ from (x, y) using Equation 3.6
              ↓
Step 4: Reconstruct SPD from basis functions
        S(λ) = S₀(λ) + M₁·S₁(λ) + M₂·S₂(λ)
        (This gives 330–700nm at 10nm)
              ↓
Step 5: Extend wavelength range
        UV (300–330nm) and IR (700–830nm) added using
        Moon's spectral absorbance data of Earth's atmosphere
              ↓
Step 6: Interpolate to 5nm
        Linear interpolation from 10nm to 5nm
              ↓
Step 7: Tabulate and freeze
        The resulting SPD becomes the authoritative D65 definition
```

### 4.3 Why D65 Cannot Be Exactly Reproduced

The tabulated D65 SPD is a **frozen artifact** of the 1964 computation. Attempting to reproduce it using the published formulas yields small differences (~0.2 units) because:

1. **Original PCA** was performed on 622 measurements at 10nm
2. **Basis functions** were tabulated with limited precision
3. **M coefficient formulas** are fitted approximations, not exact inversions
4. **Intermediate rounding** occurred before final tabulation
5. **UV/IR extension** used separate atmospheric data spliced in
6. **10nm → 5nm interpolation** was applied as a final step

The tabulated SPD is now authoritative precisely *because* you cannot perfectly reconstruct it from the component formulas. CIE 15:2004 Note 4 explicitly acknowledges these historical rounding differences.

### 4.4 The 1nm Tables

The authoritative 1nm D65 tables (ISO 11664-2:2007) are themselves interpolations from the 5nm data—they add apparent precision without adding real information from the original measurements.

---

## 5. Daylight vs. Blackbody

A common misconception is that D65 represents a 6500K blackbody. It does not.

**Blackbody (Planckian) radiation** follows Planck's law exactly. A 6500K blackbody has chromaticity approximately (0.313, 0.324)—notably different from D65's (0.31272, 0.32903).

**Daylight** differs from blackbody radiation due to:
- Rayleigh scattering (blue sky)
- Atmospheric absorption bands
- Aerosol and cloud scattering

The D65 chromaticity lies above the Planckian locus in the y-direction. This represents the physical difference between actual daylight and idealized blackbody radiation.

The **daylight locus** is a separate curve through chromaticity space, derived empirically from the 622 measurements of real daylight, that runs roughly parallel to but offset from the Planckian locus.

---

## 6. The Daylight Locus Formulas

### 6.1 Origin

As described in Section 4, Judd, MacAdam, and Wyszecki (1964) derived these formulas from analysis of 622 daylight samples.

### 6.2 The y(x) Quadratic (CIE 15:2004 Equation 3.2)

```
y_D = -3.000x_D² + 2.870x_D - 0.275
```

This defines the shape of the daylight locus—all daylight chromaticities lie approximately on this curve. It was a best-fit curve through the 622 measured chromaticities.

### 6.3 The x(T) Polynomial

The original 1964 paper tabulated chromaticity coordinates for specific temperatures (5500K, 6500K, 7500K, etc.). The CIE later added a polynomial to compute x from temperature, enabling generation of arbitrary D-illuminants. As published in CIE 15:2004:

**For 4000K ≤ T_cp ≤ 7000K (Equation 3.3):**
```
x_D = -4.6070×10⁹/T³ + 2.9678×10⁶/T² + 0.09911×10³/T + 0.244063
```

**For 7000K < T_cp ≤ 25000K (Equation 3.4):**
```
x_D = -2.0064×10⁹/T³ + 1.9018×10⁶/T² + 0.24748×10³/T + 0.237040
```

Where **T_cp is the correlated color temperature on the ITS-90 scale**.

### 6.4 The SPD Reconstruction Formula (CIE 15:2004 Equation 3.5)

Once chromaticity (x_D, y_D) is determined, the relative spectral power distribution is computed from three basis functions:

```
S(λ) = S₀(λ) + M₁·S₁(λ) + M₂·S₂(λ)
```

Where S₀(λ), S₁(λ), S₂(λ) are the daylight basis functions (Appendix B), and M₁, M₂ are coefficients derived from the chromaticity (CIE 15:2004 Equation 3.6):

```
M₁ = (-1.3515 - 1.7703x_D + 5.9114y_D) / (0.0241 + 0.2562x_D - 0.7341y_D)

M₂ = (0.0300 - 31.4424x_D + 30.0717y_D) / (0.0241 + 0.2562x_D - 0.7341y_D)
```

**For D65** (x_D = 0.31272, y_D = 0.32903):
```
M₁ = -0.2907
M₂ = -0.6687
```

### 6.5 Verification: Reconstructed SPD vs. Tabulated D65

Using M₁ = -0.2907, M₂ = -0.6687 to reconstruct the D65 SPD:

| λ (nm) | Tabulated | Reconstructed | Δ |
|--------|-----------|---------------|--------|
| 380 | 49.9755 | 50.2020 | +0.2265 |
| 400 | 82.7549 | 82.9191 | +0.1642 |
| 450 | 117.0080 | 117.1030 | +0.0950 |
| 500 | 109.3540 | 109.3937 | +0.0397 |
| 550 | 104.0460 | 104.0483 | +0.0023 |
| 560 | 100.0000 | 100.0000 | +0.0000 |
| 600 | 90.0062 | 90.0463 | +0.0401 |
| 650 | 80.0268 | 80.1292 | +0.1024 |
| 700 | 71.6091 | 71.7470 | +0.1379 |
| 780 | 63.3828 | 63.4763 | +0.0935 |

Maximum difference: +0.23 at 380nm. The M coefficients correctly reconstruct D65 to within 0.3 units across the visible spectrum.

The small differences are due to rounding in the tabulated values, which CIE 15:2004 Note 4 explicitly acknowledges.

---

## 7. The Critical Distinction: Tables vs. Polynomial

CIE 15:2004 Note 4 (Section 3.1) states:

> "The relative spectral power distributions of the D illuminants given in Table T.1 and in the CIE standard on illuminants for colorimetry (CIE, 1998c) were derived by the procedure given above with some intermediate rounding and **with some adjustments for changes in the International Temperature Scale**. Thus for historic reasons, the tabulated values are slightly different from the calculated values. For the time being **the tabulated values are the official data**."

This reveals the key distinction:

| Component | Temperature Scale | Status |
|-----------|------------------|--------|
| Tabulated D65 values (SPD, chromaticity) | 1931 scale | **Authoritative** |
| x(T) polynomial in CIE 15:2004 | ITS-90 scale | Computational tool |

The tables are frozen historical artifacts computed with the 1931 temperature scale. The polynomial expects modern ITS-90 CCT values. This is why there's a mismatch when using nominal temperatures.

---

## 8. Converting Between Temperature Scales

Per CIE 15:2004 Appendix E, to convert from the 1931 scale to ITS-90:

```
T_new = T_old × (c₂_ITS90 / c₂_1931)
T_new = T_old × (0.014388 / 0.01438)
T_new = T_old × 1.00055632823
```

For D65:
```
T_new = 6500 × 1.00055632823 = 6503.616134K
```

This is the temperature to input into the modern polynomial to recover D65's original chromaticity.

---

## 9. Empirical Proof: ITS-90 vs. CODATA

One might ask: why use the ITS-90 defined constant (0.014388) rather than the more accurate CODATA physical measurement (0.01438776877)? We can test this empirically.

### 9.1 Two Candidate Conversions

| Method | c₂ value | Conversion | Result |
|--------|----------|------------|--------|
| ITS-90 | 0.014388 m·K | 6500 × (0.014388/0.01438) | 6503.616134K |
| CODATA | 0.01438776877 m·K | 6500 × (0.01438776877/0.01438) | 6503.511614K |

### 9.2 Error Comparison

| Temperature | Δx (×10⁻⁵) | Δy (×10⁻⁵) | Euclidean (×10⁻⁵) |
|-------------|------------|------------|-------------------|
| 6503.6161K (ITS-90) | **+0.03** | +9.53 | 9.53 |
| 6503.5116K (CODATA) | +0.20 | +9.70 | 9.70 |

### 9.3 Conclusion

The ITS-90 conversion gives nearly perfect x accuracy (error 0.03×10⁻⁵), while CODATA gives slightly worse x accuracy (error 0.20×10⁻⁵). This proves:

1. **The polynomial was calibrated for ITS-90**, not CODATA physical constants
2. **The tabulated D65 was computed with the 1931 scale** (6500K nominal)
3. **The CIE deliberately chose ITS-90** for temperature scale interoperability, not physical accuracy

This is consistent with CIE 15:2004 Appendix E, which explicitly specifies ITS-90.

---

## 10. Experimental Verification

Testing the polynomial against official D65 coordinates (0.31272, 0.32903):

| Temperature | Δx (×10⁻⁵) | Δy (×10⁻⁵) | Euclidean (×10⁻⁵) |
|-------------|------------|------------|-------------------|
| 6500.0K (nominal 1931) | +5.89 | +15.35 | 16.44 |
| 6503.6161K (ITS-90) | +0.03 | +9.53 | 9.53 |
| 6503.5116K (CODATA) | +0.20 | +9.70 | 9.70 |
| 6504.0K (common rounded) | −0.59 | +8.91 | 8.93 |
| 6503.6330K (exact x match) | +0.00 | +9.50 | 9.50 |
| 6509.5412K (exact y match) | −9.56 | +0.00 | 9.56 |
| 6506.6680K (min. Euclidean) | −4.91 | +4.62 | 6.74 |

### Key Observations

1. **The ITS-90 conversion (6503.62K) gives near-perfect x match** (error 0.03×10⁻⁵), confirming the x(T) polynomial expects ITS-90 CCT as specified in CIE 15:2004.

2. **The y error persists** (~9.5×10⁻⁵) regardless of temperature choice. This is inherent error in the y(x) quadratic itself.

3. **No temperature produces exact D65.** The temperatures for exact x match (6503.6K) and exact y match (6509.5K) differ by ~5.9K. The official D65 point does not lie exactly on the daylight locus curve.

4. **The mismatch is acknowledged by the CIE (Section 3.1):** "These equations will give an illuminant whose correlated colour temperature is approximately equal to the nominal value, but not exactly so."

---

## 11. The y(x) Quadratic Error

The ~9.5×10⁻⁵ y error is **not** a temperature scale issue—it's inherent in the y(x) quadratic formula itself. This can be verified by plugging the official x values directly into the formula:

| Illuminant | x (official) | y (official) | y (from formula) | Δy (×10⁻⁵) |
|------------|--------------|--------------|------------------|------------|
| D50 | 0.34567 | 0.35850 | 0.35861 | +10.97 |
| D55 | 0.33242 | 0.34743 | 0.34754 | +10.62 |
| D65 | 0.31272 | 0.32903 | 0.32913 | +9.50 |
| D75 | 0.29902 | 0.31485 | 0.31495 | +9.85 |

**Mean absolute y error: ~10.2×10⁻⁵**

The y(x) quadratic was a best-fit curve through the 622 daylight measurements. The canonical illuminants (D50, D55, D65, D75) were computed separately using the S₀, S₁, S₂ basis functions with intermediate rounding. The quadratic was never constrained to pass exactly through these points.

---

## 12. Why the Canonical Illuminants Don't Lie on the Polynomial Curve

Several factors contribute:

1. **Independent derivation:** The canonical illuminants were computed using the S₀, S₁, S₂ basis functions (CIE 15:2004 Equation 3.5), while the y(x) quadratic was fit separately to the original 622 measurements.

2. **Intermediate rounding:** The original computations involved rounding steps not captured by the polynomial.

3. **Temperature scale adjustments:** The polynomial was adjusted for ITS-90, but the tables were frozen.

4. **Historical layering:** The tabulated SPDs predate the polynomial formalization; the polynomial is a later interpolation tool.

---

## 13. The Complete Derivation Chain

The D65 chromaticity can be derived through multiple paths:

### Path A: SPD → Integration → Chromaticity (AUTHORITATIVE)
```
D65 SPD (Table T.1) → integrate with CIE 1931 CMFs → X,Y,Z → (0.31272, 0.32903)
```

### Path B: Temperature → Polynomial → Chromaticity
```
6500K (1931) → 6503.62K (ITS-90) → x(T) polynomial → y(x) quadratic → (0.31272, 0.32913)
```
Note: ~9.5×10⁻⁵ y error from quadratic approximation.

### Path C: Temperature → Chromaticity → M₁,M₂ → Basis → SPD
```
6503.62K → (x,y) → M₁,M₂ → S₀ + M₁·S₁ + M₂·S₂ → reconstructed SPD
```

### Verification of Paths

| Method | x | y |
|--------|---|---|
| Official CIE D65 | 0.31272 | 0.32903 |
| From tabulated SPD (Path A) | 0.3127212427 | 0.3290303382 |
| From polynomial at 6503.62K (Path B) | 0.3127202733 | 0.3291252763 |
| From reconstructed SPD via M₁,M₂ (Path C) | 0.3127089233 | 0.3289234905 |

**Path A (tabulated SPD) is authoritative.** Paths B and C are computational tools for interpolation and arbitrary D-illuminant generation.

---

## 14. High-Precision Reference Values

### Radiation Constants (CIE 15:2004 Appendix E)

| Description | Value |
|-------------|-------|
| Old c₂ (1931) | 0.01438 m·K |
| New c₂ (ITS-90) | 0.014388 m·K |
| CODATA c₂ (not used) | 0.01438776877 m·K |
| Ratio (ITS-90/1931) | 1.00055632823 |

### Temperature Conversions

| Illuminant | Nominal (1931) | ITS-90 CCT |
|------------|----------------|------------|
| D50 | 5000K | 5002.7816K |
| D55 | 5500K | 5503.0598K |
| D65 | 6500K | 6503.6161K |
| D75 | 7500K | 7504.1725K |

### Official Chromaticity Coordinates (CIE 15:2004 Table T.3)

| Illuminant | x | y |
|------------|-------|-------|
| D50 | 0.34567 | 0.35850 |
| D55 | 0.33242 | 0.34743 |
| D65 | 0.31272 | 0.32903 |
| D75 | 0.29902 | 0.31485 |

### Derived D65 Chromaticity from Authoritative Source Data

Using ISO 11664-2:2007 D65 SPD and CIE 018:2019 CMF:

| Configuration | x | y | Δx (×10⁻⁵) | Δy (×10⁻⁵) |
|---------------|---|---|------------|------------|
| 380-780nm @ 5nm | 0.3127205252 | 0.3290306850 | +0.05 | +0.07 |
| 380-780nm @ 1nm | 0.3127385128 | 0.3290520326 | +1.85 | +2.20 |
| 360-830nm @ 1nm | 0.3127268710 | 0.3290232066 | +0.69 | −0.68 |

### Derived D65 Tristimulus Values

| Configuration | X | Y | Z |
|---------------|---|---|---|
| Official CIE 15:2004 | 95.04 | 100.00 | 108.88 |
| 380-780nm @ 5nm | 95.0430 | 100.0000 | 108.8801 |
| 380-780nm @ 1nm | 95.0423 | 100.0000 | 108.8610 |
| 360-830nm @ 1nm | 95.0471 | 100.0000 | 108.8829 |

### M₁, M₂ Coefficients for Canonical Illuminants

| Illuminant | M₁ | M₂ |
|------------|--------|--------|
| D50 | 0.0459 | -0.0270 |
| D55 | -0.1178 | -0.3418 |
| D65 | -0.2907 | -0.6687 |
| D75 | -0.4537 | -0.9746 |

### Chromaticity at Key Temperatures (D65)

| Temperature | x | y |
|-------------|------------|------------|
| 6500.0K (nominal) | 0.3127788762 | 0.3291834985 |
| 6503.6161K (ITS-90) | 0.3127202733 | 0.3291252763 |
| 6503.5116K (CODATA) | 0.3127219660 | 0.3291269584 |
| 6503.6330K (exact x) | 0.3127200000 | 0.3291250048 |
| 6504.0K (rounded) | 0.3127140569 | 0.3291190991 |

### Temperature Derived from RGB→XYZ Matrices

The white point implied by an RGB→XYZ matrix is the XYZ when R=G=B=1 (row sums). This can be converted to chromaticity and then to a corresponding temperature via the daylight locus polynomial.

| Matrix Source | x | y | T (ITS-90) | T (1931) |
|---------------|---|---|------------|----------|
| sRGB (IEC 61966-2-1) | 0.3127159072 | 0.3290014805 | 6503.8857K | 6500.2695K |
| BT.709 + CIE D65 | 0.3127200000 | 0.3290300000 | 6503.6330K | 6500.0169K |
| BT.709 + 4-digit D65 | 0.3127000000 | 0.3290000000 | 6504.8682K | 6501.2514K |

**Notes:**

1. The **sRGB matrix** (IEC 61966-2-1) uses rounded coefficients that imply a white point at **6500.27K (1931 scale)**—approximately 0.27K warmer than nominal D65.

2. **BT.709 + CIE D65** derives the matrix from exact BT.709 primaries and CIE D65 (0.31272, 0.32903). The resulting temperature (6500.02K) is essentially the nominal 6500K.

3. **BT.709 + 4-digit D65** uses the rounded (0.3127, 0.3290) white point, yielding 6501.25K—about 1.25K warmer than nominal.

4. The y-coordinates from matrices differ from polynomial-derived y by ~10×10⁻⁵ (the known quadratic approximation error).

See `tools/derive_k_from_matrix.py` for the derivation script.

---

## 15. Practical Recommendations

### For Software Implementations

| Goal | Approach |
|------|----------|
| Exact D65 | Hardcode (0.31272, 0.32903) |
| D65 via polynomial (CIE-correct) | Use T = 6503.62K (9.5×10⁻⁵ error) |
| D65 via polynomial (rounded) | Use T = 6504K (8.9×10⁻⁵ error) |
| Minimum error | Use T = 6506.7K (6.7×10⁻⁵ error) |

**Recommendation:** For D50, D55, D65, D75, use hardcoded official chromaticity values. Use the polynomial only for arbitrary D-illuminants at non-standard temperatures.

### Why Use the Polynomial?

The CIE tables are authoritative for **compliance and discrete lookups**, but the moment you need:

- Continuous interpolation
- Derivatives / gradients
- Optimization
- Smooth color space transformations

...you **must** use the polynomial. And if you're using the polynomial, you need the correct input temperature (ITS-90 scale).

### Perceptibility

All errors discussed are in the 5th decimal place (10⁻⁵), far below any perceptible threshold. The practical impact is limited to accumulated round-trip conversion errors in high-precision workflows.

### Inter-Space Conversions

When converting between color spaces that share the same nominal whitepoint (e.g., sRGB → Adobe RGB, both "D65"), use the **4-digit rounded values** specified by the actual standards:

| Standard | Whitepoint | Source |
|----------|------------|--------|
| ITU-R BT.709 (sRGB basis) | (0.3127, 0.3290) | ITU-R BT.709-6, Table 1 |
| Adobe RGB (1998) | (0.3127, 0.3290) | Adobe specification |
| Display P3 | (0.3127, 0.3290) | Apple/DCI specification |
| ITU-R BT.2020 | (0.3127, 0.3290) | ITU-R BT.2020-2, Table 4 |

**Rationale:** These standards all specify D65 with 4-digit precision. Images encoded in these spaces are defined relative to the 4-digit whitepoint, not the CIE-exact value. Using the standard-specified values ensures:

1. **Correct interpretation:** The image data was authored against the 4-digit whitepoint
2. **Exact round-trips:** sRGB → Adobe RGB → sRGB preserves neutrals perfectly
3. **Standard compliance:** Matches what other software expects

**Always use the standard-specified whitepoints for image conversions:**

| Conversion | Source Whitepoint | Destination Whitepoint |
|------------|-------------------|------------------------|
| sRGB → Adobe RGB | D65 (0.3127, 0.3290) | D65 (0.3127, 0.3290) |
| sRGB → ProPhoto RGB | D65 (0.3127, 0.3290) | D50 (0.3457, 0.3585) |
| Display P3 → BT.2020 | D65 (0.3127, 0.3290) | D65 (0.3127, 0.3290) |

**When to use CIE-exact values (0.31272, 0.32903):**

- Colorimetric calculations and color science research
- Measuring physical light sources against the D65 illuminant
- Spectral rendering and physically-based simulation

The CIE-exact values represent the theoretical illuminant; the 4-digit values represent the whitepoints that actual image standards define and that images are encoded against.

---

## 16. Summary

| Item | Temperature Scale | Notes |
|------|------------------|-------|
| D65 name ("65") | 1931 scale | Historical artifact |
| D65 tabulated SPD | 1931 scale | Authoritative definition |
| D65 chromaticity | Derived from SPD | (0.31272, 0.32903) |
| x(T) polynomial | ITS-90 | Input modern CCT |
| Input for D65 | 6503.62K (ITS-90) | Because 6500K (1931) = 6503.62K (ITS-90) |

The polynomial expects ITS-90 CCT as input. D65 was defined as 6500K on the 1931 scale. To recover D65 from the polynomial, convert: 6500 × (0.014388/0.01438) = 6503.616134K.

The empirical test in Section 9 confirms this: using ITS-90 gives x error of 0.03×10⁻⁵, while using CODATA gives x error of 0.20×10⁻⁵. The polynomial was calibrated for ITS-90, not physical constants.

The ~9.5×10⁻⁵ residual y error is inherent in the y(x) quadratic formula—it affects all canonical illuminants equally and reflects the fact that the quadratic was never constrained to pass through them exactly.

---

## Appendix A: D65 Spectral Power Distribution

The authoritative definition of D65. Values are relative spectral power, normalized to 100.000 at 560nm.

### Table A.1: Standard Illuminant D65 SPD (5nm intervals, 300–780nm)

| λ (nm) | S_D65(λ) | λ (nm) | S_D65(λ) | λ (nm) | S_D65(λ) |
|--------|----------|--------|----------|--------|----------|
| 300 | 0.034100 | 465 | 116.336 | 630 | 83.2886 |
| 305 | 1.6643 | 470 | 114.861 | 635 | 83.4939 |
| 310 | 3.2945 | 475 | 115.392 | 640 | 83.6992 |
| 315 | 11.7652 | 480 | 115.923 | 645 | 81.8630 |
| 320 | 20.236 | 485 | 112.367 | 650 | 80.0268 |
| 325 | 28.6447 | 490 | 108.811 | 655 | 80.1207 |
| 330 | 37.0535 | 495 | 109.082 | 660 | 80.2146 |
| 335 | 38.5011 | 500 | 109.354 | 665 | 81.2462 |
| 340 | 39.9488 | 505 | 108.578 | 670 | 82.2778 |
| 345 | 42.4302 | 510 | 107.802 | 675 | 80.2810 |
| 350 | 44.9117 | 515 | 106.296 | 680 | 78.2842 |
| 355 | 45.775 | 520 | 104.790 | 685 | 74.0027 |
| 360 | 46.6383 | 525 | 106.239 | 690 | 69.7213 |
| 365 | 49.3637 | 530 | 107.689 | 695 | 70.6652 |
| 370 | 52.0891 | 535 | 106.047 | 700 | 71.6091 |
| 375 | 51.0323 | 540 | 104.405 | 705 | 72.979 |
| 380 | 49.9755 | 545 | 104.225 | 710 | 74.349 |
| 385 | 52.3118 | 550 | 104.046 | 715 | 67.9765 |
| 390 | 54.6482 | 555 | 102.023 | 720 | 61.604 |
| 395 | 68.7015 | 560 | 100.000 | 725 | 65.7448 |
| 400 | 82.7549 | 565 | 98.1671 | 730 | 69.8856 |
| 405 | 87.1204 | 570 | 96.3342 | 735 | 72.4863 |
| 410 | 91.486 | 575 | 96.0611 | 740 | 75.087 |
| 415 | 92.4589 | 580 | 95.788 | 745 | 69.3398 |
| 420 | 93.4318 | 585 | 92.2368 | 750 | 63.5927 |
| 425 | 90.057 | 590 | 88.6856 | 755 | 55.0054 |
| 430 | 86.6823 | 595 | 89.3459 | 760 | 46.4182 |
| 435 | 95.7736 | 600 | 90.0062 | 765 | 56.6118 |
| 440 | 104.865 | 605 | 89.8026 | 770 | 66.8054 |
| 445 | 110.936 | 610 | 89.5991 | 775 | 65.0941 |
| 450 | 117.008 | 615 | 88.6489 | 780 | 63.3828 |
| 455 | 117.410 | 620 | 87.6987 | | |
| 460 | 117.812 | 625 | 85.4936 | | |

**Note:** The 5nm table above is reproduced from CIE 15:2004 Table T.1, which is in the public domain and contains some historical rounding. This table was derived by linear interpolation from 10nm data originally measured at 330–700nm, with UV/IR extensions from Moon's atmospheric data. For rigorous calculations, use the authoritative 1nm tables from ISO 11664-2:2007 / CIE S 014-2:2006 (531 values, 300–830nm at 6 significant figures). All high-precision derivations in this document were computed using the authoritative 1nm source data.

---

## Appendix B: Daylight Basis Functions S₀, S₁, S₂

These basis functions were derived from principal component analysis of 622 daylight measurements (Judd et al., 1964). They allow reconstruction of any daylight illuminant SPD.

### Origin

- **S₀(λ)**: The mean SPD of all 622 daylight samples
- **S₁(λ)**: First principal component (captures yellow-blue variation with color temperature)
- **S₂(λ)**: Second principal component (captures pink-green variation)

The original measurements were at 10nm intervals from 330–700nm. The basis functions were extended to 300–330nm and 700–830nm using Moon's spectral absorbance data of Earth's atmosphere, then interpolated to 5nm.

### Table B.1: Daylight Basis Functions (5nm intervals, 300–830nm)

| λ (nm) | S₀(λ) | S₁(λ) | S₂(λ) | | λ (nm) | S₀(λ) | S₁(λ) | S₂(λ) |
|--------|-------|-------|-------|-|--------|-------|-------|-------|
| 300 | 0.04 | 0.02 | 0.00 | | 490 | 113.50 | 20.10 | −1.80 |
| 305 | 3.02 | 2.26 | 1.00 | | 495 | 113.30 | 18.15 | −1.65 |
| 310 | 6.00 | 4.50 | 2.00 | | 500 | 113.10 | 16.20 | −1.50 |
| 315 | 17.80 | 13.45 | 3.00 | | 505 | 111.95 | 14.70 | −1.40 |
| 320 | 29.60 | 22.40 | 4.00 | | 510 | 110.80 | 13.20 | −1.30 |
| 325 | 42.45 | 32.20 | 6.25 | | 515 | 108.65 | 10.90 | −1.25 |
| 330 | 55.30 | 42.00 | 8.50 | | 520 | 106.50 | 8.60 | −1.20 |
| 335 | 56.30 | 41.30 | 8.15 | | 525 | 107.65 | 7.35 | −1.10 |
| 340 | 57.30 | 40.60 | 7.80 | | 530 | 108.80 | 6.10 | −1.00 |
| 345 | 59.55 | 41.10 | 7.25 | | 535 | 107.05 | 5.15 | −0.75 |
| 350 | 61.80 | 41.60 | 6.70 | | 540 | 105.30 | 4.20 | −0.50 |
| 355 | 61.65 | 39.80 | 6.00 | | 545 | 104.85 | 3.05 | −0.40 |
| 360 | 61.50 | 38.00 | 5.30 | | 550 | 104.40 | 1.90 | −0.30 |
| 365 | 65.15 | 40.20 | 5.70 | | 555 | 102.20 | 0.95 | −0.15 |
| 370 | 68.80 | 42.40 | 6.10 | | 560 | 100.00 | 0.00 | 0.00 |
| 375 | 66.10 | 40.45 | 4.55 | | 565 | 98.00 | −0.80 | 0.10 |
| 380 | 63.40 | 38.50 | 3.00 | | 570 | 96.00 | −1.60 | 0.20 |
| 385 | 64.60 | 36.75 | 2.10 | | 575 | 95.55 | −2.55 | 0.35 |
| 390 | 65.80 | 35.00 | 1.20 | | 580 | 95.10 | −3.50 | 0.50 |
| 395 | 80.30 | 39.20 | 0.05 | | 585 | 92.10 | −3.50 | 1.30 |
| 400 | 94.80 | 43.40 | −1.10 | | 590 | 89.10 | −3.50 | 2.10 |
| 405 | 99.80 | 44.85 | −0.80 | | 595 | 89.80 | −4.65 | 2.65 |
| 410 | 104.80 | 46.30 | −0.50 | | 600 | 90.50 | −5.80 | 3.20 |
| 415 | 105.35 | 45.10 | −0.60 | | 605 | 90.40 | −6.50 | 3.65 |
| 420 | 105.90 | 43.90 | −0.70 | | 610 | 90.30 | −7.20 | 4.10 |
| 425 | 101.35 | 40.50 | −0.95 | | 615 | 89.35 | −7.90 | 4.40 |
| 430 | 96.80 | 37.10 | −1.20 | | 620 | 88.40 | −8.60 | 4.70 |
| 435 | 105.35 | 36.90 | −1.90 | | 625 | 86.20 | −9.05 | 4.90 |
| 440 | 113.90 | 36.70 | −2.60 | | 630 | 84.00 | −9.50 | 5.10 |
| 445 | 119.75 | 36.30 | −2.75 | | 635 | 84.55 | −10.20 | 5.90 |
| 450 | 125.60 | 35.90 | −2.90 | | 640 | 85.10 | −10.90 | 6.70 |
| 455 | 125.55 | 34.25 | −2.85 | | 645 | 83.50 | −10.80 | 7.00 |
| 460 | 125.50 | 32.60 | −2.80 | | 650 | 81.90 | −10.70 | 7.30 |
| 465 | 123.40 | 30.25 | −2.70 | | 655 | 82.25 | −11.35 | 7.95 |
| 470 | 121.30 | 27.90 | −2.60 | | 660 | 82.60 | −12.00 | 8.60 |
| 475 | 121.30 | 26.10 | −2.60 | | 665 | 83.75 | −13.00 | 9.20 |
| 480 | 121.30 | 24.30 | −2.60 | | 670 | 84.90 | −14.00 | 9.80 |
| 485 | 117.40 | 22.20 | −2.20 | | 675 | 83.10 | −13.80 | 10.00 |

| λ (nm) | S₀(λ) | S₁(λ) | S₂(λ) | | λ (nm) | S₀(λ) | S₁(λ) | S₂(λ) |
|--------|-------|-------|-------|-|--------|-------|-------|-------|
| 680 | 81.30 | −13.60 | 10.20 | | 760 | 47.70 | −7.80 | 5.20 |
| 685 | 76.60 | −12.80 | 9.25 | | 765 | 58.15 | −9.50 | 6.30 |
| 690 | 71.90 | −12.00 | 8.30 | | 770 | 68.60 | −11.20 | 7.40 |
| 695 | 73.10 | −12.65 | 8.95 | | 775 | 66.80 | −10.80 | 7.10 |
| 700 | 74.30 | −13.30 | 9.60 | | 780 | 65.00 | −10.40 | 6.80 |
| 705 | 75.35 | −13.10 | 9.05 | | 785 | 65.50 | −10.50 | 6.90 |
| 710 | 76.40 | −12.90 | 8.50 | | 790 | 66.00 | −10.60 | 7.00 |
| 715 | 69.85 | −11.75 | 7.75 | | 795 | 63.50 | −10.15 | 6.70 |
| 720 | 63.30 | −10.60 | 7.00 | | 800 | 61.00 | −9.70 | 6.40 |
| 725 | 67.50 | −11.10 | 7.30 | | 805 | 57.15 | −9.00 | 5.95 |
| 730 | 71.70 | −11.60 | 7.60 | | 810 | 53.30 | −8.30 | 5.50 |
| 735 | 74.35 | −11.90 | 7.80 | | 815 | 56.10 | −8.80 | 5.80 |
| 740 | 77.00 | −12.20 | 8.00 | | 820 | 58.90 | −9.30 | 6.10 |
| 745 | 71.10 | −11.20 | 7.35 | | 825 | 60.40 | −9.55 | 6.30 |
| 750 | 65.20 | −10.20 | 6.70 | | 830 | 61.90 | −9.80 | 6.50 |
| 755 | 56.45 | −9.00 | 5.95 | | | | | |

### Usage

To compute any daylight illuminant at correlated color temperature T:

1. Compute x_D from T using Equation 3.3 or 3.4
2. Compute y_D from x_D using Equation 3.2
3. Compute M₁ and M₂ from (x_D, y_D) using Equation 3.6
4. Compute S(λ) = S₀(λ) + M₁·S₁(λ) + M₂·S₂(λ)

**Note:** The characteristic vectors S₁ and S₂ both have a zero at 560nm, since all relative SPDs were normalized to 100 at this wavelength before PCA. The 5nm basis functions above are reproduced from CIE 15:2004 Tables T.2 and T.3, which are in the public domain. Linear interpolation should be used if values at wavelengths other than those tabulated are needed. For highest accuracy, use the Lagrange-interpolated 1nm tables from CIE 15:2004 Appendix C.

---

## Appendix C: D65 Quick Reference for Continuous Computation

The CIE tabulated values are authoritative for compliance, but cannot be continuously interpolated. For any application requiring continuous mathematics (derivatives, optimization, smooth interpolation), use the CIE daylight locus polynomial with the following temperature:

**D65 Modern CCT (per CIE 15:2004 Appendix E):**
```
T = 6500K × (0.014388 / 0.01438) = 6503.616134K
```

| Constant | Value |
|----------|-------|
| Old c₂ (1931) | 0.01438 m·K |
| New c₂ (ITS-90) | 0.014388 m·K |
| CODATA c₂ (not used) | 0.01438776877 m·K |
| Conversion factor | 1.00055632823 |
| D65 modern CCT | 6503.616134K |

This yields a chromaticity error of ~9.5×10⁻⁵ from the official (0.31272, 0.32903)—an irreducible artifact of the y(x) quadratic approximation, not the temperature conversion.

---

## Appendix D: Source Data for Verification

### D.1 Data Sources Used in This Document

| Data | Source Document | Notes |
|------|-----------------|-------|
| D65 SPD (5nm) | CIE 15:2004 Table T.1 | Public domain (included in Appendices A & B) |
| D65 SPD (1nm) | ISO 11664-2:2007 / CIE S 014-2:2006 Table 1 | **Authoritative** (used for all derivations) |
| CMF 1931 2° (1nm) | CIE 018:2019 Table 6 | **Authoritative** (used for all derivations) |
| Official chromaticity | CIE 15:2004 Table T.3 | (0.31272, 0.32903) |

**Important:** The 5nm tables included in Appendices A and B are from CIE 15:2004 (public domain) and contain some historical rounding. All high-precision calculations in this document were performed using the authoritative 1nm tables from the ISO/CIE standards cited above.

### D.2 Data Verification Spot Checks

From ISO 11664-2:2007 D65 SPD and CIE 018:2019 CMF:

| Wavelength | D65 SPD | Expected | CMF ȳ | Expected |
|------------|---------|----------|-------|----------|
| 450nm | 117.008 | 117.008 | — | — |
| 555nm | 102.023 | — | 1.0000 | 1.0 |
| 560nm | 100.000 | 100.000 | — | — |

All spot checks confirm data integrity.

---

## Appendix E: Historical Timeline

| Year | Event |
|------|-------|
| 1928–31 | Wright and Guild conduct color matching experiments → CIE 1931 2° observer CMFs |
| 1931 | CIE establishes illuminants A, B, C; defines c₂ = 0.01438 m·K |
| 1960s | Budde, Condit & Grum, Henderson & Hodgkiss measure 622 daylight SPDs |
| 1964 | Judd, MacAdam, Wyszecki publish PCA analysis; derive S₀, S₁, S₂ basis functions and daylight locus |
| 1967 | CIE formally adopts D-series illuminants; D65 defined at 6500K (1931 scale) |
| 1968 | Planck's law constants revised; D65's CCT shifts to ~6504K on new scale |
| 1990 | ITS-90 temperature scale adopted; c₂ = 0.014388 m·K |
| 2004 | CIE 15:2004 publishes current polynomial formulas (expecting ITS-90 input) |

---

## References

- CIE 15:2004, Colorimetry, 3rd Edition
- CIE S 005-1998 / ISO 10526:1999, CIE Standard Illuminants for Colorimetry
- ISO 11664-2:2007 / CIE S 014-2:2006, Colorimetry — Part 2: CIE Standard Illuminants
- CIE 018:2019, The Basis of Physical Photometry, 3rd Edition
- Judd, D.B., MacAdam, D.L., Wyszecki, G., et al. (1964). "Spectral Distribution of Typical Daylight as a Function of Correlated Color Temperature." *J. Opt. Soc. Am.* 54, 1031-1040.
- Condit, H.R. and Grum, F. (1964). "Spectral Energy Distribution of Daylight." *J. Opt. Soc. Am.* 54, 937-944.
- Henderson, S.T. and Hodgkiss, D. (1963). "The Spectral Energy Distribution of Daylight." *British Journal of Applied Physics* 14, 125-131.
