# The D65 Whitepoint: Precision, History, and Derivation

## Executive Summary

D65 is the CIE standard daylight illuminant representing average noon daylight. While commonly associated with "6500K," the relationship between D65 and color temperature involves two different temperature scales: the pre-1968 scale on which D65 was originally defined, and the modern scale used in current CIE formulas. This document explains the precision issues, historical changes, and the derivation relationships between them.

---

## 1. The Authoritative Definition

D65 is defined by the CIE as a **tabulated spectral power distribution (SPD)**—a table of relative spectral radiance values from 300–830nm, published with 6 significant figures.

From this SPD, the chromaticity coordinates are derived by integration against the CIE 1931 2° standard observer color matching functions:
```
x = 0.31272
y = 0.32903
```

These coordinates are published to 5 decimal places. **The SPD is the primary definition; the chromaticity is derived from it.**

---

## 2. The Two Temperature Scales

In 1968, the International Practical Temperature Scale updated the second radiation constant in Planck's law:

| Constant | Value | Era |
|----------|-------|-----|
| Old c₂ | 0.01438 m·K | Pre-1968 |
| New c₂ | 0.01438776877 m·K | Post-1968 (CODATA) |

This change affects how **Correlated Color Temperature (CCT)** is calculated. CCT is defined as the temperature of the blackbody (on the Planckian locus) whose chromaticity is nearest to the light source. When c₂ changed, the Planckian locus shifted slightly, changing all CCT values.

**The D65 chromaticity did not change**—it is defined by its SPD. But the CCT assigned to that chromaticity changed:

| Description | Temperature |
|-------------|-------------|
| D65's old CCT (pre-1968) | 6500K |
| D65's new CCT (post-1968) | ~6504K |

The name "D65" is a historical artifact referring to the old CCT.

---

## 3. Daylight vs. Blackbody

A common misconception is that D65 represents a 6500K blackbody. It does not.

**Blackbody (Planckian) radiation** follows Planck's law exactly. A 6500K blackbody has chromaticity approximately (0.313, 0.324)—notably different from D65's (0.31272, 0.32903).

**Daylight** differs from blackbody radiation due to:
- Rayleigh scattering (blue sky)
- Atmospheric absorption bands
- Aerosol and cloud scattering

The D65 chromaticity lies above the Planckian locus in the y-direction by approximately Δy ≈ +0.005. This represents the physical difference between actual daylight and idealized blackbody radiation.

The **daylight locus** is a separate curve through chromaticity space, derived from measurements of real daylight, that runs roughly parallel to but offset from the Planckian locus.

---

## 4. The Daylight Locus Formulas

### Origin

Judd, MacAdam, and Wyszecki (1964) analyzed 622 daylight samples and found their chromaticities followed a quadratic relationship:

**The y(x) quadratic (Judd et al., 1964; CIE 15:2004 Equation 3.2):**
```
y_D = -3.000x_D² + 2.870x_D - 0.275
```

This defines the shape of the daylight locus—all daylight chromaticities lie approximately on this curve.

### The x(T) Polynomial

The CIE later added a polynomial to compute x from temperature, enabling generation of arbitrary D-illuminants. As published in CIE 15:2004:

**For 4000K ≤ T_cp ≤ 7000K (Equation 3.3):**
```
x_D = -4.6070×10⁹/T³ + 2.9678×10⁶/T² + 0.09911×10³/T + 0.244063
```

**For 7000K < T_cp ≤ 25000K (Equation 3.4):**
```
x_D = -2.0064×10⁹/T³ + 1.9018×10⁶/T² + 0.24748×10³/T + 0.237040
```

Where **T_cp is the correlated color temperature on the modern (post-1968) scale**.

---

## 5. The Critical Distinction: Tables vs. Polynomial

CIE 15:2004 Note 4 (Section 3.1) states:

> "The relative spectral power distributions of the D illuminants given in Table T.1 and in the CIE standard on illuminants for colorimetry (CIE, 1998c) were derived by the procedure given above with some intermediate rounding and **with some adjustments for changes in the International Temperature Scale**. Thus for historic reasons, the tabulated values are slightly different from the calculated values. For the time being **the tabulated values are the official data**."

This reveals the key distinction:

| Component | Temperature Scale | Status |
|-----------|------------------|--------|
| Tabulated D65 values (SPD, chromaticity) | Old K (pre-1968) | **Authoritative** |
| x(T) polynomial in CIE 15:2004 | New K (post-1968) | Computational tool |

The tables are frozen historical artifacts computed with old K = 6500. The polynomial was adjusted to use modern CCT. This is why there's a mismatch.

---

## 6. Converting Between Temperature Scales

To convert from old K to new K:

```
T_new = T_old × (new_c₂ / old_c₂)
T_new = T_old × (0.01438776877 / 0.01438)
T_new = T_old × 1.00054024826
```

For D65:
```
T_new = 6500 × 1.00054024826 = 6503.5116136996K
```

This is the temperature to input into the modern polynomial to recover D65's original chromaticity.

---

## 7. Experimental Verification

Testing the polynomial against official D65 coordinates (0.31272, 0.32903):

| Temperature | Δx (×10⁻⁵) | Δy (×10⁻⁵) | Euclidean (×10⁻⁵) |
|-------------|------------|------------|-------------------|
| 6500.0K (nominal old) | +5.89 | +15.35 | 16.44 |
| 6503.5116K (c₂ ratio) | +0.20 | +9.70 | 9.70 |
| 6504.0K (lcms/moxcms) | −0.59 | +8.91 | 8.93 |
| 6503.6330K (exact x match) | +0.00 | +9.50 | 9.50 |
| 6509.5412K (exact y match) | −9.56 | +0.00 | 9.56 |
| 6506.6680K (min. Euclidean) | −4.91 | +4.62 | 6.74 |

### Key Observations

1. **The c₂ ratio (6503.5K) gives near-perfect x match** (error 0.2×10⁻⁵), confirming the x(T) polynomial expects modern CCT.

2. **The y error persists** (~9.7×10⁻⁵) regardless of temperature choice. This is independent error in the y(x) quadratic.

3. **No temperature produces exact D65.** The temperatures for exact x match (6503.6K) and exact y match (6509.5K) differ by ~5.9K. The official D65 point does not lie exactly on the daylight locus curve.

4. **The mismatch is acknowledged by the CIE (Section 3.1):** "These equations will give an illuminant whose correlated colour temperature is approximately equal to the nominal value, but not exactly so."

---

## 8. Why D65 Doesn't Lie on the Polynomial Curve

Several factors contribute:

1. **Intermediate rounding:** The original D65 computation involved rounding steps not captured by the polynomial.

2. **Temperature scale adjustments:** The polynomial was adjusted for the new temperature scale, but the tables were frozen.

3. **Independent approximations:** The x(T) polynomial and y(x) quadratic were fit separately; neither was constrained to pass through canonical illuminants.

4. **Historical layering:** D65's SPD predates the polynomial; the polynomial is a later interpolation tool.

---

## 9. High-Precision Reference Values

### Temperatures

| Description | Value (K) |
|-------------|-----------|
| Nominal (historical, old K) | 6500.0000000000 |
| c₂ ratio conversion | 6503.5116136996 |
| Exact x match | 6503.6330064413 |
| Exact y match | 6509.5412401592 |
| Optimal (min. Euclidean) | 6506.6680000000 |
| lcms/moxcms value | 6504.0000000000 |

### Chromaticity at Each Temperature

| Temperature | x | y |
|-------------|---|---|
| 6500.0K | 0.3127788762 | 0.3291834985 |
| 6503.5116K | 0.3127219660 | 0.3291269584 |
| 6503.6330K | 0.3127200000 | 0.3291250048 |
| 6504.0K | 0.3127140569 | 0.3191190991 |
| 6506.6680K | 0.3126708751 | 0.3290761831 |
| 6509.5412K | 0.3126244185 | 0.3290300000 |

### Radiation Constants

| Description | Value |
|-------------|-------|
| Old c₂ (pre-1968) | 0.01438 m·K |
| New c₂ (CODATA) | 0.01438776877 m·K |
| Ratio (new/old) | 1.00054024826 |

### Official D65

| Coordinate | Value |
|------------|-------|
| x | 0.31272 |
| y | 0.32903 |

---

## 10. Practical Recommendations

### For Software Implementations

| Goal | Approach |
|------|----------|
| Exact D65 | Hardcode (0.31272, 0.32903) |
| D65 via polynomial | Use T = 6504K (8.9×10⁻⁵ error) |
| Theoretically correct T | Use T = 6503.5K (c₂ ratio, 9.7×10⁻⁵ error) |
| Minimum error | Use T = 6506.7K (6.7×10⁻⁵ error) |

**Recommendation:** For D50, D55, D65, D75, use hardcoded official chromaticity values. Use the polynomial only for arbitrary D-illuminants at non-standard temperatures.

### Perceptibility

All errors discussed are in the 5th decimal place (10⁻⁵), far below any perceptible threshold. The practical impact is limited to accumulated round-trip conversion errors.

---

## 11. Summary

| Item | Temperature Scale | Notes |
|------|------------------|-------|
| D65 name ("65") | Old K | Historical artifact |
| D65 tabulated SPD | Old K | Authoritative definition |
| D65 chromaticity | Derived from SPD | (0.31272, 0.32903) |
| x(T) polynomial | New K | Input modern CCT |
| Input for D65 | ~6504K (new K) | Because 6500 old K ≈ 6504 new K |

The polynomial expects modern CCT as input. D65 was defined as 6500K on the old scale. To recover D65 from the polynomial, convert: 6500 × (new_c₂/old_c₂) ≈ 6503.5K, or use the commonly rounded value 6504K.

The ~9×10⁻⁵ residual error reflects independent imprecision in the y(x) quadratic and historical rounding in the original D65 computation—not a temperature scale issue.

---

## References

- CIE 15:2004, Colorimetry, 3rd Edition
