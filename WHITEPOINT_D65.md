# The D65 Whitepoint: Precision, History, and Derivation

## Executive Summary

D65 is the CIE standard daylight illuminant representing average noon daylight. While commonly associated with "6500K," the relationship between D65 and color temperature involves two different temperature scales: the 1931 scale on which D65 was originally defined, and the modern ITS-90 scale used in current CIE formulas. This document explains the precision issues, historical changes, and the derivation relationships between them.

---

## 1. The Authoritative Definition

D65 is defined by the CIE as a **tabulated spectral power distribution (SPD)**—a table of relative spectral radiance values from 300–830nm, published with 6 significant figures in CIE S 005-1998 and CIE 15:2004.

From this SPD, the chromaticity coordinates are derived by integration against the CIE 1931 2° standard observer color matching functions:
```
x = 0.31272
y = 0.32903
```

These coordinates are published to 5 decimal places. **The SPD is the primary definition; the chromaticity is derived from it.**

---

## 2. The Two Temperature Scales

The second radiation constant (c₂) in Planck's law has changed over time. CIE 15:2004 Appendix E specifies that colorimetric calculations should use the value from the International Temperature Scale of 1990 (ITS-90):

| Constant | Value | Source |
|----------|-------|--------|
| Old c₂ | 0.01438 m·K | 1931 CIE definition |
| New c₂ | 0.014388 m·K | ITS-90 (CIE 15:2004 standard) |

Note: The CODATA physical measurement (0.01438776877 m·K) is **not** used by CIE colorimetry. The ITS-90 value is a defined constant for temperature scale interoperability.

This change affects how **Correlated Color Temperature (CCT)** is calculated. When c₂ changed, the Planckian locus shifted slightly, changing all CCT values.

**The D65 chromaticity did not change**—it is defined by its SPD. But the CCT assigned to that chromaticity changed:

| Description | Temperature |
|-------------|-------------|
| D65's old CCT (1931 scale) | 6500K |
| D65's new CCT (ITS-90 scale) | ~6504K |

The name "D65" is a historical artifact referring to the old CCT.

---

## 3. Daylight vs. Blackbody

A common misconception is that D65 represents a 6500K blackbody. It does not.

**Blackbody (Planckian) radiation** follows Planck's law exactly. A 6500K blackbody has chromaticity approximately (0.313, 0.324)—notably different from D65's (0.31272, 0.32903).

**Daylight** differs from blackbody radiation due to:
- Rayleigh scattering (blue sky)
- Atmospheric absorption bands
- Aerosol and cloud scattering

The D65 chromaticity lies above the Planckian locus in the y-direction. This represents the physical difference between actual daylight and idealized blackbody radiation.

The **daylight locus** is a separate curve through chromaticity space, derived empirically from 622 measurements of real daylight (Judd, MacAdam, Wyszecki et al., 1964), that runs roughly parallel to but offset from the Planckian locus.

---

## 4. The Daylight Locus Formulas

### Origin

Judd, MacAdam, and Wyszecki (1964) analyzed 622 daylight samples and found their chromaticities followed a quadratic relationship:

**The y(x) quadratic (CIE 15:2004 Equation 3.2):**
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

Where **T_cp is the correlated color temperature on the ITS-90 scale**.

---

## 5. The Critical Distinction: Tables vs. Polynomial

CIE 15:2004 Note 4 (Section 3.1) states:

> "The relative spectral power distributions of the D illuminants given in Table T.1 and in the CIE standard on illuminants for colorimetry (CIE, 1998c) were derived by the procedure given above with some intermediate rounding and **with some adjustments for changes in the International Temperature Scale**. Thus for historic reasons, the tabulated values are slightly different from the calculated values. For the time being **the tabulated values are the official data**."

This reveals the key distinction:

| Component | Temperature Scale | Status |
|-----------|------------------|--------|
| Tabulated D65 values (SPD, chromaticity) | 1931 scale | **Authoritative** |
| x(T) polynomial in CIE 15:2004 | ITS-90 scale | Computational tool |

The tables are frozen historical artifacts computed with the 1931 temperature scale. The polynomial expects modern ITS-90 CCT values. This is why there's a mismatch when using nominal temperatures.

---

## 6. Converting Between Temperature Scales

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

## 7. Experimental Verification

Testing the polynomial against official D65 coordinates (0.31272, 0.32903):

| Temperature | Δx (×10⁻⁵) | Δy (×10⁻⁵) | Euclidean (×10⁻⁵) |
|-------------|------------|------------|-------------------|
| 6500.0K (nominal 1931) | +5.89 | +15.35 | 16.44 |
| 6503.6161K (ITS-90) | +0.03 | +9.53 | 9.53 |
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

## 8. The y(x) Quadratic Error

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

## 9. Why the Canonical Illuminants Don't Lie on the Polynomial Curve

Several factors contribute:

1. **Independent derivation:** The canonical illuminants were computed using the S₀, S₁, S₂ basis functions (CIE 15:2004 Equation 3.5), while the y(x) quadratic was fit separately to the original 622 measurements.

2. **Intermediate rounding:** The original computations involved rounding steps not captured by the polynomial.

3. **Temperature scale adjustments:** The polynomial was adjusted for ITS-90, but the tables were frozen.

4. **Historical layering:** The tabulated SPDs predate the polynomial formalization; the polynomial is a later interpolation tool.

---

## 10. High-Precision Reference Values

### Radiation Constants (CIE 15:2004 Appendix E)

| Description | Value |
|-------------|-------|
| Old c₂ (1931) | 0.01438 m·K |
| New c₂ (ITS-90) | 0.014388 m·K |
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

### Chromaticity at Key Temperatures (D65)

| Temperature | x | y |
|-------------|------------|------------|
| 6500.0K (nominal) | 0.3127788762 | 0.3291834985 |
| 6503.6161K (ITS-90) | 0.3127202733 | 0.3291252763 |
| 6503.6330K (exact x) | 0.3127200000 | 0.3291250048 |
| 6504.0K (rounded) | 0.3127140569 | 0.3191190991 |

---

## 11. Practical Recommendations

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

---

## 12. Summary

| Item | Temperature Scale | Notes |
|------|------------------|-------|
| D65 name ("65") | 1931 scale | Historical artifact |
| D65 tabulated SPD | 1931 scale | Authoritative definition |
| D65 chromaticity | Derived from SPD | (0.31272, 0.32903) |
| x(T) polynomial | ITS-90 | Input modern CCT |
| Input for D65 | 6503.62K (ITS-90) | Because 6500K (1931) = 6503.62K (ITS-90) |

The polynomial expects ITS-90 CCT as input. D65 was defined as 6500K on the 1931 scale. To recover D65 from the polynomial, convert: 6500 × (0.014388/0.01438) = 6503.616134K.

The ~9.5×10⁻⁵ residual y error is inherent in the y(x) quadratic formula—it affects all canonical illuminants equally and reflects the fact that the quadratic was never constrained to pass through them exactly.

---

## Appendix: D65 Quick Reference for Continuous Computation

The CIE tabulated values are authoritative for compliance, but cannot be continuously interpolated. For any application requiring continuous mathematics (derivatives, optimization, smooth interpolation), use the CIE daylight locus polynomial with the following temperature:

**D65 Modern CCT (per CIE 15:2004 Appendix E):**
```
T = 6500K × (0.014388 / 0.01438) = 6503.616134K
```

| Constant | Value |
|----------|-------|
| Old c₂ (1931) | 0.01438 m·K |
| New c₂ (ITS-90) | 0.014388 m·K |
| Conversion factor | 1.00055632823 |
| D65 modern CCT | 6503.616134K |

This yields a chromaticity error of ~9.5×10⁻⁵ from the official (0.31272, 0.32903)—an irreducible artifact of the y(x) quadratic approximation, not the temperature conversion.

---

## References

- CIE 15:2004, Colorimetry, 3rd Edition
- CIE S 005-1998 / ISO 10526:1999, CIE Standard Illuminants for Colorimetry
- Judd, D.B., MacAdam, D.L., Wyszecki, G., et al. (1964). "Spectral Distribution of Typical Daylight as a Function of Correlated Color Temperature." *J. Opt. Soc. Am.* 54, 1031-1040.
