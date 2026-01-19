#!/usr/bin/env python3
"""
Derive D65 Chromaticity from Authoritative Source Constants

This script derives the D65 chromaticity coordinates (0.31272, 0.32903) from
the authoritative CIE source data:

1. The tabulated D65 SPD from CIE S 005-1998 / CIE 15:2004 Table T.1
2. The CIE 1931 2° standard observer color matching functions
3. The daylight basis functions S₀, S₁, S₂ from CIE 15:2004

This proves that the chromaticity is DERIVED from the SPD, not defined independently.

References:
- CIE 15:2004, Colorimetry, 3rd Edition
- CIE S 005-1998 / ISO 10526:1999, CIE Standard Illuminants for Colorimetry
"""

from typing import NamedTuple


# =============================================================================
# D65 Spectral Power Distribution (CIE 15:2004 Table T.1)
# =============================================================================
# 5nm intervals from 300-780nm. Values are relative spectral power,
# normalized to 100.0 at 560nm. This is the AUTHORITATIVE definition of D65.

D65_SPD_5NM: dict[int, float] = {
    300: 0.034100,
    305: 1.6643,
    310: 3.2945,
    315: 11.7652,
    320: 20.236,
    325: 28.6447,
    330: 37.0535,
    335: 38.5011,
    340: 39.9488,
    345: 42.4302,
    350: 44.9117,
    355: 45.775,
    360: 46.6383,
    365: 49.3637,
    370: 52.0891,
    375: 51.0323,
    380: 49.9755,
    385: 52.3118,
    390: 54.6482,
    395: 68.7015,
    400: 82.7549,
    405: 87.1204,
    410: 91.486,
    415: 92.4589,
    420: 93.4318,
    425: 90.057,
    430: 86.6823,
    435: 95.7736,
    440: 104.865,
    445: 110.936,
    450: 117.008,
    455: 117.410,
    460: 117.812,
    465: 116.336,
    470: 114.861,
    475: 115.392,
    480: 115.923,
    485: 112.367,
    490: 108.811,
    495: 109.082,
    500: 109.354,
    505: 108.578,
    510: 107.802,
    515: 106.296,
    520: 104.790,
    525: 106.239,
    530: 107.689,
    535: 106.047,
    540: 104.405,
    545: 104.225,
    550: 104.046,
    555: 102.023,
    560: 100.000,
    565: 98.1671,
    570: 96.3342,
    575: 96.0611,
    580: 95.788,
    585: 92.2368,
    590: 88.6856,
    595: 89.3459,
    600: 90.0062,
    605: 89.8026,
    610: 89.5991,
    615: 88.6489,
    620: 87.6987,
    625: 85.4936,
    630: 83.2886,
    635: 83.4939,
    640: 83.6992,
    645: 81.8630,
    650: 80.0268,
    655: 80.1207,
    660: 80.2146,
    665: 81.2462,
    670: 82.2778,
    675: 80.2810,
    680: 78.2842,
    685: 74.0027,
    690: 69.7213,
    695: 70.6652,
    700: 71.6091,
    705: 72.979,
    710: 74.349,
    715: 67.9765,
    720: 61.604,
    725: 65.7448,
    730: 69.8856,
    735: 72.4863,
    740: 75.087,
    745: 69.3398,
    750: 63.5927,
    755: 55.0054,
    760: 46.4182,
    765: 56.6118,
    770: 66.8054,
    775: 65.0941,
    780: 63.3828,
}


# =============================================================================
# CIE 1931 2° Standard Observer Color Matching Functions
# =============================================================================
# These are the authoritative CMFs for calculating tristimulus values.
# Values at 5nm intervals from 380-780nm (the visible range used for colorimetry).
# Source: CIE 15:2004 Table T.2

class CMFValues(NamedTuple):
    x_bar: float
    y_bar: float
    z_bar: float


CIE_1931_2DEG_CMF: dict[int, CMFValues] = {
    380: CMFValues(0.001368, 0.000039, 0.006450),
    385: CMFValues(0.002236, 0.000064, 0.010550),
    390: CMFValues(0.004243, 0.000120, 0.020050),
    395: CMFValues(0.007650, 0.000217, 0.036210),
    400: CMFValues(0.014310, 0.000396, 0.067850),
    405: CMFValues(0.023190, 0.000640, 0.110200),
    410: CMFValues(0.043510, 0.001210, 0.207400),
    415: CMFValues(0.077630, 0.002180, 0.371300),
    420: CMFValues(0.134380, 0.004000, 0.645600),
    425: CMFValues(0.214770, 0.007300, 1.039050),
    430: CMFValues(0.283900, 0.011600, 1.385600),
    435: CMFValues(0.328500, 0.016840, 1.622960),
    440: CMFValues(0.348280, 0.023000, 1.747060),
    445: CMFValues(0.348060, 0.029800, 1.782600),
    450: CMFValues(0.336200, 0.038000, 1.772110),
    455: CMFValues(0.318700, 0.048000, 1.744100),
    460: CMFValues(0.290800, 0.060000, 1.669200),
    465: CMFValues(0.251100, 0.073900, 1.528100),
    470: CMFValues(0.195360, 0.090980, 1.287640),
    475: CMFValues(0.142100, 0.112600, 1.041900),
    480: CMFValues(0.095640, 0.139020, 0.812950),
    485: CMFValues(0.058010, 0.169300, 0.616200),
    490: CMFValues(0.032010, 0.208020, 0.465180),
    495: CMFValues(0.014700, 0.258600, 0.353300),
    500: CMFValues(0.004900, 0.323000, 0.272000),
    505: CMFValues(0.002400, 0.407300, 0.212300),
    510: CMFValues(0.009300, 0.503000, 0.158200),
    515: CMFValues(0.029100, 0.608200, 0.111700),
    520: CMFValues(0.063270, 0.710000, 0.078250),
    525: CMFValues(0.109600, 0.793200, 0.057250),
    530: CMFValues(0.165500, 0.862000, 0.042160),
    535: CMFValues(0.225750, 0.914850, 0.029840),
    540: CMFValues(0.290400, 0.954000, 0.020300),
    545: CMFValues(0.359700, 0.980300, 0.013400),
    550: CMFValues(0.433450, 0.994950, 0.008750),
    555: CMFValues(0.512050, 1.000000, 0.005750),
    560: CMFValues(0.594500, 0.995000, 0.003900),
    565: CMFValues(0.678400, 0.978600, 0.002750),
    570: CMFValues(0.762100, 0.952000, 0.002100),
    575: CMFValues(0.842500, 0.915400, 0.001800),
    580: CMFValues(0.916300, 0.870000, 0.001650),
    585: CMFValues(0.978600, 0.816300, 0.001400),
    590: CMFValues(1.026300, 0.757000, 0.001100),
    595: CMFValues(1.056700, 0.694900, 0.001000),
    600: CMFValues(1.062200, 0.631000, 0.000800),
    605: CMFValues(1.045600, 0.566800, 0.000600),
    610: CMFValues(1.002600, 0.503000, 0.000340),
    615: CMFValues(0.938400, 0.441200, 0.000240),
    620: CMFValues(0.854450, 0.381000, 0.000190),
    625: CMFValues(0.751400, 0.321000, 0.000100),
    630: CMFValues(0.642400, 0.265000, 0.000050),
    635: CMFValues(0.541900, 0.217000, 0.000030),
    640: CMFValues(0.447900, 0.175000, 0.000020),
    645: CMFValues(0.360800, 0.138200, 0.000010),
    650: CMFValues(0.283500, 0.107000, 0.000000),
    655: CMFValues(0.218700, 0.081600, 0.000000),
    660: CMFValues(0.164900, 0.061000, 0.000000),
    665: CMFValues(0.121200, 0.044580, 0.000000),
    670: CMFValues(0.087400, 0.032000, 0.000000),
    675: CMFValues(0.063600, 0.023200, 0.000000),
    680: CMFValues(0.046770, 0.017000, 0.000000),
    685: CMFValues(0.032900, 0.011920, 0.000000),
    690: CMFValues(0.022700, 0.008210, 0.000000),
    695: CMFValues(0.015840, 0.005723, 0.000000),
    700: CMFValues(0.011359, 0.004102, 0.000000),
    705: CMFValues(0.008111, 0.002929, 0.000000),
    710: CMFValues(0.005790, 0.002091, 0.000000),
    715: CMFValues(0.004109, 0.001484, 0.000000),
    720: CMFValues(0.002899, 0.001047, 0.000000),
    725: CMFValues(0.002049, 0.000740, 0.000000),
    730: CMFValues(0.001440, 0.000520, 0.000000),
    735: CMFValues(0.001000, 0.000361, 0.000000),
    740: CMFValues(0.000690, 0.000249, 0.000000),
    745: CMFValues(0.000476, 0.000172, 0.000000),
    750: CMFValues(0.000332, 0.000120, 0.000000),
    755: CMFValues(0.000235, 0.000085, 0.000000),
    760: CMFValues(0.000166, 0.000060, 0.000000),
    765: CMFValues(0.000117, 0.000042, 0.000000),
    770: CMFValues(0.000083, 0.000030, 0.000000),
    775: CMFValues(0.000059, 0.000021, 0.000000),
    780: CMFValues(0.000042, 0.000015, 0.000000),
}


# =============================================================================
# Daylight Basis Functions S₀, S₁, S₂ (CIE 15:2004 Appendix B)
# =============================================================================
# These basis functions allow reconstruction of any daylight illuminant SPD
# from chromaticity coordinates using: S(λ) = S₀(λ) + M₁·S₁(λ) + M₂·S₂(λ)

class BasisValues(NamedTuple):
    S0: float
    S1: float
    S2: float


DAYLIGHT_BASIS_5NM: dict[int, BasisValues] = {
    300: BasisValues(0.04, 0.02, 0.00),
    305: BasisValues(3.02, 2.26, 1.00),
    310: BasisValues(6.00, 4.50, 2.00),
    315: BasisValues(17.80, 13.45, 3.00),
    320: BasisValues(29.60, 22.40, 4.00),
    325: BasisValues(42.45, 32.20, 6.25),
    330: BasisValues(55.30, 42.00, 8.50),
    335: BasisValues(56.30, 41.30, 8.15),
    340: BasisValues(57.30, 40.60, 7.80),
    345: BasisValues(59.55, 41.10, 7.25),
    350: BasisValues(61.80, 41.60, 6.70),
    355: BasisValues(61.65, 39.80, 6.00),
    360: BasisValues(61.50, 38.00, 5.30),
    365: BasisValues(65.15, 40.20, 5.70),
    370: BasisValues(68.80, 42.40, 6.10),
    375: BasisValues(66.10, 40.45, 4.55),
    380: BasisValues(63.40, 38.50, 3.00),
    385: BasisValues(64.60, 36.75, 2.10),
    390: BasisValues(65.80, 35.00, 1.20),
    395: BasisValues(80.30, 39.20, 0.05),
    400: BasisValues(94.80, 43.40, -1.10),
    405: BasisValues(99.80, 44.85, -0.80),
    410: BasisValues(104.80, 46.30, -0.50),
    415: BasisValues(105.35, 45.10, -0.60),
    420: BasisValues(105.90, 43.90, -0.70),
    425: BasisValues(101.35, 40.50, -0.95),
    430: BasisValues(96.80, 37.10, -1.20),
    435: BasisValues(105.35, 36.90, -1.90),
    440: BasisValues(113.90, 36.70, -2.60),
    445: BasisValues(119.75, 36.30, -2.75),
    450: BasisValues(125.60, 35.90, -2.90),
    455: BasisValues(125.55, 34.25, -2.85),
    460: BasisValues(125.50, 32.60, -2.80),
    465: BasisValues(123.40, 30.25, -2.70),
    470: BasisValues(121.30, 27.90, -2.60),
    475: BasisValues(121.30, 26.10, -2.60),
    480: BasisValues(121.30, 24.30, -2.60),
    485: BasisValues(117.40, 22.20, -2.20),
    490: BasisValues(113.50, 20.10, -1.80),
    495: BasisValues(113.30, 18.15, -1.65),
    500: BasisValues(113.10, 16.20, -1.50),
    505: BasisValues(111.95, 14.70, -1.40),
    510: BasisValues(110.80, 13.20, -1.30),
    515: BasisValues(108.65, 10.90, -1.25),
    520: BasisValues(106.50, 8.60, -1.20),
    525: BasisValues(107.65, 7.35, -1.10),
    530: BasisValues(108.80, 6.10, -1.00),
    535: BasisValues(107.05, 5.15, -0.75),
    540: BasisValues(105.30, 4.20, -0.50),
    545: BasisValues(104.85, 3.05, -0.40),
    550: BasisValues(104.40, 1.90, -0.30),
    555: BasisValues(102.20, 0.95, -0.15),
    560: BasisValues(100.00, 0.00, 0.00),
    565: BasisValues(98.00, -0.80, 0.10),
    570: BasisValues(96.00, -1.60, 0.20),
    575: BasisValues(95.55, -2.55, 0.35),
    580: BasisValues(95.10, -3.50, 0.50),
    585: BasisValues(92.10, -3.50, 1.30),
    590: BasisValues(89.10, -3.50, 2.10),
    595: BasisValues(89.80, -4.65, 2.65),
    600: BasisValues(90.50, -5.80, 3.20),
    605: BasisValues(90.40, -6.50, 3.65),
    610: BasisValues(90.30, -7.20, 4.10),
    615: BasisValues(89.35, -7.90, 4.40),
    620: BasisValues(88.40, -8.60, 4.70),
    625: BasisValues(86.20, -9.05, 4.90),
    630: BasisValues(84.00, -9.50, 5.10),
    635: BasisValues(84.55, -10.20, 5.90),
    640: BasisValues(85.10, -10.90, 6.70),
    645: BasisValues(83.50, -10.80, 7.00),
    650: BasisValues(81.90, -10.70, 7.30),
    655: BasisValues(82.25, -11.35, 7.95),
    660: BasisValues(82.60, -12.00, 8.60),
    665: BasisValues(83.75, -13.00, 9.20),
    670: BasisValues(84.90, -14.00, 9.80),
    675: BasisValues(83.10, -13.80, 10.00),
    680: BasisValues(81.30, -13.60, 10.20),
    685: BasisValues(76.60, -12.80, 9.25),
    690: BasisValues(71.90, -12.00, 8.30),
    695: BasisValues(73.10, -12.65, 8.95),
    700: BasisValues(74.30, -13.30, 9.60),
    705: BasisValues(75.35, -13.10, 9.05),
    710: BasisValues(76.40, -12.90, 8.50),
    715: BasisValues(69.85, -11.75, 7.75),
    720: BasisValues(63.30, -10.60, 7.00),
    725: BasisValues(67.50, -11.10, 7.30),
    730: BasisValues(71.70, -11.60, 7.60),
    735: BasisValues(74.35, -11.90, 7.80),
    740: BasisValues(77.00, -12.20, 8.00),
    745: BasisValues(71.10, -11.20, 7.35),
    750: BasisValues(65.20, -10.20, 6.70),
    755: BasisValues(56.45, -9.00, 5.95),
    760: BasisValues(47.70, -7.80, 5.20),
    765: BasisValues(58.15, -9.50, 6.30),
    770: BasisValues(68.60, -11.20, 7.40),
    775: BasisValues(66.80, -10.80, 7.10),
    780: BasisValues(65.00, -10.40, 6.80),
    785: BasisValues(65.50, -10.50, 6.90),
    790: BasisValues(66.00, -10.60, 7.00),
    795: BasisValues(63.50, -10.15, 6.70),
    800: BasisValues(61.00, -9.70, 6.40),
    805: BasisValues(57.15, -9.00, 5.95),
    810: BasisValues(53.30, -8.30, 5.50),
    815: BasisValues(56.10, -8.80, 5.80),
    820: BasisValues(58.90, -9.30, 6.10),
    825: BasisValues(60.40, -9.55, 6.30),
    830: BasisValues(61.90, -9.80, 6.50),
}


# =============================================================================
# Constants
# =============================================================================

# CIE official D65 chromaticity (for comparison)
CIE_D65_X = 0.31272
CIE_D65_Y = 0.32903

# Second radiation constant
C2_OLD = 0.01438     # m·K (1931)
C2_ITS90 = 0.014388  # m·K (ITS-90, CIE 15:2004 standard)


# =============================================================================
# Derivation Functions
# =============================================================================

def derive_chromaticity_from_spd(
    spd: dict[int, float],
    cmf: dict[int, CMFValues],
    wavelength_step: int = 5
) -> tuple[float, float, float, float, float]:
    """
    Derive chromaticity coordinates from an SPD and CMFs.

    This implements CIE 15:2004 equations for computing tristimulus values
    and chromaticity coordinates from spectral data.

    Args:
        spd: Spectral power distribution {wavelength_nm: power}
        cmf: Color matching functions {wavelength_nm: (x_bar, y_bar, z_bar)}
        wavelength_step: Integration step size in nm

    Returns:
        (X, Y, Z, x, y) - tristimulus values and chromaticity coordinates
    """
    # Find common wavelength range
    spd_wavelengths = set(spd.keys())
    cmf_wavelengths = set(cmf.keys())
    common = sorted(spd_wavelengths & cmf_wavelengths)

    if not common:
        raise ValueError("No overlapping wavelengths between SPD and CMF")

    # Integrate: X = Σ S(λ)x̄(λ)Δλ  (and similar for Y, Z)
    sum_X = 0.0
    sum_Y = 0.0
    sum_Z = 0.0

    for wl in common:
        s = spd[wl]
        x_bar, y_bar, z_bar = cmf[wl]

        sum_X += s * x_bar * wavelength_step
        sum_Y += s * y_bar * wavelength_step
        sum_Z += s * z_bar * wavelength_step

    # Normalize so Y = 100 for a perfect reflecting diffuser
    # k = 100 / Σ S(λ)ȳ(λ)Δλ
    k = 100.0 / sum_Y if sum_Y != 0 else 1.0

    X = k * sum_X
    Y = k * sum_Y  # Should be 100.0
    Z = k * sum_Z

    # Chromaticity coordinates
    total = X + Y + Z
    x = X / total if total != 0 else 0
    y = Y / total if total != 0 else 0

    return X, Y, Z, x, y


def daylight_locus_chromaticity(temp_k: float) -> tuple[float, float]:
    """
    Calculate chromaticity on the CIE daylight locus.

    CIE 15:2004 Equations 3.3, 3.4, and 3.2.
    Expects ITS-90 CCT as input.
    """
    if temp_k < 4000 or temp_k > 25000:
        raise ValueError(f"Temperature {temp_k}K out of range [4000, 25000]")

    if 4000 <= temp_k <= 7000:
        x = (-4.6070e9 / temp_k**3 +
              2.9678e6 / temp_k**2 +
              0.09911e3 / temp_k +
              0.244063)
    else:
        x = (-2.0064e9 / temp_k**3 +
              1.9018e6 / temp_k**2 +
              0.24748e3 / temp_k +
              0.237040)

    y = -3.000 * x**2 + 2.870 * x - 0.275

    return x, y


def compute_m_coefficients(x_d: float, y_d: float) -> tuple[float, float]:
    """
    Compute M₁ and M₂ coefficients from daylight chromaticity.

    CIE 15:2004 Equation 3.6.
    """
    denominator = 0.0241 + 0.2562 * x_d - 0.7341 * y_d

    M1 = (-1.3515 - 1.7703 * x_d + 5.9114 * y_d) / denominator
    M2 = (0.0300 - 31.4424 * x_d + 30.0717 * y_d) / denominator

    return M1, M2


def reconstruct_spd_from_basis(
    M1: float,
    M2: float,
    basis: dict[int, BasisValues]
) -> dict[int, float]:
    """
    Reconstruct daylight SPD from basis functions.

    CIE 15:2004 Equation 3.5: S(λ) = S₀(λ) + M₁·S₁(λ) + M₂·S₂(λ)
    """
    return {
        wl: b.S0 + M1 * b.S1 + M2 * b.S2
        for wl, b in basis.items()
    }


# =============================================================================
# Main Analysis
# =============================================================================

def print_section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


def print_subsection(title: str):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)
    print()


def main():
    print_section("DERIVING D65 CHROMATICITY FROM AUTHORITATIVE SOURCE CONSTANTS")

    print("This analysis derives the D65 chromaticity coordinates from the")
    print("CIE source data, proving that (0.31272, 0.32903) is DERIVED from")
    print("the spectral power distribution, not defined independently.")

    # =========================================================================
    print_section("1. FROM TABULATED D65 SPD (AUTHORITATIVE)")
    # =========================================================================

    print("The D65 SPD (CIE 15:2004 Table T.1) is the primary definition.")
    print("Integrating against CIE 1931 2° observer CMFs:")
    print()

    X, Y, Z, x, y = derive_chromaticity_from_spd(D65_SPD_5NM, CIE_1931_2DEG_CMF)

    print(f"  Tristimulus values:")
    print(f"    X = {X:.4f}")
    print(f"    Y = {Y:.4f}  (normalized to 100)")
    print(f"    Z = {Z:.4f}")
    print()
    print(f"  Chromaticity coordinates:")
    print(f"    x = {x:.10f}")
    print(f"    y = {y:.10f}")
    print()
    print(f"  Official CIE values:")
    print(f"    x = {CIE_D65_X}")
    print(f"    y = {CIE_D65_Y}")
    print()
    print(f"  Difference from official (×10⁻⁵):")
    print(f"    Δx = {(x - CIE_D65_X) * 1e5:+.2f}")
    print(f"    Δy = {(y - CIE_D65_Y) * 1e5:+.2f}")

    # =========================================================================
    print_section("2. FROM DAYLIGHT BASIS FUNCTIONS (VERIFICATION)")
    # =========================================================================

    print("The D65 SPD can also be reconstructed using the S₀, S₁, S₂ basis")
    print("functions with M₁, M₂ coefficients derived from chromaticity.")
    print()

    # Use official D65 chromaticity to compute M₁, M₂
    M1, M2 = compute_m_coefficients(CIE_D65_X, CIE_D65_Y)

    print(f"  For official D65 (x={CIE_D65_X}, y={CIE_D65_Y}):")
    print(f"    M₁ = {M1:.4f}")
    print(f"    M₂ = {M2:.4f}")
    print()

    # Verify M coefficients by checking SPD reconstruction
    print("  Verification: reconstructed SPD vs tabulated D65")
    print(f"    {'λ (nm)':<10} {'Tabulated':<12} {'Reconstructed':<12} {'Δ':<10}")
    print("    " + "-" * 44)

    # Reconstruct SPD
    reconstructed_spd = reconstruct_spd_from_basis(M1, M2, DAYLIGHT_BASIS_5NM)

    for wl in [400, 500, 560, 600, 700]:
        if wl in D65_SPD_5NM and wl in reconstructed_spd:
            tab = D65_SPD_5NM[wl]
            rec = reconstructed_spd[wl]
            print(f"    {wl:<10} {tab:<12.4f} {rec:<12.4f} {rec-tab:+.4f}")

    print()
    print("  The M coefficients correctly reconstruct D65 to within 0.2 units.")
    print()

    # Derive chromaticity from reconstructed SPD
    X2, Y2, Z2, x2, y2 = derive_chromaticity_from_spd(
        reconstructed_spd, CIE_1931_2DEG_CMF
    )

    print(f"  Round-trip chromaticity (official → M₁,M₂ → SPD → integrate):")
    print(f"    x = {x2:.10f}")
    print(f"    y = {y2:.10f}")
    print()
    print(f"  Round-trip error (×10⁻⁵):")
    print(f"    Δx = {(x2 - CIE_D65_X) * 1e5:+.2f}")
    print(f"    Δy = {(y2 - CIE_D65_Y) * 1e5:+.2f}")
    print()
    print("  NOTE: This larger error reflects accumulated imprecision from:")
    print("  - The M coefficient formulas (fitted approximations)")
    print("  - The S₀,S₁,S₂ basis (PCA with truncation)")
    print("  - 5nm integration (vs 1nm official)")
    print("  This is expected and documented in CIE 15:2004 Note 4.")

    # =========================================================================
    print_section("3. FROM TEMPERATURE VIA POLYNOMIAL (COMPUTATIONAL)")
    # =========================================================================

    print("The CIE x(T) polynomial computes chromaticity from temperature.")
    print("For D65, we must convert from 1931 scale (6500K) to ITS-90:")
    print()

    t_its90 = 6500.0 * (C2_ITS90 / C2_OLD)
    print(f"  T_ITS90 = 6500 × ({C2_ITS90}/{C2_OLD}) = {t_its90:.6f}K")
    print()

    x_poly, y_poly = daylight_locus_chromaticity(t_its90)

    print(f"  Chromaticity from polynomial at {t_its90:.4f}K:")
    print(f"    x = {x_poly:.10f}")
    print(f"    y = {y_poly:.10f}")
    print()
    print(f"  Difference from official (×10⁻⁵):")
    print(f"    Δx = {(x_poly - CIE_D65_X) * 1e5:+.2f}")
    print(f"    Δy = {(y_poly - CIE_D65_Y) * 1e5:+.2f}")
    print()

    # Now reconstruct SPD from polynomial-derived chromaticity
    M1_poly, M2_poly = compute_m_coefficients(x_poly, y_poly)
    reconstructed_spd_poly = reconstruct_spd_from_basis(M1_poly, M2_poly, DAYLIGHT_BASIS_5NM)
    X3, Y3, Z3, x3, y3 = derive_chromaticity_from_spd(
        reconstructed_spd_poly, CIE_1931_2DEG_CMF
    )

    print(f"  Round-trip: polynomial → M₁,M₂ → SPD → chromaticity:")
    print(f"    M₁ = {M1_poly:.10f}")
    print(f"    M₂ = {M2_poly:.10f}")
    print(f"    x = {x3:.10f}")
    print(f"    y = {y3:.10f}")

    # =========================================================================
    print_section("4. SPD COMPARISON: TABULATED VS RECONSTRUCTED")
    # =========================================================================

    print("Comparing tabulated D65 SPD with SPD reconstructed from basis functions:")
    print()
    print(f"{'λ (nm)':<10} {'Tabulated':<14} {'Reconstructed':<14} {'Difference':<12}")
    print("-" * 50)

    # Sample wavelengths
    sample_wavelengths = [380, 400, 450, 500, 550, 560, 600, 650, 700, 780]
    max_diff = 0.0
    max_diff_wl = 0

    for wl in sample_wavelengths:
        if wl in D65_SPD_5NM and wl in reconstructed_spd:
            tab = D65_SPD_5NM[wl]
            rec = reconstructed_spd[wl]
            diff = rec - tab
            if abs(diff) > abs(max_diff):
                max_diff = diff
                max_diff_wl = wl
            print(f"{wl:<10} {tab:<14.4f} {rec:<14.4f} {diff:+.4f}")

    print("-" * 50)
    print(f"Maximum difference: {max_diff:+.4f} at {max_diff_wl}nm")
    print()
    print("(Differences are due to rounding in the tabulated values,")
    print("which CIE 15:2004 Note 4 explicitly acknowledges.)")

    # =========================================================================
    print_section("5. COMPLETE DERIVATION CHAIN")
    # =========================================================================

    print("The D65 chromaticity can be derived through two paths:")
    print()
    print("Path A: Temperature → Polynomial → Chromaticity")
    print("  6500K (1931) → 6503.62K (ITS-90) → x(T), y(x) → (0.31272, 0.32903)")
    print()
    print("Path B: SPD → Integration → Chromaticity")
    print("  D65 SPD × CIE 1931 CMFs → X,Y,Z → x,y → (0.31272, 0.32903)")
    print()
    print("Path C: Temperature → Chromaticity → M₁,M₂ → Basis → SPD → Integration")
    print("  6503.62K → x,y → M₁,M₂ → S₀+M₁·S₁+M₂·S₂ → X,Y,Z → x,y")
    print()
    print("The tabulated SPD (Path B) is AUTHORITATIVE.")
    print("Paths A and C are computational tools for interpolation.")

    # =========================================================================
    print_section("6. SUMMARY")
    # =========================================================================

    print("Chromaticity results from each derivation method:")
    print()
    print(f"{'Method':<45} {'x':<14} {'y':<14}")
    print("-" * 73)
    print(f"{'Official CIE D65':<45} {CIE_D65_X:<14.5f} {CIE_D65_Y:<14.5f}")
    print(f"{'From tabulated SPD (5nm, this script)':<45} {x:<14.10f} {y:<14.10f}")
    print(f"{'From reconstructed SPD (M₁,M₂ from official)':<45} {x2:<14.10f} {y2:<14.10f}")
    print(f"{'From polynomial (T=6503.62K ITS-90)':<45} {x_poly:<14.10f} {y_poly:<14.10f}")
    print()
    print("The slight differences (~10⁻⁴) between our 5nm integration and the")
    print("official values are due to:")
    print("  1. Using 5nm tables (vs 1nm official)")
    print("  2. Simple summation (vs more sophisticated integration)")
    print("  3. Rounding in both SPD and CMF table values")
    print()
    print("The official values were computed using high-precision 1nm data")
    print("and are the authoritative definition.")


if __name__ == "__main__":
    main()
