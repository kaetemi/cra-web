#!/usr/bin/env python3
"""
Calculate D65 chromaticity coordinates using different radiation constants
and compare with official CIE values.

This script explores the historical quirk where D65 was defined when the
second radiation constant c2 was 0.014380, but was later recalculated as
6504K when c2 was updated to 0.014388 in 1968.
"""

import math
import numpy as np

# CIE official D65 chromaticity coordinates (5 decimal places)
CIE_D65_X = 0.31272
CIE_D65_Y = 0.32903

# Second radiation constant values
C2_OLD = 0.014380  # Pre-1968
C2_NEW = 0.014388  # Post-1968 (current CODATA value is ~0.01438776877...)

# First radiation constant (for completeness, though it cancels in chromaticity)
C1 = 3.741771e-16  # W⋅m²


def planck_spectral_radiance(wavelength_m, temp_k, c2):
    """
    Calculate spectral radiance using Planck's law.

    B(λ,T) = c1 / (λ^5 * (exp(c2/(λ*T)) - 1))

    For chromaticity calculations, c1 cancels out, but we include it for completeness.
    """
    exp_term = math.exp(c2 / (wavelength_m * temp_k)) - 1
    return C1 / (wavelength_m**5 * exp_term)


# CIE 1931 2° Standard Observer Color Matching Functions (sampled)
# These are approximate values at key wavelengths for demonstration
# For precise calculations, use full 1nm tables from CIE

def load_cie_cmf():
    """
    Load CIE 1931 2° color matching functions.
    Returns wavelengths (nm) and x_bar, y_bar, z_bar arrays.

    Using standard tabulated values from 380-780nm at 5nm intervals.
    """
    # Wavelengths from 380 to 780 nm at 5nm intervals
    wavelengths = np.arange(380, 785, 5)

    # CIE 1931 2° CMF data (abbreviated - key values)
    # Full data would be from CIE tables
    cmf_data = {
        380: (0.001368, 0.000039, 0.006450),
        385: (0.002236, 0.000064, 0.010550),
        390: (0.004243, 0.000120, 0.020050),
        395: (0.007650, 0.000217, 0.036210),
        400: (0.014310, 0.000396, 0.067850),
        405: (0.023190, 0.000640, 0.110200),
        410: (0.043510, 0.001210, 0.207400),
        415: (0.077630, 0.002180, 0.371300),
        420: (0.134380, 0.004000, 0.645600),
        425: (0.214770, 0.007300, 1.039050),
        430: (0.283900, 0.011600, 1.385600),
        435: (0.328500, 0.016840, 1.622960),
        440: (0.348280, 0.023000, 1.747060),
        445: (0.348060, 0.029800, 1.782600),
        450: (0.336200, 0.038000, 1.772110),
        455: (0.318700, 0.048000, 1.744100),
        460: (0.290800, 0.060000, 1.669200),
        465: (0.251100, 0.073900, 1.528100),
        470: (0.195360, 0.090980, 1.287640),
        475: (0.142100, 0.112600, 1.041900),
        480: (0.095640, 0.139020, 0.812950),
        485: (0.058010, 0.169300, 0.616200),
        490: (0.032010, 0.208020, 0.465180),
        495: (0.014700, 0.258600, 0.353300),
        500: (0.004900, 0.323000, 0.272000),
        505: (0.002400, 0.407300, 0.212300),
        510: (0.009300, 0.503000, 0.158200),
        515: (0.029100, 0.608200, 0.111700),
        520: (0.063270, 0.710000, 0.078250),
        525: (0.109600, 0.793200, 0.057250),
        530: (0.165500, 0.862000, 0.042160),
        535: (0.225750, 0.914850, 0.029840),
        540: (0.290400, 0.954000, 0.020300),
        545: (0.359700, 0.980300, 0.013400),
        550: (0.433450, 0.994950, 0.008750),
        555: (0.512050, 1.000000, 0.005750),
        560: (0.594500, 0.995000, 0.003900),
        565: (0.678400, 0.978600, 0.002750),
        570: (0.762100, 0.952000, 0.002100),
        575: (0.842500, 0.915400, 0.001800),
        580: (0.916300, 0.870000, 0.001650),
        585: (0.978600, 0.816300, 0.001400),
        590: (1.026300, 0.757000, 0.001100),
        595: (1.056700, 0.694900, 0.001000),
        600: (1.062200, 0.631000, 0.000800),
        605: (1.045600, 0.566800, 0.000600),
        610: (1.002600, 0.503000, 0.000340),
        615: (0.938400, 0.441200, 0.000240),
        620: (0.854450, 0.381000, 0.000190),
        625: (0.751400, 0.321000, 0.000100),
        630: (0.642400, 0.265000, 0.000050),
        635: (0.541900, 0.217000, 0.000030),
        640: (0.447900, 0.175000, 0.000020),
        645: (0.360800, 0.138200, 0.000010),
        650: (0.283500, 0.107000, 0.000000),
        655: (0.218700, 0.081600, 0.000000),
        660: (0.164900, 0.061000, 0.000000),
        665: (0.121200, 0.044580, 0.000000),
        670: (0.087400, 0.032000, 0.000000),
        675: (0.063600, 0.023200, 0.000000),
        680: (0.046770, 0.017000, 0.000000),
        685: (0.032900, 0.011920, 0.000000),
        690: (0.022700, 0.008210, 0.000000),
        695: (0.015840, 0.005723, 0.000000),
        700: (0.011359, 0.004102, 0.000000),
        705: (0.008111, 0.002929, 0.000000),
        710: (0.005790, 0.002091, 0.000000),
        715: (0.004109, 0.001484, 0.000000),
        720: (0.002899, 0.001047, 0.000000),
        725: (0.002049, 0.000740, 0.000000),
        730: (0.001440, 0.000520, 0.000000),
        735: (0.001000, 0.000361, 0.000000),
        740: (0.000690, 0.000249, 0.000000),
        745: (0.000476, 0.000172, 0.000000),
        750: (0.000332, 0.000120, 0.000000),
        755: (0.000235, 0.000085, 0.000000),
        760: (0.000166, 0.000060, 0.000000),
        765: (0.000117, 0.000042, 0.000000),
        770: (0.000083, 0.000030, 0.000000),
        775: (0.000059, 0.000021, 0.000000),
        780: (0.000042, 0.000015, 0.000000),
    }

    x_bar = np.array([cmf_data[w][0] for w in wavelengths])
    y_bar = np.array([cmf_data[w][1] for w in wavelengths])
    z_bar = np.array([cmf_data[w][2] for w in wavelengths])

    return wavelengths, x_bar, y_bar, z_bar


def blackbody_chromaticity(temp_k, c2, wavelengths, x_bar, y_bar, z_bar):
    """
    Calculate chromaticity coordinates for a blackbody at given temperature.

    Integrates Planck's law against the color matching functions.
    """
    # Calculate spectral radiance at each wavelength
    radiance = np.array([
        planck_spectral_radiance(w * 1e-9, temp_k, c2)
        for w in wavelengths
    ])

    # Integrate against CMFs (using simple rectangular integration)
    X = np.sum(radiance * x_bar)
    Y = np.sum(radiance * y_bar)
    Z = np.sum(radiance * z_bar)

    # Convert to chromaticity
    total = X + Y + Z
    x = X / total
    y = Y / total

    return x, y


def lcms_white_point_from_temp(temp_k):
    """
    Reproduce the lcms/moxcms polynomial approximation for white point from temperature.
    This is from the code shown in the conversation.
    """
    temp_k2 = temp_k * temp_k
    temp_k3 = temp_k2 * temp_k

    if 4000 < temp_k <= 7000:
        x = (-4.6070 * (1e9 / temp_k3) +
             2.9678 * (1e6 / temp_k2) +
             0.09911 * (1e3 / temp_k) +
             0.244063)
    elif 7000 < temp_k <= 25000:
        x = (-2.0064 * (1e9 / temp_k3) +
             1.9018 * (1e6 / temp_k2) +
             0.24748 * (1e3 / temp_k) +
             0.237040)
    else:
        return None, None

    # Obtain y from x
    y = -3.000 * x * x + 2.870 * x - 0.275

    return x, y


def daylight_locus_chromaticity(temp_k):
    """
    Calculate chromaticity on the CIE daylight locus.

    The daylight locus is NOT the same as the Planckian locus!
    It was derived from measurements of actual daylight, which differs
    from blackbody radiation due to atmospheric scattering and absorption.

    These formulas are from CIE 015:2004.
    """
    # Calculate x coordinate on daylight locus
    if 4000 <= temp_k <= 7000:
        x = (-4.6070e9 / temp_k**3 +
             2.9678e6 / temp_k**2 +
             0.09911e3 / temp_k +
             0.244063)
    elif 7000 < temp_k <= 25000:
        x = (-2.0064e9 / temp_k**3 +
             1.9018e6 / temp_k**2 +
             0.24748e3 / temp_k +
             0.237040)
    else:
        return None, None

    # Calculate y from x (the daylight locus curve)
    y = -3.000 * x**2 + 2.870 * x - 0.275

    return x, y


def main():
    print("=" * 70)
    print("D65 Chromaticity Coordinate Analysis")
    print("=" * 70)
    print()

    print(f"CIE Official D65: x = {CIE_D65_X}, y = {CIE_D65_Y}")
    print()

    # Load color matching functions
    wavelengths, x_bar, y_bar, z_bar = load_cie_cmf()

    print("-" * 70)
    print("Method 1: Blackbody calculation from Planck's law")
    print("-" * 70)
    print()

    # Calculate with old constant at 6500K
    x_old_6500, y_old_6500 = blackbody_chromaticity(6500, C2_OLD, wavelengths, x_bar, y_bar, z_bar)
    print(f"Old c2 ({C2_OLD}) at 6500K:")
    print(f"  x = {x_old_6500:.6f}, y = {y_old_6500:.6f}")
    print(f"  Δx = {(x_old_6500 - CIE_D65_X)*1e5:+.2f}e-5, Δy = {(y_old_6500 - CIE_D65_Y)*1e5:+.2f}e-5")
    print()

    # Calculate with new constant at 6500K
    x_new_6500, y_new_6500 = blackbody_chromaticity(6500, C2_NEW, wavelengths, x_bar, y_bar, z_bar)
    print(f"New c2 ({C2_NEW}) at 6500K:")
    print(f"  x = {x_new_6500:.6f}, y = {y_new_6500:.6f}")
    print(f"  Δx = {(x_new_6500 - CIE_D65_X)*1e5:+.2f}e-5, Δy = {(y_new_6500 - CIE_D65_Y)*1e5:+.2f}e-5")
    print()

    # Calculate with new constant at 6504K (the "corrected" temperature)
    x_new_6504, y_new_6504 = blackbody_chromaticity(6504, C2_NEW, wavelengths, x_bar, y_bar, z_bar)
    print(f"New c2 ({C2_NEW}) at 6504K:")
    print(f"  x = {x_new_6504:.6f}, y = {y_new_6504:.6f}")
    print(f"  Δx = {(x_new_6504 - CIE_D65_X)*1e5:+.2f}e-5, Δy = {(y_new_6504 - CIE_D65_Y)*1e5:+.2f}e-5")
    print()

    print("-" * 70)
    print("Method 2: CIE Daylight Locus formula (used by lcms/moxcms)")
    print("-" * 70)
    print()
    print("NOTE: This is NOT the Planckian locus! It's a polynomial fit to")
    print("measured daylight chromaticities from CIE 015:2004.")
    print()

    # Daylight locus at various temperatures
    for temp in [6500, 6504]:
        x_dl, y_dl = daylight_locus_chromaticity(temp)
        print(f"Daylight locus at {temp}K:")
        print(f"  x = {x_dl:.6f}, y = {y_dl:.6f}")
        print(f"  Δx = {(x_dl - CIE_D65_X)*1e5:+.2f}e-5, Δy = {(y_dl - CIE_D65_Y)*1e5:+.2f}e-5")
        print()

    print("-" * 70)
    print("Summary: Error magnitudes (in units of 1e-5)")
    print("-" * 70)
    print()
    print(f"{'Method':<35} {'Δx':>10} {'Δy':>10} {'Euclidean':>12}")
    print("-" * 70)

    methods = [
        ("Planckian, old c2 @ 6500K", x_old_6500, y_old_6500),
        ("Planckian, new c2 @ 6500K", x_new_6500, y_new_6500),
        ("Planckian, new c2 @ 6504K", x_new_6504, y_new_6504),
        ("Daylight locus @ 6500K", *daylight_locus_chromaticity(6500)),
        ("Daylight locus @ 6504K", *daylight_locus_chromaticity(6504)),
    ]

    for name, x, y in methods:
        dx = (x - CIE_D65_X) * 1e5
        dy = (y - CIE_D65_Y) * 1e5
        dist = math.sqrt(dx*dx + dy*dy)
        print(f"{name:<35} {dx:>+10.2f} {dy:>+10.2f} {dist:>12.2f}")

    print()
    print("-" * 70)
    print("Key Insight: Daylight Locus ≠ Planckian Locus")
    print("-" * 70)
    print()
    print("The D illuminants are based on MEASURED DAYLIGHT, not pure blackbody")
    print("radiation. Daylight differs from blackbody due to:")
    print("  - Rayleigh scattering (blue sky)")
    print("  - Atmospheric absorption bands")
    print("  - Cloud/aerosol scattering")
    print()
    print("The lcms polynomial is the CIE DAYLIGHT LOCUS formula, which was")
    print("specifically fit to measured daylight chromaticities - NOT derived")
    print("from Planck's law.")
    print()
    print("This is why pure blackbody calculation at 6500K gives y ≈ 0.3236,")
    print("while CIE D65 has y = 0.32903. The ~0.0054 difference represents")
    print("how much real daylight deviates from blackbody radiation.")
    print()
    print("-" * 70)
    print("Historical Notes:")
    print("-" * 70)
    print("- CIE D65 official coordinates: x=0.31272, y=0.32903")
    print("- Pre-1968 c2 = 0.014380 m⋅K")
    print("- Post-1968 c2 = 0.014388 m⋅K")
    print("- The '6500K' in D65 refers to CORRELATED color temperature")
    print("- CCT is the blackbody temp that appears most similar, not the actual temp")
    print("- When c2 was updated, same daylight chromaticity → CCT 6504K")
    print("- Errors shown in units of 1e-5 (the 5th decimal place)")


def find_best_temperature_for_d65():
    """
    Find the temperature that, when input to the daylight locus formula,
    produces chromaticity closest to official D65.
    """
    print("=" * 70)
    print("Experiment: Finding optimal temperature for D65")
    print("=" * 70)
    print()

    best_t = None
    best_error = float('inf')

    # Search from 6400 to 6600 in small increments
    for t_int in range(64000, 66000):
        t = t_int / 10.0  # 0.1K resolution
        x, y = daylight_locus_chromaticity(t)
        if x is None:
            continue
        error = math.sqrt((x - CIE_D65_X)**2 + (y - CIE_D65_Y)**2)
        if error < best_error:
            best_error = error
            best_t = t

    x_best, y_best = daylight_locus_chromaticity(best_t)
    print(f"Best temperature: {best_t:.1f} K")
    print(f"  Produces: x = {x_best:.6f}, y = {y_best:.6f}")
    print(f"  Error: {best_error * 1e5:.2f}e-5 (Euclidean)")
    print()

    return best_t


def test_c2_conversion_hypothesis():
    """
    Test the hypothesis: does 6500 × (new_c2/old_c2) give the right answer?
    """
    print("=" * 70)
    print("Experiment: Testing the c2 ratio conversion")
    print("=" * 70)
    print()

    old_c2 = 0.014380  # m·K (pre-1968)
    new_c2 = 0.01438776877  # m·K (current CODATA, more precise)

    # The Wikipedia formula
    converted_t = 6500 * (new_c2 / old_c2)
    print(f"Conversion: 6500 × ({new_c2}/{old_c2})")
    print(f"          = 6500 × {new_c2/old_c2:.10f}")
    print(f"          = {converted_t:.4f} K")
    print()

    x_conv, y_conv = daylight_locus_chromaticity(converted_t)
    dx = (x_conv - CIE_D65_X) * 1e5
    dy = (y_conv - CIE_D65_Y) * 1e5
    error = math.sqrt(dx*dx + dy*dy)

    print(f"Daylight locus at {converted_t:.4f} K:")
    print(f"  x = {x_conv:.6f}, y = {y_conv:.6f}")
    print(f"  Δx = {dx:+.2f}e-5, Δy = {dy:+.2f}e-5")
    print(f"  Euclidean error: {error:.2f}e-5")
    print()

    # Compare with round values
    print("Comparison with round values:")
    for t in [6500, 6503, 6503.5, 6504, 6505]:
        x, y = daylight_locus_chromaticity(t)
        dx = (x - CIE_D65_X) * 1e5
        dy = (y - CIE_D65_Y) * 1e5
        err = math.sqrt(dx*dx + dy*dy)
        print(f"  T={t:7.1f}K: Δx={dx:+7.2f}e-5, Δy={dy:+7.2f}e-5, err={err:6.2f}e-5")
    print()


def test_inverse_hypothesis():
    """
    What if the formula was derived with NEW c2, and we need to convert backwards?
    """
    print("=" * 70)
    print("Experiment: What if polynomial uses NEW c2?")
    print("=" * 70)
    print()

    old_c2 = 0.014380
    new_c2 = 0.01438776877

    # If polynomial uses new c2, and D65 was defined at old-6500K,
    # then we'd need: T_input = 6500 × (old_c2/new_c2) to get old behavior?
    # No wait, that doesn't make sense either...

    # Let's think differently:
    # D65 has a fixed chromaticity (0.31272, 0.32903)
    # This chromaticity, when projected onto the Planckian locus:
    #   - with old c2 → CCT = 6500K
    #   - with new c2 → CCT = 6504K (approximately)
    #
    # The daylight locus polynomial maps CCT → chromaticity
    # If it was fit using (old_CCT, chromaticity) pairs:
    #   - polynomial(6500) should give D65
    # If it was fit using (new_CCT, chromaticity) pairs:
    #   - polynomial(6504) should give D65

    print("The polynomial maps temperature → chromaticity")
    print("D65 chromaticity is fixed at (0.31272, 0.32903)")
    print()
    print("If polynomial was fit with OLD CCT labels:")
    print("  → polynomial(6500) should give D65")
    print("If polynomial was fit with NEW CCT labels:")
    print("  → polynomial(6504) should give D65")
    print()

    for t in [6500, 6504]:
        x, y = daylight_locus_chromaticity(t)
        err = math.sqrt((x - CIE_D65_X)**2 + (y - CIE_D65_Y)**2) * 1e5
        print(f"polynomial({t}) error: {err:.2f}e-5")
    print()
    print("Since polynomial(6504) has lower error, the polynomial")
    print("appears to expect NEW CCT values as input.")
    print()


def analyze_polynomial_origin():
    """
    Try to understand where the polynomial coefficients came from.
    """
    print("=" * 70)
    print("Experiment: Analyzing polynomial behavior")
    print("=" * 70)
    print()

    # The polynomial for x (4000-7000K range):
    # x = -4.6070e9/T³ + 2.9678e6/T² + 0.09911e3/T + 0.244063

    # Let's see what happens at different temperatures
    print("Daylight locus x,y values at various temperatures:")
    print(f"{'T (K)':<10} {'x':<12} {'y':<12}")
    print("-" * 34)
    for t in [5000, 5500, 6000, 6500, 7000, 7500, 8000, 10000]:
        x, y = daylight_locus_chromaticity(t)
        if x:
            print(f"{t:<10} {x:<12.6f} {y:<12.6f}")
    print()

    # What's the slope of x with respect to T near 6500K?
    t1, t2 = 6500, 6504
    x1, _ = daylight_locus_chromaticity(t1)
    x2, _ = daylight_locus_chromaticity(t2)
    dx_dt = (x2 - x1) / (t2 - t1)
    print(f"dx/dT near 6500K: {dx_dt:.2e} per K")
    print(f"Change in x from 6500→6504K: {(x2-x1):.6f}")
    print(f"That's {(x2-x1)*1e5:.2f}e-5")
    print()


def check_if_polynomial_changed():
    """
    The CIE daylight locus formula was published in 1967.
    The c2 change was in 1968.
    Did the polynomial get updated?
    """
    print("=" * 70)
    print("Experiment: Did the polynomial coefficients change after 1968?")
    print("=" * 70)
    print()

    # If the polynomial was NOT updated after 1968:
    #   - It expects old CCT values
    #   - polynomial(6500) should give D65
    #   - But we see polynomial(6504) is closer
    #   - Contradiction!
    #
    # Possible resolutions:
    # 1. The polynomial WAS updated to use new CCT parameterization
    # 2. The polynomial was always designed to take "true" CCT
    #    (nearest point on Planckian locus), and the labels just shifted
    # 3. The polynomial never passed through D65 exactly at any T

    print("Key question: What temperature gives EXACT D65 chromaticity?")
    print()

    # Solve for T such that daylight_locus(T) = D65
    # x(T) = 0.31272
    # -4.6070e9/T³ + 2.9678e6/T² + 0.09911e3/T + 0.244063 = 0.31272

    # Let's use numerical search
    best_t = None
    best_x_err = float('inf')

    for t_int in range(60000, 70000):
        t = t_int / 10.0
        x, y = daylight_locus_chromaticity(t)
        if x is None:
            continue
        x_err = abs(x - CIE_D65_X)
        if x_err < best_x_err:
            best_x_err = x_err
            best_t = t

    x_at_best, y_at_best = daylight_locus_chromaticity(best_t)
    print(f"Temperature where x = {CIE_D65_X}: T ≈ {best_t:.1f} K")
    print(f"  At this T: x = {x_at_best:.6f}, y = {y_at_best:.6f}")
    print(f"  x error: {(x_at_best - CIE_D65_X)*1e5:.2f}e-5")
    print(f"  y error: {(y_at_best - CIE_D65_Y)*1e5:.2f}e-5")
    print()

    # Now find T where y is closest
    best_t_y = None
    best_y_err = float('inf')

    for t_int in range(60000, 70000):
        t = t_int / 10.0
        x, y = daylight_locus_chromaticity(t)
        if y is None:
            continue
        y_err = abs(y - CIE_D65_Y)
        if y_err < best_y_err:
            best_y_err = y_err
            best_t_y = t

    x_at_best_y, y_at_best_y = daylight_locus_chromaticity(best_t_y)
    print(f"Temperature where y = {CIE_D65_Y}: T ≈ {best_t_y:.1f} K")
    print(f"  At this T: x = {x_at_best_y:.6f}, y = {y_at_best_y:.6f}")
    print(f"  x error: {(x_at_best_y - CIE_D65_X)*1e5:.2f}e-5")
    print(f"  y error: {(y_at_best_y - CIE_D65_Y)*1e5:.2f}e-5")
    print()

    print("KEY INSIGHT:")
    print(f"  Best T for x match: {best_t:.1f} K")
    print(f"  Best T for y match: {best_t_y:.1f} K")
    print(f"  These differ by {abs(best_t - best_t_y):.1f} K!")
    print()
    print("This means the official D65 point does NOT lie exactly")
    print("on the daylight locus curve defined by this polynomial!")
    print("The polynomial is an approximation that passes NEAR D65,")
    print("but not exactly through it.")
    print()


def understand_d65_origin():
    """
    The D65 SPD is computed from the daylight model using M1, M2 coefficients.
    Let's see what the model says.
    """
    print("=" * 70)
    print("Experiment: Understanding D65's actual origin")
    print("=" * 70)
    print()

    print("According to the CIE daylight model, a D-illuminant SPD is:")
    print("  S_D(λ) = S_0(λ) + M1·S_1(λ) + M2·S_2(λ)")
    print()
    print("Where M1 and M2 are computed from the chromaticity (x_D, y_D):")
    print("  M1 = (-1.3515 - 1.7703·x_D + 5.9114·y_D) / M")
    print("  M2 = (0.0300 - 31.4424·x_D + 30.0717·y_D) / M")
    print("  M  = 0.0241 + 0.2562·x_D - 0.7341·y_D")
    print()

    # Compute M1, M2 for D65
    x_d, y_d = CIE_D65_X, CIE_D65_Y
    M = 0.0241 + 0.2562 * x_d - 0.7341 * y_d
    M1 = (-1.3515 - 1.7703 * x_d + 5.9114 * y_d) / M
    M2 = (0.0300 - 31.4424 * x_d + 30.0717 * y_d) / M

    print(f"For D65 (x={x_d}, y={y_d}):")
    print(f"  M  = {M:.6f}")
    print(f"  M1 = {M1:.6f}")
    print(f"  M2 = {M2:.6f}")
    print()

    # According to the Wikipedia note, M1 and M2 should be rounded to 3 decimals
    M1_rounded = round(M1, 3)
    M2_rounded = round(M2, 3)
    print(f"Rounded to 3 decimals (as CIE specifies):")
    print(f"  M1 = {M1_rounded:.3f}")
    print(f"  M2 = {M2_rounded:.3f}")
    print()

    # Now, what if D65's chromaticity was computed from the SPD
    # using the rounded M1, M2 values?
    # The chromaticity would come from integrating the SPD against CMFs.
    # We don't have the S_0, S_1, S_2 data here, but we can note:
    print("KEY QUESTION: Which came first?")
    print("  A) The chromaticity (x_D, y_D) was specified, then SPD computed")
    print("  B) The SPD was computed first (from T → polynomial → M1,M2 → SPD),")
    print("     then chromaticity computed from SPD")
    print()
    print("If (A): The polynomial giving x_D from T is the PRIMARY definition,")
    print("        and any error means D65 doesn't quite match the SPD model.")
    print()
    print("If (B): The SPD is the PRIMARY definition, and the tabulated")
    print("        chromaticity is a DERIVED value (possibly with rounding).")
    print()

    # Let's check what chromaticity we get from the polynomial at different T
    # and what M1, M2 values that implies
    print("M1, M2 values at various temperatures:")
    print(f"{'T (K)':<10} {'x':<10} {'y':<10} {'M1':<10} {'M2':<10}")
    print("-" * 50)
    for t in [6500, 6503.5, 6504, 6506.6]:
        x, y = daylight_locus_chromaticity(t)
        m = 0.0241 + 0.2562 * x - 0.7341 * y
        m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m
        m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m
        print(f"{t:<10} {x:<10.6f} {y:<10.6f} {m1:<10.4f} {m2:<10.4f}")
    print()

    print("Official D65:")
    print(f"{'D65':<10} {CIE_D65_X:<10.6f} {CIE_D65_Y:<10.6f} {M1:<10.4f} {M2:<10.4f}")
    print()


def find_precise_d65_temperature():
    """
    Find the precise temperature that minimizes error to official D65,
    using high-precision numerical search.
    """
    print("=" * 70)
    print("Deriving the 'true' D65 temperature")
    print("=" * 70)
    print()

    def error_func(t):
        x, y = daylight_locus_chromaticity(t)
        if x is None:
            return float('inf')
        return math.sqrt((x - CIE_D65_X)**2 + (y - CIE_D65_Y)**2)

    # First, find approximate minimum with coarse search
    best_t_coarse = 6500.0
    best_err_coarse = error_func(6500.0)
    # Coarse pass at 0.1K
    for t_int in range(64000, 66000):
        t = t_int / 10.0
        err = error_func(t)
        if err < best_err_coarse:
            best_err_coarse = err
            best_t_coarse = t
    # Finer pass around the coarse minimum at 0.001K
    for t_int in range(int((best_t_coarse - 2) * 1000), int((best_t_coarse + 2) * 1000)):
        t = t_int / 1000.0
        err = error_func(t)
        if err < best_err_coarse:
            best_err_coarse = err
            best_t_coarse = t

    # Now refine with golden section search around that point
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    a = best_t_coarse - 1.0
    b = best_t_coarse + 1.0
    tol = 1e-12

    c = b - resphi * (b - a)
    d = a + resphi * (b - a)

    iterations = 0
    while abs(b - a) > tol and iterations < 1000:
        if error_func(c) < error_func(d):
            b = d
        else:
            a = c
        c = b - resphi * (b - a)
        d = a + resphi * (b - a)
        iterations += 1

    best_t = (a + b) / 2
    x_best, y_best = daylight_locus_chromaticity(best_t)
    error = error_func(best_t)

    print(f"Optimal temperature (Euclidean distance minimized):")
    print(f"  T = {best_t:.10f} K")
    print(f"  T ≈ {best_t:.4f} K")
    print()
    print(f"At this temperature:")
    print(f"  x = {x_best:.10f}")
    print(f"  y = {y_best:.10f}")
    print()
    print(f"Official D65:")
    print(f"  x = {CIE_D65_X}")
    print(f"  y = {CIE_D65_Y}")
    print()
    print(f"Errors:")
    print(f"  Δx = {(x_best - CIE_D65_X):.10f} = {(x_best - CIE_D65_X)*1e5:+.4f}e-5")
    print(f"  Δy = {(y_best - CIE_D65_Y):.10f} = {(y_best - CIE_D65_Y)*1e5:+.4f}e-5")
    print(f"  Euclidean = {error:.10f} = {error*1e5:.4f}e-5")
    print()

    # Also find the T that gives exact x match using bisection
    # x(T) is monotonically decreasing, so we can use bisection
    def x_at(t):
        x, _ = daylight_locus_chromaticity(t)
        return x

    # Find T where x = CIE_D65_X using bisection
    a, b = 6400.0, 6600.0
    while abs(b - a) > tol:
        mid = (a + b) / 2
        if x_at(mid) > CIE_D65_X:  # x decreases as T increases
            a = mid
        else:
            b = mid
    t_for_x = (a + b) / 2
    x_at_t, y_at_t = daylight_locus_chromaticity(t_for_x)

    print(f"Temperature for exact x match:")
    print(f"  T = {t_for_x:.10f} K")
    print(f"  x = {x_at_t:.10f} (error: {(x_at_t - CIE_D65_X)*1e5:+.6f}e-5)")
    print(f"  y = {y_at_t:.10f} (error: {(y_at_t - CIE_D65_Y)*1e5:+.6f}e-5)")
    print()

    # Find T where y = CIE_D65_Y using bisection
    # y(T) is also monotonically decreasing
    def y_at(t):
        _, y = daylight_locus_chromaticity(t)
        return y

    a, b = 6400.0, 6600.0
    while abs(b - a) > tol:
        mid = (a + b) / 2
        if y_at(mid) > CIE_D65_Y:  # y decreases as T increases
            a = mid
        else:
            b = mid
    t_for_y = (a + b) / 2
    x_at_t, y_at_t = daylight_locus_chromaticity(t_for_y)

    print(f"Temperature for exact y match:")
    print(f"  T = {t_for_y:.10f} K")
    print(f"  x = {x_at_t:.10f} (error: {(x_at_t - CIE_D65_X)*1e5:+.6f}e-5)")
    print(f"  y = {y_at_t:.10f} (error: {(y_at_t - CIE_D65_Y)*1e5:+.6f}e-5)")
    print()

    # Compare with various proposed values
    print("-" * 70)
    print("Comparison of proposed D65 temperatures:")
    print("-" * 70)
    print(f"{'Temperature':<25} {'Δx (e-5)':<12} {'Δy (e-5)':<12} {'Euclid (e-5)':<12}")
    print("-" * 70)

    test_temps = [
        ("6500 K (nominal)", 6500.0),
        ("6503.5116 K (c2 ratio)", 6500 * 1.0005402483),
        ("6504 K (lcms/moxcms)", 6504.0),
        (f"{best_t:.4f} K (optimal)", best_t),
        (f"{t_for_x:.4f} K (x-match)", t_for_x),
        (f"{t_for_y:.4f} K (y-match)", t_for_y),
    ]

    for name, t in test_temps:
        x, y = daylight_locus_chromaticity(t)
        dx = (x - CIE_D65_X) * 1e5
        dy = (y - CIE_D65_Y) * 1e5
        err = math.sqrt(dx*dx + dy*dy)
        print(f"{name:<25} {dx:>+11.4f} {dy:>+11.4f} {err:>11.4f}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print(f"The 'true' D65 temperature (minimizing Euclidean distance) is:")
    print(f"  T = {best_t:.6f} K")
    print()
    print(f"However, the official D65 point (0.31272, 0.32903) does NOT lie")
    print(f"exactly on the daylight locus curve. The minimum achievable error")
    print(f"is {error*1e5:.4f}e-5, which is in the 5th decimal place.")
    print()
    print(f"For practical purposes:")
    print(f"  - 6504 K gives {error_func(6504)*1e5:.2f}e-5 error (what lcms uses)")
    print(f"  - {best_t:.2f} K gives {error*1e5:.2f}e-5 error (optimal)")
    print(f"  - Improvement: {(error_func(6504) - error)*1e5:.2f}e-5")
    print()

    return best_t


def final_synthesis():
    """
    Put it all together.
    """
    print("=" * 70)
    print("FINAL SYNTHESIS")
    print("=" * 70)
    print()

    print("The situation is more complex than 'old c2 vs new c2':")
    print()
    print("1. D65 has an OFFICIAL chromaticity: (0.31272, 0.32903)")
    print("   This is the authoritative definition.")
    print()
    print("2. The daylight locus polynomial maps T → (x, y)")
    print("   But no value of T produces EXACTLY (0.31272, 0.32903)!")
    print("   - T=6503.6K gives correct x, but y is wrong by 9.5e-5")
    print("   - T=6509.5K gives correct y, but x is wrong by 9.5e-5")
    print("   - T=6506.6K minimizes total error (6.7e-5)")
    print()
    print("3. The polynomial is an APPROXIMATION to the daylight locus.")
    print("   It was fit to measured daylight chromaticities, not designed")
    print("   to pass exactly through the canonical D-illuminants.")
    print()
    print("4. The 'c2 ratio' correction (6500 × 1.000540 = 6503.5K) gives")
    print("   x very close to official D65 (error 0.2e-5), but y is still")
    print("   off by 9.7e-5. This suggests:")
    print("   - The polynomial's x(T) formula was calibrated with old c2")
    print("   - The polynomial's y(x) formula has independent error")
    print()
    print("5. The 'correct' approach depends on your goal:")
    print("   - If you want official D65: use hardcoded (0.31272, 0.32903)")
    print("   - If you want 'D65 per the polynomial': use T≈6504K")
    print("   - If you want to minimize error vs official: use T≈6506.6K")
    print()
    print("6. lcms uses 6504K, which gives ~9e-5 error. This is probably")
    print("   'close enough' for most purposes, and avoids the question")
    print("   of which fractional temperature is 'really' correct.")
    print()


if __name__ == "__main__":
    # Just run the key analysis
    find_precise_d65_temperature()
