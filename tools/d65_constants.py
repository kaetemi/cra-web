#!/usr/bin/env python3
"""
D65 Whitepoint Analysis: Precision, History, and Derivation

This script analyzes the relationship between the D65 illuminant and various
temperature values, exploring the historical c2 radiation constant change
and the daylight locus polynomial approximations.

Key findings:
- D65 is defined by its SPD, not by temperature
- The "6500K" name refers to old (pre-1968) CCT
- Modern CIE formulas expect new (post-1968) CCT (~6504K)
- The official D65 point doesn't lie exactly on the daylight locus curve
"""

import math

# =============================================================================
# Constants
# =============================================================================

# CIE official D65 chromaticity coordinates (5 decimal places)
CIE_D65_X = 0.31272
CIE_D65_Y = 0.32903

# Second radiation constant (c₂) in Planck's law
# See CIE 15:2004 Appendix E for authoritative values
C2_OLD = 0.01438     # m·K (1931 definition, used when D-illuminants were defined)
C2_ITS90 = 0.014388  # m·K (ITS-90, International Temperature Scale 1990)
C2_CODATA = 0.01438776877  # m·K (CODATA physical measurement - NOT used by CIE)

# The CIE standard (CIE 15:2004) specifies ITS-90, not CODATA
C2_NEW = C2_ITS90

# Derived values
C2_RATIO = C2_NEW / C2_OLD  # = 1.00055632...


# =============================================================================
# Daylight Locus Functions
# =============================================================================

def daylight_locus_chromaticity(temp_k: float) -> tuple[float, float] | tuple[None, None]:
    """
    Calculate chromaticity on the CIE daylight locus.

    This is the CIE formula from CIE 15:2004 for computing D-illuminant
    chromaticity from correlated color temperature.

    IMPORTANT: This formula expects MODERN (post-1968) CCT values.

    Args:
        temp_k: Correlated color temperature in Kelvin (modern scale)

    Returns:
        (x, y) chromaticity coordinates, or (None, None) if out of range
    """
    if temp_k < 4000 or temp_k > 25000:
        return None, None

    # Calculate x coordinate
    if 4000 <= temp_k <= 7000:
        x = (-4.6070e9 / temp_k**3 +
              2.9678e6 / temp_k**2 +
              0.09911e3 / temp_k +
              0.244063)
    else:  # 7000 < temp_k <= 25000
        x = (-2.0064e9 / temp_k**3 +
              1.9018e6 / temp_k**2 +
              0.24748e3 / temp_k +
              0.237040)

    # Calculate y from x (the daylight locus curve shape)
    # This quadratic was derived by Judd, MacAdam, and Wyszecki (1964)
    y = -3.000 * x**2 + 2.870 * x - 0.275

    return x, y


def compute_error(temp_k: float) -> tuple[float, float, float]:
    """
    Compute the error between daylight locus at temp_k and official D65.

    Returns:
        (dx, dy, euclidean) errors, all in raw units (not scaled)
    """
    x, y = daylight_locus_chromaticity(temp_k)
    if x is None:
        return float('inf'), float('inf'), float('inf')

    dx = x - CIE_D65_X
    dy = y - CIE_D65_Y
    euclidean = math.sqrt(dx**2 + dy**2)

    return dx, dy, euclidean


# =============================================================================
# Search Functions
# =============================================================================

def find_optimal_temperature() -> float:
    """Find temperature that minimizes Euclidean distance to D65."""
    # Coarse search
    best_t = 6500.0
    best_err = compute_error(6500.0)[2]

    for t_int in range(64000, 66000):
        t = t_int / 10.0
        err = compute_error(t)[2]
        if err < best_err:
            best_err = err
            best_t = t

    # Fine search around minimum
    for t_int in range(int((best_t - 2) * 1000), int((best_t + 2) * 1000)):
        t = t_int / 1000.0
        err = compute_error(t)[2]
        if err < best_err:
            best_err = err
            best_t = t

    # Golden section refinement
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    a, b = best_t - 0.1, best_t + 0.1
    tol = 1e-12

    c = b - resphi * (b - a)
    d = a + resphi * (b - a)

    for _ in range(1000):
        if abs(b - a) <= tol:
            break
        if compute_error(c)[2] < compute_error(d)[2]:
            b = d
        else:
            a = c
        c = b - resphi * (b - a)
        d = a + resphi * (b - a)

    return (a + b) / 2


def find_exact_x_temperature() -> float:
    """Find temperature where x exactly matches D65's x coordinate."""
    a, b = 6400.0, 6600.0
    tol = 1e-12

    while abs(b - a) > tol:
        mid = (a + b) / 2
        x, _ = daylight_locus_chromaticity(mid)
        if x > CIE_D65_X:  # x decreases as T increases
            a = mid
        else:
            b = mid

    return (a + b) / 2


def find_exact_y_temperature() -> float:
    """Find temperature where y exactly matches D65's y coordinate."""
    a, b = 6400.0, 6600.0
    tol = 1e-12

    while abs(b - a) > tol:
        mid = (a + b) / 2
        _, y = daylight_locus_chromaticity(mid)
        if y > CIE_D65_Y:  # y decreases as T increases
            a = mid
        else:
            b = mid

    return (a + b) / 2


# =============================================================================
# Output Functions
# =============================================================================

def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


def print_subsection(title: str):
    """Print a subsection header."""
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)
    print()


def main():
    """Run all analyses and output comprehensive results."""

    # =========================================================================
    print_section("D65 WHITEPOINT ANALYSIS")
    # =========================================================================

    print("This analysis explores the relationship between the D65 illuminant")
    print("and color temperature, including historical changes to the radiation")
    print("constant and the CIE daylight locus polynomial approximations.")

    # =========================================================================
    print_section("1. CONSTANTS AND DEFINITIONS")
    # =========================================================================

    print("Official D65 Chromaticity (CIE):")
    print(f"  x = {CIE_D65_X}")
    print(f"  y = {CIE_D65_Y}")
    print()

    print("Second Radiation Constant (c₂) in Planck's Law:")
    print(f"  Old c₂ (1931):      {C2_OLD} m·K")
    print(f"  New c₂ (ITS-90):    {C2_ITS90} m·K  ← CIE 15:2004 standard")
    print(f"  CODATA (physical):  {C2_CODATA} m·K  (not used by CIE)")
    print(f"  Ratio (ITS-90/old): {C2_RATIO:.11f}")
    print()

    print("Temperature Scale Conversion (per CIE 15:2004 Appendix E):")
    print(f"  T_new = T_old × (c₂_new / c₂_old)")
    print()
    c2_ratio_its90 = C2_ITS90 / C2_OLD
    c2_ratio_codata = C2_CODATA / C2_OLD
    print(f"  Using ITS-90 (correct):  6500 × ({C2_ITS90}/{C2_OLD}) = 6500 × {c2_ratio_its90:.11f} = {6500 * c2_ratio_its90:.6f}K")
    print(f"  Using CODATA (wrong):    6500 × ({C2_CODATA}/{C2_OLD}) = 6500 × {c2_ratio_codata:.11f} = {6500 * c2_ratio_codata:.6f}K")

    # =========================================================================
    print_section("2. KEY TEMPERATURES")
    # =========================================================================

    # Calculate all key temperatures
    t_nominal = 6500.0
    t_its90 = 6500.0 * (C2_ITS90 / C2_OLD)
    t_codata = 6500.0 * (C2_CODATA / C2_OLD)
    t_lcms = 6504.0
    t_exact_x = find_exact_x_temperature()
    t_exact_y = find_exact_y_temperature()
    t_optimal = find_optimal_temperature()

    print(f"{'Description':<35} {'Temperature (K)':<20}")
    print("-" * 55)
    print(f"{'Nominal (1931 scale)':<35} {t_nominal:<20.10f}")
    print(f"{'ITS-90 conversion (CIE 15:2004)':<35} {t_its90:<20.10f}")
    print(f"{'CODATA conversion (incorrect)':<35} {t_codata:<20.10f}")
    print(f"{'lcms/moxcms rounded value':<35} {t_lcms:<20.10f}")
    print(f"{'Exact x match':<35} {t_exact_x:<20.10f}")
    print(f"{'Exact y match':<35} {t_exact_y:<20.10f}")
    print(f"{'Optimal (min Euclidean)':<35} {t_optimal:<20.10f}")
    print()

    print(f"Gap between exact x and exact y temperatures: {abs(t_exact_x - t_exact_y):.4f}K")
    print("(This gap proves D65 does not lie exactly on the daylight locus curve)")

    # =========================================================================
    print_section("3. CHROMATICITY AT EACH TEMPERATURE")
    # =========================================================================

    temperatures = [
        ("6500.0K (nominal)", t_nominal),
        (f"{t_its90:.4f}K (ITS-90)", t_its90),
        (f"{t_codata:.4f}K (CODATA)", t_codata),
        ("6504.0K (lcms)", t_lcms),
        (f"{t_exact_x:.4f}K (exact x)", t_exact_x),
        (f"{t_exact_y:.4f}K (exact y)", t_exact_y),
        (f"{t_optimal:.4f}K (optimal)", t_optimal),
    ]

    print(f"{'Temperature':<25} {'x':<16} {'y':<16}")
    print("-" * 57)

    for name, t in temperatures:
        x, y = daylight_locus_chromaticity(t)
        print(f"{name:<25} {x:<16.10f} {y:<16.10f}")

    print()
    print(f"{'Official D65':<25} {CIE_D65_X:<16.10f} {CIE_D65_Y:<16.10f}")

    # =========================================================================
    print_section("4. ERROR ANALYSIS")
    # =========================================================================

    print(f"{'Temperature':<25} {'Δx (×10⁻⁵)':<14} {'Δy (×10⁻⁵)':<14} {'Euclidean (×10⁻⁵)':<18}")
    print("-" * 71)

    for name, t in temperatures:
        dx, dy, euclidean = compute_error(t)
        print(f"{name:<25} {dx*1e5:>+13.4f} {dy*1e5:>+13.4f} {euclidean*1e5:>17.4f}")

    # =========================================================================
    print_section("5. KEY INSIGHTS")
    # =========================================================================

    print("5.1 The ITS-90 Temperature Conversion")
    print("-" * 40)
    dx_its90, dy_its90, err_its90 = compute_error(t_its90)
    dx_codata, dy_codata, err_codata = compute_error(t_codata)
    print(f"ITS-90:  T = 6500 × ({C2_ITS90}/{C2_OLD}) = {t_its90:.6f}K")
    print(f"  x error: {dx_its90*1e5:+.4f}×10⁻⁵  (nearly perfect!)")
    print(f"  y error: {dy_its90*1e5:+.4f}×10⁻⁵")
    print()
    print(f"CODATA:  T = 6500 × ({C2_CODATA}/{C2_OLD}) = {t_codata:.6f}K")
    print(f"  x error: {dx_codata*1e5:+.4f}×10⁻⁵  (slightly worse)")
    print(f"  y error: {dy_codata*1e5:+.4f}×10⁻⁵")
    print()
    print("The ITS-90 value gives better x accuracy, confirming the CIE formula")
    print("uses the ITS-90 temperature scale, not CODATA physical constants.")

    print()
    print("5.2 The y(x) Quadratic Error")
    print("-" * 40)
    dx_at_exact_x, dy_at_exact_x, _ = compute_error(t_exact_x)
    print(f"At T = {t_exact_x:.4f}K (exact x match):")
    print(f"  x error: {dx_at_exact_x*1e5:+.6f}×10⁻⁵  (essentially zero)")
    print(f"  y error: {dy_at_exact_x*1e5:+.6f}×10⁻⁵  (irreducible)")
    print()
    print("The ~9.5×10⁻⁵ y error is from the quadratic y = -3x² + 2.87x - 0.275")
    print("not being perfectly accurate at D65's x coordinate.")

    print()
    print("5.3 D65 Does Not Lie on the Curve")
    print("-" * 40)
    print(f"Temperature for exact x: {t_exact_x:.4f}K")
    print(f"Temperature for exact y: {t_exact_y:.4f}K")
    print(f"Difference: {abs(t_exact_x - t_exact_y):.4f}K")
    print()
    print("No single temperature produces the exact D65 chromaticity.")
    print("The official D65 point lies near, but not on, the daylight locus curve.")

    # =========================================================================
    print_section("6. PRACTICAL RECOMMENDATIONS")
    # =========================================================================

    _, _, err_its90_rec = compute_error(t_its90)
    _, _, err_codata_rec = compute_error(t_codata)
    _, _, err_6504 = compute_error(6504.0)
    _, _, err_opt = compute_error(t_optimal)

    print(f"{'Goal':<40} {'Approach':<38}")
    print("-" * 78)
    print(f"{'Exact D65 chromaticity':<40} {'Hardcode (0.31272, 0.32903)':<38}")
    print(f"{'D65 via polynomial (CIE 15:2004)':<40} {f'T = {t_its90:.2f}K ({err_its90_rec*1e5:.1f}×10⁻⁵)':<38}")
    print(f"{'D65 via polynomial (CODATA - wrong)':<40} {f'T = {t_codata:.2f}K ({err_codata_rec*1e5:.1f}×10⁻⁵)':<38}")
    print(f"{'D65 via polynomial (rounded)':<40} {f'T = 6504K ({err_6504*1e5:.1f}×10⁻⁵)':<38}")
    print(f"{'Minimum possible error':<40} {f'T = {t_optimal:.1f}K ({err_opt*1e5:.1f}×10⁻⁵)':<38}")
    print()
    print("For D50, D55, D65, D75: Use hardcoded official values.")
    print("Use the polynomial only for arbitrary D-illuminants.")

    # =========================================================================
    print_section("7. SUMMARY")
    # =========================================================================

    print("The D65 illuminant is defined by its spectral power distribution,")
    print("from which the chromaticity (0.31272, 0.32903) is derived.")
    print()
    print("The name 'D65' refers to its correlated color temperature on the")
    print("pre-1968 temperature scale. On the modern scale, this is ~6504K.")
    print()
    print("The CIE daylight locus polynomial expects ITS-90 CCT as input.")
    print(f"To recover D65, use T = {t_its90:.6f}K (per CIE 15:2004).")
    print(f"(NOT T = {t_codata:.6f}K from CODATA constants)")
    print()
    print("The ~9×10⁻⁵ residual error reflects:")
    print("  - Independent imprecision in the y(x) quadratic")
    print("  - Historical rounding in the original D65 computation")
    print("  - The polynomial being a general fit, not constrained to pass")
    print("    exactly through the canonical D-illuminants")
    print()
    print("All errors are in the 5th decimal place—far below perceptibility.")


def analyze_y_error_nature():
    """
    Determine if the y error is an absolute offset or a scale factor
    by examining multiple canonical D-illuminants.
    """
    print_section("8. Y ERROR ANALYSIS: OFFSET VS SCALE FACTOR")

    # Canonical D-illuminants with their official CIE chromaticity coordinates
    # and nominal temperatures (old K scale)
    d_illuminants = [
        ("D50", 5000, 0.34567, 0.35850),
        ("D55", 5500, 0.33242, 0.34743),
        ("D65", 6500, 0.31272, 0.32903),
        ("D75", 7500, 0.29902, 0.31485),
    ]

    print("Comparing official chromaticity vs daylight locus polynomial")
    print("for canonical D-illuminants:")
    print()

    print(f"{'Illum':<6} {'T_old':<7} {'T_new':<12} {'x_off':<10} {'x_poly':<10} {'y_off':<10} {'y_poly':<10}")
    print("-" * 75)

    results = []
    for name, t_old, x_official, y_official in d_illuminants:
        t_new = t_old * C2_RATIO
        x_poly, y_poly = daylight_locus_chromaticity(t_new)

        print(f"{name:<6} {t_old:<7} {t_new:<12.4f} {x_official:<10.5f} {x_poly:<10.5f} {y_official:<10.5f} {y_poly:<10.5f}")
        results.append((name, t_old, x_official, y_official, x_poly, y_poly))

    print()
    print("Error analysis:")
    print()
    print(f"{'Illum':<6} {'Δx (×10⁻⁵)':<14} {'Δy (×10⁻⁵)':<14} {'Δy/y_off (×10⁻⁵)':<18} {'y_poly/y_off':<14}")
    print("-" * 70)

    for name, t_old, x_off, y_off, x_poly, y_poly in results:
        dx = (x_poly - x_off) * 1e5
        dy = (y_poly - y_off) * 1e5
        dy_relative = ((y_poly - y_off) / y_off) * 1e5  # relative error in y
        y_ratio = y_poly / y_off

        print(f"{name:<6} {dx:>+13.2f} {dy:>+13.2f} {dy_relative:>+17.2f} {y_ratio:>13.8f}")

    print()

    # Also check: what if we find the exact-x temperature for each illuminant?
    print("Finding exact-x match temperature for each illuminant:")
    print()
    print(f"{'Illum':<6} {'T_exact_x':<14} {'Δy at exact x (×10⁻⁵)':<22} {'Δy/y (×10⁻⁵)':<16}")
    print("-" * 60)

    y_errors_absolute = []
    y_errors_relative = []

    for name, t_old, x_off, y_off, _, _ in results:
        # Find temperature where polynomial x matches official x
        a, b = t_old * 0.9, t_old * 1.1
        tol = 1e-12

        while abs(b - a) > tol:
            mid = (a + b) / 2
            x_mid, _ = daylight_locus_chromaticity(mid)
            if x_mid > x_off:
                a = mid
            else:
                b = mid

        t_exact_x = (a + b) / 2
        x_at_t, y_at_t = daylight_locus_chromaticity(t_exact_x)

        dy_abs = (y_at_t - y_off) * 1e5
        dy_rel = ((y_at_t - y_off) / y_off) * 1e5

        y_errors_absolute.append(dy_abs)
        y_errors_relative.append(dy_rel)

        print(f"{name:<6} {t_exact_x:<14.4f} {dy_abs:>+21.2f} {dy_rel:>+15.2f}")

    print()
    print("-" * 60)

    # Calculate statistics
    abs_mean = sum(y_errors_absolute) / len(y_errors_absolute)
    abs_std = (sum((e - abs_mean)**2 for e in y_errors_absolute) / len(y_errors_absolute)) ** 0.5

    rel_mean = sum(y_errors_relative) / len(y_errors_relative)
    rel_std = (sum((e - rel_mean)**2 for e in y_errors_relative) / len(y_errors_relative)) ** 0.5

    print()
    print("Statistical analysis:")
    print(f"  Absolute Δy errors: mean = {abs_mean:+.2f}×10⁻⁵, std = {abs_std:.2f}×10⁻⁵")
    print(f"  Relative Δy/y errors: mean = {rel_mean:+.2f}×10⁻⁵, std = {rel_std:.2f}×10⁻⁵")
    print()

    if abs_std < rel_std:
        print("CONCLUSION: The y error appears to be an ABSOLUTE OFFSET")
        print(f"  (constant ~{abs_mean:+.1f}×10⁻⁵ regardless of y value)")
    else:
        print("CONCLUSION: The y error appears to be a SCALE FACTOR")
        print(f"  (constant ~{rel_mean:+.1f}×10⁻⁵ relative to y value)")

    print()

    # Let's also check what the quadratic predicts vs actual
    print("Checking the y(x) quadratic directly:")
    print()
    print("The formula is: y = -3x² + 2.87x - 0.275")
    print()
    print(f"{'Illum':<6} {'x_official':<12} {'y_official':<12} {'y_from_formula':<14} {'Δy (×10⁻⁵)':<14}")
    print("-" * 60)

    for name, t_old, x_off, y_off, _, _ in results:
        y_formula = -3.0 * x_off**2 + 2.87 * x_off - 0.275
        dy = (y_formula - y_off) * 1e5
        print(f"{name:<6} {x_off:<12.5f} {y_off:<12.5f} {y_formula:<14.5f} {dy:>+13.2f}")


if __name__ == "__main__":
    main()
    analyze_y_error_nature()
