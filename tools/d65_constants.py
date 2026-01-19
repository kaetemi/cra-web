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

# Second radiation constant (Planck's law)
C2_OLD = 0.01438        # m·K (pre-1968)
C2_NEW = 0.01438776877  # m·K (post-1968, CODATA)

# Derived values
C2_RATIO = C2_NEW / C2_OLD  # ≈ 1.00054025


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
    print(f"  Old c₂ (pre-1968):  {C2_OLD} m·K")
    print(f"  New c₂ (CODATA):    {C2_NEW} m·K")
    print(f"  Ratio (new/old):    {C2_RATIO:.11f}")
    print()

    print("Temperature Scale Conversion:")
    print(f"  T_new = T_old × {C2_RATIO:.11f}")
    print(f"  6500K (old) → {6500 * C2_RATIO:.10f}K (new)")

    # =========================================================================
    print_section("2. KEY TEMPERATURES")
    # =========================================================================

    # Calculate all key temperatures
    t_nominal = 6500.0
    t_c2_ratio = 6500.0 * C2_RATIO
    t_lcms = 6504.0
    t_exact_x = find_exact_x_temperature()
    t_exact_y = find_exact_y_temperature()
    t_optimal = find_optimal_temperature()

    print(f"{'Description':<35} {'Temperature (K)':<20}")
    print("-" * 55)
    print(f"{'Nominal (historical, old K)':<35} {t_nominal:<20.10f}")
    print(f"{'c₂ ratio conversion':<35} {t_c2_ratio:<20.10f}")
    print(f"{'lcms/moxcms value':<35} {t_lcms:<20.10f}")
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
        ("6503.5116K (c₂ ratio)", t_c2_ratio),
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

    print("5.1 The c₂ Ratio Correction")
    print("-" * 40)
    dx, dy, _ = compute_error(t_c2_ratio)
    print(f"At T = 6500 × (new_c₂/old_c₂) = {t_c2_ratio:.4f}K:")
    print(f"  x error: {dx*1e5:+.4f}×10⁻⁵  (nearly perfect!)")
    print(f"  y error: {dy*1e5:+.4f}×10⁻⁵  (independent error in y(x) formula)")
    print()
    print("This confirms the x(T) polynomial was calibrated with old c₂,")
    print("and expects modern CCT as input.")

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

    _, _, err_6504 = compute_error(6504.0)
    _, _, err_c2 = compute_error(t_c2_ratio)
    _, _, err_opt = compute_error(t_optimal)

    print(f"{'Goal':<35} {'Approach':<43}")
    print("-" * 78)
    print(f"{'Exact D65 chromaticity':<35} {'Hardcode (0.31272, 0.32903)':<43}")
    print(f"{'D65 via polynomial (standard)':<35} {f'Use T = 6504K ({err_6504*1e5:.1f}×10⁻⁵ error)':<43}")
    print(f"{'D65 via polynomial (precise)':<35} {f'Use T = 6503.5K ({err_c2*1e5:.1f}×10⁻⁵ error)':<43}")
    print(f"{'Minimum possible error':<35} {f'Use T = {t_optimal:.1f}K ({err_opt*1e5:.1f}×10⁻⁵ error)':<43}")
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
    print("The CIE daylight locus polynomial expects modern CCT as input.")
    print("To recover D65, use T ≈ 6504K (or precisely 6503.5116K).")
    print()
    print("The ~9×10⁻⁵ residual error reflects:")
    print("  - Independent imprecision in the y(x) quadratic")
    print("  - Historical rounding in the original D65 computation")
    print("  - The polynomial being a general fit, not constrained to pass")
    print("    exactly through the canonical D-illuminants")
    print()
    print("All errors are in the 5th decimal place—far below perceptibility.")


if __name__ == "__main__":
    main()
