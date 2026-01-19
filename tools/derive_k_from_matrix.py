#!/usr/bin/env python3
"""
Derive Correlated Color Temperature from RGB→XYZ Matrix White Point

This tool derives the x chromaticity coordinate from an RGB→XYZ matrix and finds
the temperature K that, when plugged into the CIE daylight locus polynomial,
produces that x value.

Key insight: The sRGB matrix (IEC 61966-2-1) defines a D65 white point implicitly
through its row sums. This white point differs slightly from:
- CIE official D65 (0.31272, 0.32903) from SPD integration
- 4-digit rounded D65 (0.3127, 0.3290) in BT.709/sRGB specs

This tool derives the effective temperature for any RGB→XYZ matrix's white point.

References:
- IEC 61966-2-1:1999 (sRGB specification)
- CIE 15:2004 Equations 3.3-3.4 (daylight locus polynomial)
"""

from decimal import Decimal, getcontext
from typing import NamedTuple

# Set maximum precision for Decimal arithmetic
getcontext().prec = 50


# =============================================================================
# RGB→XYZ MATRICES
# =============================================================================

class Matrix3x3(NamedTuple):
    """3x3 matrix stored row-major."""
    r00: Decimal; r01: Decimal; r02: Decimal
    r10: Decimal; r11: Decimal; r12: Decimal
    r20: Decimal; r21: Decimal; r22: Decimal

    def column_sum(self, col: int) -> Decimal:
        """Sum of a column (XYZ of a primary or white point component)."""
        if col == 0:
            return self.r00 + self.r10 + self.r20
        elif col == 1:
            return self.r01 + self.r11 + self.r21
        else:
            return self.r02 + self.r12 + self.r22

    def row_sum(self, row: int) -> Decimal:
        """Sum of a row (XYZ component of white point)."""
        if row == 0:
            return self.r00 + self.r01 + self.r02
        elif row == 1:
            return self.r10 + self.r11 + self.r12
        else:
            return self.r20 + self.r21 + self.r22

    def white_xyz(self) -> tuple[Decimal, Decimal, Decimal]:
        """XYZ of white point (sum of each row = RGB(1,1,1) → XYZ)."""
        return self.row_sum(0), self.row_sum(1), self.row_sum(2)


# IEC 61966-2-1 sRGB matrix (authoritative, 4 decimal places)
# These values are considered EXACT per the sRGB specification
SRGB_MATRIX = Matrix3x3(
    Decimal("0.4124"), Decimal("0.3576"), Decimal("0.1805"),
    Decimal("0.2126"), Decimal("0.7152"), Decimal("0.0722"),
    Decimal("0.0193"), Decimal("0.1192"), Decimal("0.9505"),
)

# =============================================================================
# RGB→XYZ MATRIX DERIVATION FROM PRIMARIES
# =============================================================================

class Chromaticity(NamedTuple):
    """CIE xy chromaticity coordinates."""
    x: Decimal
    y: Decimal

    def to_xyz(self, Y: Decimal = Decimal("1")) -> tuple[Decimal, Decimal, Decimal]:
        """Convert to XYZ with given Y (luminance)."""
        X = (self.x / self.y) * Y
        Z = ((Decimal("1") - self.x - self.y) / self.y) * Y
        return X, Y, Z


# ITU-R BT.709 / sRGB primaries (exact chromaticity coordinates)
BT709_RED = Chromaticity(Decimal("0.64"), Decimal("0.33"))
BT709_GREEN = Chromaticity(Decimal("0.30"), Decimal("0.60"))
BT709_BLUE = Chromaticity(Decimal("0.15"), Decimal("0.06"))

# D65 white point variants
D65_CIE = Chromaticity(Decimal("0.31272"), Decimal("0.32903"))  # CIE 5-digit
D65_4DIGIT = Chromaticity(Decimal("0.3127"), Decimal("0.3290"))  # Spec rounded


def derive_rgb_to_xyz_matrix(
    red: Chromaticity,
    green: Chromaticity,
    blue: Chromaticity,
    white: Chromaticity,
) -> Matrix3x3:
    """
    Derive RGB→XYZ matrix from primary chromaticities and white point.

    This implements the standard matrix derivation:
    1. Convert primary xy to XYZ (with Y=1)
    2. Solve for RGB weights that sum to white point XYZ
    3. Scale each primary's XYZ by its weight

    Returns the 3x3 matrix where each column is a scaled primary XYZ.
    """
    # Primary XYZ values (with Y=1 each)
    Xr, Yr, Zr = red.to_xyz()
    Xg, Yg, Zg = green.to_xyz()
    Xb, Yb, Zb = blue.to_xyz()

    # White point XYZ (with Y=1)
    Xw, Yw, Zw = white.to_xyz()

    # Build the primary matrix (columns are XYZ of each primary with Y=1)
    # We need to solve: [Xr Xg Xb] [Sr]   [Xw]
    #                   [Yr Yg Yb] [Sg] = [Yw]
    #                   [Zr Zg Zb] [Sb]   [Zw]
    #
    # Where Sr, Sg, Sb are the scaling factors for each primary.
    # This is a 3x3 linear system. We'll use Cramer's rule.

    # Determinant of the primary matrix
    det = (Xr * (Yg * Zb - Zg * Yb) -
           Xg * (Yr * Zb - Zr * Yb) +
           Xb * (Yr * Zg - Zr * Yg))

    if det == 0:
        raise ValueError("Degenerate primary matrix (determinant is zero)")

    # Solve for Sr (replace first column with white)
    det_Sr = (Xw * (Yg * Zb - Zg * Yb) -
              Xg * (Yw * Zb - Zw * Yb) +
              Xb * (Yw * Zg - Zw * Yg))
    Sr = det_Sr / det

    # Solve for Sg (replace second column with white)
    det_Sg = (Xr * (Yw * Zb - Zw * Yb) -
              Xw * (Yr * Zb - Zr * Yb) +
              Xb * (Yr * Zw - Zr * Yw))
    Sg = det_Sg / det

    # Solve for Sb (replace third column with white)
    det_Sb = (Xr * (Yg * Zw - Zg * Yw) -
              Xg * (Yr * Zw - Zr * Yw) +
              Xw * (Yr * Zg - Zr * Yg))
    Sb = det_Sb / det

    # Build the final matrix (rows are XYZ, columns are RGB)
    # Matrix[row][col] where row is X/Y/Z and col is R/G/B
    return Matrix3x3(
        Sr * Xr, Sg * Xg, Sb * Xb,  # X row
        Sr * Yr, Sg * Yg, Sb * Yb,  # Y row
        Sr * Zr, Sg * Zg, Sb * Zb,  # Z row
    )


# Derived matrices from first principles
SRGB_MATRIX_CIE_D65 = derive_rgb_to_xyz_matrix(
    BT709_RED, BT709_GREEN, BT709_BLUE, D65_CIE
)
SRGB_MATRIX_4DIGIT_D65 = derive_rgb_to_xyz_matrix(
    BT709_RED, BT709_GREEN, BT709_BLUE, D65_4DIGIT
)


# =============================================================================
# DAYLIGHT LOCUS POLYNOMIAL (CIE 15:2004)
# =============================================================================

def daylight_locus_x(temp_k: Decimal) -> Decimal:
    """
    Compute x chromaticity from temperature using CIE daylight locus.

    CIE 15:2004 Equations 3.3 and 3.4.
    Input is ITS-90 correlated color temperature.

    For 4000K ≤ T ≤ 7000K:
        x_D = -4.6070×10⁹/T³ + 2.9678×10⁶/T² + 0.09911×10³/T + 0.244063

    For 7000K < T ≤ 25000K:
        x_D = -2.0064×10⁹/T³ + 1.9018×10⁶/T² + 0.24748×10³/T + 0.237040
    """
    if temp_k < Decimal("4000") or temp_k > Decimal("25000"):
        raise ValueError(f"Temperature {temp_k}K out of range [4000, 25000]")

    t2 = temp_k ** 2
    t3 = temp_k ** 3

    if temp_k <= Decimal("7000"):
        x = (Decimal("-4.6070e9") / t3 +
             Decimal("2.9678e6") / t2 +
             Decimal("0.09911e3") / temp_k +
             Decimal("0.244063"))
    else:
        x = (Decimal("-2.0064e9") / t3 +
             Decimal("1.9018e6") / t2 +
             Decimal("0.24748e3") / temp_k +
             Decimal("0.237040"))

    return x


def daylight_locus_y(x: Decimal) -> Decimal:
    """
    Compute y from x using the daylight locus quadratic.

    CIE 15:2004 Equation 3.2:
        y_D = -3.000x_D² + 2.870x_D - 0.275
    """
    return Decimal("-3.000") * x ** 2 + Decimal("2.870") * x - Decimal("0.275")


def daylight_locus_chromaticity(temp_k: Decimal) -> tuple[Decimal, Decimal]:
    """Compute (x, y) chromaticity from temperature."""
    x = daylight_locus_x(temp_k)
    y = daylight_locus_y(x)
    return x, y


# =============================================================================
# TEMPERATURE SEARCH
# =============================================================================

def find_temperature_for_x(target_x: Decimal, tol: Decimal = Decimal("1e-12")) -> Decimal:
    """
    Find the temperature that produces a given x chromaticity.

    Uses binary search since x decreases monotonically with temperature
    in the D65 region (around 6500K).

    Returns temperature in Kelvin (ITS-90 scale).
    """
    # x decreases as T increases in the 4000-7000K range
    low = Decimal("4000")
    high = Decimal("7000")

    # Verify target is in achievable range
    x_at_low = daylight_locus_x(low)
    x_at_high = daylight_locus_x(high)

    if target_x > x_at_low or target_x < x_at_high:
        # Try extended range
        high = Decimal("25000")
        x_at_high = daylight_locus_x(high)

        if target_x > x_at_low or target_x < x_at_high:
            raise ValueError(
                f"Target x={target_x} outside achievable range "
                f"[{x_at_high}, {x_at_low}]"
            )

    # Binary search
    while (high - low) > tol:
        mid = (low + high) / 2
        x_mid = daylight_locus_x(mid)

        if x_mid > target_x:
            # x too high, need higher temperature
            low = mid
        else:
            # x too low, need lower temperature
            high = mid

    return (low + high) / 2


# =============================================================================
# TEMPERATURE SCALE CONSTANTS
# =============================================================================

# Second radiation constant (c₂) values
C2_1931 = Decimal("0.01438")      # m·K (1931 definition, used for original D65)
C2_ITS90 = Decimal("0.014388")    # m·K (ITS-90, CIE 15:2004 standard)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def derive_white_point_from_matrix(matrix: Matrix3x3, name: str) -> dict:
    """
    Derive white point chromaticity from an RGB→XYZ matrix.

    White point is the XYZ value when R=G=B=1.
    """
    X, Y, Z = matrix.white_xyz()
    total = X + Y + Z
    x = X / total
    y = Y / total

    return {
        "name": name,
        "X": X,
        "Y": Y,
        "Z": Z,
        "x": x,
        "y": y,
    }


def analyze_matrix(matrix: Matrix3x3, name: str):
    """Complete analysis of a matrix's white point."""
    result = derive_white_point_from_matrix(matrix, name)

    X, Y, Z = result["X"], result["Y"], result["Z"]
    x, y = result["x"], result["y"]

    print(f"\n  Matrix white point (XYZ of R=G=B=1):")
    print(f"    X = {float(X):.10f}")
    print(f"    Y = {float(Y):.10f}")
    print(f"    Z = {float(Z):.10f}")

    print(f"\n  Chromaticity coordinates:")
    print(f"    x = {float(x):.16f}")
    print(f"    y = {float(y):.16f}")

    # Find matching temperature
    temp_its90 = find_temperature_for_x(x)

    # Also compute what the y would be from the polynomial
    y_from_poly = daylight_locus_y(x)
    y_error = y - y_from_poly

    print(f"\n  Temperature analysis:")
    print(f"    x-matching temperature (ITS-90): {float(temp_its90):.10f} K")

    # Convert back to 1931 scale
    temp_1931 = temp_its90 * (C2_1931 / C2_ITS90)
    print(f"    x-matching temperature (1931):   {float(temp_1931):.10f} K")

    print(f"\n  y-coordinate analysis:")
    print(f"    y from matrix:    {float(y):.16f}")
    print(f"    y from polynomial at x: {float(y_from_poly):.16f}")
    print(f"    Δy (matrix - poly):     {float(y_error) * 1e5:+.6f} ×10⁻⁵")

    # Verify by computing x at the found temperature
    x_verify = daylight_locus_x(temp_its90)
    print(f"\n  Verification:")
    print(f"    x from matrix:              {float(x):.16f}")
    print(f"    x from polynomial at T:     {float(x_verify):.16f}")
    print(f"    Δx (should be ~0):          {float(x - x_verify) * 1e12:.3f} ×10⁻¹²")

    return {
        **result,
        "temp_its90": temp_its90,
        "temp_1931": temp_1931,
        "y_from_poly": y_from_poly,
        "y_error": y_error,
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_section("DERIVE K (TEMPERATURE) FROM RGB→XYZ MATRIX WHITE POINT")

    print("""
This tool derives the correlated color temperature (K) that corresponds to
the white point implied by an RGB→XYZ transformation matrix.

The sRGB specification (IEC 61966-2-1) defines a 3x3 matrix that transforms
linear RGB to CIE XYZ. The white point is implicit: it's the XYZ value when
R=G=B=1, which equals the sum of each row of the matrix.

This white point differs slightly from:
- CIE official D65: (0.31272, 0.32903) from SPD integration
- 4-digit D65: (0.3127, 0.3290) quoted in BT.709/sRGB specifications
""")

    # =========================================================================
    print_section("1. STANDARD sRGB MATRIX (IEC 61966-2-1)")
    # =========================================================================

    print("""
The IEC 61966-2-1 sRGB specification defines this RGB→XYZ matrix:

    | X |   | 0.4124  0.3576  0.1805 |   | R |
    | Y | = | 0.2126  0.7152  0.0722 | × | G |
    | Z |   | 0.0193  0.1192  0.9505 |   | B |

These values are considered EXACT per the specification.
""")

    result_srgb = analyze_matrix(SRGB_MATRIX, "sRGB (IEC 61966-2-1)")

    # =========================================================================
    print_section("2. DERIVED MATRIX (BT.709 PRIMARIES + CIE D65)")
    # =========================================================================

    print("""
Matrix derived from first principles using:
- ITU-R BT.709 primaries: R(0.64, 0.33), G(0.30, 0.60), B(0.15, 0.06)
- CIE D65 white point: (0.31272, 0.32903)

This should recover the CIE D65 chromaticity exactly (by construction).
""")

    result_cie = analyze_matrix(SRGB_MATRIX_CIE_D65, "BT.709 + CIE D65")

    # =========================================================================
    print_section("3. DERIVED MATRIX (BT.709 PRIMARIES + 4-DIGIT D65)")
    # =========================================================================

    print("""
Matrix derived from first principles using:
- ITU-R BT.709 primaries: R(0.64, 0.33), G(0.30, 0.60), B(0.15, 0.06)
- 4-digit D65 white point: (0.3127, 0.3290)

This is what many specifications effectively define.
""")

    result_4digit = analyze_matrix(SRGB_MATRIX_4DIGIT_D65, "BT.709 + 4-digit D65")

    # =========================================================================
    print_section("4. COMPARISON SUMMARY")
    # =========================================================================

    # Reference D65 values
    cie_d65_x = Decimal("0.31272")
    cie_d65_y = Decimal("0.32903")
    d65_4digit_x = Decimal("0.3127")
    d65_4digit_y = Decimal("0.3290")

    # Find temperatures for reference values
    temp_cie_ref = find_temperature_for_x(cie_d65_x)
    temp_4digit_ref = find_temperature_for_x(d65_4digit_x)

    print(f"\n  {'Source':<40} {'x':<20} {'y':<20} {'T (ITS-90) K':<16}")
    print("  " + "-" * 96)

    print(f"  {'CIE D65 reference':<40} "
          f"{float(cie_d65_x):<20.16f} {float(cie_d65_y):<20.16f} "
          f"{float(temp_cie_ref):<16.10f}")

    print(f"  {'4-digit D65 reference':<40} "
          f"{float(d65_4digit_x):<20.16f} {float(d65_4digit_y):<20.16f} "
          f"{float(temp_4digit_ref):<16.10f}")

    print(f"  {'sRGB matrix (IEC 61966-2-1)':<40} "
          f"{float(result_srgb['x']):<20.16f} {float(result_srgb['y']):<20.16f} "
          f"{float(result_srgb['temp_its90']):<16.10f}")

    print(f"  {'Derived: BT.709 + CIE D65':<40} "
          f"{float(result_cie['x']):<20.16f} {float(result_cie['y']):<20.16f} "
          f"{float(result_cie['temp_its90']):<16.10f}")

    print(f"  {'Derived: BT.709 + 4-digit D65':<40} "
          f"{float(result_4digit['x']):<20.16f} {float(result_4digit['y']):<20.16f} "
          f"{float(result_4digit['temp_its90']):<16.10f}")

    # =========================================================================
    print_section("5. TEMPERATURE SCALE SUMMARY")
    # =========================================================================

    print("""
  Temperature Scale Conversion:

    T_ITS90 = T_1931 × (c₂_ITS90 / c₂_1931)
            = T_1931 × (0.014388 / 0.01438)
            = T_1931 × 1.00055632823...

  Standard D65 Reference Points:
""")

    # Standard D65: 6500K on 1931 scale = 6503.62K on ITS-90
    d65_1931 = Decimal("6500")
    d65_its90 = d65_1931 * (C2_ITS90 / C2_1931)

    print(f"    D65 nominal (1931 scale):   6500.0000000000 K")
    print(f"    D65 ITS-90 equivalent:      {float(d65_its90):.10f} K")
    print()

    print("  Matrix-implied temperatures (ITS-90):")
    print(f"    sRGB standard matrix:       {float(result_srgb['temp_its90']):.10f} K")
    print(f"    BT.709 + CIE D65:           {float(result_cie['temp_its90']):.10f} K")
    print(f"    BT.709 + 4-digit D65:       {float(result_4digit['temp_its90']):.10f} K")
    print()

    print("  Matrix-implied temperatures (1931 scale):")
    print(f"    sRGB standard matrix:       {float(result_srgb['temp_1931']):.10f} K")
    print(f"    BT.709 + CIE D65:           {float(result_cie['temp_1931']):.10f} K")
    print(f"    BT.709 + 4-digit D65:       {float(result_4digit['temp_1931']):.10f} K")

    # =========================================================================
    print_section("6. KEY FINDINGS")
    # =========================================================================

    srgb_temp_1931 = result_srgb['temp_1931']
    d65_diff = srgb_temp_1931 - d65_1931

    print(f"""
  The sRGB matrix (IEC 61966-2-1) implies a white point temperature of:

    {float(result_srgb['temp_its90']):.10f} K (ITS-90 scale)
    {float(result_srgb['temp_1931']):.10f} K (1931 scale)

  This differs from nominal D65 (6500K 1931 scale) by:
    {float(d65_diff):.10f} K ({float(d65_diff):+.6f} K)

  The x-coordinate difference from CIE D65 (0.31272):
    sRGB matrix x = {float(result_srgb['x']):.16f}
    CIE D65 x     = 0.31272
    Δx            = {float(result_srgb['x'] - cie_d65_x) * 1e5:+.6f} ×10⁻⁵

  The y-coordinate difference from CIE D65 (0.32903):
    sRGB matrix y = {float(result_srgb['y']):.16f}
    CIE D65 y     = 0.32903
    Δy            = {float(result_srgb['y'] - cie_d65_y) * 1e5:+.6f} ×10⁻⁵
""")

    # =========================================================================
    print_section("7. PRACTICAL IMPLICATIONS")
    # =========================================================================

    print("""
  For image processing workflows:

  1. The sRGB specification defines its matrix as EXACT, meaning the
     matrix-implied white point IS the sRGB white point by definition.

  2. This white point (x=0.3127159..., y=0.3290015...) differs from CIE D65
     (0.31272, 0.32903) in the 5th decimal place - imperceptible.

  3. When converting between sRGB and other D65 color spaces, the choice of
     which D65 to use affects precision but not perception:

     - For maximum matrix accuracy: use the sRGB matrix-implied values
     - For CIE compliance: use CIE D65 (0.31272, 0.32903)
     - For spec compliance: use 4-digit values (0.3127, 0.3290)

  4. The corresponding temperature values serve as a reference for software
     that uses temperature-based D65 derivation (e.g., via polynomial).
""")


if __name__ == "__main__":
    main()
