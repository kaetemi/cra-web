#!/usr/bin/env python3
"""
HIGH-PRECISION D65 CHROMATICITY DERIVATION

Derives D65 chromaticity from authoritative source data:
- D65 SPD: ISO 11664-2:2007 / CIE S 014-2:2006 Table 1
- CMF: CIE 018:2019 Table 6 / CIE 1931 2° observer

Computes using three configurations:
1. Full range at 1nm (360-830nm) - maximum precision
2. Standard visible at 1nm (380-780nm) - common implementation
3. Standard visible at 5nm (380-780nm) - matches CIE 15:2004 Table T.3

Data files required:
- cie_standard_illuminants_a_d65_spd.csv (with header: wavelength_nm,SA,SD65)
- CIE_xyz_1931_2deg.csv (no header: wavelength,x_bar,y_bar,z_bar)

References:
- ISO 11664-2:2007(E) / CIE S 014-2/E:2006, Colorimetry Part 2: CIE Standard Illuminants
- CIE 015:2004, Colorimetry, 3rd Edition (Table T.3 defines official chromaticity)
- CIE 018:2019, The Basis of Physical Photometry, 3rd Edition
"""

import csv
from decimal import Decimal, getcontext
from pathlib import Path
from typing import NamedTuple

# Set maximum precision for Decimal arithmetic
getcontext().prec = 50


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CMFValues(NamedTuple):
    x_bar: Decimal
    y_bar: Decimal
    z_bar: Decimal


# =============================================================================
# CSV LOADING FUNCTIONS
# =============================================================================

def load_d65_spd(filepath: str | Path) -> dict[int, Decimal]:
    """
    Load D65 SPD from CSV file.
    
    Expected format (with header):
        wavelength_nm,SA,SD65
        300,0.930483,0.0341000
        301,0.967643,0.360140
        ...
    """
    spd: dict[int, Decimal] = {}
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wavelength = int(row['wavelength_nm'])
            power = Decimal(row['SD65'])
            spd[wavelength] = power
    
    return spd


def load_cmf(filepath: str | Path) -> dict[int, CMFValues]:
    """
    Load CIE 1931 2° observer CMF from CSV file.
    
    Expected format (no header):
        360,0.000129900000,0.0000039170000,0.000606100000
        361,0.000145847000,0.0000043935810,0.000680879200
        ...
    """
    cmf: dict[int, CMFValues] = {}
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            wavelength = int(row[0])
            x_bar = Decimal(row[1])
            y_bar = Decimal(row[2])
            z_bar = Decimal(row[3])
            cmf[wavelength] = CMFValues(x_bar, y_bar, z_bar)
    
    return cmf


# =============================================================================
# DATA FILTERING FUNCTIONS
# =============================================================================

def filter_to_range(
    data: dict[int, any],
    start_nm: int,
    end_nm: int,
    step_nm: int = 1
) -> dict[int, any]:
    """
    Filter data to specified wavelength range and interval.
    
    Args:
        data: Dictionary with wavelength keys
        start_nm: Start wavelength (inclusive)
        end_nm: End wavelength (inclusive)
        step_nm: Wavelength step (1 for all, 5 for 5nm intervals)
    
    Returns:
        Filtered dictionary
    """
    return {
        k: v for k, v in data.items()
        if start_nm <= k <= end_nm and (k - start_nm) % step_nm == 0
    }


# =============================================================================
# CONSTANTS
# =============================================================================

# Official CIE D65 chromaticity from CIE 15:2004 Table T.3
CIE_D65_X = Decimal("0.31272")
CIE_D65_Y = Decimal("0.32903")

# Official tristimulus values from CIE 15:2004 Table T.3
CIE_D65_X_TRIST = Decimal("95.04")
CIE_D65_Y_TRIST = Decimal("100.00")
CIE_D65_Z_TRIST = Decimal("108.88")

# Second radiation constants (for temperature analysis)
C2_OLD = Decimal("0.01438")      # m·K (1931)
C2_ITS90 = Decimal("0.014388")   # m·K (ITS-90)


# =============================================================================
# HIGH-PRECISION DERIVATION FUNCTIONS
# =============================================================================

def derive_chromaticity_from_spd(
    spd: dict[int, Decimal],
    cmf: dict[int, CMFValues],
    wavelength_step: Decimal = Decimal("1")
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    """
    Derive chromaticity coordinates from SPD and CMFs using Decimal arithmetic.
    
    Implements CIE 15:2004 equations:
        X = k · Σ S(λ)x̄(λ)Δλ
        Y = k · Σ S(λ)ȳ(λ)Δλ  
        Z = k · Σ S(λ)z̄(λ)Δλ
        
    Where k normalizes Y to 100 for the perfect reflecting diffuser.
    """
    # Find common wavelength range
    spd_wavelengths = set(spd.keys())
    cmf_wavelengths = set(cmf.keys())
    common = sorted(spd_wavelengths & cmf_wavelengths)
    
    if not common:
        raise ValueError("No overlapping wavelengths between SPD and CMF")
    
    # Integrate using rectangular rule
    sum_X = Decimal("0")
    sum_Y = Decimal("0")
    sum_Z = Decimal("0")
    
    for wl in common:
        s = spd[wl]
        x_bar, y_bar, z_bar = cmf[wl]
        
        sum_X += s * x_bar * wavelength_step
        sum_Y += s * y_bar * wavelength_step
        sum_Z += s * z_bar * wavelength_step
    
    # Normalize so Y = 100
    k = Decimal("100") / sum_Y if sum_Y != 0 else Decimal("1")
    
    X = k * sum_X
    Y = k * sum_Y
    Z = k * sum_Z
    
    # Chromaticity coordinates
    total = X + Y + Z
    x = X / total if total != 0 else Decimal("0")
    y = Y / total if total != 0 else Decimal("0")
    
    return X, Y, Z, x, y


def daylight_locus_chromaticity(temp_k: Decimal) -> tuple[Decimal, Decimal]:
    """
    Calculate chromaticity on CIE daylight locus.
    CIE 15:2004 Equations 3.3, 3.4, and 3.2.
    Expects ITS-90 CCT as input.
    """
    if temp_k < Decimal("4000") or temp_k > Decimal("25000"):
        raise ValueError(f"Temperature {temp_k}K out of range [4000, 25000]")
    
    if Decimal("4000") <= temp_k <= Decimal("7000"):
        x = (Decimal("-4.6070e9") / (temp_k ** 3) +
             Decimal("2.9678e6") / (temp_k ** 2) +
             Decimal("0.09911e3") / temp_k +
             Decimal("0.244063"))
    else:
        x = (Decimal("-2.0064e9") / (temp_k ** 3) +
             Decimal("1.9018e6") / (temp_k ** 2) +
             Decimal("0.24748e3") / temp_k +
             Decimal("0.237040"))
    
    y = Decimal("-3.000") * x ** 2 + Decimal("2.870") * x - Decimal("0.275")
    
    return x, y


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def print_result(
    label: str,
    X: Decimal, Y: Decimal, Z: Decimal,
    x: Decimal, y: Decimal,
    show_comparison: bool = True
):
    """Print tristimulus and chromaticity results."""
    print(f"{label}")
    print()
    print("  Tristimulus values:")
    print(f"    X = {float(X):.4f}  (official: {CIE_D65_X_TRIST})")
    print(f"    Y = {float(Y):.4f}  (official: {CIE_D65_Y_TRIST})")
    print(f"    Z = {float(Z):.4f}  (official: {CIE_D65_Z_TRIST})")
    print()
    print("  Chromaticity coordinates:")
    print(f"    x = {float(x):.10f}")
    print(f"    y = {float(y):.10f}")
    
    if show_comparison:
        delta_x = (x - CIE_D65_X) * Decimal("1e5")
        delta_y = (y - CIE_D65_Y) * Decimal("1e5")
        print()
        print(f"  Difference from official (×10⁻⁵):")
        print(f"    Δx = {float(delta_x):+.4f}")
        print(f"    Δy = {float(delta_y):+.4f}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    # File paths
    spd_file = Path("cie_standard_illuminants_a_d65_spd.csv")
    cmf_file = Path("CIE_xyz_1931_2deg.csv")
    
    print_section("HIGH-PRECISION D65 CHROMATICITY DERIVATION")
    print("Comparing different wavelength ranges and intervals")
    print()
    print("Data Sources:")
    print(f"  - D65 SPD: {spd_file}")
    print(f"  - CMF:     {cmf_file}")
    print()
    print("Official values (CIE 15:2004 Table T.3):")
    print(f"  X = {CIE_D65_X_TRIST}, Y = {CIE_D65_Y_TRIST}, Z = {CIE_D65_Z_TRIST}")
    print(f"  x = {CIE_D65_X}, y = {CIE_D65_Y}")
    print()
    print("  Note: Table T.3 specifies these were computed using")
    print("  '5 nm intervals over the range 380 nm to 780 nm'")
    
    # Load data
    print()
    print("Loading data...")
    
    try:
        d65_spd = load_d65_spd(spd_file)
        print(f"  D65 SPD: {len(d65_spd)} points "
              f"({min(d65_spd.keys())}-{max(d65_spd.keys())}nm)")
    except FileNotFoundError:
        print(f"  ERROR: Could not find {spd_file}")
        return
    
    try:
        cmf = load_cmf(cmf_file)
        print(f"  CMF:     {len(cmf)} points "
              f"({min(cmf.keys())}-{max(cmf.keys())}nm)")
    except FileNotFoundError:
        print(f"  ERROR: Could not find {cmf_file}")
        return
    
    # =========================================================================
    print_section("1. OFFICIAL METHOD: 380-780nm @ 5nm (CIE 15:2004 Table T.3)")
    # =========================================================================
    
    spd_5nm = filter_to_range(d65_spd, 380, 780, 5)
    cmf_5nm = filter_to_range(cmf, 380, 780, 5)
    
    print(f"Data points used: {len(spd_5nm)} SPD, {len(cmf_5nm)} CMF")
    print(f"Wavelengths: 380, 385, 390, ... 780nm")
    print()
    
    X1, Y1, Z1, x1, y1 = derive_chromaticity_from_spd(spd_5nm, cmf_5nm, Decimal("5"))
    print_result("Results (5nm intervals, 380-780nm):", X1, Y1, Z1, x1, y1)
    
    # =========================================================================
    print_section("2. EXTENDED PRECISION: 380-780nm @ 1nm")
    # =========================================================================
    
    spd_1nm_visible = filter_to_range(d65_spd, 380, 780, 1)
    cmf_1nm_visible = filter_to_range(cmf, 380, 780, 1)
    
    print(f"Data points used: {len(spd_1nm_visible)} SPD, {len(cmf_1nm_visible)} CMF")
    print()
    
    X2, Y2, Z2, x2, y2 = derive_chromaticity_from_spd(
        spd_1nm_visible, cmf_1nm_visible, Decimal("1")
    )
    print_result("Results (1nm intervals, 380-780nm):", X2, Y2, Z2, x2, y2)
    
    # =========================================================================
    print_section("3. FULL RANGE: 360-830nm @ 1nm")
    # =========================================================================
    
    common_range = sorted(set(d65_spd.keys()) & set(cmf.keys()))
    print(f"Data points used: {len(common_range)} (overlap of SPD and CMF)")
    print(f"Wavelength range: {min(common_range)}-{max(common_range)}nm")
    print()
    
    X3, Y3, Z3, x3, y3 = derive_chromaticity_from_spd(d65_spd, cmf, Decimal("1"))
    print_result("Results (1nm intervals, full range):", X3, Y3, Z3, x3, y3)
    
    # =========================================================================
    print_section("4. COMPARISON SUMMARY")
    # =========================================================================
    
    print(f"{'Configuration':<35} {'x':<16} {'y':<16} {'Δx×10⁻⁵':<10} {'Δy×10⁻⁵':<10}")
    print("-" * 87)
    print(f"{'Official CIE 15:2004':<35} {float(CIE_D65_X):<16.5f} {float(CIE_D65_Y):<16.5f} {'—':^10} {'—':^10}")
    
    configs = [
        ("380-780nm @ 5nm (Table T.3)", x1, y1),
        ("380-780nm @ 1nm", x2, y2),
        ("360-830nm @ 1nm (full)", x3, y3),
    ]
    
    for label, x, y in configs:
        dx = float((x - CIE_D65_X) * Decimal("1e5"))
        dy = float((y - CIE_D65_Y) * Decimal("1e5"))
        print(f"{label:<35} {float(x):<16.10f} {float(y):<16.10f} {dx:+10.4f} {dy:+10.4f}")
    
    print()
    print("Tristimulus comparison:")
    print()
    print(f"{'Configuration':<35} {'X':<12} {'Y':<12} {'Z':<12}")
    print("-" * 71)
    print(f"{'Official CIE 15:2004':<35} {float(CIE_D65_X_TRIST):<12.2f} {float(CIE_D65_Y_TRIST):<12.2f} {float(CIE_D65_Z_TRIST):<12.2f}")
    print(f"{'380-780nm @ 5nm (Table T.3)':<35} {float(X1):<12.4f} {float(Y1):<12.4f} {float(Z1):<12.4f}")
    print(f"{'380-780nm @ 1nm':<35} {float(X2):<12.4f} {float(Y2):<12.4f} {float(Z2):<12.4f}")
    print(f"{'360-830nm @ 1nm (full)':<35} {float(X3):<12.4f} {float(Y3):<12.4f} {float(Z3):<12.4f}")
    
    # =========================================================================
    print_section("5. TEMPERATURE SCALE ANALYSIS")
    # =========================================================================
    
    t_1931 = Decimal("6500")
    t_its90 = t_1931 * (C2_ITS90 / C2_OLD)
    
    print(f"D65 nominal temperature: {t_1931}K (1931 scale)")
    print(f"Converted to ITS-90:     {t_its90}K")
    print()
    
    temps = [
        (t_1931, "6500K (nominal)"),
        (t_its90, "6503.62K (ITS-90)"),
        (Decimal("6504"), "6504K (rounded)"),
    ]
    
    print(f"{'Temperature':<25} {'x':<18} {'y':<18} {'Δx×10⁻⁵':<10} {'Δy×10⁻⁵':<10}")
    print("-" * 81)
    
    for temp, label in temps:
        xp, yp = daylight_locus_chromaticity(temp)
        dx = (xp - CIE_D65_X) * Decimal("1e5")
        dy = (yp - CIE_D65_Y) * Decimal("1e5")
        print(f"{label:<25} {float(xp):<18.12f} {float(yp):<18.12f} {float(dx):+10.2f} {float(dy):+10.2f}")
    
    # =========================================================================
    print_section("6. DATA VERIFICATION")
    # =========================================================================
    
    print("Spot checks:")
    print()
    
    if 560 in d65_spd:
        print(f"  D65 SPD at 560nm: {d65_spd[560]} (should be 100.000)")
    if 450 in d65_spd:
        print(f"  D65 SPD at 450nm: {d65_spd[450]} (table: 117.008)")
    if 555 in cmf:
        print(f"  CMF ȳ at 555nm:   {cmf[555].y_bar} (should be 1.0)")
    
    print()
    print("5nm sample wavelengths in 380-780nm range:")
    sample_5nm = [380, 400, 500, 555, 600, 700, 780]
    for wl in sample_5nm:
        if wl in spd_5nm and wl in cmf_5nm:
            print(f"  {wl}nm: SPD={spd_5nm[wl]}, x̄={cmf_5nm[wl].x_bar:.6f}")


if __name__ == "__main__":
    main()
