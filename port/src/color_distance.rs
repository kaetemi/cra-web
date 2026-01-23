/// Color distance functions for perceptual color spaces.
///
/// Provides various distance metrics for CIELAB color space:
/// - CIE76: Simple Euclidean distance
/// - CIE94: Weighted for perceptual uniformity
/// - CIEDE2000: Most accurate perceptual distance
///
/// Also provides helper functions for determining which color space is in use.

use crate::colorspace_derived::f32 as cs;
use crate::dither::common::PerceptualSpace;

// ============================================================================
// CIELAB Distance Formulas
// ============================================================================

/// CIE76 (ΔE*ab): Simple Euclidean distance squared in L*a*b* space
/// This is the original CIELAB distance formula.
#[inline]
pub fn lab_distance_cie76_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let dl = l1 - l2;
    let da = a1 - a2;
    let db = b1 - b2;
    dl * dl + da * da + db * db
}

/// CIE94 (ΔE*94): Weighted distance squared accounting for perceptual non-uniformity
/// Uses graphic arts constants (kL=1, K1=0.045, K2=0.015)
#[inline]
pub fn lab_distance_cie94_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let dl = l1 - l2;
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let dc = c1 - c2;

    // ΔH² = Δa² + Δb² - ΔC²
    let da = a1 - a2;
    let db = b1 - b2;
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);

    // Weighting factors (graphic arts): SC = 1 + K1*C, SH = 1 + K2*C
    let sc = 1.0 + cs::CIE94_K1 * c1;
    let sh = 1.0 + cs::CIE94_K2 * c1;

    // ΔE94² = (ΔL/kL)² + (ΔC/SC)² + (ΔH/SH)²
    // kL = 1 for graphic arts
    let dl_term = dl;
    let dc_term = dc / sc;
    let dh_term = dh_sq.sqrt() / sh;

    dl_term * dl_term + dc_term * dc_term + dh_term * dh_term
}

/// CIEDE2000 (ΔE00): Most accurate perceptual distance squared
/// Includes lightness, chroma, and hue weighting plus rotation term for blue
#[inline]
pub fn lab_distance_ciede2000_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    use std::f32::consts::PI;
    const TWO_PI: f32 = 2.0 * PI;

    let c1_star = (a1 * a1 + b1 * b1).sqrt();
    let c2_star = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1_star + c2_star) / 2.0;

    // G factor for a' adjustment
    let c_bar_7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar_7 / (c_bar_7 + cs::CIEDE2000_POW25_7)).sqrt());

    // Adjusted a' values
    let a1_prime = a1 * (1.0 + g);
    let a2_prime = a2 * (1.0 + g);

    // Adjusted chroma
    let c1_prime = (a1_prime * a1_prime + b1 * b1).sqrt();
    let c2_prime = (a2_prime * a2_prime + b2 * b2).sqrt();

    // Hue angles (in radians, 0 to 2π)
    let h1_prime = if a1_prime == 0.0 && b1 == 0.0 {
        0.0
    } else {
        let h = b1.atan2(a1_prime);
        if h < 0.0 { h + TWO_PI } else { h }
    };

    let h2_prime = if a2_prime == 0.0 && b2 == 0.0 {
        0.0
    } else {
        let h = b2.atan2(a2_prime);
        if h < 0.0 { h + TWO_PI } else { h }
    };

    // Differences
    let dl_prime = l2 - l1;
    let dc_prime = c2_prime - c1_prime;

    // Hue difference
    let dh_prime = if c1_prime * c2_prime == 0.0 {
        0.0
    } else {
        let diff = h2_prime - h1_prime;
        if diff.abs() <= PI {
            diff
        } else if diff > PI {
            diff - TWO_PI
        } else {
            diff + TWO_PI
        }
    };

    // ΔH'
    let dh_prime_big = 2.0 * (c1_prime * c2_prime).sqrt() * (dh_prime / 2.0).sin();

    // Weighted mean values
    let l_bar_prime = (l1 + l2) / 2.0;
    let c_bar_prime = (c1_prime + c2_prime) / 2.0;

    let h_bar_prime = if c1_prime * c2_prime == 0.0 {
        h1_prime + h2_prime
    } else if (h1_prime - h2_prime).abs() <= PI {
        (h1_prime + h2_prime) / 2.0
    } else if h1_prime + h2_prime < TWO_PI {
        (h1_prime + h2_prime + TWO_PI) / 2.0
    } else {
        (h1_prime + h2_prime - TWO_PI) / 2.0
    };

    // T factor (using pre-computed radian constants)
    let t = 1.0
        - 0.17 * (h_bar_prime - cs::CIEDE2000_T_30_RAD).cos()
        + 0.24 * (2.0 * h_bar_prime).cos()
        + 0.32 * (3.0 * h_bar_prime + cs::CIEDE2000_T_6_RAD).cos()
        - 0.20 * (4.0 * h_bar_prime - cs::CIEDE2000_T_63_RAD).cos();

    // SL, SC, SH (using same K1/K2 as CIE94 for SC/SH base)
    let l_bar_minus_mid = l_bar_prime - cs::CIEDE2000_SL_L_MIDPOINT;
    let l_bar_minus_mid_sq = l_bar_minus_mid * l_bar_minus_mid;
    let sl = 1.0 + (cs::CIE94_K2 * l_bar_minus_mid_sq) / (cs::CIEDE2000_SL_DENOM_OFFSET + l_bar_minus_mid_sq).sqrt();
    let sc = 1.0 + cs::CIE94_K1 * c_bar_prime;
    let sh = 1.0 + cs::CIE94_K2 * c_bar_prime * t;

    // RT (rotation term for blue colors) - work in radians throughout
    // Δθ = 30° × exp(-((h̄' - 275°)/25°)²) but we compute in radians
    let h_bar_minus_275 = h_bar_prime - cs::CIEDE2000_RT_275_RAD;
    let delta_theta_rad: f32 = cs::CIEDE2000_RT_30_RAD
        * (-((h_bar_minus_275 / cs::CIEDE2000_RT_25_RAD).powi(2))).exp();
    let c_bar_prime_7 = c_bar_prime.powi(7);
    let rc = 2.0_f32 * (c_bar_prime_7 / (c_bar_prime_7 + cs::CIEDE2000_POW25_7)).sqrt();
    let rt = -rc * (2.0_f32 * delta_theta_rad).sin();

    // Final calculation (kL = kC = kH = 1)
    let dl_term = dl_prime / sl;
    let dc_term = dc_prime / sc;
    let dh_term = dh_prime_big / sh;

    dl_term * dl_term + dc_term * dc_term + dh_term * dh_term + rt * dc_term * dh_term
}

// ============================================================================
// Generic Distance Dispatcher
// ============================================================================

/// Compute perceptual distance squared based on the selected space/metric
/// Note: For LinearRGB mode, l/a/b actually contain linear R/G/B values
/// Note: For YCbCr mode, l/a/b actually contain Y'/Cb/Cr values
#[inline]
pub fn perceptual_distance_sq(
    space: PerceptualSpace,
    l1: f32, a1: f32, b1: f32,
    l2: f32, a2: f32, b2: f32,
) -> f32 {
    match space {
        PerceptualSpace::LabCIE76 => lab_distance_cie76_sq(l1, a1, b1, l2, a2, b2),
        PerceptualSpace::LabCIE94 => lab_distance_cie94_sq(l1, a1, b1, l2, a2, b2),
        PerceptualSpace::LabCIEDE2000 => lab_distance_ciede2000_sq(l1, a1, b1, l2, a2, b2),
        PerceptualSpace::OkLab
        | PerceptualSpace::OkLabLr
        | PerceptualSpace::LinearRGB
        | PerceptualSpace::YCbCr
        | PerceptualSpace::YCbCrBt601
        | PerceptualSpace::Srgb => {
            // OkLab uses simple Euclidean distance (it's designed for this)
            // OkLabLr uses Ottosson's revised lightness but still Euclidean distance
            // LinearRGB also uses simple Euclidean (l/a/b contain R/G/B in linear space)
            // YCbCr also uses simple Euclidean (l/a/b contain Y'/Cb/Cr values)
            // YCbCrBt601 also uses simple Euclidean (l/a/b contain Y'/Cb/Cr with BT.601 coefficients)
            // Srgb uses simple Euclidean (l/a/b contain gamma-encoded R/G/B values)
            let dl = l1 - l2;
            let da = a1 - a2;
            let db = b1 - b2;
            dl * dl + da * da + db * db
        }
        PerceptualSpace::OkLabHeavyChroma => {
            // OkLab with heavy chroma weighting (×4) for dithering
            // This penalizes chromatic differences more heavily, encouraging
            // neutral oscillations rather than complementary color oscillations.
            // Roughly matches CIE76's L:ab ratio of ~1:2.5
            let dl = l1 - l2;
            let da = a1 - a2;
            let db = b1 - b2;
            dl * dl + 4.0 * (da * da + db * db)
        }
    }
}

// ============================================================================
// Space Type Helpers
// ============================================================================

/// Check if a PerceptualSpace variant uses CIELAB
#[inline]
pub fn is_lab_space(space: PerceptualSpace) -> bool {
    matches!(space, PerceptualSpace::LabCIE76 | PerceptualSpace::LabCIE94 | PerceptualSpace::LabCIEDE2000)
}

/// Check if a PerceptualSpace variant uses linear RGB (no perceptual conversion)
#[inline]
pub fn is_linear_rgb_space(space: PerceptualSpace) -> bool {
    matches!(space, PerceptualSpace::LinearRGB)
}

/// Check if a PerceptualSpace variant uses Y'CbCr (BT.709)
#[inline]
pub fn is_ycbcr_space(space: PerceptualSpace) -> bool {
    matches!(space, PerceptualSpace::YCbCr)
}
