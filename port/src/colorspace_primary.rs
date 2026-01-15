//! Primary color space constants defined by specifications.
//!
//! These are the authoritative constants from standards documents and published research.
//! All values are stored in f64 for maximum precision during derivation.
//! Derived constants (XYZ matrices, etc.) are computed from these in colorspace_derived.rs.
//!
//! Sources:
//! - ITU-R BT.709-6 (sRGB primaries, D65)
//! - IEC 61966-2-1:1999 (sRGB transfer function)
//! - IEC 61966-2-5 (Adobe RGB)
//! - ISO 22028-2:2013 (ProPhoto/ROMM RGB)
//! - ITU-R BT.2020-2 (Rec.2020)
//! - CIE 15:2004 (illuminants)
//! - Björn Ottosson (2020) (OKLab)

// =============================================================================
// ILLUMINANT CHROMATICITY (CIE xy)
// =============================================================================

/// D65 standard illuminant - CIE authoritative definition.
/// From CIE 15:2004, derived from the D65 spectral power distribution.
/// This is the most precise definition but is not directly used by display specs.
pub mod d65_cie {
    pub const X: f64 = 0.31272;
    pub const Y: f64 = 0.32903;
}

/// D65 standard illuminant - 4-digit rounded values.
/// From ITU-R BT.709 / IEC 61966-2-1 / Adobe RGB specifications.
/// Note: D65 sRGB (derived from the sRGB matrix) is in colorspace_derived.rs.
pub mod d65 {
    pub const X: f64 = 0.3127;
    pub const Y: f64 = 0.3290;
}

/// D50 standard illuminant - CIE xy chromaticity.
/// Used by ProPhoto RGB and ICC color profiles.
/// From CIE 15:2004.
pub mod d50 {
    pub const X: f64 = 0.3457;
    pub const Y: f64 = 0.3585;
}

// =============================================================================
// sRGB / LINEAR RGB (XYZ matrix definition)
// =============================================================================

/// sRGB / Rec.709 XYZ conversion matrix.
/// From IEC 61966-2-1:1999 - these 4-digit coefficients are canonical.
/// The chromaticities are derived from these, not vice versa.
/// White point: D65
pub mod srgb_xyz {
    /// Linear sRGB → XYZ matrix (row-major).
    /// These are the exact values from the specification.
    pub const TO_XYZ: [[f64; 3]; 3] = [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ];
}

// =============================================================================
// COLOR SPACE PRIMARIES (CIE xy chromaticity)
// =============================================================================

/// Apple RGB primaries.
/// From classic Macintosh CRT phosphor specifications.
/// White point: D65
pub mod apple_rgb_primaries {
    pub const RED_X: f64 = 0.6250;
    pub const RED_Y: f64 = 0.3400;
    pub const GREEN_X: f64 = 0.2800;
    pub const GREEN_Y: f64 = 0.5950;
    pub const BLUE_X: f64 = 0.1550;
    pub const BLUE_Y: f64 = 0.0700;
}

/// Display P3 primaries.
/// Derived from DCI-P3 cinema standard, adapted for displays.
/// White point: D65
pub mod display_p3_primaries {
    pub const RED_X: f64 = 0.6800;
    pub const RED_Y: f64 = 0.3200;
    pub const GREEN_X: f64 = 0.2650;
    pub const GREEN_Y: f64 = 0.6900;
    pub const BLUE_X: f64 = 0.1500;
    pub const BLUE_Y: f64 = 0.0600;
}

/// Adobe RGB (1998) primaries.
/// From IEC 61966-2-5.
/// Note: Red and blue are identical to sRGB; only green differs.
/// White point: D65
pub mod adobe_rgb_primaries {
    pub const RED_X: f64 = 0.6400;
    pub const RED_Y: f64 = 0.3300;
    pub const GREEN_X: f64 = 0.2100;
    pub const GREEN_Y: f64 = 0.7100;
    pub const BLUE_X: f64 = 0.1500;
    pub const BLUE_Y: f64 = 0.0600;
}

/// ProPhoto RGB (ROMM RGB) primaries.
/// From ISO 22028-2:2013.
/// Note: Uses D50 white point, not D65.
/// Warning: Includes imaginary colors outside human vision.
pub mod prophoto_rgb_primaries {
    pub const RED_X: f64 = 0.7347;
    pub const RED_Y: f64 = 0.2653;
    pub const GREEN_X: f64 = 0.1596;
    pub const GREEN_Y: f64 = 0.8404;
    pub const BLUE_X: f64 = 0.0366;
    pub const BLUE_Y: f64 = 0.0001;
}

/// Rec.2020 primaries.
/// From ITU-R BT.2020-2.
/// Ultra-wide gamut for HDR/UHD television.
/// White point: D65
pub mod rec2020_primaries {
    pub const RED_X: f64 = 0.7080;
    pub const RED_Y: f64 = 0.2920;
    pub const GREEN_X: f64 = 0.1700;
    pub const GREEN_Y: f64 = 0.7970;
    pub const BLUE_X: f64 = 0.1310;
    pub const BLUE_Y: f64 = 0.0460;
}

// =============================================================================
// TRANSFER FUNCTION CONSTANTS
// =============================================================================

/// sRGB transfer function constants.
/// From IEC 61966-2-1:1999.
/// The piecewise function ensures continuity and continuous first derivative.
pub mod srgb_transfer {
    /// Linear segment threshold (encode direction).
    /// If linear <= THRESHOLD: srgb = LINEAR_SLOPE * linear
    pub const THRESHOLD: f64 = 0.0031308;

    /// Linear segment slope.
    pub const LINEAR_SLOPE: f64 = 12.92;

    /// Power curve exponent.
    pub const GAMMA: f64 = 2.4;

    /// Power curve scale factor.
    pub const SCALE: f64 = 1.055;

    /// Power curve offset.
    pub const OFFSET: f64 = 0.055;
}

/// Adobe RGB gamma.
/// From IEC 61966-2-5.
/// Often approximated as 2.2, but the exact value is 563/256.
pub mod adobe_rgb_transfer {
    /// Exact gamma value as specified: 563/256
    pub const GAMMA_NUMERATOR: u32 = 563;
    pub const GAMMA_DENOMINATOR: u32 = 256;
    // The derived f64 gamma value (563.0/256.0) goes in colorspace_derived.rs
}

/// Apple RGB gamma.
pub mod apple_rgb_transfer {
    pub const GAMMA: f64 = 1.8;
}

/// ProPhoto RGB transfer function constants.
/// From ISO 22028-2:2013 (ROMM RGB specification).
/// Similar piecewise structure to sRGB but with different parameters.
pub mod prophoto_transfer {
    /// Linear segment threshold: 1/512
    pub const THRESHOLD_NUMERATOR: u32 = 1;
    pub const THRESHOLD_DENOMINATOR: u32 = 512;

    /// Linear segment multiplier.
    pub const LINEAR_MULTIPLIER: f64 = 16.0;

    /// Power curve gamma.
    pub const GAMMA: f64 = 1.8;
}

/// Common gamma 2.2 approximation.
/// No formal standard - often conflated with sRGB but technically distinct.
pub mod gamma22_transfer {
    pub const GAMMA: f64 = 2.2;
}

// =============================================================================
// CIELAB CONSTANTS
// =============================================================================

/// CIELAB (CIE 1976 L*a*b*) constants.
/// From CIE 15:2004.
/// The fundamental constant is δ = 6/29; everything else derives from it.
pub mod cielab {
    /// Fundamental constant δ = 6/29.
    /// The f(t) function threshold is δ³, and the linear slope is (1/3)δ⁻².
    pub const DELTA_NUMERATOR: u32 = 6;
    pub const DELTA_DENOMINATOR: u32 = 29;

    /// L* channel scale factor.
    pub const L_SCALE: f64 = 116.0;

    /// L* channel offset.
    pub const L_OFFSET: f64 = 16.0;

    /// a* channel scale factor.
    pub const A_SCALE: f64 = 500.0;

    /// b* channel scale factor.
    pub const B_SCALE: f64 = 200.0;
}

// =============================================================================
// CIELAB COLOR DIFFERENCE FORMULA CONSTANTS
// =============================================================================

/// CIE94 (ΔE*94) color difference formula constants.
/// From CIE 116-1995.
/// Graphic arts application constants (kL=1). For textiles, use kL=2.
pub mod cie94 {
    /// Lightness weighting factor (1 for graphic arts, 2 for textiles).
    pub const KL: f64 = 1.0;

    /// Chroma weighting factor.
    pub const KC: f64 = 1.0;

    /// Hue weighting factor.
    pub const KH: f64 = 1.0;

    /// Chroma scaling coefficient for SC = 1 + K1*C.
    pub const K1: f64 = 0.045;

    /// Hue scaling coefficient for SH = 1 + K2*C.
    pub const K2: f64 = 0.015;
}

/// CIEDE2000 (ΔE00) color difference formula constants.
/// From CIE 142-2001.
/// The most accurate perceptual color difference metric.
pub mod ciede2000 {
    /// Lightness parametric factor.
    pub const KL: f64 = 1.0;

    /// Chroma parametric factor.
    pub const KC: f64 = 1.0;

    /// Hue parametric factor.
    pub const KH: f64 = 1.0;

    /// 25^7 - chroma correction threshold.
    /// Used in G factor and RC calculation.
    /// Exact integer: 25 × 25 × 25 × 25 × 25 × 25 × 25 = 6,103,515,625
    pub const POW25_7: f64 = 6103515625.0;

    /// Angle constants in degrees (converted to radians in colorspace_derived.rs).
    /// T factor angles:
    pub const T_ANGLE_30_DEG: f64 = 30.0;
    pub const T_ANGLE_6_DEG: f64 = 6.0;
    pub const T_ANGLE_63_DEG: f64 = 63.0;

    /// RT rotation term angles:
    pub const RT_ANGLE_275_DEG: f64 = 275.0;
    pub const RT_ANGLE_25_DEG: f64 = 25.0;
    pub const RT_ANGLE_30_DEG: f64 = 30.0;

    /// SL weighting function constants.
    /// SL = 1 + (K2 × (L̄' - SL_L_MIDPOINT)²) / √(SL_DENOM_OFFSET + (L̄' - SL_L_MIDPOINT)²)
    /// Note: K2 (0.015) is shared with CIE94 and defined in cie94 module.
    pub const SL_L_MIDPOINT: f64 = 50.0;
    pub const SL_DENOM_OFFSET: f64 = 20.0;
}

// =============================================================================
// OKLAB CONSTANTS
// =============================================================================

/// OKLab matrices.
/// From Björn Ottosson, "A perceptual color space for image processing" (2020).
/// All matrices are definitional - they were numerically optimized for
/// perceptual uniformity, not derived from colorimetric principles.
///
/// Conversion: Linear RGB → lms → lms' (cube root) → Lab
pub mod oklab {
    /// M1: Linear sRGB → LMS (cone response approximation)
    /// Row-major order: M1[row][col]
    pub const M1: [[f64; 3]; 3] = [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ];

    /// M2: LMS' → OKLab (opponent channels)
    /// Applied after cube root of LMS values.
    pub const M2: [[f64; 3]; 3] = [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ];

    /// M1_inv: LMS → Linear sRGB
    /// Inverse of M1.
    pub const M1_INV: [[f64; 3]; 3] = [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ];

    /// M2_inv: OKLab → LMS'
    /// Inverse of M2. Applied before cubing to get LMS.
    pub const M2_INV: [[f64; 3]; 3] = [
        [1.0000000000, 0.3963377774, 0.2158037573],
        [1.0000000000, -0.1055613458, -0.0638541728],
        [1.0000000000, -0.0894841775, -1.2914855480],
    ];
}

// =============================================================================
// BRADFORD CHROMATIC ADAPTATION
// =============================================================================

/// Bradford chromatic adaptation matrix.
/// From Lam (1985) and Hunt (1994), widely adopted for ICC color management.
/// Transforms XYZ to a "sharpened" cone-like response space for von Kries adaptation.
///
/// The adaptation from source white (Ws) to destination white (Wd) is:
///   M_adapt = BRADFORD_INV × diag(LMS_Wd / LMS_Ws) × BRADFORD
/// where LMS = BRADFORD × XYZ_white
pub mod bradford {
    /// Bradford matrix: XYZ → LMS (cone-like response)
    pub const MATRIX: [[f64; 3]; 3] = [
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296],
    ];
}

// =============================================================================
// Y'CbCr CONSTANTS (BT.709)
// =============================================================================

/// Y'CbCr luma coefficients for Rec.709.
/// From ITU-R BT.709-6.
/// These match the second row of the RGB→XYZ matrix (luminance).
/// Note: Applied to gamma-encoded values, not linear light.
pub mod ycbcr_bt709 {
    /// Luma coefficient for R' channel.
    pub const KR: f64 = 0.2126;

    /// Luma coefficient for G' channel.
    pub const KG: f64 = 0.7152;

    /// Luma coefficient for B' channel.
    pub const KB: f64 = 0.0722;
}

/// Y'CbCr luma coefficients for Rec.601 (NTSC/PAL).
/// From ITU-R BT.601.
/// Classic "30/59/11" formula - often incorrectly applied to sRGB content.
pub mod ycbcr_bt601 {
    pub const KR: f64 = 0.299;
    pub const KG: f64 = 0.587;
    pub const KB: f64 = 0.114;
}

// =============================================================================
// BIT DEPTH CONSTANTS
// =============================================================================

/// Canonical bit depth divisors for float conversion.
/// From the bit depth specification: uint8/255 and uint16/65535 are coherent.
pub mod bit_depth {
    /// Maximum value for uint8: 2^8 - 1
    pub const UINT8_MAX: u32 = 255;

    /// Maximum value for uint16: 2^16 - 1
    pub const UINT16_MAX: u32 = 65535;
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_xyz_matrix_valid() {
        // Sanity check that the sRGB XYZ matrix has reasonable values
        // All elements should be positive or small negative (for inverse)
        for row in &srgb_xyz::TO_XYZ {
            for &val in row {
                assert!(val >= 0.0 && val < 1.0, "Unexpected matrix value: {}", val);
            }
        }
        // Row sums should approximately match D65 white point XYZ
        let x_sum: f64 = srgb_xyz::TO_XYZ[0].iter().sum();
        let y_sum: f64 = srgb_xyz::TO_XYZ[1].iter().sum();
        let z_sum: f64 = srgb_xyz::TO_XYZ[2].iter().sum();
        // Y should be 1.0 (luminance of white)
        assert!((y_sum - 1.0).abs() < 0.001, "Y row sum: {}", y_sum);
        // X and Z should be close to D65 white point
        assert!((x_sum - 0.9505).abs() < 0.01, "X row sum: {}", x_sum);
        assert!((z_sum - 1.089).abs() < 0.01, "Z row sum: {}", z_sum);
    }

    #[test]
    fn test_srgb_transfer_continuity() {
        // At the threshold, both sides of the piecewise function should match
        let threshold = srgb_transfer::THRESHOLD;
        let linear_side = srgb_transfer::LINEAR_SLOPE * threshold;
        let power_side =
            srgb_transfer::SCALE * threshold.powf(1.0 / srgb_transfer::GAMMA) - srgb_transfer::OFFSET;
        assert!(
            (linear_side - power_side).abs() < 1e-6,
            "sRGB transfer function discontinuity: {} vs {}",
            linear_side,
            power_side
        );
    }

    #[test]
    fn test_oklab_matrices_are_inverses() {
        // M1 * M1_inv should be identity (within floating point tolerance)
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += oklab::M1[i][k] * oklab::M1_INV[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-6,
                    "M1 * M1_inv not identity at [{},{}]: {}",
                    i,
                    j,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_ycbcr_coefficients_sum_to_one() {
        let sum = ycbcr_bt709::KR + ycbcr_bt709::KG + ycbcr_bt709::KB;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Y'CbCr coefficients should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_cielab_delta() {
        let delta = cielab::DELTA_NUMERATOR as f64 / cielab::DELTA_DENOMINATOR as f64;
        assert!((delta - 6.0 / 29.0).abs() < 1e-15);
    }
}
