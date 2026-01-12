//! Generator for colorspace_derived.rs
//!
//! This tool reads primary color space constants and computes all mathematically
//! derived constants at maximum f64 precision, outputting a Rust source file.
//!
//! Run from the port/ directory:
//!   cargo run --manifest-path tools/gen_colorspace_derived/Cargo.toml > src/colorspace_derived.rs

use std::io::{self, Write};

// =============================================================================
// PRIMARY CONSTANTS (duplicated from colorspace_primary.rs for self-containment)
// =============================================================================

mod primary {
    /// D65 CIE - authoritative definition from CIE 15:2004.
    pub mod d65_cie {
        pub const X: f64 = 0.31272;
        pub const Y: f64 = 0.32903;
    }

    /// D65 4-digit - rounded values from ITU-R BT.709 / sRGB / Adobe RGB specs.
    pub mod d65 {
        pub const X: f64 = 0.3127;
        pub const Y: f64 = 0.3290;
    }

    pub mod d50 {
        pub const X: f64 = 0.3457;
        pub const Y: f64 = 0.3585;
    }

    /// sRGB XYZ matrix - these 4-digit values are canonical per IEC 61966-2-1.
    /// Chromaticities and D65 sRGB are derived from these, not vice versa.
    pub mod srgb_xyz {
        pub const TO_XYZ: [[f64; 3]; 3] = [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ];
    }

    pub mod apple_rgb_primaries {
        pub const RED_X: f64 = 0.6250;
        pub const RED_Y: f64 = 0.3400;
        pub const GREEN_X: f64 = 0.2800;
        pub const GREEN_Y: f64 = 0.5950;
        pub const BLUE_X: f64 = 0.1550;
        pub const BLUE_Y: f64 = 0.0700;
    }

    pub mod display_p3_primaries {
        pub const RED_X: f64 = 0.6800;
        pub const RED_Y: f64 = 0.3200;
        pub const GREEN_X: f64 = 0.2650;
        pub const GREEN_Y: f64 = 0.6900;
        pub const BLUE_X: f64 = 0.1500;
        pub const BLUE_Y: f64 = 0.0600;
    }

    pub mod adobe_rgb_primaries {
        pub const RED_X: f64 = 0.6400;
        pub const RED_Y: f64 = 0.3300;
        pub const GREEN_X: f64 = 0.2100;
        pub const GREEN_Y: f64 = 0.7100;
        pub const BLUE_X: f64 = 0.1500;
        pub const BLUE_Y: f64 = 0.0600;
    }

    pub mod prophoto_rgb_primaries {
        pub const RED_X: f64 = 0.7347;
        pub const RED_Y: f64 = 0.2653;
        pub const GREEN_X: f64 = 0.1596;
        pub const GREEN_Y: f64 = 0.8404;
        pub const BLUE_X: f64 = 0.0366;
        pub const BLUE_Y: f64 = 0.0001;
    }

    pub mod rec2020_primaries {
        pub const RED_X: f64 = 0.7080;
        pub const RED_Y: f64 = 0.2920;
        pub const GREEN_X: f64 = 0.1700;
        pub const GREEN_Y: f64 = 0.7970;
        pub const BLUE_X: f64 = 0.1310;
        pub const BLUE_Y: f64 = 0.0460;
    }

    pub mod srgb_transfer {
        pub const THRESHOLD: f64 = 0.0031308;
        pub const LINEAR_SLOPE: f64 = 12.92;
    }

    pub mod adobe_rgb_transfer {
        pub const GAMMA_NUMERATOR: u32 = 563;
        pub const GAMMA_DENOMINATOR: u32 = 256;
    }

    pub mod apple_rgb_transfer {
        pub const GAMMA: f64 = 1.8;
    }

    pub mod prophoto_transfer {
        pub const THRESHOLD_NUMERATOR: u32 = 1;
        pub const THRESHOLD_DENOMINATOR: u32 = 512;
        pub const LINEAR_MULTIPLIER: f64 = 16.0;
        pub const GAMMA: f64 = 1.8;
    }

    pub mod cielab {
        pub const DELTA_NUMERATOR: u32 = 6;
        pub const DELTA_DENOMINATOR: u32 = 29;
    }

    pub mod ycbcr_bt709 {
        pub const KR: f64 = 0.2126;
        pub const KG: f64 = 0.7152;
        pub const KB: f64 = 0.0722;
    }

    pub mod oklab {
        pub const M1: [[f64; 3]; 3] = [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ];

        pub const M2: [[f64; 3]; 3] = [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ];

        pub const M1_INV: [[f64; 3]; 3] = [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ];

        pub const M2_INV: [[f64; 3]; 3] = [
            [1.0000000000, 0.3963377774, 0.2158037573],
            [1.0000000000, -0.1055613458, -0.0638541728],
            [1.0000000000, -0.0894841775, -1.2914855480],
        ];
    }
}

// =============================================================================
// MATHEMATICAL DERIVATIONS
// =============================================================================

/// Convert CIE xy chromaticity to XYZ with Y=1.
fn xy_to_xyz(x: f64, y: f64) -> [f64; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Derive xy chromaticity from XYZ values.
/// x = X/(X+Y+Z), y = Y/(X+Y+Z)
fn xyz_to_xy(xyz: [f64; 3]) -> (f64, f64) {
    let sum = xyz[0] + xyz[1] + xyz[2];
    (xyz[0] / sum, xyz[1] / sum)
}

/// Extract column from a 3x3 matrix (for getting primary XYZ from RGB→XYZ matrix).
fn matrix_column(m: &Mat3, col: usize) -> [f64; 3] {
    [m[0][col], m[1][col], m[2][col]]
}

/// 3x3 matrix type for convenience
type Mat3 = [[f64; 3]; 3];

/// Compute RGB→XYZ matrix from primaries and white point.
///
/// The derivation:
/// 1. Convert each primary's xy chromaticity to XYZ (with Y=1 each)
/// 2. These form columns of matrix P
/// 3. Solve for scaling factors S such that P * S = white_XYZ
/// 4. Final matrix = P with columns scaled by S
fn compute_rgb_to_xyz_matrix(
    red_xy: (f64, f64),
    green_xy: (f64, f64),
    blue_xy: (f64, f64),
    white_xy: (f64, f64),
) -> Mat3 {
    // Convert primaries to XYZ (Y=1 for each)
    let r_xyz = xy_to_xyz(red_xy.0, red_xy.1);
    let g_xyz = xy_to_xyz(green_xy.0, green_xy.1);
    let b_xyz = xy_to_xyz(blue_xy.0, blue_xy.1);

    // White point XYZ
    let w_xyz = xy_to_xyz(white_xy.0, white_xy.1);

    // Matrix P: primaries as columns
    // P = | Xr Xg Xb |
    //     | Yr Yg Yb |
    //     | Zr Zg Zb |
    let p: Mat3 = [
        [r_xyz[0], g_xyz[0], b_xyz[0]],
        [r_xyz[1], g_xyz[1], b_xyz[1]],
        [r_xyz[2], g_xyz[2], b_xyz[2]],
    ];

    // Solve P * S = W for scaling factors S
    // S = P^-1 * W
    let p_inv = invert_3x3(p);
    let s = [
        p_inv[0][0] * w_xyz[0] + p_inv[0][1] * w_xyz[1] + p_inv[0][2] * w_xyz[2],
        p_inv[1][0] * w_xyz[0] + p_inv[1][1] * w_xyz[1] + p_inv[1][2] * w_xyz[2],
        p_inv[2][0] * w_xyz[0] + p_inv[2][1] * w_xyz[1] + p_inv[2][2] * w_xyz[2],
    ];

    // Final matrix: scale columns of P by S
    [
        [p[0][0] * s[0], p[0][1] * s[1], p[0][2] * s[2]],
        [p[1][0] * s[0], p[1][1] * s[1], p[1][2] * s[2]],
        [p[2][0] * s[0], p[2][1] * s[1], p[2][2] * s[2]],
    ]
}

/// Invert a 3x3 matrix using cofactor expansion.
fn invert_3x3(m: Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv_det = 1.0 / det;

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

/// Compute Y'CbCr matrices from luma coefficients.
fn compute_ycbcr_matrices(kr: f64, kg: f64, kb: f64) -> (Mat3, Mat3) {
    // Cb and Cr scaling factors
    let cb_scale = 2.0 * (1.0 - kb);
    let cr_scale = 2.0 * (1.0 - kr);

    // Forward matrix: RGB' → Y'CbCr
    // Y' = Kr*R' + Kg*G' + Kb*B'
    // Cb row: [-Kr/(2(1-Kb)), -Kg/(2(1-Kb)), 0.5]
    // Cr row: [0.5, -Kg/(2(1-Kr)), -Kb/(2(1-Kr))]
    let forward_simplified: Mat3 = [
        [kr, kg, kb],
        [-kr / cb_scale, -kg / cb_scale, 0.5],
        [0.5, -kg / cr_scale, -kb / cr_scale],
    ];

    // Inverse matrix: Y'CbCr → RGB'
    // R' = Y' + Cr * cr_scale
    // B' = Y' + Cb * cb_scale
    // G' = Y' - (Kr/Kg)*Cr*cr_scale - (Kb/Kg)*Cb*cb_scale
    //    = Y' - (2*Kr*(1-Kr)/Kg)*Cr - (2*Kb*(1-Kb)/Kg)*Cb
    let inverse: Mat3 = [
        [1.0, 0.0, cr_scale],
        [
            1.0,
            -2.0 * kb * (1.0 - kb) / kg,
            -2.0 * kr * (1.0 - kr) / kg,
        ],
        [1.0, cb_scale, 0.0],
    ];

    (forward_simplified, inverse)
}

/// Format a floating point number intelligently.
/// If the value is very close to a short decimal, use that clean representation.
/// Otherwise use full precision for derived values.
fn fmt_f64(v: f64) -> String {
    // Try progressively shorter decimal representations
    for precision in 1..=10 {
        let formatted = format!("{:.prec$}", v, prec = precision);
        let parsed: f64 = formatted.parse().unwrap();
        // If parsing the short representation gives us back the same value, use it
        if parsed == v {
            return formatted;
        }
    }
    // Fall back to full precision for truly derived values
    format!("{:.17}", v)
}

/// Format a 3x3 matrix as Rust code.
fn fmt_matrix(m: Mat3, indent: &str) -> String {
    format!(
        "[\n{indent}    [{}, {}, {}],\n{indent}    [{}, {}, {}],\n{indent}    [{}, {}, {}],\n{indent}]",
        fmt_f64(m[0][0]), fmt_f64(m[0][1]), fmt_f64(m[0][2]),
        fmt_f64(m[1][0]), fmt_f64(m[1][1]), fmt_f64(m[1][2]),
        fmt_f64(m[2][0]), fmt_f64(m[2][1]), fmt_f64(m[2][2]),
    )
}

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Compute all derived constants

    // D65 variants
    let d65_cie_xyz = xy_to_xyz(primary::d65_cie::X, primary::d65_cie::Y);
    let d65_xyz = xy_to_xyz(primary::d65::X, primary::d65::Y); // 4-digit

    // D50
    let d50_xyz = xy_to_xyz(primary::d50::X, primary::d50::Y);

    // sRGB / Rec.709 matrices - use canonical 4-digit XYZ matrix directly
    let srgb_to_xyz = primary::srgb_xyz::TO_XYZ;
    let xyz_to_srgb = invert_3x3(srgb_to_xyz);

    // D65 sRGB - derived from matrix row sums (white = R+G+B)
    let d65_srgb_xyz = [
        srgb_to_xyz[0][0] + srgb_to_xyz[0][1] + srgb_to_xyz[0][2],
        srgb_to_xyz[1][0] + srgb_to_xyz[1][1] + srgb_to_xyz[1][2],
        srgb_to_xyz[2][0] + srgb_to_xyz[2][1] + srgb_to_xyz[2][2],
    ];
    let d65_srgb_xy = xyz_to_xy(d65_srgb_xyz);

    // Derive sRGB chromaticities from XYZ matrix columns
    let srgb_red_xy = xyz_to_xy(matrix_column(&srgb_to_xyz, 0));
    let srgb_green_xy = xyz_to_xy(matrix_column(&srgb_to_xyz, 1));
    let srgb_blue_xy = xyz_to_xy(matrix_column(&srgb_to_xyz, 2));

    // Apple RGB matrices
    let apple_to_xyz = compute_rgb_to_xyz_matrix(
        (primary::apple_rgb_primaries::RED_X, primary::apple_rgb_primaries::RED_Y),
        (primary::apple_rgb_primaries::GREEN_X, primary::apple_rgb_primaries::GREEN_Y),
        (primary::apple_rgb_primaries::BLUE_X, primary::apple_rgb_primaries::BLUE_Y),
        (primary::d65::X, primary::d65::Y),
    );
    let xyz_to_apple = invert_3x3(apple_to_xyz);

    // Display P3 matrices
    let p3_to_xyz = compute_rgb_to_xyz_matrix(
        (primary::display_p3_primaries::RED_X, primary::display_p3_primaries::RED_Y),
        (primary::display_p3_primaries::GREEN_X, primary::display_p3_primaries::GREEN_Y),
        (primary::display_p3_primaries::BLUE_X, primary::display_p3_primaries::BLUE_Y),
        (primary::d65::X, primary::d65::Y),
    );
    let xyz_to_p3 = invert_3x3(p3_to_xyz);

    // Adobe RGB matrices
    let adobe_to_xyz = compute_rgb_to_xyz_matrix(
        (primary::adobe_rgb_primaries::RED_X, primary::adobe_rgb_primaries::RED_Y),
        (primary::adobe_rgb_primaries::GREEN_X, primary::adobe_rgb_primaries::GREEN_Y),
        (primary::adobe_rgb_primaries::BLUE_X, primary::adobe_rgb_primaries::BLUE_Y),
        (primary::d65::X, primary::d65::Y),
    );
    let xyz_to_adobe = invert_3x3(adobe_to_xyz);

    // ProPhoto RGB matrices (D50 white point)
    let prophoto_to_xyz = compute_rgb_to_xyz_matrix(
        (primary::prophoto_rgb_primaries::RED_X, primary::prophoto_rgb_primaries::RED_Y),
        (primary::prophoto_rgb_primaries::GREEN_X, primary::prophoto_rgb_primaries::GREEN_Y),
        (primary::prophoto_rgb_primaries::BLUE_X, primary::prophoto_rgb_primaries::BLUE_Y),
        (primary::d50::X, primary::d50::Y),
    );
    let xyz_to_prophoto = invert_3x3(prophoto_to_xyz);

    // Rec.2020 matrices
    let rec2020_to_xyz = compute_rgb_to_xyz_matrix(
        (primary::rec2020_primaries::RED_X, primary::rec2020_primaries::RED_Y),
        (primary::rec2020_primaries::GREEN_X, primary::rec2020_primaries::GREEN_Y),
        (primary::rec2020_primaries::BLUE_X, primary::rec2020_primaries::BLUE_Y),
        (primary::d65::X, primary::d65::Y),
    );
    let xyz_to_rec2020 = invert_3x3(rec2020_to_xyz);

    // CIELAB derived constants
    let delta = primary::cielab::DELTA_NUMERATOR as f64 / primary::cielab::DELTA_DENOMINATOR as f64;
    let lab_epsilon = delta * delta * delta; // (6/29)^3
    let lab_kappa = 1.0 / (3.0 * delta * delta); // (29/6)^2 / 3 = 1/(3*delta^2)
    let lab_offset = 4.0 / 29.0; // 4/29 = 16/116
    let lab_f_threshold = delta; // threshold in f-space for inverse

    // Transfer function derived constants
    let adobe_gamma = primary::adobe_rgb_transfer::GAMMA_NUMERATOR as f64
        / primary::adobe_rgb_transfer::GAMMA_DENOMINATOR as f64;
    let prophoto_threshold = primary::prophoto_transfer::THRESHOLD_NUMERATOR as f64
        / primary::prophoto_transfer::THRESHOLD_DENOMINATOR as f64;
    let srgb_decode_threshold = primary::srgb_transfer::THRESHOLD * primary::srgb_transfer::LINEAR_SLOPE;

    // Y'CbCr derived matrices
    let (ycbcr_forward, ycbcr_inverse) = compute_ycbcr_matrices(
        primary::ycbcr_bt709::KR,
        primary::ycbcr_bt709::KG,
        primary::ycbcr_bt709::KB,
    );
    let cb_scale = 2.0 * (1.0 - primary::ycbcr_bt709::KB);
    let cr_scale = 2.0 * (1.0 - primary::ycbcr_bt709::KR);

    // Generate the output file
    writeln!(out, "//! Derived color space constants.")?;
    writeln!(out, "//!")?;
    writeln!(out, "//! THIS FILE IS AUTO-GENERATED by gen_colorspace_derived.")?;
    writeln!(out, "//! Do not edit manually. Regenerate with:")?;
    writeln!(out, "//!   cargo run --manifest-path tools/gen_colorspace_derived/Cargo.toml > src/colorspace_derived.rs")?;
    writeln!(out, "//!")?;
    writeln!(out, "//! All constants are computed from primary constants at f64 precision.")?;
    writeln!(out)?;

    // Illuminant XYZ
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// D65 ILLUMINANT VARIANTS")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// D65 CIE - authoritative definition from CIE 15:2004.")?;
    writeln!(out, "/// Derived from xy chromaticity (0.31272, 0.32903).")?;
    writeln!(out, "pub mod d65_cie {{")?;
    writeln!(out, "    pub const X: f64 = {};", fmt_f64(primary::d65_cie::X))?;
    writeln!(out, "    pub const Y: f64 = {};", fmt_f64(primary::d65_cie::Y))?;
    writeln!(out, "    pub const XYZ_X: f64 = {};", fmt_f64(d65_cie_xyz[0]))?;
    writeln!(out, "    pub const XYZ_Y: f64 = {};", fmt_f64(d65_cie_xyz[1]))?;
    writeln!(out, "    pub const XYZ_Z: f64 = {};", fmt_f64(d65_cie_xyz[2]))?;
    writeln!(out, "}}")?;
    writeln!(out)?;
    writeln!(out, "/// D65 sRGB - derived from the canonical sRGB matrix row sums.")?;
    writeln!(out, "/// This is the operative white point for sRGB workflows.")?;
    writeln!(out, "pub mod d65_srgb {{")?;
    writeln!(out, "    pub const X: f64 = {};", fmt_f64(d65_srgb_xy.0))?;
    writeln!(out, "    pub const Y: f64 = {};", fmt_f64(d65_srgb_xy.1))?;
    writeln!(out, "    pub const XYZ_X: f64 = {};", fmt_f64(d65_srgb_xyz[0]))?;
    writeln!(out, "    pub const XYZ_Y: f64 = {};", fmt_f64(d65_srgb_xyz[1]))?;
    writeln!(out, "    pub const XYZ_Z: f64 = {};", fmt_f64(d65_srgb_xyz[2]))?;
    writeln!(out, "}}")?;
    writeln!(out)?;
    writeln!(out, "/// D65 4-digit - rounded values from ITU-R BT.709 / Adobe RGB specs.")?;
    writeln!(out, "/// Derived from xy chromaticity (0.3127, 0.3290).")?;
    writeln!(out, "pub mod d65 {{")?;
    writeln!(out, "    pub const X: f64 = {};", fmt_f64(primary::d65::X))?;
    writeln!(out, "    pub const Y: f64 = {};", fmt_f64(primary::d65::Y))?;
    writeln!(out, "    pub const XYZ_X: f64 = {};", fmt_f64(d65_xyz[0]))?;
    writeln!(out, "    pub const XYZ_Y: f64 = {};", fmt_f64(d65_xyz[1]))?;
    writeln!(out, "    pub const XYZ_Z: f64 = {};", fmt_f64(d65_xyz[2]))?;
    writeln!(out, "}}")?;
    writeln!(out)?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// D50 ILLUMINANT")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// D50 standard illuminant.")?;
    writeln!(out, "/// Derived from xy chromaticity (0.3457, 0.3585).")?;
    writeln!(out, "pub mod d50 {{")?;
    writeln!(out, "    pub const X: f64 = {};", fmt_f64(primary::d50::X))?;
    writeln!(out, "    pub const Y: f64 = {};", fmt_f64(primary::d50::Y))?;
    writeln!(out, "    pub const XYZ_X: f64 = {};", fmt_f64(d50_xyz[0]))?;
    writeln!(out, "    pub const XYZ_Y: f64 = {};", fmt_f64(d50_xyz[1]))?;
    writeln!(out, "    pub const XYZ_Z: f64 = {};", fmt_f64(d50_xyz[2]))?;
    writeln!(out, "}}")?;
    writeln!(out)?;

    // sRGB matrices
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// sRGB / Rec.709 MATRICES")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Linear sRGB → XYZ matrix.")?;
    writeln!(out, "/// These 4-digit coefficients are canonical per IEC 61966-2-1.")?;
    writeln!(out, "/// Row-major: result[row] = dot(matrix[row], rgb)")?;
    writeln!(out, "pub const SRGB_TO_XYZ: [[f64; 3]; 3] = {};", fmt_matrix(srgb_to_xyz, ""))?;
    writeln!(out)?;
    writeln!(out, "/// XYZ → Linear sRGB matrix (derived inverse).")?;
    writeln!(out, "pub const XYZ_TO_SRGB: [[f64; 3]; 3] = {};", fmt_matrix(xyz_to_srgb, ""))?;
    writeln!(out)?;
    writeln!(out, "/// sRGB chromaticities derived from XYZ matrix columns.")?;
    writeln!(out, "/// x = X/(X+Y+Z), y = Y/(X+Y+Z) for each primary.")?;
    writeln!(out, "pub mod srgb_chromaticity {{")?;
    writeln!(out, "    pub const RED_X: f64 = {};", fmt_f64(srgb_red_xy.0))?;
    writeln!(out, "    pub const RED_Y: f64 = {};", fmt_f64(srgb_red_xy.1))?;
    writeln!(out, "    pub const GREEN_X: f64 = {};", fmt_f64(srgb_green_xy.0))?;
    writeln!(out, "    pub const GREEN_Y: f64 = {};", fmt_f64(srgb_green_xy.1))?;
    writeln!(out, "    pub const BLUE_X: f64 = {};", fmt_f64(srgb_blue_xy.0))?;
    writeln!(out, "    pub const BLUE_Y: f64 = {};", fmt_f64(srgb_blue_xy.1))?;
    writeln!(out, "}}")?;
    writeln!(out)?;

    // Apple RGB matrices
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// APPLE RGB MATRICES")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Linear Apple RGB → XYZ matrix.")?;
    writeln!(out, "pub const APPLE_RGB_TO_XYZ: [[f64; 3]; 3] = {};", fmt_matrix(apple_to_xyz, ""))?;
    writeln!(out)?;
    writeln!(out, "/// XYZ → Linear Apple RGB matrix.")?;
    writeln!(out, "pub const XYZ_TO_APPLE_RGB: [[f64; 3]; 3] = {};", fmt_matrix(xyz_to_apple, ""))?;
    writeln!(out)?;

    // Display P3 matrices
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// DISPLAY P3 MATRICES")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Linear Display P3 → XYZ matrix.")?;
    writeln!(out, "pub const DISPLAY_P3_TO_XYZ: [[f64; 3]; 3] = {};", fmt_matrix(p3_to_xyz, ""))?;
    writeln!(out)?;
    writeln!(out, "/// XYZ → Linear Display P3 matrix.")?;
    writeln!(out, "pub const XYZ_TO_DISPLAY_P3: [[f64; 3]; 3] = {};", fmt_matrix(xyz_to_p3, ""))?;
    writeln!(out)?;

    // Adobe RGB matrices
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// ADOBE RGB MATRICES")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Linear Adobe RGB → XYZ matrix.")?;
    writeln!(out, "pub const ADOBE_RGB_TO_XYZ: [[f64; 3]; 3] = {};", fmt_matrix(adobe_to_xyz, ""))?;
    writeln!(out)?;
    writeln!(out, "/// XYZ → Linear Adobe RGB matrix.")?;
    writeln!(out, "pub const XYZ_TO_ADOBE_RGB: [[f64; 3]; 3] = {};", fmt_matrix(xyz_to_adobe, ""))?;
    writeln!(out)?;

    // ProPhoto RGB matrices
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// PROPHOTO RGB MATRICES (D50 white point)")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Linear ProPhoto RGB → XYZ matrix.")?;
    writeln!(out, "/// Note: Uses D50 white point, not D65.")?;
    writeln!(out, "pub const PROPHOTO_RGB_TO_XYZ: [[f64; 3]; 3] = {};", fmt_matrix(prophoto_to_xyz, ""))?;
    writeln!(out)?;
    writeln!(out, "/// XYZ → Linear ProPhoto RGB matrix.")?;
    writeln!(out, "pub const XYZ_TO_PROPHOTO_RGB: [[f64; 3]; 3] = {};", fmt_matrix(xyz_to_prophoto, ""))?;
    writeln!(out)?;

    // Rec.2020 matrices
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// REC.2020 MATRICES")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Linear Rec.2020 → XYZ matrix.")?;
    writeln!(out, "pub const REC2020_TO_XYZ: [[f64; 3]; 3] = {};", fmt_matrix(rec2020_to_xyz, ""))?;
    writeln!(out)?;
    writeln!(out, "/// XYZ → Linear Rec.2020 matrix.")?;
    writeln!(out, "pub const XYZ_TO_REC2020: [[f64; 3]; 3] = {};", fmt_matrix(xyz_to_rec2020, ""))?;
    writeln!(out)?;

    // CIELAB derived constants
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// CIELAB DERIVED CONSTANTS")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// CIELAB constants derived from δ = 6/29.")?;
    writeln!(out, "pub mod cielab {{")?;
    writeln!(out, "    /// δ = 6/29")?;
    writeln!(out, "    pub const DELTA: f64 = {};", fmt_f64(delta))?;
    writeln!(out)?;
    writeln!(out, "    /// ε = δ³ = (6/29)³ ≈ 0.008856")?;
    writeln!(out, "    /// Threshold for f(t): if t > EPSILON, use cube root.")?;
    writeln!(out, "    pub const EPSILON: f64 = {};", fmt_f64(lab_epsilon))?;
    writeln!(out)?;
    writeln!(out, "    /// κ = (29/6)² / 3 ≈ 7.787")?;
    writeln!(out, "    /// Linear segment slope: f(t) = κ*t + 16/116 for t ≤ ε")?;
    writeln!(out, "    pub const KAPPA: f64 = {};", fmt_f64(lab_kappa))?;
    writeln!(out)?;
    writeln!(out, "    /// Linear segment offset = 16/116 = 4/29")?;
    writeln!(out, "    pub const OFFSET: f64 = {};", fmt_f64(lab_offset))?;
    writeln!(out)?;
    writeln!(out, "    /// Threshold in f-space for inverse: f(ε) = δ = 6/29")?;
    writeln!(out, "    /// If f > F_THRESHOLD, use cube; otherwise use linear inverse.")?;
    writeln!(out, "    pub const F_THRESHOLD: f64 = {};", fmt_f64(lab_f_threshold))?;
    writeln!(out, "}}")?;
    writeln!(out)?;

    // Transfer function derived constants
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// TRANSFER FUNCTION DERIVED CONSTANTS")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// sRGB decode threshold in encoded space.")?;
    writeln!(out, "/// sRGB specification threshold (0.0031308) * linear_slope (12.92)")?;
    writeln!(out, "pub const SRGB_DECODE_THRESHOLD: f64 = {};", fmt_f64(srgb_decode_threshold))?;
    writeln!(out)?;
    writeln!(out, "/// Adobe RGB gamma: 563/256")?;
    writeln!(out, "pub const ADOBE_RGB_GAMMA: f64 = {};", fmt_f64(adobe_gamma))?;
    writeln!(out)?;
    writeln!(out, "/// ProPhoto RGB linear segment threshold: 1/512")?;
    writeln!(out, "pub const PROPHOTO_THRESHOLD: f64 = {};", fmt_f64(prophoto_threshold))?;
    writeln!(out)?;
    writeln!(out, "/// ProPhoto RGB decode threshold in encoded space: 16 * (1/512)")?;
    writeln!(out, "pub const PROPHOTO_DECODE_THRESHOLD: f64 = {};", fmt_f64(prophoto_threshold * primary::prophoto_transfer::LINEAR_MULTIPLIER))?;
    writeln!(out)?;

    // Y'CbCr derived constants
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// Y'CbCr BT.709 DERIVED CONSTANTS")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// Cb channel scaling factor: 2(1-Kb)")?;
    writeln!(out, "pub const YCBCR_CB_SCALE: f64 = {};", fmt_f64(cb_scale))?;
    writeln!(out)?;
    writeln!(out, "/// Cr channel scaling factor: 2(1-Kr)")?;
    writeln!(out, "pub const YCBCR_CR_SCALE: f64 = {};", fmt_f64(cr_scale))?;
    writeln!(out)?;
    writeln!(out, "/// RGB' → Y'CbCr matrix (BT.709).")?;
    writeln!(out, "/// Y' = Kr*R' + Kg*G' + Kb*B'")?;
    writeln!(out, "/// Cb = 0.5*(B'-Y')/(1-Kb)")?;
    writeln!(out, "/// Cr = 0.5*(R'-Y')/(1-Kr)")?;
    writeln!(out, "pub const RGB_TO_YCBCR: [[f64; 3]; 3] = {};", fmt_matrix(ycbcr_forward, ""))?;
    writeln!(out)?;
    writeln!(out, "/// Y'CbCr → RGB' matrix (BT.709).")?;
    writeln!(out, "pub const YCBCR_TO_RGB: [[f64; 3]; 3] = {};", fmt_matrix(ycbcr_inverse, ""))?;
    writeln!(out)?;

    // f32 versions for runtime use
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// f32 VERSIONS FOR RUNTIME USE")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "/// f32 module for runtime use. All values derived from f64 constants.")?;
    writeln!(out, "pub mod f32 {{")?;
    writeln!(out)?;

    // Illuminant XYZ (4-digit D65)
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // ILLUMINANT XYZ")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    /// D65 XYZ (4-digit variant, from specs)")?;
    writeln!(out, "    pub const D65_X: f32 = {} as f32;", fmt_f64(d65_xyz[0]))?;
    writeln!(out, "    pub const D65_Y: f32 = {} as f32;", fmt_f64(d65_xyz[1]))?;
    writeln!(out, "    pub const D65_Z: f32 = {} as f32;", fmt_f64(d65_xyz[2]))?;
    writeln!(out, "    pub const D65_XYZ: [f32; 3] = [D65_X, D65_Y, D65_Z];")?;
    writeln!(out)?;
    writeln!(out, "    /// D65 sRGB XYZ (derived from sRGB matrix row sums)")?;
    writeln!(out, "    pub const D65_SRGB_X: f32 = {} as f32;", fmt_f64(d65_srgb_xyz[0]))?;
    writeln!(out, "    pub const D65_SRGB_Y: f32 = {} as f32;", fmt_f64(d65_srgb_xyz[1]))?;
    writeln!(out, "    pub const D65_SRGB_Z: f32 = {} as f32;", fmt_f64(d65_srgb_xyz[2]))?;
    writeln!(out, "    pub const D65_SRGB_XYZ: [f32; 3] = [D65_SRGB_X, D65_SRGB_Y, D65_SRGB_Z];")?;
    writeln!(out)?;
    writeln!(out, "    pub const D50_X: f32 = {} as f32;", fmt_f64(d50_xyz[0]))?;
    writeln!(out, "    pub const D50_Y: f32 = {} as f32;", fmt_f64(d50_xyz[1]))?;
    writeln!(out, "    pub const D50_Z: f32 = {} as f32;", fmt_f64(d50_xyz[2]))?;
    writeln!(out, "    pub const D50_XYZ: [f32; 3] = [D50_X, D50_Y, D50_Z];")?;
    writeln!(out)?;

    // Illuminant chromaticity
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // ILLUMINANT CHROMATICITY")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    /// D65 chromaticity (4-digit variant)")?;
    writeln!(out, "    pub const D65_CHROMATICITY: [f32; 2] = [{} as f32, {} as f32];",
        fmt_f64(primary::d65::X), fmt_f64(primary::d65::Y))?;
    writeln!(out, "    pub const D50_CHROMATICITY: [f32; 2] = [{} as f32, {} as f32];",
        fmt_f64(primary::d50::X), fmt_f64(primary::d50::Y))?;
    writeln!(out)?;

    // RGB↔XYZ matrices
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // RGB <-> XYZ MATRICES")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;

    // Helper to write matrix as f32
    let write_matrix_f32 = |out: &mut std::io::StdoutLock, name: &str, m: &Mat3| -> io::Result<()> {
        writeln!(out, "    pub const {}: [[f32; 3]; 3] = [", name)?;
        for row in m {
            writeln!(out, "        [{} as f32, {} as f32, {} as f32],",
                fmt_f64(row[0]), fmt_f64(row[1]), fmt_f64(row[2]))?;
        }
        writeln!(out, "    ];")?;
        Ok(())
    };

    write_matrix_f32(&mut out, "SRGB_TO_XYZ", &srgb_to_xyz)?;
    write_matrix_f32(&mut out, "XYZ_TO_SRGB", &xyz_to_srgb)?;
    writeln!(out)?;
    writeln!(out, "    /// sRGB chromaticities (derived from XYZ matrix)")?;
    writeln!(out, "    pub const SRGB_RED_XY: [f32; 2] = [{} as f32, {} as f32];",
        fmt_f64(srgb_red_xy.0), fmt_f64(srgb_red_xy.1))?;
    writeln!(out, "    pub const SRGB_GREEN_XY: [f32; 2] = [{} as f32, {} as f32];",
        fmt_f64(srgb_green_xy.0), fmt_f64(srgb_green_xy.1))?;
    writeln!(out, "    pub const SRGB_BLUE_XY: [f32; 2] = [{} as f32, {} as f32];",
        fmt_f64(srgb_blue_xy.0), fmt_f64(srgb_blue_xy.1))?;
    writeln!(out)?;
    write_matrix_f32(&mut out, "APPLE_RGB_TO_XYZ", &apple_to_xyz)?;
    write_matrix_f32(&mut out, "XYZ_TO_APPLE_RGB", &xyz_to_apple)?;
    writeln!(out)?;
    write_matrix_f32(&mut out, "DISPLAY_P3_TO_XYZ", &p3_to_xyz)?;
    write_matrix_f32(&mut out, "XYZ_TO_DISPLAY_P3", &xyz_to_p3)?;
    writeln!(out)?;
    write_matrix_f32(&mut out, "ADOBE_RGB_TO_XYZ", &adobe_to_xyz)?;
    write_matrix_f32(&mut out, "XYZ_TO_ADOBE_RGB", &xyz_to_adobe)?;
    writeln!(out)?;
    write_matrix_f32(&mut out, "PROPHOTO_RGB_TO_XYZ", &prophoto_to_xyz)?;
    write_matrix_f32(&mut out, "XYZ_TO_PROPHOTO_RGB", &xyz_to_prophoto)?;
    writeln!(out)?;
    write_matrix_f32(&mut out, "REC2020_TO_XYZ", &rec2020_to_xyz)?;
    write_matrix_f32(&mut out, "XYZ_TO_REC2020", &xyz_to_rec2020)?;
    writeln!(out)?;

    // Transfer function constants
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // TRANSFER FUNCTION CONSTANTS")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    /// sRGB encode threshold (linear space)")?;
    writeln!(out, "    pub const SRGB_THRESHOLD: f32 = {} as f32;", fmt_f64(primary::srgb_transfer::THRESHOLD))?;
    writeln!(out, "    /// sRGB decode threshold (encoded space)")?;
    writeln!(out, "    pub const SRGB_DECODE_THRESHOLD: f32 = {} as f32;", fmt_f64(srgb_decode_threshold))?;
    writeln!(out, "    pub const SRGB_LINEAR_SLOPE: f32 = {} as f32;", fmt_f64(primary::srgb_transfer::LINEAR_SLOPE))?;
    writeln!(out, "    pub const SRGB_GAMMA: f32 = 2.4;")?;
    writeln!(out, "    pub const SRGB_SCALE: f32 = 1.055;")?;
    writeln!(out, "    pub const SRGB_OFFSET: f32 = 0.055;")?;
    writeln!(out)?;
    writeln!(out, "    pub const ADOBE_RGB_GAMMA: f32 = {} as f32;", fmt_f64(adobe_gamma))?;
    writeln!(out, "    pub const APPLE_RGB_GAMMA: f32 = {} as f32;", fmt_f64(primary::apple_rgb_transfer::GAMMA))?;
    writeln!(out)?;
    writeln!(out, "    pub const PROPHOTO_THRESHOLD: f32 = {} as f32;", fmt_f64(prophoto_threshold))?;
    writeln!(out, "    pub const PROPHOTO_DECODE_THRESHOLD: f32 = {} as f32;",
        fmt_f64(prophoto_threshold * primary::prophoto_transfer::LINEAR_MULTIPLIER))?;
    writeln!(out, "    pub const PROPHOTO_LINEAR_MULTIPLIER: f32 = {} as f32;",
        fmt_f64(primary::prophoto_transfer::LINEAR_MULTIPLIER))?;
    writeln!(out, "    pub const PROPHOTO_GAMMA: f32 = {} as f32;", fmt_f64(primary::prophoto_transfer::GAMMA))?;
    writeln!(out)?;
    writeln!(out, "    pub const GAMMA_22: f32 = 2.2;")?;
    writeln!(out)?;

    // CIELAB constants
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // CIELAB CONSTANTS")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    pub const CIELAB_DELTA: f32 = {} as f32;", fmt_f64(delta))?;
    writeln!(out, "    pub const CIELAB_EPSILON: f32 = {} as f32;", fmt_f64(lab_epsilon))?;
    writeln!(out, "    pub const CIELAB_KAPPA: f32 = {} as f32;", fmt_f64(lab_kappa))?;
    writeln!(out, "    pub const CIELAB_OFFSET: f32 = {} as f32;", fmt_f64(lab_offset))?;
    writeln!(out, "    pub const CIELAB_F_THRESHOLD: f32 = {} as f32;", fmt_f64(lab_f_threshold))?;
    writeln!(out, "    pub const CIELAB_L_SCALE: f32 = 116.0;")?;
    writeln!(out, "    pub const CIELAB_L_OFFSET: f32 = 16.0;")?;
    writeln!(out, "    pub const CIELAB_A_SCALE: f32 = 500.0;")?;
    writeln!(out, "    pub const CIELAB_B_SCALE: f32 = 200.0;")?;
    writeln!(out)?;

    // OKLab matrices (from primary constants)
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // OKLAB MATRICES (from primary constants)")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    /// OKLab M1: Linear sRGB → LMS")?;
    writeln!(out, "    pub const OKLAB_M1: [[f32; 3]; 3] = [")?;
    for row in &primary::oklab::M1 {
        writeln!(out, "        [{} as f32, {} as f32, {} as f32],", fmt_f64(row[0]), fmt_f64(row[1]), fmt_f64(row[2]))?;
    }
    writeln!(out, "    ];")?;
    writeln!(out)?;
    writeln!(out, "    /// OKLab M2: LMS' → Lab")?;
    writeln!(out, "    pub const OKLAB_M2: [[f32; 3]; 3] = [")?;
    for row in &primary::oklab::M2 {
        writeln!(out, "        [{} as f32, {} as f32, {} as f32],", fmt_f64(row[0]), fmt_f64(row[1]), fmt_f64(row[2]))?;
    }
    writeln!(out, "    ];")?;
    writeln!(out)?;
    writeln!(out, "    /// OKLab M1_inv: LMS → Linear sRGB")?;
    writeln!(out, "    pub const OKLAB_M1_INV: [[f32; 3]; 3] = [")?;
    for row in &primary::oklab::M1_INV {
        writeln!(out, "        [{} as f32, {} as f32, {} as f32],", fmt_f64(row[0]), fmt_f64(row[1]), fmt_f64(row[2]))?;
    }
    writeln!(out, "    ];")?;
    writeln!(out)?;
    writeln!(out, "    /// OKLab M2_inv: Lab → LMS'")?;
    writeln!(out, "    pub const OKLAB_M2_INV: [[f32; 3]; 3] = [")?;
    for row in &primary::oklab::M2_INV {
        writeln!(out, "        [{} as f32, {} as f32, {} as f32],", fmt_f64(row[0]), fmt_f64(row[1]), fmt_f64(row[2]))?;
    }
    writeln!(out, "    ];")?;
    writeln!(out)?;

    // Y'CbCr constants
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // Y'CbCr CONSTANTS")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    /// BT.709 luma coefficients")?;
    writeln!(out, "    pub const YCBCR_KR: f32 = {} as f32;", fmt_f64(primary::ycbcr_bt709::KR))?;
    writeln!(out, "    pub const YCBCR_KG: f32 = {} as f32;", fmt_f64(primary::ycbcr_bt709::KG))?;
    writeln!(out, "    pub const YCBCR_KB: f32 = {} as f32;", fmt_f64(primary::ycbcr_bt709::KB))?;
    writeln!(out, "    pub const YCBCR_CB_SCALE: f32 = {} as f32;", fmt_f64(cb_scale))?;
    writeln!(out, "    pub const YCBCR_CR_SCALE: f32 = {} as f32;", fmt_f64(cr_scale))?;
    writeln!(out)?;
    write_matrix_f32(&mut out, "RGB_TO_YCBCR", &ycbcr_forward)?;
    write_matrix_f32(&mut out, "YCBCR_TO_RGB", &ycbcr_inverse)?;
    writeln!(out)?;
    writeln!(out, "    /// BT.601 luma coefficients")?;
    writeln!(out, "    pub const YCBCR_601_KR: f32 = 0.299;")?;
    writeln!(out, "    pub const YCBCR_601_KG: f32 = 0.587;")?;
    writeln!(out, "    pub const YCBCR_601_KB: f32 = 0.114;")?;
    writeln!(out)?;

    // Bit depth
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out, "    // BIT DEPTH")?;
    writeln!(out, "    // -------------------------------------------------------------------------")?;
    writeln!(out)?;
    writeln!(out, "    pub const UINT8_MAX: f32 = 255.0;")?;
    writeln!(out, "    pub const UINT16_MAX: f32 = 65535.0;")?;

    writeln!(out, "}}")?;
    writeln!(out)?;

    // Tests
    writeln!(out, "// =============================================================================")?;
    writeln!(out, "// TESTS")?;
    writeln!(out, "// =============================================================================")?;
    writeln!(out)?;
    writeln!(out, "#[cfg(test)]")?;
    writeln!(out, "mod tests {{")?;
    writeln!(out, "    use super::*;")?;
    writeln!(out)?;
    writeln!(out, "    fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {{")?;
    writeln!(out, "        let mut r = [[0.0; 3]; 3];")?;
    writeln!(out, "        for i in 0..3 {{")?;
    writeln!(out, "            for j in 0..3 {{")?;
    writeln!(out, "                for k in 0..3 {{")?;
    writeln!(out, "                    r[i][j] += a[i][k] * b[k][j];")?;
    writeln!(out, "                }}")?;
    writeln!(out, "            }}")?;
    writeln!(out, "        }}")?;
    writeln!(out, "        r")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    fn is_identity(m: [[f64; 3]; 3], tol: f64) -> bool {{")?;
    writeln!(out, "        for i in 0..3 {{")?;
    writeln!(out, "            for j in 0..3 {{")?;
    writeln!(out, "                let expected = if i == j {{ 1.0 }} else {{ 0.0 }};")?;
    writeln!(out, "                if (m[i][j] - expected).abs() > tol {{")?;
    writeln!(out, "                    return false;")?;
    writeln!(out, "                }}")?;
    writeln!(out, "            }}")?;
    writeln!(out, "        }}")?;
    writeln!(out, "        true")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_srgb_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(SRGB_TO_XYZ, XYZ_TO_SRGB);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"SRGB matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_apple_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(APPLE_RGB_TO_XYZ, XYZ_TO_APPLE_RGB);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"Apple RGB matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_p3_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(DISPLAY_P3_TO_XYZ, XYZ_TO_DISPLAY_P3);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"Display P3 matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_adobe_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(ADOBE_RGB_TO_XYZ, XYZ_TO_ADOBE_RGB);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"Adobe RGB matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_prophoto_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(PROPHOTO_RGB_TO_XYZ, XYZ_TO_PROPHOTO_RGB);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"ProPhoto RGB matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_rec2020_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(REC2020_TO_XYZ, XYZ_TO_REC2020);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"Rec.2020 matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_ycbcr_matrices_are_inverses() {{")?;
    writeln!(out, "        let product = mat_mul(RGB_TO_YCBCR, YCBCR_TO_RGB);")?;
    writeln!(out, "        assert!(is_identity(product, 1e-10), \"Y'CbCr matrices not inverse\");")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_cielab_epsilon() {{")?;
    writeln!(out, "        let expected = (6.0_f64 / 29.0).powi(3);")?;
    writeln!(out, "        assert!((cielab::EPSILON - expected).abs() < 1e-15);")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_cielab_kappa() {{")?;
    writeln!(out, "        let expected = (29.0_f64 / 6.0).powi(2) / 3.0;")?;
    writeln!(out, "        assert!((cielab::KAPPA - expected).abs() < 1e-12);")?;
    writeln!(out, "    }}")?;
    writeln!(out)?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn test_srgb_white_matches_d65_srgb() {{")?;
    writeln!(out, "        // RGB (1,1,1) maps to white XYZ via matrix row sums.")?;
    writeln!(out, "        // This should exactly match d65_srgb since both are derived from the same matrix.")?;
    writeln!(out, "        let x = SRGB_TO_XYZ[0][0] + SRGB_TO_XYZ[0][1] + SRGB_TO_XYZ[0][2];")?;
    writeln!(out, "        let y = SRGB_TO_XYZ[1][0] + SRGB_TO_XYZ[1][1] + SRGB_TO_XYZ[1][2];")?;
    writeln!(out, "        let z = SRGB_TO_XYZ[2][0] + SRGB_TO_XYZ[2][1] + SRGB_TO_XYZ[2][2];")?;
    writeln!(out, "        assert!((x - d65_srgb::XYZ_X).abs() < 1e-14, \"X: {{}} vs {{}}\", x, d65_srgb::XYZ_X);")?;
    writeln!(out, "        assert!((y - d65_srgb::XYZ_Y).abs() < 1e-14, \"Y: {{}} vs {{}}\", y, d65_srgb::XYZ_Y);")?;
    writeln!(out, "        assert!((z - d65_srgb::XYZ_Z).abs() < 1e-14, \"Z: {{}} vs {{}}\", z, d65_srgb::XYZ_Z);")?;
    writeln!(out, "    }}")?;
    writeln!(out, "}}")?;

    Ok(())
}
