/// Rotation utilities for color correction algorithms.
/// Includes AB plane rotation (LAB) and RGB cube rotation around (1,1,1).

use std::f32::consts::PI;

/// Convert degrees to radians
#[inline]
pub fn deg_to_rad(deg: f32) -> f32 {
    deg * PI / 180.0
}

/// Compute the min/max range for A and B channels after rotation.
/// The AB plane forms a square from -127 to +127 on each axis.
/// After rotation, we need the bounding box of the rotated square.
///
/// Returns: [[a_min, a_max], [b_min, b_max]]
pub fn compute_ab_ranges(theta_deg: f32) -> [[f32; 2]; 2] {
    // Corners of the AB square
    let corners = [
        [-127.0f32, -127.0f32],
        [-127.0, 127.0],
        [127.0, -127.0],
        [127.0, 127.0],
    ];

    let theta_rad = deg_to_rad(theta_deg);
    let cos_t = theta_rad.cos();
    let sin_t = theta_rad.sin();

    // Rotate all corners and find min/max
    let mut a_min = f32::MAX;
    let mut a_max = f32::MIN;
    let mut b_min = f32::MAX;
    let mut b_max = f32::MIN;

    for [a, b] in corners {
        let a_rot = a * cos_t - b * sin_t;
        let b_rot = a * sin_t + b * cos_t;

        a_min = a_min.min(a_rot);
        a_max = a_max.max(a_rot);
        b_min = b_min.min(b_rot);
        b_max = b_max.max(b_rot);
    }

    [[a_min, a_max], [b_min, b_max]]
}

/// Rotate AB channels by theta radians
/// a and b are flat arrays of the same length
pub fn rotate_ab(a: &[f32], b: &[f32], theta_rad: f32) -> (Vec<f32>, Vec<f32>) {
    let cos_t = theta_rad.cos();
    let sin_t = theta_rad.sin();

    let a_rot: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&av, &bv)| av * cos_t - bv * sin_t)
        .collect();

    let b_rot: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&av, &bv)| av * sin_t + bv * cos_t)
        .collect();

    (a_rot, b_rot)
}

/// Compute the min/max range for A and B channels after rotation for Oklab.
/// The Oklab AB plane is roughly -0.5 to +0.5 on each axis.
/// After rotation, we need the bounding box of the rotated square.
///
/// Returns: [[a_min, a_max], [b_min, b_max]]
pub fn compute_oklab_ab_ranges(theta_deg: f32) -> [[f32; 2]; 2] {
    // Corners of the Oklab AB square (using -0.5 to +0.5 range)
    let corners = [
        [-0.5f32, -0.5f32],
        [-0.5, 0.5],
        [0.5, -0.5],
        [0.5, 0.5],
    ];

    let theta_rad = deg_to_rad(theta_deg);
    let cos_t = theta_rad.cos();
    let sin_t = theta_rad.sin();

    // Rotate all corners and find min/max
    let mut a_min = f32::MAX;
    let mut a_max = f32::MIN;
    let mut b_min = f32::MAX;
    let mut b_max = f32::MIN;

    for [a, b] in corners {
        let a_rot = a * cos_t - b * sin_t;
        let b_rot = a * sin_t + b * cos_t;

        a_min = a_min.min(a_rot);
        a_max = a_max.max(a_rot);
        b_min = b_min.min(b_rot);
        b_max = b_max.max(b_rot);
    }

    [[a_min, a_max], [b_min, b_max]]
}

/// Rodrigues' rotation formula for axis (1,1,1)/sqrt(3)
/// Used for RGB cube rotation
pub fn rotation_matrix_around_111(theta: f32) -> [[f32; 3]; 3] {
    let c = theta.cos();
    let s = theta.sin();
    let third = 1.0 / 3.0;
    let sqrt3_inv = 1.0 / 3.0f32.sqrt();

    [
        [
            third + (2.0 * third) * c,
            third - third * c - s * sqrt3_inv,
            third - third * c + s * sqrt3_inv,
        ],
        [
            third - third * c + s * sqrt3_inv,
            third + (2.0 * third) * c,
            third - third * c - s * sqrt3_inv,
        ],
        [
            third - third * c - s * sqrt3_inv,
            third - third * c + s * sqrt3_inv,
            third + (2.0 * third) * c,
        ],
    ]
}

/// Rotate RGB values around the gray axis by theta radians
/// rgb is a flat array of interleaved RGB values
pub fn rotate_rgb(rgb: &[f32], theta: f32) -> Vec<f32> {
    let rot = rotation_matrix_around_111(theta);
    let pixels = rgb.len() / 3;
    let mut result = vec![0.0f32; rgb.len()];

    for i in 0..pixels {
        let idx = i * 3;
        let r = rgb[idx];
        let g = rgb[idx + 1];
        let b = rgb[idx + 2];

        result[idx] = rot[0][0] * r + rot[0][1] * g + rot[0][2] * b;
        result[idx + 1] = rot[1][0] * r + rot[1][1] * g + rot[1][2] * b;
        result[idx + 2] = rot[2][0] * r + rot[2][1] * g + rot[2][2] * b;
    }

    result
}

/// Compute channel ranges after RGB rotation for perceptual scaling
/// perceptual_scale: optional [r, g, b] scale factors
/// Returns [[r_min, r_max], [g_min, g_max], [b_min, b_max]]
pub fn compute_rgb_channel_ranges(
    theta_deg: f32,
    perceptual_scale: Option<[f32; 3]>,
) -> [[f32; 2]; 3] {
    let scale = perceptual_scale.unwrap_or([1.0, 1.0, 1.0]);

    // 8 corners of the unit cube, scaled by perceptual factors
    let mut corners = Vec::with_capacity(8);
    for r in [0.0f32, 1.0] {
        for g in [0.0f32, 1.0] {
            for b in [0.0f32, 1.0] {
                corners.push([r * scale[0], g * scale[1], b * scale[2]]);
            }
        }
    }

    // Rotate all corners
    let theta_rad = deg_to_rad(theta_deg);
    let rot = rotation_matrix_around_111(theta_rad);

    let mut ranges = [[f32::MAX, f32::MIN]; 3];

    for corner in corners {
        let rotated = [
            rot[0][0] * corner[0] + rot[0][1] * corner[1] + rot[0][2] * corner[2],
            rot[1][0] * corner[0] + rot[1][1] * corner[1] + rot[1][2] * corner[2],
            rot[2][0] * corner[0] + rot[2][1] * corner[1] + rot[2][2] * corner[2],
        ];

        for c in 0..3 {
            ranges[c][0] = ranges[c][0].min(rotated[c]);
            ranges[c][1] = ranges[c][1].max(rotated[c]);
        }
    }

    ranges
}

/// Perceptual luminance weights (Rec.709)
pub const LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Compute perceptual scale factors
/// Compress less-important channels, match green precisely
pub fn perceptual_scale_factors() -> [f32; 3] {
    let max_weight = LUMA_WEIGHTS[1]; // Green is max
    [
        LUMA_WEIGHTS[0] / max_weight,
        1.0,
        LUMA_WEIGHTS[2] / max_weight,
    ]
}

/// Apply perceptual scaling to RGB
pub fn perceptual_scale_rgb(rgb: &[f32], scale: [f32; 3]) -> Vec<f32> {
    let pixels = rgb.len() / 3;
    let mut result = vec![0.0f32; rgb.len()];

    for i in 0..pixels {
        let idx = i * 3;
        result[idx] = rgb[idx] * scale[0];
        result[idx + 1] = rgb[idx + 1] * scale[1];
        result[idx + 2] = rgb[idx + 2] * scale[2];
    }

    result
}

/// Remove perceptual scaling from RGB
pub fn perceptual_unscale_rgb(rgb: &[f32], scale: [f32; 3]) -> Vec<f32> {
    let pixels = rgb.len() / 3;
    let mut result = vec![0.0f32; rgb.len()];

    for i in 0..pixels {
        let idx = i * 3;
        result[idx] = rgb[idx] / scale[0];
        result[idx + 1] = rgb[idx + 1] / scale[1];
        result[idx + 2] = rgb[idx + 2] / scale[2];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ab_ranges_zero_rotation() {
        let ranges = compute_ab_ranges(0.0);
        assert!((ranges[0][0] - (-127.0)).abs() < 0.01);
        assert!((ranges[0][1] - 127.0).abs() < 0.01);
        assert!((ranges[1][0] - (-127.0)).abs() < 0.01);
        assert!((ranges[1][1] - 127.0).abs() < 0.01);
    }

    #[test]
    fn test_rotate_ab_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (a_rot, b_rot) = rotate_ab(&a, &b, 0.0);
        for (i, (&orig, &rot)) in a.iter().zip(a_rot.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-6,
                "A mismatch at {}: {} vs {}",
                i,
                orig,
                rot
            );
        }
        for (i, (&orig, &rot)) in b.iter().zip(b_rot.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-6,
                "B mismatch at {}: {} vs {}",
                i,
                orig,
                rot
            );
        }
    }

    #[test]
    fn test_rgb_rotation_identity() {
        let rgb = vec![0.5, 0.5, 0.5]; // Gray should remain gray
        let rotated = rotate_rgb(&rgb, deg_to_rad(45.0));
        // Gray axis is invariant under rotation
        assert!((rotated[0] - 0.5).abs() < 1e-5);
        assert!((rotated[1] - 0.5).abs() < 1e-5);
        assert!((rotated[2] - 0.5).abs() < 1e-5);
    }
}
