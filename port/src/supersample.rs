use crate::pixel::Pixel4;

/// Expand a box-space image into tent-space.
///
/// Box space: each pixel represents the integral (total light) over a unit square.
/// Tent space: a piecewise bilinear surface where we can interpolate anywhere.
///
/// The transformation is (W, H) -> (2W+1, 2H+1):
/// - Odd coordinates (1,1), (1,3), (3,1), etc. are the "center" points
///   directly corresponding to original pixel locations
/// - Even coordinates are interpolated edges and corners between pixels
///
/// The center values are adjusted so that integrating the tent surface
/// over each pixel's unit square recovers the original box value exactly.
pub fn tent_expand(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
) -> (Vec<Pixel4>, usize, usize) {
    let dst_width = src_width * 2 + 1;
    let dst_height = src_height * 2 + 1;
    let mut dst = vec![Pixel4::new(0.0, 0.0, 0.0, 0.0); dst_width * dst_height];

    // Clamp-to-edge boundary condition for source lookups
    let get_src = |x: isize, y: isize| -> Pixel4 {
        let cx = x.clamp(0, (src_width - 1) as isize) as usize;
        let cy = y.clamp(0, (src_height - 1) as isize) as usize;
        src[cy * src_width + cx]
    };

    // =========================================================================
    // PASS 1: Compute all grid points via linear interpolation
    // =========================================================================
    //
    // Tent-space grid layout (for a 2x2 source):
    //
    //     0   1   2   3   4   (dx)
    //   +---+---+---+---+---+
    // 0 | C | E | C | E | C |   C = corner (even,even) - interp from 4 pixels
    //   +---+---+---+---+---+
    // 1 | E | P | E | P | E |   E = edge (mixed parity) - interp from 2 pixels
    //   +---+---+---+---+---+
    // 2 | C | E | C | E | C |   P = pixel center (odd,odd) - direct from source
    //   +---+---+---+---+---+
    // 3 | E | P | E | P | E |
    //   +---+---+---+---+---+
    // 4 | C | E | C | E | C |
    //   +---+---+---+---+---+
    // (dy)

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let idx = dy * dst_width + dx;

            let x_odd = dx % 2 == 1;
            let y_odd = dy % 2 == 1;

            if x_odd && y_odd {
                // Pixel center: directly maps to a source pixel
                let sx = (dx - 1) / 2;
                let sy = (dy - 1) / 2;
                dst[idx] = get_src(sx as isize, sy as isize);
            } else if x_odd && !y_odd {
                // Horizontal edge: interpolate between pixels above and below
                let sx = (dx - 1) / 2;
                let sy_above = (dy as isize / 2) - 1;
                let sy_below = dy as isize / 2;

                let p_above = get_src(sx as isize, sy_above);
                let p_below = get_src(sx as isize, sy_below);

                dst[idx] = lerp_pixel(&p_above, &p_below, 0.5);
            } else if !x_odd && y_odd {
                // Vertical edge: interpolate between pixels left and right
                let sy = (dy - 1) / 2;
                let sx_left = (dx as isize / 2) - 1;
                let sx_right = dx as isize / 2;

                let p_left = get_src(sx_left, sy as isize);
                let p_right = get_src(sx_right, sy as isize);

                dst[idx] = lerp_pixel(&p_left, &p_right, 0.5);
            } else {
                // Corner: bilinear interpolation from 4 surrounding pixels
                let sx_left = (dx as isize / 2) - 1;
                let sx_right = dx as isize / 2;
                let sy_above = (dy as isize / 2) - 1;
                let sy_below = dy as isize / 2;

                let p_tl = get_src(sx_left, sy_above);
                let p_tr = get_src(sx_right, sy_above);
                let p_bl = get_src(sx_left, sy_below);
                let p_br = get_src(sx_right, sy_below);

                let top = lerp_pixel(&p_tl, &p_tr, 0.5);
                let bottom = lerp_pixel(&p_bl, &p_br, 0.5);
                dst[idx] = lerp_pixel(&top, &bottom, 0.5);
            }
        }
    }

    // =========================================================================
    // PASS 2: Adjust center values for volume preservation
    // =========================================================================
    //
    // The integral of a bilinear surface over a unit square with values at
    // the center (M), four edges (E), and four corners (C) is:
    //
    //     volume = (1/4)*M + (1/8)*sum(E) + (1/16)*sum(C)
    //
    // We want this volume to equal the original pixel value V.
    // After pass 1, the center M equals V, but that doesn't give us
    // volume = V because the edges and corners also contribute.
    //
    // Solving for the adjusted center M' such that volume = V:
    //
    //     V = (1/4)*M' + (1/8)*sum(E) + (1/16)*sum(C)
    //     M' = 4*V - (1/2)*sum(E) - (1/4)*sum(C)
    //
    // Note: M' can go negative or exceed 1.0 for high-contrast edges.
    // This is mathematically correct for volume preservation.

    for sy in 0..src_height {
        for sx in 0..src_width {
            // Map source pixel (sx, sy) to its center in tent space
            let dx = sx * 2 + 1;
            let dy = sy * 2 + 1;
            let mid_idx = dy * dst_width + dx;

            let original_value = src[sy * src_width + sx];

            // Gather the 4 corners of this pixel's integration domain
            let corner_tl = dst[(dy - 1) * dst_width + (dx - 1)];
            let corner_tr = dst[(dy - 1) * dst_width + (dx + 1)];
            let corner_bl = dst[(dy + 1) * dst_width + (dx - 1)];
            let corner_br = dst[(dy + 1) * dst_width + (dx + 1)];
            let corner_sum = add4_pixels(&corner_tl, &corner_tr, &corner_bl, &corner_br);

            // Gather the 4 edges of this pixel's integration domain
            let edge_top = dst[(dy - 1) * dst_width + dx];
            let edge_bottom = dst[(dy + 1) * dst_width + dx];
            let edge_left = dst[dy * dst_width + (dx - 1)];
            let edge_right = dst[dy * dst_width + (dx + 1)];
            let edge_sum = add4_pixels(&edge_top, &edge_bottom, &edge_left, &edge_right);

            // Solve for adjusted center: M' = 4V - 0.5*E - 0.25*C
            let adjusted = Pixel4::new(
                4.0 * original_value[0] - 0.5 * edge_sum[0] - 0.25 * corner_sum[0],
                4.0 * original_value[1] - 0.5 * edge_sum[1] - 0.25 * corner_sum[1],
                4.0 * original_value[2] - 0.5 * edge_sum[2] - 0.25 * corner_sum[2],
                4.0 * original_value[3] - 0.5 * edge_sum[3] - 0.25 * corner_sum[3],
            );

            dst[mid_idx] = adjusted;
        }
    }

    (dst, dst_width, dst_height)
}

/// Contract a tent-space image back to box-space.
///
/// This computes the volume under the bilinear surface for each pixel's
/// unit square, recovering the original box values exactly (within
/// floating-point precision) if applied after tent_expand.
///
/// The transformation is (2W+1, 2H+1) -> (W, H).
pub fn tent_contract(
    src: &[Pixel4],
    src_width: usize,
    src_height: usize,
) -> (Vec<Pixel4>, usize, usize) {
    debug_assert!(src_width % 2 == 1, "Source width must be odd (2N+1)");
    debug_assert!(src_height % 2 == 1, "Source height must be odd (2M+1)");

    let dst_width = (src_width - 1) / 2;
    let dst_height = (src_height - 1) / 2;
    let mut dst = vec![Pixel4::new(0.0, 0.0, 0.0, 0.0); dst_width * dst_height];

    // For each output pixel, integrate the bilinear surface over its unit square.
    //
    // Integration weights for bilinear surface over unit square:
    //
    //       1/16 ----(1/8)---- 1/16
    //         |        |        |
    //         |   Q1   |   Q2   |      Each quadrant contributes 1/4 of the
    //         |        |        |      total area. Within each quadrant,
    //      (1/8)-----(1/4)----(1/8)    the bilinear weights sum to 1.
    //         |        |        |
    //         |   Q3   |   Q4   |      Center: in all 4 quadrants -> 4/16 = 1/4
    //         |        |        |      Edge: in 2 quadrants -> 2/16 = 1/8
    //       1/16 ----(1/8)---- 1/16    Corner: in 1 quadrant -> 1/16

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            // Map output pixel (dx, dy) to tent-space center
            let sx = dx * 2 + 1;
            let sy = dy * 2 + 1;

            // Gather the 9 points of the integration stencil
            let mid = src[sy * src_width + sx];

            let edge_top = src[(sy - 1) * src_width + sx];
            let edge_bottom = src[(sy + 1) * src_width + sx];
            let edge_left = src[sy * src_width + (sx - 1)];
            let edge_right = src[sy * src_width + (sx + 1)];

            let corner_tl = src[(sy - 1) * src_width + (sx - 1)];
            let corner_tr = src[(sy - 1) * src_width + (sx + 1)];
            let corner_bl = src[(sy + 1) * src_width + (sx - 1)];
            let corner_br = src[(sy + 1) * src_width + (sx + 1)];

            let edge_sum = add4_pixels(&edge_top, &edge_bottom, &edge_left, &edge_right);
            let corner_sum = add4_pixels(&corner_tl, &corner_tr, &corner_bl, &corner_br);

            // Volume = (1/4)*center + (1/8)*edges + (1/16)*corners
            let result = Pixel4::new(
                0.25 * mid[0] + 0.125 * edge_sum[0] + 0.0625 * corner_sum[0],
                0.25 * mid[1] + 0.125 * edge_sum[1] + 0.0625 * corner_sum[1],
                0.25 * mid[2] + 0.125 * edge_sum[2] + 0.0625 * corner_sum[2],
                0.25 * mid[3] + 0.125 * edge_sum[3] + 0.0625 * corner_sum[3],
            );

            dst[dy * dst_width + dx] = result;
        }
    }

    (dst, dst_width, dst_height)
}

/// Returns the tent-space dimensions for a given box-space image.
pub fn supersample_target_dimensions(width: usize, height: usize) -> (usize, usize) {
    (width * 2 + 1, height * 2 + 1)
}

/// Returns the coordinate offset needed when sampling in tent space.
///
/// In box space, pixel (0,0) is centered at (0.5, 0.5).
/// In tent space, the corresponding point is at (1, 1).
/// The scale factor is 2x, so: tent_coord = box_coord * 2 + offset
/// where offset = (0.5, 0.5) to align the centers.
pub fn supersample_scaling_offset() -> (f32, f32) {
    (0.5, 0.5)
}

fn lerp_pixel(a: &Pixel4, b: &Pixel4, t: f32) -> Pixel4 {
    Pixel4::new(
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    )
}

fn add4_pixels(a: &Pixel4, b: &Pixel4, c: &Pixel4, d: &Pixel4) -> Pixel4 {
    Pixel4::new(
        a[0] + b[0] + c[0] + d[0],
        a[1] + b[1] + c[1] + d[1],
        a[2] + b[2] + c[2] + d[2],
        a[3] + b[3] + c[3] + d[3],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_dimensions() {
        let src = vec![Pixel4::new(1.0, 1.0, 1.0, 1.0); 4];
        let (dst, w, h) = tent_expand(&src, 2, 2);
        assert_eq!(w, 5);
        assert_eq!(h, 5);
        assert_eq!(dst.len(), 25);
    }

    #[test]
    fn test_contract_dimensions() {
        let src = vec![Pixel4::new(1.0, 1.0, 1.0, 1.0); 25];
        let (dst, w, h) = tent_contract(&src, 5, 5);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(dst.len(), 4);
    }

    #[test]
    fn test_roundtrip_uniform() {
        // Uniform image should roundtrip exactly
        let value = 0.5;
        let src = vec![Pixel4::new(value, value, value, 1.0); 9];
        let (expanded, w, h) = tent_expand(&src, 3, 3);
        let (result, rw, rh) = tent_contract(&expanded, w, h);

        assert_eq!(rw, 3);
        assert_eq!(rh, 3);
        for p in &result {
            assert!((p[0] - value).abs() < 1e-5, "Expected {}, got {}", value, p[0]);
        }
    }

    #[test]
    fn test_roundtrip_nonuniform() {
        // Non-uniform image should roundtrip exactly (per-pixel check)
        let src = vec![
            Pixel4::new(0.1, 0.2, 0.3, 1.0),
            Pixel4::new(0.9, 0.1, 0.5, 1.0),
            Pixel4::new(0.3, 0.8, 0.2, 1.0),
            Pixel4::new(0.6, 0.4, 0.7, 1.0),
        ];

        let (expanded, w, h) = tent_expand(&src, 2, 2);
        let (result, _, _) = tent_contract(&expanded, w, h);

        for i in 0..4 {
            for c in 0..4 {
                assert!(
                    (result[i][c] - src[i][c]).abs() < 1e-5,
                    "Pixel {} channel {} mismatch: expected {}, got {}",
                    i, c, src[i][c], result[i][c]
                );
            }
        }
    }

    #[test]
    fn test_roundtrip_preserves_energy() {
        // Total energy should be preserved
        let src = vec![
            Pixel4::new(0.0, 0.0, 0.0, 1.0),
            Pixel4::new(1.0, 0.5, 0.0, 1.0),
            Pixel4::new(0.5, 1.0, 0.5, 1.0),
            Pixel4::new(0.2, 0.3, 1.0, 1.0),
        ];

        let src_energy: f32 = src.iter().map(|p| p[0] + p[1] + p[2]).sum();

        let (expanded, w, h) = tent_expand(&src, 2, 2);
        let (result, _, _) = tent_contract(&expanded, w, h);

        let result_energy: f32 = result.iter().map(|p| p[0] + p[1] + p[2]).sum();

        assert!(
            (src_energy - result_energy).abs() < 1e-4,
            "Energy mismatch: src={}, result={}",
            src_energy,
            result_energy
        );
    }

    #[test]
    fn test_supersample_target() {
        assert_eq!(supersample_target_dimensions(100, 50), (201, 101));
        assert_eq!(supersample_target_dimensions(1, 1), (3, 3));
    }
}
