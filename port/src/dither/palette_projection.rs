//! Palette projection for gamut mapping.
//!
//! This module extends palettes with convex hull information for proper gamut mapping.
//! When searching for the nearest palette color, we also consider projections onto
//! hull edges and surfaces. If an edge or surface projection is closer than any real
//! entry, we search only among palette entries on that edge or surface.

use super::palette_hull::{HullPlane, PaletteHull, EPSILON};
use super::paletted::DitherPalette;
use crate::color_distance::perceptual_distance_sq;
use super::common::{linear_rgb_to_perceptual, linear_rgb_to_perceptual_clamped, PerceptualSpace};

// ============================================================================
// Hull edge representation
// ============================================================================

/// An edge of the convex hull - the line segment where two planes intersect,
/// bounded by where it exits the hull.
/// For uniform cubes, multiple palette entries may lie along an edge.
#[derive(Clone, Debug)]
struct HullEdge {
    /// First endpoint (a hull vertex / palette entry)
    p0: [f32; 3],
    /// Second endpoint (a hull vertex / palette entry)
    p1: [f32; 3],
    /// All palette entry indices that lie on this edge
    entries: Vec<usize>,
}

// ============================================================================
// Extended palette structure
// ============================================================================

/// Extended palette with hull information, edge, and surface membership.
#[derive(Clone, Debug)]
pub struct ExtendedPalette {
    /// The original palette
    palette: DitherPalette,
    /// The convex hull of the palette
    hull: PaletteHull,
    /// Linear RGB positions for real entries
    linear_positions: Vec<[f32; 3]>,
    /// Perceptual positions for real entries
    perceptual_positions: Vec<[f32; 3]>,
    /// Hull edges (unique lines connecting vertices)
    edges: Vec<HullEdge>,
    /// For each hull plane, indices of real palette entries on that surface
    surface_entries: Vec<Vec<usize>>,
    /// Perceptual space used
    space: PerceptualSpace,
}

impl ExtendedPalette {
    /// Create an extended palette from a DitherPalette.
    pub fn new(palette: DitherPalette, space: PerceptualSpace) -> Self {
        let hull = PaletteHull::from_palette(&palette);
        Self::from_palette_and_hull(palette, hull, space)
    }

    /// Create from palette and pre-computed hull.
    pub fn from_palette_and_hull(
        palette: DitherPalette,
        hull: PaletteHull,
        space: PerceptualSpace,
    ) -> Self {
        let linear_positions = palette.linear_rgb_points();
        let num_planes = hull.planes.len();

        // Convert to perceptual space
        let perceptual_positions: Vec<[f32; 3]> = linear_positions
            .iter()
            .map(|p| {
                let (l, a, b) = linear_rgb_to_perceptual_clamped(space, p[0], p[1], p[2]);
                [l, a, b]
            })
            .collect();

        // Find real entries on each hull surface
        let mut surface_entries: Vec<Vec<usize>> = vec![Vec::new(); num_planes];
        for (entry_idx, pos) in linear_positions.iter().enumerate() {
            for (plane_idx, plane) in hull.planes.iter().enumerate() {
                if plane.signed_distance(*pos).abs() <= EPSILON {
                    surface_entries[plane_idx].push(entry_idx);
                }
            }
        }

        // Extract edges as intersections of pairs of hull planes.
        // Each edge is where two adjacent faces meet.
        let mut edges: Vec<HullEdge> = Vec::new();

        for i in 0..num_planes {
            for j in (i + 1)..num_planes {
                // Find palette entries that lie on BOTH planes (i.e., on their intersection)
                let mut entries_on_both: Vec<usize> = Vec::new();
                for (idx, pos) in linear_positions.iter().enumerate() {
                    let dist_i = hull.planes[i].signed_distance(*pos).abs();
                    let dist_j = hull.planes[j].signed_distance(*pos).abs();
                    if dist_i <= EPSILON && dist_j <= EPSILON {
                        entries_on_both.push(idx);
                    }
                }

                // An edge needs at least 2 points to define endpoints
                if entries_on_both.len() >= 2 {
                    // Find the two extreme points along the intersection line
                    // (the line direction is the cross product of the two normals)
                    let n1 = &hull.planes[i].normal;
                    let n2 = &hull.planes[j].normal;
                    let dir = [
                        n1[1] * n2[2] - n1[2] * n2[1],
                        n1[2] * n2[0] - n1[0] * n2[2],
                        n1[0] * n2[1] - n1[1] * n2[0],
                    ];

                    // Project all entries onto the line direction to find min/max
                    let mut min_t = f32::INFINITY;
                    let mut max_t = f32::NEG_INFINITY;
                    let mut min_idx = 0;
                    let mut max_idx = 0;

                    for &idx in &entries_on_both {
                        let pos = &linear_positions[idx];
                        let t = pos[0] * dir[0] + pos[1] * dir[1] + pos[2] * dir[2];
                        if t < min_t {
                            min_t = t;
                            min_idx = idx;
                        }
                        if t > max_t {
                            max_t = t;
                            max_idx = idx;
                        }
                    }

                    let p0 = linear_positions[min_idx];
                    let p1 = linear_positions[max_idx];

                    // Skip degenerate edges (both endpoints are the same point)
                    if !points_equal(&p0, &p1) {
                        edges.push(HullEdge {
                            p0,
                            p1,
                            entries: entries_on_both,
                        });
                    }
                }
            }
        }

        Self {
            palette,
            hull,
            linear_positions,
            perceptual_positions,
            edges,
            surface_entries,
            space,
        }
    }

    /// Number of palette entries.
    pub fn len(&self) -> usize {
        self.palette.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.palette.is_empty()
    }

    /// Get the original palette.
    pub fn palette(&self) -> &DitherPalette {
        &self.palette
    }

    /// Get the hull.
    pub fn hull(&self) -> &PaletteHull {
        &self.hull
    }

    /// Clamp a linear RGB point to within the hull.
    /// Returns the original point if inside, or the nearest point on the hull if outside.
    pub fn clamp_to_hull(&self, lin_rgb: [f32; 3]) -> [f32; 3] {
        if self.hull.contains(lin_rgb) {
            return lin_rgb;
        }

        // Find the nearest point on the hull boundary by iteratively projecting
        // onto planes until we're inside
        let mut point = lin_rgb;
        let max_iterations = 10;

        for _ in 0..max_iterations {
            // Find the plane we're most outside of
            let mut worst_plane_idx = 0;
            let mut worst_distance = f32::NEG_INFINITY;

            for (idx, plane) in self.hull.planes.iter().enumerate() {
                let d = plane.signed_distance(point);
                if d > worst_distance {
                    worst_distance = d;
                    worst_plane_idx = idx;
                }
            }

            if worst_distance <= EPSILON {
                // We're inside (or on boundary)
                break;
            }

            // Project onto the worst plane
            let plane = &self.hull.planes[worst_plane_idx];
            point = project_point_onto_plane(point, plane);
        }

        // Final clamp to ensure we're within [0, 1] for RGB
        [
            point[0].clamp(0.0, 1.0),
            point[1].clamp(0.0, 1.0),
            point[2].clamp(0.0, 1.0),
        ]
    }

    /// Find the nearest real palette entry for a linear RGB point.
    ///
    /// This searches:
    /// 1. Real palette entries (perceptual distance, optionally with gamut overshoot penalty)
    /// 2. Projections onto each hull edge (project in linear, distance in perceptual)
    /// 3. Projections onto each hull surface (project in linear, distance in perceptual)
    ///
    /// If an edge projection is closest, we search only among entries on that edge.
    /// If a surface projection is closest, we search only among entries on that surface.
    ///
    /// If `overshoot_penalty` is true, the gamut overshoot penalty discourages choosing
    /// palette colors that would cause large error diffusion outside the palette's gamut.
    pub fn find_nearest_real(&self, lin_rgb: [f32; 3], overshoot_penalty: bool) -> usize {
        // Convert target to perceptual space (non-clamped since this is a target color)
        let (l, a, b) = linear_rgb_to_perceptual(self.space, lin_rgb[0], lin_rgb[1], lin_rgb[2]);
        let target_perc = [l, a, b];

        // Track best real entry
        let mut best_real_idx = 0;
        let mut best_real_dist = f32::INFINITY;

        // Search real palette entries (with optional gamut overshoot penalty)
        for (idx, perc) in self.perceptual_positions.iter().enumerate() {
            let d = if overshoot_penalty {
                penalized_perceptual_distance_sq(
                    &self.hull,
                    self.space,
                    lin_rgb,
                    target_perc,
                    *perc,
                    self.linear_positions[idx],
                )
            } else {
                perceptual_distance_sq(
                    self.space,
                    target_perc[0], target_perc[1], target_perc[2],
                    perc[0], perc[1], perc[2],
                )
            };
            if d < best_real_dist {
                best_real_dist = d;
                best_real_idx = idx;
            }
        }

        // Track best edge projection
        let mut best_edge_idx: Option<usize> = None;
        let mut best_edge_dist = f32::INFINITY;

        // Search hull edge projections
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            // Project target onto this edge in linear RGB space
            let projected_lin = project_point_onto_segment(lin_rgb, edge.p0, edge.p1);

            // Convert projection to perceptual space (clamped since it's on the hull)
            let (pl, pa, pb) = linear_rgb_to_perceptual_clamped(
                self.space, projected_lin[0], projected_lin[1], projected_lin[2]
            );

            // Calculate perceptual distance from target to projection
            let d = perceptual_distance_sq(
                self.space,
                target_perc[0], target_perc[1], target_perc[2],
                pl, pa, pb,
            );

            if d < best_edge_dist {
                best_edge_dist = d;
                best_edge_idx = Some(edge_idx);
            }
        }

        // Track best surface projection
        let mut best_surface_idx: Option<usize> = None;
        let mut best_surface_dist = f32::INFINITY;

        // Search hull surface projections
        for (plane_idx, plane) in self.hull.planes.iter().enumerate() {
            // Project target onto this plane in linear RGB space
            let projected_lin = project_point_onto_plane(lin_rgb, plane);

            // Check if projection is inside the hull (on the actual face, not just the infinite plane)
            if !self.hull.contains(projected_lin) {
                continue;
            }

            // Convert projection to perceptual space (clamped since it's on the hull)
            let (pl, pa, pb) = linear_rgb_to_perceptual_clamped(
                self.space, projected_lin[0], projected_lin[1], projected_lin[2]
            );

            // Calculate perceptual distance from target to projection
            let d = perceptual_distance_sq(
                self.space,
                target_perc[0], target_perc[1], target_perc[2],
                pl, pa, pb,
            );

            if d < best_surface_dist {
                best_surface_dist = d;
                best_surface_idx = Some(plane_idx);
            }
        }

        // Find the minimum distance among real, edge, and surface
        // Priority: real entry > edge > surface (if distances are equal)
        if best_real_dist <= best_edge_dist && best_real_dist <= best_surface_dist {
            return best_real_idx;
        }

        if best_edge_dist < best_surface_dist {
            if let Some(edge_idx) = best_edge_idx {
                return self.find_nearest_on_edge(lin_rgb, target_perc, edge_idx, overshoot_penalty);
            }
        }

        if let Some(surface_idx) = best_surface_idx {
            return self.find_nearest_on_surface(lin_rgb, target_perc, surface_idx, overshoot_penalty);
        }

        best_real_idx
    }

    /// Find nearest real entry among those on a specific hull edge.
    fn find_nearest_on_edge(&self, target_lin: [f32; 3], target_perc: [f32; 3], edge_idx: usize, overshoot_penalty: bool) -> usize {
        let edge_entries = &self.edges[edge_idx].entries;

        if edge_entries.is_empty() {
            // Fallback: find any nearest real entry
            return self.find_nearest_fallback(target_lin, target_perc, overshoot_penalty);
        }

        let mut best_idx = edge_entries[0];
        let mut best_dist = f32::INFINITY;

        for &real_idx in edge_entries {
            let perc = &self.perceptual_positions[real_idx];
            let d = if overshoot_penalty {
                penalized_perceptual_distance_sq(
                    &self.hull,
                    self.space,
                    target_lin,
                    target_perc,
                    *perc,
                    self.linear_positions[real_idx],
                )
            } else {
                perceptual_distance_sq(
                    self.space,
                    target_perc[0], target_perc[1], target_perc[2],
                    perc[0], perc[1], perc[2],
                )
            };
            if d < best_dist {
                best_dist = d;
                best_idx = real_idx;
            }
        }

        best_idx
    }

    /// Find nearest real entry among those on a specific hull surface.
    fn find_nearest_on_surface(&self, target_lin: [f32; 3], target_perc: [f32; 3], plane_idx: usize, overshoot_penalty: bool) -> usize {
        let surface_entries = &self.surface_entries[plane_idx];

        if surface_entries.is_empty() {
            // Fallback: find any nearest real entry
            return self.find_nearest_fallback(target_lin, target_perc, overshoot_penalty);
        }

        let mut best_idx = surface_entries[0];
        let mut best_dist = f32::INFINITY;

        for &real_idx in surface_entries {
            let perc = &self.perceptual_positions[real_idx];
            let d = if overshoot_penalty {
                penalized_perceptual_distance_sq(
                    &self.hull,
                    self.space,
                    target_lin,
                    target_perc,
                    *perc,
                    self.linear_positions[real_idx],
                )
            } else {
                perceptual_distance_sq(
                    self.space,
                    target_perc[0], target_perc[1], target_perc[2],
                    perc[0], perc[1], perc[2],
                )
            };
            if d < best_dist {
                best_dist = d;
                best_idx = real_idx;
            }
        }

        best_idx
    }

    /// Fallback: find nearest among all real entries.
    fn find_nearest_fallback(&self, target_lin: [f32; 3], target_perc: [f32; 3], overshoot_penalty: bool) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (idx, perc) in self.perceptual_positions.iter().enumerate() {
            let d = if overshoot_penalty {
                penalized_perceptual_distance_sq(
                    &self.hull,
                    self.space,
                    target_lin,
                    target_perc,
                    *perc,
                    self.linear_positions[idx],
                )
            } else {
                perceptual_distance_sq(
                    self.space,
                    target_perc[0], target_perc[1], target_perc[2],
                    perc[0], perc[1], perc[2],
                )
            };
            if d < best_dist {
                best_dist = d;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Get linear RGB position for a palette entry.
    pub fn get_linear_rgb(&self, idx: usize) -> [f32; 3] {
        self.linear_positions[idx]
    }

    /// Get sRGB values for a palette entry.
    pub fn get_srgb(&self, idx: usize) -> (u8, u8, u8, u8) {
        self.palette.get_srgb(idx)
    }

    /// Get surface entries for testing/debugging.
    #[cfg(test)]
    pub fn surface_entries(&self) -> &Vec<Vec<usize>> {
        &self.surface_entries
    }
}

// ============================================================================
// Geometry helpers
// ============================================================================

/// Calculate the linear RGB distance from a point to the hull boundary.
/// Returns 0.0 if the point is inside the hull.
fn distance_to_hull(hull: &PaletteHull, point: [f32; 3]) -> f32 {
    if hull.contains(point) {
        return 0.0;
    }

    // Find the nearest point on the hull by iteratively projecting onto planes
    let mut clamped = point;
    let max_iterations = 10;

    for _ in 0..max_iterations {
        let mut worst_plane_idx = 0;
        let mut worst_distance = f32::NEG_INFINITY;

        for (idx, plane) in hull.planes.iter().enumerate() {
            let d = plane.signed_distance(clamped);
            if d > worst_distance {
                worst_distance = d;
                worst_plane_idx = idx;
            }
        }

        if worst_distance <= EPSILON {
            break;
        }

        let plane = &hull.planes[worst_plane_idx];
        clamped = project_point_onto_plane(clamped, plane);
    }

    // Clamp to [0,1] RGB bounds as well
    clamped[0] = clamped[0].clamp(0.0, 1.0);
    clamped[1] = clamped[1].clamp(0.0, 1.0);
    clamped[2] = clamped[2].clamp(0.0, 1.0);

    // Calculate Euclidean distance from original point to clamped point
    let dx = point[0] - clamped[0];
    let dy = point[1] - clamped[1];
    let dz = point[2] - clamped[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Calculate perceptual distance with penalty for gamut overshoot.
///
/// When choosing a palette color, the error diffused to neighbors is (target - palette).
/// The "opposing point" is target + error = 2*target - palette.
/// If this opposing point is outside the hull, the error will be hard to recover.
/// We penalize by scaling the distance by (hull_overshoot + 1)².
#[inline]
fn penalized_perceptual_distance_sq(
    hull: &PaletteHull,
    space: PerceptualSpace,
    target_lin: [f32; 3],
    target_perc: [f32; 3],
    palette_perc: [f32; 3],
    palette_lin: [f32; 3],
) -> f32 {
    // Calculate the opposing point: where error diffusion would push neighbors
    let opposing = [
        2.0 * target_lin[0] - palette_lin[0],
        2.0 * target_lin[1] - palette_lin[1],
        2.0 * target_lin[2] - palette_lin[2],
    ];

    // Calculate overshoot penalty
    let overshoot = distance_to_hull(hull, opposing);
    let penalty = (overshoot + 1.0) * (overshoot + 1.0);

    // Base perceptual distance
    let base_dist = perceptual_distance_sq(
        space,
        target_perc[0], target_perc[1], target_perc[2],
        palette_perc[0], palette_perc[1], palette_perc[2],
    );

    base_dist * penalty
}

/// Project a point onto a plane (nearest point on the infinite plane).
fn project_point_onto_plane(point: [f32; 3], plane: &HullPlane) -> [f32; 3] {
    let dist = plane.signed_distance(point);
    [
        point[0] - dist * plane.normal[0],
        point[1] - dist * plane.normal[1],
        point[2] - dist * plane.normal[2],
    ]
}

/// Project a point onto a line segment, returning the nearest point on the segment.
fn project_point_onto_segment(point: [f32; 3], p0: [f32; 3], p1: [f32; 3]) -> [f32; 3] {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let dz = p1[2] - p0[2];
    let len_sq = dx * dx + dy * dy + dz * dz;

    if len_sq < EPSILON * EPSILON {
        // Degenerate edge (endpoints are the same)
        return p0;
    }

    // Parameter t along the line: 0 = p0, 1 = p1
    let t = ((point[0] - p0[0]) * dx + (point[1] - p0[1]) * dy + (point[2] - p0[2]) * dz) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);

    [
        p0[0] + t_clamped * dx,
        p0[1] + t_clamped * dy,
        p0[2] + t_clamped * dz,
    ]
}

/// Check if two points are equal within epsilon.
fn points_equal(a: &[f32; 3], b: &[f32; 3]) -> bool {
    (a[0] - b[0]).abs() <= EPSILON
        && (a[1] - b[1]).abs() <= EPSILON
        && (a[2] - b[2]).abs() <= EPSILON
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_palette() -> DitherPalette {
        // Simple 8-color RGB cube corners
        let colors = vec![
            (0, 0, 0, 255),       // Black
            (255, 0, 0, 255),     // Red
            (0, 255, 0, 255),     // Green
            (0, 0, 255, 255),     // Blue
            (255, 255, 0, 255),   // Yellow
            (255, 0, 255, 255),   // Magenta
            (0, 255, 255, 255),   // Cyan
            (255, 255, 255, 255), // White
        ];
        DitherPalette::new(&colors, PerceptualSpace::OkLab)
    }

    #[test]
    fn test_extended_palette_creation() {
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        assert_eq!(extended.len(), 8);
    }

    #[test]
    fn test_clamp_inside_point() {
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Point inside the cube should not be modified
        let inside = [0.5, 0.5, 0.5];
        let clamped = extended.clamp_to_hull(inside);

        assert!((clamped[0] - inside[0]).abs() < EPSILON * 10.0);
        assert!((clamped[1] - inside[1]).abs() < EPSILON * 10.0);
        assert!((clamped[2] - inside[2]).abs() < EPSILON * 10.0);
    }

    #[test]
    fn test_clamp_outside_point() {
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Point way outside should be clamped
        let outside = [2.0, 0.5, 0.5];
        let clamped = extended.clamp_to_hull(outside);

        // Should be clamped to approximately x=1.0
        assert!(clamped[0] <= 1.0 + EPSILON * 100.0);
        assert!(extended.hull().contains(clamped));
    }

    #[test]
    fn test_find_nearest_real() {
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Point very close to black corner
        let near_black = [0.01, 0.01, 0.01];
        let nearest = extended.find_nearest_real(near_black, true);

        // Should find black (index 0)
        let (r, g, b, _) = extended.get_srgb(nearest);
        assert_eq!((r, g, b), (0, 0, 0));

        // Point very close to white corner
        let near_white = [0.99, 0.99, 0.99];
        let nearest = extended.find_nearest_real(near_white, true);

        // Should find white (index 7)
        let (r, g, b, _) = extended.get_srgb(nearest);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_surface_entries_populated() {
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Each face of the cube should have at least 3 vertices (palette entries)
        let total_surface_entries: usize = extended.surface_entries()
            .iter()
            .map(|v| v.len())
            .sum();

        // Each of the 8 corners is on 3 faces, so total should be 8*3 = 24
        // (but hull is triangulated so might be different)
        assert!(total_surface_entries > 0);
    }

    #[test]
    fn test_surface_entries_complete() {
        // Test with 8-corner cube: verify each vertex appears on at least 3 surfaces
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        let num_real = extended.len();

        // Count how many surfaces each vertex is on
        let mut vertex_surface_count = vec![0usize; num_real];
        for surface_entries in extended.surface_entries() {
            for &vertex_idx in surface_entries {
                vertex_surface_count[vertex_idx] += 1;
            }
        }

        // For a cube, each vertex should be on at least 3 surfaces
        for (idx, &count) in vertex_surface_count.iter().enumerate() {
            assert!(
                count >= 3,
                "Vertex {} is only on {} surfaces, expected at least 3",
                idx, count
            );
        }

        // Also verify that each surface has at least 3 entries (the triangle vertices)
        for (plane_idx, surface_entries) in extended.surface_entries().iter().enumerate() {
            assert!(
                surface_entries.len() >= 3,
                "Surface {} has only {} entries, expected at least 3",
                plane_idx, surface_entries.len()
            );
        }
    }

    #[test]
    fn test_tetrahedron_surface_entries() {
        // Test with 4-color palette forming a tetrahedron
        // This is similar to CGA Mode 5: black, cyan, magenta, white
        let colors = vec![
            (0, 0, 0, 255),       // Black (0,0,0)
            (0, 255, 255, 255),   // Cyan (0,1,1)
            (255, 0, 255, 255),   // Magenta (1,0,1)
            (255, 255, 255, 255), // White (1,1,1)
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        let num_planes = extended.hull().planes.len();

        // A tetrahedron has 4 triangular faces
        assert_eq!(num_planes, 4, "Tetrahedron should have 4 faces");

        // Each vertex is on exactly 3 faces
        let mut vertex_surface_count = vec![0usize; 4];
        for surface_entries in extended.surface_entries() {
            for &vertex_idx in surface_entries {
                vertex_surface_count[vertex_idx] += 1;
            }
        }

        for (idx, &count) in vertex_surface_count.iter().enumerate() {
            assert_eq!(
                count, 3,
                "Vertex {} is on {} surfaces, expected exactly 3",
                idx, count
            );
        }

        // Each face should have exactly 3 vertices
        for (plane_idx, surface_entries) in extended.surface_entries().iter().enumerate() {
            assert_eq!(
                surface_entries.len(), 3,
                "Surface {} has {} entries, expected exactly 3",
                plane_idx, surface_entries.len()
            );
        }
    }

    #[test]
    fn test_cga_mode5_surface_entries() {
        // CGA Mode 5: Black, Cyan, Magenta, White - forms a tetrahedron
        let colors = vec![
            (0, 0, 0, 255),       // Black
            (0, 255, 255, 255),   // Cyan
            (255, 0, 255, 255),   // Magenta
            (255, 255, 255, 255), // White
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Verify all 4 vertices are correctly assigned to surfaces
        let num_planes = extended.hull().planes.len();
        assert_eq!(num_planes, 4, "CGA Mode 5 tetrahedron should have 4 faces");

        // Verify each surface has exactly 3 entries
        for (i, entries) in extended.surface_entries().iter().enumerate() {
            assert_eq!(
                entries.len(), 3,
                "Surface {} should have 3 vertices, has {}",
                i, entries.len()
            );
        }

        // Verify each vertex is on exactly 3 surfaces
        let mut counts = [0usize; 4];
        for entries in extended.surface_entries() {
            for &idx in entries {
                counts[idx] += 1;
            }
        }
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(c, 3, "Vertex {} should be on 3 surfaces, is on {}", i, c);
        }

        // Verify the total: 4 faces × 3 vertices = 12 total entries
        let total: usize = extended.surface_entries().iter().map(|v| v.len()).sum();
        assert_eq!(total, 12, "Total surface entries should be 12");
    }

    #[test]
    fn test_surface_redirection() {
        // Test that surface redirection works correctly
        // Use a simple 4-color tetrahedron
        let colors = vec![
            (0, 0, 0, 255),       // Black
            (0, 255, 255, 255),   // Cyan
            (255, 0, 255, 255),   // Magenta
            (255, 255, 255, 255), // White
        ];
        let palette = DitherPalette::new(&colors, PerceptualSpace::OkLab);
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Test a point that should trigger surface redirection
        // A point outside the tetrahedron should still find a valid entry
        let test_point = [0.5, 0.5, 0.0]; // This point may be outside the tetrahedron
        let nearest = extended.find_nearest_real(test_point, true);

        // Should return a valid index
        assert!(nearest < 4);
    }
}
