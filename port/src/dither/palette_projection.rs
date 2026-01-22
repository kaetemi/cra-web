//! Palette projection for gamut mapping.
//!
//! This module extends palettes with convex hull information for proper gamut mapping.
//! When searching for the nearest palette color, we also consider projections onto
//! hull surfaces. If a surface projection is closer than any real entry, we search
//! only among palette entries on that surface.

use super::palette_hull::{HullPlane, PaletteHull, EPSILON};
use super::paletted::DitherPalette;
use crate::color_distance::perceptual_distance_sq;
use super::common::{linear_rgb_to_perceptual, linear_rgb_to_perceptual_clamped, PerceptualSpace};

// ============================================================================
// Extended palette structure
// ============================================================================

/// Extended palette with hull information and surface membership.
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

        Self {
            palette,
            hull,
            linear_positions,
            perceptual_positions,
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
    /// This searches both:
    /// 1. Real palette entries (perceptual distance)
    /// 2. Projections onto each hull surface (project in linear, distance in perceptual)
    ///
    /// If a surface projection is closer than any real entry, we search only among
    /// the palette entries on that surface.
    pub fn find_nearest_real(&self, lin_rgb: [f32; 3]) -> usize {
        // Convert target to perceptual space (non-clamped since this is a target color)
        let (l, a, b) = linear_rgb_to_perceptual(self.space, lin_rgb[0], lin_rgb[1], lin_rgb[2]);
        let target_perc = [l, a, b];

        // Track best real entry
        let mut best_real_idx = 0;
        let mut best_real_dist = f32::INFINITY;

        // Search real palette entries
        for (idx, perc) in self.perceptual_positions.iter().enumerate() {
            let d = perceptual_distance_sq(
                self.space,
                target_perc[0], target_perc[1], target_perc[2],
                perc[0], perc[1], perc[2],
            );
            if d < best_real_dist {
                best_real_dist = d;
                best_real_idx = idx;
            }
        }

        // Track best surface projection
        let mut best_surface_idx: Option<usize> = None;
        let mut best_surface_dist = f32::INFINITY;

        // Search hull surface projections
        for (plane_idx, plane) in self.hull.planes.iter().enumerate() {
            // Project target onto this plane in linear RGB space
            let projected_lin = project_point_onto_plane(lin_rgb, plane);

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

        // If a surface projection is closer than any real entry, search that surface
        if let Some(surface_idx) = best_surface_idx {
            if best_surface_dist < best_real_dist {
                return self.find_nearest_on_surface(target_perc, surface_idx);
            }
        }

        best_real_idx
    }

    /// Find nearest real entry among those on a specific hull surface.
    fn find_nearest_on_surface(&self, target_perc: [f32; 3], plane_idx: usize) -> usize {
        let surface_entries = &self.surface_entries[plane_idx];

        if surface_entries.is_empty() {
            // Fallback: find any nearest real entry
            return self.find_nearest_fallback(target_perc);
        }

        let mut best_idx = surface_entries[0];
        let mut best_dist = f32::INFINITY;

        for &real_idx in surface_entries {
            let perc = &self.perceptual_positions[real_idx];
            let d = perceptual_distance_sq(
                self.space,
                target_perc[0], target_perc[1], target_perc[2],
                perc[0], perc[1], perc[2],
            );
            if d < best_dist {
                best_dist = d;
                best_idx = real_idx;
            }
        }

        best_idx
    }

    /// Fallback: find nearest among all real entries.
    fn find_nearest_fallback(&self, target_perc: [f32; 3]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (idx, perc) in self.perceptual_positions.iter().enumerate() {
            let d = perceptual_distance_sq(
                self.space,
                target_perc[0], target_perc[1], target_perc[2],
                perc[0], perc[1], perc[2],
            );
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

/// Project a point onto a plane (nearest point on the infinite plane).
fn project_point_onto_plane(point: [f32; 3], plane: &HullPlane) -> [f32; 3] {
    let dist = plane.signed_distance(point);
    [
        point[0] - dist * plane.normal[0],
        point[1] - dist * plane.normal[1],
        point[2] - dist * plane.normal[2],
    ]
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
        let nearest = extended.find_nearest_real(near_black);

        // Should find black (index 0)
        let (r, g, b, _) = extended.get_srgb(nearest);
        assert_eq!((r, g, b), (0, 0, 0));

        // Point very close to white corner
        let near_white = [0.99, 0.99, 0.99];
        let nearest = extended.find_nearest_real(near_white);

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
        let nearest = extended.find_nearest_real(test_point);

        // Should return a valid index
        assert!(nearest < 4);
    }
}
