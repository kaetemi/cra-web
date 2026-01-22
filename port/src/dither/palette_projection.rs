//! Palette projection and ghost entry system for gamut mapping.
//!
//! This module extends palettes with "ghost" entries that lie on the convex hull
//! boundary. When the nearest color match is a ghost, the search is redirected to
//! real palette entries on that hull surface. Input colors outside the hull are
//! clamped to the nearest point on the hull boundary.

use super::palette_hull::{HullPlane, PaletteHull, EPSILON};
use super::paletted::DitherPalette;
use crate::color_distance::perceptual_distance_sq;
use super::common::{linear_rgb_to_perceptual_clamped, PerceptualSpace};

// ============================================================================
// Ghost entry and extended palette structures
// ============================================================================

/// A ghost palette entry - a virtual color on the hull boundary.
#[derive(Clone, Debug)]
pub struct GhostEntry {
    /// Linear RGB position on the hull surface
    pub lin_rgb: [f32; 3],
    /// Index of the hull plane this ghost lies on
    pub plane_idx: usize,
    /// Perceptual space coordinates (for distance calculations)
    pub perc: [f32; 3],
}

/// Entry in the extended palette (either real or ghost).
#[derive(Clone, Debug)]
pub enum ExtendedEntry {
    /// A real palette entry (index into original palette)
    Real(usize),
    /// A ghost entry on a hull surface
    Ghost(GhostEntry),
}

/// Extended palette with ghost entries and hull surface membership.
#[derive(Clone, Debug)]
pub struct ExtendedPalette {
    /// The original palette
    palette: DitherPalette,
    /// The convex hull of the palette
    hull: PaletteHull,
    /// All entries (real + ghost) for nearest-neighbor search
    entries: Vec<ExtendedEntry>,
    /// Linear RGB positions for all entries (for fast distance calc)
    positions: Vec<[f32; 3]>,
    /// Perceptual positions for all entries
    perceptual: Vec<[f32; 3]>,
    /// For each hull plane, indices of real palette entries on that surface
    surface_real_entries: Vec<Vec<usize>>,
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
        let linear_points = palette.linear_rgb_points();
        let num_real = linear_points.len();
        let num_planes = hull.planes.len();

        // Initialize with real entries
        let mut entries: Vec<ExtendedEntry> = (0..num_real)
            .map(ExtendedEntry::Real)
            .collect();

        let mut positions: Vec<[f32; 3]> = linear_points.clone();

        let mut perceptual: Vec<[f32; 3]> = linear_points
            .iter()
            .map(|p| {
                let (l, a, b) = linear_rgb_to_perceptual_clamped(space, p[0], p[1], p[2]);
                [l, a, b]
            })
            .collect();

        // (1b) Find real entries on each hull surface
        let mut surface_real_entries: Vec<Vec<usize>> = vec![Vec::new(); num_planes];
        for (entry_idx, pos) in linear_points.iter().enumerate() {
            for (plane_idx, plane) in hull.planes.iter().enumerate() {
                if plane.signed_distance(*pos).abs() <= EPSILON {
                    surface_real_entries[plane_idx].push(entry_idx);
                }
            }
        }

        // (1a) For each real entry, project onto hull and create ghosts if needed
        for pos in linear_points.iter() {
            for (plane_idx, plane) in hull.planes.iter().enumerate() {
                // Skip if this entry is already on this plane
                if plane.signed_distance(*pos).abs() <= EPSILON {
                    continue;
                }

                // Project point onto plane
                let projected = project_point_onto_plane(*pos, plane);

                // Check if projection is within hull (on the actual face, not extended plane)
                if !is_point_on_hull_face(&projected, &hull, plane_idx) {
                    continue;
                }

                // Check if there's already an entry (real or ghost) at this position
                let already_exists = positions.iter().any(|p| {
                    let d = distance_sq(*p, projected);
                    d < EPSILON * EPSILON
                });

                if !already_exists {
                    let (l, a, b) = linear_rgb_to_perceptual_clamped(
                        space, projected[0], projected[1], projected[2]
                    );

                    entries.push(ExtendedEntry::Ghost(GhostEntry {
                        lin_rgb: projected,
                        plane_idx,
                        perc: [l, a, b],
                    }));
                    positions.push(projected);
                    perceptual.push([l, a, b]);
                }
            }
        }

        Self {
            palette,
            hull,
            entries,
            positions,
            perceptual,
            surface_real_entries,
            space,
        }
    }

    /// Number of total entries (real + ghost).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of real entries.
    pub fn real_count(&self) -> usize {
        self.palette.len()
    }

    /// Number of ghost entries.
    pub fn ghost_count(&self) -> usize {
        self.entries.len() - self.palette.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the original palette.
    pub fn palette(&self) -> &DitherPalette {
        &self.palette
    }

    /// Get the hull.
    pub fn hull(&self) -> &PaletteHull {
        &self.hull
    }

    /// (2) Clamp a linear RGB point to within the hull.
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

    /// (1c) Find the nearest real palette entry for a linear RGB point.
    /// If the initial nearest is a ghost, redirects to real entries on that surface.
    pub fn find_nearest_real(&self, lin_rgb: [f32; 3]) -> usize {
        // First, find nearest entry (real or ghost) in perceptual space
        let (l, a, b) = linear_rgb_to_perceptual_clamped(self.space, lin_rgb[0], lin_rgb[1], lin_rgb[2]);
        let target_perc = [l, a, b];

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (idx, perc) in self.perceptual.iter().enumerate() {
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

        // If it's a real entry, we're done
        match &self.entries[best_idx] {
            ExtendedEntry::Real(real_idx) => *real_idx,
            ExtendedEntry::Ghost(ghost) => {
                // Redirect to real entries on this hull surface
                self.find_nearest_on_surface(target_perc, ghost.plane_idx)
            }
        }
    }

    /// Find nearest real entry among those on a specific hull surface.
    fn find_nearest_on_surface(&self, target_perc: [f32; 3], plane_idx: usize) -> usize {
        let surface_entries = &self.surface_real_entries[plane_idx];

        if surface_entries.is_empty() {
            // Fallback: find any nearest real entry
            return self.find_nearest_real_fallback(target_perc);
        }

        let mut best_idx = surface_entries[0];
        let mut best_dist = f32::INFINITY;

        for &real_idx in surface_entries {
            let perc = &self.perceptual[real_idx];
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
    fn find_nearest_real_fallback(&self, target_perc: [f32; 3]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (idx, entry) in self.entries.iter().enumerate() {
            if let ExtendedEntry::Real(real_idx) = entry {
                let perc = &self.perceptual[idx];
                let d = perceptual_distance_sq(
                    self.space,
                    target_perc[0], target_perc[1], target_perc[2],
                    perc[0], perc[1], perc[2],
                );
                if d < best_dist {
                    best_dist = d;
                    best_idx = *real_idx;
                }
            }
        }

        best_idx
    }

    /// Get linear RGB position for a real palette entry.
    pub fn get_real_linear_rgb(&self, real_idx: usize) -> [f32; 3] {
        self.positions[real_idx]
    }

    /// Get sRGB values for a real palette entry.
    pub fn get_real_srgb(&self, real_idx: usize) -> (u8, u8, u8, u8) {
        self.palette.get_srgb(real_idx)
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

/// Check if a projected point actually lies within a hull face (not just on the extended plane).
/// This is an approximation - we check if the point is "inside" all other hull planes.
fn is_point_on_hull_face(point: &[f32; 3], hull: &PaletteHull, _plane_idx: usize) -> bool {
    // A point is on a face if it's inside (or on) all other planes
    // We use a slightly relaxed epsilon since projections can have numerical error
    let relaxed_epsilon = EPSILON * 10.0;

    for plane in &hull.planes {
        if plane.signed_distance(*point) > relaxed_epsilon {
            return false;
        }
    }
    true
}

/// Squared distance between two points.
#[inline]
fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
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

        assert_eq!(extended.real_count(), 8);
        // Ghost count depends on geometry, just ensure total >= real
        assert!(extended.len() >= 8);
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
        let (r, g, b, _) = extended.get_real_srgb(nearest);
        assert_eq!((r, g, b), (0, 0, 0));

        // Point very close to white corner
        let near_white = [0.99, 0.99, 0.99];
        let nearest = extended.find_nearest_real(near_white);

        // Should find white (index 7)
        let (r, g, b, _) = extended.get_real_srgb(nearest);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_surface_entries_populated() {
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        // Each face of the cube should have at least 3 vertices (palette entries)
        // For a full cube with 8 corners, each face has 4 corners
        let total_surface_entries: usize = extended.surface_real_entries
            .iter()
            .map(|v| v.len())
            .sum();

        // Each of the 8 corners is on 3 faces, so total should be 8*3 = 24
        // (but hull is triangulated so might be different)
        assert!(total_surface_entries > 0);
    }

    #[test]
    fn test_surface_entries_complete() {
        // Test with 8-corner cube: verify each vertex appears on exactly 3 logical faces
        let palette = make_test_palette();
        let extended = ExtendedPalette::new(palette, PerceptualSpace::OkLab);

        let num_planes = extended.hull().planes.len();
        let num_real = extended.real_count();

        // Count how many surfaces each vertex is on
        let mut vertex_surface_count = vec![0usize; num_real];
        for surface_entries in &extended.surface_real_entries {
            for &vertex_idx in surface_entries {
                vertex_surface_count[vertex_idx] += 1;
            }
        }

        // For a cube, we expect either 6 planes (if merged) or 12 planes (if triangulated)
        // Each vertex should be on at least 3 surfaces (the 3 faces meeting at that corner)
        // With triangulation, each vertex might be on 6 surfaces (2 triangles per face × 3 faces)
        for (idx, &count) in vertex_surface_count.iter().enumerate() {
            assert!(
                count >= 3,
                "Vertex {} is only on {} surfaces, expected at least 3",
                idx, count
            );
        }

        // Also verify that each surface has at least 3 entries (the triangle vertices)
        for (plane_idx, surface_entries) in extended.surface_real_entries.iter().enumerate() {
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
        for surface_entries in &extended.surface_real_entries {
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
        for (plane_idx, surface_entries) in extended.surface_real_entries.iter().enumerate() {
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
        for (i, entries) in extended.surface_real_entries.iter().enumerate() {
            assert_eq!(
                entries.len(), 3,
                "Surface {} should have 3 vertices, has {}",
                i, entries.len()
            );
        }

        // Verify each vertex is on exactly 3 surfaces
        let mut counts = [0usize; 4];
        for entries in &extended.surface_real_entries {
            for &idx in entries {
                counts[idx] += 1;
            }
        }
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(c, 3, "Vertex {} should be on 3 surfaces, is on {}", i, c);
        }

        // Verify the total: 4 faces × 3 vertices = 12 total entries
        let total: usize = extended.surface_real_entries.iter().map(|v| v.len()).sum();
        assert_eq!(total, 12, "Total surface entries should be 12");
    }
}
