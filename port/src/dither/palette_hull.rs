//! Convex hull computation for palettes in linear RGB space.
//!
//! Computes the convex hull of palette colors and represents it as a set of
//! infinite planes (point + normal). This can be used to determine if a color
//! is inside the palette's gamut or to find the closest point on the hull boundary.

use super::paletted::DitherPalette;

/// Epsilon for floating point comparisons: 1 / u16::MAX
pub const EPSILON: f32 = 1.0 / 65535.0;

/// An infinite plane defined by a point and normal vector.
#[derive(Clone, Copy, Debug)]
pub struct HullPlane {
    /// A point on the plane (in linear RGB space)
    pub point: [f32; 3],
    /// The outward-facing normal vector (normalized)
    pub normal: [f32; 3],
}

impl HullPlane {
    /// Create a new plane from a point and normal.
    /// The normal will be normalized.
    pub fn new(point: [f32; 3], normal: [f32; 3]) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let normal = if len > EPSILON {
            [normal[0] / len, normal[1] / len, normal[2] / len]
        } else {
            [0.0, 0.0, 1.0] // Degenerate case
        };
        Self { point, normal }
    }

    /// Signed distance from point to plane.
    /// Positive = outside (in direction of normal), negative = inside.
    #[inline]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        let dx = p[0] - self.point[0];
        let dy = p[1] - self.point[1];
        let dz = p[2] - self.point[2];
        dx * self.normal[0] + dy * self.normal[1] + dz * self.normal[2]
    }

    /// Check if a point is on the outside of this plane (within epsilon).
    #[inline]
    pub fn is_outside(&self, p: [f32; 3]) -> bool {
        self.signed_distance(p) > EPSILON
    }
}

/// Convex hull of a palette in linear RGB space.
#[derive(Clone, Debug)]
pub struct PaletteHull {
    /// The planes forming the convex hull faces
    pub planes: Vec<HullPlane>,
    /// The vertices of the hull (linear RGB)
    pub vertices: Vec<[f32; 3]>,
}

impl PaletteHull {
    /// Build convex hull from a DitherPalette.
    pub fn from_palette(palette: &DitherPalette) -> Self {
        let points: Vec<[f32; 3]> = palette.linear_rgb_points();
        Self::from_points(&points)
    }

    /// Build convex hull from a set of points in linear RGB space.
    pub fn from_points(points: &[[f32; 3]]) -> Self {
        if points.len() < 4 {
            return Self::degenerate_hull(points);
        }

        // Use Quickhull algorithm
        quickhull(points)
    }

    /// Check if a point is inside the convex hull.
    /// Returns true if the point is inside or on the boundary.
    #[inline]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        // A point is inside if it's on the inside of all planes
        for plane in &self.planes {
            if plane.signed_distance(p) > EPSILON {
                return false;
            }
        }
        true
    }

    /// Find the maximum signed distance to any hull plane.
    /// Negative = inside, positive = outside.
    #[inline]
    pub fn max_signed_distance(&self, p: [f32; 3]) -> f32 {
        self.planes
            .iter()
            .map(|plane| plane.signed_distance(p))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Check if the hull is valid (has at least 4 planes for a 3D hull).
    pub fn is_valid(&self) -> bool {
        self.planes.len() >= 4
    }

    /// Handle degenerate cases (fewer than 4 points).
    fn degenerate_hull(points: &[[f32; 3]]) -> Self {
        match points.len() {
            0 => Self {
                planes: vec![],
                vertices: vec![],
            },
            1 => Self {
                planes: vec![],
                vertices: vec![points[0]],
            },
            2 => {
                // Line segment - create planes perpendicular to the line
                let dir = [
                    points[1][0] - points[0][0],
                    points[1][1] - points[0][1],
                    points[1][2] - points[0][2],
                ];
                Self {
                    planes: vec![
                        HullPlane::new(points[0], [-dir[0], -dir[1], -dir[2]]),
                        HullPlane::new(points[1], dir),
                    ],
                    vertices: points.to_vec(),
                }
            }
            3 => {
                // Triangle - create two planes (front and back)
                let normal = triangle_normal(points[0], points[1], points[2]);
                Self {
                    planes: vec![
                        HullPlane::new(points[0], normal),
                        HullPlane::new(points[0], [-normal[0], -normal[1], -normal[2]]),
                    ],
                    vertices: points.to_vec(),
                }
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Quickhull implementation
// ============================================================================

/// Compute convex hull using Quickhull algorithm.
fn quickhull(points: &[[f32; 3]]) -> PaletteHull {
    // Find initial tetrahedron
    let (initial_faces, mut outside_sets, interior_point) = match find_initial_simplex(points) {
        Some(result) => result,
        None => {
            // All points are coplanar or collinear
            return PaletteHull::degenerate_hull(points);
        }
    };

    let mut faces: Vec<Face> = initial_faces;
    let vertices: Vec<[f32; 3]> = points.to_vec();

    // Process faces with outside points
    let mut i = 0;
    while i < faces.len() {
        if outside_sets[i].is_empty() {
            i += 1;
            continue;
        }

        // Find furthest point from this face
        let face = &faces[i];
        let furthest_idx = find_furthest_point(face, &outside_sets[i], &vertices);
        let eye_point = vertices[furthest_idx];

        // Find all faces visible from this point
        let visible_faces: Vec<usize> = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.plane.signed_distance(eye_point) > EPSILON)
            .map(|(idx, _)| idx)
            .collect();

        if visible_faces.is_empty() {
            i += 1;
            continue;
        }

        // Find horizon edges (edges between visible and non-visible faces)
        let horizon = find_horizon_edges(&faces, &visible_faces);

        // Collect all outside points from visible faces
        let mut all_outside: Vec<usize> = Vec::new();
        for &face_idx in &visible_faces {
            all_outside.extend(&outside_sets[face_idx]);
        }
        // Remove the eye point
        all_outside.retain(|&idx| idx != furthest_idx);

        // Remove visible faces (in reverse order to preserve indices)
        let mut sorted_visible = visible_faces.clone();
        sorted_visible.sort_unstable_by(|a, b| b.cmp(a));
        for &face_idx in &sorted_visible {
            faces.swap_remove(face_idx);
            outside_sets.swap_remove(face_idx);
        }

        // Create new faces from horizon edges to eye point
        let new_faces_start = faces.len();
        for (v0, v1) in horizon {
            let mut new_face = Face::new(v0, v1, furthest_idx, &vertices);
            // Ensure normal points outward (away from interior point)
            if new_face.plane.signed_distance(interior_point) > 0.0 {
                new_face.flip();
            }
            faces.push(new_face);
            outside_sets.push(Vec::new());
        }

        // Distribute outside points to new faces
        for &pt_idx in &all_outside {
            let pt = vertices[pt_idx];
            for face_idx in new_faces_start..faces.len() {
                if faces[face_idx].plane.signed_distance(pt) > EPSILON {
                    outside_sets[face_idx].push(pt_idx);
                    break;
                }
            }
        }

        // Restart from beginning since face indices changed
        i = 0;
    }

    // Extract hull vertices (vertices that appear in faces)
    let mut hull_vertex_set: Vec<bool> = vec![false; vertices.len()];
    for face in &faces {
        hull_vertex_set[face.v[0]] = true;
        hull_vertex_set[face.v[1]] = true;
        hull_vertex_set[face.v[2]] = true;
    }
    let hull_vertices: Vec<[f32; 3]> = hull_vertex_set
        .iter()
        .enumerate()
        .filter(|&(_, on_hull)| *on_hull)
        .map(|(idx, _)| vertices[idx])
        .collect();

    // Convert faces to planes
    let planes: Vec<HullPlane> = faces.iter().map(|f| f.plane).collect();

    PaletteHull {
        planes,
        vertices: hull_vertices,
    }
}

/// A face of the convex hull (triangle).
#[derive(Clone, Debug)]
struct Face {
    /// Vertex indices (counter-clockwise when viewed from outside)
    v: [usize; 3],
    /// The plane of this face
    plane: HullPlane,
}

impl Face {
    fn new(v0: usize, v1: usize, v2: usize, vertices: &[[f32; 3]]) -> Self {
        let p0 = vertices[v0];
        let p1 = vertices[v1];
        let p2 = vertices[v2];
        let normal = triangle_normal(p0, p1, p2);
        Self {
            v: [v0, v1, v2],
            plane: HullPlane::new(p0, normal),
        }
    }

    /// Flip the face winding and normal direction.
    fn flip(&mut self) {
        self.v.swap(0, 1);
        self.plane.normal[0] = -self.plane.normal[0];
        self.plane.normal[1] = -self.plane.normal[1];
        self.plane.normal[2] = -self.plane.normal[2];
    }

    fn contains_edge(&self, e0: usize, e1: usize) -> bool {
        for i in 0..3 {
            let j = (i + 1) % 3;
            if (self.v[i] == e0 && self.v[j] == e1) || (self.v[i] == e1 && self.v[j] == e0) {
                return true;
            }
        }
        false
    }
}

/// Find initial tetrahedron for Quickhull.
/// Returns (faces, outside_sets, interior_point) where interior_point is the centroid.
fn find_initial_simplex(
    points: &[[f32; 3]],
) -> Option<(Vec<Face>, Vec<Vec<usize>>, [f32; 3])> {
    let n = points.len();
    if n < 4 {
        return None;
    }

    // Find extremal points along each axis
    let (mut min_x, mut max_x) = (0, 0);
    let (mut min_y, mut max_y) = (0, 0);
    let (mut min_z, mut max_z) = (0, 0);

    for i in 1..n {
        if points[i][0] < points[min_x][0] {
            min_x = i;
        }
        if points[i][0] > points[max_x][0] {
            max_x = i;
        }
        if points[i][1] < points[min_y][1] {
            min_y = i;
        }
        if points[i][1] > points[max_y][1] {
            max_y = i;
        }
        if points[i][2] < points[min_z][2] {
            min_z = i;
        }
        if points[i][2] > points[max_z][2] {
            max_z = i;
        }
    }

    // Find the pair of extremal points with maximum distance
    let extremals = [min_x, max_x, min_y, max_y, min_z, max_z];
    let mut best_pair = (0, 1);
    let mut best_dist = 0.0f32;

    for i in 0..extremals.len() {
        for j in (i + 1)..extremals.len() {
            let d = distance_sq(points[extremals[i]], points[extremals[j]]);
            if d > best_dist {
                best_dist = d;
                best_pair = (extremals[i], extremals[j]);
            }
        }
    }

    if best_dist < EPSILON * EPSILON {
        return None; // All points coincident
    }

    let (p0, p1) = best_pair;

    // Find third point furthest from line p0-p1
    let line_dir = [
        points[p1][0] - points[p0][0],
        points[p1][1] - points[p0][1],
        points[p1][2] - points[p0][2],
    ];
    let mut p2 = 0;
    let mut max_dist = 0.0f32;
    for i in 0..n {
        if i == p0 || i == p1 {
            continue;
        }
        let d = point_line_distance_sq(points[i], points[p0], line_dir);
        if d > max_dist {
            max_dist = d;
            p2 = i;
        }
    }

    if max_dist < EPSILON * EPSILON {
        return None; // All points collinear
    }

    // Find fourth point furthest from plane p0-p1-p2
    let plane_normal = triangle_normal(points[p0], points[p1], points[p2]);
    let mut p3 = 0;
    let mut max_dist = 0.0f32;
    for i in 0..n {
        if i == p0 || i == p1 || i == p2 {
            continue;
        }
        let d = point_plane_distance(points[i], points[p0], plane_normal).abs();
        if d > max_dist {
            max_dist = d;
            p3 = i;
        }
    }

    if max_dist < EPSILON {
        return None; // All points coplanar
    }

    // Create initial tetrahedron with outward-facing normals
    let centroid = [
        (points[p0][0] + points[p1][0] + points[p2][0] + points[p3][0]) / 4.0,
        (points[p0][1] + points[p1][1] + points[p2][1] + points[p3][1]) / 4.0,
        (points[p0][2] + points[p1][2] + points[p2][2] + points[p3][2]) / 4.0,
    ];

    // Create faces and ensure normals point outward
    let mut faces = vec![
        Face::new(p0, p1, p2, points),
        Face::new(p0, p2, p3, points),
        Face::new(p0, p3, p1, points),
        Face::new(p1, p3, p2, points),
    ];

    // Flip faces whose normals point inward
    for face in &mut faces {
        if face.plane.signed_distance(centroid) > 0.0 {
            // Normal points toward centroid, flip the face
            face.v.swap(0, 1);
            face.plane.normal[0] = -face.plane.normal[0];
            face.plane.normal[1] = -face.plane.normal[1];
            face.plane.normal[2] = -face.plane.normal[2];
        }
    }

    // Assign points to outside sets
    let initial_verts = [p0, p1, p2, p3];
    let mut outside_sets: Vec<Vec<usize>> = vec![Vec::new(); 4];

    for i in 0..n {
        if initial_verts.contains(&i) {
            continue;
        }
        let pt = points[i];
        for (face_idx, face) in faces.iter().enumerate() {
            if face.plane.signed_distance(pt) > EPSILON {
                outside_sets[face_idx].push(i);
                break;
            }
        }
    }

    Some((faces, outside_sets, centroid))
}

/// Find the point furthest from a face among the outside set.
fn find_furthest_point(face: &Face, outside_set: &[usize], vertices: &[[f32; 3]]) -> usize {
    let mut best_idx = outside_set[0];
    let mut best_dist = face.plane.signed_distance(vertices[best_idx]);

    for &idx in outside_set.iter().skip(1) {
        let d = face.plane.signed_distance(vertices[idx]);
        if d > best_dist {
            best_dist = d;
            best_idx = idx;
        }
    }

    best_idx
}

/// Find horizon edges - edges shared between visible and non-visible faces.
fn find_horizon_edges(faces: &[Face], visible_indices: &[usize]) -> Vec<(usize, usize)> {
    let visible_set: std::collections::HashSet<usize> = visible_indices.iter().copied().collect();
    let mut horizon = Vec::new();

    for &face_idx in visible_indices {
        let face = &faces[face_idx];
        for i in 0..3 {
            let j = (i + 1) % 3;
            let edge = (face.v[i], face.v[j]);

            // Check if any non-visible face shares this edge
            let mut edge_on_horizon = false;
            for (other_idx, other_face) in faces.iter().enumerate() {
                if visible_set.contains(&other_idx) {
                    continue;
                }
                if other_face.contains_edge(edge.0, edge.1) {
                    edge_on_horizon = true;
                    break;
                }
            }

            if edge_on_horizon {
                // Add edge in reverse order so new face has correct winding
                horizon.push((edge.1, edge.0));
            }
        }
    }

    horizon
}

// ============================================================================
// Geometry helpers
// ============================================================================

/// Compute triangle normal (not normalized).
fn triangle_normal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    // Cross product u × v
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

/// Squared distance between two points.
#[inline]
fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from point to line.
fn point_line_distance_sq(point: [f32; 3], line_point: [f32; 3], line_dir: [f32; 3]) -> f32 {
    let v = [
        point[0] - line_point[0],
        point[1] - line_point[1],
        point[2] - line_point[2],
    ];
    // Cross product v × line_dir
    let cross = [
        v[1] * line_dir[2] - v[2] * line_dir[1],
        v[2] * line_dir[0] - v[0] * line_dir[2],
        v[0] * line_dir[1] - v[1] * line_dir[0],
    ];
    let cross_len_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    let dir_len_sq =
        line_dir[0] * line_dir[0] + line_dir[1] * line_dir[1] + line_dir[2] * line_dir[2];
    if dir_len_sq > EPSILON * EPSILON {
        cross_len_sq / dir_len_sq
    } else {
        0.0
    }
}

/// Signed distance from point to plane.
fn point_plane_distance(point: [f32; 3], plane_point: [f32; 3], plane_normal: [f32; 3]) -> f32 {
    let v = [
        point[0] - plane_point[0],
        point[1] - plane_point[1],
        point[2] - plane_point[2],
    ];
    v[0] * plane_normal[0] + v[1] * plane_normal[1] + v[2] * plane_normal[2]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON * 10.0
    }

    #[test]
    fn test_plane_signed_distance() {
        let plane = HullPlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);

        assert!(approx_eq(plane.signed_distance([0.0, 0.0, 1.0]), 1.0));
        assert!(approx_eq(plane.signed_distance([0.0, 0.0, -1.0]), -1.0));
        assert!(approx_eq(plane.signed_distance([5.0, 3.0, 0.0]), 0.0));
    }

    #[test]
    fn test_cube_hull() {
        // Unit cube vertices
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let hull = PaletteHull::from_points(&points);

        // Cube has 6 faces (but triangulated = 12 triangles)
        // Actually quickhull produces triangular faces
        assert!(hull.planes.len() >= 6, "Expected at least 6 planes for cube");

        // Center should be inside
        assert!(hull.contains([0.5, 0.5, 0.5]), "Center should be inside");

        // Corners should be inside (on boundary)
        assert!(hull.contains([0.0, 0.0, 0.0]), "Corner should be inside");
        assert!(hull.contains([1.0, 1.0, 1.0]), "Corner should be inside");

        // Outside points
        assert!(!hull.contains([2.0, 0.5, 0.5]), "Should be outside");
        assert!(!hull.contains([-1.0, 0.5, 0.5]), "Should be outside");
    }

    #[test]
    fn test_tetrahedron_hull() {
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];

        let hull = PaletteHull::from_points(&points);

        // Tetrahedron has 4 faces
        assert_eq!(hull.planes.len(), 4, "Tetrahedron should have 4 faces");

        // Centroid should be inside
        let centroid = [
            (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0,
            (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0,
            (points[0][2] + points[1][2] + points[2][2] + points[3][2]) / 4.0,
        ];
        assert!(hull.contains(centroid), "Centroid should be inside");
    }

    #[test]
    fn test_degenerate_cases() {
        // Single point
        let hull1 = PaletteHull::from_points(&[[0.5, 0.5, 0.5]]);
        assert_eq!(hull1.vertices.len(), 1);

        // Two points (line)
        let hull2 = PaletteHull::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(hull2.vertices.len(), 2);
        assert_eq!(hull2.planes.len(), 2);

        // Three points (triangle)
        let hull3 = PaletteHull::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(hull3.vertices.len(), 3);
        assert_eq!(hull3.planes.len(), 2); // Front and back
    }

    #[test]
    fn test_max_signed_distance() {
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = PaletteHull::from_points(&points);

        // Inside point should have negative max distance
        let inside_dist = hull.max_signed_distance([0.1, 0.1, 0.1]);
        assert!(inside_dist < 0.0, "Inside point should have negative distance");

        // Outside point should have positive max distance
        let outside_dist = hull.max_signed_distance([1.0, 1.0, 1.0]);
        assert!(outside_dist > 0.0, "Outside point should have positive distance");
    }

    #[test]
    fn test_interior_points_ignored() {
        // Cube with interior point
        let mut points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        // Add interior point
        points.push([0.5, 0.5, 0.5]);

        let hull = PaletteHull::from_points(&points);

        // Interior point should not be a hull vertex
        assert_eq!(hull.vertices.len(), 8, "Interior point should not be on hull");
    }
}
