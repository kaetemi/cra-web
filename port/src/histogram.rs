/// Histogram matching implementation for uint8 images.
/// Based on the algorithm described in PORTING.md

/// Interpolation mode for f32 histogram matching
#[derive(Clone, Copy, Debug, Default)]
pub enum InterpolationMode {
    /// Snap to nearest reference value (can cause banding)
    #[allow(dead_code)]
    Nearest,
    /// Linear interpolation between adjacent reference values (smoother)
    #[default]
    Linear,
}

/// Alignment mode for f32 histogram matching
#[derive(Clone, Copy, Debug, Default)]
pub enum AlignmentMode {
    /// Endpoint-aligned: rank 0 → ref[0], rank n-1 → ref[m-1]
    /// Preserves exact min/max of reference, but may expand source range
    #[default]
    Endpoint,
    /// Midpoint-aligned: treats each rank as a bin center
    /// More statistically correct quantile sampling, doesn't force range expansion
    Midpoint,
}

use crate::dither::lowbias32;

/// Match histogram for f32 values using sort-based quantile matching.
/// No binning/quantization required - works directly on continuous values.
///
/// This approach sorts both arrays and maps quantiles directly:
/// - For each source pixel at rank r, assign the reference value at the equivalent rank
/// - With linear interpolation, smoothly interpolates between adjacent reference values
/// - Uses random tie-breaking to avoid banding artifacts on flat regions
///
/// Args:
///     source: Source values (any range, typically 0.0-255.0)
///     reference: Reference values (same range as source)
///     interp_mode: Interpolation mode (Linear recommended for smoothness)
///     align_mode: Alignment mode (Endpoint preserves ref extremes, Midpoint is statistically correct)
///     seed: Random seed for tie-breaking (use different values per pass to reduce noise when averaging)
///
/// Returns:
///     Matched values with same length as source
pub fn match_histogram_f32(
    source: &[f32],
    reference: &[f32],
    interp_mode: InterpolationMode,
    align_mode: AlignmentMode,
    seed: u32,
) -> Vec<f32> {
    if source.is_empty() || reference.is_empty() {
        return source.to_vec();
    }

    let src_len = source.len();
    let ref_len = reference.len();

    // Get sorted indices for source (argsort) with random tie-breaking.
    // This prevents horizontal banding when flat regions map to varying reference.
    // Hash the seed so consecutive seeds (0,1,2...) produce uncorrelated patterns.
    let hashed_seed = lowbias32(seed);
    let mut src_indices: Vec<usize> = (0..src_len).collect();
    src_indices.sort_unstable_by(|&a, &b| {
        match source[a].partial_cmp(&source[b]) {
            Some(std::cmp::Ordering::Equal) | None => {
                // Random tie-breaking using Wang hash of index XOR hashed seed
                lowbias32(a as u32 ^ hashed_seed).cmp(&lowbias32(b as u32 ^ hashed_seed))
            }
            Some(ord) => ord,
        }
    });

    // Sort reference values
    let mut ref_sorted: Vec<f32> = reference.to_vec();
    ref_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut output = vec![0.0f32; src_len];

    // Handle edge case: single source pixel
    if src_len == 1 {
        output[src_indices[0]] = match interp_mode {
            InterpolationMode::Nearest => ref_sorted[ref_len / 2],
            InterpolationMode::Linear => {
                // Return median of reference
                if ref_len % 2 == 0 {
                    (ref_sorted[ref_len / 2 - 1] + ref_sorted[ref_len / 2]) / 2.0
                } else {
                    ref_sorted[ref_len / 2]
                }
            }
        };
        return output;
    }

    for (rank, &src_idx) in src_indices.iter().enumerate() {
        // Map rank from source space to reference space
        let ref_rank_f = match align_mode {
            AlignmentMode::Endpoint => {
                // Endpoint-aligned: rank 0 → 0, rank (n-1) → (m-1)
                (rank as f64) * ((ref_len - 1) as f64) / ((src_len - 1).max(1) as f64)
            }
            AlignmentMode::Midpoint => {
                // Midpoint-aligned: treat each rank as bin center
                // (rank + 0.5) * ref_len / src_len - 0.5
                (rank as f64 + 0.5) * (ref_len as f64) / (src_len as f64) - 0.5
            }
        };

        match interp_mode {
            InterpolationMode::Nearest => {
                let ref_idx = ref_rank_f.round().clamp(0.0, (ref_len - 1) as f64) as usize;
                output[src_idx] = ref_sorted[ref_idx];
            }
            InterpolationMode::Linear => {
                let ref_lo = ref_rank_f.floor().max(0.0) as usize;
                let ref_hi = (ref_lo + 1).min(ref_len - 1);
                let t = (ref_rank_f - ref_lo as f64).clamp(0.0, 1.0) as f32;
                output[src_idx] = ref_sorted[ref_lo] * (1.0 - t) + ref_sorted[ref_hi] * t;
            }
        }
    }

    output
}

/// Linear interpolation helper
/// Given x, find the corresponding value by interpolating between xp and fp
fn interpolate(x: f32, xp: &[f32], fp: &[u8]) -> u8 {
    if xp.is_empty() {
        return 0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }

    // Binary search for interval
    let mut lo = 0usize;
    let mut hi = xp.len() - 1;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation between fp[lo] and fp[hi]
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    let result = fp[lo] as f32 + t * (fp[hi] as f32 - fp[lo] as f32);
    result.round() as u8
}

/// Match histogram of source to reference
/// Both source and reference are flat arrays of u8 values (single channel)
pub fn match_histogram(source: &[u8], reference: &[u8]) -> Vec<u8> {
    if source.is_empty() {
        return Vec::new();
    }

    // Step 1: Count occurrences of each value (0-255)
    let mut src_counts = [0u64; 256];
    let mut ref_counts = [0u64; 256];

    for &pixel in source {
        src_counts[pixel as usize] += 1;
    }
    for &pixel in reference {
        ref_counts[pixel as usize] += 1;
    }

    // Step 2: Compute normalized CDFs
    let src_total = source.len() as f32;
    let ref_total = reference.len() as f32;

    let mut src_cdf = [0.0f32; 256];
    let mut ref_cdf = [0.0f32; 256];

    let mut cumsum = 0u64;
    for i in 0..256 {
        cumsum += src_counts[i];
        src_cdf[i] = cumsum as f32 / src_total;
    }

    cumsum = 0;
    for i in 0..256 {
        cumsum += ref_counts[i];
        ref_cdf[i] = cumsum as f32 / ref_total;
    }

    // Step 3: Build lookup table via interpolation
    // Collect reference values that actually appear (non-zero counts)
    let mut ref_values: Vec<u8> = Vec::new();
    let mut ref_quantiles: Vec<f32> = Vec::new();

    for i in 0..256 {
        if ref_counts[i] > 0 {
            ref_values.push(i as u8);
            ref_quantiles.push(ref_cdf[i]);
        }
    }

    // Handle edge case: reference has no values
    if ref_values.is_empty() {
        return source.to_vec();
    }

    // Build lookup table
    let mut lookup = [0u8; 256];
    for i in 0..256 {
        let q = src_cdf[i];
        lookup[i] = interpolate(q, &ref_quantiles, &ref_values);
    }

    // Step 4: Apply lookup table
    source.iter().map(|&pixel| lookup[pixel as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_histogram_identity() {
        // Matching histogram to itself should return the same values
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let result = match_histogram(&data, &data);
        for (a, b) in data.iter().zip(result.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn test_match_histogram_simple() {
        // Source: all 0s, Reference: all 255s
        // Result should be all 255s
        let source = vec![0u8; 100];
        let reference = vec![255u8; 100];
        let result = match_histogram(&source, &reference);
        for &v in &result {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_match_histogram_f32_identity() {
        // Matching histogram to itself should preserve relative ordering
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let result = match_histogram_f32(&data, &data, InterpolationMode::Linear, AlignmentMode::Endpoint, 0);
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a - b).abs() < 0.01, "Expected {} but got {}", a, b);
        }
    }

    #[test]
    fn test_match_histogram_f32_simple() {
        // Source: all 0s, Reference: all 255s
        // Result should be all 255s
        let source = vec![0.0f32; 100];
        let reference = vec![255.0f32; 100];
        let result = match_histogram_f32(&source, &reference, InterpolationMode::Linear, AlignmentMode::Endpoint, 0);
        for &v in &result {
            assert!((v - 255.0).abs() < 0.01, "Expected 255.0 but got {}", v);
        }
    }

    #[test]
    fn test_match_histogram_f32_different_sizes() {
        // Source has 10 values, reference has 100 values
        let source: Vec<f32> = (0..10).map(|i| i as f32 * 25.5).collect();
        let reference: Vec<f32> = (0..100).map(|i| i as f32 * 2.55).collect();
        let result = match_histogram_f32(&source, &reference, InterpolationMode::Linear, AlignmentMode::Endpoint, 0);
        assert_eq!(result.len(), 10);
        // Results should be monotonic (preserving order)
        for i in 1..result.len() {
            assert!(
                result[i] >= result[i - 1],
                "Not monotonic at {}: {} < {}",
                i,
                result[i],
                result[i - 1]
            );
        }
    }

    #[test]
    fn test_match_histogram_f32_nearest_vs_linear() {
        let source: Vec<f32> = vec![0.0, 50.0, 100.0, 150.0, 200.0];
        let reference: Vec<f32> = vec![10.0, 110.0, 210.0];

        let nearest = match_histogram_f32(&source, &reference, InterpolationMode::Nearest, AlignmentMode::Endpoint, 0);
        let linear = match_histogram_f32(&source, &reference, InterpolationMode::Linear, AlignmentMode::Endpoint, 0);

        // Both should have same length
        assert_eq!(nearest.len(), source.len());
        assert_eq!(linear.len(), source.len());

        // Nearest should only have values from reference
        for &v in &nearest {
            assert!(
                (v - 10.0).abs() < 0.01 || (v - 110.0).abs() < 0.01 || (v - 210.0).abs() < 0.01,
                "Nearest value {} not in reference",
                v
            );
        }
    }

    #[test]
    fn test_match_histogram_f32_endpoint_vs_midpoint() {
        // With 3 source values and 1000 reference values
        let source: Vec<f32> = vec![0.0, 127.5, 255.0];
        let reference: Vec<f32> = (0..1000).map(|i| i as f32 * 0.255).collect();

        let endpoint = match_histogram_f32(&source, &reference, InterpolationMode::Linear, AlignmentMode::Endpoint, 0);
        let midpoint = match_histogram_f32(&source, &reference, InterpolationMode::Linear, AlignmentMode::Midpoint, 0);

        // Endpoint should hit exact extremes of reference
        assert!((endpoint[0] - 0.0).abs() < 0.01, "Endpoint min: {}", endpoint[0]);
        assert!((endpoint[2] - 254.745).abs() < 0.01, "Endpoint max: {}", endpoint[2]);

        // Midpoint should sample from quantile centers, not extremes
        // rank 0: (0 + 0.5) * 1000 / 3 - 0.5 = 166.17 → ref ~= 42.4
        // rank 2: (2 + 0.5) * 1000 / 3 - 0.5 = 832.83 → ref ~= 212.4
        assert!(midpoint[0] > 40.0 && midpoint[0] < 45.0, "Midpoint min: {}", midpoint[0]);
        assert!(midpoint[2] > 210.0 && midpoint[2] < 215.0, "Midpoint max: {}", midpoint[2]);
    }
}
