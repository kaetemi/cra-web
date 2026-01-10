/// Histogram matching implementation for uint8 images.
/// Based on the algorithm described in PORTING.md

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

/// Match histogram for multi-channel image
/// img and reference are flat arrays with interleaved channels (e.g., RGBRGBRGB)
#[allow(dead_code)]
pub fn match_histogram_multichannel(
    source: &[u8],
    reference: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let pixels = width * height;
    let mut output = vec![0u8; pixels * channels];

    for ch in 0..channels {
        // Extract channel
        let src_channel: Vec<u8> = (0..pixels).map(|i| source[i * channels + ch]).collect();
        let ref_channel: Vec<u8> = (0..pixels).map(|i| reference[i * channels + ch]).collect();

        // Match histogram
        let matched = match_histogram(&src_channel, &ref_channel);

        // Store result
        for i in 0..pixels {
            output[i * channels + ch] = matched[i];
        }
    }

    output
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
}
