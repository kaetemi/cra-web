/*
 * blue_noise_rng.c - Blue noise RNG demonstration and testing
 *
 * Demonstrates the blue noise random number generator and prints
 * statistics to verify correct operation.
 *
 * Usage:
 *   blue_noise_rng <count> [bit_depth] [seed]         Print values and stats
 *   blue_noise_rng --raw <count> [bit_depth] [seed]    Raw bytes to stdout
 *
 * Build:
 *   gcc -O2 -o tools/blue_noise_rng tools/blue_noise_rng.c -lm
 */

#define BLUE_NOISE_RNG_IMPLEMENTATION
#include "blue_noise_rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <count> [bit_depth] [seed]         Print values and statistics\n", prog);
    fprintf(stderr, "  %s --raw <count> [bit_depth] [seed]   Output raw bytes to stdout\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "bit_depth: 1-8 (default: 8)\n");
    fprintf(stderr, "seed:      any integer (default: 12345)\n");
}

static void demo_print(int count, int bit_depth, uint32_t seed) {
    BlueNoiseRng rng;
    blue_noise_rng_init(&rng, (uint8_t)bit_depth, seed);

    int max_val = (1 << bit_depth) - 1;
    int n_bins = max_val + 1;

    printf("Blue Noise RNG Demo\n");
    printf("Bit depth: %d (range 0-%d)\n", bit_depth, max_val);
    printf("Seed: %u\n", seed);
    printf("Count: %d\n", count);
    printf("States: %d (2^%d - 1)\n", (1 << bit_depth) - 1, bit_depth);

    /* Generate all values */
    uint8_t *values = (uint8_t *)malloc(count);
    if (!values) {
        fprintf(stderr, "Failed to allocate %d bytes\n", count);
        return;
    }

    for (int i = 0; i < count; i++) {
        values[i] = blue_noise_rng_next(&rng);
    }

    /* Print first N values */
    int show = count < 80 ? count : 80;
    printf("\nFirst %d values:", show);
    for (int i = 0; i < show; i++) {
        if (i % 20 == 0) printf("\n  ");
        printf("%3d ", values[i]);
    }
    printf("\n");

    /* Statistics */
    double sum = 0, sum_sq = 0;
    int min_seen = max_val, max_seen = 0;

    for (int i = 0; i < count; i++) {
        sum += values[i];
        sum_sq += (double)values[i] * values[i];
        if (values[i] < min_seen) min_seen = values[i];
        if (values[i] > max_seen) max_seen = values[i];
    }

    double mean = sum / count;
    double variance = sum_sq / count - mean * mean;
    double stddev = sqrt(variance);

    /* Ideal for discrete uniform on {0, ..., M}: mean = M/2, var = M(M+2)/12 */
    double ideal_mean = max_val / 2.0;
    double ideal_stddev = sqrt((double)max_val * (max_val + 2) / 12.0);

    printf("\nStatistics:\n");
    printf("  Mean:     %.2f (ideal: %.2f)\n", mean, ideal_mean);
    printf("  Std dev:  %.2f (ideal: %.2f)\n", stddev, ideal_stddev);
    printf("  Min/Max:  %d/%d\n", min_seen, max_seen);

    /* Histogram */
    int *histogram = (int *)calloc(n_bins, sizeof(int));
    for (int i = 0; i < count; i++) {
        histogram[values[i]]++;
    }

    /* Binned histogram display */
    int hist_bins = n_bins <= 16 ? n_bins : 16;
    int bin_size = n_bins / hist_bins;
    int *binned = (int *)calloc(hist_bins, sizeof(int));
    int max_count = 0;

    for (int i = 0; i < hist_bins; i++) {
        for (int j = 0; j < bin_size; j++) {
            int idx = i * bin_size + j;
            if (idx < n_bins) binned[i] += histogram[idx];
        }
        if (binned[i] > max_count) max_count = binned[i];
    }

    printf("\nHistogram (%d bins):\n", hist_bins);
    int bar_width = 30;
    for (int i = 0; i < hist_bins; i++) {
        int lo = i * bin_size;
        int hi = lo + bin_size - 1;
        int bar_len = max_count > 0 ? (binned[i] * bar_width + max_count / 2) / max_count : 0;
        printf("  [%3d-%3d] ", lo, hi);
        for (int b = 0; b < bar_len; b++) putchar('#');
        for (int b = bar_len; b < bar_width; b++) putchar(' ');
        printf(" %d\n", binned[i]);
    }

    /* Consecutive difference analysis */
    if (count > 1) {
        double diff_sum = 0;
        int max_diff = 0;

        for (int i = 1; i < count; i++) {
            int diff = abs((int)values[i] - (int)values[i - 1]);
            diff_sum += diff;
            if (diff > max_diff) max_diff = diff;
        }
        double mean_diff = diff_sum / (count - 1);

        /* Ideal E[|X-Y|] for uniform on {0,...,M}: M/3 */
        double ideal_diff = max_val / 3.0;

        printf("\nConsecutive differences:\n");
        printf("  Mean |diff|: %.1f (white noise: %.1f)\n", mean_diff, ideal_diff);
        printf("  Max |diff|:  %d\n", max_diff);
        if (mean_diff > ideal_diff) {
            printf("  -> Values more spread than white noise (blue noise property)\n");
        } else {
            printf("  -> Values less spread than white noise\n");
        }
    }

    free(histogram);
    free(binned);
    free(values);
}

static void demo_raw(int count, int bit_depth, uint32_t seed) {
    BlueNoiseRng rng;
    blue_noise_rng_init(&rng, (uint8_t)bit_depth, seed);

    for (int i = 0; i < count; i++) {
        uint8_t val = blue_noise_rng_next(&rng);
        fwrite(&val, 1, 1, stdout);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int raw_mode = 0;
    int arg_offset = 1;

    if (strcmp(argv[1], "--raw") == 0) {
        raw_mode = 1;
        arg_offset = 2;
    }

    if (argc < arg_offset + 1) {
        print_usage(argv[0]);
        return 1;
    }

    int count = atoi(argv[arg_offset]);
    int bit_depth = (argc > arg_offset + 1) ? atoi(argv[arg_offset + 1]) : 8;
    uint32_t seed = (argc > arg_offset + 2) ? (uint32_t)atoi(argv[arg_offset + 2]) : 12345;

    if (count <= 0) {
        fprintf(stderr, "Error: count must be positive\n");
        return 1;
    }
    if (bit_depth < 1 || bit_depth > 8) {
        fprintf(stderr, "Error: bit_depth must be 1-8\n");
        return 1;
    }

    if (raw_mode) {
        demo_raw(count, bit_depth, seed);
    } else {
        demo_print(count, bit_depth, seed);
    }

    return 0;
}
