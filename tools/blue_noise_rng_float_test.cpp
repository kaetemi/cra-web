/*
 * Blue Noise RNG - Floating Point Test Tool
 *
 * C++ test binary for analyzing the floating-point output of the
 * blue noise RNG. Outputs raw doubles or uint32 values to stdout
 * for analysis by the companion Python script.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o tools/blue_noise_rng_float_test tools/blue_noise_rng_float_test.cpp
 *
 * Usage:
 *   blue_noise_rng_float_test <count> [bit_depth] [seed]
 *   blue_noise_rng_float_test --raw-f64 <count> [bit_depth] [seed]
 *   blue_noise_rng_float_test --raw-u32 <count> [bit_depth] [seed]
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

// pseudo-random number generator with ~6db/octave blue noise
// this generator produces a uniform distribution
// important: blue noise properties cannot be preserved when
// the generator is used for multiple purposes simultaneously
// nor when multiple next calls are used to construct a larger value
// nor when integer outputs are used with the modulo operator
struct blue_noise_rng {
    uint8_t  bit_depth = 0;
    uint32_t seed      = 0;
    uint32_t position  = 0;

    // binary tree of 1-bit 50% duty cycle error diffusion dithering blue noise generators
    std::vector<std::array<int8_t, 2>> states; // {err0, err1} per tree node

    blue_noise_rng() = default;

    blue_noise_rng(uint8_t bit_depth, uint32_t seed) {
        init(bit_depth, seed);
    }

    static uint32_t hash(uint32_t x) { // lowbias32
        x ^= x >> 16; x *= 0x21f0aaad;
        x ^= x >> 15; x *= 0x735a2d97;
        x ^= x >> 15;
        return x;
    }

    void init(uint8_t depth, uint32_t s) {
        bit_depth = std::clamp<uint8_t>(depth, 1, 16);
        seed      = hash(s);

        const int n = (1 << bit_depth) - 1;
        states.resize(n);

        reset();
    }

    void reset() {
        const int n = (int)states.size();
        position  = 0;

        // 5 reachable states with distribution 3:3:2:1:1
        static const int8_t tbl[10][2] = {
            { 0,  0}, { 0,  0}, { 0,  0},
            {-1,  0}, {-1,  0}, {-1,  0},
            { 0, -1}, { 0, -1},
            {-2,  0},
            {-1, -1},
        };
        for (int i = 0; i < n; i++) {
            uint32_t h = hash((uint32_t)i ^ seed) % 10;
            states[i] = {tbl[h][0], tbl[h][1]}; // random initial state
        }
    }

    uint16_t next(uint32_t * hash_remainder = nullptr) {
        uint32_t h = hash(position ^ seed);
        position++;

        // traverse binary tree, one error diffusion ditherer per population split
        // thresholding output at any value still produces blue noise
        uint32_t acc = 0;
        for (int level = 0; level < bit_depth; level++) {
            auto & s = states[(1 << level) - 1 + acc]; // heap-style index

            int    out = (s[0] >= 0) ? 1 : 0;
            int8_t qe  = s[0] + (int8_t)(out ? -1 : 1); // inverse autocorrelation

            s[0] = s[1]; // step forward
            s[1] = 0;

            // error diffusion dithering using binary weight perturbation
            s[(h >> level) & 1 ? 0 : 1] += qe; // forward to t+1 or defer to t+2

            acc = acc * 2 + out;
        }

        if (hash_remainder) {
            *hash_remainder = h >> bit_depth; // unused bits from random hash
        }

        return (uint16_t)acc;
    }

    // blue noise in the upper bit_depth bits, white noise hash remainder in the lower bits
    // do not use with modulo operator, as it would just produce white noise
    uint32_t next32() {
        uint32_t rem;
        uint32_t val = next(&rem);
        return (val << (32 - bit_depth)) | rem;
    }

    // uniform double in [0, 1) with blue noise temporal autocorrelation
    double nextf() {
        double res = 0.0;
        res += hash(position ^ ~seed); // fill low bits with white noise
        res *= 1.0 / 4294967296.0;
        res += next32();
        res *= 1.0 / 4294967296.0;
        if (res >= 1.0) res = std::nextafter(1.0, 0.0);
        return res;
    }
};

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <count> [bit_depth] [seed]          Print stats\n", prog);
    fprintf(stderr, "  %s --raw-f64 <count> [bit_depth] [seed]  Raw float64 to stdout\n", prog);
    fprintf(stderr, "  %s --raw-u32 <count> [bit_depth] [seed]  Raw uint32 to stdout\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  bit_depth: 1-16 (default: 8)\n");
    fprintf(stderr, "  seed:      any integer (default: 12345)\n");
}

int main(int argc, char *argv[]) {
    enum { MODE_PRINT, MODE_RAW_F64, MODE_RAW_U32 } mode = MODE_PRINT;
    int arg_offset = 1;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--raw-f64") == 0) {
        mode = MODE_RAW_F64;
        arg_offset = 2;
    } else if (strcmp(argv[1], "--raw-u32") == 0) {
        mode = MODE_RAW_U32;
        arg_offset = 2;
    } else if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc < arg_offset + 1) {
        print_usage(argv[0]);
        return 1;
    }

    int count = atoi(argv[arg_offset]);
    int bit_depth = (argc > arg_offset + 1) ? atoi(argv[arg_offset + 1]) : 8;
    uint32_t seed = (argc > arg_offset + 2) ? (uint32_t)strtoul(argv[arg_offset + 2], nullptr, 10) : 12345;

    if (count <= 0) {
        fprintf(stderr, "Error: count must be > 0\n");
        return 1;
    }
    if (bit_depth < 1 || bit_depth > 16) {
        fprintf(stderr, "Error: bit_depth must be 1-16\n");
        return 1;
    }

    blue_noise_rng rng((uint8_t)bit_depth, seed);

    if (mode == MODE_RAW_F64) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        for (int i = 0; i < count; i++) {
            double val = rng.nextf();
            fwrite(&val, sizeof(double), 1, stdout);
        }
    } else if (mode == MODE_RAW_U32) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        for (int i = 0; i < count; i++) {
            uint32_t val = rng.next32();
            fwrite(&val, sizeof(uint32_t), 1, stdout);
        }
    } else {
        // Print mode: show some stats
        printf("Blue Noise RNG Float Test\n");
        printf("  bit_depth: %d\n", bit_depth);
        printf("  seed:      %u\n", seed);
        printf("  count:     %d\n\n", count);

        double sum = 0, sum2 = 0;
        double min_val = 1.0, max_val = 0.0;

        // Show first 20 values
        printf("First 20 values (nextf):\n");
        for (int i = 0; i < count; i++) {
            double val = rng.nextf();
            if (i < 20) {
                printf("  [%2d] %.15f\n", i, val);
            }
            sum += val;
            sum2 += val * val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        double mean = sum / count;
        double variance = sum2 / count - mean * mean;
        double stddev = sqrt(variance);

        printf("\nStatistics:\n");
        printf("  mean:   %.10f  (ideal: 0.5)\n", mean);
        printf("  stddev: %.10f  (ideal: %.10f)\n", stddev, 1.0 / sqrt(12.0));
        printf("  min:    %.15f\n", min_val);
        printf("  max:    %.15f\n", max_val);
    }

    return 0;
}
