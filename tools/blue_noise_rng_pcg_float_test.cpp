/*
 * Blue Noise RNG (PCG64-DXSM variant) - Floating Point Test Tool
 *
 * Same blue noise RNG structure but using PCG64-DXSM instead of lowbias32/mt19937
 * for the random source. For comparison testing.
 *
 * PCG64-DXSM: 128-bit LCG state, 64-bit output with DXSM permutation.
 * Melissa O'Neill, 2019. Implementation based on public domain references.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o tools/blue_noise_rng_pcg_float_test tools/blue_noise_rng_pcg_float_test.cpp
 *
 * Usage:
 *   blue_noise_rng_pcg_float_test <count> [bit_depth] [seed]
 *   blue_noise_rng_pcg_float_test --raw-f64 <count> [bit_depth] [seed]
 *   blue_noise_rng_pcg_float_test --raw-u32 <count> [bit_depth] [seed]
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <random>

// --- PCG64-DXSM implementation (portable, matches llama_dist_rng_pcg64_dxsm) ---
// reference: https://dotat.at/@/2023-06-21-pcg64-dxsm.html

struct pcg64_state {
    struct u128 { uint64_t lo, hi; };

    static constexpr uint64_t MUL_64 = 0xDA942042E4DD58B5ULL;

    u128 state;
    u128 inc;
    u128 init_state; // saved for reset()

    // 128x64 multiply
    static u128 mul128x64(u128 a, uint64_t b) {
#ifdef __SIZEOF_INT128__
        // compiler has native 128-bit support (GCC, Clang)
        unsigned __int128 full = (unsigned __int128)a.lo * b;
        uint64_t lo = (uint64_t)full;
        uint64_t hi = (uint64_t)(full >> 64) + a.hi * b;
        return {lo, hi};
#elif defined(_MSC_VER) && defined(_M_X64)
        // MSVC on x64
        uint64_t hi_lo;
        uint64_t lo = _umul128(a.lo, b, &hi_lo);
        uint64_t hi = hi_lo + a.hi * b;
        return {lo, hi};
#else
        // 32-bit fallback
        uint64_t a0 = a.lo & 0xFFFFFFFF;
        uint64_t a1 = a.lo >> 32;
        uint64_t b0 = b & 0xFFFFFFFF;
        uint64_t b1 = b >> 32;

        uint64_t p0 = a0 * b0;
        uint64_t p1 = a0 * b1;
        uint64_t p2 = a1 * b0;
        uint64_t p3 = a1 * b1;

        uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF);
        uint64_t lo  = (p0 & 0xFFFFFFFF) | (mid << 32);
        uint64_t hi  = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32) + a.hi * b;

        return {lo, hi};
#endif
    }

    static u128 add128(u128 a, u128 b) {
        uint64_t lo = a.lo + b.lo;
        uint64_t hi = a.hi + b.hi + (lo < a.lo ? 1 : 0);
        return {lo, hi};
    }

    void step() {
        state = add128(mul128x64(state, MUL_64), inc);
    }

    uint64_t output() const {
        uint64_t hi = state.hi;
        uint64_t lo = state.lo | 1;
        hi ^= hi >> 32;
        hi *= MUL_64;
        hi ^= hi >> 48;
        hi *= lo;
        return hi;
    }

    void seed_init(uint32_t seed) {
        // use std::seed_seq to derive both state and increment from a single seed
        std::seed_seq seq{seed};
        uint32_t vals[8];
        seq.generate(vals, vals + 8);

        uint64_t s_lo = (uint64_t)vals[1] << 32 | vals[0];
        uint64_t s_hi = (uint64_t)vals[3] << 32 | vals[2];
        uint64_t i_lo = (uint64_t)vals[5] << 32 | vals[4];
        uint64_t i_hi = (uint64_t)vals[7] << 32 | vals[6];

        inc = { (i_lo << 1) | 1, (i_hi << 1) | (i_lo >> 63) };
        state = {0, 0};
        step();
        state = add128(state, {s_lo, s_hi});
        step();
        init_state = state;
    }

    uint64_t next_raw() {
        uint64_t out = output();
        step();
        return out;
    }

    uint32_t next32() {
        return (uint32_t)(next_raw() >> 32);
    }

    uint64_t next64() {
        return next_raw();
    }

    double nextf() {
        return (next_raw() >> 11) * 0x1.0p-53;
    }

    void reset() {
        state = init_state;
    }
};

// --- Blue noise RNG using PCG64-DXSM ---

struct blue_noise_rng_pcg {
    uint8_t  bit_depth = 0;
    uint32_t seed      = 0;

    pcg64_state pcg;

    // binary tree of 1-bit 50% duty cycle error diffusion dithering blue noise generators
    std::vector<std::array<int8_t, 2>> states; // {err0, err1} per tree node

    blue_noise_rng_pcg() = default;

    blue_noise_rng_pcg(uint8_t bit_depth, uint32_t seed) {
        init(bit_depth, seed);
    }

    void init(uint8_t depth, uint32_t s) {
        bit_depth = std::clamp<uint8_t>(depth, 1, 16);
        seed      = s;
        pcg.seed_init(s);

        const int n = (1 << bit_depth) - 1;
        states.resize(n);

        reset_states();
    }

    void reseed(uint32_t s) {
        seed = s;
        pcg.seed_init(s);
        reset_states();
    }

    void reset_states() {
        const int n = (int)states.size();

        // 5 reachable states with distribution 3:3:2:1:1
        static const int8_t tbl[10][2] = {
            { 0,  0}, { 0,  0}, { 0,  0},
            {-1,  0}, {-1,  0}, {-1,  0},
            { 0, -1}, { 0, -1},
            {-2,  0},
            {-1, -1},
        };
        for (int i = 0; i < n; i++) {
            uint32_t h = pcg.next32() % 10;
            states[i] = {tbl[h][0], tbl[h][1]}; // random initial state
        }
    }

    uint16_t advance(uint32_t h) {
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
            s[(h >> (31 - level)) & 1 ? 0 : 1] += qe; // forward to t+1 or defer to t+2

            acc = acc * 2 + out;
        }
        return (uint16_t)acc;
    }

    uint16_t next() {
        uint32_t h = pcg.next32();
        return advance(h);
    }

    // blue noise in the upper bit_depth bits, white noise in the lower bits
    uint32_t next32() {
        uint32_t h   = pcg.next32();
        uint32_t val = advance(h);
        return (val << (32 - bit_depth)) | (h & ((1u << (32 - bit_depth)) - 1));
    }

    // blue noise in the upper bits, white noise in the lower bits
    uint64_t next64() {
        uint64_t r   = pcg.next64();
        uint32_t val = advance((uint32_t)(r >> 32));
        return ((uint64_t)val << (64 - bit_depth)) | (r & ((UINT64_C(1) << (64 - bit_depth)) - 1));
    }

    // uniform double in [0, 1) with blue noise temporal autocorrelation
    double nextf() {
        uint64_t combined = next64();
        return (combined >> 11) * 0x1.0p-53;
    }
};

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <count> [bit_depth] [seed]          Print stats\n", prog);
    fprintf(stderr, "  %s --raw-f64 <count> [bit_depth] [seed]  Raw float64 to stdout\n", prog);
    fprintf(stderr, "  %s --raw-u32 <count> [bit_depth] [seed]  Raw uint32 to stdout\n", prog);
    fprintf(stderr, "  %s --raw-white-f64 <count> [seed]        White noise float64 (PCG64-DXSM)\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  bit_depth: 1-16 (default: 8)\n");
    fprintf(stderr, "  seed:      any integer (default: 12345)\n");
}

int main(int argc, char *argv[]) {
    enum { MODE_PRINT, MODE_RAW_F64, MODE_RAW_U32, MODE_RAW_WHITE_F64 } mode = MODE_PRINT;
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
    } else if (strcmp(argv[1], "--raw-white-f64") == 0) {
        mode = MODE_RAW_WHITE_F64;
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

    if (mode == MODE_RAW_WHITE_F64) {
        // White noise using PCG64-DXSM, native 64-bit output -> double
        pcg64_state pcg_white;
        pcg_white.seed_init(seed);
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        for (int i = 0; i < count; i++) {
            double res = pcg_white.nextf();
            fwrite(&res, sizeof(double), 1, stdout);
        }
        return 0;
    }

    if (bit_depth < 1 || bit_depth > 16) {
        fprintf(stderr, "Error: bit_depth must be 1-16\n");
        return 1;
    }

    blue_noise_rng_pcg rng((uint8_t)bit_depth, seed);

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
        printf("Blue Noise RNG (PCG64-DXSM) Float Test\n");
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
