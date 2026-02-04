/*
 * blue_noise_rng.h - Blue noise random number generator
 *
 * Single-header library for generating blue noise distributed random integers.
 *
 * THE TECHNIQUE
 * =============
 * Generates multi-bit output values (e.g., 0-255 for 8-bit depth) with blue
 * noise temporal properties. Each output value is determined by N binary
 * decisions (N = bit depth), each performed by an independent 1D error
 * diffusion ditherer operating at 50% duty cycle.
 *
 * The ditherers are arranged in a binary tree:
 *   Level 0: 1 ditherer determines the MSB
 *   Level 1: 2 ditherers (one for "went high", one for "went low")
 *   Level 2: 4 ditherers
 *   ...
 *   Level N-1: 2^(N-1) ditherers determine the LSB
 *   Total: 2^N - 1 ditherer states
 *
 * Each ditherer uses the mixed-kernel 1D blue noise technique from
 * blue_dither.h: randomly selecting between a tight [1] kernel and a
 * spread [7/12, 5/12] kernel using lowbias32 hash bits.
 *
 * EFFICIENCY
 * ==========
 * Each output value only processes N states (one per bit level), traversing
 * a single root-to-leaf path in the binary tree. A single lowbias32 hash
 * per output provides kernel selection bits for all N levels.
 *
 * PROPERTIES
 * ==========
 * - Output range: 0 to 2^bit_depth - 1
 * - Nearly uniform distribution over the output range
 * - Blue noise temporal autocorrelation (consecutive values repel)
 * - Scale-invariant: thresholding output at any level gives blue noise
 * - MSB has strongest blue noise property (most samples per state)
 *
 * MEMORY
 * ======
 * Fixed allocation: 2^N - 1 states x 8 bytes each (max N=8)
 *   bit_depth=1:    1 state  =    8 bytes
 *   bit_depth=4:   15 states =  120 bytes
 *   bit_depth=8:  255 states = 2040 bytes
 * Plus 12 bytes overhead (bit_depth, seed, position).
 * No heap allocation required.
 *
 * Usage:
 *   #define BLUE_NOISE_RNG_IMPLEMENTATION
 *   #include "blue_noise_rng.h"
 *
 * Example:
 *   BlueNoiseRng rng;
 *   blue_noise_rng_init(&rng, 8, 12345);  // 8-bit, seed 12345
 *   for (int i = 0; i < 1000; i++) {
 *       uint8_t val = blue_noise_rng_next(&rng);
 *       // val is 0-255 with blue noise temporal distribution
 *   }
 *
 * BSD 3-Clause License
 *
 * Copyright (C) 2026 Jan BOON (Kaetemi) jan.boon@kaetemi.be
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef BLUE_NOISE_RNG_H
#define BLUE_NOISE_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum supported bit depth */
#define BLUE_NOISE_RNG_MAX_DEPTH 8

/* Maximum states in binary tree: 2^MAX_DEPTH - 1 */
#define BLUE_NOISE_RNG_MAX_STATES 255

/* Per-node error diffusion state */
typedef struct {
    int32_t err0;   /* Error for current step */
    int32_t err1;   /* Error for next step (deferred) */
} BlueNoiseRngState;

/* Blue noise RNG context */
typedef struct {
    uint8_t bit_depth;      /* Number of output bits (1-8) */
    uint32_t seed;          /* Random seed */
    uint32_t position;      /* Position counter for hash */
    BlueNoiseRngState states[BLUE_NOISE_RNG_MAX_STATES];
} BlueNoiseRng;

/* Initialize RNG. bit_depth: 1-8, seed: any uint32_t */
void blue_noise_rng_init(BlueNoiseRng *rng, uint8_t bit_depth, uint32_t seed);

/* Generate next value. Returns 0 to 2^bit_depth - 1 */
uint8_t blue_noise_rng_next(BlueNoiseRng *rng);

/* Reset state (clears error, resets position counter) */
void blue_noise_rng_reset(BlueNoiseRng *rng);

/* Utility: lowbias32 hash function (same as blue_dither.h) */
static inline uint32_t blue_noise_rng_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x21f0aaad;
    x ^= x >> 15;
    x *= 0x735a2d97;
    x ^= x >> 15;
    return x;
}

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * IMPLEMENTATION
 * ============================================================================ */
#ifdef BLUE_NOISE_RNG_IMPLEMENTATION

#include <string.h>

void blue_noise_rng_init(BlueNoiseRng *rng, uint8_t bit_depth, uint32_t seed) {
    if (bit_depth < 1) bit_depth = 1;
    if (bit_depth > BLUE_NOISE_RNG_MAX_DEPTH) bit_depth = BLUE_NOISE_RNG_MAX_DEPTH;
    rng->bit_depth = bit_depth;
    rng->seed = seed;
    rng->position = 0;
    memset(rng->states, 0, sizeof(rng->states));

    /* Warmup: run each state individually through 256 error diffusion
     * steps at 50% duty cycle. This fills the error buffers so the first
     * outputs don't exhibit the cold-start pattern (all states at zero
     * error produce a deterministic sequence).
     * Hash uses (warmup_step, state_index, seed) to give each state a
     * unique kernel selection sequence during warmup. */
    int num_states = (1 << bit_depth) - 1;
    for (int si = 0; si < num_states; si++) {
        BlueNoiseRngState *s = &rng->states[si];
        for (int w = 0; w < 256; w++) {
            int output = (s->err0 >= 0) ? 1 : 0;
            int32_t quant_err = s->err0 + (output ? -1 : 1);

            uint32_t hash = blue_noise_rng_hash(
                (uint32_t)w ^ ((uint32_t)si << 16) ^ seed);

            s->err0 = s->err1;
            s->err1 = 0;

            if (hash & 1) {
                s->err0 += quant_err;
            } else {
                s->err1 += quant_err;
            }
        }
    }
}

void blue_noise_rng_reset(BlueNoiseRng *rng) {
    rng->position = 0;
    memset(rng->states, 0, sizeof(rng->states));
}

uint8_t blue_noise_rng_next(BlueNoiseRng *rng) {
    /* Single hash provides kernel selection bits for all levels */
    uint32_t hash = blue_noise_rng_hash(rng->position ^ rng->seed);
    rng->position++;

    /*
     * Population splitting: traverse binary tree from root to leaf.
     * At each level, a 1D error diffusion ditherer decides high/low
     * at 50% duty cycle. Hash bit selects: forward error immediately
     * (tight) or defer to t+2 (spread).
     *
     * Error is tracked as integers: output = (err0 >= 0), then
     * quant_err = err0 + (output ? -1 : 1).
     *
     * Tree indexing (heap-style):
     *   Level k, accumulated path v -> index = 2^k - 1 + v
     *   Level 0: index 0 (root)
     *   Level 1: indices 1, 2
     *   Level 2: indices 3, 4, 5, 6
     *   ...
     *   Level 7: indices 127..254
     */
    int accumulated = 0;

    for (int level = 0; level < rng->bit_depth; level++) {
        int idx = (1 << level) - 1 + accumulated;
        BlueNoiseRngState *s = &rng->states[idx];

        int output = (s->err0 >= 0) ? 1 : 0;
        int32_t quant_err = s->err0 + (output ? -1 : 1);

        /* Shift error buffer */
        s->err0 = s->err1;
        s->err1 = 0;

        if ((hash >> level) & 1) {
            /* Tight: forward error to t+1 */
            s->err0 += quant_err;
        } else {
            /* Spread: defer error to t+2 */
            s->err1 += quant_err;
        }

        /* Build output value MSB-first */
        accumulated = accumulated * 2 + output;
    }

    return (uint8_t)accumulated;
}

#endif /* BLUE_NOISE_RNG_IMPLEMENTATION */
#endif /* BLUE_NOISE_RNG_H */
