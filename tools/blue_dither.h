/*
 * blue_dither.h - Minimal integer-only blue noise dithering
 *
 * Single-header library for blue noise dithering using integer arithmetic.
 * Suitable for embedded systems, FPGAs, or anywhere floating point is expensive.
 *
 * THE TECHNIQUE
 * =============
 * Traditional error diffusion (Floyd-Steinberg, Jarvis-Judice-Ninke) produces
 * structured patterns because each kernel has characteristic periodicities.
 * Our method randomly selects between two kernels (FS and JJN) at each pixel
 * using a hash function. This breaks up the periodic structures, producing
 * blue noise characteristics: energy concentrated at high frequencies with
 * suppressed low-frequency content.
 *
 * Why it works: FS (4 coefficients, 2 rows) and JJN (12 coefficients, 3 rows)
 * have different error propagation patterns. Random mixing prevents either
 * pattern from dominating, while both preserve the key property of error
 * diffusion: the output average equals the input average.
 *
 * The result approaches ideal blue noise (-3 to -5 dB vs void-and-cluster)
 * while being computationally simpler than precomputed dither arrays.
 *
 * INTEGER MATH
 * ============
 * FS uses denominator 16, JJN uses 48. LCM = 48, so we scale everything:
 *   FS coefficients: 7/16 -> 21/48, 3/16 -> 9/48, 5/16 -> 15/48, 1/16 -> 3/48
 *   JJN coefficients: already in 48ths (7/48, 5/48, 3/48, 1/48)
 * All arithmetic uses 32-bit integers. No floating point required.
 *
 * BUFFER MODEL
 * ============
 * The error buffer is a single circular array of (width * 4) elements.
 * Width must be a power of two for fast modulo via bitmask. There are no
 * boundaries - error wraps seamlessly from row end to row start. Processing
 * is always left-to-right with continuous flow.
 *
 * API
 * ===
 * Two usage modes with a single implementation:
 *   - Row mode: process entire rows (images)
 *   - Streaming mode: get one bit at a time (LED PWM, audio, etc.)
 *
 * Usage:
 *   #define BLUE_DITHER_IMPLEMENTATION
 *   #include "blue_dither.h"
 *
 * Example - LED PWM (streaming):
 *   BlueDither bd;
 *   blue_dither_init(&bd, 256, seed);  // width must be power of 2
 *   while (1) {
 *       int on = blue_dither_next(&bd, brightness);  // 0-255
 *       set_led(on);
 *       delay_us(100);
 *   }
 *   blue_dither_free(&bd);
 *
 * Example - Image (row mode):
 *   BlueDither bd;
 *   blue_dither_init(&bd, width, seed);  // width must be power of 2
 *   for (int y = 0; y < height; y++) {
 *       blue_dither_row(&bd, input_row, output_row);
 *   }
 *   blue_dither_free(&bd);
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

#ifndef BLUE_DITHER_H
#define BLUE_DITHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t *err;       /* Circular error buffer: 4 rows × width */
    int width;          /* Row width (must be power of 2) */
    int width_mask;     /* width - 1, for fast modulo */
    int row_shift;      /* log2(width), for row indexing */
    int cur_row;        /* Current row (0-3) */
    int x;              /* Current x position in row */
    int y;              /* Current logical row number */
    uint32_t seed;      /* Random seed */
} BlueDither;

/* Initialize ditherer. Width must be power of 2. Returns 0 on success, -1 on failure. */
int blue_dither_init(BlueDither *bd, int width, uint32_t seed);

/* Free ditherer resources */
void blue_dither_free(BlueDither *bd);

/* Reset state (clears error buffer, restarts position) */
void blue_dither_reset(BlueDither *bd);

/* Process one complete row. input: 0-255 per pixel, output: 0 or 1 per pixel */
void blue_dither_row(BlueDither *bd, const uint8_t *input, uint8_t *output);

/* Get next dithered bit for streaming use. brightness: 0-255, returns: 0 or 1 */
int blue_dither_next(BlueDither *bd, uint8_t brightness);


/* ============================================================================
 * Utility: lowbias32 hash function
 * ============================================================================ */
static inline uint32_t blue_dither_hash(uint32_t x) {
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
#ifdef BLUE_DITHER_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>

/* Check if n is a power of 2 */
static inline int is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* Compute log2 for power of 2 */
static inline int log2_pow2(int n) {
    int shift = 0;
    while ((1 << shift) < n) shift++;
    return shift;
}

int blue_dither_init(BlueDither *bd, int width, uint32_t seed) {
    if (!is_power_of_2(width)) {
        return -1;  /* Width must be power of 2 */
    }

    bd->width = width;
    bd->width_mask = width - 1;
    bd->row_shift = log2_pow2(width);
    bd->seed = seed;
    bd->cur_row = 0;
    bd->x = 0;
    bd->y = 0;

    /* Allocate circular buffer: 4 rows */
    bd->err = (int32_t *)calloc(width * 4, sizeof(int32_t));
    if (!bd->err) {
        return -1;
    }
    return 0;
}

void blue_dither_free(BlueDither *bd) {
    if (bd && bd->err) {
        free(bd->err);
        bd->err = NULL;
    }
}

void blue_dither_reset(BlueDither *bd) {
    memset(bd->err, 0, bd->width * 4 * sizeof(int32_t));
    bd->cur_row = 0;
    bd->x = 0;
    bd->y = 0;
}

/* Internal: get error buffer index with circular wrap */
static inline int err_idx(BlueDither *bd, int row, int x) {
    return ((row & 3) << bd->row_shift) | (x & bd->width_mask);
}

/* Internal: process a single pixel */
static inline int blue_dither_pixel(BlueDither *bd, int x, int y, uint8_t brightness) {
    int r0 = bd->cur_row;
    int r1 = (bd->cur_row + 1) & 3;
    int r2 = (bd->cur_row + 2) & 3;

    const int32_t threshold = 6120;  /* 127.5 * 48 */
    const int32_t white_val = 12240; /* 255 * 48 */

    int32_t pixel = (int32_t)brightness * 48 + bd->err[err_idx(bd, r0, x)];

    int output;
    int32_t quant_err;
    if (pixel >= threshold) {
        output = 1;
        quant_err = pixel - white_val;
    } else {
        output = 0;
        quant_err = pixel;
    }

    uint32_t hash = blue_dither_hash((uint32_t)x ^ ((uint32_t)y << 16) ^ bd->seed);

    if (hash & 1) {
        /* Floyd-Steinberg: 21/48, 9/48, 15/48, 3/48 */
        bd->err[err_idx(bd, r0, x + 1)] += (quant_err * 21) / 48;
        bd->err[err_idx(bd, r1, x - 1)] += (quant_err * 9) / 48;
        bd->err[err_idx(bd, r1, x)]     += (quant_err * 15) / 48;
        bd->err[err_idx(bd, r1, x + 1)] += (quant_err * 3) / 48;
    } else {
        /* Jarvis-Judice-Ninke */
        bd->err[err_idx(bd, r0, x + 1)] += (quant_err * 7) / 48;
        bd->err[err_idx(bd, r0, x + 2)] += (quant_err * 5) / 48;
        bd->err[err_idx(bd, r1, x - 2)] += (quant_err * 3) / 48;
        bd->err[err_idx(bd, r1, x - 1)] += (quant_err * 5) / 48;
        bd->err[err_idx(bd, r1, x)]     += (quant_err * 7) / 48;
        bd->err[err_idx(bd, r1, x + 1)] += (quant_err * 5) / 48;
        bd->err[err_idx(bd, r1, x + 2)] += (quant_err * 3) / 48;
        bd->err[err_idx(bd, r2, x - 2)] += (quant_err * 1) / 48;
        bd->err[err_idx(bd, r2, x - 1)] += (quant_err * 3) / 48;
        bd->err[err_idx(bd, r2, x)]     += (quant_err * 5) / 48;
        bd->err[err_idx(bd, r2, x + 1)] += (quant_err * 3) / 48;
        bd->err[err_idx(bd, r2, x + 2)] += (quant_err * 1) / 48;
    }

    return output;
}

/* Internal: advance to next row */
static inline void blue_dither_next_row(BlueDither *bd) {
    /* Clear the row we're leaving (it will be reused as row+3) */
    int row_start = (bd->cur_row & 3) << bd->row_shift;
    memset(bd->err + row_start, 0, bd->width * sizeof(int32_t));

    bd->cur_row = (bd->cur_row + 1) & 3;
    bd->y++;
    bd->x = 0;
}

void blue_dither_row(BlueDither *bd, const uint8_t *input, uint8_t *output) {
    int width = bd->width;
    int y = bd->y;

    for (int x = 0; x < width; x++) {
        output[x] = blue_dither_pixel(bd, x, y, input[x]);
    }

    blue_dither_next_row(bd);
}

int blue_dither_next(BlueDither *bd, uint8_t brightness) {
    int result = blue_dither_pixel(bd, bd->x, bd->y, brightness);

    bd->x++;
    if (bd->x >= bd->width) {
        blue_dither_next_row(bd);
    }

    return result;
}

#endif /* BLUE_DITHER_IMPLEMENTATION */
#endif /* BLUE_DITHER_H */
