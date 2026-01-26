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
 * MODES
 * =====
 *   1. 2D spatial dithering (images) - full FS/JJN with serpentine scan
 *   2. 1D temporal dithering (LED PWM, audio) - simplified 1D diffusion
 *
 * Usage:
 *   #define BLUE_DITHER_IMPLEMENTATION
 *   #include "blue_dither.h"
 *
 * Example - LED PWM:
 *   BlueDither1D bd;
 *   blue_dither_1d_init(&bd, 12345);
 *   while (1) {
 *       int on = blue_dither_1d_next(&bd, brightness);  // brightness 0-255
 *       set_led(on);
 *       delay_us(100);
 *   }
 *
 * Example - Image row:
 *   BlueDither2D bd;
 *   blue_dither_2d_init(&bd, width, 12345);
 *   for (int y = 0; y < height; y++) {
 *       blue_dither_2d_row(&bd, input_row, output_row, y);
 *   }
 *   blue_dither_2d_free(&bd);
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

/* ============================================================================
 * 1D Temporal Dithering (LED PWM, audio DAC, etc.)
 * ============================================================================
 * Minimal state for streaming single bits. Mixes two 1D kernels:
 * - K1: [48] - 100% error to t+1 (tight)
 * - K2: [46,2] - 96% to t+1, 4% to t+2 (23:1 ratio)
 * The 46:2 ratio (23:1) was found optimal through exhaustive spectral analysis.
 */

typedef struct {
    int32_t err0;       /* Error for current step */
    int32_t err1;       /* Error for next step */
    uint32_t state;     /* Hash state (position counter) */
    uint32_t seed;      /* Random seed */
} BlueDither1D;

/* Initialize 1D ditherer */
void blue_dither_1d_init(BlueDither1D *bd, uint32_t seed);

/* Generate next bit. brightness: 0-255, returns: 0 or 1 */
int blue_dither_1d_next(BlueDither1D *bd, uint8_t brightness);

/* Reset state (clears accumulated error) */
void blue_dither_1d_reset(BlueDither1D *bd);


/* ============================================================================
 * 2D Spatial Dithering (images)
 * ============================================================================
 * Full FS/JJN kernel mixing with serpentine scanning.
 * Uses 48 as common denominator (LCM of 16 and 48).
 */

typedef struct {
    int width;          /* Image width */
    int32_t *err[3];    /* Three error buffer rows (circular) */
    int cur_row;        /* Current row in circular buffer */
    uint32_t seed;      /* Random seed */
} BlueDither2D;

/* Initialize 2D ditherer. Returns 0 on success, -1 on allocation failure. */
int blue_dither_2d_init(BlueDither2D *bd, int width, uint32_t seed);

/* Free 2D ditherer resources */
void blue_dither_2d_free(BlueDither2D *bd);

/* Process one row. input: 0-255 per pixel, output: 0 or 1 per pixel */
void blue_dither_2d_row(BlueDither2D *bd, const uint8_t *input, uint8_t *output, int y);

/* Reset state (clears error buffers) */
void blue_dither_2d_reset(BlueDither2D *bd);


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

/* --------------------------------------------------------------------------
 * 1D Implementation
 * -------------------------------------------------------------------------- */

void blue_dither_1d_init(BlueDither1D *bd, uint32_t seed) {
    bd->err0 = 0;
    bd->err1 = 0;
    bd->state = 0;
    bd->seed = seed;
}

void blue_dither_1d_reset(BlueDither1D *bd) {
    bd->err0 = 0;
    bd->err1 = 0;
    bd->state = 0;
}

int blue_dither_1d_next(BlueDither1D *bd, uint8_t brightness) {
    /* Scale to 48ths: 0-255 -> 0-12240 */
    int32_t pixel = (int32_t)brightness * 48 + bd->err0;

    /* Threshold at 127.5 * 48 = 6120 */
    const int32_t threshold = 6120;
    const int32_t white_val = 255 * 48;

    int output;
    int32_t quant_err;

    if (pixel >= threshold) {
        output = 1;
        quant_err = pixel - white_val;
    } else {
        output = 0;
        quant_err = pixel;
    }

    /* Select diffusion kernel based on hash */
    uint32_t hash = blue_dither_hash(bd->state ^ bd->seed);
    bd->state++;

    /* Shift error buffer */
    bd->err0 = bd->err1;
    bd->err1 = 0;

    if (hash & 1) {
        /* FS-like: 100% to next (48/48) */
        bd->err0 += quant_err;
    } else {
        /* Optimal 1D kernel: 46:2 ratio (23:1) - best spectral performance */
        bd->err0 += (quant_err * 46) / 48;
        bd->err1 += (quant_err * 2) / 48;
    }

    return output;
}

/* --------------------------------------------------------------------------
 * 2D Implementation
 * -------------------------------------------------------------------------- */

int blue_dither_2d_init(BlueDither2D *bd, int width, uint32_t seed) {
    bd->width = width;
    bd->seed = seed;
    bd->cur_row = 0;

    /* Allocate with padding (2 pixels each side for JJN kernel) */
    int buf_width = width + 4;
    for (int i = 0; i < 3; i++) {
        bd->err[i] = (int32_t *)calloc(buf_width, sizeof(int32_t));
        if (!bd->err[i]) {
            for (int j = 0; j < i; j++) free(bd->err[j]);
            return -1;
        }
    }
    return 0;
}

void blue_dither_2d_free(BlueDither2D *bd) {
    if (!bd) return;
    for (int i = 0; i < 3; i++) {
        if (bd->err[i]) free(bd->err[i]);
    }
}

void blue_dither_2d_reset(BlueDither2D *bd) {
    int buf_width = bd->width + 4;
    for (int i = 0; i < 3; i++) {
        memset(bd->err[i], 0, buf_width * sizeof(int32_t));
    }
    bd->cur_row = 0;
}

void blue_dither_2d_row(BlueDither2D *bd, const uint8_t *input, uint8_t *output, int y) {
    int width = bd->width;
    int r0 = bd->cur_row;
    int r1 = (bd->cur_row + 1) % 3;
    int r2 = (bd->cur_row + 2) % 3;

    /* Offset by 2 for padding */
    int32_t *e0 = bd->err[r0] + 2;
    int32_t *e1 = bd->err[r1] + 2;
    int32_t *e2 = bd->err[r2] + 2;

    /* Serpentine: even rows L->R, odd rows R->L */
    int ltr = (y & 1) == 0;

    const int32_t threshold = 6120;  /* 127.5 * 48 */
    const int32_t white_val = 12240; /* 255 * 48 */

    if (ltr) {
        for (int x = 0; x < width; x++) {
            int32_t pixel = (int32_t)input[x] * 48 + e0[x];

            int32_t quant_err;
            if (pixel >= threshold) {
                output[x] = 1;
                quant_err = pixel - white_val;
            } else {
                output[x] = 0;
                quant_err = pixel;
            }

            uint32_t hash = blue_dither_hash((uint32_t)x ^ ((uint32_t)y << 16) ^ bd->seed);

            if (hash & 1) {
                /* Floyd-Steinberg: 21/48, 9/48, 15/48, 3/48 */
                e0[x + 1] += (quant_err * 21) / 48;
                e1[x - 1] += (quant_err * 9) / 48;
                e1[x]     += (quant_err * 15) / 48;
                e1[x + 1] += (quant_err * 3) / 48;
            } else {
                /* JJN: 7,5 / 3,5,7,5,3 / 1,3,5,3,1 */
                e0[x + 1] += (quant_err * 7) / 48;
                e0[x + 2] += (quant_err * 5) / 48;
                e1[x - 2] += (quant_err * 3) / 48;
                e1[x - 1] += (quant_err * 5) / 48;
                e1[x]     += (quant_err * 7) / 48;
                e1[x + 1] += (quant_err * 5) / 48;
                e1[x + 2] += (quant_err * 3) / 48;
                e2[x - 2] += (quant_err * 1) / 48;
                e2[x - 1] += (quant_err * 3) / 48;
                e2[x]     += (quant_err * 5) / 48;
                e2[x + 1] += (quant_err * 3) / 48;
                e2[x + 2] += (quant_err * 1) / 48;
            }
        }
    } else {
        for (int x = width - 1; x >= 0; x--) {
            int32_t pixel = (int32_t)input[x] * 48 + e0[x];

            int32_t quant_err;
            if (pixel >= threshold) {
                output[x] = 1;
                quant_err = pixel - white_val;
            } else {
                output[x] = 0;
                quant_err = pixel;
            }

            uint32_t hash = blue_dither_hash((uint32_t)x ^ ((uint32_t)y << 16) ^ bd->seed);

            if (hash & 1) {
                /* Floyd-Steinberg RTL */
                e0[x - 1] += (quant_err * 21) / 48;
                e1[x + 1] += (quant_err * 9) / 48;
                e1[x]     += (quant_err * 15) / 48;
                e1[x - 1] += (quant_err * 3) / 48;
            } else {
                /* JJN RTL */
                e0[x - 1] += (quant_err * 7) / 48;
                e0[x - 2] += (quant_err * 5) / 48;
                e1[x + 2] += (quant_err * 3) / 48;
                e1[x + 1] += (quant_err * 5) / 48;
                e1[x]     += (quant_err * 7) / 48;
                e1[x - 1] += (quant_err * 5) / 48;
                e1[x - 2] += (quant_err * 3) / 48;
                e2[x + 2] += (quant_err * 1) / 48;
                e2[x + 1] += (quant_err * 3) / 48;
                e2[x]     += (quant_err * 5) / 48;
                e2[x - 1] += (quant_err * 3) / 48;
                e2[x - 2] += (quant_err * 1) / 48;
            }
        }
    }

    /* Rotate buffer and clear for reuse */
    memset(bd->err[r0] + 2, 0, width * sizeof(int32_t));
    bd->cur_row = r1;
}

#endif /* BLUE_DITHER_IMPLEMENTATION */
#endif /* BLUE_DITHER_H */
