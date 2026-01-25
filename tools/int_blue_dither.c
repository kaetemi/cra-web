/*
 * int_blue_dither.c - Integer-only blue noise ditherer
 *
 * Implements Boon dithering using integer math with 48 as the common denominator.
 * FS coefficients (×3): 21, 9, 15, 3
 * JJN coefficients: 7,5 / 3,5,7,5,3 / 1,3,5,3,1
 *
 * Usage: int_blue_dither <width> <height> <gray_level> <output.bin>
 * Convert to PNG: cra -i output.bin --input-metadata '{"format":"L1","width":W,"height":H}' -o output.png
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* lowbias32 hash function - same as CRA uses */
static inline uint32_t lowbias32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x21f0aaad;
    x ^= x >> 15;
    x *= 0x735a2d97;
    x ^= x >> 15;
    return x;
}

typedef struct {
    int width;              /* Image width (stride) */
    int32_t *err[3];        /* Three circular error buffers, each width+4 for padding */
    int cur_row;            /* Current row index in circular buffer (0, 1, or 2) */
    int y;                  /* Current y coordinate */
    uint32_t seed;          /* Random seed for hash */
} BlueDither;

/* Initialize ditherer with given width */
BlueDither *blue_dither_init(int width, uint32_t seed) {
    BlueDither *bd = (BlueDither *)malloc(sizeof(BlueDither));
    if (!bd) return NULL;

    bd->width = width;
    bd->seed = seed;
    bd->cur_row = 0;
    bd->y = 0;

    /* Allocate error buffers with padding (2 pixels each side for JJN) */
    int buf_width = width + 4;
    for (int i = 0; i < 3; i++) {
        bd->err[i] = (int32_t *)calloc(buf_width, sizeof(int32_t));
        if (!bd->err[i]) {
            for (int j = 0; j < i; j++) free(bd->err[j]);
            free(bd);
            return NULL;
        }
    }

    return bd;
}

void blue_dither_free(BlueDither *bd) {
    if (!bd) return;
    for (int i = 0; i < 3; i++) {
        free(bd->err[i]);
    }
    free(bd);
}

/* Clear all error buffers */
void blue_dither_reset(BlueDither *bd) {
    int buf_width = bd->width + 4;
    for (int i = 0; i < 3; i++) {
        memset(bd->err[i], 0, buf_width * sizeof(int32_t));
    }
    bd->cur_row = 0;
    bd->y = 0;
}

/*
 * Process one row of pixels
 * Input: gray values 0-255
 * Output: 1-bit values (0 or 1)
 * Uses serpentine scanning (alternates direction each row)
 */
void blue_dither_row(BlueDither *bd, const uint8_t *input, uint8_t *output, int row_y) {
    int width = bd->width;
    int r0 = bd->cur_row;
    int r1 = (bd->cur_row + 1) % 3;
    int r2 = (bd->cur_row + 2) % 3;

    /* Offset by 2 for padding */
    int32_t *e0 = bd->err[r0] + 2;
    int32_t *e1 = bd->err[r1] + 2;
    int32_t *e2 = bd->err[r2] + 2;

    /* Serpentine: even rows L->R, odd rows R->L */
    int ltr = (row_y & 1) == 0;

    /* Threshold at 127.5 * 48 = 6120 (balanced) */
    const int32_t threshold = 6120;
    const int32_t white_val = 255 * 48;  /* 12240 */

    if (ltr) {
        /* Left to right */
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

            /* Select kernel based on hash */
            uint32_t hash = lowbias32((uint32_t)x ^ ((uint32_t)row_y << 16) ^ bd->seed);

            if (hash & 1) {
                /* Floyd-Steinberg (scaled to /48): 21, 9, 15, 3 */
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
        /* Right to left */
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

            /* Select kernel based on hash */
            uint32_t hash = lowbias32((uint32_t)x ^ ((uint32_t)row_y << 16) ^ bd->seed);

            if (hash & 1) {
                /* Floyd-Steinberg RTL: 21, 3, 15, 9 */
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

    /* Rotate circular buffer: current becomes available for row+2 */
    memset(bd->err[r0] + 2, 0, width * sizeof(int32_t));
    bd->cur_row = r1;
    bd->y++;
}

/*
 * Generate a full image with constant gray level
 * Includes 256-line warmup for clean initialization
 */
void blue_dither_generate(BlueDither *bd, uint8_t gray, int height,
                          uint8_t *output_bits) {
    int width = bd->width;
    int stride_bytes = (width + 7) / 8;

    uint8_t *input_row = (uint8_t *)malloc(width);
    uint8_t *output_row = (uint8_t *)malloc(width);

    memset(input_row, gray, width);

    blue_dither_reset(bd);

    /* Warmup: run 256 rows to initialize error buffers */
    for (int y = 0; y < 256; y++) {
        blue_dither_row(bd, input_row, output_row, y);
    }

    /* Reset row counter but keep error state */
    /* Generate actual output */
    for (int y = 0; y < height; y++) {
        blue_dither_row(bd, input_row, output_row, 256 + y);

        /* Pack into bits (MSB first, matching CRA L1 format) */
        uint8_t *out_byte = output_bits + y * stride_bytes;
        memset(out_byte, 0, stride_bytes);

        for (int x = 0; x < width; x++) {
            if (output_row[x]) {
                out_byte[x / 8] |= (0x80 >> (x % 8));
            }
        }
    }

    free(input_row);
    free(output_row);
}

/* Command line tool */
int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <width> <height> <gray_level> <output.bin>\n", argv[0]);
        fprintf(stderr, "\nGenerates 1-bit blue noise dithered image using integer math.\n");
        fprintf(stderr, "Convert to PNG with:\n");
        fprintf(stderr, "  cra -i output.bin --input-metadata '{\"format\":\"L1\",\"width\":W,\"height\":H}' -o output.png\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int gray = atoi(argv[3]);
    const char *output_path = argv[4];

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: width and height must be positive\n");
        return 1;
    }
    if (gray < 0 || gray > 255) {
        fprintf(stderr, "Error: gray level must be 0-255\n");
        return 1;
    }

    BlueDither *bd = blue_dither_init(width, 12345);
    if (!bd) {
        fprintf(stderr, "Error: failed to initialize ditherer\n");
        return 1;
    }

    int stride_bytes = (width + 7) / 8;
    uint8_t *output = (uint8_t *)malloc(stride_bytes * height);
    if (!output) {
        fprintf(stderr, "Error: failed to allocate output buffer\n");
        blue_dither_free(bd);
        return 1;
    }

    blue_dither_generate(bd, (uint8_t)gray, height, output);

    FILE *fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: failed to open output file\n");
        free(output);
        blue_dither_free(bd);
        return 1;
    }

    fwrite(output, 1, stride_bytes * height, fp);
    fclose(fp);

    printf("Generated %dx%d image at gray=%d -> %s\n", width, height, gray, output_path);
    printf("Convert with: cra -i %s --input-metadata '{\"format\":\"L1\",\"width\":%d,\"height\":%d}' -o output.png\n",
           output_path, width, height);

    free(output);
    blue_dither_free(bd);
    return 0;
}
