/*
 * int_blue_dither.c - Integer-only blue noise dithering tool
 *
 * Demonstrates both streaming (1D) and row (2D) modes.
 *
 * Usage:
 *   2D mode: int_blue_dither <width> <height> <gray_level> <output.bin>
 *   1D mode: int_blue_dither --1d <count> <gray_level>
 *
 * Convert 2D output to PNG:
 *   cra -i output.bin --input-metadata '{"format":"L1","width":W,"height":H}' -o output.png
 */

#define BLUE_DITHER_IMPLEMENTATION
#include "blue_dither.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Generate 2D image with warmup */
static void generate_2d_image(int width, int height, uint8_t gray, uint8_t *output_bits) {
    BlueDither bd;
    if (blue_dither_init(&bd, width, 12345) != 0) {
        fprintf(stderr, "Failed to initialize ditherer\n");
        return;
    }

    int stride_bytes = (width + 7) / 8;
    uint8_t *input_row = (uint8_t *)malloc(width);
    uint8_t *output_row = (uint8_t *)malloc(width);

    memset(input_row, gray, width);

    /* Warmup: 256 rows to initialize error buffers */
    for (int y = 0; y < 256; y++) {
        blue_dither_row(&bd, input_row, output_row);
    }

    /* Generate actual output */
    for (int y = 0; y < height; y++) {
        blue_dither_row(&bd, input_row, output_row);

        /* Pack into bits (MSB first) */
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
    blue_dither_free(&bd);
}

/* Demo 1D mode - prints pattern and statistics */
static void demo_1d(int count, uint8_t gray) {
    BlueDither bd;
    int row_width = 256;

    if (blue_dither_init(&bd, row_width, 12345) != 0) {
        fprintf(stderr, "Failed to initialize ditherer\n");
        return;
    }

    int ones = 0;
    int runs = 0;
    int last = -1;
    int max_run = 0;
    int cur_run = 0;

    printf("1D Blue Noise Dithering Demo (width=%d)\n", row_width);
    printf("Gray level: %d (%.1f%%)\n", gray, gray * 100.0 / 255.0);
    printf("Sample count: %d\n\n", count);

    /* Warmup: run through one full row */
    for (int i = 0; i < row_width; i++) {
        blue_dither_next(&bd, gray);
    }

    /* Generate and analyze */
    printf("Pattern (first 80): ");
    for (int i = 0; i < count; i++) {
        int bit = blue_dither_next(&bd, gray);
        ones += bit;

        /* Track runs */
        if (bit == last) {
            cur_run++;
        } else {
            if (cur_run > max_run) max_run = cur_run;
            if (last >= 0) runs++;
            cur_run = 1;
            last = bit;
        }

        if (i < 80) putchar(bit ? '#' : '.');
    }
    if (cur_run > max_run) max_run = cur_run;

    printf("\n\n");
    printf("Statistics:\n");
    printf("  Target duty: %.2f%%\n", gray * 100.0 / 255.0);
    printf("  Actual duty: %.2f%%\n", ones * 100.0 / count);
    printf("  Run changes: %d\n", runs);
    printf("  Max run len: %d\n", max_run);
    printf("  Avg run len: %.2f\n", (float)count / runs);

    /* Show LED PWM example */
    printf("\nLED PWM Example Code:\n");
    printf("  BlueDither bd;\n");
    printf("  blue_dither_init(&bd, 256, seed);\n");
    printf("  while (1) {\n");
    printf("      int on = blue_dither_next(&bd, %d);\n", gray);
    printf("      GPIO_SET(LED_PIN, on);\n");
    printf("      delay_us(100);  // 10kHz update rate\n");
    printf("  }\n");

    blue_dither_free(&bd);
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  2D mode: %s <width> <height> <gray_level> <output.bin>\n", prog);
    fprintf(stderr, "  1D mode: %s --1d <count> <gray_level>\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "2D mode generates L1 (1-bit) raw binary for image dithering.\n");
    fprintf(stderr, "1D mode demonstrates streaming for LED PWM.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Convert 2D output to PNG:\n");
    fprintf(stderr, "  cra -i output.bin --input-metadata '{\"format\":\"L1\",\"width\":W,\"height\":H}' -o output.png\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* 1D mode */
    if (strcmp(argv[1], "--1d") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: --1d requires <count> <gray_level>\n");
            return 1;
        }
        int count = atoi(argv[2]);
        int gray = atoi(argv[3]);
        if (count <= 0 || gray < 0 || gray > 255) {
            fprintf(stderr, "Error: invalid parameters\n");
            return 1;
        }
        demo_1d(count, (uint8_t)gray);
        return 0;
    }

    /* 2D mode */
    if (argc < 5) {
        print_usage(argv[0]);
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

    int stride_bytes = (width + 7) / 8;
    uint8_t *output = (uint8_t *)malloc(stride_bytes * height);
    if (!output) {
        fprintf(stderr, "Error: failed to allocate output buffer\n");
        return 1;
    }

    generate_2d_image(width, height, (uint8_t)gray, output);

    FILE *fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: failed to open output file\n");
        free(output);
        return 1;
    }

    fwrite(output, 1, stride_bytes * height, fp);
    fclose(fp);

    printf("Generated %dx%d image at gray=%d -> %s\n", width, height, gray, output_path);
    printf("Convert with: cra -i %s --input-metadata '{\"format\":\"L1\",\"width\":%d,\"height\":%d}' -o output.png\n",
           output_path, width, height);

    free(output);
    return 0;
}
