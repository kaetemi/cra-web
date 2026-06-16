# Image and Asset Format Reference

Reference for image converter tools targeting BridgeTek EVE (BT815–BT820) with LVGL integration. Covers the EVE relocatable asset container (`.reloc`), ESD Core metadata sidecar format (`.esdm`), EVE bitmap formats and stride requirements, and the LVGL binary image format (`.bin`).

## Table of Contents

1. [EVE Relocatable Asset Format (.reloc)](#eve-relocatable-asset-format-reloc)
2. [ESD Core Metadata Format (.esdm)](#esd-core-metadata-format-esdm)
3. [EVE Bitmap Formats](#eve-bitmap-formats)
4. [LVGL Binary Image Format (.bin)](#lvgl-binary-image-format-bin)
5. [LVGL Color Formats](#lvgl-color-formats)
6. [Format Mapping: LVGL to EVE](#format-mapping-lvgl-to-eve)

---

## EVE Relocatable Asset Format (.reloc)

The `.reloc` format is a container for EVE assets (fonts, images, animations) that includes compressed data and a relocation table. Loaded via `CMD_LOADASSET`, which inflates the data to the specified RAM_G address and patches all internal pointers to match the load location.

### File Structure

The asset data is always deflate-compressed. The coprocessor inflates it during `CMD_LOADASSET`.

```
+0                  uint32    signature       (0x0100aa44, little-endian)
+4                  uint32    asset_size      (inflated size in bytes)
+8                  uint32    deflated_size   (compressed data size in bytes)
+12                 byte[]    deflated_data   (deflate-compressed asset, deflated_size bytes)
+12+deflated_size   relocation_table
```

The total file size is `12 + deflated_size + relocation_table_size`.

### Relocation Table

The relocation table follows the compressed data. It consists of one or more method blocks, terminated by a 32-bit zero word:

```
method_block[]
0x00000000          (terminator)
```

Each method block:

```
+0    uint32    method      Relocation method (see below)
+4    uint32    count       Number of offsets
+8    uint32    offset[0]   Byte offset within inflated asset
+12   uint32    offset[1]
...
+8+count*4      (end of block)
```

#### Relocation Methods

| Method | Name | Description |
|--------|------|-------------|
| `P` (0x50) | Pointer | A simple 32-bit pointer at the given offset. The loader adds the load base address to the value. |
| `H` (0x48) | Bitmap Handle | A pair of display list instructions (`BITMAP_SOURCEH` + `BITMAP_SOURCE` or `PALETTE_SOURCEH` + `PALETTE_SOURCE`) at the given offset. The loader patches the embedded address. |

### CMD_LOADASSET Usage

```c
void cmd_loadasset(uint32_t ptr, uint32_t options);
```

**Parameters:**
- `ptr` — Destination RAM_G address. Animation assets must be **64-byte aligned**. Other assets (fonts, images) require **32-byte aligned**.
- `options` — Data source selection:
  - `0` — Data follows in the command buffer (default).
  - `OPT_FLASH` (64) — Flash memory source. Use `CMD_FLASHSOURCE` before this command.
  - `OPT_FS` (0x2000) — Filesystem source. Use `CMD_FSSOURCE` before this command.
  - `OPT_MEDIAFIFO` (16) — Media FIFO source. Use `CMD_MEDIAFIFO` before this command.

**Behavior:**
- Inflates the compressed data to `ptr`.
- Applies all relocations, patching internal pointers to the actual load address.
- Pads written data with 0–31 zero bytes to a **32-byte boundary**.
- Advances the allocation pointer (`CMD_GETPTR`) to the next 32-byte boundary after the asset.

### Post-Load Setup

After `CMD_LOADASSET`, the asset data is in RAM_G but not yet registered with EVE:

- **Fonts:** Call `CMD_SETFONT(handle, ptr, 0)` to register the font with a bitmap handle (0–63). After this, the handle can be used with `CMD_TEXT`, `CMD_BUTTON`, etc.
- **Animations:** Use `CMD_ANIMSTART`, `CMD_ANIMFRAME`, etc. with the load address.
- **Images:** The inflated data is a raw bitmap. Set up `BITMAP_SOURCE`, `BITMAP_LAYOUT`, and `BITMAP_SIZE` manually using metadata from an `.esdm` sidecar file.

### Limitations

The `.reloc` container does not include metadata about the asset contents (dimensions, format, glyph metrics, etc.). The host must know the asset type and properties through external means — typically an `.esdm` metadata sidecar file (see next section) or compile-time constants.

### Example: Loading a Font from Filesystem

```c
cmd_fssource("serif.reloc", 0);
cmd_loadasset(0x1000, OPT_FS);        // Load to 32-byte aligned address
cmd_setfont(8, 0x1000, 0);            // Register as handle 8
cmd_text(100, 100, 8, 0, "Hello");    // Use in widget
```

---

## ESD Core Metadata Format (.esdm)

The `.esdm` file is a binary sidecar placed alongside an asset file to provide metadata that the asset data itself does not contain. For a file named `image.raw`, the metadata file is `image.raw.esdm`. Maximum file size: 64 bytes.

The metadata is loaded by `Esd_LoadResourceEx()` in `esd_core/Esd_ResourceInfo.c` via `EVE_Util_readFile()`.

### Common Header (12 bytes)

All `.esdm` files share this header. Type-specific data follows immediately after.

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | uint32 | `signature` | Type signature, NUL-terminated ASCII: `"RES"` (0x00534552), `"BMP"` (0x00504D42), `"FNT"` (0x00544E46), or `"ANI"` (0x00494E41). Little-endian. |
| 4 | 1 | uint8 | `version` | Metadata format version. Must be `1`. |
| 5 | 1 | uint8 | `size` | Total size of the metadata structure in bytes (header + type-specific data). |
| 6 | 1 | uint8 | `compression` | Compression/load method for the asset data. See [Compression Values](#compression-values). |
| 7 | 1 | uint8 | `extLen` | String length of the complete file extension of the resource file (e.g., 4 for `.raw`). |
| 8 | 4 | uint32 | `rawSize` | Uncompressed size of the asset data in bytes. |

All multi-byte fields are **little-endian**.

#### Compression Values

| Value | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | `ESD_RESOURCE_RAW` | Implemented | Uncompressed data. Load with direct memory copy or `CMD_FLASHREAD`. |
| 1 | `ESD_RESOURCE_DEFLATE` | Implemented | Deflate/zlib compressed. Load with `CMD_INFLATE`. |
| 2 | `ESD_RESOURCE_IMAGE` | Implemented | JPEG or PNG image. Load with `CMD_LOADIMAGE`. |
| 3 | `ESD_RESOURCE_ASSET` | Planned | Relocatable asset (`.reloc`). Load with `CMD_LOADASSET`. Contains deflate-compressed data with pointer relocation table. Replaces the previously reserved `ESD_RESOURCE_VIDEO`. |
| 4 | `ESD_RESOURCE_VIDEOFRAME` | Reserved (TBD) | Video resource. Would load with `CMD_VIDEOFRAME`. Not yet implemented. |

This is a 2-bit enum in the `Esd_ResourceInfo` runtime struct (`Compressed : 2`), so only values 0–3 currently fit. The `.esdm` file stores it as a `uint8` at offset 6 and can already represent value 4. Value 3 was previously reserved for `ESD_RESOURCE_VIDEO` (`CMD_VIDEOFRAME`, never implemented) and is reassigned to `ESD_RESOURCE_ASSET`. Supporting `ESD_RESOURCE_VIDEOFRAME` at value 4 requires widening the bitfield to 3 bits (stealing 1 bit from `StorageSize : 27`, reducing max storage size from ~512 MB to ~256 MB).

### BMP Type (Bitmap Metadata)

Signature: `"BMP"` (0x00504D42). Total size: 56 bytes (12 header + 44 type-specific).

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 12 | 4 | int32 | `width` | Image width in pixels. |
| 16 | 4 | int32 | `height` | Image height in pixels. |
| 20 | 4 | int32 | `stride` | Row stride in bytes. See [EVE Stride Requirements](#stride-requirements). |
| 24 | 4 | uint32 | `format` | EVE bitmap format enum value. See [EVE Bitmap Formats](#eve-bitmap-formats). |
| 28 | 2 | uint16 | `paletteSize` | Uncompressed size of the palette file in bytes. 0 if not paletted. |
| 30 | 10 | char[] | `paletteFileExt` | Filename suffix for the palette sidecar file (e.g., `".pal.raw"`). NUL-padded. Only used for paletted formats. |
| 40 | 12 | char[] | `addtlResExt` | Filename suffix for additional resource (e.g., DXT1 auxiliary data). NUL-padded. |
| 52 | 2 | uint16 | `cells` | Number of animation cells in a cell-based sprite sheet. |
| 54 | 2 | uint16 | `swizzle` | Bitmap swizzle configuration (BT815+). Bit 12: enable flag. Bits 11-9: B channel source. Bits 8-6: G channel source. Bits 5-3: R channel source. Bits 2-0: A channel source. Channel source values: 0=ZERO, 1=ONE, 2=RED, 3=GREEN, 4=BLUE, 5=ALPHA. |

### ANI Type (Animation Metadata)

Signature: `"ANI"` (0x00494E41). Total size: 36 bytes (12 header + 24 type-specific).

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 12 | 4 | uint32 | `framesPtr` | The RAM_G address of the frames file that the object file was built with (used for relocation). |
| 16 | 4 | uint32 | `framesSize` | Size of the display list frames file. 0 = both frames and atlas files are ignored. |
| 20 | 4 | uint32 | `atlasBitmapSource` | The address in `BITMAP_SOURCE` display list format of the atlas file the frames were built with. |
| 24 | 4 | uint32 | `atlasSize` | Size of the ASTC bitmap atlas file. 0 = atlas file is ignored. |
| 28 | 2 | int16 | `rectX` | Display list bounding box X origin. |
| 30 | 2 | int16 | `rectY` | Display list bounding box Y origin. |
| 32 | 2 | int16 | `rectWidth` | Display list bounding box width. |
| 34 | 2 | int16 | `rectHeight` | Display list bounding box height. |

### FNT Type (Font Metadata)

Signature: `"FNT"` (0x00544E46). The font metadata uses the same common header. Font-specific extended fields are not defined in the current ESD Core source — font assets loaded via `CMD_LOADASSET` contain their own internal metric tables. The `.esdm` common header provides `compression` and `rawSize` for the font `.reloc` file.

### RES Type (Generic Resource)

Signature: `"RES"` (0x00534552). Uses only the 12-byte common header with no type-specific extension. Provides `compression` and `rawSize` for arbitrary data blobs.

---

## EVE Bitmap Formats

### Legacy Formats (EVE1–EVE5, format value 0–31)

These fit in the 5-bit layout field of `BITMAP_LAYOUT`. Used directly without `GLFORMAT`.

| Value | Name | BPP | Bit Layout (MSB → LSB) | Alpha | Description |
|-------|------|-----|----------------------|-------|-------------|
| 0 | `ARGB1555` | 16 | `[15] A [14:10] R [9:5] G [4:0] B` | 1-bit | |
| 1 | `L1` | 1 | `[7] px0 [6] px1 ... [0] px7` (MSB-first) | No | 1-bit luminance. BT820 decodes as (R=255, G=255, B=255, A=L). |
| 2 | `L4` | 4 | `[7:4] px0 [3:0] px1` (MSB-first) | No | 4-bit luminance. BT820 decodes as (R=255, G=255, B=255, A=L). |
| 3 | `L8` | 8 | `[7:0] L` | No | 8-bit luminance. BT820 decodes as (R=255, G=255, B=255, A=L). |
| 4 | `RGB332` | 8 | `[7:5] R [4:2] G [1:0] B` | No | |
| 5 | `ARGB2` | 8 | `[7:6] A [5:4] R [3:2] G [1:0] B` | 2-bit | Also known as ARGB2222. |
| 6 | `ARGB4` | 16 | `[15:12] A [11:8] R [7:4] G [3:0] B` | 4-bit | Also known as ARGB4444. |
| 7 | `RGB565` | 16 | `[15:11] R [10:5] G [4:0] B` | No | |
| 8 | `PALETTED` | 8 | `[7:0] index` | Via palette | EVE1 only. 256-entry palette. |
| 9 | `TEXT8X8` | 8 | Special | No | 8x8 text grid. |
| 10 | `TEXTVGA` | 8 | Special | No | VGA text mode. |
| 11 | `BARGRAPH` | 8 | Special | No | Renders data as bar graph: pixel is opaque if `byte[x] < y`, transparent otherwise. |
| 14 | `PALETTED565` | 8 | `[7:0] index` | No | EVE2+. 256-entry RGB565 palette. |
| 15 | `PALETTED4444` | 8 | `[7:0] index` | 4-bit | EVE2+. 256-entry ARGB4444 palette. |
| 16 | `PALETTED8` | 8 | `[7:0] index` | 8-bit | EVE2+. 256-entry ARGB8888 palette. |
| 17 | `L2` | 2 | `[7:6] px0 [5:4] px1 [3:2] px2 [1:0] px3` (MSB-first) | No | 2-bit luminance. BT815+. |
| 19 | `RGB8` | 24 | `[23:16] R [15:8] G [7:0] B` | No | BT820 only. |
| 20 | `ARGB8` | 32 | `[31:24] A [23:16] R [15:8] G [7:0] B` | 8-bit | BT820 only. Full 32-bit ARGB. |
| 21 | `PALETTEDARGB8` | 8 | `[7:0] index` → palette `[31:24] A [23:16] R [15:8] G [7:0] B` | 8-bit | BT820 only. 256-entry ARGB8888 palette at full speed. |
| 22 | `RGB6` | 18 | `[17:12] R [11:6] G [5:0] B` | No | BT820 only. 18 bits packed, 8-pixel aligned. |
| 23 | `ARGB6` | 24 | `[23:18] A [17:12] R [11:6] G [5:0] B` | 6-bit | BT820 only. |
| 24 | `LA1` | 2 | `[7] A0 [6] L0 [5] A1 [4] L1 [3] A2 [2] L2 [1] A3 [0] L3` | 1-bit | BT820 only. Luminance + alpha, 4 pixels/byte. |
| 25 | `LA2` | 4 | `[7:6] A0 [5:4] L0 [3:2] A1 [1:0] L1` | 2-bit | BT820 only. 2 pixels/byte. |
| 26 | `LA4` | 8 | `[7:4] A [3:0] L` | 4-bit | BT820 only. |
| 27 | `LA8` | 16 | `[15:8] A [7:0] L` | 8-bit | BT820 only. |
| 28 | `YCBCR` | 8 | 2x2 block, 32 bits/quad | No | BT820 only. Width and height must be 2-pixel aligned. |
| 31 | `GLFORMAT` | — | — | — | Sentinel value. Used in `BITMAP_LAYOUT` to indicate that the actual format is specified via `BITMAP_EXT_FORMAT`. Required for extended formats and for `BITMAP_SWIZZLE` to take effect on BT820. |

### Extended Formats (EVE2+, format value > 31)

These require `GLFORMAT` mode: set `BITMAP_LAYOUT(GLFORMAT, stride, height)` then `BITMAP_EXT_FORMAT(format_value)`. On BT820, `BITMAP_SWIZZLE(RED, GREEN, BLUE, ALPHA)` should be set to identity when using extended formats.

| Value | Name | Block Size | Bits/Pixel | Description |
|-------|------|-----------|------------|-------------|
| 37808 | `COMPRESSED_RGBA_ASTC_4x4_KHR` | 4x4 | 8.00 | ASTC compressed. |
| 37809 | `COMPRESSED_RGBA_ASTC_5x4_KHR` | 5x4 | 6.40 | |
| 37810 | `COMPRESSED_RGBA_ASTC_5x5_KHR` | 5x5 | 5.12 | |
| 37811 | `COMPRESSED_RGBA_ASTC_6x5_KHR` | 6x5 | 4.27 | |
| 37812 | `COMPRESSED_RGBA_ASTC_6x6_KHR` | 6x6 | 3.56 | |
| 37813 | `COMPRESSED_RGBA_ASTC_8x5_KHR` | 8x5 | 3.20 | |
| 37814 | `COMPRESSED_RGBA_ASTC_8x6_KHR` | 8x6 | 2.67 | |
| 37815 | `COMPRESSED_RGBA_ASTC_8x8_KHR` | 8x8 | 2.00 | |
| 37816 | `COMPRESSED_RGBA_ASTC_10x5_KHR` | 10x5 | 2.56 | |
| 37817 | `COMPRESSED_RGBA_ASTC_10x6_KHR` | 10x6 | 2.13 | |
| 37818 | `COMPRESSED_RGBA_ASTC_10x8_KHR` | 10x8 | 1.60 | |
| 37819 | `COMPRESSED_RGBA_ASTC_10x10_KHR` | 10x10 | 1.28 | |
| 37820 | `COMPRESSED_RGBA_ASTC_12x10_KHR` | 12x10 | 1.07 | |
| 37821 | `COMPRESSED_RGBA_ASTC_12x12_KHR` | 12x12 | 0.89 | |

ASTC format detection: `(format & 0xFFF0) == 0x93B0`.

### Memory Layout

EVE bitmap data is stored as tightly packed rows. Multi-byte pixel values are **little-endian** (least significant byte at lowest address). The bit layout tables above show the logical value; in memory:

| Format | Memory byte order (per pixel) | Notes |
|--------|-------------------------------|-------|
| `ARGB1555` | `byte[0]`: `[GGGBBBBB]`, `byte[1]`: `[ARRRRRGG]` | |
| `ARGB4` | `byte[0]`: `[GGGGBBBB]`, `byte[1]`: `[AAAARRRR]` | |
| `RGB565` | `byte[0]`: `[GGGBBBBB]`, `byte[1]`: `[RRRRRGGG]` | |
| `RGB8` | `byte[0]`: B, `byte[1]`: G, `byte[2]`: R | BGR order |
| `ARGB8` | `byte[0]`: B, `byte[1]`: G, `byte[2]`: R, `byte[3]`: A | BGRA order |
| `RGB6` | 18 bits packed across byte boundaries | 8-pixel aligned |
| `ARGB6` | `byte[0]`: B+G low, `byte[1]`: G high+R low, `byte[2]`: R high+A | |
| `LA8` | `byte[0]`: L, `byte[1]`: A | |
| `PALETTEDARGB8` | `byte[0]`: index (palette entries are BGRA) | |

Sub-byte formats (`L1`, `L2`, `L4`, `LA1`, `LA2`) pack **MSB-first** within each byte (pixel 0 in the highest bits).

### Stride and Pixel Alignment Requirements

The stride (`linestride` in `BITMAP_LAYOUT`) is the byte offset between consecutive rows.

**BITMAP_LAYOUT encoding:**
- Stride field: 10 bits (0–1023). For strides > 1023, use `BITMAP_LAYOUT_H` to provide the upper 2 bits (total 12 bits, max 4095).
- Height field: 9 bits (0–511). For heights > 511, use `BITMAP_LAYOUT_H`.

**Per-format stride and width alignment** (from BT820 programming guide):

Some formats require the image width to be a multiple of a specific pixel count so that rows align on byte boundaries. The "Min stride" column shows the minimum byte granularity.

| Format | Value | BPP | Min Stride (bytes) | Width Alignment (pixels) |
|--------|-------|-----|-------------------|-------------------------|
| `ARGB1555` | 0 | 16 | 2 | — |
| `L1` | 1 | 1 | 1 | 8 |
| `L4` | 2 | 4 | 1 | 2 |
| `L8` | 3 | 8 | 1 | — |
| `RGB332` | 4 | 8 | 1 | — |
| `ARGB2` | 5 | 8 | 1 | — |
| `ARGB4` | 6 | 16 | 2 | — |
| `RGB565` | 7 | 16 | 2 | — |
| `L2` | 17 | 2 | 1 | 4 |
| `RGB8` | 19 | 24 | 3 | 2 (BT820 hardware constraint) |
| `ARGB8` | 20 | 32 | 4 | — |
| `PALETTEDARGB8` | 21 | 8 | 1 | — |
| `RGB6` | 22 | 18 | 3 | 8 (18 bits not byte-aligned) |
| `ARGB6` | 23 | 24 | 3 | 2 (BT820 hardware constraint) |
| `LA1` | 24 | 2 | 1 | 4 |
| `LA2` | 25 | 4 | 1 | 2 |
| `LA4` | 26 | 8 | 1 | — |
| `LA8` | 27 | 16 | 2 | — |
| `YCBCR` | 28 | 8 | — | 2 (2x2 blocks, height also 2-aligned) |
| ASTC | 37808+ | var | 16 | Block width (4–12, each block = 16 bytes) |

**Stride formula:**
```
natural_stride = ceil(width * bpp / 8)
```

For sub-byte formats with width alignment, pad width first:
```
aligned_width = ALIGN_UP(width, pixel_alignment)
natural_stride = aligned_width * bpp / 8
```

For ASTC:
```
stride = ceil(width / block_width) * 16
```

**ESD/LVGL conventions:**
- The EVE5 LVGL driver aligns strides to 4 bytes: `ALIGN_UP(ceil(width * bpp / 8), 4)`.
- Render target allocations align width to 16 pixels: stride = `ALIGN_UP(width, 16) * bytes_per_pixel`.
- The `.esdm` metadata file stores the actual stride used, which may differ from the natural stride.

**CMD_LOADIMAGE output:**
- `CMD_LOADIMAGE` produces packed output with no stride padding: stride = `width * bytes_per_pixel`.
- The actual format, source address, dimensions, and palette address can be queried via `CMD_GETIMAGE` after decode.
- With `OPT_TRUECOLOR`: JPEG decodes to `RGB8` (3 bpp), PNG decodes to `RGB8` or `ARGB8` (depending on alpha channel), indexed PNG may produce `PALETTEDARGB8`.
- Without `OPT_TRUECOLOR`: JPEG decodes to `RGB565`, PNG decodes to `RGB565` or `ARGB4`.

---

## LVGL Binary Image Format (.bin)

LVGL's native binary image format. The built-in bin decoder (`lv_bin_decoder.c`) requires the `.bin` file extension.

### File Structure

```
+0    lv_image_header_t    (12 bytes)
+12   [palette data]       (only for indexed formats I1/I2/I4/I8)
+N    [pixel data]         (stride * height bytes)
+M    [alpha plane]        (only for RGB565A8: stride/2 * height bytes)
```

### Image Header (12 bytes)

Little-endian bitfields packed into three 32-bit words:

```
Word 0 (offset 0):
  bits  7:0   magic       (uint8)   Must be 0x19 (LV_IMAGE_HEADER_MAGIC)
  bits 15:8   cf          (uint8)   Color format enum (lv_color_format_t)
  bits 31:16  flags       (uint16)  Image flags (lv_image_flags_t)

Word 1 (offset 4):
  bits 15:0   w           (uint16)  Width in pixels
  bits 31:16  h           (uint16)  Height in pixels

Word 2 (offset 8):
  bits 15:0   stride      (uint16)  Bytes per row (0 = calculate from width and cf)
  bits 31:16  reserved    (uint16)  Reserved, must be 0
```

On big-endian systems, the bitfield order within each word is reversed (see `lv_image_dsc.h`).

### Image Flags

| Value | Name | Description |
|-------|------|-------------|
| 0x0001 | `LV_IMAGE_FLAGS_PREMULTIPLIED` | RGB is pre-multiplied by alpha. |
| 0x0002 | `LV_IMAGE_FLAGS_MODIFIABLE` | Buffer can be modified (not read-only). Set automatically for file-loaded images. |
| 0x0004 | `LV_IMAGE_FLAGS_ALLOCATED` | Buffer was allocated dynamically (freed on destroy). |

Flags 0x0008–0x0040 are internal (CLEARZERO, DISCARDABLE, etc.). Flags 0x0100–0x8000 are user-defined.

### Image Compression

LVGL supports optional compression indicated by `lv_image_compress_t` stored in the image descriptor (not in the file header directly — the bin decoder detects compression during open):

| Value | Name | Description |
|-------|------|-------------|
| 0 | `LV_IMAGE_COMPRESS_NONE` | Raw pixel data. |
| 1 | `LV_IMAGE_COMPRESS_RLE` | LVGL custom RLE compression. |
| 2 | `LV_IMAGE_COMPRESS_LZ4` | LZ4 compression. |

### Stride Calculation

When `stride` in the header is 0, it is calculated as:

```
stride = ROUND_UP(width * bpp / 8, LV_DRAW_BUF_STRIDE_ALIGN)
```

Where `LV_DRAW_BUF_STRIDE_ALIGN` is configured in `lv_conf.h` (typically 1, meaning no alignment padding). `bpp` is determined by the color format — see [LVGL Color Formats](#lvgl-color-formats).

For sub-byte formats (A1, A2, A4, I1, I2, I4): `stride = ROUND_UP((width * bpp + 7) / 8, LV_DRAW_BUF_STRIDE_ALIGN)`.

### Pixel Data Layout

**Standard formats:** Rows of packed pixel data, `stride` bytes per row, for `height` rows.

**Indexed formats (I1, I2, I4, I8):** Palette data precedes pixel data:

```
+12   palette[]     N * 4 bytes (N entries of lv_color32_t in BGRA order)
+12+N*4  pixels[]   stride * height bytes
```

Palette sizes:
| Format | Entries (N) | Palette bytes |
|--------|-------------|---------------|
| I1 | 2 | 8 |
| I2 | 4 | 16 |
| I4 | 16 | 64 |
| I8 | 256 | 1024 |

Each palette entry is `lv_color32_t`: `[B] [G] [R] [A]` (4 bytes, little-endian BGRA).

Sub-byte index formats (I1, I2, I4) pack indices MSB-first within each byte.

**RGB565A8 format:** Pixel data is followed by a separate alpha plane:

```
+12             RGB565 rows    stride * height bytes
+12+stride*h    A8 rows        (stride/2) * height bytes
```

### Total File Size

```
For standard formats:
  12 + stride * height

For indexed formats:
  12 + palette_entries * 4 + stride * height

For RGB565A8:
  12 + stride * height + (stride / 2) * height
```

---

## LVGL Color Formats

All values for the `cf` field in the LVGL image header (`lv_color_format_t` enum):

| Value | Name | BPP | Has Alpha | Description |
|-------|------|-----|-----------|-------------|
| 0x00 | `UNKNOWN` | — | — | Invalid/uninitialized. |
| 0x01 | `RAW` | — | No | Raw data, not rendered directly. |
| 0x02 | `RAW_ALPHA` | — | Yes | Raw data with alpha, not rendered directly. |
| 0x06 | `L8` | 8 | No | 8-bit luminance (grayscale). |
| 0x07 | `I1` | 1 | Via palette | 1-bit indexed. 2-entry BGRA palette. |
| 0x08 | `I2` | 2 | Via palette | 2-bit indexed. 4-entry BGRA palette. |
| 0x09 | `I4` | 4 | Via palette | 4-bit indexed. 16-entry BGRA palette. |
| 0x0A | `I8` | 8 | Via palette | 8-bit indexed. 256-entry BGRA palette. |
| 0x0B | `A1` | 1 | Yes | 1-bit alpha only. |
| 0x0C | `A2` | 2 | Yes | 2-bit alpha only. |
| 0x0D | `A4` | 4 | Yes | 4-bit alpha only. |
| 0x0E | `A8` | 8 | Yes | 8-bit alpha only. |
| 0x0F | `RGB888` | 24 | No | 24-bit RGB. Memory: `[B] [G] [R]`. |
| 0x10 | `ARGB8888` | 32 | Yes | 32-bit ARGB. Memory: `[B] [G] [R] [A]`. |
| 0x11 | `XRGB8888` | 32 | No | 32-bit RGB with unused alpha byte (always 0xFF). Memory: `[B] [G] [R] [X]`. |
| 0x12 | `RGB565` | 16 | No | 16-bit RGB. Memory: `[GGGBBBBB] [RRRRRGGG]`. Native byte order. |
| 0x13 | `ARGB8565` | 24 | Yes | 24-bit ARGB. Not supported by SW renderer. |
| 0x14 | `RGB565A8` | 16+8 | Yes | RGB565 plane followed by separate A8 plane. |
| 0x15 | `AL88` | 16 | Yes | L8 with 8-bit alpha. |
| 0x16 | `ARGB1555` | 16 | Yes | 16-bit ARGB, 1-bit alpha. |
| 0x17 | `ARGB4444` | 16 | Yes | 16-bit ARGB, 4-bit alpha. |
| 0x18 | `ARGB2222` | 8 | Yes | 8-bit ARGB, 2-bit per channel. |
| 0x1A | `ARGB8888_PREMULTIPLIED` | 32 | Yes | Same as ARGB8888 but RGB is pre-multiplied by alpha. |
| 0x1B | `RGB565_SWAPPED` | 16 | No | RGB565 with bytes swapped (big-endian wire order). |

YUV formats (0x20–0x27) and proprietary formats (0x30+) are not applicable to EVE and are omitted.

---

## Format Mapping: LVGL to EVE

How the EVE5 LVGL driver maps LVGL color formats to EVE bitmap formats during upload (`lv_draw_eve5_image_upload.c`):

| LVGL Format | EVE Format | EVE BPP | Conversion Required | Notes |
|-------------|-----------|---------|---------------------|-------|
| `A1` (0x0B) | `L1` (1) | 1 | No | Direct mapping. |
| `A2` (0x0C) | `L2` (17) | 2 | No | Direct mapping. |
| `A4` (0x0D) | `L4` (2) | 4 | No | Direct mapping. |
| `L8` (0x06) | `L8` (3) | 8 | No | Direct mapping. |
| `A8` (0x0E) | `L8` (3) | 8 | No | Direct mapping. |
| `ARGB2222` (0x18) | `ARGB2` (5) | 8 | No | Direct mapping. |
| `RGB565` (0x12) | `RGB565` (7) | 16 | No | Direct mapping. |
| `RGB565_SWAPPED` (0x1B) | `RGB565` (7) | 16 | **Yes** | Byte swap per pixel. |
| `ARGB1555` (0x16) | `ARGB1555` (0) | 16 | No | Direct mapping. |
| `ARGB4444` (0x17) | `ARGB4` (6) | 16 | No | Direct mapping. |
| `RGB888` (0x0F) | `RGB8` (19) | 24 | No | Direct mapping. BT820 only. |
| `XRGB8888` (0x11) | `RGB8` (19) | 24 | **Yes** | Strip alpha byte (4 bpp → 3 bpp). |
| `ARGB8888` (0x10) | `ARGB8` (20) | 32 | No | Direct mapping. BT820 only. |
| `ARGB8888_PREMULTIPLIED` (0x1A) | `ARGB8` (20) | 32 | No | Direct mapping. |
| `RGB565A8` (0x14) | `ARGB8` (20) | 32 | **Yes** | Merge RGB565 + A8 planes into ARGB8. |
| `I1` (0x07) | `PALETTEDARGB8` (21) | 8 | **Yes** | Expand 1-bit indices to 8-bit. Pad palette to 256 entries. |
| `I2` (0x08) | `PALETTEDARGB8` (21) | 8 | **Yes** | Expand 2-bit indices to 8-bit. Pad palette to 256 entries. |
| `I4` (0x09) | `PALETTEDARGB8` (21) | 8 | **Yes** | Expand 4-bit indices to 8-bit. Pad palette to 256 entries. |
| `I8` (0x0A) | `PALETTEDARGB8` (21) | 8 | **Yes** | Palette prefix upload (layout matches but needs separate upload). |

### EVE GPU Memory Layout for Paletted Formats

When uploading indexed images, the palette and index data are stored contiguously in RAM_G:

```
base_addr + 0:              palette (256 * 4 = 1024 bytes, BGRA entries)
base_addr + 1024:           index data (stride * height bytes, 8-bit indices)
```

The `PALETTE_SOURCE` display list command points to `base_addr`. The `BITMAP_SOURCE` command points to `base_addr + 1024`.

In the EVE5 VRAM residency descriptor (`lv_eve5_vram_res_t`), these are stored as:
- `palette_offset = 0` (palette at allocation base)
- `source_offset = 1024` (bitmap data after palette)

Both offsets are relative to the `Esd_GpuHandle` base address, making them stable across GPU memory defragmentation.
