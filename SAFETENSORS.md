# Safetensors Image Format Specification

**Version:** 1.0
**Status:** Draft

---

## Overview

This specification defines a convention for storing floating-point RGB image data in safetensors files. It is designed for use in image processing pipelines where lossless, unquantized intermediate representations are required.

Safetensors provides secure tensor storage with JSON metadata support, making it suitable for image interchange when combined with standardized metadata fields.

---

## File Structure

A conforming file is a valid safetensors file containing:

1. A tensor named `image` containing pixel data
2. A `__metadata__` object with required fields describing the image format

Multiple images may be stored by using indexed tensor names (`image.0`, `image.1`, ...) with corresponding indexed metadata fields.

---

## Tensor Format

### Name

Single image files use the tensor name `image`. Multi-image files use `image.0`, `image.1`, etc.

### Data Type

| Type | Description |
|------|-------------|
| `F32` | 32-bit float (required support) |
| `F16` | 16-bit float (optional support) |
| `BF16` | bfloat16 (optional support) |

Implementations must support `F32`. Support for `F16` and `BF16` is optional.

### Shape

The tensor shape must be one of:

| Shape | Description |
|-------|-------------|
| `[H, W, C]` | Height × Width × Channels (HWC) |
| `[C, H, W]` | Channels × Height × Width (CHW) |

The dimension order is specified in metadata. `C` must be 3 for RGB or 4 for RGBA.

### Value Range

Values are linear light intensities (unless a nonlinear transfer function such as `srgb` is specified). The nominal range is `[0.0, 1.0]` where:

- `0.0` = zero intensity (black)
- `1.0` = reference white

Values outside `[0.0, 1.0]` are permitted and must be preserved. Negative values and values exceeding 1.0 may occur in HDR content or as intermediate computation results.

---

## Metadata Fields

Metadata is stored in the safetensors `__metadata__` object. All values are JSON strings.

### Required Fields

| Field | Values | Description |
|-------|--------|-------------|
| `format` | `"sfi"` | Format identifier (Safetensors Floating-point Image) |
| `version` | `"1.0"` | Specification version |
| `primaries` | See below | Color primaries identifier |
| `transfer` | See below | Transfer characteristics identifier |
| `channels` | `"RGB"`, `"RGBA"` | Channel configuration |
| `dimension_order` | `"HWC"`, `"CHW"` | Tensor dimension ordering |

### Optional Fields

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `white_point` | See below | Native white point of primaries | White point override |
| `alpha_premultiplied` | `"true"`, `"false"` | `"false"` | Whether RGB is premultiplied by alpha |

### Multi-Image Fields

For files containing multiple images, append the image index to field names:

```
primaries.0, primaries.1, ...
transfer.0, transfer.1, ...
channels.0, channels.1, ...
```

If an indexed field is absent, the non-indexed field value applies to all images.

---

## Color Primaries

Color primaries are specified using identifiers derived from CICP (Coding-Independent Code Points) as defined in ITU-T H.273, with extensions for common workflows.

### Primaries Identifiers

| Identifier | CICP | Description | Native White Point |
|------------|------|-------------|-------------------|
| `unspecified` | 2 | Unspecified (image-referred) | N/A |
| `bt709` | 1 | ITU-R BT.709-6 / IEC 61966-2-1 | `d65` |
| `srgb` | 1 | Alias for `bt709` | `d65` |
| `bt470m` | 4 | ITU-R BT.470-6 System M | `c` |
| `bt470bg` | 5 | ITU-R BT.470-6 System B, G | `d65` |
| `bt601` | 6 | ITU-R BT.601-7 / SMPTE 170M | `d65` |
| `smpte240` | 7 | SMPTE 240M | `d65` |
| `generic_film` | 8 | Generic film (Illuminant C) | `c` |
| `bt2020` | 9 | ITU-R BT.2020-2 / BT.2100 | `d65` |
| `xyz` | 10 | CIE 1931 XYZ (SMPTE ST 428-1) | Required |
| `smpte431` | 11 | SMPTE RP 431-2 / DCI-P3 | `dci` |
| `smpte432` | 12 | SMPTE EG 432-1 / Display P3 | `d65` |
| `ebu3213` | 22 | EBU Tech. 3213-E | `d65` |
| `prophoto` | — | ISO 22028-2 / ROMM RGB | `d50` |

Note: `srgb` is a convenience alias for `bt709`. Both identifiers refer to the same color space defined by chromaticity coordinates with white point `d65` (0.3127, 0.3290). Implementations must treat them identically.

### Chromaticity Coordinates (Authoritative)

All D65 color spaces use white point chromaticity (0.3127, 0.3290) as defined by their respective specifications. Matrices are derived from these chromaticity coordinates.

| Identifier | Red x | Red y | Green x | Green y | Blue x | Blue y |
|------------|-------|-------|---------|---------|--------|--------|
| `bt709` | 0.640 | 0.330 | 0.300 | 0.600 | 0.150 | 0.060 |
| `bt470m` | 0.670 | 0.330 | 0.210 | 0.710 | 0.140 | 0.080 |
| `bt470bg` | 0.640 | 0.330 | 0.290 | 0.600 | 0.150 | 0.060 |
| `bt601` | 0.630 | 0.340 | 0.310 | 0.595 | 0.155 | 0.070 |
| `smpte240` | 0.630 | 0.340 | 0.310 | 0.595 | 0.155 | 0.070 |
| `generic_film` | 0.681 | 0.319 | 0.243 | 0.692 | 0.145 | 0.049 |
| `bt2020` | 0.708 | 0.292 | 0.170 | 0.797 | 0.131 | 0.046 |
| `xyz` | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| `smpte431` | 0.680 | 0.320 | 0.265 | 0.690 | 0.150 | 0.060 |
| `smpte432` | 0.680 | 0.320 | 0.265 | 0.690 | 0.150 | 0.060 |
| `ebu3213` | 0.630 | 0.340 | 0.295 | 0.605 | 0.155 | 0.077 |
| `prophoto` | 0.7347 | 0.2653 | 0.1596 | 0.8404 | 0.0366 | 0.0001 |

---

## Transfer Characteristics

Transfer characteristics specify the opto-electronic transfer function applied to the image data.

### Transfer Identifiers

| Identifier | CICP | Description |
|------------|------|-------------|
| `unspecified` | 2 | Unspecified (image-referred) |
| `linear` | 8 | Linear (no transfer function) |
| `srgb` | 13 | IEC 61966-2-1 sRGB |
| `prophoto` | — | ISO 22028-2 ROMM RGB |

### Transfer Functions

#### linear

No transfer function applied. Values represent linear light intensity.

#### srgb

The sRGB piecewise transfer function.

**Linear to sRGB:**

```
if linear ≤ 0.0031308:
    srgb = 12.92 × linear
else:
    srgb = 1.055 × linear^(1/2.4) − 0.055
```

**sRGB to linear:**

```
if srgb ≤ 0.04045:
    linear = srgb / 12.92
else:
    linear = ((srgb + 0.055) / 1.055)^2.4
```

#### prophoto

The ROMM RGB piecewise transfer function (ISO 22028-2).

**Linear to ProPhoto:**

```
Et = 1/512  (≈0.001953)

if linear < Et:
    encoded = 16 × linear
else:
    encoded = linear^(1/1.8)
```

**ProPhoto to linear:**

```
if encoded < 16 × Et:
    linear = encoded / 16
else:
    linear = encoded^1.8
```

Note: Many implementations approximate this as a pure γ=1.8 power function, which is adequate for most purposes but technically incorrect per ISO 22028-2.

#### unspecified

The `unspecified` value indicates that the transfer function is unknown or does not correspond to a defined standard.

**Semantics:**

- The transfer function is unknown. Implementations should treat the data as image-referred (neither linear light nor a specific gamma).
- For display purposes, assuming sRGB-like transfer is reasonable.

**Recommendations:**

- Implementations may warn when reading files with `unspecified` values
- Producers should document the actual transfer function when known
- `unspecified` should not be used as a default; prefer explicit identification
- Processing pipelines should avoid mathematical operations that assume a specific transfer function

Note: Storing nonlinear values is discouraged for processing pipelines. The `srgb` and `prophoto` transfers are provided for final output compatibility.

---

## White Points

When `white_point` is omitted, the native white point of the specified primaries is used. The `white_point` field allows overriding this default.

### White Point Identifiers

White points for D65 and D50 color spaces are defined by the standards that use them:

| Identifier | Definition | Chromaticity (x, y) | XYZ (Y=1) |
|------------|------------|---------------------|-----------|
| `d65` | ITU-R BT.709 / IEC 61966-2-1 | 0.3127, 0.3290 | 0.9504559, 1.0, 1.0890578 |
| `d50` | ICC.1 / ISO 22028-2 | (derived) | 0.9642, 1.0, 0.8249 |
| `c` | FCC (NTSC 1953) | 0.310, 0.316 | 0.9810127, 1.0, 1.1835443 |
| `dci` | SMPTE RP 431-2 | 0.314, 0.351 | 0.8945869, 1.0, 0.9544160 |
| `e` | CIE equal-energy | 1/3, 1/3 | 1.0, 1.0, 1.0 |

**Notes:**

- `d65` chromaticity (0.3127, 0.3290) is authoritative. The XYZ values are derived from this chromaticity.
- `d50` XYZ (0.9642, 1.0, 0.8249) is authoritative as defined by ICC.1 and ISO 22028-2. The chromaticity (approximately 0.3457, 0.3585) is derived from these XYZ values.
- For CIE colorimetric applications requiring the 5-digit D65 (0.31272, 0.32903) or D50 (0.34567, 0.35850), convert through XYZ with appropriate precision.

### Usage

The `white_point` field is optional for RGB primaries and defaults to the native white point of those primaries.

For `xyz` primaries, `white_point` is required and specifies the illuminant to which the XYZ values are relative.

For `unspecified` primaries, `white_point` is not applicable and should be omitted. If present, implementations should ignore it.

---

## Chromatic Adaptation

When converting between color spaces with different white points, chromatic adaptation is required. This specification uses the Bradford transform.

**Bradford matrix (XYZ → LMS):**

```
| L |   |  0.8951000  0.2664000 -0.1614000 |   | X |
| M | = | -0.7502000  1.7135000  0.0367000 | × | Y |
| S |   |  0.0389000 -0.0685000  1.0296000 |   | Z |
```

**Adaptation procedure:**

To adapt XYZ values from source white point Ws to destination white point Wd:

1. Convert both white points to LMS: `LMS_Ws = BRADFORD × XYZ_Ws`
2. Compute scale factors: `scale = LMS_Wd / LMS_Ws`
3. Build adaptation matrix: `M_adapt = BRADFORD⁻¹ × diag(scale) × BRADFORD`
4. Apply: `XYZ_adapted = M_adapt × XYZ_source`

This adaptation matrix can be precomposed with RGB↔XYZ matrices for direct RGB-to-RGB conversion.

---

## Color Space Definitions

### RGB to XYZ Conversion

For RGB primaries, the conversion to XYZ is derived from the chromaticity coordinates and white point. The following matrices are provided for common configurations.

All matrices are derived from the authoritative chromaticity coordinates specified in the Color Primaries section, using the native white point for each color space.

#### bt709 / srgb (D65)

Derived from primaries (0.640, 0.330), (0.300, 0.600), (0.150, 0.060) and white point (0.3127, 0.3290).

```
RGB → XYZ:

| X |   | 0.4123908  0.3575843  0.1804808 |   | R |
| Y | = | 0.2126390  0.7151687  0.0721923 | × | G |
| Z |   | 0.0193308  0.1191948  0.9505322 |   | B |
```

```
XYZ → RGB:

| R |   |  3.2409699 -1.5373832 -0.4986108 |   | X |
| G | = | -0.9692436  1.8759675  0.0415551 | × | Y |
| B |   |  0.0556301 -0.2039770  1.0569715 |   | Z |
```

**Truncated form (per IEC 61966-2-1):**

For compatibility with the 4-decimal matrices published in IEC 61966-2-1:

```
RGB → XYZ:

| X |   | 0.4124  0.3576  0.1805 |   | R |
| Y | = | 0.2126  0.7152  0.0722 | × | G |
| Z |   | 0.0193  0.1192  0.9505 |   | B |
```

Note: The truncated matrices are adequate for 8-bit workflows. For higher bit depths, use the full-precision matrices above.

#### smpte432 (Display P3, D65)

Derived from primaries (0.680, 0.320), (0.265, 0.690), (0.150, 0.060) and white point (0.3127, 0.3290).

```
RGB → XYZ:

| X |   | 0.4865709  0.2656677  0.1982173 |   | R |
| Y | = | 0.2289746  0.6917385  0.0792869 | × | G |
| Z |   | 0.0000000  0.0451134  1.0439444 |   | B |
```

```
XYZ → RGB:

| R |   |  2.4934969 -0.9313836 -0.4027108 |   | X |
| G | = | -0.8294890  1.7626641  0.0236247 | × | Y |
| B |   |  0.0358458 -0.0761724  0.9568845 |   | Z |
```

#### bt2020 (D65)

Derived from primaries (0.708, 0.292), (0.170, 0.797), (0.131, 0.046) and white point (0.3127, 0.3290).

```
RGB → XYZ:

| X |   | 0.6369580  0.1446169  0.1688810 |   | R |
| Y | = | 0.2627002  0.6779981  0.0593017 | × | G |
| Z |   | 0.0000000  0.0280727  1.0609851 |   | B |
```

```
XYZ → RGB:

| R |   |  1.7166512 -0.3556708 -0.2533663 |   | X |
| G | = | -0.6666844  1.6164812  0.0157685 | × | Y |
| B |   |  0.0176399 -0.0427706  0.9421031 |   | Z |
```

#### prophoto (D50)

Per ISO 22028-2, the white point is defined as XYZ (0.9642, 1.0, 0.8249).

```
XYZ (D50) → RGB:

| R |   |  1.3460  -0.2556  -0.0511 |   | X |
| G | = | -0.5446   1.5082   0.0205 | × | Y |
| B |   |  0.0000   0.0000   1.2123 |   | Z |
```

### XYZ Color Space

When `primaries` is `xyz`, the tensor contains CIE 1931 XYZ tristimulus values. The `white_point` field is required and specifies the illuminant to which the values are relative.

The Y channel represents luminance, normalized such that Y=1.0 corresponds to the reference white.

---

## Conversion Between Color Spaces

### Same White Point

When source and destination share the same white point, conversion is a single matrix multiply through XYZ:

```
RGB_dest = M_XYZ_to_dest × M_source_to_XYZ × RGB_source
```

These can be precomposed into a single 3×3 matrix.

### Different White Points

When white points differ (e.g., D65 to D50), chromatic adaptation is required:

```
RGB_source → XYZ_source → adapt(Ws → Wd) → XYZ_dest → RGB_dest
```

---

## Examples

### Minimal Example (Python)

```python
from safetensors.torch import save_file
import torch

# Image data: 1920×1080 RGB, linear sRGB, float32
image = torch.rand(1080, 1920, 3, dtype=torch.float32)

metadata = {
    "format": "sfi",
    "version": "1.0",
    "primaries": "srgb",
    "transfer": "linear",
    "channels": "RGB",
    "dimension_order": "HWC",
}

save_file({"image": image}, "output.safetensors", metadata=metadata)
```

### HDR Content (BT.2020)

```python
metadata = {
    "format": "sfi",
    "version": "1.0",
    "primaries": "bt2020",
    "transfer": "linear",
    "channels": "RGB",
    "dimension_order": "HWC",
}
```

### Display P3

```python
metadata = {
    "format": "sfi",
    "version": "1.0",
    "primaries": "smpte432",
    "transfer": "linear",
    "channels": "RGB",
    "dimension_order": "HWC",
}
```

### ProPhoto RGB

```python
metadata = {
    "format": "sfi",
    "version": "1.0",
    "primaries": "prophoto",
    "transfer": "linear",
    "channels": "RGB",
    "dimension_order": "HWC",
}
```

### XYZ with Required White Point

```python
metadata = {
    "format": "sfi",
    "version": "1.0",
    "primaries": "xyz",
    "transfer": "linear",
    "channels": "RGB",  # XYZ stored as 3 channels
    "dimension_order": "HWC",
    "white_point": "d65",  # Required for xyz primaries
}
```

### Reading Example (Python)

```python
from safetensors import safe_open

with safe_open("output.safetensors", framework="pt") as f:
    metadata = f.metadata()
    
    assert metadata.get("format") == "sfi"
    assert metadata.get("version") == "1.0"
    
    image = f.get_tensor("image")
    
    dim_order = metadata.get("dimension_order")
    if dim_order == "CHW":
        image = image.permute(1, 2, 0)  # Convert to HWC
    
    primaries = metadata.get("primaries")
    # Normalize srgb alias to bt709
    if primaries == "srgb":
        primaries = "bt709"
    
    transfer = metadata.get("transfer")
    white_point = metadata.get("white_point")  # None if using default
    
    # Validate xyz has explicit white point
    if primaries == "xyz" and white_point is None:
        raise ValueError("xyz primaries require explicit white_point")
```

### Full Metadata Example

```json
{
    "__metadata__": {
        "format": "sfi",
        "version": "1.0",
        "primaries": "srgb",
        "transfer": "linear",
        "channels": "RGBA",
        "dimension_order": "HWC",
        "alpha_premultiplied": "false"
    },
    "image": {
        "dtype": "F32",
        "shape": [1080, 1920, 4],
        "data_offsets": [0, 33177600]
    }
}
```

---

## Implementation Requirements

### Conformance Levels

**Level 1 (Minimal):**
- Read and write `F32` tensors
- Support `unspecified`, `bt709`, and `srgb` primaries (treating `srgb` as alias for `bt709`)
- Support `unspecified`, `linear`, and `srgb` transfer
- Support `RGB` channels
- Support `HWC` dimension order

**Level 2 (Recommended):**
- All Level 1 requirements
- Support `F16` and `BF16` tensors
- Support `RGBA` channels and the premultiplied alpha flag
- Support `CHW` dimension order

**Level 3 (Enhanced):**
- All Level 2 requirements
- Support `bt2020`, `smpte432`, `prophoto`, and `xyz` primaries
- Support `prophoto` transfer
- Support all white point identifiers
- Implement Bradford chromatic adaptation

**Level 4 (Full):**
- All Level 3 requirements
- Support all defined primaries
- Support multi-image files

### Error Handling

Implementations must reject files where:
- `format` is not `"sfi"`
- `version` major version exceeds supported version
- Required metadata fields are missing
- Tensor shape does not match `dimension_order` and `channels`
- `primaries` is `"xyz"` and `white_point` is absent
- `primaries`, `transfer`, or `white_point` is an unrecognized identifier

Implementations should warn but may accept files where:
- Unknown optional metadata fields are present
- `version` minor version exceeds supported version

---

## Extension Points

Future versions may add:

- Additional primaries (ACEScg, Adobe RGB, etc.)
- Additional transfer functions (PQ, HLG, etc.)
- Additional white points (D55, D75, illuminant A, etc.)
- Additional channel configurations (grayscale, arbitrary channel counts)
- Per-image descriptive metadata (capture time, GPS, etc.)
- Tiled/mipmap storage for large images
- Animation/sequence support with timing metadata

Extensions must not redefine the meaning of existing fields. Unknown fields should be preserved on round-trip.

---

## Compression

The compressed variant of this file format is simply `.safetensors.gz`.

---

## References

- Safetensors format: https://huggingface.co/docs/safetensors
- ITU-T H.273: Coding-independent code points for video signal type identification
- IEC 61966-2-1:1999: Multimedia systems and equipment — Colour measurement and management — Part 2-1: Default RGB colour space — sRGB
- ITU-R BT.709-6: Parameter values for the HDTV standards for production and international programme exchange
- ITU-R BT.470-6: Conventional analogue television systems
- ITU-R BT.601-7: Studio encoding parameters of digital television for standard 4:3 and wide screen 16:9 aspect ratios
- ITU-R BT.2020-2: Parameter values for ultra-high definition television systems for production and international programme exchange
- ITU-R BT.2100-2: Image parameter values for high dynamic range television for use in production and international programme exchange
- ISO 22028-2:2013: Photography and graphic technology — Extended colour encodings for digital image storage, manipulation and interchange — Part 2: Reference output medium metric RGB colour image encoding (ROMM RGB)
- ICC.1:2022-05: Image technology colour management — Architecture, profile format and data structure
- SMPTE ST 428-1: D-Cinema Distribution Master — Image Characteristics
- SMPTE RP 431-2: D-Cinema Quality — Reference Projector and Environment
- SMPTE EG 432-1: Digital Source Processing — Color Processing for D-Cinema
- CIE 15:2004: Colorimetry (3rd Edition)
