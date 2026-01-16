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

Values are linear light intensities (unless a nonlinear color space such as `srgb` is specified). The nominal range is `[0.0, 1.0]` where:

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
| `color_space` | See below | Color space identifier |
| `channels` | `"RGB"`, `"RGBA"` | Channel configuration |
| `dimension_order` | `"HWC"`, `"CHW"` | Tensor dimension ordering |

### Optional Fields

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `white_point` | See below | `"default"` | White point override or specification |
| `alpha_premultiplied` | `"true"`, `"false"` | `"false"` | Whether RGB is premultiplied by alpha |

### Multi-Image Fields

For files containing multiple images, append the image index to field names:

```
color_space.0, color_space.1, ...
channels.0, channels.1, ...
```

If an indexed field is absent, the non-indexed field value applies to all images.

---

## White Points

White points are specified by identifier. Each corresponds to a precise chromaticity coordinate.

### White Point Identifiers

| Identifier | x | y | Source |
|------------|---------------------|---------------------|--------|
| `d65_itu` | 0.3127 | 0.3290 | ITU-R BT.709, BT.2020, BT.2100 |
| `d65_cie` | 0.31272 | 0.32903 | CIE 15:2004 |
| `d65_srgb` | 0.31271590722158249 | 0.32900148050666228 | Derived from IEC 61966-2-1 matrix |
| `d50_cie` | 0.34567 | 0.35850 | CIE 15:2004 |
| `default` | — | — | Use color space's native white point |

### Derived XYZ Values

For implementations requiring XYZ tristimulus values (normalized to Y=1):

| Identifier | X | Y | Z |
|------------|---------|---------|---------|
| `d65_itu` | 0.95047 | 1.00000 | 1.08883 |
| `d65_cie` | 0.95047 | 1.00000 | 1.08883 |
| `d65_srgb` | 0.95050 | 1.00000 | 1.08900 |
| `d50_cie` | 0.96422 | 1.00000 | 0.82521 |

Note: `d65_itu` and `d65_cie` round to the same XYZ at 5 decimal places but differ at higher precision.

### Usage

The `white_point` field specifies which white point is in use:

- For RGB color spaces: `default` uses the space's native white point. Other values indicate the data has been adapted or the matrices were derived using the specified white point.
- For `ciexyz`: The white point is required and specifies the illuminant to which the values are relative. `default` is not valid for `ciexyz`.

---

## Color Spaces

### Supported Color Spaces

| Identifier | Description | Native White Point |
|------------|-------------|-------------------|
| `linear_srgb` | Linear RGB with sRGB/Rec.709 primaries | `d65_srgb` |
| `srgb` | sRGB with transfer function applied | `d65_srgb` |
| `linear_displayp3` | Linear RGB with Display P3 primaries | `d65_itu` |
| `linear_rec2020` | Linear RGB with Rec.2020 primaries | `d65_itu` |
| `linear_adobergb` | Linear RGB with Adobe RGB (1998) primaries | `d65_itu` |
| `ciexyz` | CIE 1931 XYZ tristimulus values | None (must be specified) |

### Chromatic Adaptation

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

### linear_srgb

Linear light RGB with sRGB/Rec.709 primaries.

**Native white point:** `d65_srgb`

**Chromaticity coordinates:**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6400 | 0.3300 |
| Green | 0.3000 | 0.6000 |
| Blue | 0.1500 | 0.0600 |

**XYZ conversion (defined as exact per IEC 61966-2-1):**

```
Linear sRGB → XYZ:

| X |   | 0.4124  0.3576  0.1805 |   | R |
| Y | = | 0.2126  0.7152  0.0722 | × | G |
| Z |   | 0.0193  0.1192  0.9505 |   | B |
```

```
XYZ → Linear sRGB:

| R |   |  3.2406255 -1.5372080 -0.4986286 |   | X |
| G | = | -0.9689307  1.8757561  0.0415175 | × | Y |
| B |   |  0.0557101 -0.2040211  1.0569959 |   | Z |
```

Note: The chromaticity coordinates above are derived from the matrix. The matrix is the normative definition.

### srgb

sRGB with the piecewise transfer function applied. Values are nonlinear.

**Native white point:** `d65_srgb`

**Transfer function (linear → sRGB):**

```
if linear ≤ 0.0031308:
    srgb = 12.92 × linear
else:
    srgb = 1.055 × linear^(1/2.4) − 0.055
```

**Transfer function (sRGB → linear):**

```
if srgb ≤ 0.04045:
    linear = srgb / 12.92
else:
    linear = ((srgb + 0.055) / 1.055)^2.4
```

Note: Storing nonlinear values is discouraged for processing pipelines. This color space is provided for final output compatibility.

### linear_displayp3

Linear light RGB with Display P3 primaries.

**Native white point:** `d65_itu`

**Chromaticity coordinates (defined as exact):**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6800 | 0.3200 |
| Green | 0.2650 | 0.6900 |
| Blue | 0.1500 | 0.0600 |

**XYZ conversion (derived from primaries and native white point):**

```
Linear Display P3 → XYZ:

| X |   | 0.4865709  0.2656677  0.1982173 |   | R |
| Y | = | 0.2289746  0.6917385  0.0792869 | × | G |
| Z |   | 0.0000000  0.0451134  1.0439444 |   | B |
```

```
XYZ → Linear Display P3:

| R |   |  2.4934969 -0.9313836 -0.4027108 |   | X |
| G | = | -0.8294890  1.7626641  0.0236247 | × | Y |
| B |   |  0.0358458 -0.0761724  0.9568845 |   | Z |
```

### linear_rec2020

Linear light RGB with Rec.2020 primaries (ITU-R BT.2020).

**Native white point:** `d65_itu`

**Chromaticity coordinates (defined as exact per ITU-R BT.2020):**

| Primary | x | y |
|---------|-------|-------|
| Red | 0.708 | 0.292 |
| Green | 0.170 | 0.797 |
| Blue | 0.131 | 0.046 |

**XYZ conversion (derived from primaries and native white point):**

```
Linear Rec.2020 → XYZ:

| X |   | 0.6369580  0.1446169  0.1688810 |   | R |
| Y | = | 0.2627002  0.6779981  0.0593017 | × | G |
| Z |   | 0.0000000  0.0280727  1.0609851 |   | B |
```

```
XYZ → Linear Rec.2020:

| R |   |  1.7166512 -0.3556708 -0.2533663 |   | X |
| G | = | -0.6666844  1.6164812  0.0157685 | × | Y |
| B |   |  0.0176399 -0.0427706  0.9421031 |   | Z |
```

### linear_adobergb

Linear light RGB with Adobe RGB (1998) primaries.

**Native white point:** `d65_itu`

**Chromaticity coordinates (defined as exact per Adobe RGB (1998)):**

| Primary | x | y |
|---------|--------|--------|
| Red | 0.6400 | 0.3300 |
| Green | 0.2100 | 0.7100 |
| Blue | 0.1500 | 0.0600 |

**XYZ conversion (derived from primaries and native white point):**

```
Linear Adobe RGB → XYZ:

| X |   | 0.5766690  0.1855582  0.1882286 |   | R |
| Y | = | 0.2973450  0.6273636  0.0752915 | × | G |
| Z |   | 0.0270314  0.0706889  0.9913375 |   | B |
```

```
XYZ → Linear Adobe RGB:

| R |   |  2.0415879 -0.5650070 -0.3447314 |   | X |
| G | = | -0.9692436  1.8759675  0.0415551 | × | Y |
| B |   |  0.0134443 -0.1183624  1.0151750 |   | Z |
```

### ciexyz

CIE 1931 XYZ tristimulus values.

**Native white point:** None (must be specified)

XYZ tristimulus values are defined by the CIE 1931 color matching functions. The `white_point` field is required and specifies the illuminant to which the values are relative. Using `default` for `white_point` with `ciexyz` is an error.

The Y channel represents luminance, normalized such that Y=1.0 corresponds to the reference white.

---

## Conversion Between Color Spaces

### Same White Point

When source and destination color spaces share the same native white point (or `white_point` field), conversion is a single matrix multiply through XYZ:

```
RGB_dest = M_XYZ_to_dest × M_source_to_XYZ × RGB_source
```

These can be precomposed into a single 3×3 matrix.

### Different White Points

When white points differ, chromatic adaptation is required:

```
RGB_source → XYZ_source → adapt(Ws → Wd) → XYZ_dest → RGB_dest
```

**Example: linear_srgb to linear_rec2020**

Source white point: `d65_srgb` (0.31272, 0.32900)
Destination white point: `d65_itu` (0.3127, 0.3290)

1. Convert linear sRGB to XYZ using the sRGB matrix
2. Apply Bradford adaptation from `d65_srgb` to `d65_itu`
3. Convert XYZ to linear Rec.2020 using the Rec.2020 matrix

Or precompose all three into a single matrix.

### Bypassing Adaptation

For workflows where white point precision is not critical, implementations may treat `d65_srgb`, `d65_itu`, and `d65_cie` as equivalent and skip adaptation. This introduces error on the order of 0.00002 in chromaticity.

To explicitly request this behavior, set `white_point` to the same value on both source and destination, which forces the matrices to use a common white point.

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
    "color_space": "linear_srgb",
    "channels": "RGB",
    "dimension_order": "HWC",
}

save_file({"image": image}, "output.safetensors", metadata=metadata)
```

### With Explicit White Point

```python
metadata = {
    "format": "sfi",
    "version": "1.0",
    "color_space": "linear_rec2020",
    "channels": "RGB",
    "dimension_order": "HWC",
    "white_point": "default",  # Use Rec.2020's native d65_itu
}
```

### XYZ with White Point

```python
metadata = {
    "format": "sfi",
    "version": "1.0",
    "color_space": "ciexyz",
    "channels": "RGB",  # XYZ stored as 3 channels
    "dimension_order": "HWC",
    "white_point": "d65_cie",  # Required for ciexyz
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
    
    color_space = metadata.get("color_space")
    white_point = metadata.get("white_point", "default")
    
    # Validate ciexyz has explicit white point
    if color_space == "ciexyz" and white_point == "default":
        raise ValueError("ciexyz requires explicit white_point")
```

### Full Metadata Example

```json
{
    "__metadata__": {
        "format": "sfi",
        "version": "1.0",
        "color_space": "linear_srgb",
        "channels": "RGBA",
        "dimension_order": "HWC",
        "white_point": "default",
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
- Support `linear_srgb` and `srgb` color spaces
- Support `RGB` channels
- Support `HWC` dimension order
- Support `default` white point

**Level 2 (Recommended):**
- All Level 1 requirements
- Support `F16` tensors
- Support `RGBA` channels
- Support `CHW` dimension order
- Support `linear_displayp3`, `linear_rec2020`, and `ciexyz` color spaces
- Support all white point identifiers
- Implement Bradford chromatic adaptation

**Level 3 (Full):**
- All Level 2 requirements
- Support `BF16` tensors
- Support all defined color spaces
- Support multi-image files

### Error Handling

Implementations must reject files where:
- `format` is not `"sfi"`
- `version` major version exceeds supported version
- Required metadata fields are missing
- Tensor shape does not match `dimension_order` and `channels`
- `white_point` is `"default"` for `ciexyz` color space
- `white_point` is an unrecognized identifier

Implementations should warn but may accept files where:
- Unknown optional metadata fields are present
- `version` minor version exceeds supported version

---

## Extension Points

Future versions may add:

- Additional color spaces (ProPhoto RGB, ACEScg, etc.)
- Additional white points (D55, D75, illuminant A, etc.)
- Additional channel configurations (grayscale, arbitrary channel counts)
- Per-image descriptive metadata (capture time, GPS, etc.)
- Tiled/mipmap storage for large images
- Animation/sequence support with timing metadata

Extensions must not redefine the meaning of existing fields. Unknown fields should be preserved on round-trip.

---

## References

- Safetensors format: https://huggingface.co/docs/safetensors
- IEC 61966-2-1:1999 (sRGB)
- ITU-R BT.709-6 (Rec.709)
- ITU-R BT.2020-2 (Rec.2020)
- CIE 15:2004 (Colorimetry)