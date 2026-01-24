- Output profile. Only really need to support sRGB and linear RGB at 8-bit currently
- Figure out the exact white points defined by all the various standard ICC profiles and review the behavior of moxcms defined whitepoints

---

- We're not yet detecting linear RGB on ICC profiles (we are on CICP profiles)
- For processing ICC profiles, we probably need to go through an ICC sRGB rather than a constructed profile, and then linearize, to be ICC-exact
  - There are multiple versions of sRGB and they match with different versions of ICC profile batches, so need to figure out a selector for that based on some table
- Similarly for CICP profiles, go through the CICP profile to linear sRGB directly

---

- An alternative way to handle the precision errors in colorspace specifications is to internally derive all the constants from first principles, so the rounded values conform to the spec, rather than exactly matching to the rounded values which are considered exact according to specification. The problem this solves is that currently we have multiple D65 whitepoints, as none of the specifications agree on the primary values used.
  - To handle ICC profiles correctly, we then need to have a mechanism to match color profiles to a most likely reference profile used. That is, the authoring display. In most case that means ICC color space -> PCS -> sRGB ICC using the ICC profile's transformations throughout, and then taking that transformation as ground reference for the image's true color.
  - In practice that means most ordinary color spaces map to sRGB, older Adobe RGB color spaces likely map to Apple RGB, and newer photographic colorspaces more likely map to Display P3.
- Then we need an explicit option to tonemap out-of-gamut colors
- Also need an explicit tonemapping filter option, basic support for going from HDR linear to tonemapped SDR

---

Currently binned histogram matching doesn't work correctly when the input file is not clamped within sRGB primaries, F32 should be fine

---

Theoretically, we would need a premultiplied alpha pipeline in parallel to the regular alpha pipeline
See https://openexr.com/en/latest/test_images/ScanLines/CandleGlass.html
This image file uses "0 alpha" to mean pure additive blending

---

We could theoretically do Bayer dithering with linear/sRGB awareness, by plotting all the quantized sRGB values in linear space, and then interpolating the intermediate values in linear space (as the dithering affects linear light emission -- it does not align with the gamma function)
Then when adding the matrix offset... add it linearly piecewise to these interpolated sections (normalize each section)

---

Add exr loading to web somehow if possible.

---

Tga saving with different formats?

---

Support re-using the palette of the input file as output palett, option PALETTE_INPUT

---

Could be possible to re-dither pngquant outputs, but pngquant generates mid-cluster palettes, which are inherently gamut limiting
--no-hull-tracing --hull-error-decay 0.95
^ this works as a workaround for now, but ideally we should generate a secondary concave hull to pre-limit the input image gamut to the generated palette
