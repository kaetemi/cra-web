- Output profile. Only really need to support sRGB and linear RGB at 8-bit currently
- Figure out the exact white points defined by all the various standard ICC profiles and review the behavior of moxcms defined whitepoints

---

- We're not yet detecting linear RGB on ICC profiles (we are on CICP profiles)
- For processing ICC profiles, we probably need to go through an ICC sRGB rather than a constructed profile, and then linearize, to be ICC-exact
  - There are multiple versions of sRGB and they match with different versions of ICC profile batches, so need to figure out a selector for that based on some table
- Similarly for CICP profiles, go through the CICP profile to linear sRGB directly

---

- Alpha channel for luminosity-only output dithering
- Separate control for alpha channel dithering (so we can dither alpha with None)

---

- When the rendering pipeline is sRGB-only without colorspace-awareness, alpha channels tend to be interpreted as being in sRGB-space rather than linear space. Might need an option to correct for that, on both input and output, even though it's non-standard? Although technically, I think, it should appear the same regardless (when comparing ARGB8 vs ARGB1), with some minor bias in how the error is propagated...

---

- An alternative way to handle the precision errors in colorspace specifications is to internally derive all the constants from first principles, so the rounded values conform to the spec, rather than exactly matching to the rounded values which are considered exact according to specification. The problem this solves is that currently we have multiple D65 whitepoints, as none of the specifications agree on the primary values used.
  - To handle ICC profiles correctly, we then need to have a mechanism to match color profiles to a most likely reference profile used. That is, the authoring display. In most case that means ICC color space -> PCS -> sRGB ICC using the ICC profile's transformations throughout, and then taking that transformation as ground reference for the image's true color.
  - In practice that means most ordinary color spaces map to sRGB, older Adobe RGB color spaces likely map to Apple RGB, and newer photographic colorspaces more likely map to Display P3.
- Then we need an explicit option to tonemap out-of-gamut colors
- Also need an explicit tonemapping filter option, basic support for going from HDR linear to tonemapped SDR

---

- For tonemapping, need both an input and output tonemapping stage (so we can tonemap going into color correction, and then reverse tonemap going back to linear space, theoretically -- this is not technically correct, but a practical option for binned histogram matching of "unbounded" linear RGB)
- Additionally, have an option to enable supersampling the tonemapping, meaning, the image gets lanczos upscaled before and downscaled after the actual tonemapping function is applied. This more accurately applies the tonemapping in physical source light space, and samples the image in physical display space.
