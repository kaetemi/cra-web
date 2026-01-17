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
