# Per-Pixel Kernel Switching Error Diffusion

A hybrid error diffusion technique that randomly switches between different diffusion kernels on a per-pixel basis, disrupting the regular patterns that each individual kernel produces.

**Our implementation:** FS-JJN-LB (Floyd-Steinberg + Jarvis-Judice-Ninke + lowbias32 hash)
- Switches between FS and JJN kernels
- Per-channel kernel selection (each RGB channel chooses independently)
- lowbias32 hash for deterministic randomization

---

## Kernel Definitions

**Floyd-Steinberg (FS)** - weights sum to 16:

```
      * 7
    3 5 1
```

**Jarvis-Judice-Ninke (JJN)** - weights sum to 48:

```
        * 7 5
    3 5 7 5 3
    1 3 5 3 1
```

For right-to-left scanning, mirror the weights horizontally.

---

## lowbias32 Hash

Optimized for low bias (0.107) with excellent spatial randomization properties.
Reference: https://github.com/skeeto/hash-prospector/issues/19

```
lowbias32(x: u32) -> u32:
  x = x ^ (x >> 16)
  x = x * 0x21f0aaad
  x = x ^ (x >> 15)
  x = x * 0x735a2d97
  x = x ^ (x >> 15)
  return x
```

All operations are 32-bit unsigned with wrapping multiplication.

---

## Kernel Selection

Pre-hash the seed once:

```
hashed_seed = lowbias32(seed)
```

For each pixel at (x, y):

```
pixel_hash = lowbias32(x ^ (y << 16) ^ hashed_seed)

use_jjn_r = (pixel_hash & 1) != 0
use_jjn_g = (pixel_hash & 2) != 0
use_jjn_b = (pixel_hash & 4) != 0
```

Each channel applies either FS or JJN based on its flag.

---

## Scanning Modes

**MixedStandard** (recommended): All rows left-to-right.

**MixedSerpentine:**
- Even rows (y % 2 == 0): left-to-right
- Odd rows (y % 2 == 1): right-to-left

**MixedRandom:**

```
row_hash = lowbias32(y ^ hashed_seed)
if (row_hash & 1) == 1: right-to-left
else: left-to-right
```

Note: Serpentine and random scanning show no improvement over standard when using per-pixel kernel switching. The kernel randomization already disrupts patterns effectively, making scan direction variation redundant.

---

## Defaults

- `seed = 12345`
