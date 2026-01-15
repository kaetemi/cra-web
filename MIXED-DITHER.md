PER-PIXEL KERNEL SWITCHING ERROR DIFFUSION

A hybrid error diffusion technique that randomly switches between different
diffusion kernels on a per-pixel basis, disrupting the regular patterns that
each individual kernel produces.

Our implementation: FS-JJN-W (Floyd-Steinberg + Jarvis-Judice-Ninke + Wang hash)
- Switches between FS and JJN kernels
- Per-channel kernel selection (each RGB channel chooses independently)
- Wang hash for deterministic randomization

================================================================================

KERNEL DEFINITIONS:

Floyd-Steinberg (FS) - weights sum to 16:
        * 7
      3 5 1

Jarvis-Judice-Ninke (JJN) - weights sum to 48:
          * 7 5
      3 5 7 5 3
      1 3 5 3 1

For right-to-left scanning, mirror the weights horizontally.

================================================================================

WANG HASH:

wang_hash(x: u32) -> u32:
  x = (x ^ 61) ^ (x >> 16)
  x = x * 9
  x = x ^ (x >> 4)
  x = x * 0x27d4eb2d
  x = x ^ (x >> 15)
  return x

All operations are 32-bit unsigned with wrapping multiplication.

================================================================================

KERNEL SELECTION:

Pre-hash the seed once:
  hashed_seed = wang_hash(seed)

For each pixel at (x, y):
  pixel_hash = wang_hash(x ^ (y << 16) ^ hashed_seed)

  use_jjn_r = (pixel_hash & 1) != 0
  use_jjn_g = (pixel_hash & 2) != 0
  use_jjn_b = (pixel_hash & 4) != 0

Each channel applies either FS or JJN based on its flag.

================================================================================

SCANNING MODES:

MixedStandard (recommended):
  All rows left-to-right.

MixedSerpentine:
  Even rows (y % 2 == 0): left-to-right
  Odd rows (y % 2 == 1): right-to-left

MixedRandom:
  row_hash = wang_hash(y ^ hashed_seed)
  if (row_hash & 1) == 1: right-to-left
  else: left-to-right

Note: Serpentine and random scanning show no improvement over standard when
using per-pixel kernel switching. The kernel randomization already disrupts
patterns effectively, making scan direction variation redundant.

================================================================================

DEFAULTS:

seed = 12345
