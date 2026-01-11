## Integer Color Format Conversion Relations

### The Canonical Model

**uint8 and uint16 are the canonical integer representations.** They are mutually coherent—bit replication and truncation between them is exact, and their float conversions agree:

```
uint8 ←→ uint16      (bit replication / truncation)
  ↓           ↓
/255       /65535
  ↓           ↓
float  ===  float    (identical values)
```

All other bit depths are defined by their relationship to uint8.

**Float is defined as uint8/255 (or equivalently uint16/65535).** This is the canonical exit to floating point.

---

### Correct Conversions

**Bit Extension (lower → higher bit depth)**
Replicate the source bits to fill the destination:
```
uint3 ABC → uint8 ABCABCAB
uint4 ABCD → uint8 ABCDABCD
uint8 ABCDEFGH → uint16 ABCDEFGH ABCDEFGH
```

**Bit Truncation (higher → lower bit depth)**
Keep the most significant bits:
```
uint8 → uint3: value >> 5
uint8 → uint4: value >> 4
uint16 → uint8: value >> 8
```

**Truncation is nearest.** In the bit-replication model, truncation always yields the value whose extension is closest to the original. Rounding (adding half before truncating) is incorrect—it imports Euclidean assumptions into a system where they don't belong.

**Integer to Float**
Extend to canonical depth first, then divide:
```
uint3 → float: extend to uint8, then /255
uint8 → float: /255
uint16 → float: /65535
```

**Float to Integer**
Multiply, then truncate to desired depth:
```
float → uint8: clamp(f * 255, 0, 255), then truncate bits if needed
```

---

### Coherent Bit Depths

Bit depths that divide evenly into 8 are fully coherent—bit replication, truncation, and float conversion all align perfectly:

| Depth | Divides evenly into |
|-------|---------------------|
| uint1 | 2, 4, 8, 16, 32 |
| uint2 | 4, 8, 16, 32 |
| uint4 | 8, 16, 32 |
| uint8 | 16, 32 |
| uint16 | 32 |

For these, bit replication to the larger depth followed by division yields identical float values to direct division by `(2^n - 1)`.

---

### Non-Divisor Bit Depths

Bit depths that don't divide evenly into 8:

| Depth | Used in |
|-------|---------|
| uint3 | RGB332 |
| uint5 | RGB565 |
| uint6 | RGB565 |

For these depths:
- **Bit replication is exact by definition.** This is what the format means.
- **Bit truncation is exact.** It is the perfect inverse of replication.
- **Float conversion via /255 is the definition.** The value `extend_to_uint8(v) / 255.0` is the correct float representation.

The theoretical interpretation `v / (2^n - 1)` yields a slightly different float value, but this interpretation was never the standard. The difference is a comparison to a fiction, not an error in the system.

---

### Incorrect Conversions

**Dividing by (2^n - 1) directly**
```
// WRONG: uint3 → float as v/7
float f = uint3_value / 7.0;

// CORRECT: extend first
float f = replicate_to_uint8(uint3_value) / 255.0;
```

The "mathematically clean" interpretation of each bit depth having its own divisor doesn't match how formats are actually defined.

**Rounding during truncation**
```
// WRONG: rounding
uint3 = (uint8 + 16) >> 5;

// CORRECT: truncation
uint3 = uint8 >> 5;
```

Rounding breaks the inverse relationship with bit extension and can overflow at max values.

**Mixing divisors across bit depths**
```
// WRONG: comparing values using their "native" divisors
float a = uint3_value / 7.0;
float b = uint5_value / 31.0;

// CORRECT: extend both to canonical depth
float a = replicate_3_to_8(uint3_value) / 255.0;
float b = replicate_5_to_8(uint5_value) / 255.0;
```

---

### Why This Works: The Infinite Bit Replication Model

A uint_n value represents an infinitely repeating binary fraction:
```
uint3 = 1 (001) → 0.001001001001... (binary)
```

This infinite series sums to exactly `v / (2^n - 1)`, which is why the endpoints work:
```
0.111111... (binary) = 1.0 (exactly, like 0.999... = 1 in decimal)
```

Bit extension reveals more digits of this infinite pattern. Bit truncation recovers the original pattern. These operations are exact inverses by construction.

When extending to a non-multiple bit depth (e.g., 3→8), the pattern truncates at a non-aligned boundary. The subsequent `/255` then interprets an 8-bit repeating pattern rather than the original 3-bit repeating pattern. This is not an error—it is the definition. The uint8 form is canonical.

---

### Summary

| Operation | Method | Exact? |
|-----------|--------|--------|
| uint_n → uint8 | bit replication | exact (by definition) |
| uint8 → uint_n | bit truncation (>>) | exact (inverse of above) |
| uint8 ↔ uint16 | bit replication / truncation | exact (8 divides 16) |
| uint8 → float | /255 | exact (by definition) |
| uint16 → float | /65535 | exact (coherent with uint8) |
| uint3 → float | extend to uint8, then /255 | exact (by definition) |

The system is fully coherent when you respect uint8 or uint16 as canonical. The only "incoherence" is between the actual standard and the theoretical `v / (2^n - 1)` model—but that model was never the standard.