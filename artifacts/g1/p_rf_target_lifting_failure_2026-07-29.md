# P-RF continuous target lifting preflight failure

> Date: 2026-07-29
>
> Source: `259663c80d85a24142af9c1294d16ce9173a571d`
>
> Test accessed: false

## Decision

The continuous target-lifting mechanism is scientifically bounded but cannot
construct a valid exact-10k P-RF target for every real K-Radar train/validation
frame. P-RF long training is not authorized.

The adapter fits a local weighted-PCA tangent plane, proposes points within a
0.6 m patch, retains only candidates that quantize to a target-occupied RAE
cell, and enforces 5 cm Euclidean spacing. It does not duplicate points, open
unobserved cells, use another frame, or change the spacing after validation.

## H200 result

- Targeted tests: `19 passed in 1.36 s`
- Sparse real frames checked: `57`
- Frames reaching certified exact-10k capacity: `47`
- Capacity failures: `10`
- Full model preflight: stopped before gradient, NFE, and memory checks

| Frame | Partition | Certified 5 cm capacity |
|---|---|---:|
| seq46/radar404 | train | 3,410 |
| seq47/radar514 | train | 4,986 |
| seq51/radar156 | validation | 9,901 |
| seq51/radar305 | validation | 6,396 |
| seq51/radar454 | validation | 4,527 |
| seq52/radar202 | train | 7,123 |
| seq57/radar205 | train | 8,223 |
| seq57/radar404 | train | 6,361 |
| seq58/radar205 | train | 8,621 |
| seq58/radar404 | train | 3,965 |

The first hard failure was:

```text
Continuous target certified 5 cm capacity 3410 is below required 10000;
refusing to widen observed RAE support
```

## Evidence boundary

No passing preflight JSON was emitted because the script uses atomic success
artifacts and failed before model execution. The result establishes a target
representation capacity obstruction, not a P-RF geometry metric. Real-frame
Cube querying, all-channel gradients, wrong-Cube sensitivity, NFE=4, and peak
H200 memory remain unexecuted for this sparse adapter.

The implementation is retained as an auditable failed mechanism. Reopening the
route requires a separately named target representation with a preregistered
physical support interpretation; post-hoc spacing relaxation, padding, or
unobserved-cell expansion is prohibited.
