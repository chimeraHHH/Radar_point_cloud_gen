# R-A1 RaLD-WCE H200 smoke

> Source: `f2a9489d40323d1ef45d85de958f4aea8126e1c8`
>
> Status: engineering preflight only
>
> Test accessed: false

The H200 smoke used two training frames, two validation frames from different
scenes, and two optimizer updates. It validates the trainer, same-query
wrong-Cube intervention, exact-10k export, and memory path. Its geometry is not
eligible for a scientific decision.

## Verification

- Targeted WCE tests: `23 passed in 1.49 s`
- Full repository tests on H200: `380 passed in 5.02 s`
- Smoke elapsed time: `43.54 s`
- Training loss: `0.54204`
- Peak training memory: `8.13 GiB` allocated, `8.27 GiB` reserved

The small smoke domain used Q0=`30,000` and Q1=`7,000`. Matched and wrong
conditions both exported exact 10,000 points with a minimum pair distance of
`5.014 cm`.

The separate full-domain pressure test used Q0=`500,000` and Q1=`200,000`.
Both arms exported exact 10,000 points; matched and wrong minimum pair
distances were `5.016 cm` and `5.018 cm`. Inference took `0.85 s`, with
`2.03 GiB` allocated and `2.27 GiB` reserved.

## Artifact hashes

| File | SHA-256 |
|---|---|
| `full_wide_inference.json` | `1a4eaea69f13c6b713fca1ab27d72608b761d37e4241527dcf6cf8c61b9df88a` |
| `metrics_epoch001.json` | `8b4c48eef8dd590dd3adaff7f6d7dfff56a3aa64bd7839a4744cac774c6b70b4` |
| `run_manifest.json` | `b8444966d6210025cf0f03174ac8b63b3a14f8b6a3dc7c168695fe41575d8389` |
| `summary.json` | `81f4e88be342a15750890eb5c42c4f077cd827ba64403af268cd09ed9972bf19` |
| `train_log.jsonl` | `f64176be475581d8ba573fb860495538a42039f8caab21d225ecf829d7c5c457` |

## Boundary

The smoke output is `XYZ+confidence`, not `XYZ+Doppler`. Wrong-Cube Chamfer
degradation was only `0.0204%`, and the matched arm won on `50%` of the two
frames. These are smoke observations, not evidence of useful condition
dependence. The formal 24-frame result must decide the frozen `1%` and `75%`
condition gates.
