# RaLD-AE-B1 hard-occupancy repair protocol

## Trigger

The preregistered matched RaLD one-frame overfit run at source commit
`d5aea417312ddc5a72a49fc157d5ad2853b2c08f` completed all 100 epochs but failed
the geometry gate:

- Chamfer distance: `11.2043 m` (required `<= 5.0 m`)
- outlier fraction at 2 m: `0.7559` (required `<= 0.5`)
- F-score at 1 m: `0.1008` (required `>= 0.2`)

The run did pass the optimization and confidence checks: train loss fell by
`72.27%`, and mean top-10k confidence was `0.0530`.

## Bounded diagnosis

Official RaLD samples points inside occupied voxels and assigns every such query
the binary label `1`. The matched adapter instead used each cached target's
measurement confidence as the occupancy label. On the overfit frame
`seq01/radar_00232`, these confidences have mean `0.2761` and median `0.1487`.
Together with the RaLD loss weights (`0.1` positive, `1.0` negative), this
changes the task from binary occupancy reconstruction to confidence regression
and drives the grid logits toward empty predictions.

## Single allowed repair

RaLD-AE-B1 changes positive query labels from target confidence to binary
occupancy `1`. Target confidence remains in the pipeline only as the sampling
weight for measured target points and as the geometry evaluation weight.

The following are frozen from the failed run:

- full 108,028,481-parameter autoencoder;
- one train frame and the same deterministic seed;
- 1,024 positive and 1,024 negative queries per epoch;
- `0.1 / 1.0 / 0.001` positive, negative, and KL loss weights;
- 100 epochs, optimizer, learning-rate schedule, top-10k grid decoding;
- no external pretraining and no CFAR query helper.

## Decision gate

Run the unchanged `verify_rald_ae_overfit.py` gate after epoch 100. B1 passes
only if all original thresholds pass. If B1 fails, stop the matched RaLD
training chain and retain it as a documented no-go baseline; do not tune a
second repair on this one-frame gate.
