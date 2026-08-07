# Selected geometry idea

> Successor revised on 2026-08-07 after the Q-Local-F0 capacity no-go and the
> fixed-range-quota contract no-go. Test access is false.

## Live primary route: Q-Local-F0R

Q-Local-F0 is terminal under its frozen `8000/1700/300` exporter: the
source-bound GT-nearest oracle passed only 10/12 train frames, so its scorer was
not trained. A separate 76-frame radial lower-bound audit then proved that the
hard per-frame quotas are themselves incompatible with the geometry contract
on frames `47:94` and `58:404`.

Q-Local-F0R is the narrowest valid successor. It keeps the source-`f2a9489`
Fresh-WCE checkpoint, all 700k candidate coordinates, Q0/Q1, residual, base
confidence, Q-Local scorer/loss, cohort, exact 10k, and true 5 cm spacing. It
changes only the exporter: one stable global score order across `[0,120)` m,
followed by greedy 5 cm selection.

The first run is the zero-training capacity gate in
`docs/qlocal_f0r_global_export_capacity_protocol.md`. It uses the same eight fit
and four unseen-train frames. All 12 frames must pass `CD <=0.8 m`, outlier
`<=5%`, exact 10k, true 5 cm, and per-target-stratum retention against the
archived fixed-quota oracle.

## Why this route is first

1. It removes one output constraint already proven impossible instead of
   changing the representation prematurely.
2. It leaves candidate support and the unresolved local Full-RAED scoring
   hypothesis bit-identical.
3. It has a decisive zero-training result before any optimizer or checkpoint is
   created.
4. Its target-stratum retention gate prevents winning by discarding real
   middle/far targets.

## Outside-family fallback

Only if Q-Local-F0R fails its frozen scientific gate, run a zero-training
representation oracle for a variable multi-return renewal-hazard field. Fixed
per-ray `K=4/6`, larger arbitrary-query pools, another binary occupancy model,
hard per-frame range quotas, and fixed-neighborhood activation remain closed.

Sparse ray-range partial transport is third priority and requires a separate
hard-rounding oracle before implementation. It is not fused with Q-Local.

## Decision boundary

- F0R 12/12 capacity pass: freeze a separate 500-update Q-Local training run
  using the same global exporter and recompute every learned baseline with it.
- F0R scientific failure: close scoring on this fresh 700k field and move to
  variable multi-return renewal hazard.
- F0R implementation-invalid: repair and rerun the identical gate; do not route
  scientifically.
- Later learning or unseen-control failure by update 500/7200 GPU-s: close this
  scorer architecture without tuning; move to variable ray hazard.
- No downstream Doppler, cycle, temporal, or test work starts until a 76/24
  geometry parent passes the complete frozen gate.
