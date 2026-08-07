# Selected geometry idea

> Successor revised on 2026-08-07 after the source-`34579a2` Q-Local-F0R
> scientific no-go. Test access is false.

## Closed predecessor: Q-Local-F0R

F0R removed only the mathematically contradictory per-frame range quotas from
the unchanged Fresh-WCE 700k field. Its formal H200 GPU0 GT-nearest oracle
passed exact 10k, true 5 cm, `CD <=0.8 m`, and outlier `<=5%` on all 12
train-only frames. It nevertheless retained every target-bearing range stratum
on only 2/12 frames. Ten frames spent almost the entire budget on dense near
returns and lost middle/far coverage.

Terminal status is `qlocal_f0r_capacity_no_go`; the Q-Local scorer was never
trained. This rejects independent pointwise quality followed by one global
capacity export on this frozen field. It does not reject the untrained scorer
in general or the existence of strong aggregate candidate support.

## Live primary route: variable multi-return renewal hazard

The successor represents each azimuth/elevation ray as an ordered radial
survival process. A predicted return ends the current survival interval; a
renewal/reset state then permits another return later on the same ray. The
number of returns is data-dependent rather than a fixed `K`, and return
allocation is coupled within a ray rather than decided independently for every
Cartesian candidate.

The first experiment is the frozen, independently audited, zero-training,
train-only paired capacity oracle in
`docs/vrh_f0_variable_return_capacity_protocol.md`. It uses one
target-independent `2R x 2A x 2E` lattice (`512 x 214 x 74`, 8,108,032
ray-range cells), a decoder-visible model-mark field separated from the GT audit
sidecar, bounded continuous event offsets, explicit renewal/STOP decoding, and
sequential-frontier versus flat exposure of one shared canonical stream. It must
establish that the representation can simultaneously support:

- exactly 10,000 unique points with true minimum spacing at least 5 cm;
- per-frame Chamfer `<=0.8 m` and 2 m outlier fraction `<=5%` on the frozen
  capacity cohort;
- explicit completeness and recall retention in every target-bearing range
  stratum;
- a target-free decoder/exporter contract, with GT identities and fitted
  distances restricted to the non-deployable audit sidecar;
- a variable number of ordered returns, with no fixed `K=4/6`, hard
  `8000/1700/300` quotas, copy, padding, jitter, or best-of-k repair;
- a renewal utility win: the decision arm passes all gates while flat exposure
  of the same decoded stream fails at least one corresponding gate.

## Why this route is first

1. F0R localizes the remaining failure to set-level range allocation rather
   than aggregate Chamfer or candidate count.
2. Renewal is the smallest representation change that can preserve multiple
   ordered surfaces along one radar ray after an earlier return.
3. It is outside the closed binary occupancy, arbitrary-query,
   fixed-neighborhood, and pointwise global-ranking families.
4. A capacity oracle can reject it before any network, optimizer, or checkpoint
   is created.

## Deferred fallback

Sparse ray-range partial transport is third priority. It requires a separately
frozen hard-rounding oracle and is not fused with renewal hazard before either
mechanism has independent evidence.

## Decision boundary

- capacity and renewal-utility pass on every frozen frame: freeze a bounded
  one-frame renewal-hazard memorization/scorer gate with a target-free Cube
  interface;
- both arms pass: record lattice-only capacity and do not authorize renewal;
- scientific capacity failure: close only the frozen recipe and route to the
  separately frozen sparse ray-range transport oracle;
- implementation- or resource-invalid: repair and rerun the identical capacity
  gate; do not route scientifically;
- no downstream Doppler, cycle, temporal, or test work starts until a 76/24
  geometry parent passes the complete frozen gate.
