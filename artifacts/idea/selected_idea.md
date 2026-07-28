# Selected parallel idea program

> Frozen on 2026-07-28 before H200 execution.

## Primary question

The next decision is whether the failure lies in **candidate support** or in
**learned point-budget allocation**. G1F-F0 is therefore the primary diagnostic:
it asks whether an explicitly unattainable GT oracle can select a valid
fixed-count subset from the frozen 32,000-point G1D measurement proposal pool.

F0 uses a coverage-oriented, capacity-one assignment. Within each fixed range
bin, every target votes for its nearest candidate; target confidence mass is
aggregated per candidate, unique candidates are ranked by covered mass and
distance, and unused quota is filled by nearest target support. This avoids the
invalid oracle that would simply cluster many candidates around one target.

If F0 fails the complete geometry gate, no selector trained on the same frozen
pool can be presented as the next answer. If F0 passes, G1F-F1 balanced
transport is authorized for one 10-epoch Stage-0 run.

## Independent mechanism branch

G1G tests the condition-bypass hypothesis. Its global Full-RAED encoder must
allocate 2,500 patch centers before any local Cube query. Only after allocation
may local spectra refine four bounded children per center. Cross-scene condition
shuffle is mandatory in preflight and evaluation.

G1G is independent from G1F: it changes the representation and information
path, not only the selector. It can proceed even if G1F-F0 fails.

## Independent temporal branch

G1T tests whether recent history adds measurement support that the current Cube
proposal pool lacks. It compares current-only, ego-warp union, and
Doppler-warp union under identical four-frame history, current-Cube rescoring,
fixed duplicate suppression, and exact 10,000-point export.

G1T is a no-train support diagnostic. Passing it does not establish future
prediction or generated Doppler; it only authorizes history as a proposal prior.

## Conservative control

G1H retains the frozen G1B `full_raed_rank2` model and replaces an equal number
of low-support isolated tail points with unused range-stratified Cube peaks. It
is deferred until the three primary Stage-0 jobs finish. It cannot be promoted
by masking points or reducing output count.

## No premature fusion

G1F, G1G, and G1T are evaluated independently. Mechanisms are combined only
after at least one learned geometry family passes its own frozen gate and the
other component shows an independent positive ablation. G1D v2 continues
unchanged to epoch 150 as a frozen control.
