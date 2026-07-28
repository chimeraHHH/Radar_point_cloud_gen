# G1G Stage-0: condition-exclusive hierarchical query allocation

## Hypothesis

G1D ignores its global radar condition because local Cube spectrum, energy, and
range are available at every query before point allocation. Requiring the global
Full-RAED tokens to allocate coarse patch centers, then allowing only bounded
local refinement, should make condition usage causal and reduce point crowding.

## Architecture

1. Encode the current Full-RAED Cube into the existing 336 spatial radar tokens.
2. Decode exactly 2,500 coarse center queries from learned query embeddings and
   global radar tokens. Center scoring and coordinates may use normalized
   coordinates and global tokens only.
3. After center allocation, sample the local Cube spectrum at each center.
4. Expand every center into exactly four children using a shared patch decoder.
5. Bound every child offset by a preregistered physical radius derived from one
   Cube-cell diagonal. Export exactly 10,000 points.

No direct local spectrum, energy, proposal score, or neighborhood feature may
enter steps 1-2. This prohibition is checked in preflight.

## Stage-0

- one seed, 20 epochs, current train/validation split;
- same geometry losses and metrics as G1D, with center repulsion and bounded
  child diversity replacing unrestricted 10k offset growth;
- zero-offset, no-local-refinement, child-collapse, and cross-scene condition
  shuffle controls;
- report center occupancy, children-per-center, duplicate fraction, range mass,
  and per-layer condition gradients.

## Promotion

Stage-0 requires all of:

- condition-shuffle Chamfer degradation `>=1%`;
- duplicate fraction `<=15%`;
- at least 30% completeness improvement over the matched G1D epoch-15 control;
- outlier fraction `<=25%`;
- no far-range regression relative to that control.

These screening conditions authorize a full frozen run but do not replace the
final geometry gate.

## Abandonment

Close G1G after Stage-0 if condition shuffle fails, if centers collapse into
fewer than 80% unique 5 cm cells, or if completeness improves only by violating
the outlier/far-range conditions.
