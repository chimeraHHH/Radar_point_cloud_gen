# STDA-F0 pre-freeze audit, round 4

> Date: 2026-08-08  
> Audited protocol SHA-256:
> `5fc5918453162cabf550cb2dfb8b7503123a10150cf80912f0d03281265da294`  
> Verdict: **NO-FREEZE**  
> Mode: independent hostile read-only audit; no repository files modified

The auditor confirmed that exact float32 spacing, the all-76 once-only cohort,
target canonicalization, graph/cost serialization, Hall verification, packed-
pointwise arithmetic, serialized CUDA children, summed process-tree NVML memory,
and the complete distinct-device mirror requirement were closed. Seven
transaction or executable-semantics blockers remained:

1. changing `NOT FROZEN` after approval would create an unaudited protocol SHA;
2. the scientific status was bound before the second copy, creating a circular
   status/root update if mirroring failed;
3. structural replay did not independently solve the global DP optimum;
4. greedy round origin and rotation direction were not fixed;
5. the audited target-denial process was not explicitly identified with the
   CUDA candidate-reconstruction child;
6. per-frame and complete-transaction timing endpoints were incomplete;
7. the generic no-go sentence overclaimed when parity support was below 10,000.

No implementation or scientific run was authorized. Revision 5 replaces the
freeze flag with external SHA-bound activation, separates immutable bundle bytes
from the terminal transaction record, defines one global commit point and crash
states, and closes the remaining executable semantics before round 5.
