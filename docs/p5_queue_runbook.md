# P5 Frozen Test Queue

`queue_p5_final.py` is the only supported test-release path. It waits for a
completed G4 queue summary before constructing a test-only temporal manifest.
G4 may pass or fail; either outcome freezes the selected temporal family, while
failure keeps temporal evidence in the appendix.

After release, the queue requires a separate eight-sequence downloader summary
and fresh CRC verification. It then runs, in order:

1. 384-frame radar-observable dense cache;
2. three same-seed parent prediction caches;
3. test-only T0/T3 and frozen T* rollout reports;
4. object radial-velocity evaluation with train-frozen Doppler sign provenance;
5. matched CUDA latency/memory benchmark;
6. descriptive scene-first bootstrap, slices and failure taxonomy.

The queue writes `p5_queue_summary_<commit>.json` only after every artifact is
complete. Test results never create a gate or select a model.
