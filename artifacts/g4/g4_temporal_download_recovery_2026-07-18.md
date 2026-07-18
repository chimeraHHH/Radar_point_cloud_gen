# G4 temporal download recovery audit (2026-07-18)

## Decision

G4 data preparation is recoverable but not ready to run. The temporal model,
training queue, and evaluation code are present. The blocking condition is the
incomplete official K-Radar download and unavailable NAS credentials, not a
model implementation failure.

## Frozen manifest

- Built on `WHUServer-H200` from source commit
  `a7d06db1abcc69c20dfed381f0c2909b1a89f026`.
- Inputs: the frozen scene-isolated G0 split, synchronized metadata, and
  official odometry.
- Output: `artifacts/g4/g4_temporal_manifest_a7d06db1.json`.
- SHA-256: `110391d79922226ccd4145e7dfec47ed9c39c0e74e1009fce588a057b0fbe8d2`.
- Gate: all 10 manifest checks pass.
- Cohort: 45 centered 48-frame windows, 2,160 frames total, with 1,776 train
  frames and 384 validation frames. Test sequences remain untouched.
- Estimated complete size: 600.773 GiB, including 632,389,680,000 Cube bytes
  and 12,685,390,560 LiDAR bytes.
- The 45 sequence IDs exactly match the existing download summary's requested
  sequence set.

## Storage audit

The data remains on the L40S storage host and is mounted read-only for project
computation through H200 SSHFS. A direct storage-side audit found:

- partial data size: 13 GiB;
- files present: 320;
- complete sequences reported by the downloader: 0 of 45;
- sequence 1-3 manifests exist, but each ended in a failed archive byte-range;
- partial compressed members remain for sequences 1-3, including nonzero
  `.part` files, and must not be deleted before resume;
- sequences 4-58 failed at Synology login before sequence manifests were
  produced;
- no active downloader remains;
- `KRADAR_NAS_USER` and `KRADAR_NAS_PASSWORD` are not present in the H200 login
  environment. Credential values were not searched for, printed, or archived.

The authoritative failed download summary is stored at:

`/storage/data/metaiot_data/wangning_radar/cube_dense_runs/g4_temporal_download_w48/manifests/summary.json`

## Recovery rule

1. Obtain a current authorized read-only NAS account and password through a
   secure channel. Never commit either value.
2. Resume the selective downloader against the frozen manifest above, retaining
   valid completed members and existing `.part` files.
3. Require all 45 sequences to appear in `completed_sequences` with zero
   failures.
4. Run exact-member, size, and CRC verification for all 2,160 Cube/LiDAR/label
   triplets before building dense targets.
5. Start `queue_g4_temporal.py` only after the download verification passes and
   the G2/G3 parent summary is final.

Until these conditions hold, G4 must remain pending and cannot contribute a
main-paper temporal claim.

## Recovery resumed

At `2026-07-18T14:43Z`, the current public read-only account documented by the
[official K-Radar dataset guide](https://github.com/kaist-avelab/K-Radar/blob/main/docs/dataset.md)
was verified without persisting its values. Direct
H200 access to the QuickConnect endpoint timed out, while the existing local
HTTP proxy at `127.0.0.1:7890` returned the endpoint and authenticated
successfully. This identifies network routing, rather than an invalid dataset
account, as the prior login failure.

The selective downloader was resumed on H200 from source commit
`188a9a2ef10c487edb887873bea446b56dc280cc`, retaining all existing `.part`
files. It uses two persistent sequence sessions and 12 byte-range workers per
session; this avoids the previous repeated concurrent-login pattern while
overlapping two sequence-local transfers. Credentials remain process-only and
are absent from the repository and logs.

The downloader now writes the global summary atomically after every completed
or failed sequence and retries only the unfinished sequence set. A process
restart may recheck already valid members, but member CRC checks and
sequence-local compressed ranges prevent silent partial-file promotion.

Runtime log:
`/home/wangning/Workspace/radar_cube_dense/launch_logs/188a9a2/g4_download.log`.

The G4 queue still waits for all 45 sequence manifests and a fresh exact-size
and CRC pass. Resuming the downloader does not alter or relax that gate.
