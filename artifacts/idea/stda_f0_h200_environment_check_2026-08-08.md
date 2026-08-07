# STDA-F0 H200 environment check

> Date: 2026-08-08  
> Role: pre-freeze package/resource availability only  
> Data/Cube/target access: false

The check ran as `wangning` on `WHUServer-H200` without loading project data or
executing STDA.

## Pinned runtime

| Component | Observed |
|---|---|
| Python | `3.10.20` |
| NumPy | `2.2.6` |
| SciPy | `1.15.3` |
| PyTorch | `2.12.1+cu130` |
| CUDA reported by PyTorch | `13.0` |
| NVIDIA driver | `595.71.05` |

## Allowed GPU

- physical index: `2`;
- model: `NVIDIA H200 NVL`;
- UUID: `GPU-000b6236-3632-a001-9667-1f02cbb61c8b`;
- PCI bus: `00000000:D1:00.0`;
- observed memory: `4 MiB / 143771 MiB`;
- observed utilization: `0%`.

GPU1 is an RTX PRO 5000 and remains forbidden. This check does not reserve GPU2
or guarantee later availability.

## Host resources

- shared evidence/data filesystem: `92 TiB` total, `41 TiB` available;
- H200 local filesystem: `3.5 TiB` total, `2.5 TiB` available;
- host RAM: `251 GiB`, `229 GiB` available at check time;
- system swap: `8.0 GiB`, `7.4 GiB` already used by unrelated host processes.

The formal gate therefore uses peak process-tree `VmSwap==0`, not global system
swap delta, so unrelated pre-existing swap cannot invalidate or rescue STDA.

## Decision

The pinned scientific runtime, allowed H200 GPU2, and storage capacity are
available. This is a package/resource check only; it is not an implementation,
capacity, geometry, or model result.
