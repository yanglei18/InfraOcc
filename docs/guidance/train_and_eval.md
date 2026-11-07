# ProSD-Occ Training and Evaluation

All released configurations evaluate on the InfraOcc validation split.

## Training

```bash
bash projects/ProSDOcc/configs/dist_train.sh \
  projects/ProSDOcc/configs/infraocc_c_4x4_36e_prosd.py 4 \
  --work-dir work_dirs/prosd_occ/c_prosd_occ_4x4_36e
```

Replace `c` with `l` or `m` for the LiDAR-only or camera-LiDAR model.

## Single-GPU evaluation

The single-GPU command is the simplest reproducibility check and needs about
8 GB of GPU memory for the camera-only checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --config projects/ProSDOcc/configs/infraocc_c_4x4_36e_prosd.py \
  --checkpoint projects/ProSDOcc/ckpt_final/c_prosd.pth \
  --launcher none \
  --compact-out \
  --work-dir work_dirs/prosd_occ/c_prosd_eval \
  --eval miou
```

## Distributed evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
  projects/ProSDOcc/configs/infraocc_c_4x4_36e_prosd.py \
  projects/ProSDOcc/ckpt_final/c_prosd.pth \
  4 \
  --compact-out \
  --work-dir work_dirs/prosd_occ/c_prosd_eval \
  --eval miou
```

Use the matching `l_prosd.pth` or `m_prosd.pth` checkpoint and configuration
for the other sensing tracks. `--compact-out` retains semantic predictions but
drops large diagnostic payloads from the serialized result.

## Reference results

| Track | mIoU | Geometric IoU | Dynamic mIoU | Static mIoU |
| --- | ---: | ---: | ---: | ---: |
| Camera-only | 60.08 | 89.08 | 28.97 | 83.42 |
| LiDAR-only | 63.66 | 90.86 | 33.87 | 85.99 |
| Camera + LiDAR | 65.87 | 91.78 | 37.92 | 86.82 |

The metrics use all 1,970 validation samples and ignore
`construction_vehicle`, `trailer`, and `other_flat` in semantic mIoU.
