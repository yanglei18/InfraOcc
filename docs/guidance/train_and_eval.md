# Training, Evaluation, and Visualization

## 1. Training

### Occ3D config

```bash
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_36e.py 4
```

### v2xreal config

```bash
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_v2xreal_36e.py 4
```

## 2. Evaluation

### Occ3D checkpoint test

```bash
bash tools/dist_test.sh \
  projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_36e.py \
  mmdet3d/ckpt/stcocc_r50_704x256_16f_occ3d.pth \
  4
```

### v2xreal checkpoint test

```bash
bash tools/dist_test.sh \
  projects/STCOcc/configs/stcocc_r50_704x256_16f_v2xreal_36e.py \
  /path/to/your_checkpoint.pth \
  4
```

## 3. Single-GPU debug run

Useful before long multi-GPU runs.

```bash
python tools/test.py \
  --config projects/STCOcc/configs/stcocc_r50_704x256_16f_v2xreal_36e.py \
  --checkpoint /path/to/your_checkpoint.pth \
  --launcher none
```

## 4. Result visualization

Use module mode (important):

```bash
python -m tools.visualizer.vis_results \
  --dataset-type occ3d \
  --vis-single-data /path/to/sample_result_or_label.npz
```

For scene-level visualization:

```bash
python -m tools.visualizer.vis_results \
  --pkl-file data/v2xreal_nuscenes/v2xreal_infos_val.pkl \
  --data-path data/v2xreal_nuscenes \
  --pred-path /path/to/pred_dir \
  --vis-scene "['scene-0001']" \
  --vis-path demo_out
```

## 5. Practical tips

- Multi-GPU progress count may exceed dataset length due to sampler padding; focus on final merged results and evaluation output.
- If you only want inference outputs, pass `--out work_dirs/results.pkl` and skip heavy visualization.
- Keep `work_dir` per experiment to avoid mixing logs/checkpoints from different configs.
