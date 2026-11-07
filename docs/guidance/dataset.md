# InfraOcc Dataset Preparation

Place the InfraOcc benchmark in `data/infraocc/` using the nuScenes-style
layout below.

```text
data/infraocc/
├── samples/
├── sweeps/                  # optional
├── v1.0-trainval/
├── gts/                     # labels.npz for every sample
├── infraocc_infos_train.pkl
└── infraocc_infos_val.pkl
```

## Generate metadata

```bash
python tools/create_info_pkl.py \
  --root-path data/infraocc \
  --extra-tag infraocc \
  --version v1.0-trainval \
  --max-sweeps 0
```

## Generate multi-scale occupancy labels

```bash
python tools/create_ms_occ.py \
  --dataset infraocc \
  --pkl_path data/infraocc/infraocc_infos_train.pkl

python tools/create_ms_occ.py \
  --dataset infraocc \
  --pkl_path data/infraocc/infraocc_infos_val.pkl
```

Each sample receives `labels_1_2.npz`, `labels_1_4.npz`, and
`labels_1_8.npz` beside its `labels.npz`. Add `--overwrite` only when these
derived labels must be regenerated.

`tools/create_info_pkl.py` requires `nuscenes-devkit`. Keep camera keys
consistent with `data_config['cams']` in the released configurations.
