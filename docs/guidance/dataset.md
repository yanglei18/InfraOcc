# Dataset Preparation

This repo uses v2xreal data organized in nuScenes format under
`data/v2xreal_nuscenes`.

## 1. Required folder structure

```text
data/
└── v2xreal_nuscenes/
    ├── samples/
    ├── sweeps/              # optional
    ├── v1.0-trainval/
    └── gts/                 # occupancy labels, each sample has labels.npz
```

## 2. Generate info pkl files

Script: `tools/create_info_pkl.py`

```bash
python tools/create_info_pkl.py \
  --root-path data/v2xreal_nuscenes \
  --extra-tag v2xreal \
  --version v1.0-trainval \
  --max-sweeps 0
```

Outputs:

- `data/v2xreal_nuscenes/v2xreal_infos_train.pkl`
- `data/v2xreal_nuscenes/v2xreal_infos_val.pkl`

## 3. Generate multi-scale occupancy GT

Script: `tools/create_ms_occ.py`

```bash
python tools/create_ms_occ.py \
  --dataset v2xreal \
  --pkl_path data/v2xreal_nuscenes/v2xreal_infos_train.pkl

python tools/create_ms_occ.py \
  --dataset v2xreal \
  --pkl_path data/v2xreal_nuscenes/v2xreal_infos_val.pkl
```

This generates the following files under each sample `occ_path` directory:

- `labels_1_2.npz`
- `labels_1_4.npz`
- `labels_1_8.npz`

Use `--overwrite` if existing multi-scale files need to be regenerated.

## 4. Notes

- `tools/create_info_pkl.py` needs `nuscenes-devkit` installed.
- Keep camera key names consistent with config `data_config['cams']`.
- Keep machine-specific dataset paths in local config overrides.
