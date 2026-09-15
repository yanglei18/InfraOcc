# Environment Installation

This project is tested with Python 3.8 + CUDA 11.6 + PyTorch 1.13.

## 1. Create conda environment

```bash
conda create -n roadocc python=3.8 -y
conda activate roadocc
```

## 2. Install PyTorch

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 3. Install OpenMMLab dependencies

```bash
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
pip install mmengine
```

## 4. Install project dependencies and build ops

```bash
pip install -r docs/requirements/runtime.txt
pip install -r docs/requirements/tests.txt
pip install -v -e .
```

## 5. Optional packages for visualization / extra tools

```bash
pip install open3d==0.16.0 numpy==1.23.4 yapf==0.40.1 nuscenes-devkit
```

## 6. Sanity check

```bash
python -c "import mmcv, mmdet, mmseg, torch; print('ok')"
```

If CUDA ops build errors occur, ensure the CUDA toolkit and `nvcc` version match your installed PyTorch CUDA version.
