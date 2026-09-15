# Environment Installation

This project is tested with Python 3.8 + CUDA 11.6 + PyTorch 1.13.

## 1. Create conda environment

```bash
conda create -n roadocc python=3.9 -y
conda activate roadocc
```

## 2. Install PyTorch

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 3. Install OpenMMLab dependencies

```bash
conda install -c conda-forge llvmlite numba
git clone https://github.com/open-mmlab/mmcv.git mmcv_src 
# A100=8.0, H100=9.0, pro6000=12.0, "8.0;9.0;12.0" = all three
# may need to adjust the C++ version in setup.py to match your system
export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
export MMCV_WITH_OPS=1
cd mmcv_src && git checkout v1.7.0
python setup.py build_ext --inplace && pip install -e . && cd ..
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
pip install mmengine
```

## 4. Install project dependencies and build ops

```bash
pip install -r docs/requirements/runtime.txt
pip install -r docs/requirements/tests.txt
pip install -e . --no-build-isolation
```

## 5. Optional packages for visualization / extra tools

```bash
pip install open3d==0.16.0 numpy==1.23.4 yapf==0.40.1 nuscenes-devkit spconv-cu124
```

## 6. Sanity check

```bash
python -c "import mmcv, mmdet, mmseg, torch; print('ok')"
```

If CUDA ops build errors occur, ensure the CUDA toolkit and `nvcc` version match your installed PyTorch CUDA version.
