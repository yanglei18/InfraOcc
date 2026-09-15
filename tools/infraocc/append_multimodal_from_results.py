import argparse
import json
import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mmdet
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core.visualizer.occ_visualization import (
    make_occ_color_map,
    render_camera_grid,
    render_occ_bev,
)
from mmdet3d.datasets import build_dataloader, build_dataset
from tools.infraocc.common import (
    flatten_results,
    get_split_cfg,
    load_config,
    load_occ_gt,
)
from tools.infraocc.generate_paper_figures import (
    CLASS_NAMES,
    EMPTY_IDX,
    first_batch_by_index,
    get_point_cloud_range,
    hconcat_resized,
    render_lidar_input_bev,
    save_image,
    unwrap_datacontainer,
)

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import compat_cfg
else:
    from mmdet3d.utils import compat_cfg

COLOR_MAP = make_occ_color_map(CLASS_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append freshly rerun multimodal visual assets.')
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('projects/InfraOcc/figures_update/rerun_occ/manifest.json'))
    parser.add_argument(
        '--results',
        type=Path,
        default=Path('projects/InfraOcc/figures_update/rerun_m_work/results.pkl'))
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(
            'projects/InfraOcc/configs/main_table_update/'
            'infraocc_m_4x4_36e_prosd.py'))
    parser.add_argument('--index', type=int, default=8)
    return parser.parse_args()


def build_sample_loader(config_path):
    cfg = compat_cfg(load_config(config_path))
    split_cfg = get_split_cfg(cfg, 'test')
    if isinstance(split_cfg, dict) and cfg.data.get('test_dataloader', {}).get(
            'samples_per_gpu', 1) > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
    dataset = build_dataset(split_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        runner_type='EpochBasedRunnerEval')
    return dataset, data_loader


def load_occ_prediction(results_path, index):
    results = flatten_results(mmcv.load(results_path))
    by_index = {int(item['index']): item for item in results}
    if index not in by_index:
        raise KeyError(f'Missing index {index} in {results_path}.')
    return np.asarray(by_index[index]['occ_results'], dtype=np.uint8)


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    figure = manifest['figures'][0]
    assets = figure.setdefault('assets', {})
    camera_asset = Path(assets['camera_grid'])
    stem = camera_asset.name.removesuffix('_camera_grid.png')
    assets_dir = camera_asset.parent

    occ = load_occ_prediction(args.results, args.index)
    dataset, data_loader = build_sample_loader(args.config)
    sample_meta = dataset.get_data_info(args.index)
    gt = load_occ_gt(dataset, args.index)
    point_cloud_range = get_point_cloud_range(args.config)

    try:
        _, data = next(
            first_batch_by_index(
                data_loader, indices=[args.index], max_samples=1))
    except StopIteration as exc:
        raise KeyError(f'Missing sample index {args.index}.') from exc

    img_inputs = unwrap_datacontainer(data['img_inputs'])
    cam_img = render_camera_grid(
        img_inputs,
        0,
        camera_num_frame=1,
        cam_names=sample_meta.get('cam_names'))
    if cam_img is None:
        cam_img = cv2.imread(str(camera_asset), cv2.IMREAD_COLOR)
    lidar_input = render_lidar_input_bev(
        data, point_cloud_range, gt_semantics=gt, color_map=COLOR_MAP)
    multimodal_input = hconcat_resized([cam_img, lidar_input], width=720)
    prosd_m_bev = render_occ_bev(occ, EMPTY_IDX, COLOR_MAP)

    multimodal_input_path = assets_dir / f'{stem}_multimodal_input.png'
    prosd_m_bev_path = assets_dir / f'{stem}_prosd_m_bev.png'
    save_image(multimodal_input_path, multimodal_input)
    save_image(prosd_m_bev_path, prosd_m_bev)

    assets['multimodal_input'] = str(multimodal_input_path)
    assets['prosd_m_bev'] = str(prosd_m_bev_path)
    args.manifest.write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Appended multimodal assets for index {args.index}.')


if __name__ == '__main__':
    main()
