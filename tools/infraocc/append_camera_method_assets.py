import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.infraocc.generate_paper_figures import (
    COLOR_MAP,
    EMPTY_IDX,
    build_runner,
    render_occ_bev,
    run_predictions,
    save_image,
    setup_multi_processes,
)
from tools.infraocc.common import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append camera-method BEV assets to a paper-figure manifest.'
    )
    parser.add_argument(
        '--manifest-dir',
        type=Path,
        required=True,
        help='Directory containing manifest.json and an assets/ directory.')
    parser.add_argument(
        '--asset-name',
        required=True,
        help='Manifest asset key, e.g. sparseocc_bev or tpvformer_bev.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    return parser.parse_args()


def resolve_stem(figure, rank):
    camera_path = figure.get('assets', {}).get('camera_grid')
    if camera_path:
        name = Path(camera_path).name
        suffix = '_camera_grid.png'
        if name.endswith(suffix):
            return name[:-len(suffix)]
    token = str(figure.get('token', 'unknown'))[:8]
    return f"sample{rank:02d}_idx{int(figure['index']):04d}_{token}"


def main():
    args = parse_args()
    manifest_path = args.manifest_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    figures = manifest.get('figures', [])
    if not figures:
        raise ValueError(f'No figures listed in {manifest_path}.')

    setup_multi_processes(load_config(args.config))
    _, _, data_loader, model = build_runner(args.config, args.checkpoint,
                                            args.gpu_id, export_analysis=False)
    indices = [int(figure['index']) for figure in figures]
    results = run_predictions(
        model, data_loader, indices=indices, export_analysis=False)
    del model
    torch.cuda.empty_cache()

    assets_dir = args.manifest_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    for rank, figure in enumerate(figures):
        index = int(figure['index'])
        if index not in results:
            raise KeyError(f'Missing prediction for index {index}.')
        stem = resolve_stem(figure, rank)
        asset_path = assets_dir / f'{stem}_{args.asset_name}.png'
        save_image(asset_path,
                   render_occ_bev(results[index]['occ'], EMPTY_IDX,
                                  COLOR_MAP))
        figure.setdefault('assets', {})[args.asset_name] = str(asset_path)

    manifest.setdefault('camera_method_assets', {})[args.asset_name] = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        indices=indices,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2),
                             encoding='utf-8')
    print(f'Appended {args.asset_name} assets to {manifest_path}')


if __name__ == '__main__':
    main()
