import argparse
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / 'work_dirs' / 'infraocc_background_overfitting'
DEFAULT_TABLE_PATH = (
    ROOT
    / 'docs'
    / 'TPAMI2026'
    / 'local-git'
    / 'latex'
    / 'tab'
    / 'calibration_perturbation_robustness.tex'
)

MODEL_SPECS = OrderedDict([
    ('plain', {
        'label': 'Plain (C)',
        'config': ROOT / 'projects' / 'InfraOcc' / 'configs' / 'ablation' / 'core_chain' / 'infraocc_c_4x4_24e_plain.py',
        'checkpoint': ROOT / 'work_dirs' / 'infraocc_ablation' / 'core_chain' / 'infraocc_c_4x4_24e_plain' / 'latest.pth',
    }),
    ('c', {
        'label': 'ProSD-Occ (C)',
        'config': ROOT / 'projects' / 'InfraOcc' / 'configs' / 'main_table' / 'infraocc_c_4x4_36e_prosd.py',
        'checkpoint': ROOT / 'projects' / 'InfraOcc' / 'checkpoints' / 'c_prosd.pth',
    }),
    ('l', {
        'label': 'ProSD-Occ (L)',
        'config': ROOT / 'projects' / 'InfraOcc' / 'configs' / 'main_table' / 'infraocc_l_4x4_36e_prosd.py',
        'checkpoint': ROOT / 'projects' / 'InfraOcc' / 'checkpoints' / 'l_prosd.pth',
    }),
    ('m', {
        'label': 'ProSD-Occ (C+L)',
        'config': ROOT / 'projects' / 'InfraOcc' / 'configs' / 'main_table' / 'infraocc_m_4x4_36e_prosd.py',
        'checkpoint': ROOT / 'projects' / 'InfraOcc' / 'checkpoints' / 'm_prosd.pth',
    }),
])

DEFAULT_PYTHON = '/home/bxk/.conda/envs/roadocc/bin/python'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run static-background re-anchoring robustness evaluation for InfraOcc.')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODEL_SPECS.keys()),
        default=['c', 'l', 'm'],
        help='Models to evaluate.')
    parser.add_argument(
        '--blocks',
        nargs='+',
        choices=['translation', 'rotation'],
        default=['translation', 'rotation'],
        help='Perturbation blocks to evaluate.')
    parser.add_argument(
        '--translation-meters',
        nargs='+',
        type=float,
        default=[0.4, 0.8, 1.2],
        help='Translation magnitudes in meters.')
    parser.add_argument(
        '--rotation-degrees',
        nargs='+',
        type=float,
        default=[0.4, 0.8, 1.2],
        help='Yaw rotation magnitudes in degrees.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='Single GPU id used for sequential inference.')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs for inference. Values >1 use tools/dist_test.sh.')
    parser.add_argument(
        '--visible-gpus',
        type=str,
        default=None,
        help='Comma-separated physical GPU ids exposed to inference, e.g. 4,5,6,7.')
    parser.add_argument(
        '--python',
        type=str,
        default=DEFAULT_PYTHON if Path(DEFAULT_PYTHON).exists() else sys.executable,
        help='Python interpreter used for test/eval subprocesses.')
    parser.add_argument(
        '--checkpoint-overrides',
        nargs='*',
        default=[],
        help='Optional model=checkpoint overrides, e.g. plain=/abs/path/epoch_24.pth')
    parser.add_argument(
        '--config-overrides',
        nargs='*',
        default=[],
        help='Optional model=config overrides, e.g. c=/abs/path/infraocc_c.py')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='Directory for intermediate results and metrics.')
    parser.add_argument(
        '--table-path',
        type=Path,
        default=DEFAULT_TABLE_PATH,
        help='LaTeX table to update.')
    parser.add_argument(
        '--overwrite-results',
        action='store_true',
        help='Re-run inference even if results.pkl already exists.')
    parser.add_argument(
        '--overwrite-metrics',
        action='store_true',
        help='Re-run evaluation even if metrics JSON already exists.')
    parser.add_argument(
        '--skip-table-update',
        action='store_true',
        help='Do not update the LaTeX table after evaluation.')
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='Skip clean evaluation and only run perturbations. Useful for smoke tests.')
    parser.add_argument(
        '--eval-in-canonical-frame',
        action='store_true',
        help=(
            'For LiDAR-based models, canonicalize inputs during re-anchored '
            'evaluation and warp predictions back. Disabled by default for the '
            'static-background overfitting stress test.'))
    return parser.parse_args()


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def parse_path_overrides(entries, flag_name, expected_kind):
    overrides = {}
    for entry in entries:
        if '=' not in entry:
            raise ValueError(
                f'Invalid {flag_name} entry {entry!r}. Expected model=/path/to.{expected_kind}')
        model_key, value = entry.split('=', 1)
        model_key = model_key.strip()
        if model_key not in MODEL_SPECS:
            raise KeyError(
                f'Unknown model {model_key!r} in {flag_name}. '
                f'Choices: {list(MODEL_SPECS.keys())}')
        overrides[model_key] = Path(value).expanduser().resolve()
    return overrides


def resolve_config(args, model_key):
    if model_key in args.config_overrides:
        return args.config_overrides[model_key]
    return MODEL_SPECS[model_key]['config']


def resolve_checkpoint(args, model_key):
    if model_key in args.checkpoint_overrides:
        return args.checkpoint_overrides[model_key]
    return MODEL_SPECS[model_key]['checkpoint']


def run_command(cmd, env=None):
    print('Running:', ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def format_value(value):
    return f'{float(value):.2f}'


def format_pair(metrics):
    return f"{format_value(metrics['mean_static'])}/{format_value(metrics['mean_dynamic'])}"


def format_dynamic(metrics):
    return format_value(metrics['mean_dynamic'])


def make_retention(clean_metrics, perturbed_metrics):
    static_ret = float(perturbed_metrics['mean_static']) / max(float(clean_metrics['mean_static']), 1e-6) * 100.0
    dynamic_ret = float(perturbed_metrics['mean_dynamic']) / max(float(clean_metrics['mean_dynamic']), 1e-6) * 100.0
    return round((static_ret + dynamic_ret) / 2.0, 2)


def make_config_text(base_config: Path, perturbation: dict,
                     eval_in_canonical_frame: bool) -> str:
    perturbation_repr = json.dumps(perturbation, indent=4)
    return f"""# Auto-generated by tools/infraocc/run_background_overfitting.py
from copy import deepcopy

from mmcv import Config

_base_ = [r'{base_config.as_posix()}']
_base_cfg = Config.fromfile(r'{base_config.as_posix()}')

_base_test_pipeline = getattr(_base_cfg, 'test_pipeline', None)
if _base_test_pipeline is None:
    _base_test_pipeline = _base_cfg.data.test.pipeline
test_pipeline = deepcopy(_base_test_pipeline)
test_pipeline = [
    step for step in test_pipeline
    if not (isinstance(step, dict)
            and step.get('type') == 'InfraOccEgoFrameReanchoring')
]

_perturbation = {perturbation_repr}
test_pipeline.insert(0, dict(
    type='InfraOccEgoFrameReanchoring',
    translation=tuple(_perturbation['translation']),
    roll_deg=float(_perturbation['roll_deg']),
    pitch_deg=float(_perturbation['pitch_deg']),
    yaw_deg=float(_perturbation['yaw_deg']),
))

reanchor_cfg = dict(
    translation=tuple(_perturbation['translation']),
    roll_deg=float(_perturbation['roll_deg']),
    pitch_deg=float(_perturbation['pitch_deg']),
    yaw_deg=float(_perturbation['yaw_deg']),
)

model = deepcopy(_base_cfg.model)
model.setdefault('modality_config', dict())
model['modality_config']['eval_reanchor_in_canonical_frame'] = {eval_in_canonical_frame!r}

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
evaluation = dict(interval=2, pipeline=test_pipeline)
"""


def write_generated_config(output_path: Path, base_config: Path,
                           perturbation: dict, eval_in_canonical_frame: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        make_config_text(base_config, perturbation, eval_in_canonical_frame),
        encoding='utf-8')


def build_model_runs(args, model_key):
    spec = MODEL_SPECS[model_key]
    runs = []
    base_config = resolve_config(args, model_key)
    checkpoint = resolve_checkpoint(args, model_key)
    ensure_exists(base_config)

    if not args.skip_clean:
        runs.append(('clean', dict(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, translation=(0.0, 0.0, 0.0))))

    if 'translation' in args.blocks:
        for meters in args.translation_meters:
            runs.append((
                f'translation_{meters:g}m',
                dict(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, translation=(float(meters), 0.0, 0.0)),
            ))

    if 'rotation' in args.blocks:
        for degrees in args.rotation_degrees:
            runs.append((
                f'rotation_{degrees:g}deg',
                dict(roll_deg=0.0, pitch_deg=0.0, yaw_deg=float(degrees), translation=(0.0, 0.0, 0.0)),
            ))

    return spec, base_config, checkpoint, runs


def evaluate_one(args, model_key, setting_name, perturbation, base_config, checkpoint):
    spec = MODEL_SPECS[model_key]
    model_root = args.output_root / model_key / setting_name
    results_path = model_root / 'results.pkl'
    metrics_path = model_root / 'metrics.json'
    config_path = model_root / f'{model_key}_{setting_name}.py'

    write_generated_config(
        config_path,
        base_config,
        perturbation,
        eval_in_canonical_frame=args.eval_in_canonical_frame)

    metrics_ready = metrics_path.exists() and not args.overwrite_metrics
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = (
        args.visible_gpus if args.visible_gpus is not None else
        (','.join(str(index) for index in range(int(args.num_gpus)))
         if int(args.num_gpus) > 1 else str(args.gpu_id)))
    env['PYTHONPATH'] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(':')

    if not metrics_ready and (args.overwrite_results or not results_path.exists()):
        if args.overwrite_results and results_path.exists():
            results_path.unlink()
        # tools/test.py only exports predictions. Table metrics are computed by
        # eval_occ_tables.py below because it applies the strict re-anchor GT warp.
        if int(args.num_gpus) > 1:
            test_cmd = [
                args.python,
                '-m',
                'torch.distributed.launch',
                '--nproc_per_node',
                str(int(args.num_gpus)),
                '--master_port',
                str(29500 + os.getpid() % 500),
                str(ROOT / 'tools' / 'test.py'),
                '--config', str(config_path),
                '--checkpoint', str(checkpoint),
                '--launcher', 'pytorch',
                '--out', str(results_path),
                '--work-dir', str(model_root / 'test_workdir'),
                '--cfg-options',
                'data.test_dataloader.samples_per_gpu=1',
                'data.test_dataloader.workers_per_gpu=2',
                'data.workers_per_gpu=2',
            ]
        else:
            test_cmd = [
                args.python,
                str(ROOT / 'tools' / 'test.py'),
                '--config', str(config_path),
                '--checkpoint', str(checkpoint),
                '--out', str(results_path),
                '--work-dir', str(model_root / 'test_workdir'),
                '--cfg-options',
                'data.test_dataloader.samples_per_gpu=1',
                'data.test_dataloader.workers_per_gpu=2',
                'data.workers_per_gpu=2',
            ]
        run_command(test_cmd, env=env)

    if args.overwrite_metrics or not metrics_path.exists():
        ensure_exists(results_path)
        if args.overwrite_metrics and metrics_path.exists():
            metrics_path.unlink()
        eval_cmd = [
            args.python,
            str(ROOT / 'tools' / 'infraocc' / 'eval_occ_tables.py'),
            '--config', str(config_path),
            '--results', str(results_path),
            '--split', 'test',
            '--output-json', str(metrics_path),
        ]
        run_command(eval_cmd, env=env)

    ensure_exists(metrics_path)
    with metrics_path.open('r', encoding='utf-8') as handle:
        metrics = json.load(handle)

    payload = dict(
        model=spec['label'],
        model_key=model_key,
        setting=setting_name,
        perturbation=perturbation,
        checkpoint=str(checkpoint),
        config=str(base_config),
        results=str(results_path),
        metrics_path=str(metrics_path),
        metrics=metrics,
    )
    return payload


def load_existing_clean_entry(output_root: Path, model_key: str):
    metrics_path = output_root / model_key / 'clean' / 'metrics.json'
    results_path = output_root / model_key / 'clean' / 'results.pkl'
    ensure_exists(metrics_path)
    with metrics_path.open('r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    return metrics, dict(
        pair=format_pair(metrics),
        gIoU=metrics['gIoU'],
        mIoU=metrics['mIoU'],
        mean_static=metrics['mean_static'],
        mean_dynamic=metrics['mean_dynamic'],
        results_path=str(results_path),
        metrics_path=str(metrics_path),
    )


def update_table(table_path: Path, summary: dict):
    text = table_path.read_text(encoding='utf-8')

    block_rows = [
        ('Translation re-anchoring', ['Clean'] + [
            f'{float(setting[:-1]):g}\\,m'
            for setting in next(iter(summary['translation']['models'].values()))['perturbations']
        ], 'translation'),
        ('Rotation re-anchoring', ['Clean'] + [
            f'${float(setting[:-3]):g}^\\circ$'
            for setting in next(iter(summary['rotation']['models'].values()))['perturbations']
        ], 'rotation'),
    ]

    for block_title, headers, block_key in block_rows:
        header_replacement = '& ' + ' & '.join(headers) + r' \\'
        block_summary = summary[block_key]
        block_header = rf"\\multicolumn\{{[56]\}}\{{l\}}\{{\\textbf\{{\\textit\{{{re.escape(block_title)}\}}\}}\}} \\\\"
        block_match = re.search(block_header, text)
        if block_match is None:
            raise RuntimeError(f'Block {block_title!r} not found in {table_path}')
        header_matches = list(
            re.finditer(r'^& Clean & .*? \\\\$',
                        text[:block_match.start()],
                        flags=re.MULTILINE))
        if not header_matches:
            raise RuntimeError(f'Failed to find header before block {block_title!r}.')
        header_match = header_matches[-1]
        text = text[:header_match.start()] + header_replacement + text[
            header_match.end():]
        block_match = re.search(block_header, text)
        block_start = block_match.end()
        next_boundary = re.search(r'(?m)^\\midrule$|^\\bottomrule$', text[block_start:])
        block_end = block_start + next_boundary.start() if next_boundary else len(text)
        block_text = text[block_start:block_end]

        for model_key, model_entry in block_summary['models'].items():
            row_name = MODEL_SPECS[model_key]['label']
            row_values = [format_dynamic(model_entry['clean'])]
            for setting in model_entry['perturbations']:
                row_values.append(
                    format_dynamic(model_entry['perturbations'][setting]))
            replacement = f"{row_name} & " + ' & '.join(row_values) + r" \\"
            pattern = rf"^{re.escape(row_name)}\s*&.*?(?:\\\\)?$"
            block_text, count = re.subn(
                pattern,
                lambda _: replacement,
                block_text,
                flags=re.MULTILINE)
            if count != 1:
                raise RuntimeError(
                    f'Failed to update exactly one row for {row_name!r} in block {block_title!r}.')
        text = text[:block_start] + block_text + text[block_end:]

    table_path.write_text(text, encoding='utf-8')


def main():
    args = parse_args()
    args.checkpoint_overrides = parse_path_overrides(
        args.checkpoint_overrides, '--checkpoint-overrides', 'ckpt')
    args.config_overrides = parse_path_overrides(
        args.config_overrides, '--config-overrides', 'py')
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = OrderedDict()
    for block in args.blocks:
        summary[block] = OrderedDict(models=OrderedDict())

    for model_key in args.models:
        spec, base_config, checkpoint, runs = build_model_runs(args, model_key)
        if not runs:
            continue
        if not checkpoint.exists():
            summary.setdefault('skipped', OrderedDict())[model_key] = dict(
                model=spec['label'],
                checkpoint=str(checkpoint),
                reason='checkpoint_missing',
            )
            print(
                f'Skipping {spec["label"]} because checkpoint is missing: {checkpoint}',
                file=sys.stderr)
            continue

        model_summary = OrderedDict()
        clean_metrics = None
        perturbation_metrics = OrderedDict()

        if args.skip_clean:
            clean_metrics, clean_entry = load_existing_clean_entry(args.output_root, model_key)
            model_summary['clean'] = clean_entry

        for setting_name, perturbation in runs:
            payload = evaluate_one(args, model_key, setting_name, perturbation, base_config, checkpoint)
            metrics = payload['metrics']
            pair = format_pair(metrics)
            entry = dict(
                pair=pair,
                gIoU=metrics['gIoU'],
                mIoU=metrics['mIoU'],
                mean_static=metrics['mean_static'],
                mean_dynamic=metrics['mean_dynamic'],
                results_path=payload['results'],
                metrics_path=payload['metrics_path'],
            )
            if setting_name == 'clean':
                clean_metrics = metrics
                model_summary['clean'] = entry
            else:
                perturbation_metrics[setting_name] = entry

        if clean_metrics is None:
            raise RuntimeError(f'Clean run was skipped for {model_key!r}; cannot compute retention.')

        if 'translation' in args.blocks:
            translation_entries = OrderedDict()
            ret_values = []
            for meters in args.translation_meters:
                setting_name = f'translation_{meters:g}m'
                entry = perturbation_metrics[setting_name]
                translation_entries[f'{meters:g}m'] = entry
                ret_values.append(make_retention(clean_metrics, {
                    'mean_static': entry['mean_static'],
                    'mean_dynamic': entry['mean_dynamic'],
                }))
            model_summary['translation'] = dict(
                clean=model_summary['clean'],
                perturbations=translation_entries,
                avg_ret=round(sum(ret_values) / len(ret_values), 2) if ret_values else None,
            )

        if 'rotation' in args.blocks:
            rotation_entries = OrderedDict()
            ret_values = []
            for degrees in args.rotation_degrees:
                setting_name = f'rotation_{degrees:g}deg'
                entry = perturbation_metrics[setting_name]
                rotation_entries[f'{degrees:g}deg'] = entry
                ret_values.append(make_retention(clean_metrics, {
                    'mean_static': entry['mean_static'],
                    'mean_dynamic': entry['mean_dynamic'],
                }))
            model_summary['rotation'] = dict(
                clean=model_summary['clean'],
                perturbations=rotation_entries,
                avg_ret=round(sum(ret_values) / len(ret_values), 2) if ret_values else None,
            )

        for block in args.blocks:
            summary[block]['models'][model_key] = model_summary[block]

    summary_path = args.output_root / 'background_overfitting_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

    if not args.skip_table_update:
        update_table(args.table_path, summary)

    print(json.dumps({
        'summary_json': str(summary_path),
        'table_path': str(args.table_path),
        'models': args.models,
        'blocks': args.blocks,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
