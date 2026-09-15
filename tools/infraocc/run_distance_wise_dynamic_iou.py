import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / 'work_dirs' / 'infraocc_table9'
DEFAULT_TABLE_PATH = (
    ROOT
    / 'docs'
    / 'TPAMI2026'
    / 'InfraOcc - TPAMI'
    / 'latex'
    / 'tab'
    / 'distance_wise_dynamic_iou.tex'
)

MODALITY_SPECS = OrderedDict([
    ('c', {
        'label': 'ProSD-Occ (C)',
        'config': ROOT / 'work_dirs' / 'infraocc_main_table' / 'c_prosd_occ_4x4_36e' / 'infraocc_c_4x4_36e_prosd.py',
        'checkpoint': ROOT / 'projects' / 'InfraOcc' / 'checkpoints' / 'c_prosd.pth',
        'expected': {'gIoU': 87.60, 'mIoU': 58.94, 'mean_dynamic': 27.71, 'mean_static': 82.36},
    }),
    ('l', {
        'label': 'ProSD-Occ (L)',
        'config': ROOT / 'work_dirs' / 'infraocc_main_table' / 'l_prosd_occ_4x4_36e' / 'infraocc_l_4x4_36e_prosd.py',
        'checkpoint': ROOT / 'work_dirs' / 'infraocc_main_table' / 'l_prosd_occ_4x4_36e' / 'epoch_36.pth',
        'expected': {'gIoU': 85.18, 'mIoU': 62.12, 'mean_dynamic': 38.84, 'mean_static': 79.58},
    }),
    ('m', {
        'label': 'ProSD-Occ (C+L)',
        'config': ROOT / 'work_dirs' / 'infraocc_main_table' / 'm_prosd_occ_4x4_36e' / 'infraocc_m_4x4_36e_prosd.py',
        'checkpoint': ROOT / 'projects' / 'InfraOcc' / 'checkpoints' / 'm_prosd.pth',
        'expected': {'gIoU': 89.90, 'mIoU': 64.62, 'mean_dynamic': 36.99, 'mean_static': 85.34},
    }),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run InfraOcc Table 9 distance-wise dynamic mIoU evaluation and fill the LaTeX table.')
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['c', 'l', 'm'],
        choices=list(MODALITY_SPECS.keys()),
        help='Modalities to evaluate.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='Single GPU id used for sequential inference.')
    parser.add_argument(
        '--python',
        type=str,
        default=sys.executable,
        help='Python interpreter used for test/eval subprocesses.')
    parser.add_argument(
        '--distance-bins',
        nargs='+',
        type=float,
        default=[0, 20, 40, 60, 80],
        help='Distance bin edges in meters.')
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
        help='Re-run distance-bin evaluation even if metrics JSON already exists.')
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip model inference and only evaluate existing results.pkl files.')
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help='Allowed absolute deviation from the main-table aggregate metrics.')
    parser.add_argument(
        '--skip-aggregate-check',
        action='store_true',
        help='Skip strict aggregate-metric validation against the main table.')
    return parser.parse_args()


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def run_command(cmd, env=None):
    print('Running:', ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def build_eval_row(metrics, distance_bins):
    names = [
        f'{distance_bins[index]:g}-{distance_bins[index + 1]:g}m'
        for index in range(len(distance_bins) - 1)
    ]
    values = []
    for name in names:
        bin_metrics = metrics['distance_bins'].get(name, {})
        value = bin_metrics.get('mean_dynamic')
        values.append('' if value is None else f'{float(value):.2f}')
    return values


def update_table(table_path: Path, rows):
    text = table_path.read_text(encoding='utf-8')
    for label, values in rows.items():
        replacement = f"{label} & " + ' & '.join(values) + r" \\"
        import re
        pattern = rf"^{re.escape(label)}\s*&.*?\\\\$"
        new_text, count = re.subn(
            pattern,
            lambda _: replacement,
            text,
            flags=re.MULTILINE)
        if count == 0:
            pattern = rf"^{re.escape(label)}\s*&.*?$"
            new_text, count = re.subn(
                pattern,
                lambda _: replacement,
                text,
                flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f'Failed to update exactly one row for {label!r} in {table_path}.')
        text = new_text
    table_path.write_text(text, encoding='utf-8')


def compare_expected(modality, metrics, tolerance):
    expected = MODALITY_SPECS[modality]['expected']
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = float(metrics[key])
        if abs(actual_value - expected_value) > tolerance:
            mismatches.append((key, expected_value, actual_value))
    return mismatches


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    table_rows = OrderedDict()
    summary = OrderedDict()

    if len(args.distance_bins) < 2:
        raise ValueError('Need at least two distance-bin edges.')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    env['PYTHONPATH'] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(':')

    for modality in args.modalities:
        spec = MODALITY_SPECS[modality]
        config_path = Path(spec['config'])
        checkpoint_path = Path(spec['checkpoint'])
        ensure_exists(config_path)
        ensure_exists(checkpoint_path)

        modality_root = args.output_root / modality
        modality_root.mkdir(parents=True, exist_ok=True)
        results_path = modality_root / 'results.pkl'
        metrics_path = modality_root / 'distance_metrics.json'

        if not args.skip_test and (args.overwrite_results or not results_path.exists()):
            test_cmd = [
                args.python,
                str(ROOT / 'tools' / 'test.py'),
                '--config', str(config_path),
                '--checkpoint', str(checkpoint_path),
                '--out', str(results_path),
                '--work-dir', str(modality_root / 'test_workdir'),
                '--eval', 'miou',
                '--cfg-options',
                'data.test_dataloader.samples_per_gpu=1',
                'data.test_dataloader.workers_per_gpu=2',
                'data.workers_per_gpu=2',
            ]
            run_command(test_cmd, env=env)

        ensure_exists(results_path)

        if args.overwrite_metrics or not metrics_path.exists():
            eval_cmd = [
                args.python,
                str(ROOT / 'tools' / 'infraocc' / 'eval_occ_tables.py'),
                '--config', str(config_path),
                '--results', str(results_path),
                '--split', 'test',
                '--output-json', str(metrics_path),
                '--distance-bins',
                *[str(edge) for edge in args.distance_bins],
            ]
            run_command(eval_cmd, env=env)

        ensure_exists(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        if not args.skip_aggregate_check:
            mismatches = compare_expected(modality, metrics, args.tolerance)
            if mismatches:
                mismatch_text = ', '.join(
                    f'{key}: expected {expected:.2f}, got {actual:.2f}'
                    for key, expected, actual in mismatches)
                raise RuntimeError(
                    f'Aggregate metrics for modality {modality} do not match the main table within '
                    f'tolerance {args.tolerance:.2f}: {mismatch_text}')

        row_values = build_eval_row(metrics, args.distance_bins)
        table_rows[spec['label']] = row_values
        summary[spec['label']] = {
            'distance_bins': metrics['distance_bins'],
            'aggregate': {
                'gIoU': metrics['gIoU'],
                'mIoU': metrics['mIoU'],
                'mean_dynamic': metrics['mean_dynamic'],
                'mean_static': metrics['mean_static'],
            },
        }

    update_table(args.table_path, table_rows)
    summary_path = args.output_root / 'table9_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

    print('\nTable 9 rows:')
    for label, values in table_rows.items():
        print(f'{label}: ' + ' | '.join(values))
    print(f'\nUpdated LaTeX table: {args.table_path}')
    print(f'Summary JSON: {summary_path}')


if __name__ == '__main__':
    main()
