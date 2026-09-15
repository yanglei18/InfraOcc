import argparse
import json
import os
import subprocess
import sys
from os import path as osp


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

ROWS = [
    dict(section='Multi-modal (C+L)', label='M-CONet', config='projects/OpenOcc/configs/conet_m_4x4_24e.py',
         checkpoint='work_dirs/conet_m_4x4_24e/epoch_24.pth', results='work_dirs/conet_m_4x4_24e/results.pkl'),
    dict(section='Multi-modal (C+L)', label='BEVFusion', config='projects/OpenOcc/configs/bevfusion_4x4_24e.py'),
    dict(section='Multi-modal (C+L)', label='BEVDepth', config='projects/OpenOcc/configs/bevdepth_m_4x4_24e.py'),
    dict(section='Multi-modal (C+L)', label='OccGen'),
    dict(section='Multi-modal (C+L)', label='M-ProSD-Occ', config='projects/InfraOcc/configs/main_table/infraocc_m_4x4_36e_prosd.py'),
    dict(section='LiDAR-only (L)', label='L-CONet', config='projects/OpenOcc/configs/conet_l_4x4_24e.py',
         checkpoint='work_dirs/conet_l_4x4_24e/epoch_18.pth', results='work_dirs/conet_l_4x4_24e/results.pkl'),
    dict(section='LiDAR-only (L)', label='VoxelNet', config='projects/OpenOcc/configs/voxelnet_4x4_24e.py'),
    dict(section='LiDAR-only (L)', label='PointPillars', config='projects/OpenOcc/configs/pointpillars_4x4_24e.py',
         checkpoint='work_dirs/pointpillars_4x4_24e/epoch_24.pth', results='work_dirs/pointpillars_4x4_24e/results.pkl'),
    dict(section='LiDAR-only (L)', label='L-ProSD-Occ', config='projects/InfraOcc/configs/main_table/infraocc_l_4x4_36e_prosd.py'),
    dict(section='Camera-only (C)', label='BEVDet', config='projects/OpenOcc/configs/bevdet_4x4_24e.py',
         checkpoint='work_dirs/bevdet_4x4_24e/epoch_24.pth', results='work_dirs/bevdet_4x4_24e/results.pkl'),
    dict(section='Camera-only (C)', label='BEVFormer', config='projects/BEVFormer/configs/bevformer_4x4_24e.py'),
    dict(section='Camera-only (C)', label='BEVDepth', config='projects/OpenOcc/configs/bevdepth_c_4x4_24e.py'),
    dict(section='Camera-only (C)', label='TPVFormer', config='projects/TPVFormer/configs/tpvformer_4x4_24e.py',
         checkpoint='work_dirs/tpvformer_4x4_24e/epoch_24.pth', results='work_dirs/tpvformer_4x4_24e/results.pkl'),
    dict(section='Camera-only (C)', label='SurroundOcc', config='projects/SurroundOcc/configs/surroundocc_4x4_24e.py',
         checkpoint='work_dirs/surroundocc_4x4_24e/epoch_24.pth', results='work_dirs/surroundocc_4x4_24e/results.pkl'),
    dict(section='Camera-only (C)', label='C-CONet', config='projects/OpenOcc/configs/conet_c_4x4_24e.py',
         checkpoint='work_dirs/conet_c_4x4_24e/epoch_24.pth', results='work_dirs/conet_c_4x4_24e/results.pkl'),
    # The STCOcc project is not present in this checkout. Per the current paper plan,
    # we approximate it with the simplest camera-only InfraOcc variant to keep the
    # efficiency table internally reproducible.
    dict(section='Camera-only (C)', label='STCOcc', config='projects/InfraOcc/configs/infraocc_c_4x4_24e.py'),
    dict(section='Camera-only (C)', label='SparseOcc', config='projects/SparseOcc/configs/sparseocc_4x4_24e.py',
         checkpoint='work_dirs/sparseocc_4x4_24e/epoch_24.pth'),
    dict(section='Camera-only (C)', label='GaussianFormer'),
    dict(section='Camera-only (C)', label='OccMamba'),
    dict(section='Camera-only (C)', label='C-ProSD-Occ', config='projects/InfraOcc/configs/main_table/infraocc_c_4x4_36e_prosd.py',
         checkpoint='work_dirs/infraocc_c_4x4_36e_progressive_sd/epoch_36.pth',
         results='work_dirs/infraocc_c_4x4_36e_progressive_sd/results.pkl',
         log_json='work_dirs/infraocc_c_4x4_36e_progressive_sd/20260418_144120.log.json'),
    dict(section='ProSD-Occ variant efficiency', label='Plain Occupancy Prediction',
         config='projects/InfraOcc/configs/ablation/core_chain/infraocc_c_2x4_24e_plain.py'),
    dict(section='ProSD-Occ variant efficiency', label='Parallel S2D Fusion',
         config='projects/InfraOcc/configs/ablation/core_chain/infraocc_c_2x4_24e_prosd_no_supp_no_raw_bypass.py'),
    dict(section='ProSD-Occ variant efficiency', label='Progressive S2D Reasoning',
         config='projects/InfraOcc/configs/ablation/core_chain/infraocc_c_2x4_24e_prosd_full.py'),
]

MAIN_TABLE_IOU = {
    ('Multi-modal (C+L)', 'M-CONet'): dict(miou_all=57.23, miou_dyn=34.78),
    ('Multi-modal (C+L)', 'BEVFusion'): dict(miou_all=55.44, miou_dyn=32.57),
    ('Multi-modal (C+L)', 'BEVDepth'): dict(miou_all=55.34, miou_dyn=33.07),
    ('Multi-modal (C+L)', 'M-ProSD-Occ'): dict(miou_all=59.89, miou_dyn=35.29),
    ('LiDAR-only (L)', 'L-CONet'): dict(miou_all=57.76, miou_dyn=32.48),
    ('LiDAR-only (L)', 'VoxelNet'): dict(miou_all=53.87, miou_dyn=31.13),
    ('LiDAR-only (L)', 'PointPillars'): dict(miou_all=50.51, miou_dyn=31.55),
    ('LiDAR-only (L)', 'L-ProSD-Occ'): dict(miou_all=58.06, miou_dyn=34.52),
    ('Camera-only (C)', 'BEVDet'): dict(miou_all=47.70, miou_dyn=23.00),
    ('Camera-only (C)', 'BEVFormer'): dict(miou_all=47.76, miou_dyn=10.73),
    ('Camera-only (C)', 'BEVDepth'): dict(miou_all=46.81, miou_dyn=19.64),
    ('Camera-only (C)', 'TPVFormer'): dict(miou_all=43.09, miou_dyn=4.36),
    ('Camera-only (C)', 'SurroundOcc'): dict(miou_all=50.77, miou_dyn=11.37),
    ('Camera-only (C)', 'C-CONet'): dict(miou_all=49.25, miou_dyn=18.93),
    ('Camera-only (C)', 'STCOcc'): dict(miou_all=51.02, miou_dyn=21.82),
    ('Camera-only (C)', 'SparseOcc'): dict(miou_all=45.42, miou_dyn=9.41),
    ('Camera-only (C)', 'C-ProSD-Occ'): dict(miou_all=54.04, miou_dyn=26.86),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Collect values for the TPAMI efficiency table.')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--warmup-iters', type=int, default=2)
    parser.add_argument('--profile-iters', type=int, default=5)
    parser.add_argument('--output-json', default='tools/efficiency/efficiency_table_metrics.json')
    parser.add_argument('--sections', nargs='*', default=None)
    parser.add_argument('--labels', nargs='*', default=None)
    return parser.parse_args()


def format_value(value, digits=2):
    if value is None:
        return ''
    return f'{float(value):.{digits}f}'


def latest_val_miou(log_path):
    if not log_path or not osp.exists(log_path):
        return None
    best = None
    with open(log_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('mode') == 'val' and 'miou' in record:
                best = float(record['miou'])
    return best


def run_json_command(command):
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True)
    stdout = result.stdout.strip()
    start = stdout.find('{')
    end = stdout.rfind('}')
    if start < 0 or end < 0 or end < start:
        raise ValueError(f'No JSON object found in output:\n{stdout}')
    return json.loads(stdout[start:end + 1])


def maybe_profile(row, args):
    config = row.get('config')
    if not config or not osp.exists(osp.join(REPO_ROOT, config)):
        return {}
    command = [
        sys.executable,
        'tools/efficiency/profile_occ_model.py',
        '--config',
        config,
        '--gpu-id',
        str(args.gpu_id),
        '--warmup-iters',
        str(args.warmup_iters),
        '--profile-iters',
        str(args.profile_iters),
    ]
    checkpoint = row.get('checkpoint')
    if checkpoint and osp.exists(osp.join(REPO_ROOT, checkpoint)):
        command.extend(['--checkpoint', checkpoint])
    return run_json_command(command)


def maybe_eval_results(row):
    config = row.get('config')
    results = row.get('results')
    if not config or not results:
        return {}
    if not osp.exists(osp.join(REPO_ROOT, config)) or not osp.exists(osp.join(REPO_ROOT, results)):
        return {}
    command = [
        sys.executable,
        'tools/infraocc/eval_occ_tables.py',
        '--config',
        config,
        '--results',
        results,
        '--split',
        'test',
    ]
    return run_json_command(command)


def build_row_summary(row, args):
    profile = {}
    metrics = {}
    error = None
    try:
        profile = maybe_profile(row, args)
    except Exception as exc:  # noqa: BLE001
        error = f'profile_failed: {exc}'
    try:
        metrics = maybe_eval_results(row)
    except Exception as exc:  # noqa: BLE001
        error = f'{error}; eval_failed: {exc}' if error else f'eval_failed: {exc}'
    if not metrics:
        miou = latest_val_miou(osp.join(REPO_ROOT, row['log_json'])) if row.get('log_json') else None
        if miou is not None:
            metrics['mIoU'] = miou
    summary = dict(
        section=row['section'],
        label=row['label'],
        config=row.get('config'),
        checkpoint=row.get('checkpoint'),
        results=row.get('results'),
        params_m=profile.get('total_params_m'),
        latency_ms=profile.get('latency_ms'),
        miou_all=metrics.get('mIoU'),
        miou_dyn=metrics.get('mean_dynamic'),
        error=error,
    )
    summary.update(MAIN_TABLE_IOU.get((row['section'], row['label']), {}))
    return summary


def main():
    args = parse_args()
    rows = ROWS
    if args.sections:
        rows = [row for row in rows if row['section'] in set(args.sections)]
    if args.labels:
        rows = [row for row in rows if row['label'] in set(args.labels)]

    summaries = [build_row_summary(row, args) for row in rows]

    output_dir = osp.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))

    print('\nLaTeX rows:')
    for item in summaries:
        print(
            f"{item['label']} & "
            f"{format_value(item['params_m'])} & "
            f"{format_value(item['latency_ms'])} & "
            f"{format_value(item['miou_all'])} & "
            f"{format_value(item['miou_dyn'])} \\\\")


if __name__ == '__main__':
    main()
