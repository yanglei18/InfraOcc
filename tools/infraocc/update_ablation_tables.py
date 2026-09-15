import argparse
import importlib.util
import json
import re
from os import path as osp


ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
DEFAULT_LATEX_ROOT = osp.join(ROOT, 'docs', 'TPAMI2026', 'InfraOcc - TPAMI')
DEFAULT_RESULTS_JSON = osp.join(ROOT, 'tools', 'infraocc', 'ablation_metrics.json')
MANIFEST_PATH = osp.join(ROOT, 'tools', 'infraocc', 'experiment_manifest.py')
SUPPORTED_TRAIN_METRIC_TABLES = {
    'core_mechanism',
    'static_guidance_quality',
    'suppression_strength_design',
    'adaptive_fusion_design',
    'static_prior_consistency_ablation',
    'static_dynamic_loss_balance',
    'branch_loss_components',
}


def load_manifest():
    spec = importlib.util.spec_from_file_location('experiment_manifest', MANIFEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXPERIMENTS


def load_results(results_json):
    if not osp.exists(results_json):
        return {}
    with open(results_json, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def save_results(payload, results_json):
    with open(results_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


ROW_ALIASES = {
    'suppression_strength_design': {
        'Learnable (alpha_init=0.25)': 'Learnable ($\\alpha_{\\mathrm{init}}=0.25$)',
        'Learnable (alpha_init=0.50)': 'Learnable ($\\alpha_{\\mathrm{init}}=0.50$)',
        'Learnable (alpha_init=0.75)': 'Learnable ($\\alpha_{\\mathrm{init}}=0.75$)',
    },
}


def canonical_row_name(table_name, row_name):
    return ROW_ALIASES.get(table_name, {}).get(row_name, row_name)


def block_marker(block_name):
    return r'\multicolumn{'


def format_metrics(table_name, row_name, metrics):
    g = f"{metrics['gIoU']:.2f}"
    m = f"{metrics['mIoU']:.2f}"
    d = f"{metrics['mean_dynamic']:.2f}"
    s = f"{metrics['mean_static']:.2f}"

    if table_name == 'suppression_strength_design':
        return [g, m, d, s]
    if table_name == 'branch_loss_components':
        return [g, m, d, s]
    if table_name == 'static_dynamic_loss_balance':
        return [g, m, d, s]
    return [g, m, d, s]


def replace_trailing_metrics(line, new_metrics):
    line_no_nl = line.rstrip('\n')
    slash_index = line_no_nl.rfind(r'\\')
    if slash_index < 0:
        raise RuntimeError(f'Row does not end with LaTeX line break: {line}')
    body = line_no_nl[:slash_index].rstrip()
    parts = [part.strip() for part in body.split('&')]
    if len(parts) < len(new_metrics) + 1:
        raise RuntimeError(f'Unexpected row format: {line}')
    parts[-len(new_metrics):] = new_metrics
    return ' & '.join(parts) + r' \\' + '\n'


def update_table_file(table_name, latex_root, latex_rel_path, block_name, row_name, metrics):
    latex_path = osp.join(latex_root, latex_rel_path)
    with open(latex_path, 'r', encoding='utf-8') as handle:
        lines = handle.readlines()

    target_row = canonical_row_name(table_name, row_name)
    replacement_metrics = format_metrics(table_name, target_row, metrics)

    inside_block = False
    updated = False
    for index, line in enumerate(lines):
        if block_name in line and block_marker(block_name) in line:
            inside_block = True
            continue
        if inside_block and ('\\midrule' in line or '\\bottomrule' in line):
            inside_block = False
        if not inside_block:
            continue
        if line.lstrip().startswith(target_row + ' &'):
            lines[index] = replace_trailing_metrics(line, replacement_metrics)
            updated = True
            break

    if not updated:
        raise RuntimeError(
            f'Failed to update row "{target_row}" under block "{block_name}" in {latex_path}')

    with open(latex_path, 'w', encoding='utf-8') as handle:
        handle.writelines(lines)


def update_for_config(config_rel_path, metrics, latex_root):
    manifest = load_manifest()
    updated = []
    for experiment in manifest:
        latex_source = experiment.get('latex_source')
        if not latex_source:
            continue
        if experiment['table'] not in SUPPORTED_TRAIN_METRIC_TABLES:
            continue
        for block in experiment['blocks']:
            for row in block['rows']:
                if row.get('config') != config_rel_path:
                    continue
                update_table_file(
                    experiment['table'],
                    latex_root,
                    experiment['latex_source'],
                    block['block'],
                    row['row'],
                    metrics,
                )
                updated.append({
                    'table': experiment['table'],
                    'latex_source': experiment['latex_source'],
                    'row': row['row'],
                })
    return updated


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update TPAMI ablation LaTeX tables from parsed InfraOcc metrics.')
    parser.add_argument('--config-rel', required=True, help='Config path relative to ablation/.')
    parser.add_argument('--metrics-json', required=True, help='Metrics JSON from extract_train_metrics.py.')
    parser.add_argument(
        '--latex-root',
        default=DEFAULT_LATEX_ROOT,
        help='Root directory containing latex/tab/*.tex.')
    parser.add_argument(
        '--results-json',
        default=DEFAULT_RESULTS_JSON,
        help='Registry JSON path for completed ablation metrics.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.metrics_json, 'r', encoding='utf-8') as handle:
        metrics = json.load(handle)

    registry = load_results(args.results_json)
    registry[args.config_rel] = metrics
    save_results(registry, args.results_json)

    updated = update_for_config(args.config_rel, metrics, args.latex_root)
    print(json.dumps({
        'config_rel': args.config_rel,
        'latex_root': osp.abspath(args.latex_root),
        'metrics_json': osp.abspath(args.metrics_json),
        'results_json': osp.abspath(args.results_json),
        'updated_rows': updated,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
