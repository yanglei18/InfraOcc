import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_PATH = (
    ROOT / 'docs' / 'TPAMI2026' / 'local-git' / 'latex' / 'tab' /
    'tab12-robustness.tex')
DEFAULT_OUTPUT_DIR = ROOT / 'projects' / 'InfraOcc' / 'figures_update' / 'robustness'

BLOCK_LABELS = {
    'translation': 'Translation re-anchoring',
    'rotation': 'Rotation re-anchoring',
}
CONDITIONS = {
    'translation': ['Clean', '0.4 m', '0.8 m', '1.2 m'],
    'rotation': ['Clean', r'$0.4^\circ$', r'$0.8^\circ$', r'$1.2^\circ$'],
}
PLAIN_LABEL = 'Plain (C)'
PROSD_LABEL = 'ProSD-Occ (C)'


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Plot C ProSD vs C plain robustness differences from the paper LaTeX table.'
    )
    parser.add_argument(
        '--table-path',
        type=Path,
        default=DEFAULT_TABLE_PATH,
        help='LaTeX robustness table to parse.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for PNG/PDF/JSON outputs.')
    parser.add_argument(
        '--output-name',
        default='robustness_c_prosd_vs_plain',
        help='Output file stem.')
    return parser.parse_args()


def strip_latex(value):
    value = re.sub(r'\\cite\{[^}]+\}', '', value)
    value = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', value)
    value = re.sub(r'\\textit\{([^{}]*)\}', r'\1', value)
    value = value.replace('~', ' ')
    value = value.replace('$', '')
    value = value.replace('\\,', ' ')
    value = value.replace('\\mathrm', '')
    value = value.replace('{', '')
    value = value.replace('}', '')
    return re.sub(r'\s+', ' ', value).strip()


def normalize_method(value):
    value = strip_latex(value)
    if value.startswith('STCOcc') or value.startswith('Plain'):
        return PLAIN_LABEL
    if value.startswith(PROSD_LABEL):
        return PROSD_LABEL
    return value


def parse_metric_value(value):
    value = strip_latex(value).replace(' ', '')
    if value == '--':
        return None
    match = re.search(r'[-+]?\d+(?:\.\d+)?', value)
    if match is None:
        return None
    return float(match.group(0))


def parse_robustness_table(table_path):
    current_block = None
    parsed = {block: {} for block in BLOCK_LABELS}
    for raw_line in table_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.split('%', 1)[0].strip()
        if not line:
            continue
        if 'Translation re-anchoring' in line:
            current_block = 'translation'
            continue
        if 'Rotation re-anchoring' in line:
            current_block = 'rotation'
            continue
        if current_block is None or '&' not in line:
            continue

        line = re.sub(r'\\\\\s*$', '', line).strip()
        cells = [cell.strip() for cell in line.split('&')]
        if len(cells) < 5:
            continue

        method = normalize_method(cells[0])
        if method not in (PLAIN_LABEL, PROSD_LABEL):
            continue

        dynamic_values = [parse_metric_value(cell) for cell in cells[1:5]]
        if any(value is None for value in dynamic_values):
            continue

        parsed[current_block][method] = [{
            'condition': condition,
            'dynamic': value,
        } for condition, value in zip(CONDITIONS[current_block], dynamic_values)]

    for block in BLOCK_LABELS:
        missing = [
            method for method in (PLAIN_LABEL, PROSD_LABEL)
            if method not in parsed[block]
        ]
        if missing:
            raise ValueError(
                f'Missing {", ".join(missing)} rows in {BLOCK_LABELS[block]} '
                f'of {table_path}.')
    return parsed


def compute_delta_summary(parsed):
    summary = {}
    for block, methods in parsed.items():
        plain = methods[PLAIN_LABEL]
        prosd = methods[PROSD_LABEL]
        summary[block] = []
        for plain_row, prosd_row in zip(plain, prosd):
            dynamic_delta = prosd_row['dynamic'] - plain_row['dynamic']
            summary[block].append({
                'condition': plain_row['condition'],
                'plain_dynamic': plain_row['dynamic'],
                'prosd_dynamic': prosd_row['dynamic'],
                'delta_dynamic': round(dynamic_delta, 2),
            })
    return summary


def plot_difference(parsed, summary, output_dir, output_name):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.72), sharey=True)
    plain_color = '#6B6B6B'
    prosd_color = '#0072B2'
    panels = [('translation', 'Translation'), ('rotation', 'Rotation')]

    for panel_index, (block, title) in enumerate(panels):
        ax = axes[panel_index]
        labels = [row['condition'] for row in parsed[block][PLAIN_LABEL]]
        x_values = list(range(len(labels)))
        plain_values = [row['dynamic'] for row in parsed[block][PLAIN_LABEL]]
        prosd_values = [row['dynamic'] for row in parsed[block][PROSD_LABEL]]
        deltas = [row['delta_dynamic'] for row in summary[block]]

        ax.plot(
            x_values,
            plain_values,
            marker='o',
            markersize=3.6,
            linewidth=1.45,
            color=plain_color)
        ax.plot(
            x_values,
            prosd_values,
            marker='s',
            markersize=3.6,
            linewidth=1.45,
            color=prosd_color)
        ax.fill_between(
            x_values,
            plain_values,
            prosd_values,
            color=prosd_color,
            alpha=0.08)

        value_min = min(plain_values + prosd_values)
        value_max = max(plain_values + prosd_values)
        padding = max((value_max - value_min) * 0.28, 2.0)
        ax.set_ylim(value_min - padding * 0.35, value_max + padding)
        for x_pos, delta, plain_value, prosd_value in zip(
                x_values, deltas, plain_values, prosd_values):
            top_value = max(plain_value, prosd_value)
            prefix = '+' if delta >= 0 else ''
            ax.text(
                x_pos,
                top_value + padding * 0.22,
                f'{prefix}{delta:.2f}',
                ha='center',
                va='bottom',
                fontsize=4.8,
                color=prosd_color if delta >= 0 else '#B23A48')

        ax.set_title(title, fontsize=6.0, pad=2.5)
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, fontsize=4.8)
        ax.tick_params(axis='y', labelsize=4.8)
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.75)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=0.7)

    png_path = output_dir / f'{output_name}.png'
    pdf_path = output_dir / f'{output_name}.pdf'
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    parsed = parse_robustness_table(args.table_path)
    summary = compute_delta_summary(parsed)
    png_path, pdf_path = plot_difference(parsed, summary, args.output_dir,
                                         args.output_name)
    json_path = args.output_dir / f'{args.output_name}.json'
    json_path.write_text(
        json.dumps(
            {
                'source_table': str(args.table_path),
                'methods': [PLAIN_LABEL, PROSD_LABEL],
                'data': parsed,
                'deltas': summary,
                'outputs': {
                    'png': str(png_path),
                    'pdf': str(pdf_path),
                },
            },
            indent=2),
        encoding='utf-8')
    print(f'Saved {png_path}')
    print(f'Saved {pdf_path}')
    print(f'Saved {json_path}')


if __name__ == '__main__':
    main()
