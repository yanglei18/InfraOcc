import argparse
import json
import os
from os import path as osp


CONFIG_ROOT = 'projects/InfraOcc/configs/ablation'


EXPERIMENTS = [
    dict(
        table='core_mechanism',
        latex_source='latex/tab/core_mechanism.tex',
        purpose='Controlled ablation of progressive static-to-dynamic reasoning.',
        blocks=[
            dict(
                block='Necessity of Progressive S2D Reasoning',
                rows=[
                    dict(row='Plain Occupancy Prediction',
                         config='core_chain/infraocc_c_4x4_24e_plain.py'),
                    dict(row='Parallel S2D Fusion',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_supp_no_raw_bypass.py',
                         status='approximate',
                         note='Current code path closest to the no-conditioning control.'),
                    dict(row='Progressive S2D Reasoning',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Design of Progressive S2D Reasoning',
                rows=[
                    dict(row='No static suppression',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_suppression.py'),
                    dict(row='Suppression only',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_raw_bypass.py'),
                    dict(row='Suppression + raw bypass',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
        ]),
    dict(
        table='static_guidance_quality',
        latex_source='latex/tab/static_guidance_quality.tex',
        purpose='Predicted, oracle, and noisy static guidance for suppression.',
        blocks=[
            dict(
                block='Guidance quality ladder',
                rows=[
                    dict(row='None',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_suppression.py',
                         note='Suppression disabled while raw bypass and fusion remain unchanged.'),
                    dict(row='Noisy',
                         config='prior/infraocc_c_4x4_24e_prosd_guidance_noisy.py'),
                    dict(row='Predicted',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                    dict(row='Oracle',
                         config='prior/infraocc_c_4x4_24e_prosd_guidance_oracle.py'),
                ]),
        ]),
    dict(
        table='suppression_strength_design',
        latex_source='latex/tab/suppression_strength_design.tex',
        purpose='Necessity of learnable suppression and sensitivity to initialization.',
        blocks=[
            dict(
                block='Necessity of learnable suppression',
                rows=[
                    dict(row='No suppression',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_suppression.py'),
                    dict(row='Fixed suppression',
                         config='progressive/infraocc_c_4x4_24e_prosd_supp_fixed_alpha050.py'),
                    dict(row='Learnable suppression',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Sensitivity to initialization',
                rows=[
                    dict(row='Learnable (alpha_init=0.25)',
                         config='progressive/infraocc_c_4x4_24e_prosd_supp_learnable_alpha025.py'),
                    dict(row='Learnable (alpha_init=0.50)',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                    dict(row='Learnable (alpha_init=0.75)',
                         config='progressive/infraocc_c_4x4_24e_prosd_supp_learnable_alpha075.py'),
                ]),
        ]),
    dict(
        table='adaptive_fusion_design',
        latex_source='latex/tab/adaptive_fusion_design.tex',
        purpose='Necessity and internal design of adaptive fusion.',
        blocks=[
            dict(
                block='Necessity of Adaptive Fusion',
                rows=[
                    dict(row='Deterministic Average Merge',
                         config='fusion/infraocc_c_4x4_24e_prosd_fusion_det_average.py'),
                    dict(row='Deterministic Group-wise Merge',
                         config='fusion/infraocc_c_4x4_24e_prosd_fusion_det_groupwise.py'),
                    dict(row='Adaptive Group-wise Fusion',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Design of Adaptive Fusion',
                rows=[
                    dict(row='Confidence-only (group-wise)',
                         config='fusion/infraocc_c_4x4_24e_prosd_fusion_conf_only.py'),
                    dict(row='Probability-only (group-wise)',
                         config='fusion/infraocc_c_4x4_24e_prosd_fusion_probs_only.py'),
                    dict(row='Prob.+Conf. (class-wise)',
                         config='fusion/infraocc_c_4x4_24e_prosd_fusion_classwise.py'),
                    dict(row='Prob.+Conf. (group-wise)',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
        ]),
    dict(
        table='static_prior_consistency_ablation',
        latex_source='latex/tab/static_prior_consistency_ablation.tex',
        purpose='Necessity and formulation of static consistency.',
        blocks=[
            dict(
                block='Necessity of Static Consistency',
                rows=[
                    dict(row='No consistency',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_static_consistency.py'),
                    dict(row='Static consistency (default)',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Design of Static Consistency',
                rows=[
                    dict(row='Batch-level consensus',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                    dict(row='Pairwise agreement',
                         config='consistency/infraocc_c_4x4_24e_prosd_consistency_all_pairs.py'),
                ]),
        ]),
    dict(
        table='prior_source',
        latex_source=None,
        purpose='Supplementary prior-source split used in README and project notes.',
        blocks=[
            dict(
                block='Prior source',
                rows=[
                    dict(row='No prior',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_prior.py'),
                    dict(row='Sparse prior',
                         config='prior/infraocc_c_4x4_24e_prosd_sparse_prior.py'),
                    dict(row='Full prior',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Prior bias vs prior consistency',
                rows=[
                    dict(row='Prior bias only',
                         config='prior/infraocc_c_4x4_24e_prosd_prior_bias_only.py'),
                    dict(row='Prior consistency only',
                         config='prior/infraocc_c_4x4_24e_prosd_prior_consistency_only.py'),
                    dict(row='Full prior',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
        ]),
    dict(
        table='static_dynamic_loss_balance',
        latex_source='latex/tab/static_dynamic_loss_balance.tex',
        purpose='Static and dynamic branch loss-weight balance.',
        blocks=[
            dict(
                block='Loss balance',
                rows=[
                    dict(row='Static-light',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_static_half.py'),
                    dict(row='Dynamic-light',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_dynamic_half.py'),
                    dict(row='Balanced',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                    dict(row='Dynamic-heavy',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_dynamic_double.py'),
                    dict(row='Static-heavy',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_static_double.py'),
                ]),
        ]),
    dict(
        table='branch_loss_components',
        latex_source='latex/tab/branch_loss_components.tex',
        purpose='Static, dynamic, and fusion-output occupancy loss components.',
        blocks=[
            dict(
                block='Static occupancy supervision',
                rows=[
                    dict(row='CE only',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_static_ce_only.py'),
                    dict(row='CE + Sem.',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_static_ce_sem.py'),
                    dict(row='CE + Lovasz',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_static_ce_lovasz.py'),
                    dict(row='CE + Sem. + Lovasz',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Dynamic occupancy supervision',
                rows=[
                    dict(row='CE only',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_dynamic_ce_only.py'),
                    dict(row='CE + Sem.',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_dynamic_ce_sem.py'),
                    dict(row='CE + Lovasz',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_dynamic_ce_lovasz.py'),
                    dict(row='CE + Sem. + Lovasz',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py'),
                ]),
            dict(
                block='Fusion-output occupancy supervision',
                rows=[
                    dict(row='CE only',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_fusion_ce_only.py',
                         note='GeoScal disabled for aligned comparison.'),
                    dict(row='CE + Sem.',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_fusion_ce_sem.py',
                         note='GeoScal disabled for aligned comparison.'),
                    dict(row='CE + Lovasz',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_fusion_ce_lovasz.py',
                         note='GeoScal disabled for aligned comparison.'),
                    dict(row='CE + Sem. + Lovasz',
                         config='loss/infraocc_c_4x4_24e_prosd_loss_fusion_ce_sem_lovasz.py',
                         note='GeoScal disabled for aligned comparison.'),
                ]),
        ]),
    dict(
        table='accuracy_efficiency_comparison',
        latex_source='latex/tab/accuracy_efficiency_comparison.tex',
        purpose='End-to-end efficiency and inference overhead decomposition.',
        blocks=[
            dict(
                block='Inference overhead decomposition',
                rows=[
                    dict(row='Plain Occupancy Prediction',
                         config='core_chain/infraocc_c_4x4_24e_plain.py',
                         tool='tools/infraocc/profile_model.py'),
                    dict(row='Parallel S2D Fusion',
                         config='core_chain/infraocc_c_4x4_24e_prosd_no_supp_no_raw_bypass.py',
                         tool='tools/infraocc/profile_model.py'),
                    dict(row='Progressive S2D Reasoning',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py',
                         tool='tools/infraocc/profile_model.py'),
                ]),
        ]),
    dict(
        table='distance_wise_dynamic_iou',
        latex_source='latex/tab/distance_wise_dynamic_iou.tex',
        purpose='Distance-wise dynamic occupancy under camera, LiDAR, and multi-modal settings.',
        blocks=[
            dict(
                block='Distance-wise dynamic mIoU',
                rows=[
                    dict(row='ProSD-Occ (C)',
                         config='core_chain/infraocc_c_4x4_24e_prosd_full.py',
                         tool='tools/infraocc/eval_occ_tables.py --distance-bins 0 20 40 60 80'),
                    dict(row='ProSD-Occ (L)',
                         config=None,
                         status='historical',
                         note='Requires LiDAR ProSD-Occ checkpoint/config outside ablation/.'),
                    dict(row='ProSD-Occ (C+L)',
                         config=None,
                         status='historical',
                         note='Requires multi-modal ProSD-Occ checkpoint/config outside ablation/.'),
                ]),
        ]),
    dict(
        table='calibration_perturbation_robustness',
        latex_source='latex/tab/calibration_perturbation_robustness.tex',
        purpose='Paper table now denotes ego-frame re-anchoring, not calibration perturbation.',
        blocks=[
            dict(
                block='Translation re-anchoring',
                rows=[
                    dict(row='Plain (C)',
                         config='core_chain/infraocc_c_4x4_24e_plain.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                    dict(row='ProSD-Occ (C)',
                         config='main_table/infraocc_c_4x4_24e_prosd.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                    dict(row='ProSD-Occ (L)',
                         config='main_table/infraocc_l_4x4_24e_prosd.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                    dict(row='ProSD-Occ (C+L)',
                         config='main_table/infraocc_m_4x4_24e_prosd.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                ]),
            dict(
                block='Rotation re-anchoring',
                rows=[
                    dict(row='Plain (C)',
                         config='core_chain/infraocc_c_4x4_24e_plain.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                    dict(row='ProSD-Occ (C)',
                         config='main_table/infraocc_c_4x4_24e_prosd.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                    dict(row='ProSD-Occ (L)',
                         config='main_table/infraocc_l_4x4_24e_prosd.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                    dict(row='ProSD-Occ (C+L)',
                         config='main_table/infraocc_m_4x4_24e_prosd.py',
                         tool='tools/infraocc/run_background_overfitting.py'),
                ]),
        ]),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Print the InfraOcc TPAMI experiment manifest.')
    parser.add_argument('--config-root', default=CONFIG_ROOT)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def with_paths(config_root):
    manifest = []
    for experiment in EXPERIMENTS:
        entry = dict(experiment)
        blocks = []
        for block in experiment['blocks']:
            rows = []
            for row in block['rows']:
                row_entry = dict(row)
                config = row.get('config')
                row_entry['config_path'] = (
                    osp.join(config_root, config) if config else None)
                rows.append(row_entry)
            blocks.append(dict(block=block['block'], rows=rows))
        entry['blocks'] = blocks
        manifest.append(entry)
    return manifest


def main():
    args = parse_args()
    manifest = with_paths(args.config_root)
    print(json.dumps(manifest, indent=2))
    if args.output_json:
        output_dir = osp.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)


if __name__ == '__main__':
    main()
