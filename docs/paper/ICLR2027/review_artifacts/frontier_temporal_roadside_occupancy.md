# Research landscape for temporal roadside 3D occupancy

## Scope

This landscape covers camera-based semantic occupancy using temporal memory, motion or occupancy flow, and sparse or adaptive fusion. RoadOcc addresses the narrower fixed-roadside setting, where the sensor frame is persistent but moving objects invalidate fixed-coordinate historical reads.

## Closest recent methods

STCOcc, published at CVPR 2025, performs occupied-state-guided sparse renovation for joint occupancy and scene-flow prediction. It is the closest implemented baseline and establishes sparse temporal interaction as a strong foundation.

Occupancy Learning with Spatiotemporal Memory, published at ICCV 2025, maintains recurrent scene memory and conditions retrieval on uncertainty and dynamics. GDFusion, published at CVPR 2025, decomposes temporal fusion into consistency, motion calibration, and geometric complementation. ALOcc and Let Occ Flow further establish cost-volume correspondence and occupancy-flow learning. LinkOcc learns temporal semantic association, while LMPOcc retrieves long-term traversal priors.

InfraOcc is the directly compatible dense fixed-roadside benchmark used by RoadOcc. RoadOcc reports 65.29 overall mIoU, 32.37 dynamic mIoU, 89.98 static mIoU, Direct MAVE of 1.669 m/s, and DSR of 61.11.

## RoadOcc's position

The field is moving from indiscriminate temporal aggregation toward selective memory, motion compensation, uncertainty-aware retrieval, and sparse computation. RoadOcc contributes a sharper decision abstraction within this trend. VVE proposes where historical evidence should be read, while P/T/R determines whether fixed-coordinate history, transported history, or current evidence should enter fusion.

The defensible novelty claim is explicit semantic source admissibility and its separation from historical address construction. It is not flow estimation, sparse temporal fusion, or adaptive weighting in isolation. The manuscript supports this boundary through sequential component ablations, capacity- and token-matched address controls, three-seed routing controls, and transfer to CRT-Fusion and STCOcc.

## Evidential strengths

RoadOcc improves STCOcc by 4.44 overall mIoU and 4.71 dynamic mIoU while increasing static mIoU by 4.24. Direct MAVE decreases from 1.946 to 1.669 and DSR rises from 57.12 to 61.11, linking occupancy gains to address quality and recovered support. The evaluation also distinguishes all-target motion error from conditional matched-voxel error.

## Remaining open problems

External validity remains the largest limitation. The available dense fixed-roadside evidence comes from one benchmark protocol and two infrastructure units. Transfer across temporal backbones supports architectural portability but not cross-site generalization.

P/T/R uses same-class voxel support without instance identities. Nearby same-class objects may therefore provide ambiguous semantic support. This is consistent with semantic occupancy, but it should not be interpreted as physical instance provenance.

Route diagnostics are conditioned on valid ground-truth dynamic voxels. They isolate address and source decisions cleanly but do not characterize all false-positive or missed predicted voxels. End-to-end dynamic IoU, next-frame IoU, Direct MAVE, and DSR provide the complementary deployed view.

Three-seed uncertainty is reported for the final routing control, while headline, transfer, and earlier component results remain point estimates.

## Sources checked

ICLR 2027 reviewer guidelines: https://iclr.cc/Conferences/2027/ReviewerGuidelines

STCOcc official CVPR 2025 page: https://openaccess.thecvf.com/content/CVPR2025/html/Liao_STCOcc_Sparse_Spatial-Temporal_Cascade_Renovation_for_3D_Occupancy_and_Scene_CVPR_2025_paper.html

GDFusion official CVPR 2025 page: https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Rethinking_Temporal_Fusion_with_a_Unified_Gradient_Descent_View_for_CVPR_2025_paper.html

ST-Occ official ICCV 2025 page: https://openaccess.thecvf.com/content/ICCV2025/html/Leng_Occupancy_Learning_with_Spatiotemporal_Memory_ICCV_2025_paper.html

LinkOcc IEEE Xplore page: https://ieeexplore.ieee.org/document/10734407

LMPOcc preprint: https://arxiv.org/abs/2504.13596
