# Citation Manifest

Evidence manifest: `docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/evidence_manifest.json`

## Source Inventory

- S1: RoadOcc submitted manuscript and appended supplement
  - Origin: submitted paper
  - Location: `docs/paper/ICLR2027/manuscript.pdf` and canonical OCR Markdown
  - Used for: method, experiments, numerical results, limitations, and references
- S2: STCOcc official CVPR 2025 page
  - Origin: web search and paper reference list
  - Location: https://openaccess.thecvf.com/content/CVPR2025/html/Liao_STCOcc_Sparse_Spatial-Temporal_Cascade_Renovation_for_3D_Occupancy_and_Scene_CVPR_2025_paper.html
  - Used for: closest sparse occupancy-and-flow baseline and novelty boundary
- S3: Occupancy Learning with Spatiotemporal Memory official ICCV 2025 page
  - Origin: web search and paper reference list
  - Location: https://openaccess.thecvf.com/content/ICCV2025/html/Leng_Occupancy_Learning_with_Spatiotemporal_Memory_ICCV_2025_paper.html
  - Used for: uncertainty-aware and dynamic-aware temporal memory positioning
- S4: GDFusion official CVPR 2025 paper
  - Origin: web search and paper reference list
  - Location: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Rethinking_Temporal_Fusion_with_a_Unified_Gradient_Descent_View_for_CVPR_2025_paper.pdf
  - Used for: motion-calibration and unified temporal-fusion positioning
- S5: Let Occ Flow paper
  - Origin: web search and paper reference list
  - Location: https://arxiv.org/abs/2407.07587
  - Used for: joint camera-only occupancy-flow positioning
- S6: OpenReview default review form
  - Origin: web search
  - Location: https://docs.openreview.net/reference/default-forms/default-review-form
  - Used for: interpretation of the 8/10 recommendation band

## External Claims Used in Review

- Claim: STCOcc already provides explicit occupied-state-guided sparse occupancy and scene-flow renovation.
  - Source id: S2
  - Review section: Motivation and Positioning, Significance
  - Provenance: web search and paper reference list
- Claim: ST-Occ already conditions temporal memory on uncertainty and dynamics.
  - Source id: S3
  - Review section: Motivation and Positioning, Significance
  - Provenance: web search and paper reference list
- Claim: GDFusion already treats motion calibration as a temporal cue.
  - Source id: S4
  - Review section: Motivation and Positioning, Significance
  - Provenance: web search and paper reference list
- Claim: Let Occ Flow jointly learns camera-only occupancy and occupancy flow.
  - Source id: S5
  - Review section: Motivation and Positioning
  - Provenance: web search and paper reference list
- Claim: An 8 recommendation corresponds to a clear accept band in the referenced OpenReview form.
  - Source id: S6
  - Review section: Overall Assessment
  - Provenance: web search

## Citation Sanity Check

- No invented bibliographic details: yes
- No unused manifest sources: yes
- No unsupported related-work claims: yes
- Search-derived claims have links or source notes: yes
- Offline limitations, if any: none for the listed claims
