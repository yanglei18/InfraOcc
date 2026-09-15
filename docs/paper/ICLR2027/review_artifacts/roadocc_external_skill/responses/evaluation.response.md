# Evaluation-Stage Artifact: RoadOcc

Evidence basis: canonical OCR Markdown `roadocc_external_skill_olmocr.md` and `evidence_manifest.json` under `docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/`. Scope note: Table A3 RoadOcc latency is `TBD`, but this is excluded from criticism per the manifest/user constraint.

## Verified Arithmetic

| Anchor | Stated Claim | Recomputed From Source Values | Status |
|---|---:|---:|---|
| Abstract; Table 2; Sec. 4.1 “InfraOcc occupancy benchmark” | RoadOcc beats STCOcc by +4.44 mIoU and +4.71 Dyn. | 65.29 - 60.85 = +4.44; 32.37 - 27.66 = +4.71 | matches |
| Sec. 4.1; Table 2 | Static mIoU rises by +4.24 to 89.98 | 89.98 - 85.74 = +4.24 | matches |
| Abstract; Table 3 | Direct MAVE drops from 1.946 to 1.669 | 1.669 - 1.946 = -0.277 | values match; delta not explicitly stated |
| Sec. 4.1; Table 3 | DSR rises from 57.12 to 61.11 | 61.11 - 57.12 = +3.99 | values match; delta not explicitly stated |
| Table 1; Sec. 2.2 | Same-class gains: slow +1.32, medium +25.57, fast +54.16 | 86.51 - 85.19 = +1.32; 96.65 - 71.08 = +25.57; 93.73 - 39.57 = +54.16 | matches |
| Table 1 | All-speed Same gain +16.28 | 89.21 - 72.93 = +16.28 | matches |
| Table 3; Sec. 4.1 | CRT-Fusion + VDSF P/T/R raises Dyn. by +1.28 and DSR by +1.89 | 28.30 - 27.02 = +1.28; 59.45 - 57.56 = +1.89 | matches |
| Table 3; Sec. 4.1 | STCOcc + VDSF P/T/R raises Dyn. by +1.14 and improves DSR | 28.80 - 27.66 = +1.14; 58.78 - 57.12 = +1.66 | matches |
| Table 3; Sec. 4.1 TP-MAVE/DSR trade-off | CRT TP-MAVE increases with higher DSR | 1.958 - 1.839 = +0.119; DSR +1.89 | matches |
| Table 4; Sec. 4.2 “Overall ablation” | VDSF alone adds +0.20 mIoU | 61.05 - 60.85 = +0.20 | matches |
| Table 4; Sec. 4.2 | DCA/VDSF transition raises Dyn. +1.55 and Nxt. +1.80 | 30.10 - 28.55 = +1.55; 29.44 - 27.64 = +1.80 | matches if the intended transition is row 61.69 to row 63.01 |
| Table 4; Sec. 4.2 | VVE adds +1.16 mIoU | 64.17 - 63.01 = +1.16 | matches |
| Table 4; Sec. 4.2 | P/T/R adds +1.12 mIoU | 65.29 - 64.17 = +1.12 | matches |
| Table 5; Sec. 4.2 | VVE read raises Dyn. +0.87, lowers dMAVE 2.146 to 1.690, raises DSR +1.51 | 30.97 - 30.10 = +0.87; 1.690 - 2.146 = -0.456; 59.35 - 57.84 = +1.51 | matches; dMAVE drop is 0.456 |
| Table 5; Sec. 4.2 | P/T/R adds +1.40 Dyn. and +1.76 DSR | 32.37 - 30.97 = +1.40; 61.11 - 59.35 = +1.76 | matches |
| Table 6; Sec. 4.2 | Full gains +1.40 Dyn., +0.94 Nxt., +1.76 DSR over None | 32.37 - 30.97 = +1.40; 31.72 - 30.78 = +0.94; 61.11 - 59.35 = +1.76 | matches |
| Fig. A4; App. D “Class-wise improvement” | Bike +10.47, Bus +6.68, Pedestrian -0.19; five of six dynamic classes improve | Bike 37.60 - 27.13 = +10.47; Bus 55.46 - 48.78 = +6.68; Pedestrian 22.66 - 22.85 = -0.19 | matches |
| Table A4; App. E “Route-wise diagnostics” | Persist + Transport target support = 96.82% | 49.42 + 47.40 = 96.82 | matches |
| Table A5; App. E “Sparse-token budget” | Small to default: Dyn. +0.63, DSR +0.83 | 32.37 - 31.74 = +0.63; 61.11 - 60.28 = +0.83 | matches |
| Table A5; App. E | Default to doubled: Dyn. +0.02, dMAVE -0.004, DSR +0.07, Nxt. -0.03 | 32.39 - 32.37 = +0.02; 1.665 - 1.669 = -0.004; 61.18 - 61.11 = +0.07; 31.69 - 31.72 = -0.03 | matches |
| Table A6; Fig. A6; App. E | Target-vector trace over ego read: Dyn. +1.31, dMAVE -0.442 | 31.41 - 30.10 = +1.31; 1.704 - 2.146 = -0.442 | matches |
| Table A6; Fig. A6; App. E | Source-consistent over target-vector: Dyn. +0.96, Nxt. +1.06, DSR +1.29, dMAVE -0.035 | 32.37 - 31.41 = +0.96; 31.72 - 30.66 = +1.06; 61.11 - 59.82 = +1.29; 1.669 - 1.704 = -0.035 | matches |
| Table A6; Fig. A6; App. E | Full address progression: Dyn. +2.27, Nxt. +2.28, DSR +3.27, dMAVE -0.477 | 32.37 - 30.10 = +2.27; 31.72 - 29.44 = +2.28; 61.11 - 57.84 = +3.27; 1.669 - 2.146 = -0.477 | matches |
| Fig. A5; App. E “Progressive training behavior” | Dyn. +3.63, Direct MAVE -0.271; final read +1.65 Dyn., -0.021 dMAVE | Source endpoint values are not tabulated in OCR Markdown | cannot verify from provided Markdown source values |

## Consistency Findings

C1. Table 4, rows with outputs `61.69/28.55/27.64` and `63.01/30.10/29.44`: the canonical OCR table markup shows the same visible design pattern for both rows, while Sec. 4.2 “Overall ablation” treats them as different sequential controls. The arithmetic is internally consistent if the intended transition is row 61.69 to row 63.01, but the table markup should be corrected or clarified.

C2. Table 2 vs. Table 3, CRT-Fusion mIoU: canonical OCR Table 2 lists CRT-Fusion as 53.85 mIoU, while Table 3 lists 53.83. This does not affect RoadOcc-vs-STCOcc headline deltas, but it is a source consistency issue for reproduced baseline accounting.

C3. Table 6 reports mean +/- standard deviation over three independent seeds, but Tables 2, 3, 4, 5, A5, and A6 do not report seed variance. Main SOTA, transfer, address-control, and sparse-budget conclusions therefore rely on point estimates except where Table 6 overlaps with the P/T/R control.

## Reviewer Judgment

Tables 1-3 are mostly fair and adequate for the paper’s main InfraOcc claim. Table 2 compares many camera-only occupancy baselines under the stated InfraOcc protocol, and Sec. 4 “Dataset and metrics” says comparisons share voxel range, ignored classes, and evaluator. The RoadOcc-vs-STCOcc headline is supported within this benchmark. However, “state of the art” should be read as InfraOcc camera-only SOTA, not broad occupancy SOTA across datasets.

Table 1 is useful motivation, not direct model evidence. It cleanly shows that GT-flow transport helps much more at medium/fast speeds than slow speeds, supporting the need for motion-aware retrieval. Because it uses generated GT-flow rather than predicted VVE, it supports the problem formulation more strongly than it supports the achieved model behavior.

Table 3 provides credible transfer evidence for adding VDSF P/T/R to CRT-Fusion and STCOcc: both backbones improve Dyn., Direct MAVE, and DSR. This supports a limited generality claim for flow-aware temporal occupancy backbones under matched InfraOcc conditions. It does not establish cross-dataset transfer or full RoadOcc portability beyond these two controlled insertions.

The sequential component controls are directionally strong but not perfectly documented. Table 4 supports the DCA -> VVE -> P/T/R story through ordered gains, Table 5 isolates address correction from source selection, and Table 6 isolates route supervision/fusion. The main weakness is Table 4’s ambiguous OCR/table row encoding and the absence of seed variance for Table 4/5.

The TP-MAVE/DSR treatment is appropriate. Table 3 correctly avoids overclaiming conditional TP-MAVE: CRT-Fusion + P/T/R worsens TP-MAVE by +0.119 while improving DSR by +1.89 and Direct MAVE by -0.237. Appendix B explicitly defines TP-MAVE as conditional on semantic matches and DSR as support recall, which is the right framing for the coverage/conditional-error trade-off.

Route fidelity is partially supported. Figure 7 and Figure A3 are GT-support visual diagnostics, and Table A4 shows high Persist/Transport precision, but Refresh is weak: target 3.19%, predicted 12.26%, precision 18.65%, IoU 17.29%. The route target is semantic voxel support, not instance identity, as stated in Appendix C.1 and F, so the evidence supports semantic source admissibility rather than object-level correspondence fidelity.

Benchmark breadth is the largest evaluation limitation. Appendix A.1 gives a reasonable justification that InfraOcc may be the only compatible dense fixed-roadside camera-only occupancy benchmark, and the paper compensates with many baselines plus controlled transfer. Still, the empirical claim remains single-benchmark. Claims about moving platforms, longer horizons, and broader deployment are appropriately left as future work in the conclusion.

Negative results are present but under-emphasized: VDSF alone adds only +0.20 mIoU in Table 4, Pedestrian drops by -0.19 in Figure A4, CRT TP-MAVE worsens after P/T/R in Table 3, Refresh routing is low-IoU in Table A4, and doubling token budget gives near-zero gains with Nxt. -0.03 in Table A5. These are valuable diagnostics, but the paper would be stronger with an explicit failure-case discussion.

Statistical rigor is limited. Table 6’s three-seed mean +/- std is helpful and supports the Full-over-None P/T/R gain, but the incremental Gate -> State -> Full gaps are small relative to reported standard deviations. No confidence intervals, significance tests, raw seed values, or seeded main/transfer results are reported. The evaluation is quantitatively coherent, but statistical evidence is not strong enough to make fine-grained superiority claims beyond the larger observed margins.
