## Summary
RoadOcc proposes camera-only roadside 3D semantic occupancy prediction as supervised temporal source routing. Instead of only aligning memory, the model selects whether each dynamic voxel should Persist with rigid history, Transport velocity-addressed history, or Refresh from current evidence. The DCA-VVE-VDSF pipeline improves InfraOcc results over STCOcc, including 65.29 mIoU, 32.37 dynamic mIoU, lower Direct MAVE, and higher DSR, with ablations supporting the address-plus-admissibility decomposition.

## Motivation and Positioning
The problem is well motivated: fixed roadside cameras repeatedly observe a stable scene, making temporal memory valuable, but dynamic objects move within the same coordinate frame, so rigid history can become stale. The paper’s key positioning is that flow only proposes where to read history; it does not decide whether the read feature is semantically admissible evidence for the current voxel.

| Prior direction | What it already covers | RoadOcc’s claimed distinction | Evidence provided |
|---|---|---|---|
| STCOcc | Sparse spatiotemporal occupancy and scene-flow renovation | Adds supervised P/T/R source admissibility | RoadOcc: 65.29 mIoU / 32.37 dyn. vs. STCOcc: 60.85 / 27.66 |
| ST-Occ | Temporal memory conditioned on uncertainty/dynamics | Explicit mutually exclusive routing target | No direct InfraOcc row reported |
| GDFusion | Motion-calibrated temporal fusion | Separates address estimation from source validity | No direct InfraOcc row reported |
| Let Occ Flow | Occupancy-flow learning | Tests whether transported history should be used | RoadOcc: 65.29 / 32.37 vs. 51.38 / 8.78 |
| ALOcc | Occupancy with cost-volume flow prediction | Fixed-roadside voxel source routing | RoadOcc: 65.29 / 32.37 vs. 43.31 / 5.86 |

The novelty is narrower than “temporal occupancy with flow,” but meaningful: supervised semantic source admissibility after motion-address construction.

## Contributions
The paper claims three main contributions: the P/T/R formulation, the DCA-VVE-VDSF architecture, and dynamic-first evaluation/diagnostics. The demonstrated contribution is strongest for the first: the authors define a supervised route target and show that adding P/T/R after VVE improves dynamic IoU and DSR. The architectural contribution is credible but assembled from several known ingredients: sparse attention, local correlation, velocity-guided memory, FIFO queues, and soft fusion.

## How the Proposed Approach Works End to End
RoadOcc receives four synchronized roadside camera images at time \(t\), calibrated to a fixed infrastructure coordinate frame. It predicts semantic occupancy over a 3D voxel grid and a planar velocity field. The core assumption is that static layout can reuse rigidly aligned history, while dynamic-class voxels may need history read from a displaced location. For a current voxel \(\mathbf{x}\), velocity defines a transported historical address \(\mathbf{x}-\Delta t v_t(\mathbf{x})\), but the model treats this only as a candidate address, not proof that history should be used.

The pipeline runs coarse-to-fine over voxel stages. DCA first identifies where additional dynamic reasoning is needed. It compares provisional current and static/free semantic distributions and combines that discrepancy with current dynamic probability to form a candidate map. This map gates a second projected deformable image cross-attention update, using visible camera anchors and depth consistency. In effect, DCA decides which voxels merit renewed current evidence before temporal fusion.

VVE then estimates motion-aware historical addresses. It projects dynamic support into BEV, constructs a local current-history correspondence volume, and recursively refines planar velocity from coarse to fine. Coarse stages capture larger motion structure, while finer stages refine boundaries. The predicted velocity supplies transport addresses for memory retrieval.

VDSF performs sparse temporal fusion. It selects top-K voxels using DCA score plus nonempty support, gathers three candidate sources for each selected voxel, and predicts a soft route distribution: Persist uses rigid aligned history at the same coordinate, Transport uses velocity-addressed history, and Refresh uses the current feature. The route target is built from current dynamic labels, nearest historical semantic occupancy, and motion targets. A stationary same-class historical support gives Persist; a moving same-class support at the backtraced address gives Transport; otherwise the target is Refresh.

For example, a car moving 4 m/s over 0.5 s is backtraced by 2 m, or five cells on the native 0.4 m grid. If the previous frame contains same-class support within the local matching radius at the same height, the voxel is labeled Transport; otherwise Refresh. This is semantic source admissibility, not instance correspondence: nearby same-class objects can satisfy support without proving identity. The evaluated parts include occupancy, dynamic/next-frame IoU, Direct MAVE, DSR, ablations, and route diagnostics; moving-platform and instance-level extensions are left for future work.

## Technical Soundness
The decomposition is technically coherent. Eq. 1 addresses where to sample memory; Eq. 7 defines source admissibility; Eq. 8 makes the three sources compete through a softmax rather than independent gates. This is a sound framing for fixed roadside sensing.

The main technical caveat is the supervision target. P/T/R uses same-class, same-height local support, with no instance-level tie-breaking. Therefore, the routing target supports semantic evidence reuse, not object identity tracking. This distinction is clearly acknowledged in the appendix and should remain explicit in the paper’s claims.

Other concerns are practical rather than fatal: DCA depends on early semantic predictions; the local VVE window may miss correspondences unless the coarse prior is good; and FIFO memory can accumulate model errors. The route diagnostics also show Refresh is weak: only 3.19% target share, 12.26% predicted share, 18.65% precision, and 17.29% IoU.

## Costs vs. Benefits
The benefits are substantial on InfraOcc: +4.44 mIoU, +4.71 dynamic mIoU, +4.24 static mIoU over STCOcc, lower Direct MAVE, and higher DSR. The ablations suggest these are not just from extra capacity.

The costs are also real: dense semantic occupancy supervision, tracklet-derived velocity targets, route-label construction, multi-scale VVE, DCA, sparse VDSF, FIFO queues, and multiple diagnostics. I do not deduct for the RoadOcc latency value marked TBD in Table A3, per instruction, but completed latency would still be important for deployment claims.

## Evaluation Assessment
The evaluation is strong within one benchmark. Table 2 gives broad InfraOcc comparisons, Table 3 tests joint occupancy-velocity behavior and transfers P/T/R into CRT-Fusion and STCOcc, and Tables 4-6/A5/A6 isolate components, address construction, route supervision, and token budget.

The main limitation is benchmark breadth. The authors justify using InfraOcc because compatible fixed-roadside dense occupancy benchmarks are scarce, but the empirical claim remains single-benchmark. Statistical rigor is mixed: Table 6 reports three-seed mean/std, but the main SOTA, transfer, and address-control results are point estimates. Negative results exist but should be discussed more directly: pedestrian slightly drops, CRT TP-MAVE worsens while DSR improves, VDSF alone gives only +0.20 mIoU, and Refresh routing is poor.

## Writing and Presentation
The central story is clear and memorable. “Flow gives an address; P/T/R decides admissibility” is consistently reinforced. The appendix is useful and contains real reproducibility details.

The paper is dense. Section 3 compresses many components and losses, Table 2 is overloaded, and some cross-references to supplementary sections appear inconsistent with the appendix’s A-F organization. An algorithm box or one-voxel trace would improve readability.

## Strengths
- S1: Clear and useful formulation of temporal memory as supervised Persist/Transport/Refresh routing.
- S2: Strong InfraOcc improvements over STCOcc in overall, dynamic, static, velocity, and support metrics.
- S3: Ablations convincingly separate address correction from source selection.
- S4: Good diagnostic design: Direct MAVE, TP-MAVE, DSR, next-frame dynamic IoU, and route statistics are interpreted carefully.
- S5: The paper explicitly distinguishes semantic admissibility from flow/address estimation.

## Weaknesses
- W1: Evidence is confined to one dense roadside benchmark.
- W2: P/T/R supervision is semantic source admissibility, not instance correspondence; same-class nearby objects can satisfy support.
- W3: Refresh routing is weak in Table A4, limiting route-level interpretability.
- W4: Most headline and ablation results lack multi-seed uncertainty.
- W5: Method complexity is high relative to the narrow evaluated setting.

## Questions for Authors
- Q1: How often do same-class nearby objects create ambiguous or wrong Transport/Persist targets?
- Q2: Can you report route accuracy separately for crowded scenes or close same-class vehicles?
- Q3: Are main Table 2/3/4/5 results stable across seeds, not only the Table 6 routing control?
- Q4: What are the dominant failure cases for Refresh?
- Q5: Can you reconcile the CRT-Fusion 53.85 vs. 53.83 mIoU discrepancy across Tables 2 and 3?

## Minor Issues
- Table 4 design indicators appear ambiguous in the OCR/table rendering.
- Some appendix cross-references use numbered sections while the supplement uses A-F labels.
- Figure 2 and Figure 7 captions could state their takeaways more explicitly.
- Table 2 is difficult to scan due to the large number of class columns.

## Venue-Specific Recommendations
- V1: For ICLR, emphasize the general learning abstraction: supervised source admissibility after memory-address proposal.
- V2: Add a compact algorithm box for DCA -> VVE -> VDSF -> P/T/R.
- V3: Add seed variance for the main comparison and key ablations.
- V4: Add an explicit limitations paragraph on semantic, not instance-level, routing.
- V5: Discuss negative results and Refresh failures.

## Overall Assessment
I lean weak accept. The paper has a clear idea, strong benchmark performance, and unusually well-aligned diagnostics. The central contribution is not merely better flow-based temporal fusion; it is the supervised decision of whether rigid, transported, or current evidence is admissible before fusion. The main reasons not to score higher are single-benchmark validation, high system complexity, limited seed reporting, and the fact that route supervision does not establish instance correspondence.

Scores: Overall Recommendation: 6/10. Soundness: 3/4. Presentation: 3/4. Contribution: 3/4. Reviewer Confidence: 4/5.

## Top Actions - Start Here
- T1: Add multi-seed uncertainty for Table 2 and the key ablation/transfer tables.
- T2: Explicitly state that P/T/R is semantic source admissibility, not instance correspondence.
- T3: Add failure analysis for Refresh and crowded same-class cases.
- T4: Clarify Table 4 row labels and the CRT-Fusion mIoU discrepancy.
- T5: Add a concise end-to-end algorithm box.

## Confidence
Confidence: 4/5. I read the submitted OCR, evidence manifest, citation manifest, numerical checks, and stage artifacts in full. My confidence is limited mainly by the absence of raw runs, code execution evidence, and multi-seed results for most headline tables.
