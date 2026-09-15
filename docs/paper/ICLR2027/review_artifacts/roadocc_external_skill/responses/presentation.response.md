# Presentation-Stage Artifact: RoadOcc (ICLR 2027)

## Scope

Evidence source: `roadocc_external_skill_olmocr.md` and `evidence_manifest.json`. The manifest retains the full 19-page PDF because the appendix contains route-supervision and diagnostic evidence needed for scoring. Per instruction, the pending latency entry in Table A3 is not treated as a presentation defect.

## Presentation Score

**7 / 10.**

The paper is substantially readable and unusually explicit about the source-routing story, but it remains dense and occasionally over-compressed. The strongest presentation feature is that the paper repeatedly ties terminology, figures, diagnostics, and ablations back to the same Persist/Transport/Refresh hypothesis. The main presentation weaknesses are mostly organizational: some definitions and reproducibility-critical contracts are split across the main paper and appendix, several tables are visually overloaded, and appendix cross-references are not always aligned with the appendix’s actual lettered structure.

## Main-Paper Page Fit

The main narrative fits the requested nine pages: the conclusion ends on page 9, and references start immediately after the page-9 footer. The appendix begins after references on page 12. This is a clean fit for a nine-page narrative paper.

## Appendix Integration

The appendix is coherently motivated and mostly well integrated. Its opening paragraph explains the role of Sections A--F: data construction, velocity-target contract, P/T/R supervision, diagnostics, capacity, and implementation scope. The appendix materially supports claims made in the main paper, especially the diagnostic contract in Table A1, target construction in A.2, routing supervision in C/C.1, and controlled design studies in E.

The integration is weakened by some mismatched or imprecise cross-references. The main method states that “Queue lengths, token budgets, source matching, and routing settings appear in supplementary Secs. 3-5,” but the appendix is organized as A--F rather than numbered sections. The appendix introduction also phrases its map as “Sections A--A.2,” “Sections C--C.1,” and “Sections B--F,” which is understandable but not reader-friendly. This is fixable, but it matters because reproducibility-facing details are deliberately deferred to the appendix.

## Writing and Organization

The writing is clear at the level of the central thesis. The abstract and introduction quickly establish the core problem: rigid memory helps static structure but can become stale for moving objects; flow gives a candidate address but not a validity decision. The Persist/Transport/Refresh framing is memorable and gives the paper a stable through-line from Figure 1 to the method and ablations.

The organization is also mostly effective. The introduction defines the motivation, Figure 1 grounds the P/T/R target visually, Section 3 follows the DCA -> VVE -> VDSF chain, and Section 4 evaluates the same chain through main results, joint occupancy-velocity behavior, and ablations. This is a strength: the paper does not present diagnostics as an afterthought but aligns them with the claimed mechanism.

The main readability issue is density. Section 3 introduces DCA, VVE, VDSF, P/T/R, FIFO memory, multi-scale routing, source-consistent addresses, route targets, and multiple losses in a compact space. The paper is followable for a reviewer already familiar with occupancy and temporal fusion, but the method section leaves little room for slower explanatory transitions. A small end-to-end algorithm box or “one voxel through the pipeline” schematic would reduce cognitive load.

## Figures and Tables

The figures are generally well chosen. Figure 1 is a strong opening figure because it shows real-scene routing evidence rather than a generic architecture diagram. Figure 4 is useful as an execution trace, Figure 5 supports the coarse-to-fine decoding story, Figure 6 isolates velocity-guided retrieval, and Figure 7 connects P/T/R predictions to GT dynamic support.

The main weakness is caption sufficiency. Some figures rely heavily on surrounding prose. Figure 2’s caption, “Static-to-dynamic persistence in InfraOcc,” is too terse for a figure that apparently motivates the temporal asymmetry of the whole problem. Figure 7’s caption names predicted P/T/R but does not by itself explain what the reader should inspect or how it differs from Figure A3. More self-contained captions would improve skimmability.

The tables contain valuable evidence but are visually heavy. Table 2 is very wide, mixing aggregate metrics and many class-wise IoUs in one table. It supports the state-of-the-art claim, but it is hard to scan. Tables 5 and 6 are more focused and effective. Table A1 is a presentation strength because it explicitly defines diagnostic scope and failure modes. Table A3 provides capacity context, with the latency field intentionally excluded from this scoring exercise.

## Terminology

Persist/Transport/Refresh is a strong terminology choice. The paper consistently uses P/T/R to distinguish rigid memory reuse, velocity-addressed historical transport, and current-evidence refresh. The distinction between “where history is queried” and “whether that source is used” is repeated clearly in the introduction, Section 3.5, Section 3.6, and the experiments.

The terminology burden is still high. DCA, VVE, VDSF, P/T/R, DSR, Direct MAVE, TP-MAVE, Nxt., Dyn., static/gIoU/RayIoU, source-consistent address, and FIFO routing all appear in a short main paper. Most terms are eventually defined, but a compact notation/metric table in the main paper would reduce friction.

## Diagnostic-Scope Disclosure

This is a concrete presentation strength. The main paper states that semantic mIoU excludes some classes, Free IoU is reported separately, Direct MAVE covers all GT dynamic voxels, TP-MAVE conditions on semantic matches, and DSR should be read with TP-MAVE. Appendix Table A1 is especially strong because it maps each metric to support, failure mode, and role. This is good reviewer-facing disclosure and helps prevent over-reading a single metric.

## Reproducibility-Facing Presentation

The paper provides many reproducibility-facing details: four cameras, ResNet-50, image size, voxel range, grid scales, 0.5 s interval, optimizer, schedule, loss weights, target construction, velocity thresholds, route supervision, token budgets, queue reset, and metric definitions. Appendix C/C.1 and F are particularly useful.

The material presentation weakness is distribution of essential details. Some contracts needed to reproduce the method, such as source matching, routing targets, token budgets, queue lengths, and implementation scope, are split between Section 3, Appendix C/C.1, Appendix E, and Appendix F. This is acceptable for a nine-page paper, but cross-references should be made exact and consistent so readers can find the relevant details quickly.

## Concrete Strengths

- **S1: Clear central narrative.** The paper’s “flow gives an address, P/T/R decides admissibility” framing is easy to remember and consistently reinforced from the abstract through Sections 3 and 4.
- **S2: Figures support the mechanism.** Figures 1, 4, 6, and 7 are aligned with the claimed evidence-routing process rather than being generic qualitative examples.
- **S3: Diagnostic scope is unusually explicit.** Main-text metric discussion and Appendix Table A1 clarify what each metric measures and what failure it exposes.
- **S4: The appendix adds real explanatory value.** Appendix A.2, B, C/C.1, E, and F document target construction, metric contracts, routing supervision, and controlled studies rather than merely adding extra results.
- **S5: Main narrative fits nine pages.** The paper reaches the conclusion by page 9, with references and appendix afterward.

## Material Presentation Weaknesses

- **W1: The method section is dense enough to slow comprehension.** Section 3 compresses the full DCA -> VVE -> VDSF -> P/T/R -> loss chain into a short space. A concise algorithm box or stepwise runtime trace would make the mechanism easier to reproduce and evaluate.
- **W2: Some appendix references are imprecise.** The main paper refers to “supplementary Secs. 3-5,” while the appendix uses A--F labels. This weakens navigation to key reproducibility details.
- **W3: Several figures need more self-contained captions.** Figure 2 and Figure 7 in particular would benefit from captions that state the intended takeaway and how to read the panels.
- **W4: Table 2 is overloaded.** The table is valuable but difficult to scan because it combines aggregate metrics with many class-wise IoUs. Moving some class-wise detail to the appendix or visually separating aggregate and class-wise sections would improve readability.
- **W5: Reproducibility details are present but scattered.** The paper reports the necessary pieces across the main text and appendix, but readers must assemble them from multiple locations. Exact cross-references would substantially improve presentation without adding much length.
