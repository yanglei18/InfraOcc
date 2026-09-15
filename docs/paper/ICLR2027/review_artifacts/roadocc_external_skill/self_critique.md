**Retained Findings**

- Keep the core positive assessment: RoadOcc has a clear source-routing idea, strong InfraOcc gains over STCOcc, and useful diagnostics tying occupancy, velocity, and support recovery together.
- Keep the single-benchmark limitation, but phrase it carefully. The appendix gives a reasonable reason for using InfraOcc only, and Table 3 transfers P/T/R to CRT-Fusion and STCOcc, but all evidence is still within one protocol.
- Keep the semantic-admissibility limitation: P/T/R is not instance correspondence. This is supported by the paper’s own target definition, but because the paper acknowledges it, it should be framed as a scope limit rather than a correctness flaw.
- Keep the Refresh concern: Table A4 shows Refresh is rare and weak as an argmax route class, with 3.19% target share, 12.26% predicted share, 18.65% precision, and 17.29% IoU. Do not overstate this as invalidating end-to-end gains.
- Keep missing multi-seed uncertainty as a real weakness. Table 6 has three seeds, but the main SOTA, transfer, address-control, token-budget, and most ablation results are point estimates.

**Corrected Findings**

- Remove every CRT-Fusion discrepancy criticism. The final review must not ask authors to reconcile 53.85 vs. 53.83. Ground truth says CRT-Fusion is 53.83 in both Tables 2 and 3; the contrary value is OCR error.
- Remove every Table 4 checkmark/row-label ambiguity criticism. Ground truth says the row checkmarks are distinct and correct.
- Do not deduct for RoadOcc latency being `TBD` in Table A3. Avoid listing it as a weakness or top action. At most say latency was excluded from scoring.
- Rephrase “negative results are under-emphasized.” The paper already discusses several of them, including VDSF alone +0.20, CRT TP-MAVE tradeoff, and Refresh rarity. A fair critique is that deeper failure-case analysis would help, not that the paper hides these results.
- Tone down “ablations convincingly separate” to “strongly support” or “largely isolate.” Table 6 supports Full over None, but fine Gate/State/Full differences are modest relative to reported standard deviations.

**Missed Findings**

- Add a strength for the evaluation contract: Direct MAVE, TP-MAVE, DSR, next-frame dynamic IoU, and Table A1 are unusually careful and prevent misleading conditional-motion conclusions.
- Add a strength for capacity control: Table A5 shows saturation when doubling sparse-token budget, supporting that gains are not simply from more tokens.
- Add a strength for static preservation: RoadOcc improves static mIoU and all static classes while improving five of six dynamic classes.
- Add nuance on Table 6: Full vs. None is meaningfully supported, but the monotonic Gate -> State -> Full story should not be treated as statistically decisive without raw seeds or significance tests.
- Credit the authors for explicitly acknowledging benchmark scope and semantic-not-instance routing. These acknowledgments reduce, but do not eliminate, the relevant weaknesses.

**Score Calibration**

A 6/10 is better calibrated than an 8/10 under the stated ICLR interpretation. The evidence supports acceptance: the idea is clear, the gains are sizable, and the diagnostics are unusually aligned with the claim. But 8/10 means clearly above the acceptance threshold, and that is too strong for a system-heavy, specialized roadside occupancy paper validated on one benchmark family with limited main-result seed reporting and a semantic rather than instance-level routing target.

The current 6/10 weak-accept recommendation follows from the evidence once the false OCR criticisms are removed. If half-points or a 7 existed, 6.5 or 7 could be defensible. Between 6 and 8, choose 6. Do not lower the score because of the corrected table artifacts.

**Exact Revision Instructions**

- Delete Q5 entirely: “Can you reconcile the CRT-Fusion 53.85 vs. 53.83 mIoU discrepancy…”
- Delete the minor issue: “Table 4 design indicators appear ambiguous…”
- Replace T4 with: “Tighten appendix cross-references and add a concise algorithm or one-voxel execution trace.”
- Merge W1 and W5 into one weakness about limited external validation relative to engineering/supervision cost.
- Rewrite W2 as a scoped limitation: “P/T/R supervises semantic source admissibility, not instance correspondence; the paper acknowledges this, so claims should remain at the semantic-occupancy level.”
- Keep W3 but qualify it: “Refresh route fidelity is weak in GT-conditioned argmax diagnostics, which limits route interpretability and motivates failure analysis, but does not negate end-to-end gains.”
- Keep W4 and add the Table 6 nuance about fine-grained differences versus standard deviations.
- Add strengths for metric design, token-budget/capacity control, and static-class preservation.
- Final recommendation should remain: Overall 6/10, Soundness 3/4, Presentation 3/4, Contribution 3/4, Confidence 4/5.
