Act as an independent quality critic of the review, not as another reviewer of the paper. Read:

- docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/final_review.md
- docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/evidence_manifest.json
- docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/checks/numerical_checks.md
- docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/citation_manifest.md
- docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/self_critique.md

Check every canonical section, factual grounding, actionability of weaknesses and rebuttal questions, absence of identity leakage or inappropriate wording, score-to-critique consistency, paper specificity, and valid Markdown structure. Confirm that the review does not reintroduce the two corrected OCR artifacts and does not penalize the pending latency field.

Output exactly this structure:

# Review Quality Report

Evidence manifest: `docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/evidence_manifest.json`
Reviewed artifact: `docs/paper/ICLR2027/review_artifacts/roadocc_external_skill/final_review.md`

## Summary
[ready | ready with low-severity issues | blocked pending fixes]

## Findings
- Q1:
  - Severity: high | medium | low
  - Category: ...
  - Location: ...
  - Issue: ...
  - Required action: ...

If there are no findings, state `- None.`

## High-Severity Gate
- High-severity findings fixed: yes/no/not applicable
- Overrides: ...

Do not introduce new paper claims.
