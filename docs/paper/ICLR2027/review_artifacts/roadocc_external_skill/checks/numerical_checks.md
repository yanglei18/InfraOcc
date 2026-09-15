# Numerical Consistency Checks: RoadOcc

Scope: all 19 pages of the submitted PDF, including the appendix. The pending RoadOcc latency in Table A3 is deliberately excluded from criticism and scoring at the user's request.

## Headline and Main-Table Checks

| Claim | Recalculation | Result |
|---|---:|---|
| RoadOcc over STCOcc in overall mIoU | 65.29 - 60.85 = 4.44 | verified |
| RoadOcc over STCOcc in dynamic mIoU | 32.37 - 27.66 = 4.71 | verified |
| RoadOcc over STCOcc in static mIoU | 89.98 - 85.74 = 4.24 | verified |
| Direct MAVE reduction over STCOcc | 1.946 - 1.669 = 0.277 m/s | verified |
| DSR gain over STCOcc | 61.11 - 57.12 = 3.99 points | verified |
| CRT-Fusion + P/T/R dynamic gain | 28.30 - 27.02 = 1.28 | verified |
| CRT-Fusion + P/T/R Direct MAVE change | 1.919 - 2.156 = -0.237 m/s | verified |
| CRT-Fusion + P/T/R DSR gain | 59.45 - 57.56 = 1.89 | verified |
| STCOcc + P/T/R dynamic gain | 28.80 - 27.66 = 1.14 | verified |
| STCOcc + P/T/R Direct MAVE change | 1.770 - 1.946 = -0.176 m/s | verified |
| STCOcc + P/T/R DSR gain | 58.78 - 57.12 = 1.66 | verified |

## Ablation Checks

| Claim | Recalculation | Result |
|---|---:|---|
| VDSF alone over baseline mIoU | 61.05 - 60.85 = 0.20 | verified |
| DCA/VDSF transition in dynamic and next-frame dynamic IoU | 30.10 - 28.55 = 1.55; 29.44 - 27.64 = 1.80 | verified, conditional on intended Table 4 row labels |
| VVE addition in mIoU | 64.17 - 63.01 = 1.16 | verified |
| P/T/R addition in mIoU | 65.29 - 64.17 = 1.12 | verified |
| Rigid read to VVE read | dynamic +0.87; Direct MAVE -0.456 m/s; DSR +1.51 | verified |
| VVE read to full P/T/R | dynamic +1.40; DSR +1.76 | verified |
| Full over no route control, Table 6 | dynamic +1.40; next dynamic +0.94; DSR +1.76; Direct MAVE -0.021 m/s | verified |

## Appendix Checks

| Claim | Recalculation | Result |
|---|---:|---|
| Five of six dynamic classes improve | bicycle +10.47; bus +6.68; pedestrian -0.19 | verified |
| Persist plus Transport target support | 49.42 + 47.40 = 96.82% | verified |
| Default to doubled token budget | dynamic +0.02; Direct MAVE -0.004; DSR +0.07; next dynamic -0.03 | verified |
| Full source-consistent address progression | dynamic +2.27; next dynamic +2.28; DSR +3.27; Direct MAVE -0.477 m/s | verified |

## Consistency Flags

- The canonical OCR loses some Table 4 checkmarks, but direct inspection of the typeset PDF confirms that the successive rows are DCA only and DCA plus VDSF. This is an OCR artifact, not a paper inconsistency.
- The canonical OCR misreads CRT-Fusion as 53.85 mIoU in one location. Direct PDF text extraction and the typeset source both give 53.83 in Tables 2 and 3, so no paper-level discrepancy remains.
- Only Table 6 reports three-seed variation. Main SOTA, address-control, and transfer claims otherwise rely on point estimates.
- Refresh is the weak route class in Table A4: 3.19% target share, 12.26% predicted share, 18.65% precision, and 17.29% IoU. This limits route-level interpretation but does not invalidate the end-to-end gains.
