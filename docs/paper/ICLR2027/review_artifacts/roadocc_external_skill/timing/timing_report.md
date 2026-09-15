# Timing Report

Generated: `2026-08-19T12:14:46.089607Z`

## Overall

- Top-level timed steps: 8
- Failed timed steps: 3
- Sum of top-level timed step durations: 11m 44.4s
- Wall-clock span covered by timing logs: 43m 39.8s
- olmOCR page requests: 19
- Sum of olmOCR page request durations: 11m 21.8s
- Mean olmOCR page request duration: 35.9s

## Time by Category

| Category | Count | Failed | Duration |
| --- | ---: | ---: | ---: |
| ocr | 1 | 0 | 11m 27.3s |
| preflight | 6 | 3 | 17.0s |
| preprocessing | 1 | 0 | 0.1s |

## Step Timings

| Step | Category | Status | Model | Thinking | Duration | Started | Ended |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| preflight.olmocr | preflight | failed |  |  | 0.0s | 2026-08-19T11:31:06.156000Z | 2026-08-19T11:31:06.158000Z |
| preflight.html_explainer | preflight | failed |  |  | 0.0s | 2026-08-19T11:31:06.247000Z | 2026-08-19T11:31:06.249000Z |
| preflight.olmocr | preflight | failed |  |  | 0.0s | 2026-08-19T11:31:23.912000Z | 2026-08-19T11:31:23.920000Z |
| preflight.html_explainer | preflight | completed |  |  | 0.4s | 2026-08-19T11:31:24.027000Z | 2026-08-19T11:31:24.413000Z |
| preflight.olmocr | preflight | completed |  |  | 16.1s | 2026-08-19T11:35:19.846000Z | 2026-08-19T11:35:35.985000Z |
| preflight.html_explainer | preflight | completed |  |  | 0.5s | 2026-08-19T11:35:36.086000Z | 2026-08-19T11:35:36.572000Z |
| scope.pdf | preprocessing | completed |  |  | 0.1s | 2026-08-19T11:35:54.004000Z | 2026-08-19T11:35:54.134000Z |
| olmocr.total | ocr | completed | gpt-5.5 | low | 11m 27.3s | 2026-08-19T11:38:34.806000Z | 2026-08-19T11:50:02.057000Z |
| explainer.start | explainer | running |  |  | 0.0s | 2026-08-19T12:14:45.994000Z | 2026-08-19T12:14:45.994000Z |

## olmOCR Page Timings

These rows are produced by the Codex-backed olmOCR shim. The wrapper defaults to `--pages_per_group 1`, so each request is intended to correspond to one PDF page.

| Request | Page Guess | Status | Model | Thinking | Duration | Images | Output Chars | Started |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 |  | completed | gpt-5.5 | low | 20.7s | 1 | 1590 | 2026-08-19T11:38:39Z |
| 2 |  | completed | gpt-5.5 | low | 30.2s | 1 | 3793 | 2026-08-19T11:39:00Z |
| 3 |  | completed | gpt-5.5 | low | 35.3s | 1 | 4564 | 2026-08-19T11:39:30Z |
| 4 |  | completed | gpt-5.5 | low | 38.8s | 1 | 4841 | 2026-08-19T11:40:05Z |
| 5 |  | completed | gpt-5.5 | low | 37.6s | 1 | 4817 | 2026-08-19T11:40:44Z |
| 6 |  | completed | gpt-5.5 | low | 26.7s | 1 | 3685 | 2026-08-19T11:41:22Z |
| 7 |  | completed | gpt-5.5 | low | 1m 26.6s | 1 | 9563 | 2026-08-19T11:41:48Z |
| 8 |  | completed | gpt-5.5 | low | 33.6s | 1 | 4452 | 2026-08-19T11:43:15Z |
| 9 |  | completed | gpt-5.5 | low | 33.0s | 1 | 3782 | 2026-08-19T11:43:49Z |
| 10 |  | completed | gpt-5.5 | low | 32.1s | 1 | 3717 | 2026-08-19T11:44:22Z |
| 11 |  | completed | gpt-5.5 | low | 41.1s | 1 | 5597 | 2026-08-19T11:44:54Z |
| 12 |  | completed | gpt-5.5 | low | 28.8s | 1 | 3631 | 2026-08-19T11:45:35Z |
| 13 |  | completed | gpt-5.5 | low | 27.1s | 1 | 3881 | 2026-08-19T11:46:04Z |
| 14 |  | completed | gpt-5.5 | low | 33.5s | 1 | 4080 | 2026-08-19T11:46:31Z |
| 15 |  | completed | gpt-5.5 | low | 39.9s | 1 | 5123 | 2026-08-19T11:47:04Z |
| 16 |  | completed | gpt-5.5 | low | 38.3s | 1 | 4036 | 2026-08-19T11:47:44Z |
| 17 |  | completed | gpt-5.5 | low | 33.2s | 1 | 4145 | 2026-08-19T11:48:23Z |
| 18 |  | completed | gpt-5.5 | low | 37.9s | 1 | 5044 | 2026-08-19T11:48:56Z |
| 19 |  | completed | gpt-5.5 | low | 27.3s | 1 | 2731 | 2026-08-19T11:49:34Z |
