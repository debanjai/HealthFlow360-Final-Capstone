# Iteration 20 - UI attribution fix: screenshot evidence

Captured 27 November 2026 on branch `fix/ui-attribution` (commit 183d1e3), local Streamlit app, synthetic test inputs only.

| File | What it shows |
|---|---|
| 00_before_emergency_shows_knowledge_badge.jpg | Before: an emergency stopped by the safety gate carried a "Knowledge" mode badge |
| 00_before_evidence_page_shows_knowledge_message.jpg | Before: after switching to Evidence, the Knowledge conversation stayed on screen under the Evidence title and sidebar |
| 01_after_emergency_safety_check_badge.jpg | After: emergency shows a "Safety check" badge, "Stopped by the safety check before any model was called", and "Handled by: safety check (no retrieval or model call)" in Details (0.8 s) |
| 02_after_injection_safety_check_badge.jpg | After: prompt injection in Evidence mode declined with the "Safety check" badge (0.9 s) |
| 03_after_mode_switch_separate_conversations.jpg | After: switching from Evidence to Knowledge shows only the Knowledge conversation, so the page and sidebar always match |
| 04_after_evidence_answer_unchanged.jpg | After: normal Evidence answer unchanged ("Evidence" badge, citations, Sources panel). 81.6 s because it was the first Evidence question after a restart (cold index load) |
| 05_after_knowledge_answer_unchanged.jpg | After: normal Knowledge answer unchanged ("Knowledge" badge, verification note, 7.5 s) |

Regression runs for this iteration: `reports/evaluation/e2e_20261127_084127/` (70/70) and `reports/evaluation/safety_regression_20261127_084748/` (30/30).
