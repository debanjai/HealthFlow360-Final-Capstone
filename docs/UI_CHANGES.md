# HealthFlow 360 - UI Changes Log

- **Branch:** `ui-fixes` (from `main` at release v2.0, commit `ea44b27`)
- **Owner:** Gautam Singh (Member 6 - Frontend and Deployment)
- **Status:** Rounds 1 to 3 complete and tested. Not yet merged into `main`.
- **Last updated:** 25 November 2026

This document records every change made to the HealthFlow 360 interface on the `ui-fixes` branch: what the problem was, what was changed, which files were touched, and how each change was tested. It is updated as work continues so the team can review everything in one place before merging.

---

## Summary

| # | Change | Round | Files | Tested |
|---|---|---|---|---|
| 1 | Knowledge answers no longer cut off mid-sentence | 1 | `src/generation.py`, `app.py` | Yes |
| 2 | Sources show titles and links instead of raw IDs | 1 | `app.py` | Yes |
| 3 | Citation numbers in the answer match the Sources list | 1 | `src/generation.py` | Yes |
| 4 | Emergencies and declined requests use different banners | 1 | `app.py` | Yes |
| 5 | "Model-generated knowledge" label only on normal Knowledge answers | 1 | `app.py` | Yes |
| 6 | Notice before the slow first Evidence question | 1 | `app.py` | Yes |
| 7 | Declined requests labelled "Declined", not "Escalated" | 2 | `app.py` | Yes |
| 8 | Each answer tagged Knowledge or Evidence | 2 | `app.py` | Yes |
| 9 | Developer details moved into a collapsed "Details" section | 2 | `app.py` | Yes |
| 10 | Each source shows the passage the answer drew on | 2 | `app.py` | Yes |
| 11 | "Not answered" replies suggest what to try next | 2 | `app.py` | Yes |
| 12 | Triage card shows readable labels instead of backend values | 2 | `app.py` | Yes |
| 13 | New visual design, "Clinical Calm" (theme, fonts, layout) | 3 | `.streamlit/config.toml`, `app.py` | Yes |
| 14 | Redesigned message states (emergency, declined, error, loading) | 3 | `app.py` | Yes |
| 15 | "Ask in Knowledge mode" button on unanswered Evidence questions | 3 | `app.py` | Yes |
| 16 | Two-column Report Triage with an urgency-coloured result card | 3 | `app.py` | Yes |

**Files changed:** `app.py` (most changes), `src/generation.py` (two small changes, see below) and a new `.streamlit/config.toml` (theme). No other module was modified. The data schemas (`src/schemas.py`) and the agent workflow (`src/agent.py`) are unchanged.

---

## Note for the team: impact on evaluation results

Two changes in `src/generation.py` alter what the model returns:

- **Change 1** adds one sentence to the Knowledge-mode system prompt asking the model to stay under about 250 words and finish its last sentence.
- **Change 3** renumbers the `[n]` citation markers in Evidence answers. The cited document IDs in `Answer.citations` are unchanged; only the numbers in the answer text change.

The v2.0 evaluation report was produced before these changes. The structural checks in `scripts/run_e2e_evaluation.py` and `scripts/run_safety_regression.py` compare citation IDs, statuses and risk levels, which these changes do not affect, but the E2E evaluation should be re-run once before merging to confirm the 70/70 result still holds.

---

## Round 1 changes

### 1. Knowledge answers cut off mid-sentence

**Problem.** Knowledge answers stopped mid-sentence, for example "...Management: Lifestyle changes ... are often recommended" with no ending. The model output limit is 450 tokens (enforced in `src/agent.py`), and long answers ran past it.

**Change.**

- `src/generation.py`: the Knowledge system prompt now ends with "Keep the answer under about 250 words and always finish your final sentence." The 450-token limit itself is unchanged.
- `app.py`: if an answer still ends without punctuation after about 280 words (close to the limit), the screen shows "This answer reached the length limit and may be incomplete. Try a narrower question." Short answers that simply end on a bullet point are not flagged.

**Tested.** "What is hypertension?" now returns a complete answer ending on a full sentence (7.6 s).

### 2. Sources showed raw IDs only

**Problem.** The Sources list showed only document IDs such as `mlp_2` and `mlp_1507`. Readers could not tell what the source was.

**Change.** `app.py` looks up each cited ID with `retrieval.get_source()` and shows a numbered title linking to the source page (for example "[1] Asthma" linking to medlineplus.gov), with the source ID underneath. If an ID cannot be found, the raw ID is shown as before.

**Tested.** "What are the symptoms of asthma?" lists "[1] Asthma" and "[2] Asthma in Children", both linking to MedlinePlus.

### 3. Citation numbers did not match the Sources list

**Problem.** The model numbers excerpts 1 to 4 as retrieved, but the Sources list removes duplicate documents. An answer could cite [1], [2] and [4] while only two sources were listed, because excerpts 1 and 4 came from the same document.

**Change.** `src/generation.py`: after the cited documents are deduplicated, the `[n]` markers in the answer text are renumbered to match the list. Excerpts from the same document share one number, numbered 1 to N in order of first citation.

**Tested.** The asthma answer now cites only [1] and [2] with two sources listed; the diabetes answer cites [1] to [3] with three sources listed.

### 4. The same red banner for emergencies and refusals

**Problem.** Every escalated response, including declined prompt injections and dosage requests, showed the same red "Clinician review or urgent professional attention required" box, which weakened the emergency warning.

**Change.** `app.py` distinguishes the two cases using the response constants in `src/agent.py`:

- **Emergency:** red banner above the answer, "Possible medical emergency. Contact local emergency services now."
- **Declined request:** neutral blue banner, "Request declined. This is outside what the assistant can safely answer."

**Tested.** The chest pain question shows the red banner; the prompt injection and ibuprofen dosage questions show the blue banner.

### 5. Wrong label on emergencies and refusals

**Problem.** "Model-generated knowledge, not verified against retrieved documents" appeared under emergency and refusal messages, which are fixed texts and not written by the model.

**Change.** `app.py` shows this label only on answered Knowledge-mode responses.

**Tested.** The label appears on the hypertension answer only.

### 6. No warning before the slow first Evidence question

**Problem.** After every app restart, the first Evidence question takes about 3 minutes while the search index is built through the embeddings service and the reranker loads (the very first run also downloads about 1 GB). The screen showed only a spinner.

**Change.** `app.py`:

- In Evidence mode, until the search models are loaded, a notice explains that the first question can take a few minutes and later questions take seconds.
- The spinner for that first question reads "Loading search models (first Evidence question only - this can take a few minutes)...".
- Both disappear once the models are loaded.

**Tested.** The notice and spinner appear on the first Evidence question after a restart and disappear afterwards. Later Evidence questions take about 7 seconds.

**Not changed.** Making the first question faster would require saving the search index to disk, which is backend work in `src/retrieval.py` and is left for the team to decide.

---

## Round 2 changes

### 7. Declined requests labelled "Escalated"

**Problem.** A declined prompt injection or dosage request showed "Status: Escalated", which suggests it was passed to a clinician.

**Change.** `app.py` labels these "Declined" and emergencies "Escalated (possible emergency)". The underlying `Answer.status` value is unchanged (`escalated`), so logging and evaluation are not affected.

**Tested.** The ibuprofen dosage request shows "Status: Declined".

### 8. Knowledge and Evidence answers mixed in one chat

**Problem.** Switching modes kept the earlier conversation on screen with no indication of which mode produced each answer.

**Change.** `app.py` shows a small "🧠 Knowledge" or "📚 Evidence" tag at the top of each answer.

**Tested.** After switching to Evidence mode, the earlier dosage response is tagged Knowledge and the asthma answer is tagged Evidence.

### 9. Developer details mixed into every answer

**Problem.** Status, request ID, processing time and an "Open Langfuse trace" link appeared under every answer. The Langfuse link only works for team members with access to the team's Langfuse account.

**Change.** `app.py` moves these into a collapsed "Details" section under each answer and each triage result. The trace link is labelled "(team Langfuse access required)". Nothing was removed; it is one click away.

**Tested.** Details is collapsed by default and shows status, request ID, processing time and the trace link when opened.

### 10. Sources did not show what the answer drew on

**Problem.** A source title alone does not let a reader check whether the answer reflects the source.

**Change.** `app.py` shows a short passage (up to 350 characters) under each source. The answer records which documents were cited but not which passage, so the passage shown is the part of that document with the most words in common with the answer. Passages that begin mid-sentence are trimmed to start at the next full sentence.

**Tested.** The asthma sources show the matching symptom passages from MedlinePlus.

### 11. "Not answered" replies gave no next step

**Problem.** When no supported answer was found, the reply did not suggest what to do.

**Change.** `app.py` adds a hint under "Not answered" replies: "Try rephrasing with the condition or topic named", plus, in Evidence mode, "or switch to Knowledge mode for a general explanation."

**Tested.** In Evidence mode, "What is the best way to learn the guitar?" returns "the evidence is insufficient" with status Not answered, and the hint appears underneath.

### 12. Triage card showed backend values

**Problem.** The triage card displayed "Confidence: not_validated" and emergency flags as a plain comma-separated list.

**Change.** `app.py`:

- "Confidence: Not clinically validated".
- Emergency flags as a bulleted list, shown above the reason.
- The model summary is headed "Report summary".
- A length-limit note appears if the summary is cut off (the summary limit is 180 tokens).

**Tested.** A synthetic head CT report with an acute subdural hematoma shows: Emergency department, Emergency urgency, "Intracranial bleeding" as a bulleted flag, an accurate report summary, and "Confidence: Not clinically validated". PDF and DOCX upload was not re-tested in this round; the upload code was not changed.

---

## Round 3: design pass

The look was explored on a Claude Design canvas ("HealthFlow 360 - UI Design Pass", owned by Gautam; ask him for access), then built into Streamlit. The direction is called "Clinical Calm": quiet, light and clinical, with one teal accent and colour used only where it carries meaning.

### 13. New visual design

**Change.**

- **New `.streamlit/config.toml` theme.**
  - Light base, background #F5F6F4, white sidebar, text #17202A, teal accent #0E7468.
  - Inter for text and Space Grotesk for headings, loaded from Google Fonts.
  - Softer borders and 10 px corners.
  - Streamlit's Deploy toolbar hidden (`toolbarMode = "minimal"`).
  - The project theme overrides any personal theme a teammate has set in `~/.streamlit`.
- **Sidebar.**
  - A logo mark and name at the top.
  - The three modes, each with a one-line description and a line icon (Material icons, no emoji).
  - "New conversation", then a "How this mode works" note that changes with the mode.
- **Page.**
  - The large yellow disclaimer box becomes a slim neutral strip.
  - Each mode has its own title and one-line description.
  - Content is held to a 1,000 px column, and the chat input matches it.
- **Answers.**
  - Each answer carries a Knowledge or Evidence tag and a source count.
  - Citations appear as small numbered chips in the text.
  - Sources sit in one card, each with number, title link, source ID and quoted passage.
  - Details collapses into one line showing the status and time, for example "Details · Answered in 6.9 s".
- **Security.** All model output, user input and corpus text is HTML-escaped before it is shown, so an answer cannot inject markup into the page.

### 14. Redesigned message states

**Change.**

- **Emergency:** a full red block with a warning icon. It is the only solid-red element in the app.
- **Declined request:** a neutral grey box with a lock icon and the refusal text.
- **Error:** an amber box with the request reference. Amber is now used only for problems, which is why the disclaimer became neutral grey.
- **First Evidence question:**
  - A "Preparing the medical library" panel with a spinner replaces the plain spinner.
  - It has no progress bar, because the index build does not report progress and a bar would be made up.
  - The spinner stops for users who have turned off motion in their system settings.

### 15. "Ask in Knowledge mode" button

**Problem.** When Evidence mode cannot answer because the library does not cover a topic (for example Kawasaki disease, which is not in the corpus), the user had to switch modes and retype the question.

**Change.** Unanswered Evidence replies now include an "Ask in Knowledge mode" button. It switches the mode and re-sends the same question for a general, uncited explanation. The original question is stored with each answer to make this possible.

### 16. Two-column Report Triage

**Change.**

- **Layout.** The report input sits on the left (upload, paste box, summary option, "Analyse report") and the result on the right. Before a report is analysed, the right column shows an empty placeholder.
- **Result header coloured by urgency:**
  - red with "Emergency department" for emergencies;
  - neutral grey for routine routing, for example "Primary care / referring clinician";
  - amber when urgency cannot be determined.
- **Result body.** Emergency flags are shown as red tags, or "None found in the report". Below them come "Why", the AI report summary in a shaded box, and a footer with "Not clinically validated" and the limitations text. Request details are collapsed.

### Round 3 testing

Tested in the running app after a restart (theme files are read only at start-up):

| Check | Result |
|---|---|
| Theme, fonts, sidebar and disclaimer strip | Applied. Sidebar fixed at 300 px and the chat input matched to the content column after a first test showed both too narrow or wide. |
| Emergency ("severe chest pain and difficulty breathing") | Red block, "Escalated (possible emergency) in 1.0 s" |
| Declined (ibuprofen dosage) | Grey box with lock icon |
| Knowledge ("What is hypertension?") | Complete answer, Knowledge tag, collapsed Details, 6.9 s |
| First Evidence question after restart | "Preparing the medical library" panel with spinner; answer after 197 s |
| Not answered ("What causes Kawasaki disease?") | "Not answered" with "Ask in Knowledge mode" button |
| "Ask in Knowledge mode" button | Switched to Knowledge, re-sent the question, answered |
| Evidence ("What are the symptoms of asthma?") | Citation chips 1 and 2, source card with linked titles and passages, 9.2 s |
| Triage, head CT with subdural hematoma | Red "Emergency department" header, "Intracranial bleeding" tag, accurate summary |
| Triage, normal chest X-ray (release smoke test 5) | Grey "Primary care / referring clinician" header, "None found in the report", accurate summary, 2.6 s |

Not re-tested in this round: PDF and DOCX upload (upload code unchanged) and phone-width layout.

## Testing approach

All changes were tested in the running app (Streamlit on localhost, Python 3.11, hosted Qwen3-8B), using the five smoke tests from the v2.0 release email plus a check for each change. Test questions used synthetic inputs only.

**Note for developers.** While `app.py` or any `src/` file is edited with the app running, Streamlit reloads the project code, which discards the loaded search index. The next Evidence question then takes about 3 minutes again. This does not happen for normal users, only during development.

## Design items now addressed

The items previously left for the design pass were handled in Round 3:

- colour and theme;
- the size of the disclaimer at the top of the page;
- the technical "Model" and "Pipeline" text in the sidebar;
- chat icons;
- the layout of the triage card;
- mobile layout.
