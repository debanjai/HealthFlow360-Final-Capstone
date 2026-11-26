# HealthFlow360

## Agentic Medical RAG Assistant
Track B — Capstone Project | v2.0 release with v2.1 UI and evaluation updates

### Overview

HealthFlow360 is an educational medical AI assistant combining
retrieval-augmented generation, agent orchestration, safety
guardrails, radiology report triage and observability.

The system is intended for educational purposes only.
It does not diagnose, prescribe or replace clinical judgment.

### Core Features

- Medical knowledge question answering
- Evidence-grounded question answering with citations
- Hybrid BM25 and dense retrieval
- Cross-encoder reranking
- Planner-executor-critic orchestration
- Session-based conversational state
- Emergency escalation and clinical review safeguards
- Prompt injection detection and blocking
- PDF and DOCX radiology report processing
- Streamlit interface ("Clinical Calm" theme) with numbered citation chips,
  source cards showing title, link and passage, and colour-coded safety states
- Langfuse observability
- Docker deployment configuration

### Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Generation | Qwen3-8B |
| Embeddings | BGE-small-en-v1.5 |
| Sparse retrieval | BM25 |
| Reranker | BGE-reranker-base |
| Frontend | Streamlit |
| Observability | Langfuse |
| Containerization | Docker |

### Architecture

User Input
  -> Safety Assessment
  -> Planner
  -> Executor
  -> Retrieval / Generation / Report Triage
  -> Critic
  -> Response

Emergency and prohibited requests may bypass generation.

### Medical Corpus

- Source: MedlinePlus, US National Library of Medicine
- Parent documents: 750
- Production chunks: 2,341
- Chunk size: 200 tokens
- Chunk overlap: 50 tokens
- Retrieval: Hybrid search with cross-encoder reranking

Source: https://medlineplus.gov/xml.html

### Evaluation Results

#### Prompt Engineering

| Strategy | Accuracy | Macro F1 |
|---|---:|---:|
| Zero-shot | 50.00% | 0.4543 |
| Few-shot | 42.22% | 0.3591 |
| Decomposition | 45.56% | 0.4270 |

Evaluated on a balanced retrospective PubMedQA subset
of 90 questions.

#### Retrieval

| Metric | Final Configuration |
|---|---:|
| Recall@1 | 80% |
| Recall@3 | 100% |
| Recall@5 | 100% |
| MRR@5 | 0.8917 |

Evaluated on 20 designated-reference retrieval questions.

#### End-to-End Evaluation

Final fresh-clone regression: 26 November 2026, application baseline commit `a50edc9`.

| Metric | Pre-remediation | v2.0 | Final v2.1 |
|---|---:|---:|---:|
| Structural task completion | 54/70 | 70/70 | 70/70 |
| Emergency detection sensitivity | 66.67% | 100% | 100% |
| Emergency detection specificity | 75% | 100% | 100% |
| Injection detection | 8/10 | 10/10 | 10/10 |
| Injection execution blocked | 0/10 | 10/10 | 10/10 |
| Valid citations | — | 20/20 | 20/20 |
| Designated reference cited | — | 19/20 | 19/20 |
| Logical generation calls | 84 | 71 | 71 |
| Weighted mean latency | 7.855 s | 5.547 s | 6.074 s |

| Category | Baseline passes | v2.0 passes | v2.1 passes | v2.1 mean (s) | v2.1 p95 (s) |
|---|---:|---:|---:|---:|---:|
| Knowledge | 20/20 | 20/20 | 20/20 | 5.192 | 6.099 |
| Evidence | 20/20 | 20/20 | 20/20 | 14.756 | 15.851 |
| Report triage | 9/10 | 10/10 | 10/10 | 1.530 | 1.794 |
| Safety | 5/10 | 10/10 | 10/10 | 1.087 | 5.444 |
| Adversarial | 0/10 | 10/10 | 10/10 | 0.002 | 0.003 |

Evidence-mode mean latency increased from 10.087 s to 14.756 s relative to v2.0.
Runtime and API conditions were uncontrolled; the cause is not established.

#### Targeted Safety Regression and Manual UI Validation

- Targeted regression: 30/30 structural passes (10 report, 10 safety, 10 adversarial).
- Emergency gate: 6 true positives, 4 true negatives, no false positives or false negatives.
- Prompt injection: 10/10 detected and blocked.
- Manual UI validation: five scenarios executed (Knowledge, Evidence, Report Triage, emergency escalation and prompt injection), with 5/5 Langfuse trace coverage. Individual pass outcomes were not separately recorded in the supplied evidence.
- The targeted regression repeats a subset of the 70 E2E cases; these are 100 executions, not 100 independent cases.

These results reflect retrospective benchmark testing on reused cases.
Structural completion is not equivalent to clinical accuracy.
Independent clinical validation has not been performed. The original qualitative review applies to v2.0; fresh predictions have not been qualitatively reviewed.

#### Evaluation Artifacts and Reproduction

- [Final evaluation report v2.2 (DOCX)](reports/evaluation_documentation/HealthFlow360_Evaluation_Report_v2.2.docx)
- [Final E2E results](reports/evaluation/e2e_v2_1/) ([summary](reports/evaluation/e2e_v2_1/summary.json))
- [Targeted safety regression results](reports/evaluation/safety_regression_final_v2_1/) ([summary](reports/evaluation/safety_regression_final_v2_1/summary.json))

Each result directory contains predictions, summary, run configuration and a manual-review CSV. A generated review CSV does not imply completed qualitative scoring.

From the repository root, with the configured environment and production corpus:

```bash
python3 scripts/run_e2e_evaluation.py
python3 scripts/run_safety_regression.py
```

The updated scripts default to timestamped output directories and refuse to overwrite existing directories. An unused output directory can also be supplied as the first argument.

### Repository Structure

    HealthFlow360-Final/
    ├── app.py
    ├── src/
    ├── scripts/
    ├── data/
    │   └── evaluation/
    ├── reports/
    │   ├── evaluation/
    │   │   ├── e2e_v2_1/
    │   │   └── safety_regression_final_v2_1/
    │   └── evaluation_documentation/
    │       └── HealthFlow360_Evaluation_Report_v2.2.docx
    ├── docs/
    │   └── UI_CHANGES.md
    ├── .streamlit/
    │   └── config.toml
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    ├── .gitignore
    ├── .env.example
    └── README.md

### Local Setup

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

Python 3.11 is required. On macOS, plain `python3` may be the system
Python 3.9, which fails on this codebase; install 3.11 with
`brew install python@3.11` if needed.

Configure authorized inference and embedding endpoints,
API credentials and trusted CA certificates in .env.

Do not commit credentials or private configuration.

### Run Locally

    streamlit run app.py

The interface theme is set in `.streamlit/config.toml` and is read only
at start-up: restart Streamlit after pulling theme changes.

### Docker Setup

    docker build -t healthflow360:2.0 .

With valid environment configuration and trusted certificates:

    docker run --rm -p 8501:8501 \
      --env-file .env \
      -v "$PWD/certs:/app/certs:ro" \
      healthflow360:2.0

Open http://localhost:8501

Docker build and local runtime validation completed successfully for the original v2.0 image. The latest UI-integrated application was validated locally from a fresh clone; updated Docker build and runtime validation were deferred.

- Platform: Apple Silicon / Linux ARM64
- Docker image: healthflow360:2.0
- CPU-only PyTorch
- Knowledge mode: Passed
- Evidence mode: Passed (12.02 seconds)
- Emergency safety: Passed (0.01 seconds)
- Radiology report triage: Passed (1.74 seconds)

All six deployment smoke checks passed.
These checks do not constitute independent clinical validation.

### Development History

The application was developed through 19 iterations covering
API integration, observability, safety, report processing,
corpus remediation, prompt engineering, retrieval optimization,
end-to-end evaluation and user interface refinement.

Iteration 19 (user interface) fixed defects found in UI testing and
applied a visual redesign:

- Knowledge answers no longer stop mid-sentence
- Evidence sources show title, link and supporting passage instead of
  raw document IDs, and citation numbers match the sources list
- Emergencies, declined requests, errors and unanswered questions
  each have a distinct visual treatment
- An "Ask in Knowledge mode" action for questions the corpus does not cover
- Two-column Report Triage with an urgency-coloured result
- "Clinical Calm" theme with a slimmer disclaimer, mode descriptions
  and collapsed request details

Intermediate post-UI regression testing retained 70/70 structural completion and 71 logical generation calls, with 5.59 s mean latency (5.55 s before). Each UI change, with test results, is documented in [docs/UI_CHANGES.md](docs/UI_CHANGES.md).

Final fresh-clone validation on 26 November 2026 (application baseline `a50edc9`) confirmed 750 documents and 2,341 chunks, 70/70 E2E structural passes, 30/30 targeted regression passes and five manual UI scenarios with Langfuse traces. Final mean E2E latency was 6.074 s; evidence-mode latency increased relative to v2.0. Evaluation scripts were updated to protect existing result directories.

Detailed development history, evaluation methodology and
failure taxonomy are maintained in the final evaluation report.

### Limitations

- Educational use only.
- No independent clinical validation.
- No medical image interpretation or OCR.
- External inference and embedding services are required.
- Cold-start latency depends on corpus initialization.
- Final evidence-mode latency increased; the cause remains unverified.
- Fresh prediction qualitative review and updated Docker validation remain outstanding.
- Citation validity does not guarantee semantic grounding.
- Session state is process-local.
- Safety rules cannot guarantee detection of every emergency.
- Do not submit identifiable patient information.

### Security and Privacy

API keys, private endpoints, certificates, environment files
and development logs are excluded from version control.

### Dataset Sources

- MedlinePlus: https://medlineplus.gov/
- PubMedQA: https://huggingface.co/datasets/qiaojin/PubMedQA
- Synthetic safety and radiology evaluation cases

Dataset usage remains subject to original source terms.

### Acknowledgements

Developed as a collaborative Track B capstone project.
