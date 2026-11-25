# HealthFlow360

## Agentic Medical RAG Assistant
Track B — Capstone Project | Release v2.0

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
- Streamlit interface
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

| Metric | Baseline | Final |
|---|---:|---:|
| Structural task completion | 54/70 | 70/70 |
| Emergency detection sensitivity | 66.67% | 100% |
| Emergency detection specificity | 75% | 100% |
| Injection execution blocked | 0/10 | 10/10 |
| Logical generation calls | 84 | 71 |
| Mean latency | 7.85 s | 5.55 s |

These results reflect retrospective benchmark testing.
Structural completion is not equivalent to clinical accuracy.
Independent clinical validation has not been performed.

### Repository Structure

    HealthFlow360-Final/
    ├── app.py
    ├── src/
    ├── scripts/
    ├── data/
    │   └── evaluation/
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    ├── .gitignore
    ├── .env.example
    └── README.md

### Local Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

Configure authorized inference and embedding endpoints,
API credentials and trusted CA certificates in .env.

Do not commit credentials or private configuration.

### Run Locally

    streamlit run app.py

### Docker Setup

    docker build -t healthflow360:2.0 .

With valid environment configuration and trusted certificates:

    docker run --rm -p 8501:8501 \
      --env-file .env \
      -v "$PWD/certs:/app/certs:ro" \
      healthflow360:2.0

Open http://localhost:8501

Docker build and local runtime validation completed successfully.

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

The application was developed through 18 iterations covering
API integration, observability, safety, report processing,
corpus remediation, prompt engineering, retrieval optimization
and end-to-end evaluation.

Detailed development history, evaluation methodology and
failure taxonomy are maintained in the final evaluation report.

### Limitations

- Educational use only.
- No independent clinical validation.
- No medical image interpretation or OCR.
- External inference and embedding services are required.
- Cold-start latency depends on corpus initialization.
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
