# 🚀 Enterprise Multi-Agent Research System (Production Platform)

<p align="center">
  <img src="https://img.shields.io/badge/Multi--LLM-Groq%20%7C%20Gemini%20%7C%20OpenAI-orange?style=for-the-badge" alt="Multi-LLM" />
  <img src="https://img.shields.io/badge/LangGraph-0.2.0-blue?style=for-the-badge&logo=python" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LangChain-0.3.0-green?style=for-the-badge" alt="LangChain" />
  <img src="https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-dc2626?style=for-the-badge" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Evaluation-RAGAS-purple?style=for-the-badge" alt="RAGAS" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" alt="Docker" />
</p>

An enterprise-grade, distributed **Agentic AI & Retrieval-Augmented Generation (RAG)** platform designed for production deployments. Built with **LangGraph**, **FastAPI**, **Qdrant**, **RAGAS**, and **Celery**, it supports multiple LLM providers (**Groq**, **Google Gemini**, and **OpenAI**) with zero vendor lock-in. It autonomously conducts multi-source web research, performs hybrid semantic & lexical retrieval, executes iterative quality-criticism loops, and synthesizes publication-ready intelligence reports.

---

## 🌟 Multi-LLM Independence (Groq, Gemini, OpenAI)

The platform features an intelligent, multi-provider model factory (`src/utils/llm_factory.py`) that eliminates vendor lock-in. You don't have to choose or configure models in the UI; simply provide your preferred API key in `.env`:

| Provider | Recommended Model | Cost / Tier | Best For |
|---|---|---|---|
| **⚡ Groq** | `llama-3.3-70b-versatile` | **Free Tier** | Lightning-fast inference (500+ tok/s), open-source sovereignty |
| **✨ Google Gemini** | `gemini-1.5-flash` | **Free Tier** | Large context window, multimodal capabilities, Google AI Studio |
| **🧠 OpenAI** | `gpt-4o-mini` / `gpt-4o` | Pay-as-you-go | Industry standard, strong JSON reasoning |

*When `LLM_PROVIDER=auto` is set in `.env`, the system automatically detects which API key you've entered and boots up with that provider!*

---

## 🏛️ System Architecture

```
                                  USER / CLIENT
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              Streamlit Frontend (UI)         FastAPI REST & WebSockets
                (Port: 8501)                   (Port: 8000)
                         │                             │
                         │                   ┌─────────┴─────────┐
                         │                   │ JWT Authentication│
                         │                   │ Redis Rate Limiter│
                         │                   │ Structlog Tracker │
                         │                   └─────────┬─────────┘
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                     Distributed Task Broker (Celery + Redis)
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
         Worker Instance #1                              Worker Instance #N
   ┌────────────────────────────────────────────────────────────────────────┐
   │                     LangGraph Stateful Agent Pipeline                  │
   │                                                                        │
   │   START                                                                │
   │     │                                                                  │
   │     ▼                                                                  │
   │  [01: SearchAgent]       ── Tavily Search (Web + News)                 │
   │     │                                                                  │
   │     ▼                                                                  │
   │  [02: ReaderAgent]       ── Multi-Strategy Concurrent Scraper          │
   │     │                       (Trafilatura + Readability + BS4)          │
   │     │                                                                  │
   │     ▼                                                                  │
   │  [03: RetrievalAgent]    ── Smart Chunking (Recursive/Semantic)        │
   │     │                       Qdrant Vector DB (Dense Embeddings)        │
   │     │                       BM25 Lexical Index (Sparse Search)         │
   │     │                       Reciprocal Rank Fusion (RRF)               │
   │     │                       Cohere Cross-Encoder Reranker              │
   │     │                                                                  │
   │     ▼                                                                  │
   │  [04: WriterAgent]       ── Grounded Synthesis & Draft Generation      │
   │     │                                                                  │
   │     ▼                                                                  │
   │  [05: CriticAgent]       ── Objective Evaluation & Rubric Scoring      │
   │     │                                                                  │
   │     ├── (Score < 7.0 & Iter < Max) ──► Re-route to [WriterAgent]       │
   │     │                                                                  │
   │     ▼ (Approved or Max Iterations Reached)                             │
   │  [06: SynthesisAgent]    ── Markdown Polish & Source Linking           │
   │     │                                                                  │
   │    END                                                                 │
   └────────────────────────────────────────────────────────────────────────┘
          │                                 │                    │
          ▼                                 ▼                    ▼
   [Qdrant Cluster]                 [PostgreSQL Store]     [LangSmith Tracing]
   - Scalar Quantization            - Runs History         - Agent Traces
   - Tenant Filtering               - Evaluation Records   - Step Latencies
   - HNSW Cosine Index              - User Accounts        - Token Consumption
```

---

## 🎯 Why This Project Gets You Hired (Fresher / AI Engineer JD Match)

Recruiters and hiring managers review hundreds of simple "OpenAI wrapper" projects. This repository demonstrates **production software engineering for AI systems**:

1. **Autonomous Self-Correction (LangGraph State Machine):**
   - Implements a cyclic agent loop (`Writer` ⇆ `Critic`).
   - If the `CriticAgent` detects weak citations, missing context, or low scores ($< 7.0$), the graph automatically re-routes back to revision before publishing.
2. **Hybrid RAG (Dense + BM25 + Reciprocal Rank Fusion):**
   - Doesn't rely on naive vector search alone; combines lexical keyword matching (BM25) with semantic embeddings via Reciprocal Rank Fusion (RRF: $\sum \frac{1}{k + r}$).
3. **Rust-Powered Vector Storage (Qdrant):**
   - High-throughput vector search with Scalar Quantization ($int8$) and automatic in-memory fallback.
4. **Quantitative Evaluation Framework (RAGAS):**
   - Validates generated reports on **Faithfulness** (hallucination prevention), **Answer Relevancy**, **Context Precision**, and **Source Coverage**.
5. **Decoupled Distributed Architecture:**
   - Stateless FastAPI REST API + background Celery workers + Redis queue + PostgreSQL persistence.
6. **Zero-Lock-in Multi-LLM Factory:**
   - Seamlessly swaps between Groq (free/fast), Google Gemini, and OpenAI.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Configure `.env`
Copy `.env.example` to `.env` (or open the existing `.env`):
```bash
cp .env.example .env
```
Open `.env` and fill in **ANY ONE** LLM key:
```ini
# Option A: Groq (FREE: https://console.groq.com)
GROQ_API_KEY=gsk_your_groq_key_here

# Option B: Google Gemini (FREE: https://aistudio.google.com)
GEMINI_API_KEY=AIzaSy_your_gemini_key_here

# Option C: OpenAI (https://platform.openai.com)
OPENAI_API_KEY=sk-proj-your_openai_key_here

# Tavily Search API (FREE 1,000 queries: https://tavily.com)
TAVILY_API_KEY=tvly-your_tavily_key_here
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
Launch the interactive Streamlit research assistant:
```bash
streamlit run app.py
```
*The app automatically detects your active LLM provider from `.env` and executes the full multi-agent pipeline!*

---

## 🐳 Docker Deployment

To launch the full containerized stack (FastAPI backend + Celery worker + Streamlit frontend + Qdrant + Redis + Postgres):

```bash
docker-compose up --build
```

Access the services:
- **Streamlit Interface:** `http://localhost:8501`
- **FastAPI Documentation:** `http://localhost:8000/api/docs`
- **Qdrant Console:** `http://localhost:6333/dashboard`

---

## ☁️ Cloud Deployment Guide (Render / Railway / Streamlit Cloud)

### Deploying to Streamlit Community Cloud (Easiest & Free)
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repo and set main file path to `app.py`.
4. In **Advanced Settings → Secrets**, paste your `.env` variables (`GROQ_API_KEY`, `TAVILY_API_KEY`, etc.).
5. Click **Deploy**!

### Deploying to Render / Railway / Cloud VM
Use the included Dockerfiles:
- **Web Service:** `docker/Dockerfile.api` (FastAPI) or `docker/Dockerfile.frontend` (Streamlit)
- Set environment variables in the cloud dashboard.

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Register new user account |
| `POST` | `/api/v1/auth/login` | Authenticate and obtain JWT access token |
| `POST` | `/api/v1/research/run` | Submit asynchronous research pipeline job |
| `GET` | `/api/v1/research/status/{run_id}` | Poll execution progress and current state |
| `GET` | `/api/v1/research/result/{run_id}` | Retrieve synthesized report and evaluation metrics |
| `GET` | `/api/v1/research/history` | List paginated history of research runs |
| `WS` | `/api/v1/research/stream/{run_id}` | Real-time WebSocket event stream |
| `GET` | `/api/v1/health` | Service liveness probe |
| `GET` | `/api/v1/health/detailed` | Deep health check (DB, Redis, Qdrant) |

---

## 🧪 Testing

Run unit and integration tests with pytest:
```bash
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
