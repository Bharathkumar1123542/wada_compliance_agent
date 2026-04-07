# WADA Compliance Agent

A production-ready, multi-agent AI system that checks sports supplements against WADA (World Anti-Doping Agency) regulations using **Google ADK** + **Gemini 2.0 Flash**.

---

## Architecture

```
POST /analyze-supplement
        │
        ▼
┌─────────────────────────────────┐
│   Agent 5: Orchestrator         │  ← Adaptive ReAct controller
│   (LoopAgent + SequentialAgent) │    Decides, retries, escalates
└──────────┬──────────────────────┘
           │
     ┌─────┴──────┐
     │  iteration │  (max 3 loops, retries on low confidence)
     └─────┬──────┘
           │
    ┌──────┼──────────────┐
    ▼      ▼              ▼
Agent 1  Agent 2       Agent 3
Extract  WADA Know.   Compliance
   │        │              │
 OCR     fetch()        evaluate
Scrape   embed()         retrace
Parse    search()       decide
    └──────┬──────────────┘
           ▼
       Agent 4
     Explanation
    (Gemini output)
           │
           ▼
   AnalyzeResponse JSON
```

### ReAct Pattern

Every agent follows `Thought → Action → Observation → Final Answer`:

```
THOUGHT: I received a text input with ingredient data.
ACTION:  text_parser_tool("caffeine 200mg, stanozolol 50mg")
OBSERVE: Extracted 2 ingredients: ["caffeine", "stanozolol"]
THOUGHT: Now check against WADA list.
ACTION:  fetch_wada_list_tool() → keyword_match_tool("stanozolol")
OBSERVE: Exact match found: S1 - Anabolic agents, prohibited_in=all
THOUGHT: Confidence is 1.0, no re-verification needed.
ANSWER:  risk_level=PROHIBITED, flagged=["stanozolol"]
```

---

## Project Structure

```
wada_compliance_agent/
├── main.py                   # Entrypoint (ADK web UI + FastAPI)
├── schemas.py                # Pydantic models (shared I/O contracts)
├── tools.py                  # All tool implementations
├── pyproject.toml            # Dependencies
├── .env.example              # Config template
├── Dockerfile                # Multi-stage production image
├── deploy.sh                 # Cloud Run deployment script
├── agents/
│   ├── extraction_agent.py   # Agent 1: OCR / scrape / parse
│   ├── knowledge_agent.py    # Agent 2: Dynamic WADA fetch + search
│   ├── compliance_agent.py   # Agent 3: Risk decision + reasoning trace
│   ├── explanation_agent.py  # Agent 4: Athlete-friendly Gemini output
│   └── orchestrator.py       # Agent 5: Adaptive ReAct controller
├── api/
│   └── app.py                # FastAPI: /analyze-supplement, /analyze-batch
└── tests/
    └── test_agents.py        # 30+ unit + integration tests
```

---

## Setup (Local)

### 1. Prerequisites

- Python 3.11+
- Google Cloud account + project
- Gemini API key ([Get one here](https://aistudio.google.com/))

### 2. Install dependencies

```bash
cd wada_compliance_agent
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Run locally

```bash
# FastAPI server (production mode)
python main.py

# ADK web UI (dev/debug mode)
python main.py --adk
```

API docs available at: `http://localhost:8080/docs`

### 5. Test a supplement

```bash
curl -X POST http://localhost:8080/analyze-supplement \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "data": "caffeine 200mg, beta-alanine 3g, stanozolol 50mg",
    "athlete_sport": "athletics"
  }'
```

---

## API Reference

### POST /analyze-supplement

**Request:**
```json
{
  "input_type": "text | image | url",
  "data": "raw text, base64 image, or product URL",
  "athlete_sport": "optional sport name"
}
```

**Response:**
```json
{
  "risk_level": "SAFE | CAUTION | PROHIBITED",
  "ingredients": ["caffeine", "stanozolol"],
  "flagged": ["stanozolol"],
  "confidence": 0.97,
  "explanation": "This supplement contains stanozolol...",
  "athlete_advice": "Do not use this supplement...",
  "trace": [{"step": "[12:01:00] THOUGHT: evaluating stanozolol"}],
  "wada_list_version": "2024",
  "disclaimer": "Always verify with your national anti-doping organization..."
}
```

### POST /analyze-batch

Accepts up to 10 items in parallel. Same schema per item.

---

## Cloud Run Deployment

### Quick deploy

```bash
chmod +x deploy.sh
./deploy.sh --project YOUR_PROJECT_ID --region us-central1
```

### Manual steps

```bash
# 1. Set project
gcloud config set project YOUR_PROJECT_ID

# 2. Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/wada-compliance-agent .

# 3. Store API key
echo -n "YOUR_GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=-

# 4. Deploy
gcloud run deploy wada-compliance-agent \
  --image gcr.io/YOUR_PROJECT_ID/wada-compliance-agent \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Gemini API key |
| `GOOGLE_CLOUD_PROJECT` | No | — | GCP project ID |
| `WADA_CACHE_TTL_HOURS` | No | 6 | WADA list cache duration |
| `PORT` | No | 8080 | Server port |
| `LOG_LEVEL` | No | INFO | Logging level |

---

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

Tests cover:
- Tool functions (OCR, scraper, parser, WADA fetch, keyword, semantic, explanation)
- Each agent independently
- Orchestrator end-to-end
- FastAPI endpoints
- Edge cases: hidden ingredients, misleading labels, incomplete pages

---

## Key Design Decisions

**No hardcoded WADA list** — Agent 2 dynamically fetches from `wada-ama.org` on every request (with a configurable in-memory TTL cache). Falls back to a minimal baseline only if the network is unreachable.

**Hybrid retrieval** — Keyword matching is fast and deterministic. Semantic search with Gemini embeddings catches aliases and near-matches (e.g. "ma huang" → ephedrine).

**Adaptive orchestration** — The orchestrator re-runs knowledge checks on flagged ingredients when compliance confidence is below threshold. This is not a fixed pipeline.

**Stateless backend** — All state lives in the request/response cycle. No database required for core functionality. Compatible with Cloud Run's auto-scaling and zero-instance scaling.

**Reasoning traces** — Every compliance decision includes a full `Thought → Action → Observation` trace in the API response, making the system auditable and explainable.
