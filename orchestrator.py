"""
agents/ — Multi-agent package for WADA Compliance System

This package contains five ReAct-style agents that together form the
WADA compliance pipeline:

  Agent 1 — ExtractionAgent    (extraction_agent.py)
      Accepts image (OCR), URL (web scraper), or raw text.
      Normalizes ingredient names using a synonym map.
      Returns ExtractionOutput with confidence scores.

  Agent 2 — KnowledgeAgent     (knowledge_agent.py)
      Dynamically fetches the live WADA prohibited list.
      Runs hybrid retrieval: keyword match → semantic search.
      Returns KnowledgeOutput with WADAMatch objects.

  Agent 3 — ComplianceAgent    (compliance_agent.py)
      Evaluates each ingredient + WADA match.
      Re-verifies borderline cases (confidence 0.65–0.80).
      Returns ComplianceOutput with risk_level + reasoning_trace.

  Agent 4 — ExplanationAgent   (explanation_agent.py)
      Sends all findings to Gemini 2.0 Flash.
      Returns athlete-friendly summary, advice, and flagged details.

  Agent 5 — Orchestrator       (orchestrator.py)
      Adaptive ReAct controller (not a fixed pipeline).
      Retries extraction on low confidence.
      Re-runs knowledge check if compliance confidence is too low.
      Assembles final AnalyzeResponse.

──────────────────────────────────────────────────────────────────────
Import paths
──────────────────────────────────────────────────────────────────────
Direct Python APIs (used by orchestrator, FastAPI, and tests):

    from agents import run_extraction
    from agents import run_knowledge_check
    from agents import run_compliance_check
    from agents import run_explanation
    from agents import run_orchestrator

ADK agent factory functions (used by ADK web runner in main.py --adk):

    from agents import create_extraction_agent
    from agents import create_knowledge_agent
    from agents import create_compliance_agent
    from agents import create_explanation_agent
    from agents import create_orchestrator_agent
"""

from agents.extraction_agent import run_extraction, create_extraction_agent
from agents.knowledge_agent import run_knowledge_check, create_knowledge_agent
from agents.compliance_agent import run_compliance_check, create_compliance_agent
from agents.explanation_agent import run_explanation, create_explanation_agent
from agents.orchestrator import run_orchestrator, create_orchestrator_agent

__all__ = [
    # ── Direct async Python APIs ──────────────────────────────────────────
    # Used by: orchestrator.py, api/app.py, tests/test_agents.py
    "run_extraction",       # ExtractionInput  → ExtractionOutput
    "run_knowledge_check",  # list[str]         → KnowledgeOutput
    "run_compliance_check", # list[str]+matches → ComplianceOutput
    "run_explanation",      # all outputs       → ExplanationOutput
    "run_orchestrator",     # AnalyzeRequest    → AnalyzeResponse

    # ── ADK LlmAgent / LoopAgent factories ───────────────────────────────
    # Used by: main.py --adk  (ADK web UI dev mode)
    "create_extraction_agent",
    "create_knowledge_agent",
    "create_compliance_agent",
    "create_explanation_agent",
    "create_orchestrator_agent",
]
