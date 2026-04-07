"""
Agent 5 — Orchestrator (Adaptive ReAct Controller)

This is NOT a simple pipeline. It:
- Decides which agents to call based on runtime state
- Retries extraction when confidence is too low
- Escalates to re-verification when compliance is ambiguous
- Handles partial failures gracefully
- Produces a unified AnalyzeResponse

ReAct loop:
  Thought  → assess current state, decide next agent
  Action   → call agent
  Observe  → evaluate output quality
  Loop     → continue reasoning until confident or max iterations hit
  Answer   → assemble final AnalyzeResponse
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import FunctionTool

from schemas import (
    AnalyzeRequest, AnalyzeResponse, ExtractionInput,
    RiskLevel, InputType,
    ExtractionOutput, KnowledgeOutput, ComplianceOutput, ExplanationOutput,
)
from agents.extraction_agent import run_extraction
from agents.knowledge_agent import run_knowledge_check
from agents.compliance_agent import run_compliance_check
from agents.explanation_agent import run_explanation

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 3
_MIN_EXTRACTION_CONFIDENCE = 0.50
_MIN_COMPLIANCE_CONFIDENCE = 0.65


# ── ADK Agent definitions (for ADK web runner) ────────────────────────────────

def create_orchestrator_agent() -> LoopAgent:
    """
    Compose all sub-agents into an ADK LoopAgent orchestrator.
    The LoopAgent runs the inner SequentialAgent in a reasoning loop,
    with an EscalationChecker deciding when to stop.
    """
    from agents.extraction_agent import create_extraction_agent
    from agents.knowledge_agent import create_knowledge_agent
    from agents.compliance_agent import create_compliance_agent
    from agents.explanation_agent import create_explanation_agent

    pipeline = SequentialAgent(
        name="wada_pipeline",
        sub_agents=[
            create_extraction_agent(),
            create_knowledge_agent(),
            create_compliance_agent(),
            create_explanation_agent(),
        ],
    )

    return LoopAgent(
        name="wada_orchestrator",
        sub_agents=[pipeline],
        max_iterations=_MAX_ITERATIONS,
    )


# ── Direct Python API (used by FastAPI) ───────────────────────────────────────

class OrchestratorState:
    """Tracks reasoning state across the ReAct loop."""

    def __init__(self):
        self.iteration = 0
        self.extraction: Optional[ExtractionOutput] = None
        self.knowledge: Optional[KnowledgeOutput] = None
        self.compliance: Optional[ComplianceOutput] = None
        self.explanation: Optional[ExplanationOutput] = None
        self.errors: list[str] = []
        self.log: list[str] = []

    def record(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        logger.info(entry)


async def run_orchestrator(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Main entrypoint. Orchestrates all 4 agents with adaptive ReAct reasoning.
    """
    state = OrchestratorState()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1 — EXTRACTION (with retry on low confidence)
    # ──────────────────────────────────────────────────────────────────────
    while state.iteration < _MAX_ITERATIONS:
        state.iteration += 1
        state.record(
            f"THOUGHT [iter={state.iteration}]: Starting extraction for input_type={request.input_type}"
        )

        try:
            state.record("ACTION: calling run_extraction")
            extraction = await run_extraction(
                ExtractionInput(input_type=request.input_type, data=request.data)
            )
            state.record(
                f"OBSERVATION: extracted {len(extraction.ingredients)} ingredients, "
                f"confidence={extraction.confidence_scores[0] if extraction.confidence_scores else 0:.2f}"
            )
            state.extraction = extraction
        except Exception as e:
            state.errors.append(f"Extraction error (iter {state.iteration}): {e}")
            state.record(f"OBSERVATION: extraction failed — {e}")
            if state.iteration >= _MAX_ITERATIONS:
                break
            await asyncio.sleep(1)
            continue

        # Check if we got meaningful results
        avg_conf = (
            sum(extraction.confidence_scores) / len(extraction.confidence_scores)
            if extraction.confidence_scores else 0.0
        )

        if extraction.ingredients and avg_conf >= _MIN_EXTRACTION_CONFIDENCE:
            state.record(
                f"THOUGHT: Extraction successful (confidence={avg_conf:.2f}). Proceeding."
            )
            break
        elif not extraction.ingredients:
            state.record(
                "THOUGHT: No ingredients extracted. "
                f"{'Retrying.' if state.iteration < _MAX_ITERATIONS else 'Max retries reached.'}"
            )
        else:
            state.record(
                f"THOUGHT: Low confidence ({avg_conf:.2f}) but have {len(extraction.ingredients)} ingredients. "
                "Proceeding with caution."
            )
            break

    # Guard: if extraction completely failed
    if not state.extraction or not state.extraction.ingredients:
        state.record("OBSERVATION: Could not extract any ingredients. Returning early.")
        return AnalyzeResponse(
            risk_level=RiskLevel.SAFE,
            ingredients=[],
            flagged=[],
            confidence=0.0,
            explanation="Could not extract any ingredients from the provided input. Please try again with clearer text or image.",
            athlete_advice="Unable to analyze this supplement. Please check the input and try again.",
            trace=[{"step": e} for e in state.log],
            wada_list_version="",
        )

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2 — WADA KNOWLEDGE CHECK
    # ──────────────────────────────────────────────────────────────────────
    state.record(
        f"THOUGHT: Running WADA knowledge check for {len(state.extraction.ingredients)} ingredients"
    )
    try:
        state.record("ACTION: calling run_knowledge_check")
        state.knowledge = await run_knowledge_check(state.extraction.ingredients)
        state.record(
            f"OBSERVATION: found {len(state.knowledge.matches)} WADA matches "
            f"(list version={state.knowledge.wada_list_version})"
        )
    except Exception as e:
        state.errors.append(f"Knowledge check error: {e}")
        state.record(f"OBSERVATION: knowledge check failed — {e}")
        # Non-fatal: proceed with empty matches
        from schemas import KnowledgeOutput
        state.knowledge = KnowledgeOutput(matches=[], wada_list_version="error", retrieval_timestamp="")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 3 — COMPLIANCE DECISION
    # ──────────────────────────────────────────────────────────────────────
    state.record("THOUGHT: Running compliance decision engine")
    try:
        state.record("ACTION: calling run_compliance_check")
        state.compliance = await run_compliance_check(
            ingredients=state.extraction.ingredients,
            wada_matches=state.knowledge.matches,
        )
        state.record(
            f"OBSERVATION: risk_level={state.compliance.risk_level}, "
            f"flagged={state.compliance.flagged_ingredients}, "
            f"confidence={state.compliance.confidence:.2f}"
        )
    except Exception as e:
        state.errors.append(f"Compliance check error: {e}")
        state.record(f"OBSERVATION: compliance check failed — {e}")
        from schemas import ComplianceOutput
        state.compliance = ComplianceOutput(
            risk_level=RiskLevel.CAUTION,
            flagged_ingredients=[],
            confidence=0.0,
            reasoning_trace=[],
            safe_ingredients=state.extraction.ingredients,
        )

    # ── Adaptive: if compliance confidence is too low, re-run knowledge check ──
    if state.compliance.confidence < _MIN_COMPLIANCE_CONFIDENCE and state.compliance.flagged_ingredients:
        state.record(
            f"THOUGHT: Compliance confidence ({state.compliance.confidence:.2f}) is low. "
            "Re-running knowledge check for flagged ingredients only."
        )
        try:
            state.record("ACTION: re-running knowledge check on flagged ingredients")
            enhanced_knowledge = await run_knowledge_check(state.compliance.flagged_ingredients)
            # Merge with existing matches
            existing_non_flagged = [
                m for m in state.knowledge.matches
                if m.ingredient not in state.compliance.flagged_ingredients
            ]
            state.knowledge.matches = existing_non_flagged + enhanced_knowledge.matches
            state.record(
                f"OBSERVATION: re-verification complete, total matches={len(state.knowledge.matches)}"
            )

            # Re-run compliance with enhanced knowledge
            state.record("ACTION: re-running compliance check with enhanced knowledge")
            state.compliance = await run_compliance_check(
                ingredients=state.extraction.ingredients,
                wada_matches=state.knowledge.matches,
            )
            state.record(
                f"OBSERVATION: revised risk_level={state.compliance.risk_level}, "
                f"confidence={state.compliance.confidence:.2f}"
            )
        except Exception as e:
            state.errors.append(f"Re-verification error: {e}")
            state.record(f"OBSERVATION: re-verification failed — {e}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 4 — EXPLANATION
    # ──────────────────────────────────────────────────────────────────────
    state.record("THOUGHT: Generating athlete-friendly explanation")
    try:
        state.record("ACTION: calling run_explanation")
        state.explanation = await run_explanation(
            extraction=state.extraction,
            knowledge=state.knowledge,
            compliance=state.compliance,
            athlete_sport=request.athlete_sport,
        )
        state.record("OBSERVATION: explanation generated")
    except Exception as e:
        state.errors.append(f"Explanation error: {e}")
        state.record(f"OBSERVATION: explanation failed — {e}")
        state.explanation = ExplanationOutput(
            summary=f"Risk level: {state.compliance.risk_level.value}",
            athlete_advice="Please consult your national anti-doping authority.",
            flagged_details=[],
        )

    # ──────────────────────────────────────────────────────────────────────
    # FINAL ANSWER — assemble AnalyzeResponse
    # ──────────────────────────────────────────────────────────────────────
    state.record(
        f"FINAL ANSWER: risk={state.compliance.risk_level}, "
        f"ingredients={len(state.extraction.ingredients)}, "
        f"flagged={len(state.compliance.flagged_ingredients)}"
    )

    trace_data = [{"step": entry} for entry in state.log]

    return AnalyzeResponse(
        risk_level=state.compliance.risk_level,
        ingredients=state.extraction.ingredients,
        flagged=state.compliance.flagged_ingredients,
        confidence=state.compliance.confidence,
        explanation=state.explanation.summary,
        athlete_advice=state.explanation.athlete_advice,
        trace=trace_data,
        wada_list_version=state.knowledge.wada_list_version,
    )
