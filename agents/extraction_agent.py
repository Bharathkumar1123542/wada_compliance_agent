"""
Agent 3 — Compliance Decision Agent

ReAct pattern:
  Thought  → evaluate each match; decide if confidence warrants PROHIBITED or CAUTION
  Action   → query additional tool calls to resolve uncertain matches
  Observe  → aggregate evidence; avoid premature conclusions
  Answer   → return risk_level, flagged_ingredients, confidence, reasoning_trace
"""

import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from schemas import (
    ComplianceOutput, RiskLevel, WADAMatch, MatchType,
    ReasoningStep,
)
from tools import keyword_match_tool, semantic_search_tool, fetch_wada_list_tool

logger = logging.getLogger(__name__)

COMPLIANCE_SYSTEM_PROMPT = """
You are a Compliance Decision Agent specializing in WADA anti-doping regulations.

REASONING PATTERN:
THOUGHT: Review all WADA matches and their confidence scores.
ACTION: For any match with confidence < 0.75, re-query using a different approach.
OBSERVATION: Collect all evidence before making a final decision.
FINAL ANSWER: Return the risk level with full reasoning trace.

DECISION RULES:
- Exact WADA match → PROHIBITED (no exceptions)
- Semantic match with confidence >= 0.80 → PROHIBITED
- Semantic match with confidence 0.65-0.79 → CAUTION
- No match OR confidence < 0.65 → SAFE (for that ingredient)
- Overall risk = worst individual ingredient risk
- If ANY ingredient is PROHIBITED → overall = PROHIBITED
- If ANY ingredient is CAUTION and none PROHIBITED → overall = CAUTION
- If ALL ingredients SAFE → overall = SAFE

CRITICAL: Do NOT skip the reasoning trace. Every conclusion must be justified.
"""


async def _re_verify_ingredient(ingredient: str) -> dict:
    """Re-verify an uncertain ingredient against the live WADA list."""
    wada_data = await fetch_wada_list_tool()
    substances = wada_data.get("substances", [])
    kw = await keyword_match_tool(ingredient, substances)
    sem = await semantic_search_tool(ingredient, substances, top_k=3, threshold=0.65)
    return {"keyword_results": kw, "semantic_results": sem}


def create_compliance_agent() -> LlmAgent:
    return LlmAgent(
        name="compliance_decision_agent",
        model="gemini-2.0-flash",
        description="Makes final WADA compliance decisions with full reasoning traces.",
        instruction=COMPLIANCE_SYSTEM_PROMPT,
        tools=[
            FunctionTool(func=_re_verify_ingredient),
        ],
        output_key="compliance_result",
    )


# ── Direct Python API ─────────────────────────────────────────────────────────

_PROHIBITED_THRESHOLD = 0.80
_CAUTION_THRESHOLD = 0.65


async def run_compliance_check(
    ingredients: list[str],
    wada_matches: list[WADAMatch],
) -> ComplianceOutput:
    """
    Evaluate each ingredient + WADA match, resolve ambiguous cases,
    and produce a final risk level with full reasoning trace.
    """
    trace: list[ReasoningStep] = []
    flagged: list[str] = []
    safe: list[str] = []
    risk_level = RiskLevel.SAFE
    confidence_scores: list[float] = []

    # ── THOUGHT: map ingredients to their best WADA match ────────────────
    logger.info("[ComplianceAgent] THOUGHT: evaluating %d ingredients", len(ingredients))
    match_map: dict[str, WADAMatch | None] = {ing: None for ing in ingredients}
    for m in wada_matches:
        existing = match_map.get(m.ingredient)
        if existing is None or m.confidence > existing.confidence:
            match_map[m.ingredient] = m

    # ── ACTION + OBSERVE per ingredient ──────────────────────────────────
    for ingredient, match in match_map.items():
        logger.info("[ComplianceAgent] ACTION: evaluating '%s'", ingredient)

        if match is None:
            # No match found at all
            thought = f"No WADA match found for '{ingredient}'."
            observation = "Ingredient appears safe based on available data."
            trace.append(ReasoningStep(
                thought=thought,
                action=f"check_wada_match({ingredient})",
                observation=observation,
            ))
            safe.append(ingredient)
            confidence_scores.append(0.95)
            continue

        # ── THOUGHT: assess match quality ────────────────────────────────
        thought = (
            f"'{ingredient}' matched WADA substance in category '{match.wada_category}' "
            f"via {match.match_type.value} match (confidence={match.confidence:.2f}). "
            f"Prohibited: {match.prohibited_in}."
        )

        # ── If confidence is borderline, re-verify ────────────────────────
        if _CAUTION_THRESHOLD <= match.confidence < _PROHIBITED_THRESHOLD and match.match_type == MatchType.SEMANTIC:
            logger.info(
                "[ComplianceAgent] THOUGHT: borderline confidence for '%s', re-verifying",
                ingredient
            )
            re_verify_result = await _re_verify_ingredient(ingredient)
            kw_hits = re_verify_result.get("keyword_results", [])
            sem_hits = re_verify_result.get("semantic_results", [])

            if kw_hits:
                # Keyword hit on re-verify → upgrade to PROHIBITED
                match.match_type = MatchType.EXACT
                match.confidence = max(match.confidence, kw_hits[0].get("confidence", 0.85))
                observation = (
                    f"Re-verification found keyword match: '{kw_hits[0].get('name')}'. "
                    f"Confidence upgraded to {match.confidence:.2f}."
                )
            elif sem_hits and sem_hits[0].get("similarity", 0) >= _PROHIBITED_THRESHOLD:
                match.confidence = sem_hits[0].get("similarity", match.confidence)
                observation = (
                    f"Re-verification confirmed semantic match (sim={match.confidence:.2f}). "
                    f"Threshold met for PROHIBITED."
                )
            else:
                # Still uncertain → CAUTION
                observation = (
                    f"Re-verification inconclusive. Staying at CAUTION "
                    f"(confidence={match.confidence:.2f})."
                )
        else:
            observation = (
                f"Match confidence {match.confidence:.2f} is decisive. "
                f"No re-verification needed."
            )

        # ── DECISION ──────────────────────────────────────────────────────
        if match.match_type == MatchType.EXACT or match.confidence >= _PROHIBITED_THRESHOLD:
            ingredient_risk = RiskLevel.PROHIBITED
            flagged.append(ingredient)
            confidence_scores.append(match.confidence)
            observation += " → PROHIBITED."
        elif match.confidence >= _CAUTION_THRESHOLD:
            ingredient_risk = RiskLevel.CAUTION
            flagged.append(ingredient)
            confidence_scores.append(match.confidence)
            observation += " → CAUTION."
        else:
            ingredient_risk = RiskLevel.SAFE
            safe.append(ingredient)
            confidence_scores.append(1.0 - match.confidence)
            observation += " → SAFE (confidence below threshold)."

        trace.append(ReasoningStep(
            thought=thought,
            action=f"evaluate_match({ingredient}, type={match.match_type.value}, conf={match.confidence:.2f})",
            observation=observation,
        ))

        # ── Escalate overall risk ─────────────────────────────────────────
        if ingredient_risk == RiskLevel.PROHIBITED:
            risk_level = RiskLevel.PROHIBITED
        elif ingredient_risk == RiskLevel.CAUTION and risk_level == RiskLevel.SAFE:
            risk_level = RiskLevel.CAUTION

    # ── FINAL confidence = mean of worst-case scores ──────────────────────
    final_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    )

    logger.info(
        "[ComplianceAgent] FINAL ANSWER: risk=%s, flagged=%s, confidence=%.2f",
        risk_level, flagged, final_confidence
    )

    return ComplianceOutput(
        risk_level=risk_level,
        flagged_ingredients=flagged,
        confidence=round(final_confidence, 3),
        reasoning_trace=trace,
        safe_ingredients=safe,
    )
