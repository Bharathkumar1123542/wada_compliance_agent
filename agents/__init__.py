"""
Agent 2 — Dynamic WADA Knowledge Agent

ReAct pattern:
  Thought  → decide retrieval strategy (keyword first, then semantic)
  Action   → fetch WADA list, then keyword match, then semantic search
  Observe  → evaluate match quality; escalate to semantic if keyword misses
  Answer   → return WADAMatch list with source citations
"""

import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from schemas import KnowledgeOutput, WADAMatch, MatchType
from tools import (
    fetch_wada_list_tool,
    keyword_match_tool,
    semantic_search_tool,
)

logger = logging.getLogger(__name__)

KNOWLEDGE_SYSTEM_PROMPT = """
You are a WADA (World Anti-Doping Agency) Knowledge Agent.
You have access to the live WADA prohibited substance list.

REASONING PATTERN — follow this for EVERY query:

THOUGHT: I need to check if each ingredient is on the WADA prohibited list.
         I will first fetch the current list, then check each ingredient.
ACTION: Call fetch_wada_list_tool to get the current prohibited substances.
OBSERVATION: I have the list. Now I will check each ingredient.
ACTION: For each ingredient, call keyword_match_tool first (fast, exact).
OBSERVATION: For any ingredients with no exact match, run semantic_search_tool.
FINAL ANSWER: Return all matches with their WADA category and prohibition scope.

RULES:
- NEVER assume a substance is safe without running both keyword and semantic search.
- An exact match = PROHIBITED. A semantic match above threshold = CAUTION.
- Always include the WADA category (S1, S2, etc.) in the match output.
- Always include the source URL from the fetched list.
"""


async def _fetch_wada_wrapper() -> dict:
    """Fetch the current WADA prohibited substances list dynamically."""
    return await fetch_wada_list_tool()


async def _keyword_match_wrapper(ingredient: str, substances: list) -> list:
    """Exact and partial keyword match for an ingredient against WADA substances."""
    return await keyword_match_tool(ingredient, substances)


async def _semantic_search_wrapper(query: str, substances: list, top_k: int = 5) -> list:
    """Semantic similarity search for an ingredient against WADA substances."""
    return await semantic_search_tool(query, substances, top_k=top_k)


def create_knowledge_agent() -> LlmAgent:
    return LlmAgent(
        name="wada_knowledge_agent",
        model="gemini-2.0-flash",
        description="Dynamically fetches and queries the WADA prohibited substance list.",
        instruction=KNOWLEDGE_SYSTEM_PROMPT,
        tools=[
            FunctionTool(func=_fetch_wada_wrapper),
            FunctionTool(func=_keyword_match_wrapper),
            FunctionTool(func=_semantic_search_wrapper),
        ],
        output_key="knowledge_result",
    )


# ── Direct Python API ─────────────────────────────────────────────────────────

async def run_knowledge_check(ingredients: list[str]) -> KnowledgeOutput:
    """
    For each ingredient, run keyword then semantic search against the live WADA list.
    """
    from datetime import datetime, timezone

    if not ingredients:
        return KnowledgeOutput(matches=[], wada_list_version="", retrieval_timestamp="")

    # ── THOUGHT: fetch the live WADA list first ───────────────────────────
    logger.info("[KnowledgeAgent] THOUGHT: fetching live WADA prohibited list")

    # ── ACTION: fetch WADA list ───────────────────────────────────────────
    logger.info("[KnowledgeAgent] ACTION: fetch_wada_list_tool")
    wada_data = await fetch_wada_list_tool()
    substances = wada_data.get("substances", [])
    version = wada_data.get("version", "unknown")
    timestamp = wada_data.get("fetched_at", datetime.now(timezone.utc).isoformat())
    source = wada_data.get("source", "")

    logger.info(
        "[KnowledgeAgent] OBSERVATION: fetched %d substances (version=%s)",
        len(substances), version
    )

    all_matches: list[WADAMatch] = []
    matched_ingredients: set[str] = set()

    # ── ACTION: keyword match per ingredient ──────────────────────────────
    logger.info("[KnowledgeAgent] ACTION: keyword_match_tool for %d ingredients", len(ingredients))
    for ingredient in ingredients:
        kw_results = await keyword_match_tool(ingredient, substances)

        if kw_results:
            best = kw_results[0]
            match_type = MatchType.EXACT if best.get("match_type") == "exact" else MatchType.SEMANTIC
            all_matches.append(WADAMatch(
                ingredient=ingredient,
                match_type=match_type,
                wada_category=best.get("category", ""),
                prohibited_in=best.get("prohibited_in", "all"),
                source=source,
                confidence=float(best.get("confidence", 0.8)),
            ))
            matched_ingredients.add(ingredient)
            logger.info(
                "[KnowledgeAgent] OBSERVATION: keyword match for '%s' → '%s' (%s)",
                ingredient, best.get("name"), best.get("match_type")
            )

    # ── THOUGHT: run semantic search for unmatched ingredients ────────────
    unmatched = [i for i in ingredients if i not in matched_ingredients]
    logger.info(
        "[KnowledgeAgent] THOUGHT: %d ingredients unmatched, running semantic search",
        len(unmatched)
    )

    # ── ACTION: semantic search for unmatched ────────────────────────────
    for ingredient in unmatched:
        logger.info("[KnowledgeAgent] ACTION: semantic_search_tool for '%s'", ingredient)
        sem_results = await semantic_search_tool(ingredient, substances, top_k=3, threshold=0.70)

        if sem_results:
            best = sem_results[0]
            all_matches.append(WADAMatch(
                ingredient=ingredient,
                match_type=MatchType.SEMANTIC,
                wada_category=best.get("category", ""),
                prohibited_in=best.get("prohibited_in", "all"),
                source=source,
                confidence=float(best.get("similarity", 0.7)),
            ))
            logger.info(
                "[KnowledgeAgent] OBSERVATION: semantic match for '%s' → '%s' (sim=%.2f)",
                ingredient, best.get("name"), best.get("similarity", 0.0)
            )
        else:
            logger.info("[KnowledgeAgent] OBSERVATION: no match found for '%s'", ingredient)

    logger.info(
        "[KnowledgeAgent] FINAL ANSWER: %d matches from %d ingredients",
        len(all_matches), len(ingredients)
    )

    return KnowledgeOutput(
        matches=all_matches,
        wada_list_version=version,
        retrieval_timestamp=timestamp,
    )
