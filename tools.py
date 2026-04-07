"""
tools.py — All tool implementations.
Each tool is a plain async function; agents call them explicitly (ReAct pattern).
"""

import os
import re
import json
import base64
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

logger = logging.getLogger(__name__)

# ── Gemini client setup ───────────────────────────────────────────────────────

_gemini_client: Optional[genai.GenerativeModel] = None


def _get_gemini() -> genai.GenerativeModel:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_client


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — OCR (image → ingredient text)
# ─────────────────────────────────────────────────────────────────────────────

async def ocr_tool(image_b64: str) -> dict:
    """
    Send a base64-encoded image to Gemini Vision and extract ingredient text.

    Returns:
        {"raw_text": str, "ingredients_section": str, "confidence": float}
    """
    try:
        model = _get_gemini()
        image_data = {
            "mime_type": "image/jpeg",
            "data": image_b64,
        }
        prompt = (
            "You are an ingredient extraction assistant. "
            "Look at this supplement label image. "
            "Extract ONLY the ingredients or 'Supplement Facts' section as a plain list. "
            "If no ingredients are visible, say 'NO_INGREDIENTS_FOUND'. "
            "Do NOT add commentary."
        )
        response = model.generate_content([prompt, image_data])
        raw = response.text.strip()
        confidence = 0.3 if "NO_INGREDIENTS_FOUND" in raw else 0.85
        return {
            "raw_text": raw,
            "ingredients_section": raw,
            "confidence": confidence,
        }
    except Exception as e:
        logger.error("ocr_tool failed: %s", e)
        return {"raw_text": "", "ingredients_section": "", "confidence": 0.0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Web scraper (URL → ingredient text)
# ─────────────────────────────────────────────────────────────────────────────

_INGREDIENT_KEYWORDS = [
    "ingredients", "supplement facts", "other ingredients",
    "active ingredients", "contains", "composition",
]

async def web_scrape_tool(url: str) -> dict:
    """
    Fetch a product page and extract the ingredient section.

    Returns:
        {"raw_html": str, "ingredients_text": str, "confidence": float}
    """
    try:
        headers = {"User-Agent": "WADAComplianceBot/1.0"}
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to find ingredient sections by common patterns
        candidates = []
        for kw in _INGREDIENT_KEYWORDS:
            for tag in soup.find_all(string=re.compile(kw, re.IGNORECASE)):
                parent = tag.find_parent()
                if parent:
                    section = parent.get_text(separator=" ", strip=True)
                    candidates.append(section)

        if not candidates:
            # Fallback: grab all text and let the parser handle it
            body_text = soup.get_text(separator="\n", strip=True)
            candidates = [body_text[:3000]]
            confidence = 0.4
        else:
            confidence = 0.75

        ingredients_text = "\n".join(candidates[:5])
        return {
            "raw_html": resp.text[:500],
            "ingredients_text": ingredients_text[:3000],
            "confidence": confidence,
        }
    except httpx.HTTPError as e:
        logger.error("web_scrape_tool HTTP error: %s", e)
        return {"raw_html": "", "ingredients_text": "", "confidence": 0.0, "error": str(e)}
    except Exception as e:
        logger.error("web_scrape_tool failed: %s", e)
        return {"raw_html": "", "ingredients_text": "", "confidence": 0.0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Text parser (raw text → normalized ingredient list)
# ─────────────────────────────────────────────────────────────────────────────

# Common synonyms/aliases that map to a canonical name
_SYNONYM_MAP: dict[str, str] = {
    "ephedra": "ephedrine",
    "ma huang": "ephedrine",
    "dmaa": "1,3-dimethylamylamine",
    "methylhexaneamine": "1,3-dimethylamylamine",
    "1,3-dimethylpentylamine": "1,3-dimethylamylamine",
    "clen": "clenbuterol",
    "beta-2": "clenbuterol",
    "epo": "erythropoietin",
    "hgh": "human growth hormone",
    "somatropin": "human growth hormone",
    "stanozolol": "stanozolol",
    "winstrol": "stanozolol",
    "nandrolone": "nandrolone",
    "deca": "nandrolone",
    "ostarine": "enobosarm",
    "mk-2866": "enobosarm",
    "rad140": "rad-140",
    "ligandrol": "lgd-4033",
    "cardarine": "gw501516",
    "meldonium": "meldonium",
    "mildronate": "meldonium",
}


async def text_parser_tool(raw_text: str) -> dict:
    """
    Parse raw text into a normalized ingredient list using Gemini + synonym map.

    Returns:
        {"ingredients": list[str], "raw_ingredients": list[str], "confidence": float}
    """
    if not raw_text or not raw_text.strip():
        return {"ingredients": [], "raw_ingredients": [], "confidence": 0.0}

    try:
        model = _get_gemini()
        prompt = (
            "Extract ALL individual ingredient names from the following supplement label text. "
            "Return ONLY a JSON array of ingredient name strings. "
            "Remove dosage amounts, percentages, and special characters. "
            "Example output: [\"caffeine\", \"vitamin c\", \"beta-alanine\"]\n\n"
            f"TEXT:\n{raw_text[:2000]}"
        )
        response = model.generate_content(prompt)
        raw_json = response.text.strip()
        # Strip markdown fences if present
        raw_json = re.sub(r"```json|```", "", raw_json).strip()
        raw_ingredients: list[str] = json.loads(raw_json)
    except Exception as e:
        logger.warning("Gemini parse failed (%s), falling back to regex", e)
        raw_ingredients = _regex_parse(raw_text)

    # Normalize
    normalized = []
    for ing in raw_ingredients:
        clean = ing.strip().lower()
        canonical = _SYNONYM_MAP.get(clean, clean)
        if canonical:
            normalized.append(canonical)

    confidence = min(0.9, 0.5 + len(normalized) * 0.05) if normalized else 0.0
    return {
        "ingredients": normalized,
        "raw_ingredients": raw_ingredients,
        "confidence": round(confidence, 2),
    }


def _regex_parse(text: str) -> list[str]:
    """Fallback: split by common delimiters."""
    # Remove dosage patterns like "500mg", "2g", "(50%)"
    text = re.sub(r"\d+\s*(mg|g|mcg|iu|%|ml)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*?\)", "", text)
    parts = re.split(r"[,\n;•·|/]", text)
    return [p.strip().lower() for p in parts if len(p.strip()) > 2]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — WADA list fetcher (dynamic, no hardcoding)
# ─────────────────────────────────────────────────────────────────────────────

_WADA_CACHE: dict = {}  # simple in-memory cache
_WADA_CACHE_TTL_HOURS = 6

_WADA_SOURCES = [
    "https://www.wada-ama.org/en/prohibited-list",
    "https://list.wada-ama.org/prohibited-list/search",
]


async def fetch_wada_list_tool() -> dict:
    """
    Dynamically fetch and parse the WADA prohibited substance list.
    Caches in memory for TTL hours to avoid hammering the endpoint.

    Returns:
        {"substances": list[dict], "version": str, "fetched_at": str, "source": str}
    """
    now = datetime.now(timezone.utc)

    # Check cache
    if _WADA_CACHE:
        fetched = _WADA_CACHE.get("fetched_at")
        if fetched:
            age_hours = (now - fetched).seconds / 3600
            if age_hours < _WADA_CACHE_TTL_HOURS:
                logger.info("Using cached WADA list (age=%.1fh)", age_hours)
                return _WADA_CACHE.get("data", {})

    try:
        headers = {"User-Agent": "WADAComplianceResearch/1.0"}
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(_WADA_SOURCES[0], headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract substances from WADA page structure
        substances = []
        seen = set()

        # Method 1: Look for substance headings and lists
        for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            text = heading.get_text(strip=True)
            if any(kw in text.upper() for kw in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "P1", "M1", "M2", "M3"]):
                category = text
                # Get sibling list items
                sibling = heading.find_next_sibling()
                while sibling and sibling.name in ["ul", "ol", "p", "div"]:
                    for li in sibling.find_all(["li", "p"]):
                        substance_name = li.get_text(strip=True).lower()
                        if substance_name and substance_name not in seen and len(substance_name) > 2:
                            seen.add(substance_name)
                            substances.append({
                                "name": substance_name,
                                "category": category,
                                "prohibited_in": _infer_prohibition_scope(category),
                            })
                    sibling = sibling.find_next_sibling()

        # Method 2: Any text mentioning "prohibited" near substance names
        if len(substances) < 10:
            substances.extend(_extract_fallback_substances(soup))

        # Deduplicate
        seen2: set[str] = set()
        unique_substances = []
        for s in substances:
            if s["name"] not in seen2:
                seen2.add(s["name"])
                unique_substances.append(s)

        version = _extract_wada_version(soup) or str(now.year)
        result = {
            "substances": unique_substances,
            "version": version,
            "fetched_at": now.isoformat(),
            "source": _WADA_SOURCES[0],
            "total": len(unique_substances),
        }

        _WADA_CACHE["data"] = result
        _WADA_CACHE["fetched_at"] = now
        logger.info("Fetched %d substances from WADA list", len(unique_substances))
        return result

    except Exception as e:
        logger.error("WADA fetch failed: %s — using extended baseline", e)
        return _get_extended_baseline_substances(now.isoformat())


def _infer_prohibition_scope(category_text: str) -> str:
    cat = category_text.upper()
    if "IN-COMPETITION" in cat and "OUT" not in cat:
        return "in-competition"
    if "OUT-OF-COMPETITION" in cat:
        return "out-of-competition"
    return "all"


def _extract_wada_version(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(string=re.compile(r"20\d\d", re.IGNORECASE)):
        match = re.search(r"(20\d\d)", tag)
        if match:
            return match.group(1)
    return ""


def _extract_fallback_substances(soup: BeautifulSoup) -> list[dict]:
    substances = []
    all_text = soup.get_text(separator="\n")
    # Look for capitalized substance names in lists
    pattern = re.compile(r"^[A-Z][a-zA-Z\-\s]{3,40}$", re.MULTILINE)
    for match in pattern.finditer(all_text):
        name = match.group().strip().lower()
        if name not in ("the", "and", "prohibited", "substances", "section"):
            substances.append({
                "name": name,
                "category": "unknown",
                "prohibited_in": "all",
            })
    return substances[:50]


def _get_extended_baseline_substances(timestamp: str) -> dict:
    """
    Minimal baseline only used when live fetch fails entirely.
    Covers the most commonly encountered prohibited substances in supplements.
    """
    substances = [
        # S0 — Non-approved substances
        {"name": "sarms", "category": "S0 - Non-approved substances", "prohibited_in": "all"},
        {"name": "enobosarm", "category": "S0 - Non-approved substances", "prohibited_in": "all"},
        {"name": "lgd-4033", "category": "S0 - Non-approved substances", "prohibited_in": "all"},
        {"name": "rad-140", "category": "S0 - Non-approved substances", "prohibited_in": "all"},
        {"name": "gw501516", "category": "S0 - Non-approved substances", "prohibited_in": "all"},
        # S1 — Anabolic agents
        {"name": "stanozolol", "category": "S1 - Anabolic agents", "prohibited_in": "all"},
        {"name": "nandrolone", "category": "S1 - Anabolic agents", "prohibited_in": "all"},
        {"name": "testosterone", "category": "S1 - Anabolic agents", "prohibited_in": "all"},
        {"name": "dhea", "category": "S1 - Anabolic agents", "prohibited_in": "all"},
        {"name": "androstenedione", "category": "S1 - Anabolic agents", "prohibited_in": "all"},
        # S2 — Peptide hormones
        {"name": "human growth hormone", "category": "S2 - Peptide hormones", "prohibited_in": "all"},
        {"name": "erythropoietin", "category": "S2 - Peptide hormones", "prohibited_in": "all"},
        {"name": "igf-1", "category": "S2 - Peptide hormones", "prohibited_in": "all"},
        # S3 — Beta-2 agonists
        {"name": "clenbuterol", "category": "S3 - Beta-2 agonists", "prohibited_in": "all"},
        {"name": "salbutamol", "category": "S3 - Beta-2 agonists", "prohibited_in": "all"},
        # S6 — Stimulants
        {"name": "ephedrine", "category": "S6 - Stimulants", "prohibited_in": "in-competition"},
        {"name": "1,3-dimethylamylamine", "category": "S6 - Stimulants", "prohibited_in": "in-competition"},
        {"name": "amphetamine", "category": "S6 - Stimulants", "prohibited_in": "in-competition"},
        {"name": "modafinil", "category": "S6 - Stimulants", "prohibited_in": "in-competition"},
        {"name": "cocaine", "category": "S6 - Stimulants", "prohibited_in": "in-competition"},
        # S7 — Narcotics
        {"name": "morphine", "category": "S7 - Narcotics", "prohibited_in": "in-competition"},
        {"name": "oxycodone", "category": "S7 - Narcotics", "prohibited_in": "in-competition"},
        # S8 — Cannabinoids
        {"name": "thc", "category": "S8 - Cannabinoids", "prohibited_in": "in-competition"},
        # S9 — Glucocorticoids
        {"name": "prednisolone", "category": "S9 - Glucocorticoids", "prohibited_in": "in-competition"},
        # P1 — Beta-blockers (in specific sports)
        {"name": "propranolol", "category": "P1 - Beta-blockers", "prohibited_in": "specific-sports"},
        {"name": "atenolol", "category": "P1 - Beta-blockers", "prohibited_in": "specific-sports"},
        # M1 — Prohibited methods
        {"name": "meldonium", "category": "M1 - Manipulation", "prohibited_in": "all"},
    ]
    return {
        "substances": substances,
        "version": "baseline-fallback",
        "fetched_at": timestamp,
        "source": "local-baseline",
        "total": len(substances),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — Embedding generator + FAISS semantic search
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


async def generate_embedding_tool(text: str) -> list[float]:
    """
    Generate an embedding for a text string using Gemini embeddings.
    Falls back to a simple character-based hash vector if API fails.
    """
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )
        return result["embedding"]
    except Exception as e:
        logger.warning("Embedding API failed: %s — using fallback", e)
        return _char_hash_embedding(text)


def _char_hash_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic pseudo-embedding for offline/fallback use."""
    vec = [0.0] * dim
    for i, ch in enumerate(text.lower()):
        vec[i % dim] += ord(ch) / 1000.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


async def semantic_search_tool(
    query: str,
    substances: list[dict],
    top_k: int = 5,
    threshold: float = 0.72,
) -> list[dict]:
    """
    Embed the query ingredient and all substance names, then return
    top-k semantic matches above threshold.

    Returns:
        list of {"name": str, "category": str, "similarity": float, "prohibited_in": str}
    """
    if not substances:
        return []

    query_vec = await generate_embedding_tool(query)
    tasks = [generate_embedding_tool(s["name"]) for s in substances]
    sub_vecs = await asyncio.gather(*tasks)

    scored = []
    for substance, vec in zip(substances, sub_vecs):
        sim = _cosine_similarity(query_vec, vec)
        if sim >= threshold:
            scored.append({
                "name": substance["name"],
                "category": substance.get("category", ""),
                "prohibited_in": substance.get("prohibited_in", "all"),
                "similarity": round(sim, 4),
            })

    return sorted(scored, key=lambda x: x["similarity"], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — Keyword matcher (exact + partial)
# ─────────────────────────────────────────────────────────────────────────────

async def keyword_match_tool(ingredient: str, substances: list[dict]) -> list[dict]:
    """
    Exact and partial string matching against the WADA substance list.

    Returns:
        list of {"name": str, "category": str, "match_type": "exact"|"partial", ...}
    """
    results = []
    query = ingredient.strip().lower()

    for s in substances:
        name = s["name"].strip().lower()
        if query == name:
            results.append({**s, "match_type": "exact", "confidence": 1.0})
        elif query in name or name in query:
            overlap = len(set(query.split()) & set(name.split()))
            confidence = min(0.9, 0.5 + overlap * 0.15)
            results.append({**s, "match_type": "partial", "confidence": round(confidence, 2)})

    return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 — Gemini explanation generator
# ─────────────────────────────────────────────────────────────────────────────

async def generate_explanation_tool(
    risk_level: str,
    flagged_ingredients: list[str],
    wada_matches: list[dict],
    all_ingredients: list[str],
    reasoning_trace: list[dict],
    athlete_sport: Optional[str] = None,
) -> dict:
    """
    Use Gemini to generate a clear, athlete-friendly explanation.

    Returns:
        {"summary": str, "athlete_advice": str, "flagged_details": list[dict]}
    """
    sport_context = f" The athlete competes in {athlete_sport}." if athlete_sport else ""

    flagged_str = json.dumps(flagged_ingredients, indent=2)
    matches_str = json.dumps(wada_matches[:5], indent=2)

    prompt = f"""
You are a sports compliance advisor helping athletes understand supplement safety.{sport_context}

ANALYSIS RESULTS:
- Risk level: {risk_level}
- All ingredients found: {", ".join(all_ingredients[:20]) or "none"}
- Flagged ingredients: {flagged_str}
- WADA matches: {matches_str}

Write a response in this EXACT JSON format (no markdown fences):
{{
  "summary": "2-3 sentence plain English summary of the overall risk",
  "athlete_advice": "Clear, actionable advice — what the athlete should do next",
  "flagged_details": [
    {{
      "ingredient": "name",
      "why_flagged": "plain explanation",
      "wada_category": "category name",
      "risk": "what could happen (e.g. competition ban, health risk)",
      "recommendation": "what to do (avoid, consult doctor, etc.)"
    }}
  ]
}}

Rules:
- Use plain language, no jargon
- Be factual, not alarmist
- If SAFE, still remind athlete to check with their national body
- If PROHIBITED, be clear about the consequences
"""
    try:
        model = _get_gemini()
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("Explanation generation failed: %s", e)
        return {
            "summary": f"Analysis complete. Risk level: {risk_level}.",
            "athlete_advice": "Please consult your national anti-doping authority for confirmation.",
            "flagged_details": [{"ingredient": ing, "why_flagged": "Found in WADA prohibited list"} for ing in flagged_ingredients],
        }
