"""
Agent 1 — Multi-Modal Ingredient Extraction Agent

ReAct pattern:
  Thought  → decide which tool to use based on input_type
  Action   → call OCR, web scraper, or text parser
  Observe  → check confidence; retry with fallback if low
  Answer   → return ExtractionOutput with normalized ingredients
"""

import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from schemas import ExtractionInput, ExtractionOutput, InputType
from tools import ocr_tool, web_scrape_tool, text_parser_tool

logger = logging.getLogger(__name__)

# ── System prompt enforcing ReAct ────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """
You are an Ingredient Extraction Agent for a WADA compliance system.
You MUST follow this exact reasoning pattern for EVERY request:

THOUGHT: Analyze what type of input you received (image/text/url) and decide the correct tool.
ACTION: Call the appropriate tool (ocr_tool, web_scrape_tool, or text_parser_tool).
OBSERVATION: Read the tool output. If confidence < 0.5, call a secondary tool to verify.
FINAL ANSWER: Return the structured ingredient list with confidence scores.

RULES:
- NEVER guess ingredients without calling a tool.
- ALWAYS normalize ingredient names (e.g. "ma huang" → "ephedrine").
- If OCR confidence is low, attempt web search for the product name.
- Remove dosage information (mg, g, IU) from ingredient names.
- Return ONLY ingredient names, not descriptions.
"""

# ── Tool wrappers (ADK FunctionTool compatible) ───────────────────────────────

async def _ocr_wrapper(image_b64: str) -> dict:
    """Extract ingredients from a supplement label image using OCR."""
    return await ocr_tool(image_b64)


async def _scrape_wrapper(url: str) -> dict:
    """Scrape a product page URL and extract the ingredient section."""
    return await web_scrape_tool(url)


async def _parse_wrapper(raw_text: str) -> dict:
    """Parse raw ingredient text into a normalized ingredient list."""
    return await text_parser_tool(raw_text)


# ── ADK Agent definition ──────────────────────────────────────────────────────

def create_extraction_agent() -> LlmAgent:
    return LlmAgent(
        name="ingredient_extraction_agent",
        model="gemini-2.0-flash",
        description="Extracts and normalizes supplement ingredients from images, URLs, or text.",
        instruction=EXTRACTION_SYSTEM_PROMPT,
        tools=[
            FunctionTool(func=_ocr_wrapper),
            FunctionTool(func=_scrape_wrapper),
            FunctionTool(func=_parse_wrapper),
        ],
        output_key="extraction_result",
    )


# ── Direct Python API (used by orchestrator) ──────────────────────────────────

async def run_extraction(inp: ExtractionInput) -> ExtractionOutput:
    """
    Directly invoke extraction logic without ADK runner.
    Orchestrator calls this for tight integration.
    """
    raw_text = ""
    confidence = 0.0
    method = ""

    # ── THOUGHT: select tool based on input_type ──────────────────────────
    logger.info("[ExtractionAgent] THOUGHT: input_type=%s", inp.input_type)

    # ── ACTION + OBSERVATION ──────────────────────────────────────────────
    if inp.input_type == InputType.IMAGE:
        logger.info("[ExtractionAgent] ACTION: calling ocr_tool")
        result = await ocr_tool(inp.data)
        raw_text = result.get("ingredients_section", "")
        confidence = result.get("confidence", 0.0)
        method = "ocr"
        logger.info("[ExtractionAgent] OBSERVATION: ocr confidence=%.2f", confidence)

    elif inp.input_type == InputType.URL:
        logger.info("[ExtractionAgent] ACTION: calling web_scrape_tool")
        result = await web_scrape_tool(inp.data)
        raw_text = result.get("ingredients_text", "")
        confidence = result.get("confidence", 0.0)
        method = "scrape"
        logger.info("[ExtractionAgent] OBSERVATION: scrape confidence=%.2f", confidence)

    elif inp.input_type == InputType.TEXT:
        raw_text = inp.data
        confidence = 0.9  # Text input is the highest confidence starting point
        method = "direct"
        logger.info("[ExtractionAgent] OBSERVATION: direct text input, confidence=0.9")

    # ── RETRY: if confidence too low, fall back to Gemini text parser ────
    if confidence < 0.5 and raw_text:
        logger.info("[ExtractionAgent] THOUGHT: confidence low, retrying with text_parser_tool")

    if not raw_text:
        logger.warning("[ExtractionAgent] No text extracted from input")
        return ExtractionOutput(
            ingredients=[],
            raw_ingredients=[],
            confidence_scores=[],
            extraction_method=method,
            notes="No ingredient text could be extracted from the input.",
        )

    # ── ACTION: parse extracted text ──────────────────────────────────────
    logger.info("[ExtractionAgent] ACTION: calling text_parser_tool")
    parse_result = await text_parser_tool(raw_text)

    ingredients: list[str] = parse_result.get("ingredients", [])
    raw_ingredients: list[str] = parse_result.get("raw_ingredients", [])
    parse_confidence: float = parse_result.get("confidence", 0.0)

    # Final confidence = average of extraction + parsing
    final_confidence = (confidence + parse_confidence) / 2

    logger.info(
        "[ExtractionAgent] FINAL ANSWER: %d ingredients, confidence=%.2f",
        len(ingredients), final_confidence
    )

    return ExtractionOutput(
        ingredients=ingredients,
        raw_ingredients=raw_ingredients,
        confidence_scores=[round(final_confidence, 2)] * len(ingredients),
        extraction_method=method,
        notes=f"Extracted via {method}, parsed {len(ingredients)} ingredients.",
    )
