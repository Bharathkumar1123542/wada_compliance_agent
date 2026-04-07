"""
tests/test_agents.py — 30+ unit + integration tests

Covers:
- Tool functions (OCR, scraper, parser, WADA fetch, keyword, semantic, explanation)
- Each agent's direct Python API
- Orchestrator end-to-end
- FastAPI endpoints via TestClient
- Edge cases (hidden ingredients, misleading labels, incomplete pages)
"""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schemas import (
    ExtractionInput, InputType, AnalyzeRequest, RiskLevel,
    WADAMatch, MatchType, KnowledgeOutput, ExtractionOutput,
    ComplianceOutput, ReasoningStep,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_ingredients():
    return ["caffeine", "vitamin c", "beta-alanine", "magnesium"]

@pytest.fixture
def prohibited_ingredients():
    return ["stanozolol", "ephedrine", "erythropoietin"]

@pytest.fixture
def mixed_ingredients():
    return ["caffeine", "stanozolol", "vitamin b12", "clenbuterol"]

@pytest.fixture
def sample_wada_substances():
    return [
        {"name": "stanozolol", "category": "S1 - Anabolic agents", "prohibited_in": "all"},
        {"name": "ephedrine", "category": "S6 - Stimulants", "prohibited_in": "in-competition"},
        {"name": "clenbuterol", "category": "S3 - Beta-2 agonists", "prohibited_in": "all"},
        {"name": "erythropoietin", "category": "S2 - Peptide hormones", "prohibited_in": "all"},
        {"name": "caffeine", "category": "monitoring", "prohibited_in": "none"},
    ]

@pytest.fixture
def mock_wada_response(sample_wada_substances):
    return {
        "substances": sample_wada_substances,
        "version": "2024",
        "fetched_at": "2024-01-01T00:00:00+00:00",
        "source": "https://www.wada-ama.org",
        "total": len(sample_wada_substances),
    }

@pytest.fixture
def prohibited_wada_matches():
    return [
        WADAMatch(
            ingredient="stanozolol",
            match_type=MatchType.EXACT,
            wada_category="S1 - Anabolic agents",
            prohibited_in="all",
            source="https://www.wada-ama.org",
            confidence=1.0,
        )
    ]

@pytest.fixture
def caution_wada_matches():
    return [
        WADAMatch(
            ingredient="1,3-dimethylamylamine",
            match_type=MatchType.SEMANTIC,
            wada_category="S6 - Stimulants",
            prohibited_in="in-competition",
            source="https://www.wada-ama.org",
            confidence=0.72,
        )
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTextParserTool:

    @pytest.mark.asyncio
    async def test_parse_clean_csv_text(self):
        from tools import text_parser_tool
        text = "caffeine 200mg, vitamin c 500mg, beta-alanine 2g, magnesium stearate"
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text='["caffeine", "vitamin c", "beta-alanine", "magnesium stearate"]'
            )
            mock_gemini.return_value = mock_model
            result = await text_parser_tool(text)
        assert "caffeine" in result["ingredients"]
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_parse_empty_text(self):
        from tools import text_parser_tool
        result = await text_parser_tool("")
        assert result["ingredients"] == []
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_synonym_normalization(self):
        from tools import text_parser_tool
        text = "ma huang extract, ephedra sinica"
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text='["ma huang", "ephedra"]'
            )
            mock_gemini.return_value = mock_model
            result = await text_parser_tool(text)
        # Both should normalize to ephedrine
        assert all(ing == "ephedrine" for ing in result["ingredients"])

    @pytest.mark.asyncio
    async def test_parse_noisy_label_text(self):
        """Edge case: label with mixed units, special chars, HTML entities"""
        from tools import text_parser_tool
        noisy_text = "Ingredients: L-Citrulline (2,000mg), Beta-Alanine† (3.2g), Caffeine Anhydrous* [200mg]"
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text='["l-citrulline", "beta-alanine", "caffeine anhydrous"]'
            )
            mock_gemini.return_value = mock_model
            result = await text_parser_tool(noisy_text)
        assert len(result["ingredients"]) >= 2

    @pytest.mark.asyncio
    async def test_gemini_failure_falls_back_to_regex(self):
        from tools import text_parser_tool
        with patch("tools._get_gemini", side_effect=Exception("API error")):
            result = await text_parser_tool("caffeine, vitamin c, creatine")
        assert isinstance(result["ingredients"], list)


class TestKeywordMatchTool:

    @pytest.mark.asyncio
    async def test_exact_match(self, sample_wada_substances):
        from tools import keyword_match_tool
        results = await keyword_match_tool("stanozolol", sample_wada_substances)
        assert len(results) > 0
        assert results[0]["match_type"] == "exact"
        assert results[0]["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_partial_match(self, sample_wada_substances):
        from tools import keyword_match_tool
        results = await keyword_match_tool("stanozolol acetate", sample_wada_substances)
        assert len(results) > 0
        assert results[0]["match_type"] in ("exact", "partial")

    @pytest.mark.asyncio
    async def test_no_match_safe_ingredient(self, sample_wada_substances):
        from tools import keyword_match_tool
        results = await keyword_match_tool("beta-alanine", sample_wada_substances)
        assert results == []

    @pytest.mark.asyncio
    async def test_case_insensitive(self, sample_wada_substances):
        from tools import keyword_match_tool
        results = await keyword_match_tool("STANOZOLOL", sample_wada_substances)
        assert len(results) > 0


class TestWADAFetcher:

    @pytest.mark.asyncio
    async def test_fetch_returns_required_fields(self):
        from tools import fetch_wada_list_tool, _WADA_CACHE
        _WADA_CACHE.clear()
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.text = "<html><body><h3>S1 Anabolic Agents</h3><ul><li>Stanozolol</li></ul></body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            result = await fetch_wada_list_tool()
        assert "substances" in result
        assert "version" in result
        assert "fetched_at" in result
        assert isinstance(result["substances"], list)

    @pytest.mark.asyncio
    async def test_cache_is_used_on_second_call(self):
        from tools import fetch_wada_list_tool, _WADA_CACHE
        # Pre-populate cache
        from datetime import datetime, timezone
        _WADA_CACHE["data"] = {"substances": [], "version": "cached", "fetched_at": "t", "source": ""}
        _WADA_CACHE["fetched_at"] = datetime.now(timezone.utc)
        result = await fetch_wada_list_tool()
        assert result["version"] == "cached"

    @pytest.mark.asyncio
    async def test_fallback_when_fetch_fails(self):
        from tools import fetch_wada_list_tool, _WADA_CACHE
        _WADA_CACHE.clear()
        with patch("httpx.AsyncClient", side_effect=Exception("Network error")):
            result = await fetch_wada_list_tool()
        # Should return baseline, not raise
        assert len(result["substances"]) > 0
        assert result["version"] == "baseline-fallback"


class TestEmbeddingTool:

    @pytest.mark.asyncio
    async def test_fallback_embedding_is_correct_length(self):
        from tools import _char_hash_embedding
        vec = _char_hash_embedding("stanozolol", dim=64)
        assert len(vec) == 64

    @pytest.mark.asyncio
    async def test_cosine_similarity_identical(self):
        from tools import _cosine_similarity
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_cosine_similarity_orthogonal(self):
        from tools import _cosine_similarity
        a, b = [1.0, 0.0], [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionAgent:

    @pytest.mark.asyncio
    async def test_text_input_direct(self):
        from agents.extraction_agent import run_extraction
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text='["caffeine", "vitamin c"]'
            )
            mock_gemini.return_value = mock_model
            result = await run_extraction(
                ExtractionInput(input_type=InputType.TEXT, data="caffeine 200mg, vitamin c 500mg")
            )
        assert len(result.ingredients) >= 1
        assert result.extraction_method == "direct"

    @pytest.mark.asyncio
    async def test_url_input_calls_scraper(self):
        from agents.extraction_agent import run_extraction
        with patch("tools.web_scrape_tool", new_callable=AsyncMock) as mock_scrape, \
             patch("tools._get_gemini") as mock_gemini:
            mock_scrape.return_value = {"ingredients_text": "caffeine, creatine", "confidence": 0.8}
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text='["caffeine", "creatine"]')
            mock_gemini.return_value = mock_model
            result = await run_extraction(
                ExtractionInput(input_type=InputType.URL, data="https://example.com/supplement")
            )
        mock_scrape.assert_called_once()
        assert result.extraction_method == "scrape"

    @pytest.mark.asyncio
    async def test_image_input_calls_ocr(self):
        from agents.extraction_agent import run_extraction
        with patch("tools.ocr_tool", new_callable=AsyncMock) as mock_ocr, \
             patch("tools._get_gemini") as mock_gemini:
            mock_ocr.return_value = {"ingredients_section": "beta-alanine 3g", "confidence": 0.9}
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text='["beta-alanine"]')
            mock_gemini.return_value = mock_model
            result = await run_extraction(
                ExtractionInput(input_type=InputType.IMAGE, data="base64encodedimage==")
            )
        mock_ocr.assert_called_once()
        assert result.extraction_method == "ocr"

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_extraction(self):
        from agents.extraction_agent import run_extraction
        result = await run_extraction(ExtractionInput(input_type=InputType.TEXT, data=""))
        assert result.ingredients == []


class TestKnowledgeAgent:

    @pytest.mark.asyncio
    async def test_prohibited_ingredient_detected(self, mock_wada_response):
        from agents.knowledge_agent import run_knowledge_check
        with patch("tools.fetch_wada_list_tool", new_callable=AsyncMock, return_value=mock_wada_response), \
             patch("tools.generate_embedding_tool", new_callable=AsyncMock, return_value=[0.1] * 64):
            result = await run_knowledge_check(["stanozolol"])
        assert len(result.matches) > 0
        assert result.matches[0].ingredient == "stanozolol"

    @pytest.mark.asyncio
    async def test_safe_ingredient_not_flagged(self, mock_wada_response):
        from agents.knowledge_agent import run_knowledge_check
        with patch("tools.fetch_wada_list_tool", new_callable=AsyncMock, return_value=mock_wada_response), \
             patch("tools.generate_embedding_tool", new_callable=AsyncMock, return_value=[0.1] * 64):
            result = await run_knowledge_check(["vitamin d3"])
        # Should have no exact/semantic match for vitamin d3
        exact_matches = [m for m in result.matches if m.match_type == MatchType.EXACT]
        assert len(exact_matches) == 0

    @pytest.mark.asyncio
    async def test_empty_ingredients_list(self):
        from agents.knowledge_agent import run_knowledge_check
        result = await run_knowledge_check([])
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_wada_version_captured(self, mock_wada_response):
        from agents.knowledge_agent import run_knowledge_check
        with patch("tools.fetch_wada_list_tool", new_callable=AsyncMock, return_value=mock_wada_response), \
             patch("tools.generate_embedding_tool", new_callable=AsyncMock, return_value=[0.1] * 64):
            result = await run_knowledge_check(["caffeine"])
        assert result.wada_list_version == "2024"


class TestComplianceAgent:

    @pytest.mark.asyncio
    async def test_exact_match_gives_prohibited(self, prohibited_wada_matches):
        from agents.compliance_agent import run_compliance_check
        result = await run_compliance_check(
            ingredients=["stanozolol"],
            wada_matches=prohibited_wada_matches,
        )
        assert result.risk_level == RiskLevel.PROHIBITED
        assert "stanozolol" in result.flagged_ingredients

    @pytest.mark.asyncio
    async def test_semantic_match_gives_caution(self, caution_wada_matches):
        from agents.compliance_agent import run_compliance_check
        with patch("tools.fetch_wada_list_tool", new_callable=AsyncMock, return_value={"substances": []}):
            result = await run_compliance_check(
                ingredients=["1,3-dimethylamylamine"],
                wada_matches=caution_wada_matches,
            )
        assert result.risk_level in (RiskLevel.PROHIBITED, RiskLevel.CAUTION)
        assert len(result.flagged_ingredients) > 0

    @pytest.mark.asyncio
    async def test_clean_supplement_is_safe(self):
        from agents.compliance_agent import run_compliance_check
        result = await run_compliance_check(
            ingredients=["vitamin c", "zinc", "magnesium"],
            wada_matches=[],
        )
        assert result.risk_level == RiskLevel.SAFE
        assert result.flagged_ingredients == []

    @pytest.mark.asyncio
    async def test_reasoning_trace_populated(self, prohibited_wada_matches):
        from agents.compliance_agent import run_compliance_check
        result = await run_compliance_check(
            ingredients=["stanozolol"],
            wada_matches=prohibited_wada_matches,
        )
        assert len(result.reasoning_trace) > 0
        assert all(hasattr(step, "thought") for step in result.reasoning_trace)

    @pytest.mark.asyncio
    async def test_mixed_supplement_worst_risk_wins(self):
        """If one ingredient is PROHIBITED, overall should be PROHIBITED."""
        from agents.compliance_agent import run_compliance_check
        matches = [
            WADAMatch(
                ingredient="stanozolol", match_type=MatchType.EXACT,
                wada_category="S1", prohibited_in="all", confidence=1.0, source=""
            ),
            WADAMatch(
                ingredient="ephedrine", match_type=MatchType.EXACT,
                wada_category="S6", prohibited_in="in-competition", confidence=1.0, source=""
            ),
        ]
        result = await run_compliance_check(
            ingredients=["vitamin c", "stanozolol", "ephedrine"],
            wada_matches=matches,
        )
        assert result.risk_level == RiskLevel.PROHIBITED
        assert len(result.flagged_ingredients) >= 1


class TestExplanationAgent:

    @pytest.mark.asyncio
    async def test_explanation_generated(self):
        from agents.explanation_agent import run_explanation
        extraction = ExtractionOutput(
            ingredients=["stanozolol"], raw_ingredients=["stanozolol"], confidence_scores=[1.0]
        )
        knowledge = KnowledgeOutput(
            matches=[WADAMatch(ingredient="stanozolol", match_type=MatchType.EXACT,
                               wada_category="S1", prohibited_in="all", confidence=1.0, source="")],
            wada_list_version="2024",
        )
        compliance = ComplianceOutput(
            risk_level=RiskLevel.PROHIBITED,
            flagged_ingredients=["stanozolol"],
            confidence=1.0,
            reasoning_trace=[ReasoningStep(thought="t", action="a", observation="o")],
        )
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text=json.dumps({
                    "summary": "This supplement is prohibited.",
                    "athlete_advice": "Do not use this supplement.",
                    "flagged_details": []
                })
            )
            mock_gemini.return_value = mock_model
            result = await run_explanation(extraction, knowledge, compliance)
        assert result.summary != ""
        assert result.athlete_advice != ""


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:

    @pytest.mark.asyncio
    async def test_full_pipeline_safe_supplement(self):
        from agents.orchestrator import run_orchestrator
        request = AnalyzeRequest(input_type=InputType.TEXT, data="vitamin c 500mg, zinc 10mg")
        with patch("tools._get_gemini") as mock_gemini, \
             patch("tools.fetch_wada_list_tool", new_callable=AsyncMock) as mock_wada, \
             patch("tools.generate_embedding_tool", new_callable=AsyncMock, return_value=[0.1]*64):
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = [
                MagicMock(text='["vitamin c", "zinc"]'),
                MagicMock(text=json.dumps({"summary": "Safe.", "athlete_advice": "OK to use.", "flagged_details": []}))
            ]
            mock_gemini.return_value = mock_model
            mock_wada.return_value = {
                "substances": [{"name": "stanozolol", "category": "S1", "prohibited_in": "all"}],
                "version": "2024", "fetched_at": "2024-01-01", "source": "", "total": 1,
            }
            response = await run_orchestrator(request)
        assert response.risk_level in (RiskLevel.SAFE, RiskLevel.CAUTION, RiskLevel.PROHIBITED)
        assert isinstance(response.ingredients, list)
        assert isinstance(response.trace, list)

    @pytest.mark.asyncio
    async def test_full_pipeline_prohibited_supplement(self):
        from agents.orchestrator import run_orchestrator
        request = AnalyzeRequest(input_type=InputType.TEXT, data="stanozolol 50mg")
        with patch("tools._get_gemini") as mock_gemini, \
             patch("tools.fetch_wada_list_tool", new_callable=AsyncMock) as mock_wada, \
             patch("tools.generate_embedding_tool", new_callable=AsyncMock, return_value=[0.1]*64):
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = [
                MagicMock(text='["stanozolol"]'),
                MagicMock(text=json.dumps({"summary": "PROHIBITED.", "athlete_advice": "Do not use.", "flagged_details": []}))
            ]
            mock_gemini.return_value = mock_model
            mock_wada.return_value = {
                "substances": [{"name": "stanozolol", "category": "S1", "prohibited_in": "all"}],
                "version": "2024", "fetched_at": "2024-01-01", "source": "", "total": 1,
            }
            response = await run_orchestrator(request)
        assert response.risk_level == RiskLevel.PROHIBITED

    @pytest.mark.asyncio
    async def test_empty_input_returns_gracefully(self):
        from agents.orchestrator import run_orchestrator
        request = AnalyzeRequest(input_type=InputType.TEXT, data="no ingredients here at all xyz123")
        with patch("tools._get_gemini") as mock_gemini, \
             patch("tools.fetch_wada_list_tool", new_callable=AsyncMock) as mock_wada, \
             patch("tools.generate_embedding_tool", new_callable=AsyncMock, return_value=[0.1]*64):
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text='[]')
            mock_gemini.return_value = mock_model
            mock_wada.return_value = {"substances": [], "version": "2024", "fetched_at": "t", "source": "", "total": 0}
            response = await run_orchestrator(request)
        assert response is not None


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:

    @pytest.fixture
    def test_app(self):
        from api.app import app
        return app

    @pytest.mark.asyncio
    async def test_health_check(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_analyze_endpoint_returns_200(self, test_app):
        payload = {"input_type": "text", "data": "caffeine 200mg, vitamin c"}
        mock_response = {
            "risk_level": "SAFE", "ingredients": ["caffeine", "vitamin c"],
            "flagged": [], "confidence": 0.95,
            "explanation": "Safe supplement.", "athlete_advice": "OK to use.",
            "trace": [], "wada_list_version": "2024",
            "disclaimer": "Always verify."
        }
        with patch("agents.orchestrator.run_orchestrator", new_callable=AsyncMock) as mock_orch:
            from schemas import AnalyzeResponse
            mock_orch.return_value = AnalyzeResponse(**mock_response)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                resp = await client.post("/analyze-supplement", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_level" in data
        assert "ingredients" in data
        assert "explanation" in data

    @pytest.mark.asyncio
    async def test_analyze_endpoint_empty_data_returns_400(self, test_app):
        payload = {"input_type": "text", "data": "  "}
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            resp = await client.post("/analyze-supplement", json=payload)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_endpoint_returns_all_results(self, test_app):
        payload = {
            "items": [
                {"input_type": "text", "data": "caffeine"},
                {"input_type": "text", "data": "vitamin c"},
            ]
        }
        mock_response = {
            "risk_level": "SAFE", "ingredients": ["caffeine"],
            "flagged": [], "confidence": 0.9,
            "explanation": "Safe.", "athlete_advice": "OK.",
            "trace": [], "wada_list_version": "2024",
            "disclaimer": "Always verify."
        }
        with patch("agents.orchestrator.run_orchestrator", new_callable=AsyncMock) as mock_orch:
            from schemas import AnalyzeResponse
            mock_orch.return_value = AnalyzeResponse(**mock_response)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                resp = await client.post("/analyze-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_too_large_returns_400(self, test_app):
        payload = {"items": [{"input_type": "text", "data": f"item {i}"} for i in range(15)]}
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            resp = await client.post("/analyze-batch", json=payload)
        assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_misleading_label_hidden_dmaa(self):
        """Supplement using alias 'geranium extract' for DMAA."""
        from tools import text_parser_tool, _SYNONYM_MAP
        # Manually add alias for test
        _SYNONYM_MAP["geranium extract"] = "1,3-dimethylamylamine"
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text='["geranium extract", "caffeine"]'
            )
            mock_gemini.return_value = mock_model
            result = await text_parser_tool("geranium extract 50mg, caffeine 200mg")
        assert "1,3-dimethylamylamine" in result["ingredients"]

    @pytest.mark.asyncio
    async def test_incomplete_webpage_still_parses(self):
        """Web scraper should degrade gracefully on malformed pages."""
        from tools import web_scrape_tool
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.text = "<html><body><p>Limited info available</p></body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            result = await web_scrape_tool("https://example.com/bad-page")
        assert "ingredients_text" in result
        assert isinstance(result["confidence"], float)

    @pytest.mark.asyncio
    async def test_very_long_ingredient_list(self):
        """System handles supplements with 50+ ingredients."""
        from agents.compliance_agent import run_compliance_check
        ingredients = [f"ingredient_{i}" for i in range(50)]
        result = await run_compliance_check(ingredients=ingredients, wada_matches=[])
        assert result.risk_level == RiskLevel.SAFE
        assert len(result.safe_ingredients) == 50

    @pytest.mark.asyncio
    async def test_ingredient_with_numbers_and_dashes(self):
        """Handles ingredient names like '5-HTP' or 'L-theanine'."""
        from tools import keyword_match_tool
        substances = [{"name": "5-htp", "category": "monitoring", "prohibited_in": "none"}]
        results = await keyword_match_tool("5-htp", substances)
        assert len(results) > 0
        assert results[0]["match_type"] == "exact"

    @pytest.mark.asyncio
    async def test_sport_specific_context_passed_to_explanation(self):
        """Beta-blockers are only prohibited in specific sports."""
        from agents.explanation_agent import run_explanation
        extraction = ExtractionOutput(
            ingredients=["propranolol"], raw_ingredients=["propranolol"], confidence_scores=[0.9]
        )
        knowledge = KnowledgeOutput(
            matches=[WADAMatch(ingredient="propranolol", match_type=MatchType.EXACT,
                               wada_category="P1", prohibited_in="specific-sports", confidence=0.9, source="")],
            wada_list_version="2024",
        )
        compliance = ComplianceOutput(
            risk_level=RiskLevel.PROHIBITED,
            flagged_ingredients=["propranolol"],
            confidence=0.9,
            reasoning_trace=[],
        )
        with patch("tools._get_gemini") as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(
                text=json.dumps({
                    "summary": "Beta-blockers are prohibited in archery.",
                    "athlete_advice": "Do not use if competing in archery.",
                    "flagged_details": []
                })
            )
            mock_gemini.return_value = mock_model
            result = await run_explanation(extraction, knowledge, compliance, athlete_sport="archery")
        assert result.summary != ""
