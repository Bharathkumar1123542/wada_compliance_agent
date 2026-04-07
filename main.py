"""
main.py — Unified entrypoint

Modes:
  python main.py          → FastAPI server (production / Cloud Run)
  python main.py --adk    → ADK web UI (development / debugging)
"""

import argparse
import os
import sys
import uvicorn

# ── Runtime args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="WADA Compliance Agent")
parser.add_argument("--adk", action="store_true", help="Launch ADK web UI instead of FastAPI")
parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)), help="Port to bind")
parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev only)")
args, _ = parser.parse_known_args()


if args.adk:
    # ── ADK web runner (for local debugging with ADK's built-in UI) ───────
    from google.adk.cli.fast_api import get_fast_api_app
    from agents.orchestrator import create_orchestrator_agent

    agent = create_orchestrator_agent()
    app = get_fast_api_app(agent=agent, session_service_uri="inmemory://")

    print(f"ADK web UI starting at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

else:
    # ── Production FastAPI server ─────────────────────────────────────────
    from api.app import app  # noqa: F401

    print(f"FastAPI server starting at http://{args.host}:{args.port}")
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,          # Cloud Run: 1 worker per container instance
        log_level="info",
        access_log=True,
    )
