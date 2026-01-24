#!/usr/bin/env python3
"""
Visible Learning Demo Server

Serves the learning demo UI and proxies requests to the orchestrator.
Adds additional endpoints for learning visualization.
"""
import json
import logging
import sys
from pathlib import Path
import asyncio

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ORCHESTRATOR_URL = "http://localhost:8000"
DEMO_PORT = 8088
DEMO_DIR = Path(__file__).parent

app = FastAPI(title="RFSN Visible Learning Demo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client
client = httpx.AsyncClient(timeout=60.0)


@app.get("/")
async def serve_demo():
    """Serve the demo HTML page."""
    html_path = DEMO_DIR / "visible_learning_demo.html"
    if not html_path.exists():
        raise HTTPException(404, "Demo file not found")
    return FileResponse(html_path, media_type="text/html")


@app.get("/api/health")
async def health_check():
    """Check orchestrator health."""
    try:
        response = await client.get(f"{ORCHESTRATOR_URL}/api/health")
        return response.json()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=503)


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """
    Proxy chat request to orchestrator with enhanced response.
    Adds learning metrics to the response for visualization.
    """
    try:
        body = await request.json()
        
        # Ensure we get action info back
        body["include_action"] = True
        
        response = await client.post(
            f"{ORCHESTRATOR_URL}/api/chat",
            json=body,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            return JSONResponse(
                {"error": "Orchestrator error", "detail": response.text},
                status_code=response.status_code
            )
        
        data = response.json()
        
        # Enhance with learning info if available
        enhanced = {
            "response": data.get("response", data.get("text", "")),
            "action": data.get("action", "UNKNOWN"),
            "reward": data.get("reward", 0.5),
            "affinity_change": data.get("affinity_change", 0.0),
            "temporal_adjustments": data.get("temporal_adjustments", {}),
            "bandit_stats": data.get("bandit_stats", {}),
        }
        
        return JSONResponse(enhanced)
        
    except httpx.ConnectError:
        # Fallback to simulation mode
        logger.warning("Orchestrator not available, using simulation mode")
        return JSONResponse({
            "response": "The orchestrator is not available. Running in simulation mode.",
            "action": "GREET",
            "reward": 0.5,
            "simulated": True
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/api/learning/stats")
async def get_learning_stats():
    """Get learning statistics from orchestrator."""
    try:
        response = await client.get(f"{ORCHESTRATOR_URL}/api/learning/stats")
        return response.json()
    except Exception as e:
        logger.warning(f"Could not get learning stats: {e}")
        return {
            "temporal_memory_size": 0,
            "bandit_arms": {},
            "total_updates": 0
        }


@app.get("/api/learning/temporal")
async def get_temporal_memory():
    """Get temporal memory contents for visualization."""
    try:
        response = await client.get(f"{ORCHESTRATOR_URL}/api/learning/temporal")
        return response.json()
    except Exception as e:
        logger.warning(f"Could not get temporal memory: {e}")
        return {"experiences": [], "stats": {}}


def main():
    """Run the demo server."""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           🧠 RFSN Visible Learning Demo                       ║
╠═══════════════════════════════════════════════════════════════╣
║  Demo UI:       http://localhost:{DEMO_PORT}                         ║
║  Orchestrator:  {ORCHESTRATOR_URL}                    ║
╚═══════════════════════════════════════════════════════════════╝

Make sure the orchestrator is running:
  cd Python && source ../.venv/bin/activate && uvicorn orchestrator:app

Then open http://localhost:{DEMO_PORT} in your browser.
Press Ctrl+C to stop.
""")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=DEMO_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
