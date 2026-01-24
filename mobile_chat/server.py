#!/usr/bin/env python3
"""
Mobile Chat Server - All-in-one server for iPhone access
Serves the mobile UI and proxies chat requests to the orchestrator.
"""

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import httpx

# Config
ORCHESTRATOR_URL = "http://localhost:8000"
STATIC_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mobile-server")

app = FastAPI(title="RFSN Mobile Chat")

# CORS for any origin (mobile access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    npc_id: str
    npc_name: str
    player_text: str
    npc_state: Dict[str, Any]
    session_id: str
    enable_voice: bool = False

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the mobile chat UI"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Mobile Chat UI not found</h1>")

@app.post("/api/chat/stream")
async def proxy_chat_stream(request: ChatRequest):
    """Proxy chat requests to orchestrator"""
    logger.info(f"Chat request for NPC: {request.npc_name}")
    
    orchestrator_payload = {
        "user_input": request.player_text,
        "npc_state": request.npc_state,
        "enable_voice": request.enable_voice,
        "tts_engine": "kokoro"
    }
    
    url = f"{ORCHESTRATOR_URL}/api/dialogue/stream"
    
    async def event_generator():
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", url, json=orchestrator_payload) as response:
                    if response.status_code != 200:
                        yield f"event: error\ndata: {{\"message\": \"API error: {response.status_code}\"}}\n\n"
                        return
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        data_str = line[len("data: "):].strip()
                        if not data_str:
                            continue
                        
                        try:
                            data = json.loads(data_str)
                            if "sentence" in data:
                                yield f"event: text\ndata: {json.dumps({'text': data['sentence'], 'done': False})}\n\n"
                            elif "error" in data:
                                yield f"event: error\ndata: {json.dumps({'message': data['error']})}\n\n"
                        except json.JSONDecodeError:
                            continue
                    
                    yield "event: done\ndata: {}\n\n"
                    
            except Exception as e:
                logger.error(f"Proxy error: {e}")
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ORCHESTRATOR_URL}/api/health")
            orchestrator_status = resp.json() if resp.status_code == 200 else {"status": "down"}
    except:
        orchestrator_status = {"status": "unreachable"}
    
    return {
        "status": "healthy",
        "orchestrator": orchestrator_status
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Mobile Chat Server starting on http://0.0.0.0:8080")
    print("📱 Access from iPhone: http://<your-ip>:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
