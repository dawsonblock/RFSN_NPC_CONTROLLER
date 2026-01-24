import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ORCHESTRATOR_BASE_URL = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
ORCHESTRATOR_STREAM_PATH = os.getenv("ORCHESTRATOR_STREAM_PATH", "/api/dialogue/stream")
EPISODES_DIR = Path("episodes")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-chat-backend")

# Ensure episodes directory exists
EPISODES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RFSN Web Chat Proxy")

# Mount episodes directory for downloading logs
from fastapi.staticfiles import StaticFiles
app.mount("/api/episodes", StaticFiles(directory=EPISODES_DIR), name="episodes")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev
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

@app.on_event("startup")
async def startup_event():
    logger.info(f"Orchestrator Base URL: {ORCHESTRATOR_BASE_URL}")
    logger.info(f"Orchestrator Stream Path: {ORCHESTRATOR_STREAM_PATH}")
    logger.info(f"Episodes Directory: {EPISODES_DIR.absolute()}")

def log_event(session_id: str, event_type: str, data: Any):
    """Append event to session JSONL file"""
    try:
        file_path = EPISODES_DIR / f"{session_id}.jsonl"
        entry = {
            "t": event_type,
            "ts": datetime.utcnow().isoformat(),
            "data": data
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log event: {e}")

@app.post("/api/chat/stream")
async def proxy_chat_stream(request: ChatRequest):
    """
    Proxy request to orchestrator and stream back SSE events.
    """
    logger.info(f"Received chat request for NPC {request.npc_name} (Session: {request.session_id})")
    
    # 1. Log User Input
    log_event(request.session_id, "user", {"text": request.player_text})

    # Prepare payload for Orchestrator
    # Ensure npc_name is in npc_state as required by RFSNState
    npc_state = request.npc_state.copy()
    npc_state["npc_name"] = request.npc_name 

    orchestrator_payload = {
        "user_input": request.player_text,
        "npc_state": npc_state,
        "enable_voice": request.enable_voice, 
                               # Actually user wants "streaming assistant output".
                               # RFSN generates text chunks. Voice generation happens on orchestrator side.
                               # If we disable voice, orchestrator might skip TTS but still stream text?
                               # Looking at orchestrator code: streaming_engine.voice.process_stream is used.
                               # It yields SentenceChunk. Even if TTS is mock or disabled, it yields chunks.
                               # Let's keep enable_voice=True or False? default is True.
                               # Let's set False to save resources if only text is needed, 
                               # BUT verify if orchestrator streams text if voice is false.
                               # Reading code: `if request.enable_voice` isn't checked in `stream_dialogue`.
                               # It seems `enable_voice` field exists in DialogueRequest but isn't explicitly used to gate the generator?
                               # Actually `DialogueRequest` has `enable_voice`.
                               # In `stream_dialogue`: `request` argument.
                               # It initializes `piper_engine` etc at startup.
                               # `streaming_engine.generate_streaming` is called.
                               # It generates tokens and pipes to `self.voice.process_stream`.
                               # So text IS generated.
        "tts_engine": "kokoro" 
    }

    url = f"{ORCHESTRATOR_BASE_URL}{ORCHESTRATOR_STREAM_PATH}"
    
    async def event_generator():
        client = httpx.AsyncClient(timeout=60.0)
        try:
            async with client.stream("POST", url, json=orchestrator_payload) as response:
                if response.status_code != 200:
                    error_msg = f"Orchestrator returned status {response.status_code}"
                    logger.error(error_msg)
                    yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"
                    return

                # Buffer for full assistant text to log at end
                assistant_full_text = ""
                is_first_event = True

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[len("data: "):].strip()
                    if not data_str:
                        continue
                        
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Logic to determine event type
                    # First event from orchestrator is always metadata: { "player_signal": ... }
                    # Subsequent events are { "sentence": ..., "is_final": ... }
                    # Or { "error": ... }

                    if "error" in data:
                        logger.error(f"Orchestrator error: {data['error']}")
                        yield f"event: error\ndata: {json.dumps({'message': data['error']})}\n\n"
                        continue

                    if is_first_event:
                        # This is the META event
                        is_first_event = False
                        # Check if it has 'player_signal' etc.
                        if "player_signal" in data or "npc_action" in data:
                            log_event(request.session_id, "meta", data)
                            yield f"event: meta\ndata: {json.dumps(data)}\n\n"
                            continue
                        else:
                            # Fallback if first event isn't meta (shouldn't happen with current code)
                            pass
                    
                    # Text event
                    if "sentence" in data:
                        text_chunk = data["sentence"]
                        # We send delta. orchestrator sends "sentence" which is a full sentence chunk?
                        # Looking at `SentenceChunk`: text is a full sentence or fragment?
                        # `StreamTokenizer` yields sentences.
                        # `data: { 'sentence': clean_text ... }`
                        # So it chunks by sentence.
                        
                        assistant_full_text += text_chunk + " "
                        
                        # Send to frontend
                        payload = {
                            "text": text_chunk, # This is a sentence chunk
                            "done": False
                        }
                        yield f"event: text\ndata: {json.dumps(payload)}\n\n"

                # Stream finished
                log_event(request.session_id, "assistant", {"text": assistant_full_text.strip()})
                yield f"event: done\ndata: {{}}\n\n"

        except Exception as e:
            logger.error(f"Proxy error: {e}")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            await client.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Memory Editor API
# -----------------------------------------------------------------------------

MEMORY_DIR = Path("../../memory")  # Relative to web_chat_ui/backend

@app.get("/api/memories")
async def list_memories():
    """List all available NPC memory files."""
    if not MEMORY_DIR.exists():
        return []
    
    files = []
    for f in MEMORY_DIR.glob("*.json"):
        files.append(f.name)
    return sorted(files)

@app.get("/api/memories/{filename}")
async def get_memory(filename: str):
    """Get the content of a memory file."""
    file_path = MEMORY_DIR / filename
    
    # Simple path traversal check
    if not file_path.resolve().is_relative_to(MEMORY_DIR.resolve()):
         raise HTTPException(status_code=403, detail="Invalid path")
         
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Memory file not found")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memories/{filename}")
async def update_memory(filename: str, request: Request):
    """Update an existing memory file."""
    file_path = MEMORY_DIR / filename
    
    if not file_path.resolve().is_relative_to(MEMORY_DIR.resolve()):
         raise HTTPException(status_code=403, detail="Invalid path")
         
    try:
        content = await request.json()
        # Verify it's valid JSON list (standard memory format) or object
        if not isinstance(content, (list, dict)):
             raise HTTPException(status_code=400, detail="Content must be a JSON list or object")
             
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
            
        return {"status": "success", "file": filename}
    except Exception as e:
        logger.error(f"Failed to save memory {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CreateMemoryRequest(BaseModel):
    filename: str
    content: Any = []

@app.post("/api/memories")
async def create_memory(payload: CreateMemoryRequest):
    """Create a new memory file."""
    filename = payload.filename
    if not filename.endswith(".json"):
        filename += ".json"
        
    file_path = MEMORY_DIR / filename
    
    if not file_path.resolve().is_relative_to(MEMORY_DIR.resolve()):
         raise HTTPException(status_code=403, detail="Invalid path")
         
    if file_path.exists():
        raise HTTPException(status_code=409, detail="File already exists")
        
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload.content, f, indent=2)
        return {"status": "created", "file": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Print discovered path
    print(f"DISCOVERED ORCHESTRATOR ROUTE: {ORCHESTRATOR_STREAM_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=3001)
