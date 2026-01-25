import logging
import json
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

from services.dialogue_service import DialogueService
from prometheus_metrics import inc_requests

router = APIRouter(prefix="/api/dialogue", tags=["dialogue"])
logger = logging.getLogger("orchestrator")

class DialogueRequest(BaseModel):
    user_input: str
    npc_state: Dict[str, Any]
    enable_voice: bool = True
    tts_engine: str = "piper"  # piper or xvasynth

@router.post("/stream")
async def stream_dialogue(request: Request, body: DialogueRequest):
    """
    Main dialogue streaming endpoint.
    Delegates to DialogueService.
    """
    inc_requests()
    
    # Validation
    if not body.user_input.strip():
        raise HTTPException(status_code=400, detail="Empty input")
        
    service: DialogueService = request.app.state.dialogue_service
    
    # Check readiness
    # (Service checks internals, but we can do a quick check here if needed)
    
    return StreamingResponse(
        service.stream_dialogue(body, memory_dir="memory"),
        media_type="text/event-stream"
    )
