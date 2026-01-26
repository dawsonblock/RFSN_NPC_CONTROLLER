"""
CGW Debug Router: API endpoints for the Cognitive Global Workspace.

Debug/testing endpoints for:
- Forced signal injection
- CGW state inspection
- Manual attention override
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from learning.cgw_integration import get_cgw_manager, reset_cgw_manager
from world_model import NPCAction

router = APIRouter(prefix="/api/cgw", tags=["cgw", "debug"])
logger = logging.getLogger("cgw.router")


class ForceSignalRequest(BaseModel):
    """Request to inject a forced signal."""
    action: str  # NPCAction enum value (e.g., "refuse", "greet")
    source: str = "debug"  # Source module injecting
    reason: str = "MANUAL_OVERRIDE"  # Reason for override


class ForceSignalResponse(BaseModel):
    """Response from forced signal injection."""
    success: bool
    slot_id: str
    action: str
    reason: str
    queue_depth: int


@router.post("/force", response_model=ForceSignalResponse)
async def force_signal(request: ForceSignalRequest):
    """
    Inject a forced signal into the CGW gate.
    
    Forced signals bypass normal competition and are processed immediately.
    Use cases: safety overrides, combat reactions, emergency responses.
    
    Example:
        POST /api/cgw/force
        {"action": "refuse", "source": "safety_gate", "reason": "TOXIC_CONTENT"}
    """
    try:
        # Validate action
        action = NPCAction(request.action.lower())
    except ValueError:
        valid_actions = [a.value for a in NPCAction]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Valid: {valid_actions}"
        )
    
    cgw = get_cgw_manager()
    
    slot_id = cgw.inject_forced(
        source=request.source,
        npc_action=action,
        reason=request.reason
    )
    
    logger.info(f"[CGW] Forced signal injected: {slot_id} ({action.value})")
    
    return ForceSignalResponse(
        success=True,
        slot_id=slot_id,
        action=action.value,
        reason=request.reason,
        queue_depth=len(cgw.gate.forced_queue)
    )


@router.get("/state")
async def get_cgw_state() -> Dict[str, Any]:
    """
    Get current CGW state and statistics.
    
    Returns:
        Current attended content, cycle counts, queue depths.
    """
    cgw = get_cgw_manager()
    state = cgw.get_current_state()
    
    attended = None
    if state:
        attended = {
            "slot_id": state.content_id(),
            "cycle_id": state.cycle_id,
            "forced": state.causal_trace.forced_override,
            "reason": state.causal_trace.winner_reason.value,
        }
    
    return {
        "stats": cgw.get_stats(),
        "attended_content": attended,
        "has_forced_pending": cgw.has_forced_pending(),
    }


@router.post("/reset")
async def reset_cgw() -> Dict[str, str]:
    """
    Reset the CGW manager.
    
    Clears all state and reinitializes. Use for testing/debugging.
    """
    reset_cgw_manager()
    logger.info("[CGW] Manager reset")
    return {"status": "reset", "message": "CGW manager reset successfully"}


@router.post("/tick")
async def tick_cgw() -> Dict[str, Any]:
    """
    Run one CGW cycle manually.
    
    Selects winner from candidates (or forced queue) and commits to workspace.
    """
    cgw = get_cgw_manager()
    had_winner = cgw.tick()
    
    state = cgw.get_current_state()
    
    return {
        "had_winner": had_winner,
        "cycle": cgw.cgw.cycle_counter,
        "attended": state.content_id() if state else None,
    }
