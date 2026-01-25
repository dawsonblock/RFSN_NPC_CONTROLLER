from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
from pathlib import Path

from memory_manager import ConversationManager, list_backups
from prometheus_metrics import inc_errors
from security import require_auth
from hot_config import get_config

router = APIRouter(prefix="/api/memory", tags=["memory"])

# Config path (should be injected or global ref)
MEMORY_DIR = Path("memory")

class SafeResetRequest(BaseModel):
    npc_name: str

class RestoreRequest(BaseModel):
    filename: str

@router.get("/{npc_name}/stats")
async def get_memory_stats(npc_name: str):
    """Get usage stats for specific NPC memory"""
    try:
        memory = ConversationManager(npc_name, str(MEMORY_DIR))
        return {
            "npc_name": npc_name,
            "message_count": len(memory),
            "token_usage": memory.estimate_token_usage(),
            "last_active": "now"  # Placeholder
        }
    except Exception as e:
        inc_errors()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{npc_name}/safe_reset")
async def safe_reset_memory(npc_name: str, request: SafeResetRequest):
    """Reset memory but keep a backup reference"""
    if npc_name != request.npc_name:
        raise HTTPException(status_code=400, detail="NPC Name mismatch")
    
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    backup_path = memory.create_backup()
    memory.clear()
    
    return {
        "status": "reset",
        "backup_created": backup_path is not None,
        "backup_path": backup_path,
        "npc_name": npc_name
    }

@router.get("/{npc_name}/backups", dependencies=[Depends(require_auth)])
async def get_backups(npc_name: str):
    """List available backups for an NPC"""
    return {
        "npc_name": npc_name,
        "backups": list_backups(str(MEMORY_DIR), npc_name)
    }

@router.post("/restore", dependencies=[Depends(require_auth)])
async def restore_backup(request: RestoreRequest):
    """Restore a specific backup file"""
    backup_path = MEMORY_DIR / "backups" / request.filename
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")
        
    npc_name = request.filename.split("_backup_")[0]
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    
    try:
        memory.load_from_backup(str(backup_path))
        return {
            "status": "restored",
            "npc_name": npc_name,
            "messages_restored": len(memory),
            "backup_filename": request.filename
        }
    except Exception as e:
        inc_errors()
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@router.delete("/{npc_name}", dependencies=[Depends(require_auth)])
async def clear_memory(npc_name: str):
    """Clear memory without backup"""
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    memory.clear()
    return {"status": "cleared", "npc_name": npc_name}
