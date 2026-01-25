from fastapi import APIRouter, Depends
from runtime_state import Runtime
from prometheus_metrics import router as metrics_router
from multi_npc import MultiNPCManager
from emotional_tone import get_emotion_manager, EmotionalTone
import time

router = APIRouter()
runtime = Runtime()
# We need access to multi_manager. In a real app we'd inject it, 
# but for now we can instantiate the singleton or access via app state if we set it.
# MultiNPCManager is a singleton pattern, so this safe:
multi_manager = MultiNPCManager()

@router.get("/api/health")
async def health_check():
    """Operational health check"""
    state_ok = runtime.get() is not None
    engine = runtime.get().streaming_engine if state_ok else None
    
    return {
        "status": "healthy" if engine else "initializing",
        "model_loaded": engine is not None,
        "active_npc": list(multi_manager.sessions.keys()),
        "uptime": time.time()
    }

@router.get("/api/status")
async def get_status():
    """System status"""
    state = runtime.get()
    engine = state.streaming_engine if state else None
    tts = state.tts_engine if state else None
    
    return {
        "status": "healthy",
        "version": "9.0",
        "components": {
            "streaming_llm": engine is not None,
            "kokoro_tts": tts is not None,
            "active_npc_sessions": len(multi_manager.sessions)
        }
    }

@router.get("/api/learning/stats")
async def get_learning_stats():
    """Get learning statistics for visualization."""
    state = runtime.get()
    if not state: return {}
    
    temporal = state.temporal_memory
    bandit = state.contextual_bandit
    
    stats = {
        "temporal_memory": {
            "size": len(temporal) if temporal else 0,
        },
        "bandit": {
            "total_arms": 0,
            "total_updates": 0,
        }
    }
    # (Simplified bandit stats extraction)
    return stats

@router.get("/api/learning/emotions")
async def get_emotional_states():
    """Get all NPC emotional states for visualization."""
    emotion_manager = get_emotion_manager()
    return {
        "states": emotion_manager.list_states(),
        "tones": [t.value for t in EmotionalTone]
    }

# Include prometheus
router.include_router(metrics_router, tags=["monitoring"])
