#!/usr/bin/env python3
"""
RFSN GenAI Orchestrator - Production Streaming Build
Complete system with Kokoro TTS, Ollama LLM, security, metrics, and multi-NPC support.
"""
from version import ORCHESTRATOR_VERSION, STREAMING_ENGINE_VERSION, get_version_string
from runtime_state import Runtime, RuntimeState

import asyncio
import json
import logging
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from learning.learning_contract import (
    LearningContract, LearningConstraints, StateSnapshot as LearningStateSnapshot,
    LearningUpdate, EvidenceType, WriteGateError
)
from reward_shaping import RewardAccumulator

# Import core modules
# Import core modules
from streaming_engine import RFSNState, StreamingMetrics # Keep types top-level
# Lazy loaded in startup: StreamingMantellaEngine
# Lazy loaded in startup: KokoroTTSEngine, setup_kokoro_voice
from voice_router import VoiceRouter, VoiceRequest, VoiceIntensity, VoiceConfig
from ollama_client import OllamaClient, ensure_ollama_ready
from memory_manager import ConversationManager, list_backups

# Import enhancements
from model_manager import ModelManager, setup_models
from security import setup_security, APIKeyManager, JWTManager, require_auth, optional_auth
from structured_logging import configure_logging, get_logger, RequestLoggingMiddleware
from multi_npc import MultiNPCManager
from prometheus_metrics import router as metrics_router, registry, inc_requests, inc_errors, observe_request_duration, inc_tokens, observe_first_token
from hot_config import init_config, get_config
# Lazy loaded: XVASynthEngine

# Import learning layer
from learning import (
    PolicyAdapter, RewardModel, Trainer, ActionMode
)
from learning.contextual_bandit import ContextualBandit, BanditContext
from learning.mode_bandit import ModeBandit, PhrasingMode, ModeContext
from learning.metrics_guard import MetricsGuard
from learning.temporal_memory import TemporalMemory

from memory_governance import MemoryGovernance, GovernedMemory, MemoryType, MemorySource
from intent_extraction import HybridIntentGate, IntentExtractor, IntentType, SafetyFlag
from streaming_pipeline import StreamingPipeline, DropPolicy, TimeoutConfig
from observability import StructuredLogger, MetricsCollector
from event_recorder import EventRecorder, EventType
from state_machine import StateMachine, RFSNStateMachine

# Import world model and action scoring
from world_model import (
    WorldModel, StateSnapshot, NPCAction, PlayerSignal
)
from action_scorer import ActionScorer, UtilityFunction
from llm_action_prompts import render_action_block
from emotional_tone import get_emotion_manager, EmotionalTone
import re

# Configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
MEMORY_DIR = Path(__file__).parent.parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
API_KEYS_PATH = Path(__file__).parent.parent / "api_keys.json"

# Initialize Hot Config
config_watcher = init_config(str(CONFIG_PATH))
config = config_watcher.get_all()

# Configure Logging
configure_logging(
    level=config.get("log_level", "INFO"),
    json_format=config.get("json_logging", True)
)
logger = get_logger("orchestrator")


# API Models
class DialogueRequest(BaseModel):
    user_input: str
    npc_state: Dict[str, Any]
    enable_voice: bool = True
    tts_engine: str = "piper"  # piper or xvasynth

class SafeResetRequest(BaseModel):
    npc_name: str

class RestoreRequest(BaseModel):
    filename: str

class PerformanceSettings(BaseModel):
    temperature: Optional[float] = None
    max_queue_size: Optional[int] = None
    max_tokens: Optional[int] = None


# FastAPI App
app = FastAPI(
    title=f"RFSN GenAI Orchestrator v{ORCHESTRATOR_VERSION}",
    description="Production-hardened streaming system with Security, Metrics, and Multi-NPC capabilities",
    version=ORCHESTRATOR_VERSION
)

# Setup Security (CORS + Rate Limiting)
setup_security(app)

# Setup Request Logging
app.add_middleware(RequestLoggingMiddleware)

# Include Prometheus Metrics
app.include_router(metrics_router, tags=["monitoring"])

# Global instances - wrapped by runtime for atomic swaps
runtime = Runtime()
streaming_engine: Optional["StreamingMantellaEngine"] = None  # Lazy loaded
tts_engine: Optional["KokoroTTSEngine"] = None  # Lazy loaded
voice_router: Optional[VoiceRouter] = None  # Dual-TTS router (Chatterbox-Turbo + Full)
xva_engine: Optional["XVASynthEngine"] = None  # Lazy loaded
multi_manager = MultiNPCManager()
active_ws: List[WebSocket] = []
conversation_managers: Dict[str, ConversationManager] = {}
conversation_managers_lock = asyncio.Lock()

# Learning layer instances
policy_adapter: Optional[PolicyAdapter] = None
reward_model: Optional[RewardModel] = None
trainer: Optional[Trainer] = None
learning_contract: Optional[LearningContract] = None
memory_governance: Optional[MemoryGovernance] = None
memory_consolidator = None  # MemoryConsolidator instance
intent_gate: Optional[HybridIntentGate] = None
streaming_pipeline: Optional[StreamingPipeline] = None
structured_logger: Optional[StructuredLogger] = None
metrics_collector: Optional[MetricsCollector] = None
# Routers
from routers import dialogue, system, memory, cgw
from services.dialogue_service import DialogueService


@app.on_event("startup")
async def startup_event():
    """Initialize all engines on startup"""
    global streaming_engine, tts_engine, xva_engine
    
    logger.info("=" * 70)
    logger.info(get_version_string() + " - STARTING UP")
    logger.info("=" * 70)
    
    # Verify models (optional for Ollama mode)
    if not os.environ.get("SKIP_MODELS"):
        await asyncio.to_thread(setup_models)
    else:
        logger.warning("Skipping model downloads (SKIP_MODELS=1)")
    
    # Setup xVASynth (legacy support)
    if config_watcher.get("xvasynth_enabled", False):
        from xvasynth_engine import XVASynthEngine
        xva_engine = XVASynthEngine()
        logger.info(f"xVASynth initialized (Available: {xva_engine.available})")

    # Initialize TTS (Kokoro)
    tts_config = config.get("tts", {})
    tts_backend = tts_config.get("backend", "kokoro")
    
    if tts_backend == "kokoro":
        try:
            project_root = Path(__file__).parent.parent
            kokoro_model = tts_config.get("kokoro_model", "Models/kokoro/kokoro-v1.0.onnx")
            kokoro_voices = tts_config.get("kokoro_voices", "Models/kokoro/voices-v1.0.bin")
            
            model_path = str(project_root / kokoro_model)
            voices_path = str(project_root / kokoro_voices)
            
            # Auto-download if not present
            if not Path(model_path).exists() or not Path(voices_path).exists():
                logger.info("[Kokoro] Model not found, downloading...")
                from kokoro_tts import setup_kokoro_voice
                model_path, voices_path = setup_kokoro_voice()
            
            if model_path and voices_path:
                tts_speed = tts_config.get("speed", 1.0)
                voice = tts_config.get("voice", "af_bella")
                q_size = tts_config.get("max_queue_size", 10)
                
                logger.info(f"Setting up Kokoro TTS: voice={voice}, speed={tts_speed}")
                from kokoro_tts import KokoroTTSEngine
                tts_engine = KokoroTTSEngine(
                    model_path=model_path,
                    voices_path=voices_path,
                    voice=voice,
                    speed=tts_speed,
                    max_queue_size=q_size
                )
                logger.info("Kokoro TTS Engine initialized successfully")
            else:
                logger.warning("Kokoro TTS model download failed. Running in mock mode.")
                tts_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            tts_engine = None

    # Initialize streaming LLM with Ollama
    llm_config = config.get("llm", {})
    llm_backend = llm_config.get("backend", "ollama")
    
    if os.environ.get("SKIP_MODELS"):
        logger.info("Using MOCK LLM (SKIP_MODELS set)")
        from streaming_engine import StreamingMantellaEngine
        streaming_engine = StreamingMantellaEngine(backend="mock")
    elif llm_backend == "ollama":
        # Ollama backend
        ollama_config = {
            "ollama_host": llm_config.get("ollama_host", "http://localhost:11434"),
            "ollama_model": llm_config.get("ollama_model", "llama3.2"),
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 150)
        }
        from streaming_engine import StreamingMantellaEngine
        streaming_engine = StreamingMantellaEngine(
            backend="ollama",
            ollama_config=ollama_config
        )
        logger.info(f"Ollama LLM initialized (model={ollama_config['ollama_model']})")
    else:
        # Fallback to llama-cpp
        from model_manager import ensure_llm_model_exists
        cfg_model_path = llm_config.get("model_path", config_watcher.get("model_path"))
        resolved = ensure_llm_model_exists(cfg_model_path)
        
        if resolved is None:
            logger.error(f"Configured model_path not found: {cfg_model_path}")
            # streaming_engine = StreamingMantellaEngine(backend="mock") # Fallback
            from streaming_engine import StreamingMantellaEngine
            streaming_engine = StreamingMantellaEngine(backend="mock")
        else:
            from streaming_engine import StreamingMantellaEngine
            streaming_engine = StreamingMantellaEngine(
                model_path=str(resolved),
                backend="llama_cpp"
            )
    
    # Connect TTS engine
    if tts_engine and streaming_engine:
        streaming_engine.voice.set_tts_engine(tts_engine)
    
    # Initialize VoiceRouter for dual-TTS engine support (Chatterbox-Turbo + Full)
    global voice_router
    tts_backend = tts_config.get("backend", "kokoro")
    if tts_backend == "chatterbox":
        try:
            # Load config thresholds
            thresholds = tts_config.get("intensity_thresholds", {})
            voice_config = VoiceConfig(
                high_emotion_threshold=thresholds.get("high_emotion", 0.7),
                high_valence_threshold=thresholds.get("high_valence", 0.7),
                high_arousal_threshold=thresholds.get("high_arousal", 0.8)
            )
            
            # Initialize router with Chatterbox settings
            chatterbox_cfg = tts_config.get("chatterbox", {})
            device = chatterbox_cfg.get("device", "cuda")
            voice_router = VoiceRouter(
                config=voice_config,
                device=device,
                enable_queue=True,
                fallback_to_kokoro=True
            )
            
            # Register voice prompts
            voice_prompts = tts_config.get("voice_prompts", {})
            for npc_id, audio_path in voice_prompts.items():
                voice_router.set_voice_prompt(npc_id, audio_path)
            
            logger.info(f"VoiceRouter initialized (device={device}, voices={len(voice_prompts)})")
        except Exception as e:
            logger.error(f"VoiceRouter initialization failed: {e}, falling back to Kokoro only")
            voice_router = None
    else:
        logger.info("VoiceRouter disabled (TTS backend is not 'chatterbox')")
    
    # Sync global API Key Manager
    from security import api_key_manager
    api_key_manager.keys_file = str(API_KEYS_PATH)
    api_key_manager._load_keys()
    
    # Initialize learning layer
    global policy_adapter, reward_model, trainer, learning_contract, memory_governance, intent_gate
    global streaming_pipeline, structured_logger, metrics_collector, event_recorder, state_machine, memory_consolidator
    policy_adapter = PolicyAdapter(epsilon=0.08)
    reward_model = RewardModel()
    trainer = Trainer(learning_rate=0.05, decay_rate=0.9999)
    
    # Initialize LearningContract with constraints
    constraints = LearningConstraints()
    learning_contract = LearningContract(
        constraints=constraints,
        snapshot_dir=Path(__file__).parent.parent / "data" / "learning" / "snapshots"
    )
    
    # Initialize MemoryGovernance
    memory_governance = MemoryGovernance(
        storage_path=Path(__file__).parent.parent / "data" / "memory" / "governed"
    )
    
    # Initialize MemoryConsolidator (using simple LLM shim)
    # Note: We need a blocking LLM call for consolidation. 
    # For now, we'll try to use a simple completion via llama-cpp (if available) or mock it.
    # In full production, we'd hook into the streaming engine's queue or a separate model.
    # For now, consolidation is manual or placeholder.
    from memory_consolidator import MemoryConsolidator

    def _blocking_llm_generate(prompt: str, max_tokens: int = 220, temperature: float = 0.2) -> str:
        """
        Minimal blocking completion for consolidation.
        Uses llama-cpp via the loaded engine when available; otherwise returns "".
        """
        try:
            if not streaming_engine or not getattr(streaming_engine, "llm", None):
                return ""

            out = streaming_engine.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<|eot_id|>", "<|end|>", "</s>"],
                stream=False,
                echo=False,
                temperature=temperature,
            )

            if isinstance(out, dict) and out.get("choices"):
                return (out["choices"][0].get("text") or "").strip()

            return ""
        except Exception:
            logger.exception("Error during blocking LLM generation for consolidation")
            return ""

    memory_consolidator = MemoryConsolidator(memory_governance, llm_generate_fn=_blocking_llm_generate)
    
    # Initialize HybridIntentGate with LLM-powered extraction (Ollama + regex fallback)
    llm_config = config.get("llm", {})
    intent_gate = HybridIntentGate(
        block_unsafe=True,
        require_min_confidence=0.3,
        use_llm=llm_config.get("intent_llm_enabled", True),
        ollama_host=llm_config.get("ollama_host", "http://localhost:11434"),
        model=llm_config.get("intent_model", llm_config.get("model", "llama3.2"))
    )
    logger.info(f"HybridIntentGate initialized (LLM={llm_config.get('intent_llm_enabled', True)})")
    
    # Initialize StreamingPipeline for message delivery guarantees
    # Initialize StreamingPipeline for message delivery guarantees
    streaming_pipeline = StreamingPipeline(
        text_queue_size=50,
        audio_queue_size=50,
        text_drop_policy=DropPolicy.DROP_NEWEST,
        audio_drop_policy=DropPolicy.DROP_NEWEST,
        timeout_config=TimeoutConfig(llm_generation_timeout=30.0)
    )
    
    # Initialize Observability components
    structured_logger = StructuredLogger(name="rfsn-orchestrator")
    metrics_collector = MetricsCollector()
    
    # Initialize EventRecorder for deterministic replay
    event_recorder = EventRecorder(
        session_id=str(int(time.time())),
        output_dir=Path(__file__).parent.parent / "data" / "recordings"
    )
    
    # Initialize StateMachine for RFSN state invariants
    state_machine = RFSNStateMachine()

    # Initialize WorldModel for consequence prediction
    transitions_path = Path("data/world_transitions.json")
    world_model = WorldModel(
        retrieval_k=5,
        transitions_path=transitions_path if transitions_path.exists() else None
    )

    # Initialize ActionScorer for decision pipeline
    action_scorer = ActionScorer(
        world_model=world_model,
        utility_fn=UtilityFunction(
            affinity_weight=0.3,
            trust_weight=0.2,
            fear_penalty=0.3,
            quest_bonus=0.2
        )
    )

    # Initialize v1.3 Learning Components
    contextual_bandit = ContextualBandit(Path(__file__).parent.parent / "data/learning/contextual_bandit.json")
    mode_bandit = ModeBandit(Path(__file__).parent.parent / "data/learning/mode_bandit.json")
    metrics_guard = MetricsGuard(Path(__file__).parent.parent / "data/learning/metrics_guard.json")

    # Initialize TemporalMemory for anticipatory action selection
    temporal_memory_config = config.get("learning", {}).get("temporal_memory", {})
    temporal_memory = TemporalMemory(
        max_size=temporal_memory_config.get("max_size", 50),
        decay_rate=temporal_memory_config.get("decay_rate", 0.95),
        adjustment_scale=temporal_memory_config.get("adjustment_scale", 0.1)
    )
    
    # Load persisted learning state (if available)
    try:
        temporal_memory.load()
        logger.info(f"Loaded temporal memory ({len(temporal_memory)} experiences)")
    except Exception as e:
        logger.warning(f"Could not load temporal memory: {e}")
    
    # Load persisted emotional states
    try:
        emotion_manager = get_emotion_manager()
        emotion_manager.load()
        logger.info(f"Loaded emotional states ({len(emotion_manager.list_states())} NPCs)")
    except Exception as e:
        logger.warning(f"Could not load emotional states: {e}")
    
    # Swap engines into RuntimeState atomically (Patch 1: atomic engine pointers)
    runtime.swap(RuntimeState(
        streaming_engine=streaming_engine,
        tts_engine=tts_engine,
        xva_engine=xva_engine,
        policy_adapter=policy_adapter,
        trainer=trainer,
        reward_model=reward_model,
        learning_contract=learning_contract,
        memory_governance=memory_governance,
        intent_gate=intent_gate,
        streaming_pipeline=streaming_pipeline,
        observability=structured_logger,
        event_recorder=event_recorder,
        state_machine=state_machine,
        world_model=world_model,
        action_scorer=action_scorer,
        contextual_bandit=contextual_bandit,
        mode_bandit=mode_bandit,
        metrics_guard=metrics_guard,
        temporal_memory=temporal_memory
    ))
    # Initialize DialogueService
    dialogue_service = DialogueService(runtime, config_watcher, multi_manager)
    app.state.dialogue_service = dialogue_service

    logger.info("Runtime state initialized atomically")

    # Generate initial API Key if missing
    if not API_KEYS_PATH.exists():
        admin_key = api_key_manager.generate_key("admin", ["admin_role"])
        logger.warning(f"Created initial Admin API Key: {admin_key}")
    
    logger.info("Startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown with learning state persistence"""
    logger.info("Shutting down engines...")
    
    # Save learning state
    try:
        state = runtime.get()
        
        # Save temporal memory
        if state.temporal_memory:
            state.temporal_memory.save()
            logger.info("Saved temporal memory")
        
        # Save bandit state
        if state.npc_action_bandit:
            state.npc_action_bandit.save()
            logger.info("Saved bandit state")
        
        # Save emotional states
        emotion_manager = get_emotion_manager()
        emotion_manager.save()
        logger.info("Saved emotional states")
        
    except Exception as e:
        logger.error(f"Failed to save learning state: {e}")
    
    # Shutdown engines
    if tts_engine:
        tts_engine.shutdown()
    if voice_router:
        voice_router.shutdown()
    if xva_engine:
        xva_engine.shutdown()
    if streaming_engine:
        streaming_engine.shutdown()
    
    logger.info("Shutdown complete")


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """Real-time metrics streaming to dashboard"""
    await websocket.accept()
    active_ws.append(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "Metrics stream active"})
        while True:
            # Broadcast live metrics every 0.5s
            if streaming_engine and streaming_engine.voice:
                await websocket.send_json({
                    "type": "metrics",
                    "metrics": asdict(streaming_engine.voice.metrics),
                    "config": config_watcher.get_all()
                })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_ws:
            active_ws.remove(websocket)


async def broadcast_metrics(metrics: StreamingMetrics):
    """Broadcast performance metrics to all connected dashboards"""
    for ws in list(active_ws):
        try:
            await ws.send_json({
                "type": "metrics",
                "first_token_ms": metrics.first_token_ms,
                "first_sentence_ms": metrics.first_sentence_ms,
                "total_generation_ms": metrics.total_generation_ms,
                "tts_queue_size": metrics.tts_queue_size,
                "dropped_sentences": metrics.dropped_sentences
            })
        except Exception:
            pass





@app.post("/api/memory/{npc_name}/safe_reset", dependencies=[Depends(require_auth)])
async def safe_reset_memory(npc_name: str):
    """Safe reset with automatic backup"""
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    
    if len(memory) == 0:
        return {"status": "nothing_to_backup", "message": "No conversation history to backup"}
    
    backup_path = memory.safe_reset()
    
    if backup_path:
        backup_data = json.loads(backup_path.read_text())
        return {
            "status": "reset_with_backup",
            "backup_path": str(backup_path),
            "messages_archived": len(backup_data),
            "file_size_bytes": backup_path.stat().st_size,
            "timestamp": backup_path.stat().st_mtime
        }
    
    inc_errors()
    return {"status": "error", "message": "Failed to create backup"}


@app.get("/api/memory/{npc_name}/stats")
async def get_memory_stats(npc_name: str):
    """Get detailed memory statistics"""
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    return memory.get_stats()


@app.get("/api/memory/backups")
async def get_backups():
    """List all available memory backups"""
    backups = list_backups(str(MEMORY_DIR))
    return {"backups": backups, "total_backups": len(backups)}


@app.post("/api/memory/restore", dependencies=[Depends(require_auth)])
async def restore_backup(request: RestoreRequest):
    """Restore NPC memory from backup file"""
    backup_path = MEMORY_DIR / request.filename
    
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail=f"Backup not found: {request.filename}")
    
    npc_name = request.filename.split("_backup_")[0]
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    
    try:
        memory.load_from_backup(backup_path)
        return {
            "status": "restored",
            "npc_name": npc_name,
            "messages_restored": len(memory),
            "backup_filename": request.filename
        }
    except Exception as e:
        inc_errors()
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")


@app.delete("/api/memory/{npc_name}", dependencies=[Depends(require_auth)])
async def clear_memory(npc_name: str):
    """Clear memory without backup"""
    memory = ConversationManager(npc_name, str(MEMORY_DIR))
    memory.clear()
    return {"status": "cleared", "npc_name": npc_name}


# Mount Routers
app.include_router(system.router)  # Includes health, status, metrics
app.include_router(memory.router)  # Memory management
app.include_router(dialogue.router) # Streaming endpoints
app.include_router(cgw.router)     # CGW debug endpoints


@app.get("/api/learning/stats")
async def get_learning_stats():
    """Get learning statistics for visualization."""
    npc_action_bandit = runtime.get().npc_action_bandit
    temporal_memory = runtime.get().temporal_memory
    
    stats = {
        "temporal_memory": {
            "size": len(temporal_memory) if temporal_memory else 0,
            "max_size": temporal_memory.max_size if temporal_memory else 0,
        },
        "bandit": {
            "total_arms": 0,
            "total_updates": 0,
        }
    }
    
    if npc_action_bandit:
        total_arms = 0
        total_updates = 0
        for key, arms in npc_action_bandit._arms.items():
            total_arms += len(arms)
            for arm in arms.values():
                total_updates += arm.get("n", 0)
        stats["bandit"]["total_arms"] = total_arms
        stats["bandit"]["total_updates"] = int(total_updates)
    
    return stats


@app.get("/api/learning/temporal")
async def get_temporal_memory():
    """Get temporal memory contents for visualization."""
    temporal_memory = runtime.get().temporal_memory
    
    if not temporal_memory:
        return {"experiences": [], "stats": {}}
    
    # Get recent experiences (don't expose full state for privacy)
    experiences = []
    for exp in list(temporal_memory.memory)[-10:]:  # Last 10
        experiences.append({
            "action": exp.action.value,
            "reward": exp.reward,
            "age_seconds": (temporal_memory._now() - exp.timestamp)
        })
    
    return {
        "experiences": experiences,
        "stats": temporal_memory.stats()
    }


@app.get("/api/learning/bandit")
async def get_bandit_arms():
    """Get bandit arm statistics for visualization."""
    npc_action_bandit = runtime.get().npc_action_bandit
    
    if not npc_action_bandit:
        return {"arms": {}}
    
    # Summarize arm stats per context bucket
    summary = {}
    for key, arms in npc_action_bandit._arms.items():
        summary[key] = {}
        for action, stats in arms.items():
            n = stats.get("n", 0)
            if n > 0:
                alpha = stats.get("alpha", 1)
                beta = stats.get("beta", 1)
                mean = alpha / (alpha + beta)
                summary[key][action] = {
                    "trials": int(n),
                    "mean_reward": round(mean, 3)
                }
    
    return {"arms": summary}


@app.get("/api/learning/emotions")
async def get_emotional_states():
    """Get all NPC emotional states for visualization."""
    emotion_manager = get_emotion_manager()
    return {
        "states": emotion_manager.list_states(),
        "tones": [t.value for t in EmotionalTone]
    }


@app.get("/api/learning/emotions/{npc_name}")
async def get_npc_emotional_state(npc_name: str):
    """Get emotional state for a specific NPC."""
    emotion_manager = get_emotion_manager()
    state = emotion_manager.get_state(npc_name)
    return state.to_dict()


@app.post("/api/tune-performance", dependencies=[Depends(require_auth)])
async def tune_performance(settings: PerformanceSettings):
    # Adjust performance settings at runtime
    # Hot config handles the persistence and reload
    import json
    
    current_conf = config_watcher.get_all()
    
    if settings.temperature is not None:
        current_conf["temperature"] = settings.temperature
    if settings.max_queue_size is not None:
        # Validate queue size before applying (Phase 1 fix)
        if not (1 <= settings.max_queue_size <= 50):
            raise HTTPException(status_code=400, detail="max_queue_size must be between 1 and 50")
        current_conf["max_queue_size"] = settings.max_queue_size
    if settings.max_tokens is not None:
        current_conf["max_tokens"] = settings.max_tokens
    
    CONFIG_PATH.write_text(json.dumps(current_conf, indent=2))
    
    # Apply to running engine immediately
    if streaming_engine:
        streaming_engine.apply_tuning(
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_queue_size=settings.max_queue_size
        )
    
    return {"status": "updated", "new_settings": current_conf}



# Mount Dashboard (Must be after API routes to avoid masking)
DASHBOARD_DIR = Path(__file__).parent.parent / "Dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
    logger.info(f"Dashboard mounted at / from {DASHBOARD_DIR}")
else:
    logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("RFSN ORCHESTRATOR v8.2 - PRODUCTION STREAMING MODE")
    print("Security: Enabled | Metrics: Enabled | Multi-NPC: Active")
    print("=" * 70)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
