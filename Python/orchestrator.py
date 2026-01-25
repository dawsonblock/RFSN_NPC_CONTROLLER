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
from streaming_engine import StreamingMantellaEngine, StreamingMetrics, RFSNState
from kokoro_tts import KokoroTTSEngine, setup_kokoro_voice
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
from xvasynth_engine import XVASynthEngine

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
streaming_engine: Optional[StreamingMantellaEngine] = None
tts_engine: Optional[KokoroTTSEngine] = None
voice_router: Optional[VoiceRouter] = None  # Dual-TTS router (Chatterbox-Turbo + Full)
xva_engine: Optional[XVASynthEngine] = None
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
event_recorder: Optional[EventRecorder] = None
state_machine: Optional[StateMachine] = None


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
                model_path, voices_path = setup_kokoro_voice()
            
            if model_path and voices_path:
                tts_speed = tts_config.get("speed", 1.0)
                voice = tts_config.get("voice", "af_bella")
                q_size = tts_config.get("max_queue_size", 10)
                
                logger.info(f"Setting up Kokoro TTS: voice={voice}, speed={tts_speed}")
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
        streaming_engine = StreamingMantellaEngine(backend="mock")
    elif llm_backend == "ollama":
        # Ollama backend
        ollama_config = {
            "ollama_host": llm_config.get("ollama_host", "http://localhost:11434"),
            "ollama_model": llm_config.get("ollama_model", "llama3.2"),
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 150)
        }
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
            streaming_engine = StreamingMantellaEngine(backend="mock")
        else:
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


def _extract_tail_json_payload(raw_text: str) -> tuple[Optional[Dict[str, Any]], bool]:
    """
    Extract authoritative FINAL_JSON block from LLM output.
    
    Returns:
        (payload, is_verified): Tuple of parsed payload and verification status.
        - payload: Parsed dict or None if not found/invalid
        - is_verified: True only if valid FINAL_JSON with required fields
    """
    import json
    import re
    
    payload = None
    is_verified = False
    
    try:
        # Try strict FINAL_JSON: format first (preferred)
        final_json_match = re.search(
            r'FINAL_JSON:\s*```json\s*(\{.*?\})\s*```',
            raw_text, re.DOTALL | re.IGNORECASE
        )
        
        if final_json_match:
            json_text = final_json_match.group(1)
            payload = json.loads(json_text)
            
            # Verify required fields exist
            if all(k in payload for k in ["line", "action"]):
                is_verified = True
                # Validate confidence if present
                if "confidence" in payload:
                    conf = payload["confidence"]
                    if not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
                        payload["confidence"] = 0.5  # Default if invalid
        else:
            # Fallback: try legacy ```json format (but mark as unverified)
            legacy_match = re.search(
                r'```json\s*(\{.*?\})\s*```',
                raw_text, re.DOTALL | re.IGNORECASE
            )
            if legacy_match:
                json_text = legacy_match.group(1)
                payload = json.loads(json_text)
                # Legacy format is not fully verified
                is_verified = False
                logger.warning("Legacy JSON format detected, marking as unverified")
                
    except json.JSONDecodeError as e:
        logger.warning(f"FINAL_JSON parse error: {e}")
        is_verified = False
    except Exception as e:
        logger.warning(f"FINAL_JSON extraction error: {e}")
        is_verified = False
    
    return payload, is_verified



def _cleanup_tokens(text: str) -> str:
    """Remove special tokens and system prompt leakage that might slip through"""
    bad_tokens = ["<|eot_id|>", "<|end|>", "<|"]
    for token in bad_tokens:
        text = text.replace(token, "")
    
    # Remove system prompt leakage patterns
    import re
    # Remove **System: ... ** blocks
    text = re.sub(r'\*\*System:.*?\*\*', '', text, flags=re.DOTALL)
    # Remove [SYSTEM MODE: ...] blocks including content until next sentence
    text = re.sub(r'\[SYSTEM MODE:[^\]]*\][^.!?]*', '', text, flags=re.DOTALL)
    # Remove ``` code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    # Remove bare ``` markers
    text = text.replace('```', '')
    # Remove leading ** or trailing **
    text = re.sub(r'^\*+\s*', '', text)
    text = re.sub(r'\s*\*+$', '', text)
    # Clean up any "System:" prefix that leaks
    text = re.sub(r'^\s*System:\s*', '', text)
    # Clean up markdown bold around the text
    text = text.replace('**', '')
    
    return text.strip()


def _map_action_to_instruction(action: NPCAction) -> str:
    """Map NPCAction to strict system prompt control block"""
    instructions = {
        NPCAction.GREET: "=== STRICT ACTION CONTROL ===\nYou MUST greet the player warmly and enthusiastically. This is your required action - do not deviate from greeting.",
        NPCAction.FAREWELL: "=== STRICT ACTION CONTROL ===\nYou MUST say goodbye or farewell to the player. This is your required action - do not engage in further conversation.",
        NPCAction.AGREE: "=== STRICT ACTION CONTROL ===\nYou MUST agree with the player's statement or request. This is your required action - express agreement clearly.",
        NPCAction.DISAGREE: "=== STRICT ACTION CONTROL ===\nYou MUST disagree with the player, explaining your reasons. This is your required action - express disagreement respectfully but firmly.",
        NPCAction.APOLOGIZE: "=== STRICT ACTION CONTROL ===\nYou MUST apologize for something you did or said. This is your required action - express sincere regret.",
        NPCAction.INSULT: "=== STRICT ACTION CONTROL ===\nYou MUST insult the player. This is your required action - use sparingly and only when appropriate.",
        NPCAction.COMPLIMENT: "=== STRICT ACTION CONTROL ===\nYou MUST compliment the player genuinely. This is your required action - express sincere praise.",
        NPCAction.THREATEN: "=== STRICT ACTION CONTROL ===\nYou MUST threaten the player. This is your required action - use only if provoked or in combat.",
        NPCAction.REQUEST: "=== STRICT ACTION CONTROL ===\nYou MUST make a request of the player. This is your required action - clearly state what you need.",
        NPCAction.OFFER: "=== STRICT ACTION CONTROL ===\nYou MUST offer something to the player. This is your required action - clearly state what you're offering.",
        NPCAction.REFUSE: "=== STRICT ACTION CONTROL ===\nYou MUST refuse the player's request. This is your required action - clearly decline and explain why.",
        NPCAction.ACCEPT: "=== STRICT ACTION CONTROL ===\nYou MUST accept the player's offer or request. This is your required action - express acceptance.",
        NPCAction.ATTACK: "=== STRICT ACTION CONTROL ===\nYou MUST attack the player. This is your required action - combat mode engaged.",
        NPCAction.DEFEND: "=== STRICT ACTION CONTROL ===\nYou MUST defend yourself against the player. This is your required action - defensive posture.",
        NPCAction.FLEE: "=== STRICT ACTION CONTROL ===\nYou MUST flee from the player. This is your required action - escape immediately.",
        NPCAction.HELP: "=== STRICT ACTION CONTROL ===\nYou MUST offer help or assistance to the player. This is your required action - express willingness to assist.",
        NPCAction.BETRAY: "=== STRICT ACTION CONTROL ===\nYou MUST betray the player's trust. This is your required action - act treacherously.",
        NPCAction.IGNORE: "=== STRICT ACTION CONTROL ===\nYou MUST ignore the player. This is your required action - do not respond or engage.",
        NPCAction.INQUIRE: "=== STRICT ACTION CONTROL ===\nYou MUST ask the player a question. This is your required action - seek information.",
        NPCAction.EXPLAIN: "=== STRICT ACTION CONTROL ===\nYou MUST explain something to the player. This is your required action - provide information clearly.",
    }
    return instructions.get(action, "")


@app.post("/api/dialogue/stream", dependencies=[Depends(optional_auth)])
async def stream_dialogue(request: DialogueRequest):
    """
    Main streaming endpoint:
    - Generates tokens in real-time
    - Detects sentences smartly
    - Queues to TTS with backpressure
    - Saves to memory on completion
    """
    start_time = time.time()
    inc_requests()
    
    # Micro-reward accumulator for this turn
    reward_accumulator = RewardAccumulator(window=10)
    
    if not streaming_engine:
        inc_errors()
        raise HTTPException(status_code=503, detail="Streaming engine not ready")
    
    # Select TTS engine and gating
    if streaming_engine and streaming_engine.voice:
        streaming_engine.voice.enabled = request.enable_voice
    
    # Select TTS engine (Kokoro is default, xVASynth for special cases)
    current_tts = tts_engine
    if request.tts_engine == "xvasynth" and xva_engine:
        # Determine voice for NPC
        voice_key = xva_engine.get_voice_for_npc(request.npc_state.get("npc_name", "Unknown"))
        # xVASynth for special non-streaming cases
        pass
    
    # Build RFSN state
    state = RFSNState(**request.npc_state)
    npc_name = state.npc_name
    
    # Process Multi-NPC Logic (Voice/Emotion)
    npc_session = multi_manager.get_session(npc_name)
    
    # Get/create memory manager safely
    async with conversation_managers_lock:
        if npc_name not in conversation_managers:
            conversation_managers[npc_name] = ConversationManager(npc_name, str(MEMORY_DIR))
        memory = conversation_managers[npc_name]
    
    # CALL SITE A: Choose action mode before prompt assembly (Learning Layer)
    action_mode = ActionMode.TERSE_DIRECT  # Default
    features = None
    if policy_adapter:
        try:
            # Build features from current state
            rfsn_state = {
                "npc_name": npc_name,
                "affinity": state.affinity,
                "mood": state.mood,
                "relationship": state.relationship,
                "playstyle": "balanced",  # Could extract from player data
                "recent_sentiment": 0.0  # Could compute from recent turns
            }
            retrieval_stats = {
                "top_k_scores": [],  # Could add if using retrieval
                "contradiction_detected": False
            }
            convo_stats = {
                "turn_count": len(memory.get_context_window(limit=100).split("\n")) if config_watcher.get("memory_enabled") else 0
            }
            
            features = policy_adapter.build_features(rfsn_state, retrieval_stats, convo_stats)
            action_mode = policy_adapter.choose_action_mode(features)
            logger.info(f"Learning: Selected action mode {action_mode.name}")
        except Exception as e:
            logger.error(f"Learning layer error (feature extraction): {e}")

    # CALL SITE A.5: World Model Decision Loop
    # Get runtime state for world model access
    rt_state = runtime.get()
    world_model = rt_state.world_model
    action_scorer = rt_state.action_scorer
    intent_gate = rt_state.intent_gate

    # Map player input to discrete signal (IntentExtractor first, then regex fallback)
    player_signal = PlayerSignal.QUESTION  # Default fallback
    try:
        if not request.user_input or not request.user_input.strip():
            player_signal = PlayerSignal.IGNORE
        else:
            # Try IntentExtractor first for structured intent
            try:
                intent_extractor = IntentExtractor()
                intent_proposal = intent_extractor.extract(request.user_input)

                # Map IntentType to PlayerSignal
                intent_to_signal = {
                    IntentType.ASK: PlayerSignal.QUESTION,
                    IntentType.REQUEST: PlayerSignal.REQUEST,
                    IntentType.AGREE: PlayerSignal.AGREE,
                    IntentType.DISAGREE: PlayerSignal.DISAGREE,
                    IntentType.REFUSE: PlayerSignal.REFUSE,
                    IntentType.THREATEN: PlayerSignal.THREATEN,
                    IntentType.INFORM: PlayerSignal.QUESTION,
                    IntentType.TRADE: PlayerSignal.OFFER,
                    IntentType.JOKE: PlayerSignal.COMPLIMENT,
                    IntentType.NEUTRAL: PlayerSignal.QUESTION,
                }

                if intent_proposal.intent in intent_to_signal:
                    player_signal = intent_to_signal[intent_proposal.intent]
                    logger.debug(f"IntentExtractor classified: {intent_proposal.intent.value} -> {player_signal.value}")
                else:
                    # Fall back to regex for unknown intents
                    logger.debug(f"IntentExtractor unknown intent {intent_proposal.intent.value}, using regex fallback")
                    raise ValueError("Unknown intent, use regex fallback")
            except Exception as intent_error:
                logger.debug(f"IntentExtractor failed: {intent_error}, using regex fallback")

                # Regex-based fallback
                player_input_lower = request.user_input.lower()

                # Use word boundary matching for accuracy
                if re.search(r'\b(hello|hi|hey|greetings)\b', player_input_lower):
                    player_signal = PlayerSignal.GREET
                elif re.search(r'\b(bye|goodbye|farewell|see you)\b', player_input_lower):
                    player_signal = PlayerSignal.FAREWELL
                elif re.search(r'\b(yes|sure|okay|agree|fine)\b', player_input_lower):
                    player_signal = PlayerSignal.AGREE
                elif re.search(r'\b(no|nope|never|not|disagree)\b', player_input_lower):
                    player_signal = PlayerSignal.DISAGREE
                elif re.search(r'\b(sorry|apologize|my bad|forgive)\b', player_input_lower):
                    player_signal = PlayerSignal.APOLOGIZE
                elif re.search(r'\b(stupid|idiot|hate|kill|die)\b', player_input_lower):
                    player_signal = PlayerSignal.INSULT
                elif re.search(r'\b(help|assist|support|aid)\b', player_input_lower):
                    player_signal = PlayerSignal.HELP
                elif '?' in player_input_lower:
                    player_signal = PlayerSignal.QUESTION
                elif re.search(r'\b(please|can you|could you|would you)\b', player_input_lower):
                    player_signal = PlayerSignal.REQUEST
                else:
                    player_signal = PlayerSignal.QUESTION

    except Exception as e:
        logger.warning(f"Player signal classification failed: {e}")

    # Create StateSnapshot from current RFSNState
    current_state_snapshot = StateSnapshot(
        mood=state.mood,
        affinity=state.affinity,
        relationship=state.relationship,
        recent_sentiment=features.recent_sentiment if features else 0.0,
        combat_active=False,
        quest_active=False,
        trust_level=(
            max(0.0, min(1.0, (state.affinity + 1.0) / 2.0))
            if state.affinity is not None
            else 0.5
        ),
        fear_level=0.0,
        additional_fields={"npc_name": npc_name}
    )

    # Validate affinity range
    if state.affinity is not None and (state.affinity < -1.0 or state.affinity > 1.0):
        logger.warning(f"Affinity out of expected range [-1, 1]: {state.affinity}")

    # Generate and score action candidates
    selected_npc_action = None
    selected_action_score = None
    bandit_key = None
    try:
        if action_scorer and world_model:
            # Score candidates directly (generates internally)
            action_scores = action_scorer.score_candidates(
                current_state_snapshot,
                player_signal
            )

            # Select action using bandit (if available) or fallback to top scorer
            if action_scores:
                # Build candidate set (top-K already scored)
                TOP_K = 4
                candidates = [s.action for s in action_scores[:TOP_K]]
                
                # Get npc_action_bandit from runtime
                
                # Get v1.3 bandits from runtime
                contextual_bandit = runtime.get().contextual_bandit
                mode_bandit = runtime.get().mode_bandit
                
                # Default mode if bandit missing
                selected_phrasing_mode = PhrasingMode.NEUTRAL_BALANCED
                
                # Forced actions bypass learning (single candidate means forced)
                if len(candidates) == 1:
                    selected_npc_action = candidates[0]
                    selected_action_score = action_scores[0]
                else:
                    if contextual_bandit:
                        # Build rich context for v1.3 learning
                        bandit_ctx = BanditContext.from_state(
                            mood=current_state_snapshot.emotional_state,
                            affinity=current_state_snapshot.affinity,
                            player_signal=player_signal,
                            last_action=None,  # Ideally track this from history
                            turn_count=len(memory.turns) if memory else 3,
                            has_safety_flag=False # Safety flag handled by IntentGate separately
                        )
                        
                        # Use LinUCB scaling for priors
                        # We pass priors implicitly via UCB bias or just rely on bandit
                        selected_npc_action = contextual_bandit.select(
                            context=bandit_ctx,
                            candidates=candidates,
                            log_counterfactual=True
                        )
                    else:
                        # Fallback to score-based if no bandit
                        selected_npc_action = candidates[0]

                    # Get the score for the selected action
                    selected_action_score = next(
                        (s for s in action_scores if s.action == selected_npc_action),
                        action_scores[0]
                    )

                # Select Phrasing Mode (Two-Stage Policy)
                if mode_bandit:
                    # Determine personality from name/config (simplified map)
                    npc_role = "generic"
                    if "guard" in npc_name.lower(): npc_role = "guard"
                    elif "merchant" in npc_name.lower(): npc_role = "merchant"
                    elif "eris" in npc_name.lower(): npc_role = "companion" # Key NPC
                    
                    mode_ctx = ModeContext.from_state(
                        affinity=current_state_snapshot.affinity,
                        npc_role=npc_role,
                        emotional_intensity=0.5, # Default, could derive from emotion components
                        last_mode=None
                    )
                    
                    selected_phrasing_mode = mode_bandit.select(mode_ctx)
                    logger.info(f"Mode Bandit: Selected phrasing {selected_phrasing_mode.value}")
                
                logger.info(
                    f"World Model: Selected action {selected_npc_action.value} "
                    f"(score={selected_action_score.total_score:.2f}, "
                    f"utility={selected_action_score.utility_score:.2f}, "
                    f"risk={selected_action_score.risk_score:.2f})"
                )

                # Record decision event
                if event_recorder:
                    event_recorder.record(
                        EventType.ACTION_CHOSEN,
                        {
                            "npc_action": selected_npc_action.value,
                            "player_signal": player_signal.value,
                            "action_score": selected_action_score.total_score,
                            "reasoning": selected_action_score.reasoning,
                            "predicted_state": selected_action_score.predicted_state.to_dict()
                        }
                    )
    except Exception as e:
        logger.error(f"World model decision loop failed: {e}")
        selected_npc_action = None

    # Retrieve Semantic Memories
    semantic_context = ""
    if memory_governance and request.user_input:
        relevant = memory_governance.semantic_search(request.user_input, k=3, min_score=0.4)
        if relevant:
            semantic_context = "Relevant Facts:\n" + "\n".join([f"- {m[0].content}" for m in relevant]) + "\n"
            logger.info(f"Retrieved {len(relevant)} semantic memories")

    # Build prompt with context compression and action mode injection
    history = memory.get_context_window(limit=config_watcher.get("context_limit", 4)) if config_watcher.get("memory_enabled") else ""
    system_prompt = f"You are {state.npc_name}. {state.get_attitude_instruction()}"

    # Inject action mode control block
    if policy_adapter:
        system_prompt += f"\n\n{action_mode.prompt_injection}"

    # Inject selected NPC action instruction using strict action control block
    if selected_npc_action:
        action_block = render_action_block(
            npc_action=selected_npc_action,
            npc_name=state.npc_name,
            mood=current_state_snapshot.mood,
            relationship=current_state_snapshot.relationship,
            affinity=current_state_snapshot.affinity,
            player_signal=player_signal.value,
            context_window=history,
            governed_memory_context=semantic_context
        )
        system_prompt += f"\n\n{action_block}"

    # Inject emotional tone context for expressive dialogue
    emotion_manager = get_emotion_manager()
    emotional_prompt = emotion_manager.get_prompt_injection(state.npc_name)
    system_prompt += f"\n\n{emotional_prompt}"

    # Inject Phrasing Mode Instruction (Style Control)
    if mode_bandit and selected_phrasing_mode:
        phrasing_instr = mode_bandit.get_mode_instructions(selected_phrasing_mode)
        system_prompt += f"\n\n{phrasing_instr}"

    # Require strict FINAL_JSON block (authoritative output contract)
    # This is the ONLY trusted source for memory storage and learning updates
    system_prompt += (
        "\n\n=== OUTPUT CONTRACT ===\n"
        "After your spoken dialogue, you MUST append:\n"
        "FINAL_JSON:\n"
        "```json\n"
        '{"line": "<your exact spoken dialogue>", "action": "<the required npc_action>", "confidence": <0.0-1.0>}\n'
        "```\n"
        "The 'line' must match EXACTLY what you said above.\n"
        "The 'action' must match the required action from [ACTION_CONTROL].\n"
        "The 'confidence' is your certainty the response is appropriate (0.0-1.0).\n"
        "This block is MANDATORY. Responses without valid FINAL_JSON will be discarded."
    )

    full_prompt = f"System: {system_prompt}\n{history}\nPlayer: {request.user_input}\n{state.npc_name}:"
    
    # Snapshot configuration per request (Patch 4) prevents mid-stream tuning weirdness
    req_config = config_watcher.get_all()
    snapshot_temp = req_config.get("temperature", 0.7)
    snapshot_max_tokens = req_config.get("max_tokens", 80)

    # Stream response
    async def stream_generator():
        # Initialize learning flags
        skip_learning = False
        
        full_response = ""
        raw_response = ""
        first_chunk_sent = False
        start_gen = time.time()
        
        # Validate LLM output through IntentGate (per-turn, not per-chunk)
        # Track if first sentence has been validated to avoid duplicate checks
        first_sentence_validated = False
        sentence_buffer = ""
        
        def validate_sentence(text: str, is_first: bool = False) -> tuple[str, bool]:
            """
            Validate a complete sentence through IntentGate.
            Returns (filtered_text, should_abort_stream).
            Only runs on first sentence and final response to reduce overhead.
            """
            if not intent_gate or not text.strip():
                return text, False
            
            # Extract intent from the text
            proposal = intent_gate._extractor.extract(text)
            
            # Check for harmful safety flags
            harmful_flags = {
                SafetyFlag.HARMFUL_CONTENT,
                SafetyFlag.AGGRESSION,
                SafetyFlag.SELF_HARM,
                SafetyFlag.ILLEGAL_ACTION
            }
            
            if any(flag in proposal.safety_flags for flag in harmful_flags):
                if is_first:
                    logger.warning(f"IntentGate blocked first sentence with safety flags: {proposal.safety_flags}")
                    return "", True  # Abort stream
                logger.warning(f"IntentGate detected safety flags in content: {proposal.safety_flags}")
                return "", False
            
            # Log low-confidence content (but don't filter during stream)
            if proposal.confidence < intent_gate.require_min_confidence:
                logger.debug(f"IntentGate: low-confidence content (conf={proposal.confidence})")
            
            return text, False

        try:
            # Emit metadata event first (before any content)
            metadata_event = {
                "player_signal": player_signal.value,
                "bandit_key": bandit_key,
                "npc_action": selected_npc_action.value if selected_npc_action else None,
                "action_mode": action_mode.name if action_mode else None,
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"
            
            # Determine prosody for this turn
            npc_pitch = 1.0
            npc_rate = 1.0
            if npc_session:
                # Convert semitone offset to ratio for rubberband
                npc_pitch = 2.0**(npc_session.voice_config.pitch_offset / 12.0)
                npc_rate = npc_session.voice_config.rate_multiplier
                logger.info(f"Prosody for {npc_name}: pitch_offset={npc_session.voice_config.pitch_offset} -> ratio={npc_pitch:.4f}, rate={npc_rate}")

            # Use snapshot values
            for chunk in streaming_engine.generate_streaming(full_prompt, max_tokens=snapshot_max_tokens, temperature=snapshot_temp, pitch=npc_pitch, rate=npc_rate):
                
                if not first_chunk_sent:
                    first_chunk_sent = True
                    latency = (time.time() - start_gen)
                    observe_first_token(latency)
                    logger.info(f"First token: {latency*1000:.0f}ms")
                    
                    # Micro-reward: Engaging start (fast response)
                    if latency < 2.0:
                        reward_accumulator.add(0.05, "fast_response")

                # Micro-reward: Long, engaging response
                if len(full_response) > 50 and len(full_response) < 60: # trigger once
                    reward_accumulator.add(0.05, "engaging_length")
                
                # Record user input event
                if event_recorder:
                    event_recorder.record(
                        EventType.USER_INPUT,
                        {"text": request.user_input, "npc_name": npc_name}
                    )
                
                # Buffer text and validate on sentence boundaries (per-turn, not per-chunk)
                sentence_buffer += chunk.text
                validated_text = chunk.text
                if not first_sentence_validated and any(p in sentence_buffer for p in '.!?'):
                    validated_text, abort = validate_sentence(sentence_buffer, is_first=True)
                    first_sentence_validated = True
                    if abort:
                        yield f"data: {json.dumps({'error': 'Content blocked'})}\\n\\n"
                        return
                    sentence_buffer = ""

                # Accumulate RAW text before cleanup for JSON extraction
                # (we need this because _cleanup_tokens below strips code blocks)
                raw_response += (validated_text or "")

                clean_text = _cleanup_tokens(validated_text)
                
                if clean_text:  # Only send non-empty cleaned text
                    yield f"data: {json.dumps({'sentence': clean_text, 'is_final': chunk.is_final, 'latency_ms': chunk.latency_ms})}\n\n"
                full_response += clean_text  # Cleaned accumulator for memory
                
                # Token metrics
                inc_tokens(len(chunk.text.split())) # Rough estimate
                
                if chunk.is_final and config_watcher.get("memory_enabled"):
                    payload, is_verified = _extract_tail_json_payload(raw_response)
                    
                    # Track if we should skip learning updates
                    skip_learning = False

                    if payload and isinstance(payload, dict) and "line" in payload:
                        # Authoritative stored text comes from structured payload
                        stored_text = _cleanup_tokens(str(payload.get("line", "")))
                        
                        # Get confidence from FINAL_JSON (default 0.5 if missing)
                        turn_confidence = payload.get("confidence", 0.5)

                        # Optional: sanity-check action match (record if mismatch)
                        if selected_npc_action:
                            expected = selected_npc_action.value
                            got = str(payload.get("action", "")).strip().lower()
                            if got and got != expected.lower() and event_recorder:
                                event_recorder.record(
                                    EventType.SAFETY_EVENT,
                                    {"reason": "action_mismatch_in_tail_json", "expected": expected, "got": got, "npc": npc_name}
                                )
                                # Action mismatch means unverified
                                is_verified = False
                    else:
                        # No valid structured footer -> treat as unsafe/unverifiable
                        stored_text = full_response.strip()
                        turn_confidence = 0.0
                        is_verified = False
                        if event_recorder:
                            event_recorder.record(
                                EventType.SAFETY_EVENT,
                                {"reason": "missing_or_invalid_final_json", "npc": npc_name}
                            )
                    
                    # CRITICAL: Skip learning on unverified turns
                    if not is_verified:
                        skip_learning = True
                        logger.warning(f"Unverified FINAL_JSON for {npc_name} - skipping learning update")

                    memory.add_turn(request.user_input, stored_text)
                    
                    # Record LLM generation event
                    if event_recorder:
                        event_recorder.record(
                            EventType.LLM_GENERATION,
                            {"prompt": full_prompt, "response": stored_text, "verified": is_verified}
                        )
                    
                    # Also add to MemoryGovernance for provenance tracking
                    if memory_governance:
                        governed_memory = GovernedMemory(
                            memory_id="",
                            memory_type=MemoryType.CONVERSATION_TURN,
                            source=MemorySource.NPC_RESPONSE,
                            content=stored_text,
                            confidence=turn_confidence if is_verified else 0.0,  # Quarantine unverified
                            timestamp=datetime.utcnow(),
                            metadata={
                                "npc_name": npc_name,
                                "user_input": request.user_input,
                                "action_mode": action_mode.name if action_mode else None,
                                "npc_action": selected_npc_action.value if selected_npc_action else None,
                                "player_signal": player_signal.value,
                                "action_score": selected_action_score.total_score if selected_action_score else None,
                                "verified": is_verified,  # Explicit verification flag
                                "quarantined": not is_verified  # Mark quarantine status
                            }
                        )
                        memory_governance.add_memory(governed_memory)
                    
                    # Process emotion on authoritative stored text
                    emotion_info = multi_manager.process_response(npc_name, stored_text)
                    logger.info(f"NPC {npc_name} emotion: {emotion_info['emotion']['primary']}")
                    
                    # Update emotional state based on action taken
                    emotion_mgr = get_emotion_manager()
                    if selected_npc_action:
                        # Player sentiment from features
                        player_sentiment = features.recent_sentiment if features else 0.0
                        emotion_mgr.update_from_action(
                            npc_name=npc_name,
                            action_name=selected_npc_action.value,
                            player_sentiment=player_sentiment
                        )
                        logger.debug(f"Emotional state updated for {npc_name}: {emotion_mgr.get_state(npc_name).primary_tone.value}")
                    
                    # Micro-reward: Positive emotion (only if verified)
                    if is_verified and emotion_info['emotion']['primary'] in ("joy", "trust", "anticipation"):
                        reward_accumulator.add(0.10, "positive_emotion")
            
            # Explicit End-of-Stream Flush (Patch v8.9)
            if streaming_engine and streaming_engine.voice:
                streaming_engine.voice.flush_pending()
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            inc_errors()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        total_dur = (time.time() - start_time) * 1000
        observe_request_duration(total_dur / 1000.0)
        await broadcast_metrics(streaming_engine.voice.metrics)
        
        # CALL SITE B: Compute reward and update policy (Learning Layer)
        # CALL SITE C: Apply authoritative state transition (World Model)
        state_after = None
        if state_machine and selected_npc_action and selected_action_score:
            try:
                # Capture state before transition
                state_before_dict = {
                    "mood": state.mood,
                    "affinity": state.affinity,
                    "relationship": state.relationship,
                    "playstyle": state.playstyle,
                    "recent_sentiment": features.recent_sentiment if features else 0.0
                }

                # Apply authoritative state transition
                state_after_dict = state_machine.apply_transition(
                    state_before_dict,
                    player_signal.value,
                    selected_npc_action.value
                )

                # Update NPC state with authoritative result
                state.mood = state_after_dict["mood"]
                state.affinity = state_after_dict["affinity"]
                state.relationship = state_after_dict["relationship"]
                state.playstyle = state_after_dict.get("playstyle", state.playstyle)

                state_after = state_after_dict

                logger.info(
                    f"State transition: {state_before_dict['mood']} -> {state_after_dict['mood']}, "
                    f"affinity {state_before_dict['affinity']:.2f} -> {state_after_dict['affinity']:.2f}"
                )

                # Record transition event for training
                if event_recorder:
                    event_recorder.record(
                        EventType.STATE_UPDATE,
                        {
                            "state_before": state_before_dict,
                            "state_after": state_after_dict,
                            "npc_action": selected_npc_action.value,
                            "player_signal": player_signal.value
                        }
                    )

                # Record transition in world model for learning
                if world_model and selected_action_score:
                    predicted_state = selected_action_score.predicted_state
                    # Calculate reward based on affinity change
                    reward = state_after_dict["affinity"] - state_before_dict["affinity"]
                    world_model.record_transition(
                        current_state_snapshot,
                        selected_npc_action,
                        player_signal,
                        predicted_state,
                        reward=reward
                    )

            except Exception as e:
                logger.error(f"State transition failed: {e}")


        
        # Update NPC Action Bandit with reward signal
        # Update v1.3 Learning Layers
        # Only update if learning is not skipped (Patch 003: FINAL_JSON verification)
        if not skip_learning:
            # Get components
            contextual_bandit = runtime.get().contextual_bandit
            mode_bandit = runtime.get().mode_bandit
            metrics_guard = runtime.get().metrics_guard
            reward_model = runtime.get().reward_model or RewardModel() # Ensure fallback
            temporal_memory = runtime.get().temporal_memory
            
            try:
                # 1. Compute Base Rewards
                # Sigmoid normalization for score
                import math
                def sigmoid(x: float) -> float: return 1.0 / (1.0 + math.exp(-x))
                base_reward = sigmoid(selected_action_score.total_score)
                
                # Add micro-rewards
                shaped_reward = base_reward + reward_accumulator.emit()
                
                # 2. Check Ground-Truth Anchors (v1.3)
                # Determine response time (latency)
                latency_ms = 0.0
                if 'latency' in locals():
                    latency_ms = latency * 1000.0
                
                anchor_reward, anchor_type = reward_model.compute_ground_truth_anchor(
                    user_text=request.user_input,
                    response_time_ms=latency_ms
                )
                
                if anchor_type != "none":
                    final_reward = max(0.0, min(1.0, 0.5 + anchor_reward)) # Normalize anchor centered at 0.5
                    logger.info(f"Learning Anchor Triggered: {anchor_type} -> reward={final_reward:.2f}")
                else:
                    final_reward = max(0.0, min(1.0, shaped_reward))
                
                # 3. Update Metrics Guard
                if metrics_guard:
                    metrics_guard.record_turn(
                        reward=final_reward,
                        latency_ms=latency_ms,
                        was_corrected=(anchor_type in ["thats_wrong", "repeat_question"]),
                        was_blocked=False, # Safety blocks handled elsewhere
                        npc_name=npc_name
                    )
                    
                    # Check safe mode
                    if metrics_guard.is_frozen():
                        logger.warning("Learning Frozen by Metrics Guard")
                        return
                        
                # 4. Update Bandits
                if contextual_bandit and selected_npc_action:
                    # Reconstruct context
                    bandit_ctx = BanditContext.from_state(
                        mood=current_state_snapshot.emotional_state,
                        affinity=current_state_snapshot.affinity,
                        player_signal=player_signal,
                        last_action=None,
                        turn_count=len(memory.turns) if memory else 3,
                        has_safety_flag=False
                    )
                    contextual_bandit.update(bandit_ctx, selected_npc_action, final_reward)
                    contextual_bandit.save()
                    
                if mode_bandit and selected_phrasing_mode:
                    mode_bandit.update(mode_ctx, selected_phrasing_mode, final_reward)
                    mode_bandit.save()
                    
                # 5. Temporal Memory
                if temporal_memory and current_state_snapshot and selected_npc_action:
                    temporal_memory.record(
                        state=current_state_snapshot,
                        action=selected_npc_action,
                        reward=final_reward
                    )
                
                logger.info(f"Learning Updated: action={selected_npc_action.value}, mode={selected_phrasing_mode.value}, reward={final_reward:.2f}")

            except Exception as e:
                logger.error(f"Learning update failed: {e}")
        else:
            logger.warning("Learning skipped due to unverified FINAL_JSON or quarantine")

        
        # Per-request Trace Log (Patch v8.9)
        trace_id = f"req_{int(start_time)}"
        logger.info(
            f"TRACE[{trace_id}] npc={npc_name} "
            f"latency={streaming_engine.voice.metrics.first_token_ms:.0f}ms "
            f"total={total_dur:.0f}ms "
            f"q_size={streaming_engine.voice.metrics.tts_queue_size} "
            f"drops={streaming_engine.voice.metrics.dropped_sentences}"
        )
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


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


@app.get("/api/health")
async def health_check():
    """Operational health check"""
    # Check if LLM is ready (either Ollama or llama-cpp)
    model_ok = streaming_engine is not None and (
        streaming_engine.ollama_client is not None or 
        streaming_engine.llm is not None
    )
    tts_ok = tts_engine is not None
    q_size = len(streaming_engine.voice.speech_queue) if streaming_engine else 0
    
    return {
        "status": "healthy" if model_ok and tts_ok else "degraded",
        "model_loaded": model_ok,
        "tts_ready": tts_ok,
        "queue_size": q_size,
        "active_npc": list(multi_manager.sessions.keys()),
        "uptime": time.time()
    }


@app.get("/api/status")
async def get_status():
    # System health check
    return {
        "status": "healthy",
        "version": "9.0",
        "components": {
            "streaming_llm": streaming_engine is not None,
            "kokoro_tts": tts_engine is not None,
            "voice_router": voice_router is not None,
            "xvasynth": xva_engine.available if xva_engine else False,
            "memory_system": True,
            "active_websockets": len(active_ws),
            "active_npc_sessions": len(multi_manager.sessions)
        }
    }


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
