#!/usr/bin/env python3
"""
Voice Router - Automatic TTS Engine Selection Based on Narrative Weight

Routes dialogue to appropriate TTS engine:
- LOW intensity → Chatterbox-Turbo (guards, shopkeepers, ambient)
- MEDIUM intensity → Chatterbox-Turbo with stronger exaggeration
- HIGH intensity → Chatterbox-Full (companions, emotional moments, memory callbacks)

The routing decision is made automatically from existing NPC state,
eliminating the need for per-line authoring.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Any
import logging
import time

from emotional_tone import EmotionalState, EmotionalTone

logger = logging.getLogger(__name__)


class VoiceIntensity(IntEnum):
    """
    Voice intensity levels that determine TTS engine selection.
    
    Computed automatically from NPC emotional state and context.
    """
    LOW = 0     # Ambient NPCs, barks, filler - uses Turbo
    MEDIUM = 1  # Normal companion dialogue - uses Turbo + exaggeration
    HIGH = 2    # Emotional moments, memory callbacks - uses Full


@dataclass
class VoiceRequest:
    """
    Request for voice synthesis with full context.
    
    Contains all information needed for routing decision.
    """
    text: str
    npc_id: str
    emotional_state: EmotionalState
    context: Dict[str, Any] = field(default_factory=dict)
    npc_type: str = "ambient"  # ambient | companion | main
    
    # Optional overrides
    force_intensity: Optional[VoiceIntensity] = None
    voice_prompt_path: Optional[str] = None


@dataclass
class VoiceConfig:
    """Configuration for voice routing thresholds and parameters"""
    
    # Intensity thresholds
    high_emotion_threshold: float = 0.7  # emotion.intensity above this → HIGH
    high_valence_threshold: float = 0.7  # abs(valence) above this → HIGH
    high_arousal_threshold: float = 0.8  # arousal above this → MEDIUM
    
    # Exaggeration presets by intensity
    low_exaggeration: float = 0.3
    medium_exaggeration: float = 0.6
    high_exaggeration: float = 0.8
    
    # CFG presets by intensity
    low_cfg: float = 0.5
    medium_cfg: float = 0.4
    high_cfg: float = 0.3
    
    # Companion minimum intensity (companions never drop below MEDIUM feel)
    companion_min_exaggeration: float = 0.5


class VoiceRouter:
    """
    Central voice routing system.
    
    Automatically selects between Chatterbox-Turbo and Chatterbox-Full
    based on narrative weight computed from NPC state and context.
    """
    
    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        device: str = "cuda",
        enable_queue: bool = True,
        fallback_to_kokoro: bool = True
    ):
        """
        Initialize the voice router with dual TTS engines.
        
        Args:
            config: VoiceConfig with thresholds and parameters
            device: "cuda" or "cpu" (cuda strongly recommended)
            enable_queue: Enable async queue mode for background playback
            fallback_to_kokoro: Fall back to Kokoro if Chatterbox unavailable
        """
        self.config = config or VoiceConfig()
        self.device = device
        self.enable_queue = enable_queue
        
        # Per-NPC voice prompts for cloning
        self.voice_prompts: Dict[str, str] = {}
        
        # Statistics
        self._route_counts = {VoiceIntensity.LOW: 0, VoiceIntensity.MEDIUM: 0, VoiceIntensity.HIGH: 0}
        self._total_latency_ms = 0.0
        self._total_routes = 0
        
        # Initialize engines
        self._init_engines(fallback_to_kokoro)
        
        logger.info(f"[VoiceRouter] Initialized (device={device}, fallback={fallback_to_kokoro})")
    
    def _init_engines(self, fallback_to_kokoro: bool):
        """Initialize TTS engines with graceful fallback"""
        self.turbo_engine: Any = None
        self.full_engine: Any = None
        self.fallback_engine: Any = None
        self._using_fallback = False
        
        try:
            from chatterbox_tts import ChatterboxTTSEngine
            
            # Try to load Chatterbox engines
            self.turbo_engine = ChatterboxTTSEngine(
                model_variant="turbo",
                device=self.device,
                enable_queue=self.enable_queue
            )
            
            self.full_engine = ChatterboxTTSEngine(
                model_variant="full",
                device=self.device,
                enable_queue=self.enable_queue
            )
            
            if not self.turbo_engine.is_available or not self.full_engine.is_available:
                raise RuntimeError("Chatterbox models not available")
            
            logger.info("[VoiceRouter] Chatterbox engines loaded successfully")
            
        except Exception as e:
            logger.warning(f"[VoiceRouter] Chatterbox unavailable ({e}), attempting fallback...")
            
            if fallback_to_kokoro:
                try:
                    from kokoro_tts import KokoroTTSEngine
                    self.fallback_engine = KokoroTTSEngine(enable_queue=self.enable_queue)
                    self.turbo_engine = self.fallback_engine
                    self.full_engine = self.fallback_engine
                    self._using_fallback = True
                    logger.info("[VoiceRouter] Fallback to Kokoro TTS successful")
                except Exception as fallback_error:
                    logger.error(f"[VoiceRouter] Kokoro fallback also failed: {fallback_error}")
                    self._using_fallback = True
            else:
                self._using_fallback = True
    
    def set_voice_prompt(self, npc_id: str, audio_path: str):
        """
        Register a voice cloning prompt for an NPC.
        
        Args:
            npc_id: NPC identifier
            audio_path: Path to reference audio file (WAV recommended)
        """
        self.voice_prompts[npc_id] = audio_path
        logger.info(f"[VoiceRouter] Voice prompt registered for {npc_id}: {audio_path}")
    
    def compute_intensity(self, request: VoiceRequest) -> VoiceIntensity:
        """
        Compute voice intensity from NPC state and context.
        
        This is the core routing logic that eliminates per-line authoring.
        
        Args:
            request: VoiceRequest with full context
            
        Returns:
            VoiceIntensity level (LOW, MEDIUM, HIGH)
        """
        # Allow explicit override
        if request.force_intensity is not None:
            return request.force_intensity
        
        state = request.emotional_state
        ctx = request.context
        cfg = self.config
        
        # Rule 1: High emotional intensity always gets full treatment
        if state.intensity > cfg.high_emotion_threshold:
            logger.debug(f"[VoiceRouter] HIGH: emotion intensity {state.intensity:.2f}")
            return VoiceIntensity.HIGH
        
        # Rule 2: Strong positive or negative valence
        if abs(state.valence) > cfg.high_valence_threshold:
            logger.debug(f"[VoiceRouter] HIGH: valence {state.valence:.2f}")
            return VoiceIntensity.HIGH
        
        # Rule 3: Memory callbacks and relationship moments always HIGH
        if ctx.get("is_memory_callback") or ctx.get("is_relationship_moment"):
            logger.debug(f"[VoiceRouter] HIGH: memory/relationship context")
            return VoiceIntensity.HIGH
        
        # Rule 4: Cutscene or high-stakes dialogue
        if ctx.get("is_cutscene") or ctx.get("is_high_stakes"):
            logger.debug(f"[VoiceRouter] HIGH: cutscene/high-stakes")
            return VoiceIntensity.HIGH
        
        # Rule 5: Companions get minimum MEDIUM
        if request.npc_type in ("companion", "main"):
            # Elevated state for companion
            if state.intensity > 0.4 or state.arousal > cfg.high_arousal_threshold:
                logger.debug(f"[VoiceRouter] HIGH: companion elevated state")
                return VoiceIntensity.HIGH
            logger.debug(f"[VoiceRouter] MEDIUM: companion baseline")
            return VoiceIntensity.MEDIUM
        
        # Rule 6: High arousal (combat, fear) → MEDIUM for ambient
        if state.arousal > cfg.high_arousal_threshold:
            logger.debug(f"[VoiceRouter] MEDIUM: high arousal {state.arousal:.2f}")
            return VoiceIntensity.MEDIUM
        
        # Rule 7: Aggressive or defensive tones → MEDIUM
        if state.primary_tone in (EmotionalTone.AGGRESSIVE, EmotionalTone.DEFENSIVE):
            logger.debug(f"[VoiceRouter] MEDIUM: {state.primary_tone.value} tone")
            return VoiceIntensity.MEDIUM
        
        # Default: LOW for ambient NPCs
        logger.debug(f"[VoiceRouter] LOW: ambient default")
        return VoiceIntensity.LOW
    
    def get_params_for_intensity(self, intensity: VoiceIntensity) -> Dict[str, float]:
        """
        Get exaggeration and CFG parameters for intensity level.
        
        Args:
            intensity: VoiceIntensity level
            
        Returns:
            Dict with 'exaggeration' and 'cfg' keys
        """
        cfg = self.config
        
        if intensity == VoiceIntensity.LOW:
            return {"exaggeration": cfg.low_exaggeration, "cfg": cfg.low_cfg}
        elif intensity == VoiceIntensity.MEDIUM:
            return {"exaggeration": cfg.medium_exaggeration, "cfg": cfg.medium_cfg}
        else:  # HIGH
            return {"exaggeration": cfg.high_exaggeration, "cfg": cfg.high_cfg}
    
    def route(self, request: VoiceRequest) -> bool:
        """
        Route voice request to appropriate TTS engine.
        
        Args:
            request: VoiceRequest with text and context
            
        Returns:
            True if successfully queued/spoken, False otherwise
        """
        start_time = time.perf_counter()
        
        # Compute intensity
        intensity = self.compute_intensity(request)
        self._route_counts[intensity] += 1
        
        # Get voice parameters
        params = self.get_params_for_intensity(intensity)
        
        # Select engine
        if intensity == VoiceIntensity.HIGH:
            engine = self.full_engine
            engine_name = "Full"
        else:
            engine = self.turbo_engine
            engine_name = "Turbo"
        
        if engine is None:
            logger.warning(f"[VoiceRouter] No engine available for {request.npc_id}")
            return False
        
        # Get voice prompt for this NPC
        voice_prompt = request.voice_prompt_path or self.voice_prompts.get(request.npc_id)
        
        logger.info(
            f"[VoiceRouter] {request.npc_id} → {engine_name} "
            f"(intensity={intensity.name}, exagg={params['exaggeration']:.1f})"
        )
        
        # Dispatch to engine
        try:
            if self.enable_queue:
                # Async mode - Chatterbox engine handles exaggeration internally
                if hasattr(engine, 'speak') and callable(getattr(engine, 'speak')):
                    result = engine.speak(
                        request.text,
                        exaggeration=params['exaggeration'],
                        cfg=params['cfg']
                    )
                else:
                    # Fallback engine (Kokoro) - simpler interface
                    result = engine.speak(request.text)
            else:
                # Sync mode
                if hasattr(engine, 'speak_sync'):
                    engine.speak_sync(
                        request.text,
                        exaggeration=params['exaggeration'],
                        cfg=params['cfg'],
                        audio_prompt_path=voice_prompt
                    )
                    result = True
                else:
                    engine.speak_sync(request.text)
                    result = True
            
            # Track latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._total_latency_ms += latency_ms
            self._total_routes += 1
            
            return result
            
        except Exception as e:
            logger.error(f"[VoiceRouter] Routing failed: {e}")
            return False
    
    def route_sync(self, request: VoiceRequest):
        """
        Synchronously route and play voice request (blocking).
        
        Args:
            request: VoiceRequest with text and context
        """
        # Temporarily switch to sync mode
        original_queue = self.enable_queue
        self.enable_queue = False
        try:
            self.route(request)
        finally:
            self.enable_queue = original_queue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "route_counts": {k.name: v for k, v in self._route_counts.items()},
            "total_routes": self._total_routes,
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_routes),
            "using_fallback": self._using_fallback,
            "registered_voices": list(self.voice_prompts.keys())
        }
    
    def shutdown(self):
        """Graceful shutdown of all engines"""
        if self.turbo_engine:
            self.turbo_engine.shutdown()
        if self.full_engine and self.full_engine != self.turbo_engine:
            self.full_engine.shutdown()
        if self.fallback_engine and self.fallback_engine != self.turbo_engine:
            self.fallback_engine.shutdown()
        logger.info("[VoiceRouter] Shutdown complete")


# Convenience function for single-shot routing
def speak_with_emotion(
    text: str,
    npc_id: str,
    emotional_state: EmotionalState,
    npc_type: str = "ambient",
    context: Optional[Dict[str, Any]] = None,
    router: Optional[VoiceRouter] = None
) -> bool:
    """
    Convenience function for single-shot voice routing.
    
    Args:
        text: Text to speak
        npc_id: NPC identifier
        emotional_state: Current NPC emotional state
        npc_type: "ambient", "companion", or "main"
        context: Additional routing context
        router: Optional VoiceRouter instance (creates new if None)
        
    Returns:
        True if successfully routed
    """
    if router is None:
        router = VoiceRouter(enable_queue=False)
    
    request = VoiceRequest(
        text=text,
        npc_id=npc_id,
        emotional_state=emotional_state,
        npc_type=npc_type,
        context=context or {}
    )
    
    return router.route(request)


if __name__ == "__main__":
    # Demo: Voice routing based on emotional state
    print("Voice Router Demo")
    print("=" * 60)
    
    # Create router (will fall back to Kokoro if no CUDA)
    router = VoiceRouter(enable_queue=False, fallback_to_kokoro=True)
    
    # Test 1: Low intensity (ambient guard)
    print("\n[Test 1] Ambient guard (LOW intensity expected):")
    guard_state = EmotionalState(
        valence=0.0,
        arousal=0.3,
        dominance=0.6,
        intensity=0.3,
        primary_tone=EmotionalTone.NEUTRAL
    )
    request = VoiceRequest(
        text="Move along, citizen.",
        npc_id="guard_001",
        emotional_state=guard_state,
        npc_type="ambient"
    )
    intensity = router.compute_intensity(request)
    print(f"Computed intensity: {intensity.name}")
    router.route(request)
    
    # Test 2: Medium intensity (companion casual)
    print("\n[Test 2] Companion casual (MEDIUM intensity expected):")
    lydia_state = EmotionalState(
        valence=0.3,
        arousal=0.4,
        dominance=0.5,
        intensity=0.4,
        primary_tone=EmotionalTone.WARM
    )
    request = VoiceRequest(
        text="I am sworn to carry your burdens.",
        npc_id="lydia",
        emotional_state=lydia_state,
        npc_type="companion"
    )
    intensity = router.compute_intensity(request)
    print(f"Computed intensity: {intensity.name}")
    router.route(request)
    
    # Test 3: High intensity (memory callback)
    print("\n[Test 3] Memory callback (HIGH intensity expected):")
    lydia_emotional = EmotionalState(
        valence=0.8,
        arousal=0.7,
        dominance=0.4,
        intensity=0.85,
        primary_tone=EmotionalTone.WARM
    )
    request = VoiceRequest(
        text="I remember when you saved my life in Helgen. I never forgot that moment.",
        npc_id="lydia",
        emotional_state=lydia_emotional,
        npc_type="companion",
        context={"is_memory_callback": True}
    )
    intensity = router.compute_intensity(request)
    print(f"Computed intensity: {intensity.name}")
    router.route(request)
    
    # Print stats
    print("\n" + "=" * 60)
    print("Routing Statistics:")
    stats = router.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    router.shutdown()
    print("\nDemo complete!")
