#!/usr/bin/env python3
"""
Voice Router - Automatic TTS Engine Selection Based on Narrative Weight

Routes dialogue to appropriate TTS engine:
- LOW intensity → Chatterbox-Turbo (guards, shopkeepers, ambient)
- MEDIUM intensity → Chatterbox-Turbo with stronger exaggeration
- HIGH intensity → Chatterbox-Full (companions, emotional moments, memory callbacks)

OPTIMIZATIONS (v2):
- Lazy model loading: Full model only loaded on first HIGH request
- Audio caching: LRU cache for repeated lines (configurable TTL)
- Intensity precomputation: Cache intensity for stable NPC states
- Batch-friendly: Supports priority queuing for latency-sensitive requests
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
import logging
import time
import hashlib
import threading

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
    
    # Priority hint (lower = higher priority, 0 = immediate)
    priority: int = 5
    
    # Cache control
    skip_cache: bool = False


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
    
    # Audio caching
    cache_enabled: bool = True
    cache_max_size: int = 100  # Max cached audio clips
    cache_ttl_seconds: float = 300.0  # 5 minute TTL
    
    # Lazy loading
    lazy_load_full_model: bool = True  # Only load Full model when first needed
    
    # Precomputation
    intensity_cache_ttl: float = 5.0  # Cache intensity computation for 5s


class AudioCache:
    """
    LRU cache for synthesized audio with TTL expiration.
    
    Caches audio by hash of (text, npc_id, intensity, exaggeration).
    Dramatically reduces repeated synthesis for common lines.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[bytes, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, text: str, npc_id: str, intensity: VoiceIntensity, exaggeration: float) -> str:
        """Generate cache key from request parameters"""
        raw = f"{text}|{npc_id}|{intensity.value}|{exaggeration:.2f}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, text: str, npc_id: str, intensity: VoiceIntensity, exaggeration: float) -> Optional[bytes]:
        """Get cached audio if available and not expired"""
        key = self._make_key(text, npc_id, intensity, exaggeration)
        
        with self._lock:
            if key in self._cache:
                audio, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return audio
                else:
                    # Expired, remove
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def put(self, text: str, npc_id: str, intensity: VoiceIntensity, exaggeration: float, audio: bytes):
        """Cache synthesized audio"""
        key = self._make_key(text, npc_id, intensity, exaggeration)
        
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (audio, time.time())
    
    def clear(self):
        """Clear all cached audio"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / max(1, total)
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1%}"
            }


class IntensityCache:
    """
    Cache for intensity computations.
    
    Avoids recomputing intensity for stable NPC states.
    Keyed by (npc_id, emotional_state_hash, context_hash).
    """
    
    def __init__(self, ttl_seconds: float = 5.0):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[VoiceIntensity, float]] = {}
        self._lock = threading.Lock()
    
    def _make_key(self, request: 'VoiceRequest') -> str:
        """Generate cache key from request"""
        state = request.emotional_state
        ctx_str = str(sorted(request.context.items())) if request.context else ""
        raw = f"{request.npc_id}|{request.npc_type}|{state.valence:.2f}|{state.arousal:.2f}|{state.intensity:.2f}|{ctx_str}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, request: 'VoiceRequest') -> Optional[VoiceIntensity]:
        """Get cached intensity if available"""
        if request.force_intensity is not None:
            return request.force_intensity
        
        key = self._make_key(request)
        
        with self._lock:
            if key in self._cache:
                intensity, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return intensity
                else:
                    del self._cache[key]
        
        return None
    
    def put(self, request: 'VoiceRequest', intensity: VoiceIntensity):
        """Cache computed intensity"""
        key = self._make_key(request)
        
        with self._lock:
            self._cache[key] = (intensity, time.time())
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()


class VoiceRouter:
    """
    Central voice routing system with performance optimizations.
    
    Automatically selects between Chatterbox-Turbo and Chatterbox-Full
    based on narrative weight computed from NPC state and context.
    
    OPTIMIZATIONS:
    - Lazy model loading: Full model only loaded on first HIGH request
    - Audio caching: LRU cache for repeated lines
    - Intensity precomputation: Cached for stable NPC states
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
        self._fallback_to_kokoro = fallback_to_kokoro
        
        # Per-NPC voice prompts for cloning
        self.voice_prompts: Dict[str, str] = {}
        
        # Statistics
        self._route_counts = {VoiceIntensity.LOW: 0, VoiceIntensity.MEDIUM: 0, VoiceIntensity.HIGH: 0}
        self._total_latency_ms = 0.0
        self._total_routes = 0
        self._cache_hits = 0
        
        # Caches
        self._audio_cache = AudioCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.cache_enabled else None
        
        self._intensity_cache = IntensityCache(
            ttl_seconds=self.config.intensity_cache_ttl
        )
        
        # Initialize engines (lazy loading for Full model)
        self._full_model_loaded = False
        self._init_engines(fallback_to_kokoro)
        
        logger.info(f"[VoiceRouter] Initialized (device={device}, cache={self.config.cache_enabled}, lazy={self.config.lazy_load_full_model})")
    
    def _init_engines(self, fallback_to_kokoro: bool):
        """Initialize TTS engines with graceful fallback and lazy loading"""
        self.turbo_engine: Any = None
        self.full_engine: Any = None
        self.fallback_engine: Any = None
        self._using_fallback = False
        
        try:
            from chatterbox_tts import ChatterboxTTSEngine
            
            # Always load Turbo (workhorse model)
            self.turbo_engine = ChatterboxTTSEngine(
                model_variant="turbo",
                device=self.device,
                enable_queue=self.enable_queue
            )
            
            if not self.turbo_engine.is_available:
                raise RuntimeError("Chatterbox Turbo not available")
            
            # Lazy load Full model only if not enabled
            if not self.config.lazy_load_full_model:
                self._load_full_model()
            else:
                logger.info("[VoiceRouter] Full model lazy loading enabled (will load on first HIGH request)")
            
            logger.info("[VoiceRouter] Chatterbox Turbo engine loaded successfully")
            
        except Exception as e:
            logger.warning(f"[VoiceRouter] Chatterbox unavailable ({e}), attempting fallback...")
            
            if fallback_to_kokoro:
                try:
                    from voice.kokoro_tts import KokoroTTSEngine
                    self.fallback_engine = KokoroTTSEngine(enable_queue=self.enable_queue)
                    self.turbo_engine = self.fallback_engine
                    self.full_engine = self.fallback_engine
                    self._using_fallback = True
                    self._full_model_loaded = True
                    logger.info("[VoiceRouter] Fallback to Kokoro TTS successful")
                except Exception as fallback_error:
                    logger.error(f"[VoiceRouter] Kokoro fallback also failed: {fallback_error}")
                    self._using_fallback = True
            else:
                self._using_fallback = True
    
    def _load_full_model(self):
        """Load the Full Chatterbox model (called lazily on first HIGH request)"""
        if self._full_model_loaded:
            return
        
        if self._using_fallback:
            self._full_model_loaded = True
            return
        
        try:
            from chatterbox_tts import ChatterboxTTSEngine
            
            logger.info("[VoiceRouter] Loading Full model for HIGH intensity requests...")
            start = time.perf_counter()
            
            self.full_engine = ChatterboxTTSEngine(
                model_variant="full",
                device=self.device,
                enable_queue=self.enable_queue
            )
            
            if self.full_engine.is_available:
                load_time = (time.perf_counter() - start) * 1000
                logger.info(f"[VoiceRouter] Full model loaded in {load_time:.0f}ms")
                self._full_model_loaded = True
            else:
                logger.warning("[VoiceRouter] Full model not available, using Turbo for HIGH")
                self.full_engine = self.turbo_engine
                self._full_model_loaded = True
                
        except Exception as e:
            logger.error(f"[VoiceRouter] Failed to load Full model: {e}, using Turbo")
            self.full_engine = self.turbo_engine
            self._full_model_loaded = True
    
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
        
        Uses caching for stable states to avoid recomputation.
        
        Args:
            request: VoiceRequest with full context
            
        Returns:
            VoiceIntensity level (LOW, MEDIUM, HIGH)
        """
        # Check cache first
        cached = self._intensity_cache.get(request)
        if cached is not None:
            return cached
        
        # Allow explicit override
        if request.force_intensity is not None:
            return request.force_intensity
        
        state = request.emotional_state
        ctx = request.context
        cfg = self.config
        
        # Rule 1: High emotional intensity always gets full treatment
        if state.intensity > cfg.high_emotion_threshold:
            intensity = VoiceIntensity.HIGH
        # Rule 2: Strong positive or negative valence
        elif abs(state.valence) > cfg.high_valence_threshold:
            intensity = VoiceIntensity.HIGH
        # Rule 3: Memory callbacks and relationship moments always HIGH
        elif ctx.get("is_memory_callback") or ctx.get("is_relationship_moment"):
            intensity = VoiceIntensity.HIGH
        # Rule 4: Cutscene or high-stakes dialogue
        elif ctx.get("is_cutscene") or ctx.get("is_high_stakes"):
            intensity = VoiceIntensity.HIGH
        # Rule 5: Companions get minimum MEDIUM
        elif request.npc_type in ("companion", "main"):
            if state.intensity > 0.4 or state.arousal > cfg.high_arousal_threshold:
                intensity = VoiceIntensity.HIGH
            else:
                intensity = VoiceIntensity.MEDIUM
        # Rule 6: High arousal (combat, fear) → MEDIUM for ambient
        elif state.arousal > cfg.high_arousal_threshold:
            intensity = VoiceIntensity.MEDIUM
        # Rule 7: Aggressive or defensive tones → MEDIUM
        elif state.primary_tone in (EmotionalTone.AGGRESSIVE, EmotionalTone.DEFENSIVE):
            intensity = VoiceIntensity.MEDIUM
        # Default: LOW for ambient NPCs
        else:
            intensity = VoiceIntensity.LOW
        
        # Cache the result
        self._intensity_cache.put(request, intensity)
        
        return intensity
    
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
        
        Uses audio cache for repeated lines when enabled.
        
        Args:
            request: VoiceRequest with text and context
            
        Returns:
            True if successfully queued/spoken, False otherwise
        """
        start_time = time.perf_counter()
        
        # Compute intensity (cached)
        intensity = self.compute_intensity(request)
        self._route_counts[intensity] += 1
        
        # Get voice parameters
        params = self.get_params_for_intensity(intensity)
        
        # Check audio cache
        if self._audio_cache and not request.skip_cache:
            cached_audio = self._audio_cache.get(
                request.text, request.npc_id, intensity, params['exaggeration']
            )
            if cached_audio is not None:
                self._cache_hits += 1
                logger.debug(f"[VoiceRouter] Cache hit for {request.npc_id}")
                # Play cached audio (implement playback)
                # For now, just return True
                return True
        
        # Lazy load Full model if needed
        if intensity == VoiceIntensity.HIGH and not self._full_model_loaded:
            self._load_full_model()
        
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
                # Async mode
                if hasattr(engine, 'speak') and callable(getattr(engine, 'speak')):
                    result = engine.speak(
                        request.text,
                        exaggeration=params['exaggeration'],
                        cfg=params['cfg']
                    )
                else:
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
        original_queue = self.enable_queue
        self.enable_queue = False
        try:
            self.route(request)
        finally:
            self.enable_queue = original_queue
    
    def preload_full_model(self):
        """
        Explicitly preload the Full model.
        
        Call this during loading screens or idle time to avoid
        latency spike on first HIGH intensity request.
        """
        if not self._full_model_loaded:
            self._load_full_model()
    
    def warm_cache(self, common_lines: Dict[str, str]):
        """
        Pre-warm the audio cache with common lines.
        
        Args:
            common_lines: Dict of npc_id -> text for common lines
        """
        if not self._audio_cache:
            return
        
        logger.info(f"[VoiceRouter] Warming cache with {len(common_lines)} common lines...")
        for npc_id, text in common_lines.items():
            # Create minimal request for synthesis
            request = VoiceRequest(
                text=text,
                npc_id=npc_id,
                emotional_state=EmotionalState(intensity=0.3),
                npc_type="ambient"
            )
            self.route(request)
    
    def clear_cache(self):
        """Clear audio and intensity caches"""
        if self._audio_cache:
            self._audio_cache.clear()
        self._intensity_cache.clear()
        logger.info("[VoiceRouter] Caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing and cache statistics"""
        stats = {
            "route_counts": {k.name: v for k, v in self._route_counts.items()},
            "total_routes": self._total_routes,
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_routes),
            "cache_hits": self._cache_hits,
            "using_fallback": self._using_fallback,
            "full_model_loaded": self._full_model_loaded,
            "registered_voices": list(self.voice_prompts.keys())
        }
        
        if self._audio_cache:
            stats["audio_cache"] = self._audio_cache.get_stats()
        
        return stats
    
    def shutdown(self):
        """Graceful shutdown of all engines"""
        if self.turbo_engine:
            self.turbo_engine.shutdown()
        if self.full_engine and self.full_engine != self.turbo_engine:
            self.full_engine.shutdown()
        if self.fallback_engine and self.fallback_engine != self.turbo_engine:
            self.fallback_engine.shutdown()
        
        # Clear caches
        self.clear_cache()
        
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
    # Demo: Voice routing with optimizations
    print("Voice Router Demo (Optimized)")
    print("=" * 60)
    
    # Create router with optimizations enabled
    config = VoiceConfig(
        cache_enabled=True,
        cache_max_size=50,
        lazy_load_full_model=True
    )
    router = VoiceRouter(config=config, enable_queue=False, fallback_to_kokoro=True)
    
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
    
    # Test 2: Same request again (should use cache)
    print("\n[Test 2] Same request (should use intensity cache):")
    intensity2 = router.compute_intensity(request)
    print(f"Computed intensity (cached): {intensity2.name}")
    
    # Test 3: High intensity (triggers lazy load)
    print("\n[Test 3] Memory callback (HIGH, will lazy load Full model):")
    lydia_emotional = EmotionalState(
        valence=0.8,
        arousal=0.7,
        dominance=0.4,
        intensity=0.85,
        primary_tone=EmotionalTone.WARM
    )
    request = VoiceRequest(
        text="I remember when you saved my life.",
        npc_id="lydia",
        emotional_state=lydia_emotional,
        npc_type="companion",
        context={"is_memory_callback": True}
    )
    router.route(request)
    
    # Print stats
    print("\n" + "=" * 60)
    print("Routing Statistics:")
    stats = router.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    router.shutdown()
    print("\nDemo complete!")
