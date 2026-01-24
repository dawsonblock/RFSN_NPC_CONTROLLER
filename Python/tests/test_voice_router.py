"""
Unit tests for Voice Router - automatic TTS engine selection.

Tests cover:
- VoiceIntensity computation from emotional state
- Routing decisions based on NPC type and context
- Parameter mapping for each intensity level
- Fallback behavior
"""

import pytest
from unittest.mock import MagicMock, patch

from voice_router import (
    VoiceRouter, VoiceRequest, VoiceConfig, VoiceIntensity
)
from emotional_tone import EmotionalState, EmotionalTone


class TestVoiceIntensity:
    """Test VoiceIntensity enum values"""
    
    def test_intensity_ordering(self):
        """Verify LOW < MEDIUM < HIGH ordering"""
        assert VoiceIntensity.LOW < VoiceIntensity.MEDIUM
        assert VoiceIntensity.MEDIUM < VoiceIntensity.HIGH
        assert VoiceIntensity.LOW.value == 0
        assert VoiceIntensity.MEDIUM.value == 1
        assert VoiceIntensity.HIGH.value == 2


class TestVoiceRouterIntensityComputation:
    """Test intensity computation from NPC state and context"""
    
    @pytest.fixture
    def router(self):
        """Create router with mock engines"""
        with patch('voice_router.VoiceRouter._init_engines'):
            router = VoiceRouter()
            router._using_fallback = True
            return router
    
    @pytest.fixture
    def neutral_state(self):
        """Neutral emotional state"""
        return EmotionalState(
            valence=0.0,
            arousal=0.3,
            dominance=0.5,
            intensity=0.3,
            primary_tone=EmotionalTone.NEUTRAL
        )
    
    @pytest.fixture
    def high_emotion_state(self):
        """High emotional intensity state"""
        return EmotionalState(
            valence=0.5,
            arousal=0.6,
            dominance=0.5,
            intensity=0.85,  # Above threshold (0.7)
            primary_tone=EmotionalTone.WARM
        )
    
    @pytest.fixture
    def high_valence_state(self):
        """High positive valence state"""
        return EmotionalState(
            valence=0.9,  # Above threshold (0.7)
            arousal=0.5,
            dominance=0.5,
            intensity=0.5,
            primary_tone=EmotionalTone.ENTHUSIASTIC
        )
    
    def test_ambient_low_emotion_returns_low(self, router, neutral_state):
        """Ambient NPC with neutral emotion → LOW"""
        request = VoiceRequest(
            text="Move along.",
            npc_id="guard_001",
            emotional_state=neutral_state,
            npc_type="ambient"
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.LOW
    
    def test_high_emotion_intensity_returns_high(self, router, high_emotion_state):
        """High emotional intensity → HIGH regardless of NPC type"""
        request = VoiceRequest(
            text="I am so grateful!",
            npc_id="shopkeeper",
            emotional_state=high_emotion_state,
            npc_type="ambient"
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH
    
    def test_high_valence_returns_high(self, router, high_valence_state):
        """Strong valence (positive or negative) → HIGH"""
        request = VoiceRequest(
            text="This is wonderful!",
            npc_id="villager",
            emotional_state=high_valence_state,
            npc_type="ambient"
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH
    
    def test_memory_callback_returns_high(self, router, neutral_state):
        """Memory callback context → HIGH"""
        request = VoiceRequest(
            text="I remember when you saved me.",
            npc_id="lydia",
            emotional_state=neutral_state,
            npc_type="companion",
            context={"is_memory_callback": True}
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH
    
    def test_relationship_moment_returns_high(self, router, neutral_state):
        """Relationship moment context → HIGH"""
        request = VoiceRequest(
            text="You mean so much to me.",
            npc_id="serana",
            emotional_state=neutral_state,
            npc_type="companion",
            context={"is_relationship_moment": True}
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH
    
    def test_companion_baseline_returns_medium(self, router, neutral_state):
        """Companion with neutral state → MEDIUM minimum"""
        request = VoiceRequest(
            text="I am sworn to carry your burdens.",
            npc_id="lydia",
            emotional_state=neutral_state,
            npc_type="companion"
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.MEDIUM
    
    def test_companion_elevated_returns_high(self, router):
        """Companion with elevated state → HIGH"""
        elevated_state = EmotionalState(
            valence=0.4,
            arousal=0.7,
            dominance=0.5,
            intensity=0.5,  # Above 0.4 threshold for companions
            primary_tone=EmotionalTone.WARM
        )
        request = VoiceRequest(
            text="Be careful out there.",
            npc_id="lydia",
            emotional_state=elevated_state,
            npc_type="companion"
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH
    
    def test_high_arousal_ambient_returns_medium(self, router):
        """Ambient NPC with high arousal (combat) → MEDIUM"""
        combat_state = EmotionalState(
            valence=-0.3,
            arousal=0.9,  # Above threshold (0.8)
            dominance=0.7,
            intensity=0.5,
            primary_tone=EmotionalTone.AGGRESSIVE
        )
        request = VoiceRequest(
            text="Stop right there!",
            npc_id="guard",
            emotional_state=combat_state,
            npc_type="ambient"
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.MEDIUM
    
    def test_force_intensity_override(self, router, neutral_state):
        """Force intensity overrides computed value"""
        request = VoiceRequest(
            text="Force this to HIGH.",
            npc_id="test_npc",
            emotional_state=neutral_state,
            npc_type="ambient",
            force_intensity=VoiceIntensity.HIGH
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH
    
    def test_cutscene_context_returns_high(self, router, neutral_state):
        """Cutscene context → HIGH"""
        request = VoiceRequest(
            text="This is the moment of truth.",
            npc_id="jarl",
            emotional_state=neutral_state,
            npc_type="main",
            context={"is_cutscene": True}
        )
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH


class TestVoiceRouterParams:
    """Test parameter mapping for intensity levels"""
    
    @pytest.fixture
    def router(self):
        with patch('voice_router.VoiceRouter._init_engines'):
            return VoiceRouter()
    
    def test_low_intensity_params(self, router):
        """LOW intensity → low exaggeration, high cfg"""
        params = router.get_params_for_intensity(VoiceIntensity.LOW)
        assert params["exaggeration"] == 0.3
        assert params["cfg"] == 0.5
    
    def test_medium_intensity_params(self, router):
        """MEDIUM intensity → medium exaggeration, medium cfg"""
        params = router.get_params_for_intensity(VoiceIntensity.MEDIUM)
        assert params["exaggeration"] == 0.6
        assert params["cfg"] == 0.4
    
    def test_high_intensity_params(self, router):
        """HIGH intensity → high exaggeration, low cfg"""
        params = router.get_params_for_intensity(VoiceIntensity.HIGH)
        assert params["exaggeration"] == 0.8
        assert params["cfg"] == 0.3


class TestVoiceRouterConfig:
    """Test VoiceConfig customization"""
    
    def test_custom_thresholds(self):
        """Custom thresholds affect intensity computation"""
        config = VoiceConfig(
            high_emotion_threshold=0.5,  # Lower threshold
            high_valence_threshold=0.5
        )
        
        with patch('voice_router.VoiceRouter._init_engines'):
            router = VoiceRouter(config=config)
        
        # State that would be MEDIUM with default thresholds
        medium_state = EmotionalState(
            valence=0.3,
            arousal=0.4,
            dominance=0.5,
            intensity=0.6,  # Above custom threshold (0.5)
            primary_tone=EmotionalTone.WARM
        )
        
        request = VoiceRequest(
            text="Test",
            npc_id="test",
            emotional_state=medium_state,
            npc_type="ambient"
        )
        
        intensity = router.compute_intensity(request)
        assert intensity == VoiceIntensity.HIGH


class TestVoiceRouterStats:
    """Test statistics tracking"""
    
    def test_route_count_tracking(self):
        """Route counts are tracked per intensity"""
        with patch('voice_router.VoiceRouter._init_engines'):
            router = VoiceRouter()
            router._using_fallback = True
            router.turbo_engine = MagicMock()
            router.turbo_engine.speak = MagicMock(return_value=True)
            router.full_engine = MagicMock()
            router.full_engine.speak = MagicMock(return_value=True)
        
        neutral_state = EmotionalState(intensity=0.3)
        
        # Route a LOW request
        request = VoiceRequest(
            text="Low test",
            npc_id="test",
            emotional_state=neutral_state,
            npc_type="ambient"
        )
        router.route(request)
        
        stats = router.get_stats()
        assert stats["route_counts"]["LOW"] == 1


class TestVoiceRequest:
    """Test VoiceRequest dataclass"""
    
    def test_default_values(self):
        """VoiceRequest has sensible defaults"""
        request = VoiceRequest(
            text="Hello",
            npc_id="test",
            emotional_state=EmotionalState()
        )
        assert request.npc_type == "ambient"
        assert request.context == {}
        assert request.force_intensity is None
        assert request.voice_prompt_path is None


class TestAudioCache:
    """Test AudioCache LRU cache with TTL"""
    
    def test_cache_put_and_get(self):
        """Cache stores and retrieves audio"""
        from voice_router import AudioCache
        cache = AudioCache(max_size=10, ttl_seconds=60.0)
        
        cache.put("hello", "npc1", VoiceIntensity.LOW, 0.3, b"audio_data")
        result = cache.get("hello", "npc1", VoiceIntensity.LOW, 0.3)
        
        assert result == b"audio_data"
    
    def test_cache_miss(self):
        """Cache returns None for missing entries"""
        from voice_router import AudioCache
        cache = AudioCache(max_size=10, ttl_seconds=60.0)
        
        result = cache.get("nonexistent", "npc1", VoiceIntensity.LOW, 0.3)
        assert result is None
    
    def test_cache_eviction(self):
        """Cache evicts oldest entries when at capacity"""
        from voice_router import AudioCache
        cache = AudioCache(max_size=2, ttl_seconds=60.0)
        
        cache.put("first", "npc1", VoiceIntensity.LOW, 0.3, b"first")
        cache.put("second", "npc1", VoiceIntensity.LOW, 0.3, b"second")
        cache.put("third", "npc1", VoiceIntensity.LOW, 0.3, b"third")
        
        # First should be evicted
        assert cache.get("first", "npc1", VoiceIntensity.LOW, 0.3) is None
        assert cache.get("second", "npc1", VoiceIntensity.LOW, 0.3) == b"second"
        assert cache.get("third", "npc1", VoiceIntensity.LOW, 0.3) == b"third"
    
    def test_cache_stats(self):
        """Cache tracks hit/miss statistics"""
        from voice_router import AudioCache
        cache = AudioCache(max_size=10, ttl_seconds=60.0)
        
        cache.put("hello", "npc1", VoiceIntensity.LOW, 0.3, b"data")
        cache.get("hello", "npc1", VoiceIntensity.LOW, 0.3)  # hit
        cache.get("miss", "npc1", VoiceIntensity.LOW, 0.3)   # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "50.0%"


class TestIntensityCache:
    """Test IntensityCache precomputation"""
    
    def test_intensity_cache_hit(self):
        """Intensity cache returns cached value"""
        from voice_router import IntensityCache
        cache = IntensityCache(ttl_seconds=60.0)
        
        state = EmotionalState(intensity=0.5)
        request = VoiceRequest(text="test", npc_id="npc1", emotional_state=state)
        
        cache.put(request, VoiceIntensity.MEDIUM)
        result = cache.get(request)
        
        assert result == VoiceIntensity.MEDIUM
    
    def test_intensity_cache_miss(self):
        """Intensity cache returns None for new requests"""
        from voice_router import IntensityCache
        cache = IntensityCache(ttl_seconds=60.0)
        
        state = EmotionalState(intensity=0.5)
        request = VoiceRequest(text="test", npc_id="npc1", emotional_state=state)
        
        result = cache.get(request)
        assert result is None
    
    def test_intensity_force_override_skips_cache(self):
        """Force intensity returns immediately without cache lookup"""
        from voice_router import IntensityCache
        cache = IntensityCache(ttl_seconds=60.0)
        
        state = EmotionalState(intensity=0.5)
        request = VoiceRequest(
            text="test", 
            npc_id="npc1", 
            emotional_state=state,
            force_intensity=VoiceIntensity.HIGH
        )
        
        # Should return force_intensity immediately
        result = cache.get(request)
        assert result == VoiceIntensity.HIGH


class TestVoiceRouterOptimizations:
    """Test VoiceRouter optimization features"""
    
    @pytest.fixture
    def optimized_router(self):
        """Create router with all optimizations enabled"""
        config = VoiceConfig(
            cache_enabled=True,
            cache_max_size=50,
            lazy_load_full_model=True,
            intensity_cache_ttl=5.0
        )
        with patch('voice_router.VoiceRouter._init_engines'):
            router = VoiceRouter(config=config)
            router._using_fallback = True
            router._full_model_loaded = False
            return router
    
    def test_lazy_load_not_triggered_for_low(self, optimized_router):
        """Full model not loaded for LOW intensity requests"""
        optimized_router.turbo_engine = MagicMock()
        optimized_router.turbo_engine.speak = MagicMock(return_value=True)
        
        state = EmotionalState(intensity=0.3)
        request = VoiceRequest(text="Low", npc_id="guard", emotional_state=state)
        
        optimized_router.route(request)
        
        # Full model should NOT be loaded
        assert not optimized_router._full_model_loaded
    
    def test_stats_include_cache_info(self, optimized_router):
        """Stats include audio cache information"""
        stats = optimized_router.get_stats()
        
        assert "audio_cache" in stats
        assert "hits" in stats["audio_cache"]
        assert "hit_rate" in stats["audio_cache"]
    
    def test_clear_cache(self, optimized_router):
        """clear_cache clears both caches"""
        # Add something to intensity cache
        state = EmotionalState(intensity=0.5)
        request = VoiceRequest(text="test", npc_id="npc1", emotional_state=state)
        optimized_router._intensity_cache.put(request, VoiceIntensity.MEDIUM)
        
        optimized_router.clear_cache()
        
        # Cache should be empty
        assert optimized_router._intensity_cache.get(request) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
