"""
Tests for Temporal Memory and Action Lattice Expansion.
"""
import pytest
import time
from world_model import NPCAction, StateSnapshot
from learning.temporal_memory import TemporalMemory, Experience


class TestNPCActionExpansion:
    """Test the expanded NPCAction enum with nuance variants."""
    
    def test_total_action_count(self):
        """Verify 32 total actions (20 core + 12 nuance)."""
        assert len(NPCAction) == 32
    
    def test_get_base_action_for_variant(self):
        """Nuance variants should map back to their base action."""
        assert NPCAction.get_base_action(NPCAction.AGREE_RELUCTANTLY) == NPCAction.AGREE
        assert NPCAction.get_base_action(NPCAction.AGREE_ENTHUSIASTICALLY) == NPCAction.AGREE
        assert NPCAction.get_base_action(NPCAction.REFUSE_POLITELY) == NPCAction.REFUSE
        assert NPCAction.get_base_action(NPCAction.HELP_GRUDGINGLY) == NPCAction.HELP
        assert NPCAction.get_base_action(NPCAction.WARN_STERNLY) == NPCAction.THREATEN
        assert NPCAction.get_base_action(NPCAction.DEFLECT) == NPCAction.IGNORE
        assert NPCAction.get_base_action(NPCAction.PROBE) == NPCAction.INQUIRE
        assert NPCAction.get_base_action(NPCAction.CONFIDE) == NPCAction.EXPLAIN
        assert NPCAction.get_base_action(NPCAction.HINT) == NPCAction.EXPLAIN
    
    def test_get_base_action_for_core_action(self):
        """Core actions should map to themselves."""
        assert NPCAction.get_base_action(NPCAction.GREET) == NPCAction.GREET
        assert NPCAction.get_base_action(NPCAction.AGREE) == NPCAction.AGREE
        assert NPCAction.get_base_action(NPCAction.HELP) == NPCAction.HELP
    
    def test_get_nuance_variants(self):
        """Base actions should return their nuance variants."""
        agree_variants = NPCAction.get_nuance_variants(NPCAction.AGREE)
        assert NPCAction.AGREE_RELUCTANTLY in agree_variants
        assert NPCAction.AGREE_ENTHUSIASTICALLY in agree_variants
        
        help_variants = NPCAction.get_nuance_variants(NPCAction.HELP)
        assert NPCAction.HELP_GRUDGINGLY in help_variants
        assert NPCAction.HELP_EAGERLY in help_variants
        
        explain_variants = NPCAction.get_nuance_variants(NPCAction.EXPLAIN)
        assert NPCAction.CONFIDE in explain_variants
        assert NPCAction.HINT in explain_variants
    
    def test_get_nuance_variants_empty_for_non_expandable(self):
        """Actions without variants should return empty list."""
        assert NPCAction.get_nuance_variants(NPCAction.GREET) == []
        assert NPCAction.get_nuance_variants(NPCAction.ATTACK) == []
    
    def test_is_nuance_variant(self):
        """Correctly identify nuance vs core actions."""
        assert NPCAction.AGREE_RELUCTANTLY.is_nuance_variant() is True
        assert NPCAction.HELP_EAGERLY.is_nuance_variant() is True
        assert NPCAction.CONFIDE.is_nuance_variant() is True
        
        assert NPCAction.GREET.is_nuance_variant() is False
        assert NPCAction.AGREE.is_nuance_variant() is False
        assert NPCAction.HELP.is_nuance_variant() is False


class TestTemporalMemory:
    """Test the TemporalMemory anticipation mechanism."""
    
    def setup_method(self):
        """Create fresh TemporalMemory for each test."""
        self.tm = TemporalMemory(max_size=50, decay_rate=0.95)
        self.state = StateSnapshot(
            mood="neutral",
            affinity=0.5,
            relationship="friend",
            recent_sentiment=0.3
        )
    
    def test_record_experience(self):
        """Recording should add to memory."""
        assert len(self.tm) == 0
        self.tm.record(self.state, NPCAction.HELP, 0.8)
        assert len(self.tm) == 1
    
    def test_max_size_limit(self):
        """Memory should not exceed max_size."""
        tm = TemporalMemory(max_size=5)
        for i in range(10):
            tm.record(self.state, NPCAction.HELP, 0.8)
        assert len(tm) == 5
    
    def test_positive_reward_positive_adjustment(self):
        """High reward should yield positive adjustment."""
        self.tm.record(self.state, NPCAction.HELP, 0.9)  # Good outcome
        
        similar_state = StateSnapshot(
            mood="neutral",
            affinity=0.55,
            relationship="friend",
            recent_sentiment=0.35
        )
        adjustments = self.tm.get_prior_adjustments(similar_state)
        
        assert NPCAction.HELP in adjustments
        assert adjustments[NPCAction.HELP] > 0  # Should be positive
    
    def test_negative_reward_negative_adjustment(self):
        """Low reward should yield negative adjustment."""
        self.tm.record(self.state, NPCAction.REFUSE, 0.1)  # Bad outcome
        
        similar_state = StateSnapshot(
            mood="neutral",
            affinity=0.55,
            relationship="friend",
            recent_sentiment=0.35
        )
        adjustments = self.tm.get_prior_adjustments(similar_state)
        
        assert NPCAction.REFUSE in adjustments
        assert adjustments[NPCAction.REFUSE] < 0  # Should be negative
    
    def test_dissimilar_state_no_adjustment(self):
        """Very different state should not trigger adjustments."""
        self.tm.record(self.state, NPCAction.HELP, 0.9)
        
        very_different_state = StateSnapshot(
            mood="angry",
            affinity=-0.8,
            relationship="enemy",
            recent_sentiment=-0.9,
            combat_active=True
        )
        adjustments = self.tm.get_prior_adjustments(very_different_state)
        
        # May have no adjustments or very weak ones due to low similarity
        if NPCAction.HELP in adjustments:
            assert abs(adjustments[NPCAction.HELP]) < 0.05  # Very small
    
    def test_empty_memory_no_adjustments(self):
        """Empty memory should return empty adjustments."""
        adjustments = self.tm.get_prior_adjustments(self.state)
        assert adjustments == {}
    
    def test_adjustment_clamping(self):
        """Adjustments should be clamped to adjustment_scale."""
        # Record many high-reward experiences
        for _ in range(20):
            self.tm.record(self.state, NPCAction.HELP, 1.0)
        
        adjustments = self.tm.get_prior_adjustments(self.state)
        
        if NPCAction.HELP in adjustments:
            # Should not exceed adjustment_scale (default 0.1)
            assert adjustments[NPCAction.HELP] <= 0.1
    
    def test_clear_memory(self):
        """Clear should empty the memory."""
        self.tm.record(self.state, NPCAction.HELP, 0.8)
        assert len(self.tm) == 1
        self.tm.clear()
        assert len(self.tm) == 0
    
    def test_get_similar_experiences(self):
        """Should return most similar experiences."""
        state1 = StateSnapshot(mood="neutral", affinity=0.5, relationship="friend", recent_sentiment=0.3)
        state2 = StateSnapshot(mood="angry", affinity=-0.5, relationship="enemy", recent_sentiment=-0.5)
        
        self.tm.record(state1, NPCAction.HELP, 0.8)
        self.tm.record(state2, NPCAction.ATTACK, 0.3)
        
        # Query with similar to state1
        similar = self.tm.get_similar_experiences(state1, top_k=1)
        assert len(similar) == 1
        assert similar[0].action == NPCAction.HELP


class TestNPCActionBanditTemporalIntegration:
    """Test that NPCActionBandit accepts temporal adjustments."""
    
    def test_select_with_temporal_adjustments(self):
        """Temporal adjustments should bias selection."""
        from learning.npc_action_bandit import NPCActionBandit
        
        bandit = NPCActionBandit(path=None)  # In-memory only
        
        candidates = [NPCAction.HELP, NPCAction.REFUSE, NPCAction.AGREE]
        
        # No temporal adjustments - random selection (due to explore_bias)
        # With positive HELP adjustment, should bias toward HELP
        temporal_adjustments = {
            NPCAction.HELP: 0.1,  # Strong positive
            NPCAction.REFUSE: -0.1,  # Strong negative
        }
        
        # Run many times and check bias
        help_count = 0
        n_trials = 100
        for _ in range(n_trials):
            selected = bandit.select(
                key="test",
                candidates=candidates,
                temporal_adjustments=temporal_adjustments,
                explore_bias=0.0  # Disable exploration to see bias
            )
            if selected == NPCAction.HELP:
                help_count += 1
        
        # With temporal boost, HELP should be selected more often than 33%
        assert help_count > n_trials * 0.4, f"HELP selected only {help_count}/{n_trials} times"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
