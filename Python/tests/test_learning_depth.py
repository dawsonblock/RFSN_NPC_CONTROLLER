"""
Tests for v1.3 Learning Depth modules.
Covers contextual bandit, mode bandit, metrics guard, and ground-truth rewards.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.contextual_bandit import (
    BanditContext, LinUCBArm, ContextualBandit, ConversationPhase
)
from learning.mode_bandit import ModeBandit, PhrasingMode, ModeContext
from learning.metrics_guard import MetricsGuard, TurnMetrics
from learning.reward_model import RewardModel
from world_model import NPCAction, PlayerSignal


class TestBanditContext:
    """Test BanditContext feature vector creation."""
    
    def test_feature_vector_dimension(self):
        """Feature vector should have correct dimension."""
        ctx = BanditContext(
            mood_bucket="neutral",
            affinity_bucket="neutral",
            player_signal="greet",
            last_action=None,
            conversation_phase="early",
            has_safety_flag=False,
            turn_count=1
        )
        
        vec = ctx.to_feature_vector()
        assert len(vec) == BanditContext.feature_dim()
        assert len(vec) == 12  # 3+3+3+1+1+1
    
    def test_feature_vector_one_hot(self):
        """One-hot encodings should have exactly one 1."""
        ctx = BanditContext(
            mood_bucket="hostile",
            affinity_bucket="pos",
            player_signal="attack",
            last_action="greet",
            conversation_phase="late",
            has_safety_flag=True,
            turn_count=5
        )
        
        vec = ctx.to_feature_vector()
        
        # Mood (first 3): hostile = [1,0,0]
        assert vec[0] == 1.0
        assert sum(vec[:3]) == 1.0
        
        # Affinity (next 3): pos = [0,0,1]
        assert vec[5] == 1.0
        assert sum(vec[3:6]) == 1.0
    
    def test_from_state_creates_valid_context(self):
        """from_state should create valid context."""
        ctx = BanditContext.from_state(
            mood="angry",
            affinity=-0.5,
            player_signal=PlayerSignal.ATTACK,
            last_action=NPCAction.GREET,
            turn_count=3,
            has_safety_flag=True
        )
        
        assert ctx.mood_bucket == "hostile"
        assert ctx.affinity_bucket == "neg"
        assert ctx.conversation_phase == "mid"


class TestLinUCBArm:
    """Test LinUCB arm operations."""
    
    def test_initial_ucb(self):
        """Initial UCB should be reasonable for uniform prior."""
        arm = LinUCBArm(dim=5)
        x = np.ones(5) / np.sqrt(5)  # Unit vector
        
        ucb = arm.get_ucb(x, alpha=1.0)
        # With identity A and zero b, UCB should be just exploration term
        assert ucb > 0
    
    def test_update_increases_trials(self):
        """Update should increase trial count."""
        arm = LinUCBArm(dim=5)
        x = np.array([1, 0, 0, 0, 1], dtype=np.float64)
        
        assert arm.n == 0
        arm.update(x, reward=0.8)
        assert arm.n == 1
    
    def test_serialization_roundtrip(self):
        """Arm should serialize and deserialize correctly."""
        arm = LinUCBArm(dim=4)
        x = np.array([1, 0, 1, 0], dtype=np.float64)
        arm.update(x, 0.7)
        
        data = arm.to_dict()
        arm2 = LinUCBArm.from_dict(data)
        
        assert arm2.n == arm.n
        assert np.allclose(arm2.A_inv, arm.A_inv)  # Sherman-Morrison uses A_inv
        assert np.allclose(arm2.b, arm.b)


class TestContextualBandit:
    """Test contextual bandit selection and update."""
    
    def test_select_with_candidates(self):
        """Select should return one of the candidates."""
        bandit = ContextualBandit()
        ctx = BanditContext(
            mood_bucket="neutral",
            affinity_bucket="neutral",
            player_signal="greet",
            last_action=None,
            conversation_phase="early",
            has_safety_flag=False
        )
        
        candidates = [NPCAction.GREET, NPCAction.INQUIRE, NPCAction.AGREE]
        selected = bandit.select(ctx, candidates)
        
        assert selected in candidates
    
    def test_update_modifies_arm(self):
        """Update should modify the arm for that action."""
        bandit = ContextualBandit()
        ctx = BanditContext(
            mood_bucket="friendly",
            affinity_bucket="pos",
            player_signal="greet",
            last_action=None,
            conversation_phase="early",
            has_safety_flag=False
        )
        
        action = NPCAction.GREET
        bandit.update(ctx, action, reward=0.9)
        
        arm = bandit._get_arm(action)
        assert arm.n == 1


class TestModeBandit:
    """Test mode bandit for phrasing selection."""
    
    def test_select_returns_valid_mode(self):
        """Mode selection should return valid PhrasingMode."""
        bandit = ModeBandit()
        ctx = ModeContext(
            relationship="friend",
            npc_personality="merchant",
            emotional_intensity="medium",
            last_mode_used=None,
            correction_rate=0.0
        )
        
        mode = bandit.select(ctx)
        assert isinstance(mode, PhrasingMode)
    
    def test_mode_instructions_exist(self):
        """Each mode should have instructions."""
        bandit = ModeBandit()
        
        for mode in PhrasingMode:
            instructions = bandit.get_mode_instructions(mode)
            assert len(instructions) > 0


class TestMetricsGuard:
    """Test metrics guard for regression detection."""
    
    def test_initial_learning_rate(self):
        """Initial learning rate should be normal."""
        guard = MetricsGuard()
        assert guard.get_learning_rate() == MetricsGuard.NORMAL_LEARNING_RATE
    
    def test_record_turn(self):
        """Recording turns should update stats."""
        guard = MetricsGuard()
        
        guard.record_turn(
            reward=0.8,
            latency_ms=150,
            was_corrected=False,
            was_blocked=False,
            npc_name="TestNPC"
        )
        
        stats = guard.get_stats()
        assert stats["total_turns"] == 1
    
    def test_high_correction_reduces_rate(self):
        """High correction rate should reduce learning rate."""
        guard = MetricsGuard()
        
        # Record many corrections
        for _ in range(50):
            guard.record_turn(
                reward=0.3,
                latency_ms=100,
                was_corrected=True,
                was_blocked=False
            )
        
        # Should have reduced or frozen learning
        assert guard.get_learning_rate() < MetricsGuard.NORMAL_LEARNING_RATE


class TestGroundTruthRewards:
    """Test ground-truth reward anchors."""
    
    def test_thats_wrong_anchor(self):
        """'That's wrong' should trigger negative anchor."""
        model = RewardModel()
        
        reward, anchor_type = model.compute_ground_truth_anchor(
            "No that's wrong, I asked about dragons"
        )
        
        assert anchor_type == "thats_wrong"
        assert reward == RewardModel.ANCHOR_THATS_WRONG
    
    def test_explicit_thanks_anchor(self):
        """'Thank you' should trigger positive anchor."""
        model = RewardModel()
        
        reward, anchor_type = model.compute_ground_truth_anchor(
            "Thank you, that's exactly what I needed!"
        )
        
        assert anchor_type == "explicit_thanks"
        assert reward == RewardModel.ANCHOR_EXPLICIT_THANKS
    
    def test_repeat_question_anchor(self):
        """Repeated question should trigger negative anchor."""
        model = RewardModel()
        
        reward, anchor_type = model.compute_ground_truth_anchor(
            "Where can I find the blacksmith?",
            previous_user_text="Where is the blacksmith?"
        )
        
        assert anchor_type == "repeat_question"
        assert reward == RewardModel.ANCHOR_REPEAT_QUESTION
    
    def test_no_anchor_for_normal_input(self):
        """Normal input should return no anchor."""
        model = RewardModel()
        
        reward, anchor_type = model.compute_ground_truth_anchor(
            "Hello there"
        )
        
        assert anchor_type == "none"
        assert reward == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
