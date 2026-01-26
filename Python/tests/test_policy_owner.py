"""
Test suite for PolicyOwner and DecisionRecord.

Tests:
1. DecisionRecord creation and persistence
2. decision_id uniqueness
3. PolicyOwner.select_action() produces DecisionRecord
4. Reward join correctness (requires valid decision_id)
5. Trace reward propagation
6. Single policy owner enforcement
7. Deterministic mode (LEARNING_DISABLED)
"""
import os
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set test mode before imports
os.environ["LEARNING_DISABLED"] = "0"
os.environ["STRICT_REWARDS"] = "1"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.decision_record import (
    DecisionRecord, PolicyDecision, DecisionLogger, get_decision_logger
)
from learning.policy_owner import (
    PolicyOwner, AbstractContext, get_policy_owner, reset_policy_owner,
    LEARNING_DISABLED
)
from learning.action_trace import ActionTracer
from learning.reward_signal import RewardChannel, RewardValidationError
from learning.state_abstraction import StateAbstractor
from world_model import NPCAction


class TestDecisionRecord:
    """Tests for DecisionRecord creation and serialization."""
    
    def test_decision_record_creation(self):
        """Test that DecisionRecord generates unique decision_id."""
        dr1 = DecisionRecord(npc_id="guard", session_id="s1", chosen_action_id="GREET")
        dr2 = DecisionRecord(npc_id="guard", session_id="s1", chosen_action_id="GREET")
        
        assert dr1.decision_id != dr2.decision_id, "decision_ids should be unique"
        assert len(dr1.decision_id) == 36, "decision_id should be UUID format"
    
    def test_decision_record_serialization(self):
        """Test JSON serialization roundtrip."""
        dr = DecisionRecord(
            npc_id="merchant",
            session_id="session123",
            abstract_state_key="LOCATION:shop|MOOD:friendly",
            chosen_action_id="OFFER",
            exploration=True
        )
        
        json_str = dr.to_json()
        dr2 = DecisionRecord.from_json(json_str)
        
        assert dr2.decision_id == dr.decision_id
        assert dr2.npc_id == dr.npc_id
        assert dr2.chosen_action_id == dr.chosen_action_id
        assert dr2.exploration == True
    
    def test_state_hash_determinism(self):
        """Test that same state produces same hash."""
        state = {"mood": "happy", "affinity": 0.5, "location": "tavern"}
        
        hash1 = DecisionRecord.compute_state_hash(state)
        hash2 = DecisionRecord.compute_state_hash(state)
        
        assert hash1 == hash2, "Same state should produce same hash"
        
        # Different order should also produce same hash (sorted keys)
        state2 = {"affinity": 0.5, "location": "tavern", "mood": "happy"}
        hash3 = DecisionRecord.compute_state_hash(state2)
        assert hash1 == hash3, "Key order should not affect hash"


class TestDecisionLogger:
    """Tests for DecisionLogger persistence."""
    
    @pytest.fixture
    def temp_logger(self, tmp_path):
        """Create a temporary decision logger."""
        log_path = tmp_path / "decisions.jsonl"
        return DecisionLogger(log_path)
    
    def test_log_and_retrieve(self, temp_logger):
        """Test logging and retrieving decisions."""
        dr = DecisionRecord(
            npc_id="guard",
            session_id="test-session",
            chosen_action_id="GREET"
        )
        
        temp_logger.log(dr)
        
        assert temp_logger.exists(dr.decision_id)
        
        retrieved = temp_logger.get(dr.decision_id)
        assert retrieved is not None
        assert retrieved.npc_id == "guard"
        assert retrieved.chosen_action_id == "GREET"
    
    def test_unknown_decision_not_found(self, temp_logger):
        """Test that unknown decision_id returns None."""
        assert not temp_logger.exists("unknown-id")
        assert temp_logger.get("unknown-id") is None


class TestRewardBinding:
    """Tests for reward binding to decision_id."""
    
    @pytest.fixture
    def reward_channel(self, tmp_path):
        """Create temporary reward channel."""
        log_path = tmp_path / "rewards.jsonl"
        quarantine_path = tmp_path / "quarantine.jsonl"
        return RewardChannel(str(log_path), str(quarantine_path))
    
    @pytest.fixture
    def decision_logger(self, tmp_path):
        """Create temporary decision logger."""
        log_path = tmp_path / "decisions.jsonl"
        return DecisionLogger(log_path)
    
    def test_reward_requires_decision_id(self, reward_channel):
        """Test that reward submission requires decision_id."""
        with patch.object(reward_channel, '_decision_logger') as mock_logger:
            mock_logger.exists.return_value = False
            
            # In STRICT_REWARDS mode, should raise exception
            with pytest.raises(RewardValidationError, match="required"):
                reward_channel.submit_reward(
                    decision_id="",
                    reward=0.8,
                    source="test"
                )
    
    def test_unknown_decision_id_rejected(self, reward_channel):
        """Test that unknown decision_id is rejected."""
        with patch.object(reward_channel, '_decision_logger') as mock_logger:
            mock_logger.exists.return_value = False
            
            # In STRICT_REWARDS mode, should raise exception
            with pytest.raises(RewardValidationError, match="Unknown"):
                reward_channel.submit_reward(
                    decision_id="unknown-decision-id",
                    reward=0.8,
                    source="test"
                )
    
    def test_valid_reward_accepted(self, tmp_path):
        """Test that reward with valid decision_id is accepted."""
        # Create logger and log a decision
        log_path = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path)
        
        dr = DecisionRecord(npc_id="guard", session_id="s1", chosen_action_id="GREET")
        logger.log(dr)
        
        # Create reward channel with same logger
        reward_channel = RewardChannel(
            str(tmp_path / "rewards.jsonl"),
            str(tmp_path / "quarantine.jsonl")
        )
        reward_channel._decision_logger = logger
        
        success, error = reward_channel.submit_reward(
            decision_id=dr.decision_id,
            reward=0.9,
            source="test"
        )
        
        assert success
        assert error is None


class TestActionTracer:
    """Tests for action trace credit assignment."""
    
    def test_trace_stores_decision_id(self):
        """Test that trace stores decision_id correctly."""
        tracer = ActionTracer(trace_length=5)
        
        tracer.record_step(
            decision_id="dec-001",
            npc_id="guard",
            session_id="session-1",
            abstract_state_key="STATE:key",
            action_id="GREET"
        )
        
        recent = tracer.get_recent("guard", "session-1", count=1)
        assert len(recent) == 1
        assert recent[0]["decision_id"] == "dec-001"
        assert recent[0]["action_id"] == "GREET"
    
    def test_credit_propagation(self):
        """Test backward credit propagation."""
        tracer = ActionTracer(trace_length=5, gamma=0.9)
        
        # Record 3 decisions
        for i in range(3):
            tracer.record_step(
                decision_id=f"dec-{i}",
                npc_id="guard",
                session_id="session-1",
                abstract_state_key=f"STATE:key-{i}",
                action_id="ACTION"
            )
        
        # Propagate reward from last decision
        updates = tracer.propagate_reward(
            decision_id="dec-2",
            npc_id="guard",
            session_id="session-1",
            reward=1.0
        )
        
        assert len(updates) == 3
        # Last decision gets full reward
        assert updates[0]["decision_id"] == "dec-2"
        assert updates[0]["reward"] == 1.0
        # Previous decisions get decayed rewards
        assert updates[1]["reward"] == 0.9
        assert updates[2]["reward"] == pytest.approx(0.81, 0.01)
    
    def test_session_isolation(self):
        """Test that traces are isolated per session."""
        tracer = ActionTracer()
        
        tracer.record_step("dec-1", "npc1", "session-A", "STATE:1", "ACTION")
        tracer.record_step("dec-2", "npc1", "session-B", "STATE:2", "ACTION")
        
        session_a = tracer.get_recent("npc1", "session-A")
        session_b = tracer.get_recent("npc1", "session-B")
        
        assert len(session_a) == 1
        assert len(session_b) == 1
        assert session_a[0]["decision_id"] == "dec-1"
        assert session_b[0]["decision_id"] == "dec-2"


class TestStateAbstractor:
    """Tests for state abstraction stability."""
    
    def test_deterministic_abstraction(self):
        """Test same state produces same abstract key."""
        abstractor = StateAbstractor()
        
        state = {"mood": "happy", "affinity": 0.5, "relationship": "friendly"}
        
        key1 = abstractor.abstract(state)
        key2 = abstractor.abstract(state)
        
        assert key1 == key2, "Same state should produce same key"
    
    def test_version_consistency(self):
        """Test abstractor has version info."""
        abstractor = StateAbstractor()
        
        assert hasattr(abstractor, 'VERSION')
        assert hasattr(abstractor, 'SCHEMA_ID')
        assert abstractor.VERSION == "2.0"
    
    def test_bucket_rules(self):
        """Test bucket rules work correctly."""
        abstractor = StateAbstractor()
        
        assert abstractor.bucket_affinity(-0.5) == "neg"
        assert abstractor.bucket_affinity(0.0) == "neutral"
        assert abstractor.bucket_affinity(0.5) == "pos"
        
        assert abstractor.bucket_mood("angry") == "hostile"
        assert abstractor.bucket_mood("happy") == "friendly"
        assert abstractor.bucket_mood("calm") == "neutral"


class TestPolicyOwner:
    """Tests for PolicyOwner as single authoritative policy."""
    
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        """Reset global policy owner before each test."""
        reset_policy_owner()
        yield
        reset_policy_owner()
    
    def test_singleton_pattern(self):
        """Test that get_policy_owner returns same instance."""
        owner1 = get_policy_owner()
        owner2 = get_policy_owner()
        
        assert owner1 is owner2
    
    def test_select_produces_decision_record(self):
        """Test that select_action produces DecisionRecord."""
        reset_policy_owner()
        
        # Create mock context
        ctx = AbstractContext(
            npc_id="guard",
            session_id="test-session",
            abstract_state_key="STATE:test",
            raw_state_hash="abc123",
            mood_bucket="neutral",
            affinity_bucket="neutral",
            player_signal="question"
        )
        
        candidates = [NPCAction.GREET, NPCAction.EXPLAIN]
        
        # Use LEARNING_DISABLED mode for deterministic selection
        with patch.dict(os.environ, {"LEARNING_DISABLED": "1"}):
            with patch('learning.policy_owner.LEARNING_DISABLED', True):
                owner = PolicyOwner(seed=42)
                decision = owner.select_action(ctx, candidates)
        
        assert isinstance(decision, PolicyDecision)
        assert isinstance(decision.record, DecisionRecord)
        assert decision.record.npc_id == "guard"
        assert decision.record.session_id == "test-session"


class TestDeterministicMode:
    """Tests for deterministic mode (LEARNING_DISABLED)."""
    
    def test_learning_disabled_produces_deterministic_output(self):
        """Test that with LEARNING_DISABLED, same input produces same output."""
        with patch.dict(os.environ, {"LEARNING_DISABLED": "1"}):
            # Force reimport to pick up env var
            import importlib
            import learning.policy_owner as po
            importlib.reload(po)
            
            # This test would need more setup to fully test determinism
            # For now, just verify the flag is checked
            assert po.LEARNING_DISABLED == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
