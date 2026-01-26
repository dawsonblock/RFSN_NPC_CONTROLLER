"""
Tests for CGW SSL Guard gate integration.
"""
import pytest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.cgw_integration import (
    CGWManager, AttentionCandidate, get_cgw_manager, reset_cgw_manager
)
from world_model import NPCAction


class TestCGWManager:
    """Tests for CGWManager wrapper."""
    
    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset global manager before each test."""
        reset_cgw_manager()
        yield
        reset_cgw_manager()
    
    def test_singleton_pattern(self):
        """Test that get_cgw_manager returns same instance."""
        mgr1 = get_cgw_manager()
        mgr2 = get_cgw_manager()
        assert mgr1 is mgr2
    
    def test_submit_candidate(self):
        """Test submitting a candidate."""
        mgr = get_cgw_manager()
        
        candidate = AttentionCandidate(
            decision_id="test-001",
            npc_id="guard",
            action=NPCAction.GREET,
            score=0.8
        )
        
        mgr.submit_candidate(candidate)
        assert len(mgr.gate.candidates) == 1
    
    def test_forced_signal_bypass(self):
        """Test that forced signals bypass competition."""
        mgr = get_cgw_manager()
        
        # Submit normal candidate
        candidate = AttentionCandidate(
            decision_id="normal-001",
            npc_id="guard",
            action=NPCAction.GREET,
            score=0.8
        )
        mgr.submit_candidate(candidate)
        
        # Inject forced signal
        slot_id = mgr.inject_forced("safety_gate", NPCAction.REFUSE, "SAFETY_OVERRIDE")
        
        # Forced should win
        action, decision_id, is_forced = mgr.select_and_commit()
        
        assert action == NPCAction.REFUSE
        assert is_forced == True
        assert decision_id == slot_id
    
    def test_normal_competition(self):
        """Test normal candidate competition."""
        mgr = get_cgw_manager()
        
        # Submit two candidates
        c1 = AttentionCandidate(
            decision_id="high-score",
            npc_id="guard",
            action=NPCAction.GREET,
            score=0.9
        )
        c2 = AttentionCandidate(
            decision_id="low-score",
            npc_id="guard",
            action=NPCAction.EXPLAIN,
            score=0.3
        )
        mgr.submit_candidate(c1)
        mgr.submit_candidate(c2)
        
        # High score should win
        had_winner = mgr.tick()
        assert had_winner == True
        
        state = mgr.get_current_state()
        assert state is not None
        assert state.content_id() == "high-score"
    
    def test_has_forced_pending(self):
        """Test checking for pending forced signals."""
        mgr = get_cgw_manager()
        
        assert mgr.has_forced_pending() == False
        
        mgr.inject_forced("test", NPCAction.REFUSE)
        assert mgr.has_forced_pending() == True
        
        # Process forced
        mgr.tick()
        assert mgr.has_forced_pending() == False
    
    def test_get_stats(self):
        """Test getting CGW statistics."""
        mgr = get_cgw_manager()
        
        stats = mgr.get_stats()
        assert "gate_cycle" in stats
        assert "cgw_cycle" in stats
        assert "forced_queue_size" in stats
        assert stats["forced_queue_size"] == 0


class TestAttentionCandidate:
    """Tests for AttentionCandidate."""
    
    def test_to_candidate_conversion(self):
        """Test conversion to CGW Candidate."""
        ac = AttentionCandidate(
            decision_id="test-123",
            npc_id="merchant",
            action=NPCAction.OFFER,
            score=0.75,
            urgency=0.5
        )
        
        cgw_candidate = ac.to_candidate()
        
        assert cgw_candidate.slot_id == "test-123"
        assert cgw_candidate.saliency == 0.75
        assert cgw_candidate.urgency == 0.5
        assert b"merchant" in cgw_candidate.content_payload
        assert b"offer" in cgw_candidate.content_payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
