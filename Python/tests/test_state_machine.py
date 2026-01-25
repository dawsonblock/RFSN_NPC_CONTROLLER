"""
Tests for state machine correctness after surgical upgrade.
Verifies that NPCAction case normalization works correctly.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state_machine import RFSNStateMachine


class TestApplyTransitionCaseInsensitive:
    """Verify that apply_transition handles case normalization correctly."""
    
    def test_lowercase_insult_decreases_affinity(self):
        """INSULT action passed as lowercase should still decrease affinity."""
        sm = RFSNStateMachine()
        state_before = {
            "mood": "neutral",
            "affinity": 0.0,
            "relationship": "stranger"
        }
        
        # Pass lowercase (as orchestrator does via NPCAction.value)
        result = sm.apply_transition(state_before, "greet", "insult")
        
        assert result["affinity"] < 0.0, "INSULT should decrease affinity"
        assert result["mood"] == "angry", "INSULT should make NPC angry"
    
    def test_lowercase_apologize_increases_affinity(self):
        """APOLOGIZE action passed as lowercase should increase affinity."""
        sm = RFSNStateMachine()
        state_before = {
            "mood": "angry",
            "affinity": -0.3,
            "relationship": "rival"
        }
        
        result = sm.apply_transition(state_before, "apologize", "apologize")
        
        assert result["affinity"] > state_before["affinity"], "APOLOGIZE should increase affinity"
    
    def test_lowercase_compliment_increases_affinity(self):
        """COMPLIMENT action passed as lowercase should increase affinity."""
        sm = RFSNStateMachine()
        state_before = {
            "mood": "neutral",
            "affinity": 0.0,
            "relationship": "stranger"
        }
        
        result = sm.apply_transition(state_before, "greet", "compliment")
        
        assert result["affinity"] > 0.0, "COMPLIMENT should increase affinity"
        assert result["mood"] == "happy", "COMPLIMENT should make NPC happy"
    
    def test_uppercase_still_works(self):
        """Uppercase actions should also work correctly."""
        sm = RFSNStateMachine()
        state_before = {
            "mood": "neutral",
            "affinity": 0.0,
            "relationship": "stranger"
        }
        
        result = sm.apply_transition(state_before, "greet", "HELP")
        
        assert result["affinity"] > 0.0, "HELP should increase affinity"
    
    def test_mixed_case_works(self):
        """Mixed case actions should work correctly via normalization."""
        sm = RFSNStateMachine()
        state_before = {
            "mood": "neutral",
            "affinity": 0.0,
            "relationship": "stranger"
        }
        
        result = sm.apply_transition(state_before, "Greet", "Agree")
        
        assert result["affinity"] >= 0.0, "AGREE should not decrease affinity"


class TestStateTransitionEffects:
    """Test that state transitions produce correct side effects."""
    
    def test_threaten_makes_enemy(self):
        """THREATEN should eventually make NPC an enemy."""
        sm = RFSNStateMachine()
        state = {
            "mood": "neutral",
            "affinity": -0.3,
            "relationship": "rival"
        }
        
        # Multiple threatens should push to enemy
        for _ in range(3):
            state = sm.apply_transition(state, "threaten", "threaten")
        
        assert state["affinity"] < -0.5, "Multiple THREATEN should significantly decrease affinity"
        assert state["relationship"] == "enemy", "Low affinity should result in enemy relationship"
    
    def test_help_builds_trust(self):
        """HELP actions should build trust and improve relationship."""
        sm = RFSNStateMachine()
        state = {
            "mood": "neutral",
            "affinity": 0.3,
            "relationship": "friend"
        }
        
        # Multiple helps should push to ally
        for _ in range(3):
            state = sm.apply_transition(state, "help", "help")
        
        assert state["affinity"] > 0.5, "Multiple HELP should increase affinity"
        assert state["relationship"] == "ally", "High affinity should result in ally relationship"
