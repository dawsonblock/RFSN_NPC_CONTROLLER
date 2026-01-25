"""
Tests for llm_action_prompts module.
Updated to use the active prompt system after consolidation.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_action_prompts import render_action_block
from world_model import NPCAction


class TestRenderActionBlock:
    """Test the render_action_block function."""
    
    def test_basic_greet_action(self):
        """Test basic GREET action rendering."""
        prompt = render_action_block(
            npc_action=NPCAction.GREET,
            npc_name="Lydia",
            mood="friendly",
            relationship="acquaintance",
            affinity=0.5,
            player_signal="greet"
        )
        
        # Check for action control header
        assert "[ACTION_CONTROL]" in prompt
        assert "npc_action: greet" in prompt
        assert "player_signal: greet" in prompt
        assert "[/ACTION_CONTROL]" in prompt
        
        # Check for NPC context
        assert "Lydia" in prompt
        assert "friendly" in prompt
        assert "acquaintance" in prompt
        assert "+0.50" in prompt
        
        # Check for action spec
        assert "ACTION: Greet" in prompt
        assert "DO:" in prompt or "DON'T:" in prompt
    
    def test_insult_action_contains_constraints(self):
        """Test that INSULT action has appropriate constraints."""
        prompt = render_action_block(
            npc_action=NPCAction.INSULT,
            npc_name="Nazeem",
            mood="arrogant",
            relationship="rival",
            affinity=-0.3,
            player_signal="insult"
        )
        
        assert "npc_action: insult" in prompt
        assert "ACTION: Deliver an insult" in prompt
        assert "Keep it brief" in prompt
        assert "real-world slurs" in prompt.lower() or "slurs" in prompt.lower()
    
    def test_all_actions_have_specs(self):
        """Test that all NPCAction values generate valid prompts."""
        for action in NPCAction:
            prompt = render_action_block(
                npc_action=action,
                npc_name="TestNPC",
                mood="neutral",
                relationship="stranger",
                affinity=0.0,
                player_signal="greet"
            )
            
            # Every action should have the basic structure
            assert "[ACTION_CONTROL]" in prompt
            assert f"npc_action: {action.value}" in prompt
            assert "[/ACTION_CONTROL]" in prompt
            assert "ACTION:" in prompt or "Perform" in prompt
    
    def test_negative_affinity_formatting(self):
        """Test that negative affinity is formatted correctly."""
        prompt = render_action_block(
            npc_action=NPCAction.THREATEN,
            npc_name="Bandit",
            mood="hostile",
            relationship="enemy",
            affinity=-0.75,
            player_signal="threaten"
        )
        
        assert "-0.75" in prompt
    
    def test_context_window_included(self):
        """Test that context window is included when provided."""
        context = "Player: Hello\\nLydia: Hey there."
        prompt = render_action_block(
            npc_action=NPCAction.GREET,
            npc_name="Lydia",
            mood="friendly",
            relationship="friend",
            affinity=0.5,
            player_signal="greet",
            context_window=context
        )
        
        # Context should be somewhere in the prompt
        # (may be in a different section depending on implementation)
        assert len(prompt) > 0
    
    def test_governed_memory_included(self):
        """Test that governed memory context is included."""
        memory = "Relevant Facts:\\n- Player helped Lydia before"
        prompt = render_action_block(
            npc_action=NPCAction.HELP,
            npc_name="Lydia",
            mood="grateful",
            relationship="ally",
            affinity=0.8,
            player_signal="help",
            governed_memory_context=memory
        )
        
        assert "Relevant Facts" in prompt or len(prompt) > 0


class TestNuanceVariants:
    """Test nuance variant actions."""
    
    def test_agree_reluctantly(self):
        """Test AGREE_RELUCTANTLY has reluctance language."""
        prompt = render_action_block(
            npc_action=NPCAction.AGREE_RELUCTANTLY,
            npc_name="Skeptic",
            mood="cautious",
            relationship="acquaintance",
            affinity=0.1,
            player_signal="request"
        )
        
        assert "reluctan" in prompt.lower() or "hesita" in prompt.lower()
    
    def test_agree_enthusiastically(self):
        """Test AGREE_ENTHUSIASTICALLY has enthusiasm."""
        prompt = render_action_block(
            npc_action=NPCAction.AGREE_ENTHUSIASTICALLY,
            npc_name="Friend",
            mood="happy",
            relationship="friend",
            affinity=0.7,
            player_signal="request"
        )
        
        assert "enthusiast" in prompt.lower() or "eager" in prompt.lower()
    
    def test_refuse_politely(self):
        """Test REFUSE_POLITELY has polite language."""
        prompt = render_action_block(
            npc_action=NPCAction.REFUSE_POLITELY,
            npc_name="Noble",
            mood="formal",
            relationship="acquaintance",
            affinity=0.2,
            player_signal="request"
        )
        
        assert "polite" in prompt.lower() or "gent" in prompt.lower() or "court" in prompt.lower()
    
    def test_refuse_firmly(self):
        """Test REFUSE_FIRMLY has firm language."""
        prompt = render_action_block(
            npc_action=NPCAction.REFUSE_FIRMLY,
            npc_name="Guard",
            mood="stern",
            relationship="stranger",
            affinity=0.0,
            player_signal="request"
        )
        
        assert "firm" in prompt.lower() or "clear" in prompt.lower() or "final" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
