#!/usr/bin/env python3
"""
Memory Consolidator
Uses LLM to extract permanent facts from conversation history.
"""
import logging
import json
import re
from typing import List, Optional
from datetime import datetime

from memory_manager import ConversationTurn, ConversationManager
from memory_governance import MemoryGovernance, GovernedMemory, MemoryType, MemorySource

logger = logging.getLogger(__name__)

class MemoryConsolidator:
    """
    Consolidates raw conversation logs into semantic memories (facts).
    """
    def __init__(self, 
                 memory_governance: MemoryGovernance,
                 llm_generate_fn = None):
        """
        Args:
            memory_governance: Governance system to store facts
            llm_generate_fn: Function(prompt) -> str to call LLM
        """
        self.governance = memory_governance
        self.generate_fn = llm_generate_fn
        self.consolidation_prompt_template = """
You are a Memory System for an NPC. Analyze the conversation below and extract PERMANENT FACTS about the Player or the World.
Do not summarize the chit-chat. Only extract specific details that should be remembered long-term.

Conversation:
{history}

Instructions:
- Extract facts as a JSON list of strings.
- Example: ["Player's name is Dawson", "Player owns a red car", "Player likes sci-fi"]
- If no facts are found, return []
- OUTPUT RAW JSON ONLY.

JSON Facts:
"""

    def consolidate(self, manager: ConversationManager, recent_n: int = 10) -> int:
        """
        Run consolidation on recent turns.
        Returns number of new facts created.
        """
        if not self.generate_fn:
            logger.warning("Consolidation skipped: No LLM generation function provided.")
            return 0
            
        recent_turns = manager.get_recent(recent_n)
        if not recent_turns:
            return 0
            
        # Format history
        history_text = ""
        for turn in recent_turns:
            history_text += f"Player: {turn.user_input}\nNPC: {turn.npc_response}\n"
            
        prompt = self.consolidation_prompt_template.format(history=history_text)
        
        try:
            logger.info("Running memory consolidation...")
            response = self.generate_fn(prompt)
            
            # Extract JSON
            facts = self._parse_json(response)
            
            added_count = 0
            for fact in facts:
                # Create governed memory
                mem = GovernedMemory(
                    memory_id="", # Auto-hash
                    memory_type=MemoryType.FACT_CLAIM,
                    source=MemorySource.LEARNER_INFERENCE,
                    content=fact,
                    confidence=0.8,
                    timestamp=datetime.utcnow(),
                    metadata={"consolidation_source": "llm_summary"}
                )
                
                success, reason, _ = self.governance.add_memory(mem)
                if success:
                    added_count += 1
                    logger.info(f"Consolidated Fact: {fact}")
            
            return added_count
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return 0

    def _parse_json(self, text: str) -> List[str]:
        """Robust JSON extraction from LLM output"""
        try:
            # Try direct parse
            return json.loads(text)
        except:
            pass
            
        try:
            # Find list pattern
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
            
        return []
