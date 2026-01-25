"""
Mode Bandit for Two-Stage Policy (v1.3 Learning Depth).

Separate bandit for phrasing style selection.
Stage A: Action bandit decides WHAT to do (NPCAction)
Stage B: Mode bandit decides HOW to say it (phrasing style)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class PhrasingMode(Enum):
    """Phrasing styles for NPC responses."""
    TERSE_DIRECT = "terse_direct"        # Short, factual (3-4 sentences)
    WARM_FRIENDLY = "warm_friendly"       # Empathetic, relational
    LORE_RICH = "lore_rich"              # Detailed world-building
    PLAYFUL_WITTY = "playful_witty"      # Humorous, light-hearted
    FORMAL_RESPECTFUL = "formal_respectful"  # Distant, proper
    NEUTRAL_BALANCED = "neutral_balanced"  # Default balanced


@dataclass
class ModeContext:
    """
    Context for mode selection.
    Focused on dialogue style preferences, not action context.
    """
    relationship: str       # stranger/acquaintance/friend/ally
    npc_personality: str    # guard/merchant/companion/noble
    emotional_intensity: str  # low/medium/high
    last_mode_used: Optional[str]
    correction_rate: float  # Recent correction rate (0-1)
    
    @classmethod
    def from_state(
        cls,
        affinity: float,
        npc_role: str,
        emotional_intensity: float,
        last_mode: Optional[PhrasingMode],
        correction_rate: float = 0.0
    ) -> "ModeContext":
        """Create context from state."""
        # Relationship bucket
        if affinity <= -0.3:
            relationship = "stranger"
        elif affinity <= 0.2:
            relationship = "acquaintance"
        elif affinity <= 0.6:
            relationship = "friend"
        else:
            relationship = "ally"
        
        # Personality bucket (simplified)
        role_lower = npc_role.lower()
        if any(g in role_lower for g in ["guard", "soldier", "warrior"]):
            personality = "guard"
        elif any(m in role_lower for m in ["merchant", "trader", "shopkeeper"]):
            personality = "merchant"
        elif any(c in role_lower for c in ["companion", "follower", "friend"]):
            personality = "companion"
        elif any(n in role_lower for n in ["noble", "jarl", "lord", "lady"]):
            personality = "noble"
        else:
            personality = "generic"
        
        # Intensity bucket
        if emotional_intensity <= 0.3:
            intensity = "low"
        elif emotional_intensity <= 0.7:
            intensity = "medium"
        else:
            intensity = "high"
        
        return cls(
            relationship=relationship,
            npc_personality=personality,
            emotional_intensity=intensity,
            last_mode_used=last_mode.value if last_mode else None,
            correction_rate=correction_rate
        )
    
    def to_key(self) -> str:
        """Convert to bucket key."""
        return f"{self.relationship}:{self.npc_personality}:{self.emotional_intensity}"


class ModeBandit:
    """
    Thompson Sampling bandit for phrasing mode selection.
    
    Learns which PhrasingMode works best per context.
    Uses separate reward function from action bandit:
    - Helpfulness (user engagement)
    - Verbosity match (not too long, not too short)
    - Correction rate (lower is better)
    """
    
    EXPLORE_RATE = 0.15  # 15% exploration
    
    # Mode constraints per personality
    PERSONALITY_MODE_PRIORS = {
        "guard": {PhrasingMode.TERSE_DIRECT: 0.3, PhrasingMode.FORMAL_RESPECTFUL: 0.2},
        "merchant": {PhrasingMode.WARM_FRIENDLY: 0.3, PhrasingMode.PLAYFUL_WITTY: 0.2},
        "companion": {PhrasingMode.WARM_FRIENDLY: 0.3, PhrasingMode.PLAYFUL_WITTY: 0.2},
        "noble": {PhrasingMode.FORMAL_RESPECTFUL: 0.3, PhrasingMode.LORE_RICH: 0.2},
        "generic": {}
    }
    
    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path("data/learning/mode_bandit.json")
        
        # Arms: key -> mode -> {alpha, beta, n}
        self._arms: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.load()
    
    def _get_arm(self, key: str, mode: PhrasingMode) -> Dict[str, float]:
        """Get or create arm."""
        if key not in self._arms:
            self._arms[key] = {}
        
        if mode.value not in self._arms[key]:
            # Prior from personality
            self._arms[key][mode.value] = {"alpha": 1.0, "beta": 1.0, "n": 0.0}
        
        return self._arms[key][mode.value]
    
    def select(
        self,
        context: ModeContext,
        candidates: Optional[List[PhrasingMode]] = None
    ) -> PhrasingMode:
        """
        Select phrasing mode for context.
        
        Args:
            context: Mode selection context
            candidates: Available modes (default: all)
            
        Returns:
            Selected PhrasingMode
        """
        if candidates is None:
            candidates = list(PhrasingMode)
        
        key = context.to_key()
        
        # Exploration
        if random.random() < self.EXPLORE_RATE:
            return random.choice(candidates)
        
        # Thompson sampling
        best_mode = candidates[0]
        best_sample = -1.0
        
        for mode in candidates:
            arm = self._get_arm(key, mode)
            sample = random.betavariate(arm["alpha"], arm["beta"])
            
            # Add personality prior
            priors = self.PERSONALITY_MODE_PRIORS.get(context.npc_personality, {})
            prior_bump = priors.get(mode, 0.0)
            
            # Penalty if same mode used twice in a row
            repeat_penalty = 0.0
            if context.last_mode_used == mode.value:
                repeat_penalty = -0.1
            
            score = sample + prior_bump + repeat_penalty
            
            if score > best_sample:
                best_sample = score
                best_mode = mode
        
        return best_mode
    
    def update(
        self,
        context: ModeContext,
        mode: PhrasingMode,
        reward: float
    ) -> None:
        """
        Update mode arm with observed reward.
        
        Reward should reflect:
        - Helpfulness (engagement)
        - Verbosity match
        - Low correction rate
        """
        key = context.to_key()
        arm = self._get_arm(key, mode)
        
        r = max(0.0, min(1.0, reward))
        arm["alpha"] += r
        arm["beta"] += (1.0 - r)
        arm["n"] += 1.0
    
    def get_mode_instructions(self, mode: PhrasingMode) -> str:
        """Get prompt instructions for phrasing mode."""
        instructions = {
            PhrasingMode.TERSE_DIRECT: (
                "Style: TERSE. Be brief and direct. 1-2 sentences max. "
                "No filler words. State facts plainly."
            ),
            PhrasingMode.WARM_FRIENDLY: (
                "Style: WARM. Be empathetic and relational. Show care. "
                "Use the player's name if known. Express genuine interest."
            ),
            PhrasingMode.LORE_RICH: (
                "Style: LORE. Include world-building details. Reference history, "
                "culture, or traditions. Make the world feel alive."
            ),
            PhrasingMode.PLAYFUL_WITTY: (
                "Style: PLAYFUL. Be light-hearted with subtle humor. "
                "Include a clever observation or gentle teasing if appropriate."
            ),
            PhrasingMode.FORMAL_RESPECTFUL: (
                "Style: FORMAL. Maintain distance and proper address. "
                "Use titles. Be courteous but not warm."
            ),
            PhrasingMode.NEUTRAL_BALANCED: (
                "Style: BALANCED. Natural conversation. Neither too formal "
                "nor too casual. Match the player's energy."
            )
        }
        return instructions.get(mode, "")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        stats = {}
        for key, modes in self._arms.items():
            stats[key] = {}
            for mode_name, arm in modes.items():
                mean = arm["alpha"] / (arm["alpha"] + arm["beta"])
                stats[key][mode_name] = {
                    "trials": arm["n"],
                    "mean": round(mean, 3)
                }
        return stats
    
    def save(self) -> None:
        """Persist bandit state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {"arms": self._arms}
        
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        import os
        os.replace(tmp_path, self.path)
    
    def load(self) -> None:
        """Load bandit state."""
        if not self.path.exists():
            return
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._arms = data.get("arms", {})
        except Exception as e:
            print(f"Error loading mode bandit: {e}")
            self._arms = {}
