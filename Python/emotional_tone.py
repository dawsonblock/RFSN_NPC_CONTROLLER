"""
Emotional Tone Module for RFSN NPC Controller.

Provides emotional context injection for LLM prompts and manages
NPC emotional state transitions based on interactions.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple
import math


class EmotionalTone(Enum):
    """Primary emotional tones for LLM response generation."""
    NEUTRAL = "neutral"
    WARM = "warm"
    COLD = "cold"
    ENTHUSIASTIC = "enthusiastic"
    RESERVED = "reserved"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    SYMPATHETIC = "sympathetic"
    SUSPICIOUS = "suspicious"
    PLAYFUL = "playful"


@dataclass
class EmotionalState:
    """
    Complete emotional state of an NPC.
    
    Uses a dimensional model with:
    - valence: negative (-1) to positive (+1)
    - arousal: calm (0) to excited (1)
    - dominance: submissive (0) to dominant (1)
    
    Plus discrete primary emotion for clarity.
    """
    valence: float = 0.0  # -1 to +1 (sad/happy)
    arousal: float = 0.5  # 0 to 1 (calm/excited)
    dominance: float = 0.5  # 0 to 1 (submissive/dominant)
    
    primary_tone: EmotionalTone = EmotionalTone.NEUTRAL
    intensity: float = 0.5  # 0 to 1
    
    # Recent emotional events for continuity
    recent_triggers: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._clamp_values()
    
    def _clamp_values(self):
        """Ensure values are in valid ranges."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.intensity = max(0.0, min(1.0, self.intensity))
    
    def apply_stimulus(
        self,
        valence_delta: float = 0.0,
        arousal_delta: float = 0.0,
        dominance_delta: float = 0.0,
        trigger: Optional[str] = None,
        decay_factor: float = 0.9
    ):
        """
        Apply an emotional stimulus with natural decay.
        
        Args:
            valence_delta: Change in valence
            arousal_delta: Change in arousal  
            dominance_delta: Change in dominance
            trigger: Description of what caused this emotion
            decay_factor: How much previous state affects new state
        """
        # Apply with momentum (emotions don't change instantly)
        self.valence = self.valence * decay_factor + valence_delta * (1 - decay_factor)
        self.arousal = self.arousal * decay_factor + arousal_delta * (1 - decay_factor)
        self.dominance = self.dominance * decay_factor + dominance_delta * (1 - decay_factor)
        
        # Update intensity based on how far from neutral
        self.intensity = math.sqrt(self.valence**2 + (self.arousal - 0.5)**2)
        
        # Track trigger
        if trigger:
            self.recent_triggers.append(trigger)
            self.recent_triggers = self.recent_triggers[-5:]  # Keep last 5
        
        self._clamp_values()
        self._update_primary_tone()
    
    def _update_primary_tone(self):
        """Derive primary tone from dimensional values."""
        # High valence + high arousal = enthusiastic
        if self.valence > 0.3 and self.arousal > 0.6:
            self.primary_tone = EmotionalTone.ENTHUSIASTIC
        # High valence + low arousal = warm
        elif self.valence > 0.3 and self.arousal <= 0.6:
            self.primary_tone = EmotionalTone.WARM
        # Low valence + high arousal + high dominance = aggressive
        elif self.valence < -0.3 and self.arousal > 0.6 and self.dominance > 0.5:
            self.primary_tone = EmotionalTone.AGGRESSIVE
        # Low valence + high arousal + low dominance = defensive
        elif self.valence < -0.3 and self.arousal > 0.6 and self.dominance <= 0.5:
            self.primary_tone = EmotionalTone.DEFENSIVE
        # Low valence + low arousal = cold
        elif self.valence < -0.3 and self.arousal <= 0.4:
            self.primary_tone = EmotionalTone.COLD
        # Moderate valence + low dominance = reserved
        elif self.dominance < 0.3:
            self.primary_tone = EmotionalTone.RESERVED
        # Moderate negative + moderate arousal = suspicious
        elif -0.3 <= self.valence < 0 and 0.4 < self.arousal < 0.7:
            self.primary_tone = EmotionalTone.SUSPICIOUS
        # Positive + playful cues
        elif self.valence > 0.2 and 0.5 < self.arousal < 0.8:
            self.primary_tone = EmotionalTone.PLAYFUL
        # Positive + caring response to negative situation
        elif self.valence > 0 and self.arousal < 0.5:
            self.primary_tone = EmotionalTone.SYMPATHETIC
        else:
            self.primary_tone = EmotionalTone.NEUTRAL
    
    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "primary_tone": self.primary_tone.value,
            "intensity": round(self.intensity, 3),
            "recent_triggers": self.recent_triggers[-3:]
        }


# Emotion stimulus mappings for common situations
EMOTION_STIMULI = {
    # Player actions
    "compliment": {"valence_delta": 0.15, "arousal_delta": 0.05, "trigger": "received compliment"},
    "insult": {"valence_delta": -0.2, "arousal_delta": 0.15, "dominance_delta": -0.1, "trigger": "was insulted"},
    "threat": {"valence_delta": -0.3, "arousal_delta": 0.25, "trigger": "was threatened"},
    "help_offered": {"valence_delta": 0.1, "arousal_delta": 0.05, "trigger": "player offered help"},
    "gift": {"valence_delta": 0.2, "arousal_delta": 0.1, "trigger": "received gift"},
    "agreement": {"valence_delta": 0.1, "arousal_delta": 0.0, "trigger": "player agreed"},
    "disagreement": {"valence_delta": -0.1, "arousal_delta": 0.05, "trigger": "player disagreed"},
    "question": {"valence_delta": 0.0, "arousal_delta": 0.05, "trigger": "asked question"},
    "greeting": {"valence_delta": 0.05, "arousal_delta": 0.0, "trigger": "greeted"},
    "farewell": {"valence_delta": 0.0, "arousal_delta": -0.1, "trigger": "said goodbye"},
    
    # NPC internal states
    "task_success": {"valence_delta": 0.15, "arousal_delta": 0.1, "dominance_delta": 0.1, "trigger": "task succeeded"},
    "task_failure": {"valence_delta": -0.15, "arousal_delta": 0.1, "dominance_delta": -0.1, "trigger": "task failed"},
    "danger_nearby": {"valence_delta": -0.1, "arousal_delta": 0.3, "trigger": "sensed danger"},
    "safe_environment": {"valence_delta": 0.05, "arousal_delta": -0.15, "trigger": "feels safe"},
}


def get_emotional_prompt_injection(
    emotional_state: EmotionalState,
    npc_name: str
) -> str:
    """
    Generate emotional tone instructions for LLM prompt.
    
    This tells the LLM how to express the NPC's current emotional state.
    """
    tone = emotional_state.primary_tone
    intensity = emotional_state.intensity
    valence = emotional_state.valence
    
    # Build tone description
    tone_descriptions = {
        EmotionalTone.NEUTRAL: "Speak in a balanced, even-tempered manner.",
        EmotionalTone.WARM: "Speak warmly and kindly. Show genuine interest and care.",
        EmotionalTone.COLD: "Speak curtly and distantly. Keep responses brief and impersonal.",
        EmotionalTone.ENTHUSIASTIC: "Speak with energy and excitement! Show passion and eagerness.",
        EmotionalTone.RESERVED: "Speak cautiously and measured. Hold back, don't reveal too much.",
        EmotionalTone.AGGRESSIVE: "Speak forcefully and confrontationally. Challenge and assert.",
        EmotionalTone.DEFENSIVE: "Speak guardedly, protecting yourself. Be wary and cautious.",
        EmotionalTone.SYMPATHETIC: "Speak with compassion and understanding. Show empathy.",
        EmotionalTone.SUSPICIOUS: "Speak with doubt and skepticism. Question motives.",
        EmotionalTone.PLAYFUL: "Speak with wit and levity. Be light-hearted and teasing.",
    }
    
    base_instruction = tone_descriptions.get(tone, tone_descriptions[EmotionalTone.NEUTRAL])
    
    # Add intensity modifier
    if intensity > 0.7:
        intensity_mod = "Express these emotions strongly. "
    elif intensity > 0.4:
        intensity_mod = ""
    else:
        intensity_mod = "Keep emotional expression subtle. "
    
    # Add valence hint
    if valence > 0.3:
        valence_hint = "Your overall mood is positive."
    elif valence < -0.3:
        valence_hint = "Your overall mood is negative."
    else:
        valence_hint = ""
    
    # Build emotional context block
    block = f"""
[EMOTIONAL_CONTEXT]
tone: {tone.value}
{base_instruction}
{intensity_mod}{valence_hint}
[/EMOTIONAL_CONTEXT]
""".strip()
    
    return block


def get_emotion_from_action(action_name: str) -> Tuple[float, float, float]:
    """
    Map NPC action to emotional response (for state update).
    
    Returns: (valence_delta, arousal_delta, dominance_delta)
    """
    action_emotions = {
        # Positive actions
        "greet": (0.05, 0.0, 0.0),
        "agree": (0.1, 0.0, 0.05),
        "agree_enthusiastically": (0.15, 0.1, 0.05),
        "agree_reluctantly": (0.0, -0.05, -0.05),
        "help": (0.1, 0.05, 0.1),
        "help_eagerly": (0.15, 0.1, 0.1),
        "help_grudgingly": (0.0, 0.0, 0.05),
        "compliment": (0.15, 0.05, 0.1),
        "offer": (0.05, 0.05, 0.05),
        "accept": (0.1, 0.0, 0.0),
        "confide": (0.1, 0.0, -0.1),
        
        # Neutral actions
        "inquire": (0.0, 0.05, 0.0),
        "probe": (0.0, 0.1, 0.1),
        "explain": (0.0, 0.0, 0.05),
        "hint": (0.0, 0.0, 0.0),
        "deflect": (-0.05, 0.0, 0.0),
        "farewell": (0.0, -0.1, 0.0),
        "ignore": (-0.1, -0.1, 0.1),
        
        # Negative actions
        "disagree": (-0.05, 0.05, 0.1),
        "refuse": (-0.1, 0.05, 0.1),
        "refuse_politely": (-0.05, 0.0, 0.05),
        "refuse_firmly": (-0.1, 0.1, 0.15),
        "warn_sternly": (-0.1, 0.15, 0.2),
        "threaten": (-0.2, 0.25, 0.25),
        "insult": (-0.15, 0.2, 0.15),
        "attack": (-0.25, 0.3, 0.25),
        "apologize": (0.05, -0.05, -0.1),
        
        # Special actions
        "defend": (-0.1, 0.2, 0.0),
        "flee": (-0.2, 0.3, -0.3),
        "betray": (-0.3, 0.2, 0.1),
        "comply_with_hesitation": (0.0, 0.05, -0.1),
    }
    
    action_lower = action_name.lower().replace(" ", "_")
    return action_emotions.get(action_lower, (0.0, 0.0, 0.0))


class EmotionalStateManager:
    """
    Manages emotional states across NPCs.
    
    Persists emotional state between interactions for continuity.
    """
    
    def __init__(self):
        self._states: Dict[str, EmotionalState] = {}
    
    def get_state(self, npc_name: str) -> EmotionalState:
        """Get or create emotional state for NPC."""
        if npc_name not in self._states:
            self._states[npc_name] = EmotionalState()
        return self._states[npc_name]
    
    def update_from_action(
        self,
        npc_name: str,
        action_name: str,
        player_sentiment: float = 0.0
    ):
        """
        Update NPC emotional state based on action taken.
        
        Args:
            npc_name: NPC identifier
            action_name: The action taken (e.g., "AGREE", "THREATEN")
            player_sentiment: Player's recent sentiment (-1 to 1)
        """
        state = self.get_state(npc_name)
        
        # Get emotion deltas from action
        v_delta, a_delta, d_delta = get_emotion_from_action(action_name)
        
        # Modulate by player sentiment (responses to negative players are stronger)
        if player_sentiment < -0.2:
            a_delta *= 1.2  # More aroused when player is negative
        elif player_sentiment > 0.2:
            v_delta *= 1.1  # More positive when player is positive
        
        state.apply_stimulus(
            valence_delta=v_delta,
            arousal_delta=a_delta,
            dominance_delta=d_delta,
            trigger=f"performed {action_name}"
        )
    
    def update_from_player_action(
        self,
        npc_name: str,
        player_action: str
    ):
        """
        Update NPC emotional state based on player action.
        
        Args:
            npc_name: NPC identifier
            player_action: What the player did (e.g., "insult", "compliment")
        """
        state = self.get_state(npc_name)
        
        if player_action in EMOTION_STIMULI:
            stimulus = EMOTION_STIMULI[player_action]
            state.apply_stimulus(**stimulus)
    
    def get_prompt_injection(self, npc_name: str) -> str:
        """Get emotional prompt injection for NPC."""
        state = self.get_state(npc_name)
        return get_emotional_prompt_injection(state, npc_name)
    
    def decay_all(self, decay_factor: float = 0.95):
        """
        Apply decay to all emotional states.
        
        Call this periodically to let emotions naturally subside.
        """
        for state in self._states.values():
            # Decay toward neutral
            state.apply_stimulus(
                valence_delta=-state.valence * (1 - decay_factor),
                arousal_delta=(0.5 - state.arousal) * (1 - decay_factor),
                dominance_delta=(0.5 - state.dominance) * (1 - decay_factor),
                decay_factor=1.0  # Don't double-decay
            )
    
    def list_states(self) -> Dict[str, dict]:
        """List all NPC emotional states."""
        return {
            name: state.to_dict()
            for name, state in self._states.items()
        }


# Singleton instance
_emotion_manager: Optional[EmotionalStateManager] = None

def get_emotion_manager() -> EmotionalStateManager:
    """Get the global emotion manager instance."""
    global _emotion_manager
    if _emotion_manager is None:
        _emotion_manager = EmotionalStateManager()
    return _emotion_manager
