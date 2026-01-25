"""
Reward Model: Computes scalar reward from observable signals
"""
from .schemas import RewardSignals
import logging

logger = logging.getLogger(__name__)


class RewardModel:
    """
    Computes reward signal from observable turn outcomes
    Uses simple weighted sum of binary signals
    """
    
    # Reward weights for each signal
    CONTRADICTION_PENALTY = -1.0
    USER_CORRECTION_PENALTY = -0.7
    TTS_OVERRUN_PENALTY = -0.2
    CONVERSATION_CONTINUED_REWARD = +0.4
    FOLLOW_UP_QUESTION_REWARD = +0.2
    
    # Reward bounds
    MIN_REWARD = -2.0
    MAX_REWARD = +2.0
    
    def compute(self, signals: RewardSignals) -> float:
        """
        Compute scalar reward from signals with normalization.
        
        Args:
            signals: RewardSignals with boolean flags
            
        Returns:
            Clamped reward in [-1.0, 1.0]
        """
        # Collect individual components for logging
        components = {}
        
        # Negative signals (normalized to [-1, 0])
        if signals.contradiction_detected:
            components["contradiction"] = -1.0
        
        if signals.user_correction:
            components["user_correction"] = -0.7
        
        if signals.tts_overrun:
            components["tts_overrun"] = -0.2
        
        # Positive signals (normalized to [0, 1])
        if signals.conversation_continued:
            components["conversation_continued"] = 0.4
        
        if signals.follow_up_question:
            components["follow_up_question"] = 0.2
        
        # Sum components
        raw_reward = sum(components.values())
        
        # Normalize by count of active signals to prevent extreme values
        n_active = max(1, len(components))
        normalized_reward = raw_reward / n_active
        
        # Clamp once at end to [-1.0, 1.0]
        final_reward = max(-1.0, min(1.0, normalized_reward))
        
        # Log per-component breakdown for debugging
        if components:
            component_str = ", ".join(f"{k}={v:+.2f}" for k, v in components.items())
            logger.info(f"Reward components: [{component_str}] -> raw={raw_reward:.2f}, norm={normalized_reward:.2f}, final={final_reward:.2f}")
        else:
            logger.info("Reward: no active signals, final=0.00")
        
        return final_reward
    
    @staticmethod
    def detect_user_correction(user_text: str) -> bool:
        """
        Detect if user is correcting the NPC
        
        Args:
            user_text: User's input text
            
        Returns:
            True if correction detected
        """
        correction_phrases = [
            "no", "wrong", "not true", "that's not what i said",
            "incorrect", "you're mistaken", "actually", "false"
        ]
        user_lower = user_text.lower()
        return any(phrase in user_lower for phrase in correction_phrases)
    
    @staticmethod
    def detect_follow_up_question(user_text: str) -> bool:
        """
        Detect if user asked a follow-up question
        
        Args:
            user_text: User's input text
            
        Returns:
            True if question detected
        """
        question_markers = ["?", "what", "why", "how", "when", "where", "who"]
        user_lower = user_text.lower()
        return any(marker in user_lower for marker in question_markers)
