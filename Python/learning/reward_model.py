"""
Reward Model: Computes scalar reward from observable signals
"""
from .schemas import RewardSignals
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


class RewardModel:
    """
    Computes reward signal from observable turn outcomes.
    Uses weighted sum of signals plus ground-truth anchors.
    """
    
    # Reward weights for each signal
    CONTRADICTION_PENALTY = -1.0
    USER_CORRECTION_PENALTY = -0.7
    TTS_OVERRUN_PENALTY = -0.2
    CONVERSATION_CONTINUED_REWARD = +0.4
    FOLLOW_UP_QUESTION_REWARD = +0.2
    
    # Ground-truth anchors (deterministic, high-confidence signals)
    ANCHOR_THATS_WRONG = -0.5         # "that's wrong", "no that's incorrect"
    ANCHOR_REPEAT_QUESTION = -0.3     # User repeats same question
    ANCHOR_CONTINUES_DETAIL = +0.3    # User continues with detail
    ANCHOR_IMMEDIATE_DISENGAGE = -0.2 # User immediately leaves/ends
    ANCHOR_EXPLICIT_THANKS = +0.4     # User explicitly thanks NPC
    
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
    
    def compute_ground_truth_anchor(
        self,
        user_text: str,
        previous_user_text: str = "",
        response_time_ms: float = 0.0
    ) -> Tuple[float, str]:
        """
        Compute ground-truth reward anchor from deterministic patterns.
        
        These anchors are high-confidence signals that override normal rewards.
        
        Args:
            user_text: Current user input
            previous_user_text: Previous user input (for repeat detection)
            response_time_ms: How quickly user responded
            
        Returns:
            (anchor_reward, anchor_type) or (0.0, "none")
        """
        user_lower = user_text.lower().strip()
        
        # Strong negative: explicit correction
        strong_negative_patterns = [
            r"\bthat'?s?\s*(not\s+)?wrong\b",
            r"\bthat'?s?\s*incorrect\b",
            r"\bno,?\s*that'?s?\s*not\b",
            r"\byou'?re\s*mistaken\b",
            r"\bwrong\s*answer\b"
        ]
        for pattern in strong_negative_patterns:
            if re.search(pattern, user_lower):
                logger.info(f"Ground-truth anchor: THATS_WRONG ({self.ANCHOR_THATS_WRONG})")
                return self.ANCHOR_THATS_WRONG, "thats_wrong"
        
        # Repeat question detection
        if previous_user_text:
            prev_lower = previous_user_text.lower().strip()
            # Simple similarity: if sufficient word overlap, likely repeat
            # Filter out very common stop words to focus on content
            stop_words = {"the", "a", "an", "is", "are", "do", "does", "can", "could", "would"}
            curr_words = set(w for w in user_lower.split() if w not in stop_words)
            prev_words = set(w for w in prev_lower.split() if w not in stop_words)
            
            if len(curr_words) >= 2 and len(prev_words) >= 2:
                overlap = len(curr_words & prev_words)
                # Jaccard index
                similarity = overlap / len(curr_words | prev_words)
                
                # Threshold tuned for "where is blacksmith" vs "where find blacksmith"
                if similarity >= 0.4:
                    logger.info(f"Ground-truth anchor: REPEAT_QUESTION ({self.ANCHOR_REPEAT_QUESTION})")
                    return self.ANCHOR_REPEAT_QUESTION, "repeat_question"
        
        # Immediate disengage (very fast response with terminating content)
        disengage_patterns = [
            r"\b(bye|goodbye|leave|exit|quit|nevermind|never\s*mind)\b"
        ]
        if response_time_ms > 0 and response_time_ms < 1500:
            for pattern in disengage_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"Ground-truth anchor: IMMEDIATE_DISENGAGE ({self.ANCHOR_IMMEDIATE_DISENGAGE})")
                    return self.ANCHOR_IMMEDIATE_DISENGAGE, "immediate_disengage"
        
        # Explicit thanks (positive)
        thanks_patterns = [
            r"\bthank(s|\s*you)\b",
            r"\bthat'?s?\s*(very\s+)?helpful\b",
            r"\bgreat,?\s*(thanks|thank)\b",
            r"\bperfect\b"
        ]
        for pattern in thanks_patterns:
            if re.search(pattern, user_lower):
                logger.info(f"Ground-truth anchor: EXPLICIT_THANKS ({self.ANCHOR_EXPLICIT_THANKS})")
                return self.ANCHOR_EXPLICIT_THANKS, "explicit_thanks"
        
        # User continues with detail (positive)
        if len(user_lower.split()) > 10 and "?" not in user_text:
            logger.info(f"Ground-truth anchor: CONTINUES_DETAIL ({self.ANCHOR_CONTINUES_DETAIL})")
            return self.ANCHOR_CONTINUES_DETAIL, "continues_detail"
        
        return 0.0, "none"
    
    @staticmethod
    def detect_user_correction(user_text: str) -> bool:
        """
        Detect if user is correcting the NPC.
        
        Args:
            user_text: User's input text
            
        Returns:
            True if correction detected
        """
        correction_phrases = [
            "no", "wrong", "not true", "that's not what i said",
            "incorrect", "you're mistaken", "actually", "false",
            "that's not right", "i didn't say that"
        ]
        user_lower = user_text.lower()
        return any(phrase in user_lower for phrase in correction_phrases)
    
    @staticmethod
    def detect_follow_up_question(user_text: str) -> bool:
        """
        Detect if user asked a follow-up question.
        
        Args:
            user_text: User's input text
            
        Returns:
            True if question detected
        """
        question_markers = ["?", "what", "why", "how", "when", "where", "who"]
        user_lower = user_text.lower()
        return any(marker in user_lower for marker in question_markers)

