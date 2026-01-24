"""
Learning Event Logger for RFSN NPC Controller.

Logs learning events (state, action, reward) to CSV for analysis and debugging.
Enables post-hoc hyperparameter tuning and learning visualization.
"""
import csv
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    """A single learning event for logging."""
    timestamp: float
    npc_name: str
    action: str
    reward: float
    bandit_key: str
    temporal_bias: float
    emotion_tone: str
    affinity_before: float
    affinity_after: float
    player_signal: str
    mood: str
    
    def to_row(self) -> list:
        """Convert to CSV row."""
        return [
            self.timestamp,
            self.npc_name,
            self.action,
            round(self.reward, 4),
            self.bandit_key,
            round(self.temporal_bias, 4),
            self.emotion_tone,
            round(self.affinity_before, 4),
            round(self.affinity_after, 4),
            self.player_signal,
            self.mood
        ]


class LearningEventLogger:
    """
    Thread-safe CSV logger for learning events.
    
    Enables:
    - Post-hoc analysis of learning performance
    - Hyperparameter tuning insights
    - Action pattern visualization
    - Reward distribution analysis
    """
    
    COLUMNS = [
        "timestamp",
        "npc_name",
        "action",
        "reward",
        "bandit_key",
        "temporal_bias",
        "emotion_tone",
        "affinity_before",
        "affinity_after",
        "player_signal",
        "mood"
    ]
    
    def __init__(self, log_path: Optional[Path] = None, enabled: bool = True):
        """
        Initialize the event logger.
        
        Args:
            log_path: Path to CSV file (default: data/learning/events.csv)
            enabled: Whether logging is active
        """
        self.log_path = log_path or Path("data/learning/events.csv")
        self.enabled = enabled
        self._lock = threading.Lock()
        self._event_count = 0
        
        # Ensure directory exists
        if self.enabled:
            self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create log file with headers if it doesn't exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)
            logger.info(f"[LearningEventLogger] Created {self.log_path}")
    
    def log(
        self,
        npc_name: str,
        action: str,
        reward: float,
        bandit_key: str = "",
        temporal_bias: float = 0.0,
        emotion_tone: str = "neutral",
        affinity_before: float = 0.0,
        affinity_after: float = 0.0,
        player_signal: str = "",
        mood: str = "neutral"
    ) -> None:
        """
        Log a learning event.
        
        Thread-safe append to CSV file.
        """
        if not self.enabled:
            return
        
        event = LearningEvent(
            timestamp=time.time(),
            npc_name=npc_name,
            action=action,
            reward=reward,
            bandit_key=bandit_key,
            temporal_bias=temporal_bias,
            emotion_tone=emotion_tone,
            affinity_before=affinity_before,
            affinity_after=affinity_after,
            player_signal=player_signal,
            mood=mood
        )
        
        with self._lock:
            try:
                with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(event.to_row())
                self._event_count += 1
                
                # Log every 100 events
                if self._event_count % 100 == 0:
                    logger.info(f"[LearningEventLogger] {self._event_count} events logged")
                    
            except Exception as e:
                logger.error(f"[LearningEventLogger] Failed to log event: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged events."""
        if not self.log_path.exists():
            return {"total_events": 0, "file_size_kb": 0}
        
        file_size = self.log_path.stat().st_size / 1024
        
        # Count lines (excluding header)
        with open(self.log_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f) - 1
        
        return {
            "total_events": max(0, line_count),
            "file_size_kb": round(file_size, 2),
            "session_events": self._event_count
        }
    
    def rotate(self, max_size_mb: float = 10.0) -> bool:
        """
        Rotate log file if it exceeds max size.
        
        Args:
            max_size_mb: Maximum file size before rotation
            
        Returns:
            True if rotation occurred
        """
        if not self.log_path.exists():
            return False
        
        file_size_mb = self.log_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb < max_size_mb:
            return False
        
        # Rotate: rename current to .old, start fresh
        with self._lock:
            old_path = self.log_path.with_suffix(".csv.old")
            
            # Remove old backup if exists
            if old_path.exists():
                old_path.unlink()
            
            # Rename current to .old
            self.log_path.rename(old_path)
            
            # Create fresh file
            self._ensure_file_exists()
            
            logger.info(f"[LearningEventLogger] Rotated log ({file_size_mb:.1f}MB)")
            return True
    
    def clear(self) -> None:
        """Clear all logged events."""
        with self._lock:
            if self.log_path.exists():
                self.log_path.unlink()
            self._ensure_file_exists()
            self._event_count = 0
            logger.info("[LearningEventLogger] Cleared all events")


# Singleton instance
_event_logger: Optional[LearningEventLogger] = None


def get_event_logger() -> LearningEventLogger:
    """Get the global event logger instance."""
    global _event_logger
    if _event_logger is None:
        _event_logger = LearningEventLogger()
    return _event_logger
