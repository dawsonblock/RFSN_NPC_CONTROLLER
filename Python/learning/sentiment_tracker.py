"""
Player Sentiment Tracker for RFSN NPC Controller.

Tracks player sentiment over time for trend detection.
Enables NPCs to notice "player is warming up to me" patterns.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque
import time


@dataclass
class SentimentEntry:
    """A single sentiment observation."""
    value: float  # -1 to +1
    timestamp: float = field(default_factory=time.time)


class PlayerSentimentTracker:
    """
    Track player sentiment over time for trend analysis.
    
    Features:
    - Rolling window of recent sentiment values
    - Trend detection (improving/declining)
    - Volatility measurement
    - Per-NPC tracking
    """
    
    def __init__(self, window_size: int = 10, decay_hours: float = 2.0):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of observations to keep
            decay_hours: Hours after which old observations are discarded
        """
        self.window_size = window_size
        self.decay_seconds = decay_hours * 3600
        self.history: Deque[SentimentEntry] = deque(maxlen=window_size)
    
    def add(self, sentiment: float) -> None:
        """
        Add a sentiment observation.
        
        Args:
            sentiment: Sentiment value (-1 to +1)
        """
        # Clamp to valid range
        sentiment = max(-1.0, min(1.0, sentiment))
        self.history.append(SentimentEntry(value=sentiment))
        self._prune_old()
    
    def _prune_old(self) -> None:
        """Remove observations older than decay threshold."""
        now = time.time()
        cutoff = now - self.decay_seconds
        
        # Remove from front while too old
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()
    
    @property
    def current(self) -> float:
        """Get most recent sentiment value."""
        self._prune_old()
        if not self.history:
            return 0.0
        return self.history[-1].value
    
    @property
    def average(self) -> float:
        """Get average sentiment over window."""
        self._prune_old()
        if not self.history:
            return 0.0
        return sum(e.value for e in self.history) / len(self.history)
    
    @property
    def trend(self) -> float:
        """
        Calculate sentiment trend.
        
        Returns:
            Positive if improving, negative if declining, ~0 if stable
        """
        self._prune_old()
        if len(self.history) < 3:
            return 0.0
        
        # Compare first half average to second half average
        mid = len(self.history) // 2
        early = list(self.history)[:mid]
        late = list(self.history)[mid:]
        
        early_avg = sum(e.value for e in early) / len(early)
        late_avg = sum(e.value for e in late) / len(late)
        
        return late_avg - early_avg
    
    @property
    def volatility(self) -> float:
        """
        Measure sentiment volatility (standard deviation).
        
        High volatility = player is erratic
        Low volatility = player is consistent
        """
        self._prune_old()
        if len(self.history) < 2:
            return 0.0
        
        avg = self.average
        variance = sum((e.value - avg) ** 2 for e in self.history) / len(self.history)
        return variance ** 0.5
    
    @property
    def is_improving(self) -> bool:
        """Check if player sentiment is trending positive."""
        return self.trend > 0.1
    
    @property
    def is_declining(self) -> bool:
        """Check if player sentiment is trending negative."""
        return self.trend < -0.1
    
    @property
    def is_hostile(self) -> bool:
        """Check if player has been consistently negative."""
        return self.average < -0.3 and len(self.history) >= 3
    
    @property
    def is_friendly(self) -> bool:
        """Check if player has been consistently positive."""
        return self.average > 0.3 and len(self.history) >= 3
    
    def get_insight(self) -> str:
        """Get human-readable insight about player sentiment."""
        self._prune_old()
        
        if len(self.history) < 2:
            return "Not enough data"
        
        avg = self.average
        trend = self.trend
        vol = self.volatility
        
        # Base assessment
        if avg > 0.5:
            base = "Player is very positive"
        elif avg > 0.2:
            base = "Player is somewhat positive"
        elif avg < -0.5:
            base = "Player is very negative"
        elif avg < -0.2:
            base = "Player is somewhat negative"
        else:
            base = "Player is neutral"
        
        # Trend modifier
        if trend > 0.15:
            trend_str = "and improving"
        elif trend < -0.15:
            trend_str = "and declining"
        else:
            trend_str = ""
        
        # Volatility note
        if vol > 0.4:
            vol_str = " (erratic behavior)"
        else:
            vol_str = ""
        
        return f"{base} {trend_str}{vol_str}".strip()
    
    def to_dict(self) -> dict:
        """Serialize for API responses."""
        self._prune_old()
        return {
            "current": round(self.current, 3),
            "average": round(self.average, 3),
            "trend": round(self.trend, 3),
            "volatility": round(self.volatility, 3),
            "observations": len(self.history),
            "insight": self.get_insight(),
            "is_improving": self.is_improving,
            "is_declining": self.is_declining,
            "is_hostile": self.is_hostile,
            "is_friendly": self.is_friendly
        }
    
    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()


class MultiPlayerSentimentTracker:
    """
    Track sentiment across multiple player-NPC pairs.
    
    Organized by player_id -> npc_name -> tracker
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._trackers: dict[str, dict[str, PlayerSentimentTracker]] = {}
    
    def get_tracker(self, player_id: str, npc_name: str) -> PlayerSentimentTracker:
        """Get or create tracker for player-NPC pair."""
        if player_id not in self._trackers:
            self._trackers[player_id] = {}
        
        if npc_name not in self._trackers[player_id]:
            self._trackers[player_id][npc_name] = PlayerSentimentTracker(
                window_size=self.window_size
            )
        
        return self._trackers[player_id][npc_name]
    
    def add(self, player_id: str, npc_name: str, sentiment: float) -> None:
        """Add sentiment observation."""
        tracker = self.get_tracker(player_id, npc_name)
        tracker.add(sentiment)
    
    def get_trend(self, player_id: str, npc_name: str) -> float:
        """Get sentiment trend for player-NPC pair."""
        tracker = self.get_tracker(player_id, npc_name)
        return tracker.trend
    
    def get_all_for_player(self, player_id: str) -> dict[str, dict]:
        """Get all NPC sentiment stats for a player."""
        if player_id not in self._trackers:
            return {}
        
        return {
            npc_name: tracker.to_dict()
            for npc_name, tracker in self._trackers[player_id].items()
        }
    
    def list_players(self) -> list[str]:
        """List all tracked players."""
        return list(self._trackers.keys())


# Singleton instance for default NPC tracking (player_id = "default")
_sentiment_tracker: Optional[MultiPlayerSentimentTracker] = None


def get_sentiment_tracker() -> MultiPlayerSentimentTracker:
    """Get the global sentiment tracker."""
    global _sentiment_tracker
    if _sentiment_tracker is None:
        _sentiment_tracker = MultiPlayerSentimentTracker()
    return _sentiment_tracker
