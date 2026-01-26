"""
StateAbstractor: Deterministic state abstraction with versioning.

Maps high-dimensionality raw state to low-dimensionality categorical keys.
Ensures stable, versioned abstraction for reproducible learning.
"""
import hashlib
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger("learning.abstraction")


class StateAbstractor:
    """
    Maps high-dimensionality RawState to low-dimensionality AbstractState buckets.
    
    Key guarantees:
    - Deterministic: same input → same output
    - Versioned: abstraction_version changes when logic changes
    - Stable: small irrelevant changes don't affect keys
    """
    
    VERSION = "2.0"
    SCHEMA_ID = "rfsn_state_v2"
    
    # Default bucket configuration
    DEFAULT_CONFIG = {
        "key_features": ["location", "quest_state", "relationship", "threat_level"],
        "bucket_rules": {
            "affinity": [(-1.0, -0.3, "neg"), (-0.3, 0.3, "neutral"), (0.3, 1.0, "pos")],
            "threat_level": [(0, 0, "none"), (1, 1, "low"), (2, 99, "high")],
            "turn_count": [(0, 2, "early"), (3, 6, "mid"), (7, 999, "late")]
        }
    }
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        key_features: Optional[List[str]] = None
    ):
        # Load config from file or use defaults
        self.config = self._load_config(config_path)
        
        # Allow override of key features
        self.key_features = key_features or self.config.get(
            "key_features", 
            self.DEFAULT_CONFIG["key_features"]
        )
        
        self.bucket_rules = self.config.get(
            "bucket_rules",
            self.DEFAULT_CONFIG["bucket_rules"]
        )
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load config from YAML or use defaults."""
        if config_path and config_path.exists():
            try:
                # Try YAML first
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        return yaml.safe_load(f)
                except ImportError:
                    # Fall back to JSON
                    with open(config_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return self.DEFAULT_CONFIG.copy()
    
    def abstract(self, raw_state: Dict[str, Any]) -> str:
        """
        Convert raw state dictionary into a deterministic abstract key.
        
        Args:
            raw_state: The full state dictionary from the game/world model.
            
        Returns:
            A string identifier (e.g., "LOCATION:tavern|QUEST_STATE:active|...")
            
        Guarantees:
            - Same raw_state → same abstract key (deterministic)
            - Key format is versioned via VERSION/SCHEMA_ID
        """
        segments = []
        
        for feature in sorted(self.key_features):
            val = raw_state.get(feature, "unk")
            
            # Apply bucket rules if available
            if feature in self.bucket_rules:
                val = self._apply_bucket_rule(feature, val)
            else:
                # Normalize value
                val = self._normalize_value(val)
            
            segments.append(f"{feature.upper()}:{val}")
        
        # Create readable, deterministic key
        abstract_key = "|".join(segments)
        
        return abstract_key
    
    def _apply_bucket_rule(self, feature: str, value: Any) -> str:
        """Apply bucket rule to convert numeric values to categories."""
        rules = self.bucket_rules.get(feature, [])
        
        # Try to get numeric value
        try:
            if isinstance(value, str):
                numeric_val = float(value) if '.' in value else int(value)
            else:
                numeric_val = float(value)
        except (ValueError, TypeError):
            return self._normalize_value(value)
        
        # Find matching bucket
        for rule in rules:
            if len(rule) == 3:
                low, high, label = rule
                if low <= numeric_val <= high:
                    return label
        
        return self._normalize_value(value)
    
    def _normalize_value(self, val: Any) -> str:
        """Normalize a value to a consistent string representation."""
        if val is None:
            return "unk"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return val.lower().strip()
        return str(val).lower().strip()
    
    def get_context_id(self, raw_state: Dict[str, Any]) -> str:
        """
        Returns a hashed version of the abstract state for efficient storage keys.
        """
        abstract_key = self.abstract(raw_state)
        return hashlib.sha256(abstract_key.encode()).hexdigest()[:16]
    
    def compute_raw_hash(self, raw_state: Dict[str, Any]) -> str:
        """Compute stable hash of full raw state for reproducibility."""
        canonical = json.dumps(raw_state, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def bucket_affinity(self, affinity: float) -> str:
        """Bucket affinity value to category."""
        if affinity < -0.3:
            return "neg"
        elif affinity > 0.3:
            return "pos"
        return "neutral"
    
    def bucket_mood(self, mood: str) -> str:
        """Bucket mood to simplified category."""
        mood_lower = mood.lower() if mood else "neutral"
        
        hostile = ["angry", "hostile", "aggressive", "furious", "enraged"]
        friendly = ["happy", "friendly", "warm", "welcoming", "pleased"]
        
        if mood_lower in hostile:
            return "hostile"
        elif mood_lower in friendly:
            return "friendly"
        return "neutral"
    
    def bucket_turn_count(self, turn_count: int) -> str:
        """Bucket turn count to conversation phase."""
        if turn_count <= 2:
            return "early"
        elif turn_count <= 6:
            return "mid"
        return "late"
    
    def get_versioned_key(self, raw_state: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Get abstract key with version info.
        
        Returns:
            (abstract_key, version, schema_id)
        """
        return (self.abstract(raw_state), self.VERSION, self.SCHEMA_ID)


# Default shared instance
_abstractor: Optional[StateAbstractor] = None

def get_state_abstractor() -> StateAbstractor:
    """Get or create shared StateAbstractor."""
    global _abstractor
    if _abstractor is None:
        _abstractor = StateAbstractor()
    return _abstractor
