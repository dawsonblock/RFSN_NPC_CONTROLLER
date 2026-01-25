import hashlib
import json
from typing import Dict, Any, Optional, List

class StateAbstractor:
    """
    Maps high-dimensionality RawState to low-dimensionality AbstractState buckets.
    Ensures deterministic, hashable identifiers for bandit contexts.
    """
    
    def __init__(self, key_features: Optional[List[str]] = None):
        # Default features if none provided
        self.key_features = key_features or [
            "location",
            "relationship", 
            "threat_level",
            "quest_state"
        ]
        
    def abstract(self, raw_state: Dict[str, Any]) -> str:
        """
        Convert raw state dictionary into a deterministic abstract key.
        
        Args:
            raw_state: The full state dictionary from the game/world model.
            
        Returns:
            A string identifier (e.g., "LOC:tavern|REL:friendly|THREAT:0")
        """
        segments = []
        for feature in sorted(self.key_features):
            val = raw_state.get(feature, "unk")
            # Normalize value
            if isinstance(val, (int, float)):
                val = str(val)
            elif isinstance(val, str):
                val = val.lower().strip()
            else:
                val = str(val)
                
            segments.append(f"{feature.upper()}:{val}")
            
        # Create a readable but unique string
        # We don't hash yet to keep it debuggable, unless it gets too long
        abstract_key = "|".join(segments)
        
        return abstract_key

    def get_context_id(self, raw_state: Dict[str, Any]) -> str:
        """
        Returns a hashed version of the abstract state for efficient storage keys.
        """
        abstract_key = self.abstract(raw_state)
        return hashlib.sha256(abstract_key.encode()).hexdigest()[:16]
