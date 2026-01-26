"""
DecisionIndex: Fast lookup from JSONL decision logs.

Provides efficient decision_id → record mapping for:
- Reward validation
- Trace lookups
- Offline evaluation
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator
import logging

logger = logging.getLogger("tools.decision_index")


class DecisionIndex:
    """
    Fast in-memory index of decision records.
    
    Supports:
    - Streaming parse of large JSONL files
    - decision_id → full record lookup
    - Session-based queries
    - Stats aggregation
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("data/learning/decisions.jsonl")
        
        # In-memory index: decision_id → record
        self._index: Dict[str, Dict[str, Any]] = {}
        
        # Session index: (npc_id, session_id) → [decision_ids]
        self._sessions: Dict[tuple, List[str]] = {}
    
    def load(self) -> int:
        """
        Load/refresh index from log file.
        
        Returns:
            Number of records indexed
        """
        self._index.clear()
        self._sessions.clear()
        
        if not self.log_path.exists():
            logger.warning(f"Decision log not found: {self.log_path}")
            return 0
        
        count = 0
        for record in self._stream_records():
            decision_id = record.get("decision_id")
            if decision_id:
                self._index[decision_id] = record
                
                # Build session index
                npc_id = record.get("npc_id", "")
                session_id = record.get("session_id", "")
                session_key = (npc_id, session_id)
                
                if session_key not in self._sessions:
                    self._sessions[session_key] = []
                self._sessions[session_key].append(decision_id)
                
                count += 1
        
        logger.info(f"[INDEX] Loaded {count} decisions from {self.log_path}")
        return count
    
    def _stream_records(self) -> Iterator[Dict[str, Any]]:
        """Stream records from JSONL without loading all into memory."""
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            logger.error(f"Failed to read {self.log_path}: {e}")
    
    def exists(self, decision_id: str) -> bool:
        """Check if decision_id exists in index."""
        return decision_id in self._index
    
    def get(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get decision record by ID."""
        return self._index.get(decision_id)
    
    def get_session_decisions(
        self, 
        npc_id: str, 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Get all decisions for a session."""
        session_key = (npc_id, session_id)
        decision_ids = self._sessions.get(session_key, [])
        return [self._index[did] for did in decision_ids if did in self._index]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        action_counts: Dict[str, int] = {}
        exploration_count = 0
        safety_veto_count = 0
        
        for record in self._index.values():
            action = record.get("chosen_action_id", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
            
            if record.get("exploration"):
                exploration_count += 1
            if record.get("safety_vetoed"):
                safety_veto_count += 1
        
        return {
            "total_decisions": len(self._index),
            "total_sessions": len(self._sessions),
            "action_distribution": action_counts,
            "exploration_rate": exploration_count / max(1, len(self._index)),
            "safety_veto_rate": safety_veto_count / max(1, len(self._index))
        }
    
    def get_state_action_pairs(self) -> Dict[str, Dict[str, int]]:
        """
        Get action distribution per abstract state.
        
        Returns:
            {abstract_state_key: {action: count}}
        """
        pairs: Dict[str, Dict[str, int]] = {}
        
        for record in self._index.values():
            state_key = record.get("abstract_state_key", "unknown")
            action = record.get("chosen_action_id", "unknown")
            
            if state_key not in pairs:
                pairs[state_key] = {}
            pairs[state_key][action] = pairs[state_key].get(action, 0) + 1
        
        return pairs


def build_index(log_path: Optional[Path] = None) -> DecisionIndex:
    """Build a fresh decision index from logs."""
    index = DecisionIndex(log_path)
    index.load()
    return index


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Decision Index Tool")
    parser.add_argument("--logdir", default="data/learning", help="Log directory")
    parser.add_argument("--stats", action="store_true", help="Print stats")
    args = parser.parse_args()
    
    log_path = Path(args.logdir) / "decisions.jsonl"
    index = build_index(log_path)
    
    if args.stats:
        stats = index.get_stats()
        print(json.dumps(stats, indent=2))
