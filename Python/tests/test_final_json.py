"""
Tests for FINAL_JSON contract enforcement.
Verifies that only verified outputs enter memory and learning.
"""
import pytest
import json
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import _extract_tail_json_payload


class TestFinalJsonExtraction:
    """Test FINAL_JSON parsing and verification."""
    
    def test_valid_final_json_verified(self):
        """Valid FINAL_JSON: block should be verified."""
        raw = '''Hello there, traveler!

FINAL_JSON:
```json
{"line": "Hello there, traveler!", "action": "greet", "confidence": 0.95}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is True
        assert payload is not None
        assert payload["line"] == "Hello there, traveler!"
        assert payload["action"] == "greet"
        assert payload["confidence"] == 0.95
    
    def test_legacy_json_unverified(self):
        """Legacy ```json format without FINAL_JSON: prefix should be unverified."""
        raw = '''Hello there!

```json
{"line": "Hello there!", "action": "greet", "tone": "friendly"}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is False
        assert payload is not None
        assert payload["line"] == "Hello there!"
    
    def test_missing_line_field_unverified(self):
        """FINAL_JSON missing 'line' field should be unverified."""
        raw = '''FINAL_JSON:
```json
{"action": "greet", "confidence": 0.9}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        # Payload exists but is unverified due to missing required field
        assert is_verified is False
    
    def test_missing_action_field_unverified(self):
        """FINAL_JSON missing 'action' field should be unverified."""
        raw = '''FINAL_JSON:
```json
{"line": "Hello!", "confidence": 0.9}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is False
    
    def test_malformed_json_returns_none(self):
        """Malformed JSON should return None payload and unverified."""
        raw = '''FINAL_JSON:
```json
{"line": "Hello, broken JSON
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is False
        assert payload is None
    
    def test_no_json_block_returns_none(self):
        """No JSON block at all should return None and unverified."""
        raw = "Hello there, traveler! Welcome to our village."
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is False
        assert payload is None
    
    def test_confidence_out_of_range_clamped(self):
        """Confidence values outside [0,1] should be normalized."""
        raw = '''FINAL_JSON:
```json
{"line": "Hello!", "action": "greet", "confidence": 5.0}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is True
        assert payload["confidence"] == 0.5  # Defaulted due to invalid range
    
    def test_confidence_negative_clamped(self):
        """Negative confidence should be normalized."""
        raw = '''FINAL_JSON:
```json
{"line": "Hello!", "action": "greet", "confidence": -0.5}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is True
        assert payload["confidence"] == 0.5  # Defaulted
    
    def test_confidence_missing_defaults(self):
        """Missing confidence should still be verified (optional field)."""
        raw = '''FINAL_JSON:
```json
{"line": "Hello!", "action": "greet"}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is True
        assert "confidence" not in payload  # Not added if missing


class TestFinalJsonCaseSensitivity:
    """Test case handling in FINAL_JSON."""
    
    def test_final_json_case_insensitive(self):
        """FINAL_JSON: should match case-insensitively."""
        raw = '''final_json:
```json
{"line": "Hello!", "action": "greet", "confidence": 0.8}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is True
        assert payload is not None


class TestQuarantineLogic:
    """Test that unverified responses are properly quarantined."""
    
    def test_verified_response_has_confidence(self):
        """Verified responses should maintain their confidence value."""
        raw = '''FINAL_JSON:
```json
{"line": "I can help you.", "action": "help", "confidence": 0.92}
```'''
        payload, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is True
        assert payload["confidence"] == 0.92
    
    def test_unverified_should_have_zero_confidence_in_metadata(self):
        """This tests the contract - unverified turns get confidence=0.0 in storage."""
        # The actual quarantine logic is in orchestrator, but we verify
        # that the parser correctly identifies unverified outputs
        raw = '''Just some rambling without proper JSON format'''
        _, is_verified = _extract_tail_json_payload(raw)
        
        assert is_verified is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
