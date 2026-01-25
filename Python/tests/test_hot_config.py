"""
Tests for hot config nested key access after surgical upgrade.
Verifies that dot-path config access works correctly.
"""
import pytest
import json
import tempfile
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hot_config import ConfigWatcher


class TestNestedConfigAccess:
    """Verify that ConfigWatcher supports dot-path nested key access."""
    
    @pytest.fixture
    def nested_config_file(self, tmp_path):
        """Create a temporary config file with nested structure."""
        config = {
            "llm": {
                "backend": "ollama",
                "temperature": 0.7,
                "max_tokens": 150,
                "nested": {
                    "deep_key": "deep_value"
                }
            },
            "tts": {
                "backend": "kokoro",
                "voice": "af_bella"
            },
            "flat_key": "flat_value"
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        return str(config_path)
    
    def test_dot_path_access_single_level(self, nested_config_file):
        """Test accessing single-level nested keys."""
        watcher = ConfigWatcher(nested_config_file)
        
        temp = watcher.get("llm.temperature", 0.5)
        assert temp == 0.7, "Should read nested llm.temperature"
        
        backend = watcher.get("llm.backend", "unknown")
        assert backend == "ollama", "Should read nested llm.backend"
    
    def test_dot_path_access_multi_level(self, nested_config_file):
        """Test accessing deeply nested keys."""
        watcher = ConfigWatcher(nested_config_file)
        
        deep = watcher.get("llm.nested.deep_key", "default")
        assert deep == "deep_value", "Should read deeply nested key"
    
    def test_flat_key_still_works(self, nested_config_file):
        """Test that flat keys still work after adding dot-path support."""
        watcher = ConfigWatcher(nested_config_file)
        
        flat = watcher.get("flat_key", "default")
        assert flat == "flat_value", "Should read flat key"
    
    def test_missing_nested_key_returns_default(self, nested_config_file):
        """Test that missing nested keys return the default value."""
        watcher = ConfigWatcher(nested_config_file)
        
        missing = watcher.get("llm.nonexistent", "my_default")
        assert missing == "my_default", "Should return default for missing nested key"
        
        missing_deep = watcher.get("llm.nested.nonexistent", "deep_default")
        assert missing_deep == "deep_default", "Should return default for missing deep key"
    
    def test_path_through_non_dict_returns_default(self, nested_config_file):
        """Test that paths through non-dict values return default."""
        watcher = ConfigWatcher(nested_config_file)
        
        # llm.temperature is a float, not a dict
        result = watcher.get("llm.temperature.invalid", "default")
        assert result == "default", "Should return default when traversing non-dict"
    
    def test_empty_path_segment_handled(self, nested_config_file):
        """Test that empty path segments are handled gracefully."""
        watcher = ConfigWatcher(nested_config_file)
        
        # This shouldn't crash
        result = watcher.get("", "default")
        assert result == "default", "Empty string should return default"


class TestConfigWatcherBackwardCompatibility:
    """Ensure backward compatibility with flat config access patterns."""
    
    @pytest.fixture
    def flat_config_file(self, tmp_path):
        """Create a temporary config file with flat structure."""
        config = {
            "temperature": 0.8,
            "max_tokens": 200,
            "memory_enabled": True
        }
        config_path = tmp_path / "flat_config.json"
        config_path.write_text(json.dumps(config))
        return str(config_path)
    
    def test_flat_config_all_keys_work(self, flat_config_file):
        """Test that flat configs work unchanged."""
        watcher = ConfigWatcher(flat_config_file)
        
        assert watcher.get("temperature") == 0.8
        assert watcher.get("max_tokens") == 200
        assert watcher.get("memory_enabled") is True
    
    def test_get_all_returns_full_config(self, flat_config_file):
        """Test that get_all still returns the full config."""
        watcher = ConfigWatcher(flat_config_file)
        
        config = watcher.get_all()
        assert "temperature" in config
        assert "max_tokens" in config
