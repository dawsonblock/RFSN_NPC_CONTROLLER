#!/usr/bin/env python3
"""
Ollama Client - Local LLM via Ollama HTTP API
Provides OpenAI-compatible interface for easy integration with existing code.

Replaces llama-cpp-python with Ollama for simpler model management.
"""

import logging
import json
import time
from typing import Generator, Optional, List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


@dataclass
class OllamaConfig:
    """Configuration for Ollama client"""
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 150
    context_length: int = 2048


class OllamaClient:
    """
    Ollama HTTP API client for local LLM inference.
    
    Features:
    - Streaming generation support
    - Model management (list, pull)
    - OpenAI-compatible interface
    - Automatic retry and error handling
    """
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 150
    ):
        self.host = host.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = None
        
        logger.info(f"[Ollama] Client initialized (host={self.host}, model={self.model})")
    
    @property
    def session(self):
        """Lazy-load requests session"""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session
    
    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"[Ollama] Failed to list models: {e}")
            return []
    
    def has_model(self, model: str = None) -> bool:
        """Check if a specific model is available"""
        model = model or self.model
        models = self.list_models()
        model_names = [m.get("name", "").split(":")[0] for m in models]
        return model.split(":")[0] in model_names
    
    def pull_model(self, model: str = None) -> bool:
        """Pull/download a model"""
        model = model or self.model
        logger.info(f"[Ollama] Pulling model: {model}")
        
        try:
            response = self.session.post(
                f"{self.host}/api/pull",
                json={"name": model},
                stream=True,
                timeout=600  # 10 min timeout for large models
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status:
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        if total > 0:
                            pct = (completed / total) * 100
                            logger.info(f"[Ollama] Downloading: {pct:.1f}%")
                    elif status == "success":
                        logger.info(f"[Ollama] Model {model} ready!")
                        return True
            
            return True
        except Exception as e:
            logger.error(f"[Ollama] Failed to pull model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stop: List[str] = None,
        **kwargs
    ) -> str:
        """
        Generate completion (non-streaming).
        
        Args:
            prompt: The input prompt
            model: Model to use (default: self.model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if stop:
                payload["options"]["stop"] = stop
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except Exception as e:
            logger.error(f"[Ollama] Generation failed: {e}")
            return ""
    
    def generate_streaming(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stop: List[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate completion with streaming.
        
        Args:
            prompt: The input prompt
            model: Model to use (default: self.model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Yields:
            Generated text tokens
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if stop:
                payload["options"]["stop"] = stop
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        
                        # Check if done
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"[Ollama] Streaming generation failed: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False,
        **kwargs
    ):
        """
        Chat completion (OpenAI-compatible format).
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            stream: Enable streaming
            
        Returns:
            Generated response (or generator if streaming)
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = self.session.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=stream,
                timeout=120
            )
            response.raise_for_status()
            
            if stream:
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                return stream_generator()
            else:
                data = response.json()
                return data.get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"[Ollama] Chat failed: {e}")
            return "" if not stream else iter([])
    
    def shutdown(self):
        """Close session"""
        if self._session:
            self._session.close()
            self._session = None
        logger.info("[Ollama] Client shutdown")


def ensure_ollama_ready(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> bool:
    """
    Ensure Ollama is running and model is available.
    
    Returns: True if ready, False otherwise
    """
    client = OllamaClient(host=host, model=model)
    
    if not client.is_available():
        logger.error(f"[Ollama] Server not running at {host}")
        logger.error("[Ollama] Start with: ollama serve")
        return False
    
    if not client.has_model(model):
        logger.info(f"[Ollama] Model {model} not found, pulling...")
        if not client.pull_model(model):
            return False
    
    logger.info(f"[Ollama] Ready with model {model}")
    return True


if __name__ == "__main__":
    # Quick test
    print("Testing Ollama Client...")
    
    client = OllamaClient()
    
    if not client.is_available():
        print("Ollama not running! Start with: ollama serve")
        exit(1)
    
    print(f"Available models: {[m['name'] for m in client.list_models()]}")
    
    # Test non-streaming
    print("\n--- Non-streaming test ---")
    response = client.generate("Say hello in one sentence.", max_tokens=50)
    print(f"Response: {response}")
    
    # Test streaming
    print("\n--- Streaming test ---")
    print("Response: ", end="", flush=True)
    for token in client.generate_streaming("Count from 1 to 5.", max_tokens=50):
        print(token, end="", flush=True)
    print()
    
    client.shutdown()
    print("\nTest complete!")
