#!/usr/bin/env python3
"""
Kokoro TTS Engine - Lightweight ONNX-based Text-to-Speech
Uses kokoro-onnx for fast, local TTS synthesis (~300MB model)

Replaces Piper TTS with a more lightweight and faster alternative.
"""

import threading
import queue
import logging
import tempfile
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model download URLs
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class KokoroTTSEngine:
    """
    Production Kokoro TTS using kokoro-onnx.
    
    Features:
    - Lightweight ~300MB model (quantized ~80MB)
    - Real-time synthesis on M1 Macs
    - Multiple voice support
    - Thread-safe queue-based playback
    """
    
    def __init__(
        self, 
        model_path: str = None, 
        voices_path: str = None,
        voice: str = "af_bella",
        speed: float = 1.0,
        enable_queue: bool = True,
        max_queue_size: int = 10
    ):
        self.model_path = Path(model_path) if model_path else None
        self.voices_path = Path(voices_path) if voices_path else None
        self.voice = voice
        self.speed = speed
        self.enable_queue = enable_queue
        
        # Audio queue with backpressure
        self.audio_queue = queue.Queue(maxsize=max_queue_size) if enable_queue else None
        self._shutdown = False
        self._dropped_count = 0
        
        # Kokoro model instance
        self._kokoro = None
        self._load_model()
        
        # Start synthesis worker if queue mode enabled
        self.worker = None
        if enable_queue and self._kokoro is not None:
            self.worker = threading.Thread(target=self._synthesis_worker, daemon=True)
            self.worker.start()
        
        mode = "async queue" if enable_queue else "sync-only"
        logger.info(f"[Kokoro] Engine ready (model={self._kokoro is not None}, mode={mode})")
    
    def _load_model(self):
        """Load the Kokoro ONNX model"""
        try:
            from kokoro_onnx import Kokoro
            
            if self.model_path and self.model_path.exists() and self.voices_path and self.voices_path.exists():
                self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
                logger.info(f"[Kokoro] Model loaded: {self.model_path.name}")
            else:
                # Try default paths
                default_model = Path(__file__).parent.parent / "Models" / "kokoro" / "kokoro-v1.0.onnx"
                default_voices = Path(__file__).parent.parent / "Models" / "kokoro" / "voices-v1.0.bin"
                
                if default_model.exists() and default_voices.exists():
                    self._kokoro = Kokoro(str(default_model), str(default_voices))
                    self.model_path = default_model
                    self.voices_path = default_voices
                    logger.info(f"[Kokoro] Model loaded from default path")
                else:
                    logger.warning("[Kokoro] Model not found. Run setup_kokoro_voice() to download.")
                    self._kokoro = None
        except ImportError:
            logger.warning("[Kokoro] kokoro-onnx not installed. Install with: pip install kokoro-onnx")
            self._kokoro = None
        except Exception as e:
            logger.error(f"[Kokoro] Failed to load model: {e}")
            self._kokoro = None
    
    def _synthesis_worker(self):
        """Background thread for synthesis + playback"""
        while not self._shutdown:
            try:
                text = self.audio_queue.get(timeout=30)
                if text is None:  # Shutdown signal
                    break
                
                try:
                    self._synthesize_and_play(text)
                except Exception as synth_error:
                    logger.error(f"[Kokoro] Synthesis failed for '{text[:30]}...': {synth_error}")
                    continue
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Kokoro] Worker error: {e}", exc_info=True)
    
    def _synthesize_and_play(self, text: str, pitch: float = 1.0, rate: float = 1.0):
        """
        Synthesize text to audio and play it.
        """
        if self._kokoro is None:
            logger.info(f"[Kokoro-Mock] Would speak: {text[:50]}...")
            return
        
        try:
            start_time = time.perf_counter()
            
            # Generate audio using Kokoro
            # Kokoro.create() returns (samples, sample_rate)
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self.voice,
                speed=self.speed * rate
            )
            
            gen_time = time.perf_counter() - start_time
            duration = len(samples) / sample_rate
            logger.info(f"[Kokoro] Generated {duration:.2f}s audio in {gen_time:.2f}s (RTF: {gen_time/duration:.2f})")
            
            # Write to temp file and play
            self._play_samples(samples, sample_rate)
            
        except Exception as e:
            logger.error(f"[Kokoro] Synthesis error: {e}")
            raise
    
    def _play_samples(self, samples, sample_rate: int):
        """Play audio samples using platform-appropriate command"""
        import numpy as np
        import wave
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Convert float32 samples to int16
            if hasattr(samples, 'numpy'):
                samples = samples.numpy()
            samples_int16 = (samples * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples_int16.tobytes())
            
            # Play using platform-appropriate command
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.run(["afplay", tmp_path], check=True, timeout=60)
            elif system == "Linux":
                if shutil.which("aplay"):
                    subprocess.run(["aplay", "-q", tmp_path], check=True, timeout=60)
                elif shutil.which("paplay"):
                    subprocess.run(["paplay", tmp_path], check=True, timeout=60)
                else:
                    logger.warning("[Kokoro] No audio player found on Linux")
            elif system == "Windows":
                ps_cmd = f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"
                subprocess.run(["powershell", "-Command", ps_cmd], check=True, timeout=60)
            else:
                logger.warning(f"[Kokoro] Unsupported platform: {system}")
                
        except subprocess.TimeoutExpired:
            logger.warning("[Kokoro] Audio playback timed out")
        except Exception as e:
            logger.error(f"[Kokoro] Audio playback error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def speak(self, text: str) -> bool:
        """
        Queue text for synthesis with backpressure handling.
        Returns: True if queued, False if dropped.
        """
        if not self.enable_queue:
            raise RuntimeError(
                "KokoroTTSEngine.speak() called but enable_queue=False. "
                "Use speak_sync() for synchronous mode."
            )
        
        if not text or self._shutdown:
            return False
        
        try:
            self.audio_queue.put_nowait(text)
            return True
        except queue.Full:
            # Backpressure: drop oldest, add new
            try:
                old_text = self.audio_queue.get_nowait()
                self._dropped_count += 1
                logger.warning(f"[Kokoro] Dropped audio due to backpressure: '{old_text[:30]}...'")
                self.audio_queue.put_nowait(text)
                return True
            except queue.Empty:
                return False
    
    def speak_sync(self, text: str, pitch: float = 1.0, rate: float = 1.0):
        """Synchronously synthesize and play text (blocking)"""
        self._synthesize_and_play(text, pitch=pitch, rate=rate)
    
    def wait_until_done(self):
        """Block until all queued audio is spoken"""
        if self.audio_queue:
            self.audio_queue.join()
    
    def shutdown(self):
        """Graceful shutdown of worker"""
        self._shutdown = True
        if self.enable_queue and self.audio_queue is not None:
            try:
                self.audio_queue.put_nowait(None)
            except queue.Full:
                pass
            if self.worker:
                self.worker.join(timeout=5)
        logger.info("[Kokoro] Engine shutdown complete")
    
    def get_queue_size(self) -> int:
        """Get current queue size for metrics"""
        return self.audio_queue.qsize() if self.audio_queue else 0
    
    def get_dropped_count(self) -> int:
        """Get count of dropped sentences due to backpressure"""
        return self._dropped_count
    
    def list_voices(self) -> list:
        """List available voices"""
        # Kokoro v1.0 voices (from VOICES.md)
        return [
            "af_bella", "af_sarah", "af_nicole", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]


def setup_kokoro_voice() -> tuple:
    """
    Download Kokoro model files if not present.
    Returns: (model_path, voices_path)
    """
    kokoro_dir = Path(__file__).parent.parent / "Models" / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = kokoro_dir / "kokoro-v1.0.onnx"
    voices_path = kokoro_dir / "voices-v1.0.bin"
    
    if not model_path.exists() or not voices_path.exists():
        logger.info("[Kokoro] Downloading model files...")
        
        try:
            import requests
            from tqdm import tqdm
            
            def download_with_progress(url: str, dest: Path):
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total = int(response.headers.get('content-length', 0))
                
                with open(dest, 'wb') as f:
                    with tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            if not model_path.exists():
                logger.info(f"[Kokoro] Downloading model (~300MB)...")
                download_with_progress(KOKORO_MODEL_URL, model_path)
            
            if not voices_path.exists():
                logger.info(f"[Kokoro] Downloading voices...")
                download_with_progress(KOKORO_VOICES_URL, voices_path)
            
            logger.info(f"[Kokoro] Download complete!")
            
        except ImportError:
            logger.error("[Kokoro] requests/tqdm not installed. Cannot download model.")
            return None, None
        except Exception as e:
            logger.error(f"[Kokoro] Failed to download model: {e}")
            return None, None
    
    return str(model_path), str(voices_path)


if __name__ == "__main__":
    # Quick test
    print("Testing Kokoro TTS Engine...")
    
    model_path, voices_path = setup_kokoro_voice()
    if model_path:
        engine = KokoroTTSEngine(model_path, voices_path)
        engine.speak_sync("Hello, I am a test of the Kokoro text to speech engine.")
        engine.shutdown()
    else:
        print("Running in mock mode (no model)")
        engine = KokoroTTSEngine()
        engine.speak_sync("This is a mock test.")
        engine.shutdown()
    
    print("Test complete!")
