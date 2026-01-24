#!/usr/bin/env python3
"""
Chatterbox TTS Engine - ResembleAI's Production-Grade Voice Synthesis

Supports two model variants:
- Chatterbox-Full: Acting model with deep emotional range
- Chatterbox-Turbo: Workhorse for ambient NPCs and high-throughput

Features:
- Emotion exaggeration control (0.0-1.0)
- CFG scale for pacing control
- Zero-shot voice cloning via audio prompts
- Multilingual support (23 languages)
"""

import threading
import queue
import logging
import tempfile
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatterboxTTSEngine:
    """
    Chatterbox TTS with dual-model support (Full + Turbo).
    
    The engine can load either variant and provides consistent interface
    for the VoiceRouter to switch between them based on narrative weight.
    """
    
    def __init__(
        self,
        model_variant: str = "turbo",
        device: str = "cuda",
        enable_queue: bool = True,
        max_queue_size: int = 10
    ):
        """
        Initialize Chatterbox TTS Engine.
        
        Args:
            model_variant: "turbo" for workhorse, "full" for acting model
            device: "cuda" or "cpu" (cuda strongly recommended)
            enable_queue: Enable async queue mode
            max_queue_size: Maximum queued utterances
        """
        self.model_variant = model_variant
        self.device = device
        self.enable_queue = enable_queue
        
        # Audio queue with backpressure
        self.audio_queue = queue.Queue(maxsize=max_queue_size) if enable_queue else None
        self._shutdown = False
        self._dropped_count = 0
        
        # Chatterbox model instance
        self._model = None
        self._sample_rate = 24000  # Default Chatterbox sample rate
        
        # Voice cloning prompt (optional)
        self._audio_prompt: Optional[str] = None
        
        # Load model
        self._load_model()
        
        # Start synthesis worker if queue mode enabled
        self.worker = None
        if enable_queue and self._model is not None:
            self.worker = threading.Thread(target=self._synthesis_worker, daemon=True)
            self.worker.start()
        
        mode = "async queue" if enable_queue else "sync-only"
        logger.info(f"[Chatterbox] Engine ready (variant={model_variant}, device={device}, mode={mode})")
    
    def _load_model(self):
        """Load the Chatterbox model"""
        try:
            if self.model_variant == "turbo":
                from chatterbox.tts import ChatterboxTTS
                self._model = ChatterboxTTS.from_pretrained(device=self.device, variant="turbo")
                logger.info(f"[Chatterbox] Loaded Turbo model on {self.device}")
            else:
                from chatterbox.tts import ChatterboxTTS
                self._model = ChatterboxTTS.from_pretrained(device=self.device)
                logger.info(f"[Chatterbox] Loaded Full model on {self.device}")
            
            self._sample_rate = self._model.sr
            
        except ImportError:
            logger.warning("[Chatterbox] chatterbox-tts not installed. Install with: pip install chatterbox-tts")
            self._model = None
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.warning(f"[Chatterbox] CUDA not available: {e}. Running in mock mode.")
            else:
                logger.error(f"[Chatterbox] Failed to load model: {e}")
            self._model = None
        except Exception as e:
            logger.error(f"[Chatterbox] Failed to load model: {e}")
            self._model = None
    
    def set_voice_prompt(self, audio_path: str):
        """
        Set voice cloning prompt from audio file.
        
        Args:
            audio_path: Path to reference audio file (WAV recommended)
        """
        if Path(audio_path).exists():
            self._audio_prompt = audio_path
            logger.info(f"[Chatterbox] Voice prompt set: {audio_path}")
        else:
            logger.warning(f"[Chatterbox] Voice prompt not found: {audio_path}")
    
    def clear_voice_prompt(self):
        """Clear voice cloning prompt, use default voice"""
        self._audio_prompt = None
    
    def _synthesis_worker(self):
        """Background thread for synthesis + playback"""
        while not self._shutdown:
            try:
                item = self.audio_queue.get(timeout=30)
                if item is None:  # Shutdown signal
                    break
                
                text, exaggeration, cfg = item
                
                try:
                    self._synthesize_and_play(text, exaggeration, cfg)
                except Exception as synth_error:
                    logger.error(f"[Chatterbox] Synthesis failed: {synth_error}")
                    continue
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Chatterbox] Worker error: {e}", exc_info=True)
    
    def _synthesize_and_play(
        self, 
        text: str, 
        exaggeration: float = 0.5, 
        cfg: float = 0.5
    ):
        """
        Synthesize text to audio and play it.
        
        Args:
            text: Text to synthesize
            exaggeration: Emotion intensity (0.0-1.0, higher = more dramatic)
            cfg: CFG scale for pacing (lower = slower, more deliberate)
        """
        if self._model is None:
            # Mock mode for testing without GPU
            logger.info(f"[Chatterbox-Mock] Would speak (exagg={exaggeration:.1f}): {text[:50]}...")
            time.sleep(len(text) * 0.05)  # Simulate TTS delay
            return
        
        try:
            start_time = time.perf_counter()
            
            # Generate audio using Chatterbox
            if self._audio_prompt:
                wav = self._model.generate(
                    text,
                    audio_prompt_path=self._audio_prompt,
                    exaggeration=exaggeration,
                    cfg_weight=cfg
                )
            else:
                wav = self._model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg
                )
            
            gen_time = time.perf_counter() - start_time
            duration = len(wav[0]) / self._sample_rate if len(wav.shape) > 0 else len(wav) / self._sample_rate
            rtf = gen_time / duration if duration > 0 else 0
            logger.info(f"[Chatterbox] Generated {duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f})")
            
            # Write to temp file and play
            self._play_samples(wav, self._sample_rate)
            
        except Exception as e:
            logger.error(f"[Chatterbox] Synthesis error: {e}")
            raise
    
    def _play_samples(self, samples, sample_rate: int):
        """Play audio samples using platform-appropriate command"""
        import numpy as np
        import wave
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Handle tensor vs numpy conversion
            if hasattr(samples, 'cpu'):
                samples = samples.cpu().numpy()
            if hasattr(samples, 'numpy'):
                samples = samples.numpy()
            
            # Flatten if needed
            if len(samples.shape) > 1:
                samples = samples.squeeze()
            
            # Convert float32 samples to int16
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
                    logger.warning("[Chatterbox] No audio player found on Linux")
            elif system == "Windows":
                ps_cmd = f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"
                subprocess.run(["powershell", "-Command", ps_cmd], check=True, timeout=60)
            else:
                logger.warning(f"[Chatterbox] Unsupported platform: {system}")
                
        except subprocess.TimeoutExpired:
            logger.warning("[Chatterbox] Audio playback timed out")
        except Exception as e:
            logger.error(f"[Chatterbox] Audio playback error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def speak(self, text: str, exaggeration: float = 0.5, cfg: float = 0.5) -> bool:
        """
        Queue text for synthesis with backpressure handling.
        
        Args:
            text: Text to synthesize
            exaggeration: Emotion intensity (0.0-1.0)
            cfg: CFG scale for pacing
            
        Returns: True if queued, False if dropped.
        """
        if not self.enable_queue:
            raise RuntimeError(
                "ChatterboxTTSEngine.speak() called but enable_queue=False. "
                "Use speak_sync() for synchronous mode."
            )
        
        if not text or self._shutdown:
            return False
        
        try:
            self.audio_queue.put_nowait((text, exaggeration, cfg))
            return True
        except queue.Full:
            # Backpressure: drop oldest, add new
            try:
                old_item = self.audio_queue.get_nowait()
                self._dropped_count += 1
                old_text = old_item[0] if isinstance(old_item, tuple) else old_item
                logger.warning(f"[Chatterbox] Dropped audio due to backpressure: '{old_text[:30]}...'")
                self.audio_queue.put_nowait((text, exaggeration, cfg))
                return True
            except queue.Empty:
                return False
    
    def speak_sync(
        self, 
        text: str, 
        exaggeration: float = 0.5, 
        cfg: float = 0.5,
        audio_prompt_path: Optional[str] = None
    ):
        """
        Synchronously synthesize and play text (blocking).
        
        Args:
            text: Text to synthesize
            exaggeration: Emotion intensity (0.0-1.0)
            cfg: CFG scale for pacing
            audio_prompt_path: Optional voice cloning reference (overrides set prompt)
        """
        # Temporarily set voice prompt if provided
        original_prompt = self._audio_prompt
        if audio_prompt_path:
            self._audio_prompt = audio_prompt_path
        
        try:
            self._synthesize_and_play(text, exaggeration, cfg)
        finally:
            self._audio_prompt = original_prompt
    
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
        logger.info(f"[Chatterbox] Engine shutdown complete ({self.model_variant})")
    
    def get_queue_size(self) -> int:
        """Get current queue size for metrics"""
        return self.audio_queue.qsize() if self.audio_queue else 0
    
    def get_dropped_count(self) -> int:
        """Get count of dropped sentences due to backpressure"""
        return self._dropped_count
    
    @property
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model is not None


# Multilingual support
class ChatterboxMultilingualTTS:
    """
    Multilingual Chatterbox TTS for 23 languages.
    
    Use for system voice, narration, or non-English NPCs.
    """
    
    SUPPORTED_LANGUAGES = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", 
        "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
        "pl", "pt", "ru", "sv", "sw", "tr", "zh"
    ]
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as MTL
            self._model = MTL.from_pretrained(device=self.device)
            logger.info(f"[Chatterbox-MTL] Multilingual model loaded on {self.device}")
        except ImportError:
            logger.warning("[Chatterbox-MTL] chatterbox-tts not installed")
            self._model = None
        except Exception as e:
            logger.error(f"[Chatterbox-MTL] Failed to load: {e}")
            self._model = None
    
    def generate(self, text: str, language_id: str = "en") -> Tuple:
        """Generate audio for given text and language"""
        if self._model is None:
            logger.warning(f"[Chatterbox-MTL-Mock] Would generate ({language_id}): {text[:30]}...")
            return None, 24000
        
        wav = self._model.generate(text, language_id=language_id)
        return wav, self._model.sr


if __name__ == "__main__":
    # Quick test
    print("Testing Chatterbox TTS Engine...")
    print("=" * 60)
    
    # Test Turbo variant
    print("\n[Test 1] Chatterbox-Turbo (workhorse):")
    turbo = ChatterboxTTSEngine(model_variant="turbo", device="cuda", enable_queue=False)
    if turbo.is_available:
        turbo.speak_sync("Hello, I am a guard. Move along citizen.", exaggeration=0.3, cfg=0.5)
    else:
        print("Running in mock mode (no CUDA)")
        turbo.speak_sync("Hello, I am a guard.", exaggeration=0.3, cfg=0.5)
    turbo.shutdown()
    
    # Test Full variant  
    print("\n[Test 2] Chatterbox-Full (acting model):")
    full = ChatterboxTTSEngine(model_variant="full", device="cuda", enable_queue=False)
    if full.is_available:
        full.speak_sync(
            "I remember when you saved my life... I never forgot that moment.", 
            exaggeration=0.8, 
            cfg=0.3
        )
    else:
        print("Running in mock mode (no CUDA)")
        full.speak_sync("I remember...", exaggeration=0.8, cfg=0.3)
    full.shutdown()
    
    print("\n" + "=" * 60)
    print("Test complete!")
