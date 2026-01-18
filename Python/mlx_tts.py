import time
import numpy as np
import queue
import logging
import tempfile
import os
import subprocess
import wave
import shutil
from pathlib import Path
from mlx_audio.tts.utils import load_model, get_model_path
import mlx.core as mx

logger = logging.getLogger("orchestrator")

class MlxTTSEngine:
    def __init__(self, model_path: str, speed: float = 1.0, max_queue_size: int = 10):
        self.model_path = model_path
        self.speech_speed = speed
        self.audio_queue = queue.Queue(maxsize=max_queue_size)
        
        logger.info(f"Initializing MLX TTS with model={model_path}, speed={speed}")
        
        # Load model
        # load_model expects a Path or str.
        self.model = load_model(model_path)
        logger.info("MLX TTS Model loaded successfully.")
        
        # Warmup to prevent delay on first user request
        self._warmup()

    def _warmup(self):
        """Generate a short silent burst to warm up the model weights."""
        try:
            logger.info("Warming up MLX TTS model...")
            start = time.perf_counter()
            # Generate a single short token
            list(self.model.generate(text=".", speed=1.0, stream=False))
            logger.info(f"MLX TTS Warmup complete in {time.perf_counter() - start:.2f}s")
        except Exception as e:
            logger.warning(f"MLX TTS Warmup failed: {e}")

    def speak_sync(self, text: str):
        """Alias for say() to match StreamingVoiceSystem interface."""
        self.say(text)

    def say(self, text: str, priority: int = 0):
        """
        Synthesize text to audio and push raw bytes to the queue.
        """
        try:
            start_t = time.perf_counter()
            
            # Use model.generate() directly
            # It returns a generator yielding results with .audio (mx.array)
            # We enforce blocking generation here for simplicity, or we could yielding
            # But the Orchestrator expects us to push to self.audio_queue
            
            # Chatterbox/Kokoro models usually accept text, speed, etc.
            # We rely on defaults for voice/lang if not specified.
            # Chatterbox is single speaker usually? Or has embedded styles.
            
            # Generate returns an iterator of results.
            # We'll collect them or push chunk by chunk.
            # The 'stream' argument in model.generate might control chunking.
            
            # Let's try to generate one big chunk for now to ensure cohesion, 
            # or stream if we want lower latency.
            # Piper engine pushes PCM bytes.
            
            generator = self.model.generate(
                text=text, 
                speed=self.speech_speed,
                stream=False # Get full result at once for safety first
            )
            
            full_audio = []
            sample_rate = 24000 
            
            for result in generator:
                # result has .audio, .sample_rate, .audio_duration
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate
                
                if hasattr(result, "audio"):
                    # result.audio is mx.array
                    full_audio.append(result.audio)
            
            if not full_audio:
                logger.warning("MLX TTS produced no audio.")
                return

            # Concatenate if multiple chunks
            if len(full_audio) > 1:
                audio_mx = mx.concatenate(full_audio, axis=0)
            else:
                audio_mx = full_audio[0]

            # Convert to numpy
            audio_np = np.array(audio_mx)
            
            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Get bytes
            audio_bytes = audio_int16.tobytes()
            
            duration = len(audio_np) / sample_rate
            gen_time = time.perf_counter() - start_t
            
            logger.info(f"[MLX-TTS] Generated {duration:.2f}s audio in {gen_time:.2f}s (RTF: {gen_time/duration:.2f})")
            
            # Push to queue (for potential external consumption)
            try:
                self.audio_queue.put(audio_bytes, timeout=0.1)
            except queue.Full:
                pass
            
            # Synchronous Playback (Patch 4.1)
            self._play_bytes_as_wav(audio_bytes, sample_rate)
            
        except Exception as e:
            logger.error(f"MLX TTS Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _play_bytes_as_wav(self, audio_bytes: bytes, sample_rate: int):
        """Write PCM bytes to a temp WAV and play it."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(1) # Mono
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
            
            # Play using afplay (Mac) or aplay/paplay (Linux)
            import platform
            system = platform.system()
            
            if system == "Darwin":
                subprocess.run(["afplay", tmp_path], check=True)
            elif system == "Linux":
                if shutil.which("aplay"):
                    subprocess.run(["aplay", "-q", tmp_path], check=True)
                elif shutil.which("paplay"):
                    subprocess.run(["paplay", tmp_path], check=True)
            elif system == "Windows":
                 from ctypes import windll
                 windll.winmm.PlaySoundW(tmp_path, None, 0x00020000 | 0x00000001)
        
        except Exception as e:
            logger.error(f"MLX Audio Playback failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def stream_audio(self):
        """
        Yields chunks of audio bytes from the queue.
        """
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def shutdown(self):
        self.audio_queue.put(None)
