"""
Voice package: TTS engines and voice routing.

Re-exports for backward compatibility.
"""
# Re-export for backward compatibility
from voice.kokoro_tts import KokoroTTSEngine, setup_kokoro_voice
from voice.xvasynth_engine import XVASynthEngine
from voice.voice_router import VoiceRouter, VoiceRequest, VoiceIntensity, VoiceConfig
from voice.chatterbox_tts import ChatterboxTTSEngine, ChatterboxMultilingualTTS
from voice.streaming_voice_system import DequeSpeechQueue, VoiceChunk

__all__ = [
    "KokoroTTSEngine",
    "setup_kokoro_voice",
    "XVASynthEngine",
    "VoiceRouter",
    "VoiceRequest",
    "VoiceIntensity",
    "VoiceConfig",
    "ChatterboxTTSEngine",
    "ChatterboxMultilingualTTS",
    "DequeSpeechQueue",
    "VoiceChunk",
]
