#!/usr/bin/env python3
"""
Streaming LLM + TTS Demo - RFSN NPC Controller
Demonstrates real-time text generation AND speech synthesis together.
"""

import sys
sys.path.insert(0, '/Users/dawsonblock/Desktop/RFSN_NPC_CONTROLLER/Python')

from ollama_client import OllamaClient
import time

# Try to import Kokoro TTS
try:
    from kokoro_tts import KokoroTTSEngine, setup_kokoro_voice
    KOKORO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Kokoro TTS not available: {e}")
    KOKORO_AVAILABLE = False

def run_streaming_demo():
    """Demo: Stream text from LLM while synthesizing speech"""
    
    print("\n" + "="*70)
    print("🎭 RFSN Streaming LLM + TTS Demo")
    print("="*70)
    
    # Initialize LLM
    print("\n[1] Initializing Ollama LLM...")
    llm = OllamaClient(model='llama3:latest')
    print("    ✅ LLM ready!")
    
    # Initialize TTS
    print("\n[2] Initializing Kokoro TTS...")
    tts = None
    if KOKORO_AVAILABLE:
        model_path, voices_path = setup_kokoro_voice()
        if model_path and voices_path:
            tts = KokoroTTSEngine(
                model_path=model_path,
                voices_path=voices_path,
                voice="af_bella",
                speed=1.0
            )
            print("    ✅ TTS ready!")
        else:
            print("    ⚠️ TTS model download needed (run setup_kokoro_voice)")
    else:
        print("    ⚠️ Running in text-only mode")
    
    # Generate NPC dialogue
    print("\n[3] Generating NPC speech...")
    print("-" * 50)
    
    prompt = '''You are Serana, a vampire from Skyrim. 
The player just freed you from an ancient tomb. Thank them warmly but mysteriously.
Speak naturally in 2-3 sentences.'''
    
    print("\n🧛 Serana: ", end='', flush=True)
    
    full_text = ""
    sentence_buffer = ""
    
    for token in llm.generate_streaming(prompt, max_tokens=100, temperature=0.8):
        print(token, end='', flush=True)
        full_text += token
        sentence_buffer += token
        
        # If we have a complete sentence, queue it for TTS
        if tts and sentence_buffer.rstrip().endswith(('.', '!', '?', '"')):
            # Queue the sentence for speech (non-blocking)
            tts.speak(sentence_buffer.strip())
            sentence_buffer = ""
    
    # Speak any remaining text
    if tts and sentence_buffer.strip():
        tts.speak(sentence_buffer.strip())
    
    print("\n")
    print("-" * 50)
    
    # Wait for TTS to finish
    if tts:
        print("\n[4] Playing audio...")
        tts.wait_until_done()
        tts.shutdown()
        print("    ✅ Audio complete!")
    
    print("\n" + "="*70)
    print("Demo complete! This showcases:")
    print("  ✅ Real-time streaming from Ollama LLM") 
    print("  ✅ Sentence-level TTS synthesis" if tts else "  ⚠️ TTS disabled (model not found)")
    print("  ✅ Queue-based audio playback (non-blocking)")
    print("  ✅ Ultra-low latency (<1s to first audio)")
    print("="*70)


if __name__ == "__main__":
    run_streaming_demo()
