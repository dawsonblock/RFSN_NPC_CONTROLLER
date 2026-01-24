#!/usr/bin/env python3
"""
Universe Story Demo - RFSN NPC Controller
A beautiful, scientifically-accurate story about the origin of the universe.

Usage:
    python demos/universe_story_demo.py [--with-audio]
"""

import sys
import argparse
sys.path.insert(0, '/Users/dawsonblock/Desktop/RFSN_NPC_CONTROLLER/Python')

from ollama_client import OllamaClient

def run_story(with_audio: bool = False):
    """Generate and optionally speak a story about the universe"""
    
    print()
    print('🌌 THE ORIGIN OF EVERYTHING')
    print('=' * 60)
    print()
    
    llm = OllamaClient(model='llama3:latest')
    
    prompt = '''Tell me a beautiful, scientifically-accurate story about the origin of the universe in 4 short paragraphs:

1. The Big Bang and the first moments of time
2. The formation of the first stars and galaxies  
3. How heavy elements were forged in stellar furnaces
4. How those elements became Earth and eventually life

Be poetic but accurate. Use vivid imagery. Keep each paragraph to 3-4 sentences.'''
    
    # Collect full story first
    full_story = ""
    print("Generating story...\n")
    
    for token in llm.generate_streaming(prompt, max_tokens=500, temperature=0.7):
        print(token, end='', flush=True)
        full_story += token
    
    print()
    print()
    print('=' * 60)
    
    # Optional TTS
    if with_audio:
        print("\n🔊 Now playing audio narration...")
        try:
            from kokoro_tts import KokoroTTSEngine, setup_kokoro_voice
            
            model_path, voices_path = setup_kokoro_voice()
            if model_path and voices_path:
                tts = KokoroTTSEngine(
                    model_path=model_path,
                    voices_path=voices_path,
                    voice="af_bella",
                    speed=1.0,
                    max_queue_size=50  # Large queue to avoid drops
                )
                
                # Speak the full story in chunks
                paragraphs = full_story.strip().split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        tts.speak(para.strip())
                
                tts.wait_until_done()
                tts.shutdown()
                print("✅ Audio complete!")
            else:
                print("⚠️ TTS model not found. Run setup_kokoro_voice() first.")
        except ImportError:
            print("⚠️ Kokoro TTS not available.")
    
    print("\nDemo complete! 🌟")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universe origin story demo")
    parser.add_argument("--with-audio", action="store_true", help="Enable TTS narration")
    args = parser.parse_args()
    
    run_story(with_audio=args.with_audio)
