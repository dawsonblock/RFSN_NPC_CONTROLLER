#!/usr/bin/env python3
"""
Multi-NPC Dialogue Demo - RFSN NPC Controller
Demonstrates streaming LLM responses from multiple NPCs in a tavern scene.
"""

import sys
sys.path.insert(0, '/Users/dawsonblock/Desktop/RFSN_NPC_CONTROLLER/Python')

from ollama_client import OllamaClient
import time

# ANSI colors for different NPCs
COLORS = {
    'narrator': '\033[90m',  # Gray
    'lydia': '\033[94m',     # Blue  
    'innkeeper': '\033[93m', # Yellow
    'bard': '\033[95m',      # Magenta
    'reset': '\033[0m'
}

def speak(npc_name: str, client: OllamaClient, prompt: str, max_tokens: int = 100):
    """Stream an NPC's dialogue with color coding"""
    color = COLORS.get(npc_name.lower(), COLORS['reset'])
    
    print(f"\n{color}[{npc_name.upper()}]:{COLORS['reset']} ", end='', flush=True)
    
    for token in client.generate_streaming(prompt, max_tokens=max_tokens, temperature=0.85):
        print(f"{color}{token}{COLORS['reset']}", end='', flush=True)
        time.sleep(0.02)  # Slight delay for dramatic effect
    
    print()

def run_tavern_scene():
    """Run a multi-NPC tavern scene demo"""
    
    print("\n" + "="*70)
    print("🍺 THE BANNERED MARE - Multi-NPC Dialogue Demo")
    print("="*70)
    print(f"{COLORS['narrator']}*The door creaks open as you enter the warm, firelit tavern*{COLORS['reset']}")
    
    client = OllamaClient(model='llama3:latest')
    
    # Scene setup
    npcs = [
        {
            'name': 'Innkeeper',
            'prompt': '''You are Hulda, the innkeeper at The Bannered Mare in Whiterun, Skyrim.
A weary traveler just entered your tavern. Greet them warmly and offer them a drink.
Speak in character, 2 sentences max. Be warm but busy.'''
        },
        {
            'name': 'Lydia', 
            'prompt': '''You are Lydia, a loyal housecarl waiting at the tavern for your Thane.
You just spotted your Thane entering the tavern. Express relief and mention you have news about a dragon sighting.
Speak in character, 2 sentences max. Be urgent but respectful.'''
        },
        {
            'name': 'Bard',
            'prompt': '''You are Mikael, the bard at The Bannered Mare.
You see a legendary dragonslayer enter. Offer to compose a song about their deeds.
Speak in character, 2 sentences max. Be flattering and musical.'''
        }
    ]
    
    # Run the scene
    for npc in npcs:
        speak(npc['name'], client, npc['prompt'], max_tokens=80)
        time.sleep(0.5)
    
    # Player response prompt
    print(f"\n{COLORS['narrator']}*The tavern awaits your response...*{COLORS['reset']}")
    print("\n" + "="*70)
    print("Demo complete! This showcases:")
    print("  ✅ Multiple NPCs with distinct personalities")
    print("  ✅ Real-time streaming from local Ollama LLM")
    print("  ✅ Color-coded dialogue for easy reading")
    print("  ✅ Low-latency response generation")
    print("="*70)

if __name__ == "__main__":
    run_tavern_scene()
