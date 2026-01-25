
import re

def _clean_for_tts(text: str) -> str:
    print(f"Original: '{text}'")
    
    # Current logic reproduction
    text = re.sub(r'\([^\)]+\)', '', text)
    text = re.sub(r'\*[^\*]+\*', '', text)
    
    text = text.replace('!', '.')
    text = text.replace('?', '.') 
    text = text.replace(':', ' ')
    text = text.replace(';', ' ')
    text = text.replace('...', ' ')
    
    text = ' '.join(text.split())
    print(f"Cleaned:  '{text}'")
    return text

test_input = "(smiling politely) Hello!I'm here to ensure your visit to The Velvet Room is enjoyable and memorable.Is there something specific you'd like to know or discuss?"

_clean_for_tts(test_input)
