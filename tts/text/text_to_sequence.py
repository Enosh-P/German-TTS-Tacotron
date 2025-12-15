"""
Text to Sequence Conversion for German TTS
"""
import re
from .symbols import symbol_to_id, id_to_symbol, BOS_ID, EOS_ID, PAD_ID

# German text normalization patterns
_abbreviations = {
    'Dr.': 'Doktor',
    'Prof.': 'Professor',
    'St.': 'Sankt',
    'Nr.': 'Nummer',
    'Str.': 'Straße',
    'z.B.': 'zum Beispiel',
    'u.a.': 'unter anderem',
    'd.h.': 'das heißt',
}

def normalize_text(text):
    """Normalize German text"""
    # Convert to lowercase
    text = text.lower()
    
    # Expand abbreviations
    for abbr, expansion in _abbreviations.items():
        text = text.replace(abbr.lower(), expansion.lower())
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up punctuation spacing
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    return text.strip()

def text_to_sequence(text, add_bos=True, add_eos=True):
    """
    Convert text to sequence of symbol IDs
    
    Args:
        text: Input text string
        add_bos: Add beginning-of-sequence token
        add_eos: Add end-of-sequence token
    
    Returns:
        List of integer IDs
    """
    text = normalize_text(text)
    sequence = []
    
    if add_bos:
        sequence.append(BOS_ID)
    
    for char in text:
        if char in symbol_to_id:
            sequence.append(symbol_to_id[char])
        else:
            # Skip unknown characters
            continue
    
    if add_eos:
        sequence.append(EOS_ID)
    
    return sequence

def sequence_to_text(sequence):
    """
    Convert sequence of IDs back to text
    
    Args:
        sequence: List of integer IDs
    
    Returns:
        Text string
    """
    
    result = []
    for symbol_id in sequence:
        if symbol_id in [BOS_ID, EOS_ID, PAD_ID]:
            continue
        if symbol_id in id_to_symbol:
            result.append(id_to_symbol[symbol_id])
    
    return ''.join(result)

