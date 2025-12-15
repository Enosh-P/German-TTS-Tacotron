"""
German Text Symbols for TTS
Defines all characters and special tokens used in the system
"""

# Special tokens
_pad = '_'
_eos = '~'
_bos = '^'

# German letters (including umlauts)
_letters = 'abcdefghijklmnopqrstuvwxyzäöüß'
_letters_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ'

# Punctuation
_punctuation = '!\'(),.:;? '
_special = '-'

# Combine all symbols
symbols = [_pad, _bos, _eos] + list(_letters) + list(_letters_upper) + list(_punctuation) + list(_special)

# Create mapping dictionaries
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Export useful constants
PAD_ID = symbol_to_id[_pad]
BOS_ID = symbol_to_id[_bos]
EOS_ID = symbol_to_id[_eos]


