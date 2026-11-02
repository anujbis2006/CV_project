"""Offline character tokenizer for the debug implementation."""
from dataclasses import dataclass
@dataclass(frozen=True)
class CharTokenizer:
    vocab_size:int=260; pad_id:int=0; bos_id:int=1; eos_id:int=2; unk_id:int=3
    def encode(self, text, max_length, add_eos=True):
        ids=[ord(c)+4 if ord(c)<256 else self.unk_id for c in text]
        return (ids+([self.eos_id] if add_eos else []))[:max_length] or [self.eos_id]
    def decode(self, ids):
        return ''.join(chr(i-4) for i in ids if 4<=i<self.vocab_size)
