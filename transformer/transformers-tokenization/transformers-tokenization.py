import numpy as np
from typing import List, Dict
import re

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def _tokenize_text(self, text):
        return [t for t in re.split(r'[^\w]+', text.lower()) if t != ""]
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {
            self.pad_token : 0,
            self.unk_token : 1,
            self.bos_token : 2,
            self.eos_token : 3,
        }

        raw_text = " ".join(text.lower() for text in texts)
        preprocessed_text = self._tokenize_text(raw_text)
        postprocessed_text = sorted(set(text for text in preprocessed_text))
        for idx, text in enumerate(postprocessed_text):
            self.word_to_id[text] = idx + 4
            
        for text, idx in self.word_to_id.items():
            self.id_to_word[idx] = text

        self.vocab_size = len(self.word_to_id)
        return        
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        res = []
        preprocessed_text = self._tokenize_text(text.lower())
        for text in preprocessed_text:
            res.append(self.word_to_id.get(text, 1))

        return res
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        # res = ""
        # for id in ids:
        #     res = " ".join(self.id_to_word.get(id, self.unk_token))

        return " ".join(
            self.id_to_word.get(i, self.unk_token)
            for i in ids
        )
        
