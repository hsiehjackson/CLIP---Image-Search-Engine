import torch
from clip.simple_tokenizer import SimpleTokenizer

class ClipTokenizer():
    "Tokenizer from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py"
    def __init__(self, bpe_path, context_length=77):
        self._tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        self.context_length = context_length
        self.vocab_size = len(self._tokenizer.encoder)

    def __call__(self:str, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text)[:self.context_length-2] + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result