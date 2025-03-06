import re

def tokenize_text(text, tokenizer_type='word'):
    if tokenizer_type == 'letter':
        return list(text)
    elif tokenizer_type == 'word':
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens
    else:
        raise ValueError("Type de tokenizer inconnu.")

def build_vocab(tokens):
    vocab = sorted(list(set(tokens)))
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = {i: s for i, s in enumerate(vocab)}
    return stoi, itos