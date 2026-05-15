import hashlib
from collections import Counter
from kiwipiepy import Kiwi
from qdrant_client import models

kiwi = Kiwi()

def get_stable_hash(word: str) -> int:
    hash_bytes = hashlib.md5(word.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')

def embed_sparse_text(text: str) -> models.SparseVector:
    tokens = [
        token.form for token in kiwi.tokenize(text) 
        if token.tag.startswith(('N', 'V', 'R')) and len(token.form) > 1
    ]
    
    token_counts = Counter(tokens)
    
    sparse_dict = {}
    for token, count in token_counts.items():
        idx = get_stable_hash(token)
        sparse_dict[idx] = sparse_dict.get(idx, 0.0) + count
        
    return models.SparseVector(
        indices=list(sparse_dict.keys()),
        values=list(sparse_dict.values())
    )