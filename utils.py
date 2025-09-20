# utils.py
import re
from collections import Counter
import nltk

_token_re = re.compile(r"\b[a-zA-Z]+\b")

def tokenize(text):
    """lowercase alphabetic tokenizer"""
    return [t.lower() for t in _token_re.findall(text)]

def simple_clean_join(tokens):
    """join tokens to single string (useful for storing cleaned doc text)"""
    return " ".join(tokens)

def load_reuters_corpus(limit=None, keep_raw=True):
    """
    loads nltk reuters corpus into dict {docid: raw_text}
    if limit provided, loads first `limit` fileids.
    requires nltk.download('reuters') to be run beforehand.
    """
    from nltk.corpus import reuters
    fileids = reuters.fileids()
    if limit:
        fileids = fileids[:limit]
    corpus = {fid: reuters.raw(fid) for fid in fileids}
    if keep_raw:
        return corpus
    # else return cleaned text
    cleaned = {}
    for fid, raw in corpus.items():
        toks = tokenize(raw)
        cleaned[fid] = simple_clean_join(toks)
    return cleaned

def save_text_file(path, text):
    with open(path, "w", encoding="utf8") as f:
        f.write(text)

def read_text_file(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read()
