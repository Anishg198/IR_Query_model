# indexer.py
import math
import pickle
from collections import defaultdict, Counter
from typing import Dict
from utils import tokenize
from soundex import soundex

def build_index(corpus: Dict[str, str]):
    """
    corpus: dict {doc_id: raw_text}
    returns index_data dict containing postings, df, doc_lengths, doc_term_freqs, N, soundex_index
    """
    postings = defaultdict(list)
    doc_term_freqs = {}
    for doc_id, raw in corpus.items():
        tokens = tokenize(raw)
        tf = Counter(tokens)
        doc_term_freqs[doc_id] = tf
        for t, f in tf.items():
            postings[t].append((doc_id, f))

    df = {t: len(lst) for t, lst in postings.items()}

    # soundex index: code -> set(terms)
    soundex_index = defaultdict(set)
    for t in postings.keys():
        code = soundex(t)
        soundex_index[code].add(t)

    # doc lengths using lnc (tf weight = 1 + log10(tf)) and euclidean norm
    doc_lengths = {}
    for doc_id, tf in doc_term_freqs.items():
        acc = 0.0
        for t, f in tf.items():
            w = 1.0 + math.log10(f) if f > 0 else 0.0
            acc += w * w
        doc_lengths[doc_id] = math.sqrt(acc) if acc > 0 else 1.0

    index_data = {
        "postings": dict(postings),
        "df": df,
        "doc_term_freqs": doc_term_freqs,
        "doc_lengths": doc_lengths,
        "soundex_index": dict(soundex_index),
        "N": len(corpus)
    }
    return index_data

def save_index(index_data, path):
    with open(path, "wb") as f:
        pickle.dump(index_data, f)

def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)
