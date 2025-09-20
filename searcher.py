# searcher.py
# final tuned searcher: vsm (lnc.ltc) + light wordnet + soundex fallback
# phrase/title/proximity boosts, headline dedupe, multi-line highlighting (green tokens)

import re
import math
import heapq
import pickle
import html
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# project helpers (must exist in project)
from utils import tokenize
from soundex import soundex

# optional wordnet (download via nltk if you use expansion)
try:
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None

# token regex for scanning
_token_re = re.compile(r"\b[a-zA-Z]+\b")

# ----- tuneable params -----
SYNONYM_WEIGHT = 0.7
MAX_SYNS = 2
PROXIMITY_WINDOW = 8
PROXIMITY_BOOST = 1.5
TITLE_BOOST = 1.3
PHRASE_BOOST = 1.3
DEDUP_CHARS = 250

# ----- wordnet expansion -----
def expand_with_wordnet(term: str, max_synonyms: int = MAX_SYNS) -> List[str]:
    syns = []
    if wn is None:
        return syns
    try:
        for syn in wn.synsets(term):
            for lemma in syn.lemmas():
                name = lemma.name().lower().replace("_", " ")
                if " " in name:
                    continue
                if name == term:
                    continue
                if name not in syns:
                    syns.append(name)
                if len(syns) >= max_synonyms:
                    break
            if len(syns) >= max_synonyms:
                break
    except Exception:
        return []
    return syns

# ----- index helpers -----
def get_term_candidates(term: str, index_data: dict) -> List[str]:
    postings = index_data["postings"]
    if term in postings:
        return [term]
    code = soundex(term)
    return list(index_data["soundex_index"].get(code, []))

def build_query_vector(query: str, index_data: dict, synonym_weight: float = SYNONYM_WEIGHT) -> Dict[str, float]:
    tokens = tokenize(query)
    q_tf = Counter(tokens)
    N = index_data["N"]
    df = index_data["df"]

    weights: Dict[str, float] = {}
    for qterm, raw_tf in q_tf.items():
        tf_q = 1.0 + math.log10(raw_tf) if raw_tf > 0 else 0.0

        exact_candidates = get_term_candidates(qterm, index_data)
        seen_cands = set()
        for cand in exact_candidates:
            if cand in seen_cands:
                continue
            seen_cands.add(cand)
            df_c = df.get(cand, 0)
            if df_c == 0:
                continue
            idf = math.log10(N / df_c)
            weights[cand] = weights.get(cand, 0.0) + (tf_q * idf)

        synonyms = expand_with_wordnet(qterm)
        for syn in synonyms:
            syn_cands = get_term_candidates(syn, index_data)
            for sc in syn_cands:
                if sc in seen_cands:
                    continue
                seen_cands.add(sc)
                df_sc = df.get(sc, 0)
                if df_sc == 0:
                    continue
                idf_sc = math.log10(N / df_sc)
                weights[sc] = weights.get(sc, 0.0) + (tf_q * idf_sc * synonym_weight)

    denom = math.sqrt(sum(v * v for v in weights.values())) if weights else 1.0
    if denom != 0:
        for t in list(weights.keys()):
            weights[t] = weights[t] / denom
    return weights

# ----- optional doc vector helper (for rocchio if used) -----
def get_doc_vector(doc_id: str, index_data: dict) -> Dict[str, float]:
    tf_map = index_data["doc_term_freqs"].get(doc_id, {})
    length = index_data["doc_lengths"].get(doc_id, 1.0)
    vec = {}
    for term, tf in tf_map.items():
        if tf <= 0:
            continue
        w = 1.0 + math.log10(tf)
        vec[term] = (w / length) if length != 0 else w
    return vec

# ----- proximity helper -----
def _has_proximity(tokens: List[str], doc_text: str, window: int = PROXIMITY_WINDOW) -> bool:
    if len(tokens) < 2:
        return False
    words = _token_re.findall(doc_text.lower())
    pos_map = {}
    for idx, w in enumerate(words):
        if w in pos_map:
            pos_map[w].append(idx)
        elif w in tokens:
            pos_map[w] = [idx]
    tokens_present = [t for t in tokens if t in pos_map]
    for i in range(len(tokens_present)):
        for j in range(i + 1, len(tokens_present)):
            a = tokens_present[i]; b = tokens_present[j]
            for pa in pos_map[a]:
                for pb in pos_map[b]:
                    if abs(pa - pb) <= window:
                        return True
    return False

# ----- main ranking: lnc docs, ltc query, boosts, dedupe by headline -----
def rank_documents(query: str, index_data: dict, corpus: Dict[str, str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
    postings = index_data["postings"]
    doc_lengths = index_data["doc_lengths"]

    q_weights = build_query_vector(query, index_data)
    if not q_weights:
        return []

    scores = defaultdict(float)
    for term, q_w in q_weights.items():
        if term not in postings:
            continue
        for doc_id, tf in postings[term]:
            doc_w = 1.0 + math.log10(tf) if tf > 0 else 0.0
            denom = doc_lengths.get(doc_id, 1.0)
            doc_w_norm = doc_w / denom if denom != 0 else doc_w
            scores[doc_id] += doc_w_norm * q_w

    if not scores:
        return []

    tokens = tokenize(query)
    tokens_lower = [t.lower() for t in tokens]
    bigrams = set(" ".join(bg) for bg in zip(tokens_lower, tokens_lower[1:])) if len(tokens_lower) > 1 else set()

    if corpus:
        for doc_id in list(scores.keys()):
            text = corpus.get(doc_id, "").lower()
            first_chunk = text.splitlines()[0] if text.splitlines() else text[:200]
            if any(t in first_chunk for t in tokens_lower):
                scores[doc_id] *= TITLE_BOOST
            for phrase in bigrams:
                if phrase in text:
                    scores[doc_id] *= PHRASE_BOOST
                    break
            if _has_proximity(tokens_lower, text, window=PROXIMITY_WINDOW):
                scores[doc_id] *= PROXIMITY_BOOST

    # dedupe by headline
    heap = [(-score, doc_id) for doc_id, score in scores.items()]
    heapq.heapify(heap)

    headline_map = {}
    other_docs = []
    while heap:
        neg_score, doc_id = heapq.heappop(heap)
        score = -neg_score
        raw = corpus.get(doc_id, "") if corpus else ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        headline = lines[0] if lines else ""
        headline_key = re.sub(r"[^\w\s]", "", headline.lower()).strip()
        headline_key = re.sub(r"\s+", " ", headline_key)
        if headline_key:
            existing = headline_map.get(headline_key)
            if (existing is None) or (score > existing[0]):
                headline_map[headline_key] = (score, doc_id)
        else:
            other_docs.append((score, doc_id))

    candidates = sorted(headline_map.values(), key=lambda x: -x[0])
    other_docs.sort(key=lambda x: -x[0])
    combined = candidates + other_docs

    top = []
    seen_docids = set()
    for score, doc_id in combined:
        if doc_id in seen_docids:
            continue
        top.append((doc_id, score))
        seen_docids.add(doc_id)
        if len(top) >= top_k:
            break

    top.sort(key=lambda x: (-x[1], x[0]))
    return top

# ----- improved line selection + multi-token highlighting -----
def _find_line_with_match(text: str, tokens: list, return_top_n: int = 2):
    """
    Return up to return_top_n best lines with matched token sets:
    - prefer exact phrase (immediate best)
    - else rank lines by number of distinct query tokens present
    - soundex fallback if no direct hits
    """
    if not text:
        return []

    text = html.unescape(text)
    lines = text.splitlines()
    tokens_lower = [t.lower() for t in tokens if t]

    phrase = " ".join(tokens_lower) if len(tokens_lower) > 1 else None
    scored = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        line_low = line.lower()
        if phrase and phrase in line_low:
            matched = set(tokens_lower) & set(_token_re.findall(line_low))
            scored.append((10 + len(matched), line, i + 1, matched))
            continue
        matched = set()
        for t in set(tokens_lower):
            if re.search(rf"\b{re.escape(t)}\b", line, flags=re.IGNORECASE):
                matched.add(t)
        if matched:
            scored.append((len(matched), line, i + 1, matched))

    if scored:
        scored.sort(key=lambda x: (-x[0], x[2]))
        return [(line, idx, matched) for score, line, idx, matched in scored[:return_top_n]]

    token_soundexes = {t: soundex(t) for t in tokens_lower}
    for i, line in enumerate(lines):
        for wmatch in _token_re.finditer(line):
            w = wmatch.group(0)
            w_code = soundex(w)
            for t, code in token_soundexes.items():
                if code == w_code:
                    return [(line, i + 1, {t})]

    for i, line in enumerate(lines):
        if line.strip():
            return [(line, i + 1, set())]
    return []

def highlight_line_with_colors(text: str, query: str, max_lines: int = 2):
    """
    Return (headline, rendered_lines) where rendered_lines is list of
    (line_no, rendered_line, matched_tokens_set).
    Highlight matched tokens in green; rest remains default color.
    """
    if not text:
        return ("(no title)", [])

    text = html.unescape(text)
    lines = text.splitlines()
    headline = "(no title)"
    for ln in lines:
        if ln.strip():
            headline = ln.strip()
            break

    tokens = [t for t in tokenize(query) if t]
    top_lines = _find_line_with_match(text, tokens, return_top_n=max_lines)

    rendered = []
    if not top_lines:
        rendered.append((1, headline, set()))
        return (headline, rendered)

    if tokens:
        pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b", flags=re.IGNORECASE)
        for line, ln_no, matched_set in top_lines:
            def repl(m):
                return f"\033[92m{m.group(0)}\033[0m"
            highlighted_line = pat.sub(repl, line)
            rendered.append((ln_no, highlighted_line, matched_set))
    else:
        for line, ln_no, matched_set in top_lines:
            rendered.append((ln_no, line, matched_set))

    return (headline, rendered)

# ----- wrapper: search + return headline + line info -----
def search_and_snippet(query: str, corpus: Dict[str, str], index_data: dict, top_k: int = 10):
    toks = tokenize(query)
    if toks:
        print("soundex codes:")
        for t in toks:
            print(f"  {t} -> {soundex(t)}")
    else:
        print("soundex codes: (no tokens)")
    for t in toks:
        syns = expand_with_wordnet(t)
        if syns:
            print(f"WordNet expansion for '{t}': {syns}")


    ranked = rank_documents(query, index_data, corpus=corpus, top_k=top_k)
    results = []
    for doc_id, score in ranked:
        raw = corpus.get(doc_id, "")
        headline, rendered_lines = highlight_line_with_colors(raw, query)
        results.append((doc_id, score, headline, rendered_lines))
    return results

# ----- CLI -----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="searcher cli - vsm with soundex/wordnet + headline + line highlighting")
    parser.add_argument("--index", required=True, help="path to index pickle (index.pkl)")
    parser.add_argument("--corpus", required=True, help="path to corpus pickle (corpus.pkl)")
    parser.add_argument("--q", required=True, help="query string (wrap in quotes)")
    parser.add_argument("--k", type=int, default=10, help="top-k results")
    args = parser.parse_args()

    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
    with open(args.corpus, "rb") as f:
        corpus = pickle.load(f)

    results = search_and_snippet(args.q, corpus, index_data, top_k=args.k)
    if not results:
        print("no results")
    else:
        for i, (doc_id, score, headline, line_info) in enumerate(results, start=1):
            print(f"{i}. \033[1m{headline}\033[0m (score={score:.4f}) [{doc_id}]")
            for ln_no, rendered_line, matched_set in line_info:
                print(f"   line {ln_no}: {rendered_line}")
            print()
