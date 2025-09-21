
import re
import math
import heapq
import pickle
import html
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from utils import tokenize
from soundex import soundex

try:
    from nltk.corpus import wordnet as wn # type: ignore
except Exception:
    wn = None

_token_re = re.compile(r"\b[a-zA-Z]+\b")

# ----- params -----
SYNONYM_WEIGHT = 0.7
MAX_SYNS = 2
PROXIMITY_WINDOW = 8
PROXIMITY_BOOST = 1.2
TITLE_BOOST = 1.3
PHRASE_BOOST = 1.3

# ----- helpers: ensure soundex index exists -----
def ensure_soundex_index(index_data: dict):
    """Build index_data['soundex_index'] from postings if it's missing or empty."""
    if index_data is None:
        return
    if "soundex_index" in index_data and index_data["soundex_index"]:
        return
    sidx = defaultdict(list)
    postings = index_data.get("postings", {})
    for term in postings.keys():
        if term is None:
            continue
        t_low = term.lower()
        code = soundex(t_low)
        sidx[code].append(t_low)
    index_data["soundex_index"] = dict(sidx)

# ----- WordNet expansion -----
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

# ----- index helpers (case-robust) -----
def get_term_candidates(term: str, index_data: dict) -> List[str]:
    """Return vocabulary terms matching `term` exactly or by soundex (lowercased)."""
    if index_data is None:
        return []
    postings = index_data.get("postings", {})
    # try exact term and lowercase term
    if term in postings:
        return [term]
    tl = term.lower()
    if tl in postings:
        return [tl]
    # ensure soundex index
    ensure_soundex_index(index_data)
    sidx = index_data.get("soundex_index", {})
    code = soundex(tl)
    return list(sidx.get(code, []))

# ----- build query vector + track synonyms used -----
def build_query_vector(query: str, index_data: dict, synonym_weight: float = SYNONYM_WEIGHT):
    tokens = tokenize(query)
    q_tf = Counter(tokens)
    N = index_data.get("N", 1)
    df = index_data.get("df", {})

    weights: Dict[str, float] = {}
    synonym_terms = set()   # index-vocab terms we used as synonym candidates (lowercased)
    raw_synonyms = set()    # raw WordNet synonyms (lowercased) for highlighting/debug

    for qterm, raw_tf in q_tf.items():
        tf_q = 1.0 + math.log10(raw_tf) if raw_tf > 0 else 0.0

        # exact or soundex candidates for the query term
        exact_candidates = get_term_candidates(qterm, index_data)
        seen_cands = set()
        for cand in exact_candidates:
            cand_l = cand.lower()
            if cand_l in seen_cands:
                continue
            seen_cands.add(cand_l)
            df_c = df.get(cand_l, 0)
            if df_c == 0:
                continue
            idf = math.log10(N / df_c) if df_c else 0.0
            weights[cand_l] = weights.get(cand_l, 0.0) + (tf_q * idf)

        # WordNet synonyms & fallbacks
        synonyms = expand_with_wordnet(qterm)
        if synonyms:
            print(f"WordNet expansion for '{qterm}': {synonyms}")
        for syn in synonyms:
            raw_synonyms.add(syn.lower())

            # direct mapping
            syn_cands = get_term_candidates(syn, index_data)

            # substring fallback (match morphological variants)
            if not syn_cands:
                syn_low = syn.lower()
                syn_cands = [t for t in index_data.get("postings", {}).keys() if syn_low in t.lower()]
                if syn_cands:
                    print(f"  substring candidates for '{syn}': {syn_cands[:6]}{'...' if len(syn_cands)>6 else ''}")

            # soundex fallback
            if not syn_cands:
                ensure_soundex_index(index_data)
                scode = soundex(syn.lower())
                syn_cands = list(index_data.get("soundex_index", {}).get(scode, []))
                if syn_cands:
                    print(f"  soundex candidates for '{syn}': {syn_cands[:6]}{'...' if len(syn_cands)>6 else ''}")

            if not syn_cands:
                print(f"  no vocab candidates found for synonym '{syn}' (skipping)")
                continue

            # add weights for synonym-mapped terms (lowercased)
            for sc in syn_cands:
                sc_l = sc.lower()
                if sc_l in seen_cands:
                    continue
                seen_cands.add(sc_l)
                df_sc = df.get(sc_l, 0)
                if df_sc == 0:
                    continue
                idf_sc = math.log10(N / df_sc) if df_sc else 0.0
                w_add = (tf_q * idf_sc * synonym_weight)
                weights[sc_l] = weights.get(sc_l, 0.0) + w_add
                synonym_terms.add(sc_l)
                print(f"    added weight for '{sc_l}' from synonym '{syn}': idf={idf_sc:.3f} add={w_add:.4f}")

    # normalize (ltc)
    denom = math.sqrt(sum(v * v for v in weights.values())) if weights else 1.0
    if denom != 0:
        for t in list(weights.keys()):
            weights[t] = weights[t] / denom

    return weights, synonym_terms, raw_synonyms

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

# ----- ranking: returns (top_list, synonym_terms_set, raw_synonyms_set) -----
def rank_documents(query: str, index_data: dict, corpus: Dict[str, str] = None, top_k: int = 10):
    postings = index_data.get("postings", {})
    doc_lengths = index_data.get("doc_lengths", {})

    q_weights, synonym_terms, raw_synonyms = build_query_vector(query, index_data)
    if not q_weights:
        return [], set(), set()

    scores = defaultdict(float)
    for term, q_w in q_weights.items():
        if term not in postings:
            # try lowercase or other variants
            term_l = term.lower()
            if term_l not in postings:
                continue
            term = term_l
        for doc_id, tf in postings[term]:
            doc_w = 1.0 + math.log10(tf) if tf > 0 else 0.0
            denom = doc_lengths.get(doc_id, 1.0)
            doc_w_norm = doc_w / denom if denom != 0 else doc_w
            scores[doc_id] += doc_w_norm * q_w

    if not scores:
        return [], synonym_terms, raw_synonyms

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
    return top, synonym_terms, raw_synonyms

# ----- line selection & highlighting (returns rendered_headline and rendered lines) -----
def _find_line_with_match(text: str, tokens: list, return_top_n: int = 2):
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

    for i, line in enumerate(lines):
        if line.strip():
            return [(line, i + 1, set())]
    return []

def highlight_line_with_colors(text: str, query: str, synonym_terms=None, raw_synonyms=None, max_lines: int = 2):
    """
    Returns (rendered_headline, rendered_lines) where:
      - rendered_headline is headline with ANSI color escapes
      - rendered_lines is list of (line_no, rendered_line, matched_set)
    Colors:
      - green (92) = direct query tokens
      - yellow (93) = synonyms (either mapped index syn terms or raw wordnet synonyms)
    """
    if not text:
        return ("(no title)", [])

    if synonym_terms is None:
        synonym_terms = set()
    if raw_synonyms is None:
        raw_synonyms = set()

    text = html.unescape(text)
    lines = text.splitlines()
    raw_headline = "(no title)"
    for ln in lines:
        if ln.strip():
            raw_headline = ln.strip()
            break

    tokens = [t for t in tokenize(query) if t]
    top_lines = _find_line_with_match(text, tokens, return_top_n=max_lines)

    direct_terms = set(t.lower() for t in tokens)
    syn_terms = set(t.lower() for t in (synonym_terms or []))
    raw_syns = set(t.lower() for t in (raw_synonyms or []))

    # union of synonyms used for highlighting (include raw synonyms too)
    synonym_highlight_terms = syn_terms | raw_syns
    highlight_terms = list(direct_terms | synonym_highlight_terms)

    # rendered headline
    rendered_headline = raw_headline
    if highlight_terms:
        pat_head = re.compile(r"\b(" + "|".join(re.escape(t) for t in highlight_terms) + r")\b", flags=re.IGNORECASE)

        def repl_head(m):
            w = m.group(0).lower()
            if w in synonym_highlight_terms:
                return f"\033[93m{m.group(0)}\033[0m"
            else:
                return f"\033[92m{m.group(0)}\033[0m"

        rendered_headline = pat_head.sub(repl_head, raw_headline)

    rendered = []
    if not top_lines:
        rendered.append((1, raw_headline, set()))
        return (rendered_headline, rendered)

    if highlight_terms:
        pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in highlight_terms) + r")\b", flags=re.IGNORECASE)
        for line, ln_no, matched_set in top_lines:
            def repl(m):
                w = m.group(0).lower()
                if w in synonym_highlight_terms:
                    return f"\033[93m{m.group(0)}\033[0m"
                else:
                    return f"\033[92m{m.group(0)}\033[0m"
            highlighted_line = pat.sub(repl, line)
            rendered.append((ln_no, highlighted_line, matched_set))
    else:
        for line, ln_no, matched_set in top_lines:
            rendered.append((ln_no, line, matched_set))

    return (rendered_headline, rendered)

# ----- wrapper: returns results with rendered headline and lines -----
def search_and_snippet(query: str, corpus: Dict[str, str], index_data: dict, top_k: int = 10):
    toks = tokenize(query)
    if toks:
        print("soundex codes:")
        for t in toks:
            print(f"  {t} -> {soundex(t)}")
    else:
        print("soundex codes: (no tokens)")

    ranked, synonym_terms, raw_synonyms = rank_documents(query, index_data, corpus=corpus, top_k=top_k)

    # helpful debug: show raw WordNet synonyms (per query tokens)
    if raw_synonyms:
        print("WordNet raw synonyms (used for highlighting):", sorted(raw_synonyms))

    results = []
    for doc_id, score in ranked:
        raw = corpus.get(doc_id, "")
        rendered_headline, rendered_lines = highlight_line_with_colors(raw, query, synonym_terms, raw_synonyms)
        results.append((doc_id, score, rendered_headline, rendered_lines))
    return results

# ----- CLI entrypoint for single-query runs -----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="searcher cli - vsm + synonyms + soundex + highlighting")
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
        for i, (doc_id, score, rendered_headline, line_info) in enumerate(results, start=1):
            print(f"{i}. {rendered_headline} (score={score:.4f}) [{doc_id}]")
            for ln_no, rendered_line, matched_set in line_info:
                print(f"   line {ln_no}: {rendered_line}")
            print()
