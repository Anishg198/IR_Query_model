# 🔎 VSM Search Engine (lnc.ltc + Soundex + WordNet) for IR assignment

This assignment implements a **Vector Space Model (VSM)** search engine over the [Reuters-21578 dataset](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection).  
It supports **interactive querying** with highlighting, synonym expansion, and fuzzy matching.

## 📐 Retrieval Model

We use the **lnc.ltc weighting scheme**:

- **Document weight (lnc):**
w(d,t) = (1 + log10(tf_d,t)) / |d|

- **Query weight (ltc):**
w(q,t) = (1 + log10(tf_q,t)) * log10(N / df_t)

- **Synonyms (WordNet):** scaled by **0.7** of direct query term weight.  
- **Soundex-mapped terms (typos → canonical terms):** treated as full-weight direct terms.  
- **Final score:** cosine similarity between query and document vectors.

## ✨ Features

- **Core model:** Vector Space Model with lnc.ltc weighting.  
- **Soundex fuzzy matching:** query typos map to canonical terms (e.g., `fues` → `fuse`).  
- **WordNet synonym expansion:** expands query with semantic synonyms (reduced weight).  
- **Boosts:**
  - **Title boost** (+30%) if query terms appear in the headline.  
  - **Phrase boost** (+30%) if query phrase occurs verbatim.  
  - **Proximity boost** (+20%) if query terms appear near each other (window ≤ 8).  
- **Output formatting:**
  - Top **5 documents** shown per query.  
  - Top **2 relevant lines** (snippets) shown per document.  
  - **Headlines:** bold.  
  - **Matching words:** green.  
  - **Synonyms:** green.  
- **Interactive search:** type queries live, get ranked results instantly.

## 📂 Project Structure

IR_Query_model<br>

| File/Folder      | Description                                      |
|------------------|--------------------------------------------------|
| indexer.py       | builds index.pkl and corpus.pkl                   |
| searcher.py      | core retrieval engine (VSM + Soundex + WordNet)   |
| interactive.py   | interactive console to type queries               |
| run_example.py   | example run with preset queries                   |
| utils.py         | tokenizer and helper functions                    |
| soundex.py       | Soundex implementation                            |
| index.pkl        | built index (after running indexer)               |
| corpus.pkl       | serialized corpus (after running indexer)         |
| README.md        | this file                                         |


## 🚀 Running model

1. Install dependencies
   pip install nltk colorama

   Then download WordNet (only once):
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')

2. Build index
   python indexer.py

3. Run interactive search
   python interactive.py

## 💻 Sample Session

🔎 interactive VSM searcher
type a query and press enter (type 'exit' or 'quit' to stop)

query > gold price

1. BELGIUM LAUNCHES BONDS WITH GOLD WARRANTS (score=0.2646) [test/15471]
   line 3: The Kingdom of Belgium is launching 100 mln Swiss francs of seven ...

2. CRA SOLD FORREST GOLD FOR 76 MLN DLRS (score=0.2139) [test/14865]
   line 1: CRA sold Forrest GOLD for 76 mln dollars ...

## 🆚 Novelty Beyond Assignment Requirements

Compared to a basic VSM implementation, this project adds:

- **Soundex Fuzzy Matching** → handles typos and spelling variations.  
- **WordNet Synonym Expansion** → improves recall by retrieving semantically related docs.  
- **Boosting Heuristics** → titles, phrases, and proximity are rewarded.  
- **Color-coded Output** → improves readability in terminal:  
  - Headlines = **bold**  
  - Matching words = **🟩 green**  
  - Synonyms = **🟨 yellow**  
- **Interactive Console** → lets you search dynamically without rerunning scripts.  
- **Line-level Snippets** → shows the most relevant lines instead of whole documents.  

## 👨‍💻 Authors

Built by *Anish Gupta*, *Prakhar Sethi* and *Ritwik Bhattacharya* as part of **Lab 1 — VSM Information Retrieval** assignment.
