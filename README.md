# 🔎 VSM Search Engine (lnc.ltc + Soundex + WordNet)

This project implements a **Vector Space Model (VSM)** search engine over the [Reuters-21578 dataset](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection).  
It supports **interactive querying** with rich highlighting, synonym expansion, and fuzzy matching.

---

## 📐 Retrieval Model

We use the **lnc.ltc weighting scheme**:

- **Document weight (lnc)**:  
  \[
  w_{d,t} = \frac{1 + \log_{10}(tf_{d,t})}{|d|}
  \]

- **Query weight (ltc)**:  
  \[
  w_{q,t} = (1 + \log_{10}(tf_{q,t})) \cdot \log_{10}\frac{N}{df_t}
  \]

- **Synonyms (WordNet)**: scaled by **0.7** of direct query term weight.  
- **Soundex-mapped terms** (typos → canonical terms): treated as full-weight direct terms.  
- **Final score**: cosine similarity between query and document vectors.

---

## ✨ Features

- **Core model**: Vector Space Model with lnc.ltc weighting.  
- **Soundex fuzzy matching**: query typos map to canonical terms (e.g., `fues` → `fuse`).  
- **WordNet synonym expansion**: expands query with semantic synonyms (reduced weight).  
- **Boosts**:
  - **Title boost** (+30%) if query terms appear in the headline.
  - **Phrase boost** (+30%) if query phrase occurs verbatim.
  - **Proximity boost** (+20%) if query terms appear near each other (window ≤ 8).
- **Output formatting**:
  - Top **5 documents** shown per query.
  - Top **2 relevant lines** (snippets) shown per document.
  - **Headlines**: always bold (yellow in interactive mode).
  - **Synonyms**: highlighted in green.
- **Interactive search**: type queries live, get ranked results instantly.

---

## 📂 Project Structure


---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install nltk colorama

#Then download WordNet (only once):
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
