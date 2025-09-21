# interactive.py
# simple interactive console for querying the VSM system

import pickle
from searcher import search_and_snippet

def main():
    # load index + corpus once
    with open("index.pkl", "rb") as f:
        index_data = pickle.load(f)
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    print("🔎 interactive VSM searcher")
    print("type a query and press enter (type 'exit' or 'quit' to stop)")
    print("-" * 50)

    while True:
        query = input("\nquery > ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("bye 👋")
            break

        results = search_and_snippet(query, corpus, index_data, top_k=5)
        if not results:
            print("  no results found")
            continue

        for i, (doc_id, score, headline, line_info) in enumerate(results, start=1):
            print(f"\n{i}. \033[1m{headline}\033[0m (score={score:.4f}) [{doc_id}]")
            for ln_no, rendered_line, matched_set in line_info:
                print(f"   line {ln_no}: {rendered_line}")

if __name__ == "__main__":
    main()
