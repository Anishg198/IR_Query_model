# run_example.py
# demo runner for the updated searcher that returns (doc_id, score, headline, line_info)
import pickle
from searcher import search_and_snippet
import argparse

def main():
    parser = argparse.ArgumentParser(description="demo runner for searcher")
    parser.add_argument("--index", default="index.pkl", help="path to index pickle")
    parser.add_argument("--corpus", default="corpus.pkl", help="path to corpus pickle")
    parser.add_argument("--k", type=int, default=5, help="top-k results per query")
    parser.add_argument("--q", nargs="*", help="single query to run (wrap in quotes)")
    args = parser.parse_args()

    print("loading index and corpus...")
    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
    with open(args.corpus, "rb") as f:
        corpus = pickle.load(f)

    # demo queries if none provided
    queries = [
        "gold price",
        "robert",
        "oil production",
        "vegetable",
        "silver",
        " retrieval"
    ]
    if args.q:
        queries = [" ".join(args.q)]

    for q in queries:
        print("\n== query:", q)
        results = search_and_snippet(q, corpus, index_data, top_k=args.k)
        if not results:
            print("  no results")
            continue

        for rnk, (doc_id, score, headline, line_info) in enumerate(results, start=1):
            print(f"{rnk}. \033[1m{headline}\033[0m (score={score:.4f}) [{doc_id}]")
            # line_info: list of (line_no, rendered_line, matched_set)
            for ln_no, rendered_line, matched_set in line_info:
                # rendered_line already contains ANSI color escapes for matches
                print(f"   line {ln_no}: {rendered_line}")
            print()

if __name__ == "__main__":
    main()
