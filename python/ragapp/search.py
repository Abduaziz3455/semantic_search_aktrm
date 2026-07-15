"""Search the index from the command line.

Usage: python -m ragapp.search "how do I undo a release?" [--limit N] [--threshold T]
With no query argument, drops into a REPL.
"""
import argparse

from . import store


def run_query(query: str, limit: int, threshold: float) -> None:
    hits = store.search(query, limit=limit, threshold=threshold)
    if not hits:
        print("No confident match found.")
        return
    for rank, h in enumerate(hits, 1):
        snippet = h.answer if len(h.answer) <= 120 else h.answer[:117] + "..."
        print(f"{rank}. [{h.score:.4f}] {h.question}")
        print(f"   {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over the FAQ index.")
    parser.add_argument("query", nargs="?", help="query text; omit for a REPL")
    parser.add_argument("--limit", type=int, default=store.DEFAULT_LIMIT)
    parser.add_argument("--threshold", type=float, default=store.DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if args.query:
        run_query(args.query, args.limit, args.threshold)
        return

    print("Semantic search REPL — empty line to exit.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        run_query(query, args.limit, args.threshold)


if __name__ == "__main__":
    main()
