"""Recreate the collection and index qa.jsonl.

Usage: python -m ragapp.ingest [path/to/qa.jsonl]
"""
import json
import sys
from pathlib import Path

from . import store


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[2] / "qa.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    c = store.client()
    store.recreate_collection(c)
    store.upsert_entries(c, entries)
    print(f"Indexed {len(entries)} entries from {path} into '{store.COLLECTION}'.")


if __name__ == "__main__":
    main()
