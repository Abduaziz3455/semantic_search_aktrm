# Native Semantic Search — Python + Java, one shared index

A semantic search engine over a small Q&A file, implemented **twice** and fully native
each time: a pure-Python app and a pure-Java app. Neither shells out to the other. Both
read and write the **same** Qdrant collection, so an index built by one is searchable by
the other — that cross-runtime agreement is the whole point.

There is **no LLM and no answer generation**. A natural-language query returns a ranked
list of matching Q&A entries. That's the entire scope.

On top of the Python engine sits a **Gradio web app** — the surface used to demo semantic
search live to programmers of any stack.

## Why this model

We use [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
specifically because it runs natively in **both** runtimes:

- Standard BERT architecture — no `trust_remote_code`, no custom modeling code.
- The HF repo ships `onnx/model.onnx` and `tokenizer.json`, so the JVM runs it directly
  via ONNX Runtime + the DJL HuggingFace tokenizer (JNI, no Python).
- 384-dimensional output, 94 languages, MIT licence.

The "engine" is therefore a **contract**, not a library: tokenize (max 512, padded),
mask-weighted mean pooling over the last hidden state, L2 normalize, and E5's asymmetric
prefixes (`passage: ` for documents, `query: ` for queries). Follow the contract in any
language and you land in the same vector space. The parity test below proves Python and
Java agree to 1e-4 per element.

## Prerequisites

- Docker + Docker Compose (for Qdrant)
- Python 3.10+
- Java 17+ and Maven
- `curl` (for the model download)

Qdrant is shared by everything:

```bash
docker compose up -d          # REST on 6333 (Python), gRPC on 6334 (Java)
```

## Python app — first run

```bash
cd python
pip install -r requirements.txt
python -m ragapp.ingest                       # recreate 'faq' collection + index qa.jsonl
python -m ragapp.search "how do I undo a release?"
```

`search.py` takes the query as an argument, or drops into a REPL if given none. Flags:
`--limit` (default 3) and `--threshold` (default 0.80).

## Java app — first run

```bash
cd java
./download-model.sh           # curl model.onnx + tokenizer.json into models/
mvn -q compile
mvn -q exec:java -Dexec.mainClass=com.example.rag.Ingest
mvn -q exec:java -Dexec.mainClass=com.example.rag.Search -Dexec.args="how do I undo a release?"
```

## Gradio UI — the demo

```bash
cd python
python -m ragapp.app          # then open the printed http://127.0.0.1:7860
```

The UI is a thin view over the Python engine (`ragapp.store` / `ragapp.embedder`) — it
adds no retrieval logic of its own. It reads the same shared Qdrant index, so it works
whether the index was built by the **Python or the Java** app.

### Demo run-of-show

Do the one-time setup **before** the talk (`docker compose up -d` → `python -m ragapp.ingest`
→ `python -m ragapp.app`) — nobody wants to watch a model download live. Then, on stage:

1. **Show `qa.jsonl` for ten seconds.** ~12 plain Q&A entries, a few not in English. This
   is the entire "database" — no training, no fine-tuning.
2. **Click example 1** (*"how do I undo a release?"* → the rollback entry). Point out the
   top hit shares **no keywords** with the query — grep would find nothing here. That's
   semantic search in one row.
3. **Click example 2** (English query → a Russian entry). One model, one vector space, 94
   languages — nothing was translated.
4. **Click example 3** (an Uzbek query → the matching entry) for the reverse direction.
5. **Click example 4** (off-topic query). "No confident match found" — it returns nothing
   rather than nonsense, because cosine scores are comparable and thresholdable.
6. **Optional punchline:** in a terminal, run the *Java* search against the same index the
   UI is showing. Same vectors, same hits, different runtime — the engine is a contract,
   not a library.

Under 3 minutes; steps 2–5 are one click each.

## Proving both apps agree

**Cross-runtime index (the core proof).** Ingest with one app, search with the other:

```bash
# Python builds the index, Java searches it:
python -m ragapp.ingest
cd java && mvn -q exec:java -Dexec.mainClass=com.example.rag.Search -Dexec.args="how do I undo a release?"

# ...and the reverse:
mvn -q exec:java -Dexec.mainClass=com.example.rag.Ingest
cd ../python && python -m ragapp.search "how do I undo a release?"
```

**Vector parity.** Embed `passage: hello world` in both and diff the 384 floats — they
agree to within 1e-4:

```bash
# Python
cd python && python -c "from ragapp.embedder import embed_raw; print('\n'.join(f'{x:.6f}' for x in embed_raw('passage: hello world')))" > /tmp/py.txt
# Java
cd ../java && mvn -q exec:java -Dexec.mainClass=com.example.rag.Embedder -Dexec.args="passage: hello world" > /tmp/java.txt
paste /tmp/py.txt /tmp/java.txt | awk '{d=$1-$2; if (d<0) d=-d; if (d>m) m=d} END {print "max abs diff:", m}'
```

## Layout

```
qa.jsonl                       # shared data: ~12 entries, incl. Russian + Uzbek
docker-compose.yml             # Qdrant, ports 6333 (REST) + 6334 (gRPC)
python/
  requirements.txt
  ragapp/
    embedder.py                # model load + E5 prefixes + normalize (one place)
    store.py                   # collection lifecycle, upsert, search()
    ingest.py                  # python -m ragapp.ingest
    search.py                  # python -m ragapp.search "..."
    app.py                     # python -m ragapp.app  (Gradio UI)
java/
  pom.xml
  download-model.sh
  src/main/java/com/example/rag/
    Embedder.java              # ONNX Runtime + tokenizer, masked pooling by hand
    QdrantStore.java           # gRPC client
    Ingest.java / Search.java  # main() entry points
```
