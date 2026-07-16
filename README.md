# Cross-lingual Semantic Search — Python + Java, one shared index

A semantic search engine over a small **Uzbek + Russian news feed**, implemented **twice**
and fully native each time: a pure-Python app and a pure-Java app. Neither shells out to
the other. Both read and write the **same** Qdrant collection, so an index built by one is
searchable by the other — that cross-runtime agreement is the whole point.

There is **no LLM and no answer generation**. A natural-language query (in *any* language)
returns a ranked list of matching news items. Because the model is cross-lingual, an
English query finds the relevant Russian or Uzbek article, and a Russian query finds the
relevant Uzbek one.

On top of the Python engine sits a **Gradio web app** — the surface used to demo the idea
live to programmers of any stack.

![Architecture](docs/architecture.svg)

## Quick start (one command)

```bash
docker compose up --build
```

That starts Qdrant, waits until it is healthy, **auto-ingests** `qa.jsonl`, then serves the
Gradio UI at **http://localhost:7860**. First run downloads the embedding model (~470 MB)
into a cached volume, so subsequent starts are fast. Stop with `docker compose down`.

The three services (`qdrant`, `ingest`, `web`) share one Qdrant collection (`faq`). The
`ingest` service populates it once and exits; `web` waits for that to finish, then serves.

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

## Data

`qa.jsonl` holds ~16 short news items, one JSON object per line, entirely in **Uzbek and
Russian** across varied topics (technology, sport, economy, science, weather, health,
culture, transport…). Each line is `{id, question, answer, tags}` where `question` is the
headline and `answer` is the summary; both are embedded together as one chunk.

## Running the apps natively

Bring up just Qdrant (the apps connect to it on the host):

```bash
docker compose up -d qdrant
```

### Python app

```bash
cd python
pip install -r requirements.txt
python -m ragapp.ingest                                  # recreate 'faq' + index qa.jsonl
python -m ragapp.search "satellite launched for remote internet"
python -m ragapp.app                                     # or launch the Gradio UI
```

`search.py` takes the query as an argument, or drops into a REPL if given none. Flags:
`--limit` (default 3) and `--threshold` (default 0.80).

### Java app

```bash
cd java
./download-model.sh           # curl model.onnx + tokenizer.json into models/
mvn -q compile
mvn -q exec:java -Dexec.mainClass=com.example.rag.Ingest
mvn -q exec:java -Dexec.mainClass=com.example.rag.Search -Dexec.args="satellite launched for remote internet"
```

Both apps read `QDRANT_URL` / `QDRANT_HOST`+`QDRANT_PORT` from the environment, defaulting
to localhost, so the same binaries work on the host or inside compose.

## Demo run-of-show

Do the one-time setup **before** the talk (`docker compose up --build`, wait for the model
download, confirm the UI loads). Then, on stage — every step is one click of a built-in
example, no typing:

1. **Show `qa.jsonl` for ten seconds.** ~16 news items, all in Uzbek and Russian. This is
   the entire "database" — no training, no fine-tuning.
2. **Example 1** — an English query (*"satellite launched to bring internet to remote
   areas"*) surfaces the **Russian** article about a communications satellite. No shared
   keywords, different language: grep finds nothing here. That's semantic + cross-lingual
   in one row.
3. **Example 2** — an **Uzbek** query (*"shaharda yangi kasalxona ochildi"*) surfaces the
   **Russian** article about a new hospital. One model, one vector space.
4. **Example 3** — a **Russian** query (*"новая ветка метро в столице"*) surfaces the
   **Uzbek** article about a new metro line — the reverse direction.
5. **Example 4** — an off-topic query (*"recipe for a chocolate cake"*) returns
   "No confident match found": scores are comparable and thresholdable, so the system
   knows when it doesn't know.
6. **Optional punchline** — in a terminal, run the *Java* search against the same index the
   UI is showing. Same vectors, same hits, different runtime — the engine is a contract,
   not a library.

## Proving both apps agree

**Cross-runtime index (the core proof).** Ingest with one app, search with the other:

```bash
# Python builds the index, Java searches it:
cd python && python -m ragapp.ingest
cd ../java && mvn -q exec:java -Dexec.mainClass=com.example.rag.Search -Dexec.args="satellite launched for remote internet"

# ...and the reverse:
mvn -q exec:java -Dexec.mainClass=com.example.rag.Ingest
cd ../python && python -m ragapp.search "satellite launched for remote internet"
```

**Vector parity.** Embed `passage: hello world` in both and diff the 384 floats — they
agree to within 1e-4:

```bash
cd python && python -c "from ragapp.embedder import embed_raw; print('\n'.join(f'{x:.6f}' for x in embed_raw('passage: hello world')))" > /tmp/py.txt
cd ../java && mvn -q exec:java -Dexec.mainClass=com.example.rag.Embedder -Dexec.args="passage: hello world" | grep -E '^-?[0-9]' > /tmp/java.txt
paste /tmp/py.txt /tmp/java.txt | awk '{d=$1-$2; if (d<0) d=-d; if (d>m) m=d} END {print "max abs diff:", m}'
```

## Layout

```
qa.jsonl                       # shared data: ~16 news items, Uzbek + Russian
docker-compose.yml             # qdrant + auto-ingest + Gradio, one command
docs/architecture.svg          # the diagram above
python/
  Dockerfile                   # image used by the ingest + web services
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
