# Build Prompt: Native Semantic Search in Python and Java (with a Gradio UI)

> Paste everything below into your coding agent. It is self-contained.

---

Build a **semantic search engine over a Q&A file**, implemented **twice**: once as a
pure Python application, once as a pure Java application. Both must be fully native
to their runtime — the Java app must not shell out to Python, spawn a Python process,
or call any Python HTTP service, and vice versa. There is **no LLM and no answer
generation**. The system takes a natural-language query and returns ranked matching
Q&A entries. That is the whole scope.

On top of the two engines, ship a **Gradio web app** that puts a browser UI on the
search. It is a thin presentation layer over the Python engine — it does **not**
introduce a third code path for embedding or retrieval, and it does **not** relax the
no-cross-language rule (Gradio is Python, so it uses the Python engine natively).

## Non-negotiable constraints

1. **No cross-language runtime dependency.** Each engine stands alone. The Java app
   never touches Python; the Python app (and the Gradio UI on top of it) never touches
   the JVM.
2. **No generative model.** Retrieval only. Output is a ranked list, not prose. This
   applies to the Gradio UI too — it renders the ranked hits, it does not summarize or
   answer.
3. **One shared index.** Both apps read and write the same Qdrant collection, and
   vectors written by one must be searchable by the other. The Gradio UI reads that
   same collection.
4. **The Gradio UI adds no new embedding or search logic.** It imports and calls the
   exact same `ragapp.embedder` and `ragapp.store` functions the Python CLI uses, so
   there is a single source of truth for prefixes, pooling, normalization, and ranking.

## Fixed technical decisions — do not substitute these

**Embedding model: `intfloat/multilingual-e5-small`**

Chosen specifically because it is runnable natively in both languages:

- Standard BERT architecture — no `trust_remote_code`, no custom modeling code.
- The HF repo ships `onnx/model.onnx` and `tokenizer.json` at `main`, so the JVM can
  run it directly via ONNX Runtime. Download URLs:
  - `https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx`
  - `https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json`
- 384-dimensional output, 94 languages, MIT licence, ~470 MB fp32 ONNX.

**The model's contract — both implementations must follow it exactly:**

| Property | Value |
|---|---|
| Output dimensions | 384 |
| Max sequence length | 512 tokens (truncate) |
| Pooling | **mean pooling over the last hidden state, masked by `attention_mask`** |
| Normalization | **L2 normalize after pooling** |
| Document prefix | `passage: ` |
| Query prefix | `query: ` |

E5 is an **asymmetric** model. Documents and queries get *different* prefixes.
Omitting them or mixing them up silently degrades recall — it will not error. Put the
prefixing in exactly one function per app so it cannot drift.

**Vector DB: Qdrant** (via Docker, `qdrant/qdrant`)

- Collection name `faq`, size 384, distance **Cosine**.
- Python talks to it over **REST on port 6333**.
- Java talks to it over **gRPC on port 6334**.
- The Gradio UI talks to it over REST on 6333 (same client as the Python CLI).
- Provide a `docker-compose.yml` exposing both ports with a named volume for storage.

## Input data

A single shared `qa.jsonl` at the repo root, one JSON object per line:

```json
{"id": 1, "question": "How do I roll back a bad deploy?", "answer": "Run 'acme releases list', then 'acme rollback --to <release-id>'. Rollbacks are instant because the previous image is kept warm for 24 hours.", "tags": ["deploy", "operations"]}
```

Write ~12 entries for a fictional developer product. **Include at least two non-English
entries** (e.g. Russian, Uzbek) — the point is to demonstrate cross-lingual retrieval,
where an English query matches a Russian entry and vice versa.

**Indexing strategy:** embed `question + "\n" + answer` as one chunk per line. The
question carries the phrasing a user would type; the answer carries the vocabulary they
might search by. Store `question`, `answer`, and `tags` in the Qdrant payload; use the
JSONL `id` as the point ID.

## Deliverable 1 — Python app

Stack: `sentence-transformers`, `qdrant-client`. Nothing else for the engine.

```
python/
  requirements.txt
  ragapp/
    __init__.py
    embedder.py    # loads the model once (lru_cache), applies prefixes, normalizes
    store.py       # collection lifecycle, upsert, search
    ingest.py      # python -m ragapp.ingest [qa.jsonl]
    search.py      # python -m ragapp.search "how do I undo a release?"
    app.py         # python -m ragapp.app  -> launches the Gradio UI (Deliverable 3)
```

- `embedder.py` exports `embed_documents(list[str]) -> list[list[float]]` and
  `embed_query(str) -> list[float]`. Use `normalize_embeddings=True`.
- `store.py` exports a reusable `search(query_text, limit, threshold) -> list[Hit]`
  (or equivalent) so both `search.py` and `app.py` call the same function — the Gradio
  UI must not re-implement ranking or threshold logic.
- `ingest.py` recreates the collection and upserts with `wait=True`.
- `search.py` takes the query as argv, or drops into a REPL loop if no argv is given.

## Deliverable 2 — Java app

Stack (Maven, Java 17+):

- `com.microsoft.onnxruntime:onnxruntime` — runs `model.onnx` in-process.
- `ai.djl.huggingface:tokenizers` — loads `tokenizer.json` natively (JNI, no Python).
- `io.qdrant:client` — official gRPC client.
- `com.fasterxml.jackson.core:jackson-databind` — parse the JSONL.

```
java/
  pom.xml
  download-model.sh          # curl the two files into models/multilingual-e5-small/
  src/main/java/com/example/rag/
    Embedder.java            # implements AutoCloseable
    QdrantStore.java
    Ingest.java              # main()
    Search.java              # main(), argv query or REPL
```

`Embedder.java` requirements:

- Build the tokenizer with `HuggingFaceTokenizer.builder().optTokenizerPath(...)`,
  padding on, truncation on, max length 512. Use `batchEncode` for ingest.
- **Inspect `session.getInputNames()` at runtime and only bind the inputs the graph
  actually declares.** This export may or may not require `token_type_ids` — do not
  hardcode the assumption. Feed a zero tensor for it only if the name is present.
- Feed `long[]` tensors shaped `[batch, seq]` via `LongBuffer`.
- Read the last hidden state as `float[][][]` shaped `[batch, seq, 384]`, then do
  mask-weighted mean pooling and L2 normalization **by hand**. Do not skip the mask —
  averaging over padding tokens corrupts the vector.
- Reuse a single `OrtEnvironment` and `OrtSession`; do not create one per call.

## Deliverable 3 — Gradio web app (the demo surface)

A browser UI over the Python engine. Pure Python, no new retrieval logic.

**Purpose: this is what gets shown live in a presentation to programmers of any
stack.** The audience should understand what semantic search does within one query —
type a question in plain language, get ranked matches with scores — without needing to
know Python, Java, or embeddings. Optimize for legibility on a projector, not for
features.

Simplicity budget: **one file, one screen, no tabs, no custom CSS, no state.** If a
feature needs explaining before the demo makes sense, cut it. Target well under ~100
lines.

Stack: add `gradio` to `python/requirements.txt`. Nothing else.

Location: `python/ragapp/app.py`, launched with `python -m ragapp.app`.

Behaviour:

- **Reuses the engine, doesn't reimplement it.** `app.py` imports `embed_query` /
  the shared `search()` from `ragapp.store` (which uses `ragapp.embedder`). It must not
  contain its own prefixing, pooling, normalization, or Qdrant-ranking code. This keeps
  vector parity (acceptance criterion 3) intact through the UI.
- **Model loads once.** Rely on the existing `lru_cache` in `embedder.py`; do not load
  the model per request. Optionally warm it at startup so the first query is fast.
- **Inputs:** a query textbox (submit on Enter), a `limit` slider (1–10, default 3),
  and a score-`threshold` slider (0.0–1.0, default matching the CLI apps' default).
- **Outputs:** a results table (Gradio `Dataframe`) with columns
  `rank, score, question, answer, tags`, sorted by score descending. Round scores to
  3 dp and truncate answers to ~120 chars — the table must stay readable on a
  projector. When the top score is below the threshold, show a clear
  "No confident match found" message instead of a table of weak hits.
- **Examples:** wire up 3–4 `gr.Examples` — these ARE the demo script, so choose them
  deliberately:
  1. A query that shares **no keywords** with its best match (e.g. "undo a release" →
     the rollback entry) — proves it's semantic, not keyword search. This is the
     one-click opener.
  2. An English query whose best match is the Russian or Uzbek entry — the
     cross-lingual moment.
  3. A non-English query matching an English entry — the reverse.
  4. An off-topic query (e.g. "best pizza in town") that lands below the threshold —
     shows the system knows when it doesn't know.
- **Graceful failure:** if the Qdrant collection is missing/empty (i.e. ingest hasn't
  run), catch it and render a friendly message telling the user to run
  `python -m ragapp.ingest` first — do not dump a raw stack trace into the UI.
- **Launch:** `demo.launch()` binding to `127.0.0.1` (expose `server_name="0.0.0.0"`
  and read the port from an env var only if a comment explains why). Print the local URL.
- **No `share=True` by default.** Keep it local unless the user opts in.

The Gradio app is a *view*. Anything that changes retrieval quality lives in
`embedder.py` / `store.py`, never in `app.py`.

## Acceptance criteria

The build is done when all of these pass:

1. `docker compose up -d`, then `python -m ragapp.ingest`, then
   `mvn -q exec:java -Dexec.mainClass=com.example.rag.Search -Dexec.args="how do I undo a release?"`
   returns the rollback entry as the top hit. **The Java app searching an index built
   by the Python app is the core proof that both are native and agree.**
2. The reverse also works: ingest with Java, search with Python.
3. **Vector parity test.** Embed the string `passage: hello world` in both apps, print
   all 384 floats. They must agree to within `1e-4` per element. If they do not, the
   cause is almost always one of: a missing/wrong prefix, pooling without the attention
   mask, or a forgotten L2 normalize. Fix it rather than loosening the tolerance.
4. Cross-lingual check: an English query returns the Russian or Uzbek entry when that
   entry is the best semantic match.
5. Both CLI apps print results as `rank, score, question, answer-snippet`, sorted by
   score descending, with a `--limit` (default 3) and a score threshold below which they
   say no confident match was found.
6. **Gradio UI works end-to-end.** With Qdrant up and the collection ingested,
   `python -m ragapp.app` launches, opens in a browser, and a natural-language query
   returns the same ranked hits the Python CLI returns for that query (same order, same
   scores). The `limit` and `threshold` controls take effect, and a below-threshold query
   shows "No confident match found". **All four built-in examples produce their
   scripted outcome on one click** (no-keyword-overlap match, cross-lingual both ways,
   below-threshold rejection) — the demo must not require typing during the talk. The
   UI reuses the engine — searching the same shared Qdrant collection that Java wrote
   (criterion 1) also works through the UI.

## Demo flow (run-of-show for the presentation)

Put this in the README as a "Demo" section. It is the exact sequence to perform live,
in order, after the one-time setup (`docker compose up -d` → `python -m ragapp.ingest`
→ `python -m ragapp.app`, done *before* the talk — nobody watches a model download):

1. **Show `qa.jsonl` for ten seconds.** ~12 plain Q&A entries, a few not in English.
   This is the entire "database". No training, no fine-tuning.
2. **Click example 1** ("undo a release" → rollback entry). Point at the top hit:
   *no shared keywords* — grep would find nothing here. That's the whole pitch of
   semantic search in one row.
3. **Click example 2** (English query → Russian/Uzbek entry). One model, one vector
   space, 94 languages — nothing was translated.
4. **Click example 4** (off-topic query). "No confident match found" — it returns
   nothing rather than nonsense, because scores are comparable and thresholdable.
5. **Optional, for the polyglot punchline:** in a terminal, run the *Java* search
   against the same index the UI is using. Same vectors, same hits, different runtime —
   the engine is a contract (model + pooling + prefixes), not a library.

Total demo time: under 3 minutes. Steps 2–4 are one click each — that is why the
examples are specified so precisely in Deliverable 3.

## Output style

Include a top-level `README.md` with: prerequisites, the exact command sequence for
first run for each app, and a short "why this model" note explaining the ONNX/native
constraint.

- **Python CLI:** compose up → ingest → search.
- **Java CLI:** compose up → download model → ingest → search.
- **Gradio UI:** compose up → ingest (Python) → `python -m ragapp.app` → open the
  printed URL. Note that it shares the same Qdrant index, so an index built by *either*
  the Python or Java app is searchable from the UI.

Comment the non-obvious parts only — the E5 prefixes, the masked pooling, the runtime
input-name inspection, and the fact that `app.py` deliberately delegates all retrieval
to `store.py`/`embedder.py`. Skip comments that restate the code.
