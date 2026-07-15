"""Gradio demo UI. A view over the engine: all retrieval lives in store.py /
embedder.py — this file only renders results.

Usage: python -m ragapp.app
"""
import gradio as gr

from . import store

# Each example is one beat of the live demo:
# 1. no shared keywords with its match  -> semantic, not grep
# 2. English query, Russian best match  -> cross-lingual
# 3. Uzbek query, English best match    -> cross-lingual, reversed
# 4. off-topic                          -> below threshold, honest "no match"
EXAMPLES = [
    "how do I undo a release?",
    "why is my app slow after being idle for a while?",
    "loyihani qanday o'chirib tashlayman?",
    "what's the best pizza in town?",
]


def do_search(query: str, limit: int, threshold: float):
    if not query.strip():
        return gr.update(visible=False), gr.update(visible=False)
    try:
        hits = store.search(query, limit=int(limit), threshold=threshold)
    except Exception:
        msg = (
            "**Could not reach the index.** Is Qdrant running and ingested?\n\n"
            "```\ndocker compose up -d\npython -m ragapp.ingest\n```"
        )
        return gr.update(visible=False), gr.update(value=msg, visible=True)

    if not hits:
        return (
            gr.update(visible=False),
            gr.update(value="**No confident match found.**", visible=True),
        )

    rows = [
        [
            rank,
            round(h.score, 3),
            h.question,
            h.answer if len(h.answer) <= 120 else h.answer[:117] + "...",
            ", ".join(h.tags),
        ]
        for rank, h in enumerate(hits, 1)
    ]
    return gr.update(value=rows, visible=True), gr.update(visible=False)


with gr.Blocks(title="Semantic FAQ Search") as demo:
    gr.Markdown(
        "# Semantic FAQ Search\n"
        "No keywords, no LLM — a query and 12 Q&A entries meet in the same "
        "384-dimensional vector space."
    )
    query = gr.Textbox(label="Ask in any language", placeholder="how do I undo a release?")
    with gr.Row():
        limit = gr.Slider(1, 10, value=store.DEFAULT_LIMIT, step=1, label="Max results")
        threshold = gr.Slider(0.0, 1.0, value=store.DEFAULT_THRESHOLD, step=0.01, label="Score threshold")
    results = gr.Dataframe(
        headers=["rank", "score", "question", "answer", "tags"],
        visible=False,
        interactive=False,
    )
    message = gr.Markdown(visible=False)

    inputs, outputs = [query, limit, threshold], [results, message]
    # run_on_click makes each example a complete one-click demo beat.
    gr.Examples(
        examples=[[q, store.DEFAULT_LIMIT, store.DEFAULT_THRESHOLD] for q in EXAMPLES],
        inputs=inputs,
        fn=do_search,
        outputs=outputs,
        run_on_click=True,
    )
    query.submit(do_search, inputs, outputs)
    limit.release(do_search, inputs, outputs)
    threshold.release(do_search, inputs, outputs)


def main() -> None:
    try:
        store.search("warm up")  # load model + touch Qdrant so the first live query is instant
    except Exception:
        pass  # not ingested yet — the UI shows instructions on first query
    demo.launch()


if __name__ == "__main__":
    main()
