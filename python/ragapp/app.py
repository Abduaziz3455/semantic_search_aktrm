"""Gradio demo UI. A view over the engine: all retrieval lives in store.py /
embedder.py — this file only renders results.

Usage: python -m ragapp.app
"""
import gradio as gr

from . import store

# The news feed is entirely Uzbek + Russian. Each example is one demo beat:
# 1. English query      -> Russian article  (cross-lingual, zero shared words)
# 2. Uzbek query        -> Russian article  (cross-lingual between the two langs)
# 3. Russian query      -> Uzbek article    (the reverse direction)
# 4. off-topic          -> below threshold, honest "no match"
EXAMPLES = [
    "satellite launched to bring internet to remote areas",
    "shaharda yangi kasalxona ochildi",
    "новая ветка метро в столице",
    "recipe for a chocolate cake",
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
            h.answer if len(h.answer) <= 160 else h.answer[:157] + "...",
            ", ".join(h.tags),
        ]
        for rank, h in enumerate(hits, 1)
    ]
    return gr.update(value=rows, visible=True), gr.update(visible=False)


with gr.Blocks(title="Semantic FAQ Search") as demo:
    gr.Markdown(
        "# Cross-lingual News Search\n"
        "No keywords, no LLM. The news feed is written in **Uzbek and Russian** — "
        "search in any language and your query meets the articles in the same "
        "1024-dimensional vector space."
    )
    query = gr.Textbox(label="Search the news (any language)",
                       placeholder="satellite launched to bring internet to remote areas")
    with gr.Row():
        limit = gr.Slider(1, 10, value=store.DEFAULT_LIMIT, step=1, label="Max results")
        threshold = gr.Slider(0.0, 1.0, value=store.DEFAULT_THRESHOLD, step=0.01, label="Score threshold")
    results = gr.Dataframe(
        headers=["rank", "score", "headline", "summary", "topics"],
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
