# Python + Gradio image for ingest and the demo UI.
# Build context is the repo root (see docker-compose.yml) so we can copy qa.jsonl.
# The embedding model is NOT baked in — it is fetched once into a cached volume
# (see the `model-download` service), so this image stays small.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/python \
    HF_HOME=/hf-cache

WORKDIR /app

# CPU-only torch keeps the image small (sentence-transformers would otherwise
# pull multi-GB CUDA wheels). No GPU is needed for this model.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY python/requirements.txt python/requirements.txt
RUN pip install --no-cache-dir -r python/requirements.txt

COPY python/ python/
COPY qa.jsonl qa.jsonl

EXPOSE 7860
CMD ["python", "-m", "ragapp.app"]
