FROM python:3.11-slim as builder

#RUN apk add --no-cache build-base linux-headers
RUN pip install poetry

WORKDIR /app

# Install split into two steps (the dependencies and the sources)
# in order to leverage the Docker caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root --no-directory --only main

COPY . .
RUN poetry install --no-interaction --no-ansi --no-cache --only main

# nltk 3.9 actually uses punkt_tab and averaged_perceptron_tagger_eng
# but we have to download punkt and averaged_perceptron_tagger as well, because unstructured will try to download it if missing
RUN . ./.venv/bin/activate && python -m nltk.downloader -d /usr/share/nltk_data stopwords punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng

ENV BGE_EMBEDDINGS_MODEL_PATH=/embeddings_model/bge-small-en
COPY scripts/download_model.py .

# Model: https://huggingface.co/aliakseilabanau/bge-small-en
RUN . ./.venv/bin/activate && python download_model.py "aliakseilabanau/bge-small-en" "$BGE_EMBEDDINGS_MODEL_PATH" "openvino"

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        # Libreoffice is required for MS office documents
        libreoffice \
        libmagic1 \
        poppler-utils \
        # Dependency for opencv library
        libgl1-mesa-glx && \
    # Cleanup apt cache in the same command to reduce size
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Copy the sources and virtual env. No poetry.
RUN adduser -u 1001 --disabled-password --gecos "" appuser
COPY --chown=appuser --from=builder /app .
COPY --chown=appuser --from=builder /usr/share/nltk_data /usr/share/nltk_data
COPY --chown=appuser --from=builder "$BGE_EMBEDDINGS_MODEL_PATH" "$BGE_EMBEDDINGS_MODEL_PATH"

COPY ./scripts/docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENV LOG_LEVEL=INFO
ENV WEB_CONCURRENCY=1

# DIAL RAG related envs
ENV DO_NOT_TRACK=true
ENV UNSTRUCTURED_XY_CUT_BBOX_SHRINK_FACTOR=1.0
ENV USE_DESCRIPTION_INDEX=true

EXPOSE 5000

USER appuser
ENTRYPOINT ["/docker_entrypoint.sh"]

CMD ["uvicorn", "general_mindmap.v2.app:app", "--host", "0.0.0.0", "--port", "5000"]
