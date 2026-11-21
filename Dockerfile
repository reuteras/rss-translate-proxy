FROM ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d AS uv
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

# Distro security updates the base image may lag behind.
RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
# uv is only needed for this step: bind-mount it instead of copying it
# into the image (its bundled Rust crates would otherwise ship too).
RUN --mount=type=bind,from=uv,source=/uv,target=/bin/uv \
    uv sync --frozen --no-dev
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY app.py worker.py config.yaml ./
RUN useradd --no-create-home --shell /bin/false --uid 1000 app \
  && mkdir -p /app/data \
  && chown -R app:app /app

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8086/healthz')"]

USER 1000

EXPOSE 8086
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8086"]
