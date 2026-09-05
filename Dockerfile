FROM ghcr.io/astral-sh/uv:0.12.9 AS uv
FROM python:3.14-slim

WORKDIR /app
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY app.py worker.py config.yaml ./
RUN useradd --no-create-home --shell /bin/false --uid 1000 app \
  && mkdir -p /app/data \
  && chown -R app:app /app

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8086/healthz')" || exit 1

USER app

EXPOSE 8086
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8086"]
