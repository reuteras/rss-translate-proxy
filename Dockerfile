FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
  && uv sync --frozen --no-dev
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY app.py worker.py config.yaml ./
RUN mkdir -p /app/data

EXPOSE 8086
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8086"]
