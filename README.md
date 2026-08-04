# rss-translate-proxy

A small proxy that fetches RSS/Atom feeds, translates each item into a target
language, and re-serves them as a clean RSS feed. Useful for reading
non-English sources (e.g. threat-intel feeds) in your regular RSS reader,
with the original text kept alongside the translation.

## How it works

- **`worker.py`** polls the configured source feeds on an interval, optionally
  fetches full article content, translates new/changed items, and caches
  the rendered feed XML, an HTML entry list, and one HTML page per entry in
  SQLite (`data/cache.sqlite3`).
- **`app.py`** is a FastAPI app that serves the cached feed XML at
  `/feeds/<id>.xml`, a human-readable list of the same entries at
  `/feeds/<id>` (each linking to its own page at `/feeds/<id>/<entry_id>`),
  and extracted images at `/images/<name>`. It never calls the translation
  API itself — it only reads what the worker has cached.
- Two processes are meant to run side by side (see `docker-compose.yml`):
  the web app and the worker.

### Translation providers

- **DeepL** (default) — requires `DEEPL_API_KEY`.
- **LibreTranslate** — self-hosted (a `libretranslate` service is included in
  `docker-compose.yml`) or a remote instance via `LIBRETRANSLATE_ENDPOINT` /
  `LIBRETRANSLATE_API_KEY`.

If DeepL is selected but unavailable (no key, or quota exceeded), the worker
falls back to LibreTranslate automatically.

### Other features

- IOC preservation: URLs, CVE IDs, hashes, IPv4 addresses, and email
  addresses are protected from mangling during translation.
- Optional full-content fetching: pull the full article (via its link or a
  per-feed JSON/XML API) instead of translating just the feed summary.
- Original text is kept in the output feed (`original_mode: text | link | none`).
- Embedded images (including inline `data:` URIs) are extracted and re-served
  locally under `/images/`.

## Configuration

Copy the example files and fill in your values:

```sh
cp .env.example .env
cp config.yaml.example config.yaml
```

`.env` holds secrets and endpoints (`DEEPL_API_KEY`, `DEEPL_ENDPOINT`,
`LIBRETRANSLATE_ENDPOINT`, `LIBRETRANSLATE_API_KEY`).

`config.yaml` holds everything else: server host/port, translation provider
and language settings, cache/TTL settings, and the list of feeds to proxy.
Each feed entry defines its source URL, item limit, and optional full-content
fetching rules. See `config.yaml.example` for a fully annotated sample.

## Running

### With Docker (recommended)

```sh
docker compose up -d
```

This starts the web app, the worker, and (optionally) a local LibreTranslate
instance. The feed will be available at
`http://localhost:8086/feeds/<id>.xml` once the worker has completed its
first run.

### Locally with uv

```sh
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8086 &
uv run python worker.py
```

## Endpoints

- `GET /` — service info and list of available feeds.
- `GET /healthz` — health check.
- `GET /feeds/{feed_id}.xml` — translated RSS feed for the given feed id.
- `GET /feeds/{feed_id}` — browser-readable list of the feed's translated
  entries. Each entry's title still links to the original article; a
  separate link opens that entry's own local page.
- `GET /feeds/{feed_id}/{entry_id}` — a single translated entry (translation
  plus original, per `original_mode`) on its own page.
- `GET /images/{name}` — locally cached images extracted from feed content.

## Development

```sh
just lock          # regenerate uv.lock
just lock-update    # upgrade and regenerate uv.lock
just bump-render-version  # bump cache.render_version to force feed XML rebuild
```

Lint with `ruff` (configured in `pyproject.toml`).

If you change how feed XML is rendered (not just how items are translated),
bump `render_version` in `config.yaml` so cached XML is rebuilt without
needing to retranslate everything.
